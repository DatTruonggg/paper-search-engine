import asyncio
import json
from datetime import datetime
from io import BytesIO
import pandas as pd
from minio import Minio
from logs import log 
from config import cfg
from .arxiv_nlp_pipeline import ArxivDataPipeline
from .arxiv_pdf_downloader import ArxivPDFDownloader


class NLPPaperDownloader:
    def __init__(self):
        # Use global cfg from config
        self.cfg = cfg
        # Initialize pipeline (no local dirs)
        self.pipeline = ArxivDataPipeline(bucket_name=self.cfg.minio_client.bucket_name)
        # Initialize MinIO client for metadata IO
        self.minio = Minio(
            endpoint=self.cfg.minio_client.endpoint,
            access_key=self.cfg.minio_client.access_key,
            secret_key=self.cfg.minio_client.secret_key,
            secure=bool(self.cfg.minio_client.secure),
        )
        # Ensure bucket exists
        if not self.minio.bucket_exists(self.cfg.minio_client.bucket_name):
            self.minio.make_bucket(self.cfg.minio_client.bucket_name)
        # PDF downloader configured for MinIO-only
        self.pdf_downloader = ArxivPDFDownloader(
            config=self.cfg,
            bucket_name=self.cfg.minio_client.bucket_name,
            raw_prefix=self.cfg.minio_client.raw_prefix,
            json_metadata_prefix=self.cfg.minio_client.json_metadata_prefix,
            max_concurrent=self.cfg.minio_client.max_concurrent,
            retry_attempts=self.cfg.minio_client.retry_attempts,
            delay_between_requests=self.cfg.minio_client.delay_between_requests
        )
    
    def read_metadata_from_minio(self) -> pd.DataFrame:
        """Read ArXiv JSONL metadata from MinIO into a DataFrame."""
        obj = self.minio.get_object(self.cfg.minio_client.bucket_name, self.cfg.metadata_object)
        try:
            # Stream into pandas read_json with lines=True
            data = obj.read()
            df = pd.read_json(BytesIO(data), lines=True)
            log.info(f"Loaded metadata from s3://{self.cfg.minio_client.bucket_name}/{self.cfg.metadata_object} -> {len(df):,} rows")
            return df
        finally:
            obj.close()
            obj.release_conn()
    
    def process_full_dataset(self, df: pd.DataFrame, limit_papers: int | None = None) -> pd.DataFrame:
        """Optionally limit the in-memory DataFrame for testing."""
        if limit_papers is not None:
            df = df.head(limit_papers)
            log.info(f"Using limited dataset: {len(df):,} rows")
        return df
    
    async def find_and_download_nlp_papers(self):
        """
        Main function to find and download latest NLP papers
        """
        log.info("="*80)
        log.info("DOWNLOADING LATEST NLP PAPERS FROM ARXIV")
        log.info("="*80)
        log.info(f"Target papers: {self.cfg.target_papers}")
        log.info(f"Use categories: {self.cfg.use_categories}")
        log.info(f"Use keywords: {self.cfg.use_keywords}")
        log.info(f"Min keyword matches: {self.cfg.min_keyword_matches}")
        
        # Step 1: Load metadata DataFrame from MinIO
        try:
            df = self.read_metadata_from_minio()
            df = self.process_full_dataset(df, self.cfg.limit_dataset)
        except Exception as e:
            log.error(f"Failed to read metadata from MinIO: {e}")
            return None
        
        # Step 3: Filter NLP papers with early stopping
        try:
            # The new method already returns the latest papers up to target amount
            latest_papers = self.pipeline.filter_nlp_papers_advanced(
                df,
                use_categories=self.cfg.use_categories,
                use_keywords=self.cfg.use_keywords,
                min_keyword_matches=self.cfg.min_keyword_matches,
                target_papers=self.cfg.target_papers,
            )
            log.info(f"Found {len(latest_papers):,} NLP papers")
        except Exception as e:
            log.error(f"Failed to filter papers: {e}")
            return None
        
        if len(latest_papers) == 0:
            log.error("No papers found after filtering")
            return None
        
        # Step 6: Prepare paper data for download
        papers_to_download = []
        for _, paper in latest_papers.iterrows():
            paper_data = {
                'id': paper.get('id', ''),
                'title': paper.get('title', ''),
                'authors': paper.get('authors', ''),
                'abstract': paper.get('abstract', ''),
                'categories': paper.get('categories', ''),
                'year': paper.get('year', None),
                'matched_keywords': paper.get('matched_nlp_keywords', []),
                'keyword_count': paper.get('keyword_match_count', 0)
            }
            papers_to_download.append(paper_data)
        
        # Step 7: Download PDFs
        log.info("="*80)
        log.info("STARTING PDF DOWNLOADS")
        log.info("="*80)
        
        try:
            results = await self.pdf_downloader.download_batch(papers_to_download)
            # Aggregate simple stats and store to MinIO
            successful_papers = [paper for paper, result in zip(papers_to_download, results) if result.get('status') == 'success']
            summary = {
                'download_timestamp': datetime.now().isoformat(),
                'total_requested': len(papers_to_download),
                'successful_downloads': len(successful_papers),
            }
            summary_bytes = json.dumps(summary, indent=2).encode('utf-8')
            summary_key = f"{self.cfg.logs_prefix}/nlp_papers_download_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            self.minio.put_object(
                self.cfg.bucket_name,
                summary_key,
                BytesIO(summary_bytes),
                len(summary_bytes),
                content_type="application/json",
            )
            log.info(f"Download complete! Summary at s3://{self.cfg.minio_client.bucket_name}/{summary_key}")
            return results
            
        except Exception as e:
            log.error(f"Failed to download PDFs: {e}")
            return None
    
    def save_analysis(self, analysis: dict, papers_df):
        """Save analysis results to file"""
        analysis_file = self.data_dir / 'logs' / f'nlp_filtering_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        
        # Convert any numpy types to native Python types for JSON serialization
        def convert_for_json(obj):
            if hasattr(obj, 'item'):  # numpy scalar
                return obj.item()
            elif hasattr(obj, 'tolist'):  # numpy array
                return obj.tolist()
            return obj
        
        # Clean the analysis data
        clean_analysis = {}
        for key, value in analysis.items():
            if isinstance(value, dict):
                clean_analysis[key] = {k: convert_for_json(v) for k, v in value.items()}
            else:
                clean_analysis[key] = convert_for_json(value)
        
        with open(analysis_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'analysis': clean_analysis,
                'sample_papers': papers_df.head(10)[['id', 'title', 'categories', 'matched_nlp_keywords']].to_dict('records')
            }, f, indent=2)
        
        log.info(f"Analysis saved to {analysis_file}")
        
        # Print key statistics
        log.info("\n" + "="*60)
        log.info("NLP FILTERING ANALYSIS")
        log.info("="*60)
        log.info(f"Total filtered papers: {analysis.get('total_papers', 0):,}")
        log.info(f"Papers with abstracts: {analysis.get('papers_with_abstract', 0):,}")
        
        if 'top_keywords' in analysis:
            log.info("\nTop matched keywords:")
            for keyword, count in list(analysis['top_keywords'].items())[:10]:
                log.info(f"  {keyword}: {count}")
        
        if 'top_categories' in analysis:
            log.info("\nTop categories:")
            for category, count in list(analysis['top_categories'].items())[:10]:
                log.info(f"  {category}: {count}")
        
        if 'year_distribution' in analysis:
            log.info("\nYear distribution:")
            for year, count in list(analysis['year_distribution'].items())[:5]:
                log.info(f"  {year}: {count}")


async def main():
    downloader = NLPPaperDownloader()
    log.info("Starting NLP paper download process...")
    results = await downloader.find_and_download_nlp_papers()
    if results:
        successful = sum(1 for r in results if r.get('status') == 'success')
        log.info(f"\n✓ Download process completed!")
        log.info(f"✓ Successfully downloaded {successful}/{len(results)} papers")
    else:
        log.error("Download process failed!")


if __name__ == "__main__":
    asyncio.run(main())