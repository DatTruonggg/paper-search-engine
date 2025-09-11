#!/usr/bin/env python3
"""
Download Latest NLP Papers from ArXiv
Uses advanced filtering with categories and keywords to find and download recent NLP papers
"""

import asyncio
import json
import logging
from pathlib import Path
from datetime import datetime
from arxiv_nlp_pipeline import ArxivDataPipeline
from arxiv_pdf_downloader import ArxivPDFDownloader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('../data/logs/download_nlp_papers.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class NLPPaperDownloader:
    def __init__(self, data_dir: str = "../data"):
        self.pipeline = ArxivDataPipeline(data_dir)
        self.pdf_downloader = ArxivPDFDownloader(
            pdfs_dir=str(self.pipeline.pdfs_dir),
            max_concurrent=5,  # Conservative to avoid rate limiting
            delay_between_requests=1.0  # 1 second delay
        )
        self.data_dir = Path(data_dir)
    
    def load_or_download_dataset(self):
        """Load dataset from cache or download if not available"""
        # Check if we already have the dataset
        json_files = list(self.pipeline.raw_dir.glob("*.json"))
        
        if json_files:
            logger.info(f"Found existing dataset at {json_files[0]}")
            return True
        else:
            logger.info("Dataset not found, downloading from Kaggle...")
            try:
                self.pipeline.download_dataset()
                return True
            except Exception as e:
                logger.error(f"Failed to download dataset: {e}")
                return False
    
    def process_full_dataset(self, limit_papers: int = None):
        """Process the full dataset without the 500k limit"""
        json_file = self.pipeline.raw_dir / "arxiv-metadata-oai-snapshot.json"
        
        if not json_file.exists():
            json_files = list(self.pipeline.raw_dir.glob("*.json"))
            if json_files:
                json_file = json_files[0]
            else:
                raise FileNotFoundError("No metadata JSON file found")
        
        logger.info(f"Loading full dataset from {json_file}")
        
        papers = []
        with open(json_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                if line_num % 100000 == 0 and line_num > 0:
                    logger.info(f"Loaded {line_num:,} papers...")
                
                if line.strip():
                    try:
                        paper = json.loads(line)
                        papers.append(paper)
                    except json.JSONDecodeError:
                        continue
                
                # Optional limit for testing
                if limit_papers and line_num >= limit_papers:
                    logger.info(f"Reached limit of {limit_papers:,} papers")
                    break
        
        logger.info(f"Total papers loaded: {len(papers):,}")
        return papers
    
    async def find_and_download_nlp_papers(self, 
                                   num_papers: int = 1000,
                                   use_categories: bool = True,
                                   use_keywords: bool = True,
                                   min_keyword_matches: int = 1,
                                   limit_dataset: int = None):
        """
        Main function to find and download latest NLP papers
        """
        logger.info("="*80)
        logger.info("DOWNLOADING LATEST NLP PAPERS FROM ARXIV")
        logger.info("="*80)
        logger.info(f"Target papers: {num_papers}")
        logger.info(f"Use categories: {use_categories}")
        logger.info(f"Use keywords: {use_keywords}")
        logger.info(f"Min keyword matches: {min_keyword_matches}")
        
        # Step 1: Load dataset
        if not self.load_or_download_dataset():
            logger.error("Failed to load dataset")
            return None
        
        # Step 2: Load papers into dataframe
        try:
            import pandas as pd
            papers_data = self.process_full_dataset(limit_dataset)
            df = pd.DataFrame(papers_data)
            logger.info(f"Created dataframe with {len(df):,} papers")
            logger.info(f"Columns: {df.columns.tolist()}")
        except Exception as e:
            logger.error(f"Failed to load papers: {e}")
            return None
        
        # Step 3: Filter NLP papers with early stopping
        try:
            # The new method already returns the latest papers up to target amount
            latest_papers = self.pipeline.filter_nlp_papers_advanced(
                df, 
                use_categories=use_categories,
                use_keywords=use_keywords,
                min_keyword_matches=min_keyword_matches,
                target_papers=num_papers  # Will stop after finding this many papers
            )
            logger.info(f"Found {len(latest_papers):,} NLP papers")
        except Exception as e:
            logger.error(f"Failed to filter papers: {e}")
            return None
        
        if len(latest_papers) == 0:
            logger.error("No papers found after filtering")
            return None
        
        # Step 5: Analyze filtering results
        try:
            analysis = self.pipeline.analyze_nlp_filtering_results(latest_papers)
            self.save_analysis(analysis, latest_papers)
        except Exception as e:
            logger.warning(f"Failed to analyze results: {e}")
        
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
        logger.info("="*80)
        logger.info("STARTING PDF DOWNLOADS")
        logger.info("="*80)
        
        try:
            results = await self.pdf_downloader.download_batch(papers_to_download)
            self.pdf_downloader.print_statistics()
            self.pdf_downloader.save_download_log(results, '../data/logs/nlp_papers_download_log.json')
            
            # Save successful downloads info
            successful_papers = [paper for paper, result in zip(papers_to_download, results) 
                               if result.get('status') == 'success']
            
            success_file = self.data_dir / 'logs' / f'successful_nlp_downloads_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            with open(success_file, 'w') as f:
                json.dump({
                    'download_timestamp': datetime.now().isoformat(),
                    'total_requested': len(papers_to_download),
                    'successful_downloads': len(successful_papers),
                    'papers': successful_papers
                }, f, indent=2)
            
            logger.info(f"Download complete! Check {success_file} for details")
            return results
            
        except Exception as e:
            logger.error(f"Failed to download PDFs: {e}")
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
        
        logger.info(f"Analysis saved to {analysis_file}")
        
        # Print key statistics
        logger.info("\n" + "="*60)
        logger.info("NLP FILTERING ANALYSIS")
        logger.info("="*60)
        logger.info(f"Total filtered papers: {analysis.get('total_papers', 0):,}")
        logger.info(f"Papers with abstracts: {analysis.get('papers_with_abstract', 0):,}")
        
        if 'top_keywords' in analysis:
            logger.info("\nTop matched keywords:")
            for keyword, count in list(analysis['top_keywords'].items())[:10]:
                logger.info(f"  {keyword}: {count}")
        
        if 'top_categories' in analysis:
            logger.info("\nTop categories:")
            for category, count in list(analysis['top_categories'].items())[:10]:
                logger.info(f"  {category}: {count}")
        
        if 'year_distribution' in analysis:
            logger.info("\nYear distribution:")
            for year, count in list(analysis['year_distribution'].items())[:5]:
                logger.info(f"  {year}: {count}")


async def main():
    """Main execution function"""
    downloader = NLPPaperDownloader()
    
    # Configuration
    NUM_PAPERS = 1000
    USE_CATEGORIES = True
    USE_KEYWORDS = True
    MIN_KEYWORD_MATCHES = 1
    LIMIT_DATASET = None  # Set to None for full dataset, or number for testing
    
    logger.info("Starting NLP paper download process...")
    
    results = await downloader.find_and_download_nlp_papers(
        num_papers=NUM_PAPERS,
        use_categories=USE_CATEGORIES,
        use_keywords=USE_KEYWORDS,
        min_keyword_matches=MIN_KEYWORD_MATCHES,
        limit_dataset=LIMIT_DATASET
    )
    
    if results:
        successful = sum(1 for r in results if r.get('status') == 'success')
        logger.info(f"\n✓ Download process completed!")
        logger.info(f"✓ Successfully downloaded {successful}/{len(results)} papers")
        logger.info(f"✓ PDFs saved to: {downloader.pipeline.pdfs_dir}")
        logger.info(f"✓ Logs saved to: {downloader.pipeline.logs_dir}")
    else:
        logger.error("Download process failed!")


if __name__ == "__main__":
    asyncio.run(main())