#!/usr/bin/env python3
import os, asyncio, json, pandas as pd
from logs import log
from datetime import datetime, timezone
from pathlib import Path
from arxiv_nlp_pipeline import ArxivDataPipeline
from arxiv_pdf_downloader import ArxivPDFDownloader
from docling_pdf_parser import run as parse_pdfs
from ingest_papers import PaperProcessor

# Hardcoded configuration
MINIO_ENDPOINT = "103.3.247.120:9002"
MINIO_ACCESS_KEY = "admin"
MINIO_SECRET_KEY = "password"
MINIO_BUCKET = "papers"
ES_HOST = "103.3.247.120:9200"

os.environ["MINIO_ENDPOINT"] = MINIO_ENDPOINT
os.environ["MINIO_ACCESS_KEY"] = MINIO_ACCESS_KEY
os.environ["MINIO_SECRET_KEY"] = MINIO_SECRET_KEY
os.environ["MINIO_BUCKET"] = MINIO_BUCKET

async def main():
    log.info("[PIPELINE] Starting full ArXiv NLP pipeline")
    stats = {"start": datetime.now(timezone.utc).isoformat()}

    # Create local data directories
    base_dir = Path.cwd() / "data"
    (base_dir / "kaggle").mkdir(parents=True, exist_ok=True)
    (base_dir / "pdfs").mkdir(parents=True, exist_ok=True)
    (base_dir / "parsed").mkdir(parents=True, exist_ok=True)
    (base_dir / "logs").mkdir(parents=True, exist_ok=True)

    # 1. Download/ensure Kaggle dataset locally
    pipeline = ArxivDataPipeline()
    local_dataset = base_dir / "kaggle" / "arxiv-metadata-oai-snapshot.json"

    if not local_dataset.exists():
        log.info("[PIPELINE] Downloading dataset from Kaggle to local...")
        import kagglehub
        kaggle_path = Path(kagglehub.dataset_download("Cornell-University/arxiv"))
        for f in kaggle_path.glob("*.json"):
            f.rename(local_dataset)
            break
        log.info(f"[PIPELINE] Dataset saved to {local_dataset}")
    else:
        log.info(f"[PIPELINE] Using existing dataset: {local_dataset}")

    # 2. Load and filter cs.CL papers locally
    log.info("[PIPELINE] Loading dataset from local file...")
    papers = []
    with open(local_dataset, 'r') as f:
        for line in f:
            if line.strip():
                try: papers.append(json.loads(line))
                except: pass
            if len(papers) % 100000 == 0:
                log.info(f"[PIPELINE] Loaded {len(papers):,} papers...")

    df = pd.DataFrame(papers)
    log.info(f"[PIPELINE] Total papers: {len(df)}")

    # Filter cs.CL only
    nlp_df = df[df['categories'].str.contains('cs.CL', na=False)]
    log.info(f"[PIPELINE] cs.CL papers: {len(nlp_df)}")
    stats["total_nlp_papers"] = len(nlp_df)

    # 3. Save filtered papers metadata locally
    metadata_file = base_dir / "kaggle" / "cs_cl_papers.json"
    nlp_df.to_json(metadata_file, orient="records", lines=True)
    log.info(f"[PIPELINE] Saved {len(nlp_df)} papers metadata to {metadata_file}")

    # 4. Download PDFs locally in batches
    log.info("[PIPELINE] Downloading PDFs locally...")
    pdf_dir = base_dir / "pdfs"
    batch_size = 500
    total_downloaded = 0

    for i in range(0, len(nlp_df), batch_size):
        batch = nlp_df.iloc[i:i+batch_size]
        batch_num = i//batch_size + 1

        for _, paper in batch.iterrows():
            paper_id = paper.get("id", "").replace("/", "_")
            if not paper_id: continue

            pdf_path = pdf_dir / f"{paper_id}.pdf"
            if pdf_path.exists():
                continue

            # Download PDF from arxiv
            import aiohttp
            pdf_url = f"https://arxiv.org/pdf/{paper.get('id')}.pdf"
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(pdf_url) as resp:
                        if resp.status == 200:
                            pdf_path.write_bytes(await resp.read())
                            total_downloaded += 1
                await asyncio.sleep(1)  # Rate limit
            except: pass

        log.info(f"[PIPELINE] Batch {batch_num}: Downloaded {total_downloaded} PDFs total")

    stats["pdfs_downloaded"] = total_downloaded

    # 5. Parse PDFs with Docling locally
    log.info("[PIPELINE] Parsing PDFs with Docling...")
    parsed_dir = base_dir / "parsed"
    parsed_count = 0

    from docling.document_converter import DocumentConverter
    converter = DocumentConverter()

    for pdf_file in (base_dir / "pdfs").glob("*.pdf"):
        paper_id = pdf_file.stem
        md_file = parsed_dir / f"{paper_id}.md"

        if md_file.exists():
            continue

        try:
            conversion = converter.convert(str(pdf_file))
            md_content = conversion.document.export_to_markdown()
            md_file.write_text(md_content)
            parsed_count += 1
        except Exception as e:
            log.error(f"[PIPELINE] Failed parsing {paper_id}: {e}")

    stats["parsed_count"] = parsed_count
    log.info(f"[PIPELINE] Parsed {parsed_count} PDFs")

    # 6. Ingest to Elasticsearch using ingest_papers.py
    log.info("[PIPELINE] Ingesting to Elasticsearch...")

    # Save metadata JSON files for ingest_papers to use
    metadata_dir = base_dir / "metadata"
    metadata_dir.mkdir(exist_ok=True)

    for _, paper in nlp_df.iterrows():
        paper_id = paper.get('id', '').replace('/', '_')
        if paper_id:
            meta_file = metadata_dir / f"{paper_id}.json"
            meta_data = {
                "paper_id": paper_id,
                "title": paper.get("title", ""),
                "authors": paper.get("authors", []),
                "abstract": paper.get("abstract", ""),
                "categories": paper.get("categories", ""),
                "publish_date": paper.get("update_date", None)
            }
            meta_file.write_text(json.dumps(meta_data))

    # Use PaperProcessor to ingest all parsed markdown files
    processor = PaperProcessor(
        es_host=ES_HOST,
        chunk_size=1024,
        chunk_overlap=100,
        json_metadata_dir=str(metadata_dir)
    )

    # Process all markdown files
    processor.ingest_directory(
        markdown_dir=parsed_dir,
        batch_size=50,
        max_files=None
    )

    stats["ingested"] = len(list(parsed_dir.glob("*.md")))
    stats["end"] = datetime.now(timezone.utc).isoformat()
    # Save stats to log file
    stats_file = base_dir / "logs" / f"pipeline_run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    stats_file.write_text(json.dumps(stats, indent=2))

    log.info(f"[PIPELINE] Complete. Stats saved to {stats_file}")
    log.info(json.dumps(stats, indent=2))
    return stats

if __name__ == "__main__":
    asyncio.run(main())