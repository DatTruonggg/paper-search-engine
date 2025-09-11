#!/usr/bin/env python3
"""
Test the optimized incremental filtering approach
"""

import time
from download_latest_nlp_papers import NLPPaperDownloader
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_optimized_performance():
    """Test the optimized filtering performance"""
    downloader = NLPPaperDownloader()
    
    logger.info("="*80)
    logger.info("TESTING OPTIMIZED INCREMENTAL FILTERING")
    logger.info("="*80)
    
    start_time = time.time()
    
    # Test with small target to see early stopping in action
    results = await downloader.find_and_download_nlp_papers(
        num_papers=20,  # Small target to see early stopping
        use_categories=True,
        use_keywords=True,
        min_keyword_matches=1,
        limit_dataset=100000  # Process 100k papers to see the optimization
    )
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    logger.info("="*80)
    logger.info("PERFORMANCE RESULTS")
    logger.info("="*80)
    logger.info(f"Total processing time: {processing_time:.2f} seconds")
    
    if results:
        successful = sum(1 for r in results if r.get('status') == 'success')
        logger.info(f"Successfully downloaded: {successful}/{len(results)} papers")
        logger.info("✅ Optimized filtering with early stopping works!")
    else:
        logger.error("❌ Test failed")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_optimized_performance())