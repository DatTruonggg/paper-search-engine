#!/usr/bin/env python3
"""
Explore ArXiv categories from the Kaggle dataset
Downloads and analyzes the metadata to show all available categories
"""

import kagglehub
import json
import pandas as pd
from pathlib import Path
from collections import Counter
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def download_and_explore():
    """Download ArXiv dataset and explore categories"""
    
    logger.info("Downloading ArXiv dataset metadata from Kaggle...")
    logger.info("This will download the full dataset (~3.4GB compressed)")
    
    try:
        # Download the dataset
        path = kagglehub.dataset_download("Cornell-University/arxiv")
        logger.info(f"Dataset downloaded to: {path}")
        
        # Find the metadata JSON file
        dataset_path = Path(path)
        json_files = list(dataset_path.glob("*.json"))
        
        if not json_files:
            logger.error("No JSON metadata file found")
            return None
        
        json_file = json_files[0]
        logger.info(f"Found metadata file: {json_file}")
        
        # Process the file to extract categories
        logger.info("Processing metadata to extract categories...")
        
        all_categories = []
        paper_count = 0
        category_papers = {}  # Track which papers belong to each category
        
        # Read the JSON lines file
        with open(json_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                if line_num % 100000 == 0 and line_num > 0:
                    logger.info(f"Processed {line_num:,} papers...")
                
                if line.strip():
                    try:
                        paper = json.loads(line)
                        paper_count += 1
                        
                        # Extract categories (space-separated string)
                        if 'categories' in paper and paper['categories']:
                            cats = paper['categories'].split()
                            all_categories.extend(cats)
                            
                            # Track paper IDs for each category
                            for cat in cats:
                                if cat not in category_papers:
                                    category_papers[cat] = []
                                # Store only first 5 papers as examples
                                if len(category_papers[cat]) < 5:
                                    category_papers[cat].append({
                                        'id': paper.get('id', ''),
                                        'title': paper.get('title', '')[:100]  # Truncate long titles
                                    })
                    
                    except json.JSONDecodeError:
                        continue
                
                # Process first 1M papers for faster exploration
                if line_num >= 10000000:
                    logger.info(f"Processed first 10,000,000 papers for exploration")
                    break
        
        logger.info(f"Total papers processed: {paper_count:,}")
        
        # Analyze categories
        category_counts = Counter(all_categories)
        
        # Group by main category
        main_categories = {}
        for cat, count in category_counts.items():
            main = cat.split('.')[0] if '.' in cat else cat
            if main not in main_categories:
                main_categories[main] = []
            main_categories[main].append((cat, count))
        
        # Sort main categories by total count
        main_cat_totals = {}
        for main, subcats in main_categories.items():
            main_cat_totals[main] = sum(count for _, count in subcats)
        
        # Print comprehensive analysis
        print("\n" + "="*80)
        print("ARXIV CATEGORIES COMPREHENSIVE ANALYSIS")
        print("="*80)
        print(f"Total papers analyzed: {paper_count:,}")
        print(f"Total unique categories: {len(category_counts)}")
        print(f"Total category assignments: {sum(category_counts.values()):,}")
        
        print("\n" + "="*80)
        print("MAIN CATEGORY GROUPS (sorted by total papers)")
        print("="*80)
        
        for main, total in sorted(main_cat_totals.items(), key=lambda x: x[1], reverse=True):
            print(f"\n{main}: {total:,} papers")
            print("-" * 40)
            
            # Show subcategories
            subcats = sorted(main_categories[main], key=lambda x: x[1], reverse=True)
            for subcat, count in subcats[:10]:  # Show top 10 subcategories
                print(f"  {subcat:25} {count:8,} papers")
                
                # Show example papers
                if subcat in category_papers and category_papers[subcat]:
                    print(f"    Example papers:")
                    for ex in category_papers[subcat][:2]:  # Show 2 examples
                        print(f"      - {ex['id']}: {ex['title'][:60]}...")
            
            if len(subcats) > 10:
                print(f"  ... and {len(subcats) - 10} more subcategories")
        
        # Identify NLP-related categories
        print("\n" + "="*80)
        print("NLP & AI RELATED CATEGORIES (for filtering)")
        print("="*80)
        
        nlp_keywords = {
            'Primary NLP': ['cs.CL'],  # Computation and Language
            'AI & ML': ['cs.AI', 'cs.LG', 'stat.ML'],  # AI, Learning, ML
            'Information Retrieval': ['cs.IR'],  # Information Retrieval
            'Computer Vision (may have NLP)': ['cs.CV'],  # Computer Vision
            'Human-Computer Interaction': ['cs.HC'],  # HCI
        }
        
        recommended_categories = []
        
        for group, cats in nlp_keywords.items():
            print(f"\n{group}:")
            group_total = 0
            for cat in cats:
                if cat in category_counts:
                    count = category_counts[cat]
                    group_total += count
                    recommended_categories.append(cat)
                    print(f"  {cat:15} {count:8,} papers")
                    
                    # Show examples
                    if cat in category_papers and category_papers[cat]:
                        print(f"    Example papers:")
                        for ex in category_papers[cat][:2]:
                            print(f"      - {ex['id']}: {ex['title'][:50]}...")
            
            print(f"  Group total: {group_total:,} papers")
        
        # Save detailed analysis
        output_file = Path("../data/logs/category_analysis_detailed.json")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump({
                "total_papers": paper_count,
                "total_categories": len(category_counts),
                "category_counts": dict(category_counts.most_common()),
                "main_categories": {k: dict(v) for k, v in main_categories.items()},
                "recommended_nlp_categories": recommended_categories,
                "sample_papers": {k: v for k, v in category_papers.items() if k in recommended_categories}
            }, f, indent=2)
        
        logger.info(f"\nDetailed analysis saved to: {output_file}")
        
        print("\n" + "="*80)
        print("RECOMMENDED CATEGORIES FOR NLP PAPER COLLECTION:")
        print("="*80)
        print("Primary focus (pure NLP):")
        print("  - cs.CL (Computation and Language)")
        print("\nSecondary focus (AI/ML with NLP components):")
        print("  - cs.AI (Artificial Intelligence)")
        print("  - cs.LG (Machine Learning)")
        print("  - cs.IR (Information Retrieval)")
        print("  - stat.ML (Statistics - Machine Learning)")
        print("\nOptional (may contain NLP research):")
        print("  - cs.CV (Computer Vision - for multimodal models)")
        print("  - cs.HC (Human-Computer Interaction)")
        
        total_nlp_papers = sum(category_counts.get(cat, 0) for cat in recommended_categories)
        print(f"\nTotal papers in recommended categories: {total_nlp_papers:,}")
        
        return category_counts, recommended_categories
        
    except Exception as e:
        logger.error(f"Error: {e}")
        logger.info("\nAlternative: Using predefined category list based on ArXiv taxonomy")
        
        # Provide known categories as fallback
        print("\n" + "="*80)
        print("KNOWN ARXIV NLP-RELATED CATEGORIES")
        print("="*80)
        print("""
Based on ArXiv's taxonomy, here are the main NLP-related categories:

PRIMARY NLP CATEGORIES:
- cs.CL    : Computation and Language (main NLP category)

RELATED AI/ML CATEGORIES:
- cs.AI    : Artificial Intelligence
- cs.LG    : Machine Learning
- cs.IR    : Information Retrieval
- stat.ML  : Machine Learning (Statistics)

INTERDISCIPLINARY CATEGORIES (may contain NLP):
- cs.CV    : Computer Vision (vision-language models)
- cs.HC    : Human-Computer Interaction
- cs.SI    : Social and Information Networks
- cs.CY    : Computers and Society
- cs.DL    : Digital Libraries

For a comprehensive NLP dataset, recommend filtering by:
['cs.CL', 'cs.AI', 'cs.LG', 'cs.IR', 'stat.ML']
        """)
        
        return None, ['cs.CL', 'cs.AI', 'cs.LG', 'cs.IR', 'stat.ML']


if __name__ == "__main__":
    categories, recommended = download_and_explore()
    
    if categories:
        print(f"\n✓ Analysis complete! Found {len(categories)} unique categories")
        print(f"✓ Recommended filtering by {len(recommended)} categories for NLP papers")