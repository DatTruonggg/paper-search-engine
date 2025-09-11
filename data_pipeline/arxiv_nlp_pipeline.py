#!/usr/bin/env python3
"""
ArXiv NLP Papers Data Pipeline
Downloads ArXiv dataset from Kaggle, explores categories, and downloads papers
"""

import os
import json
import pandas as pd
import kagglehub
from pathlib import Path
from typing import List, Dict, Set
from collections import Counter
import logging
import re
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('../data/logs/pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ArxivDataPipeline:
    def __init__(self, data_dir: str = "../data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.pdfs_dir = self.data_dir / "pdfs"
        self.processed_dir = self.data_dir / "processed"
        self.logs_dir = self.data_dir / "logs"
        
        # Create directories if they don't exist
        for dir_path in [self.raw_dir, self.pdfs_dir, self.processed_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Define NLP-related categories
        self.nlp_categories = [
            'cs.CL',    # Computation and Language (primary NLP)
            'cs.AI',    # Artificial Intelligence
            'cs.LG',    # Machine Learning
            'cs.IR',    # Information Retrieval
            'stat.ML',  # Statistics - Machine Learning
            'cs.CV',    # Computer Vision (for multimodal models)
            'cs.HC',    # Human-Computer Interaction
        ]
        
        # Define comprehensive NLP keywords for abstract/title search
        self.nlp_keywords = {
            # Core NLP terms
            'core_nlp': [
                'natural language processing', 'nlp', 'computational linguistics',
                'text mining', 'text analysis', 'language model', 'language modeling'
            ],
            
            # Modern architectures
            'architectures': [
                'transformer', 'transformers', 'bert', 'gpt', 'gpt-2', 'gpt-3', 'gpt-4',
                't5', 'bart', 'roberta', 'electra', 'albert', 'xlnet', 'deberta',
                'attention mechanism', 'self-attention', 'multi-head attention',
                'large language model', 'llm', 'foundation model', 'pre-trained model'
            ],
            
            # RAG and retrieval
            'rag_retrieval': [
                'rag', 'retrieval augmented generation', 'retrieval-augmented',
                'vector database', 'semantic search', 'dense retrieval',
                'knowledge grounding', 'knowledge-grounded', 'retrieval system'
            ],
            
            # Dialogue and chatbots
            'dialogue': [
                'chatbot', 'conversational ai', 'dialogue system', 'virtual assistant',
                'conversation model', 'dialogue generation', 'chat model',
                'human-computer dialogue', 'conversational agent'
            ],
            
            # NLP tasks
            'tasks': [
                'machine translation', 'neural machine translation', 'mt',
                'sentiment analysis', 'emotion recognition', 'opinion mining',
                'named entity recognition', 'ner', 'entity extraction',
                'question answering', 'qa', 'reading comprehension',
                'text summarization', 'abstractive summarization', 'extractive summarization',
                'text classification', 'document classification', 'text categorization',
                'information extraction', 'relation extraction', 'knowledge extraction',
                'part-of-speech tagging', 'pos tagging', 'parsing', 'dependency parsing',
                'semantic role labeling', 'word sense disambiguation', 'coreference resolution'
            ],
            
            # Techniques and methods
            'techniques': [
                'tokenization', 'word embedding', 'sentence embedding', 'contextualized embedding',
                'fine-tuning', 'prompt engineering', 'in-context learning',
                'zero-shot', 'few-shot learning', 'one-shot learning',
                'instruction tuning', 'rlhf', 'reinforcement learning from human feedback',
                'knowledge distillation', 'transfer learning'
            ],
            
            # Evaluation metrics
            'evaluation': [
                'bleu', 'rouge', 'perplexity', 'f1 score', 'accuracy',
                'human evaluation', 'automatic evaluation'
            ],
            
            # Language and text processing
            'language': [
                'multilingual', 'cross-lingual', 'low-resource language',
                'text generation', 'language generation', 'text-to-text',
                'speech recognition', 'automatic speech recognition', 'asr',
                'text-to-speech', 'speech synthesis'
            ]
        }
        
        # Flatten all keywords for easy searching
        self.all_nlp_keywords = []
        for category, keywords in self.nlp_keywords.items():
            self.all_nlp_keywords.extend(keywords)
    
    def download_dataset(self) -> str:
        """Download ArXiv dataset from Kaggle"""
        logger.info("Downloading ArXiv dataset from Kaggle...")
        
        try:
            # Download latest version
            path = kagglehub.dataset_download("Cornell-University/arxiv")
            logger.info(f"Dataset downloaded to: {path}")
            
            # Copy to our raw data directory
            import shutil
            dataset_files = list(Path(path).glob("*"))
            for file in dataset_files:
                dest = self.raw_dir / file.name
                if not dest.exists():
                    shutil.copy2(file, dest)
                    logger.info(f"Copied {file.name} to {dest}")
            
            return str(self.raw_dir)
        
        except Exception as e:
            logger.error(f"Failed to download dataset: {e}")
            raise
    
    def load_metadata(self) -> pd.DataFrame:
        """Load ArXiv metadata from downloaded dataset"""
        json_file = self.raw_dir / "arxiv-metadata-oai-snapshot.json"
        
        if not json_file.exists():
            logger.warning(f"Metadata file not found at {json_file}")
            # Try to find any JSON file
            json_files = list(self.raw_dir.glob("*.json"))
            if json_files:
                json_file = json_files[0]
                logger.info(f"Using metadata file: {json_file}")
            else:
                raise FileNotFoundError("No metadata JSON file found in raw data directory")
        
        logger.info(f"Loading metadata from {json_file}")
        
        # Load JSON lines file
        papers = []
        with open(json_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                if line_num % 100000 == 0:
                    logger.info(f"Loaded {line_num} papers...")
                
                if line.strip():  # Skip empty lines
                    try:
                        paper = json.loads(line)
                        papers.append(paper)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Error parsing line {line_num}: {e}")
                
                # For initial exploration, limit to first 500k papers
                if line_num >= 500000:
                    logger.info("Loaded first 500,000 papers for exploration")
                    break
        
        logger.info(f"Total papers loaded: {len(papers)}")
        df = pd.DataFrame(papers)
        return df
    
    def explore_categories(self, df: pd.DataFrame) -> Dict:
        """Explore and analyze all categories in the dataset"""
        logger.info("Exploring ArXiv categories...")
        
        # Extract all categories
        all_categories = []
        for categories_str in df['categories'].dropna():
            # Categories are space-separated
            categories = categories_str.split()
            all_categories.extend(categories)
        
        # Count category frequencies
        category_counts = Counter(all_categories)
        
        # Separate by main category prefix
        main_categories = {}
        for cat, count in category_counts.items():
            main_cat = cat.split('.')[0] if '.' in cat else cat
            if main_cat not in main_categories:
                main_categories[main_cat] = {}
            main_categories[main_cat][cat] = count
        
        # Sort categories
        sorted_categories = dict(sorted(category_counts.items(), key=lambda x: x[1], reverse=True))
        
        # Print analysis
        logger.info(f"\n{'='*60}")
        logger.info(f"ARXIV CATEGORIES ANALYSIS")
        logger.info(f"{'='*60}")
        logger.info(f"Total unique categories: {len(category_counts)}")
        logger.info(f"Total category assignments: {sum(category_counts.values())}")
        
        logger.info(f"\n{'='*60}")
        logger.info("TOP 50 CATEGORIES BY FREQUENCY:")
        logger.info(f"{'='*60}")
        for i, (cat, count) in enumerate(list(sorted_categories.items())[:50], 1):
            logger.info(f"{i:3}. {cat:20} : {count:8,} papers")
        
        # Identify NLP-related categories
        nlp_keywords = ['cl', 'lg', 'ai', 'ir', 'ml', 'computational', 'language', 
                       'information', 'retrieval', 'learning', 'intelligence']
        
        potential_nlp_categories = []
        for cat in category_counts:
            cat_lower = cat.lower()
            if any(keyword in cat_lower for keyword in nlp_keywords):
                potential_nlp_categories.append((cat, category_counts[cat]))
        
        logger.info(f"\n{'='*60}")
        logger.info("POTENTIAL NLP-RELATED CATEGORIES:")
        logger.info(f"{'='*60}")
        for cat, count in sorted(potential_nlp_categories, key=lambda x: x[1], reverse=True):
            logger.info(f"{cat:20} : {count:8,} papers")
        
        # Save category analysis to file
        analysis_file = self.logs_dir / f"category_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        analysis_data = {
            "total_categories": len(category_counts),
            "total_assignments": sum(category_counts.values()),
            "all_categories": sorted_categories,
            "main_categories": main_categories,
            "potential_nlp_categories": dict(potential_nlp_categories)
        }
        
        with open(analysis_file, 'w') as f:
            json.dump(analysis_data, f, indent=2)
        logger.info(f"\nCategory analysis saved to: {analysis_file}")
        
        return analysis_data
    
    def filter_nlp_papers(self, df: pd.DataFrame, categories: List[str]) -> pd.DataFrame:
        """Filter papers by specified categories"""
        logger.info(f"Filtering papers for categories: {categories}")
        
        # Create a mask for papers that have any of the specified categories
        mask = df['categories'].apply(
            lambda x: any(cat in x.split() if pd.notna(x) else [] for cat in categories)
        )
        
        filtered_df = df[mask].copy()
        logger.info(f"Found {len(filtered_df)} papers in specified categories")
        
        # Add year extraction if available
        if 'update_date' in filtered_df.columns:
            filtered_df['year'] = pd.to_datetime(filtered_df['update_date']).dt.year
        
        return filtered_df
    
    def contains_nlp_keywords(self, text: str) -> tuple[bool, List[str]]:
        """
        Check if text contains NLP-related keywords
        Returns tuple of (contains_keywords, matched_keywords)
        """
        if pd.isna(text) or not text:
            return False, []
        
        text_lower = text.lower()
        matched_keywords = []
        
        for keyword in self.all_nlp_keywords:
            # Use word boundaries for exact matching
            pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
            if re.search(pattern, text_lower):
                matched_keywords.append(keyword)
        
        return len(matched_keywords) > 0, matched_keywords
    
    def filter_nlp_papers_advanced(self, df: pd.DataFrame, 
                                  use_categories: bool = True,
                                  use_keywords: bool = True,
                                  min_keyword_matches: int = 1,
                                  target_papers: int = 1000) -> pd.DataFrame:
        """
        Advanced filtering combining categories and keywords with early stopping
        
        Args:
            df: Input dataframe
            use_categories: Whether to filter by NLP categories
            use_keywords: Whether to filter by keywords in title/abstract
            min_keyword_matches: Minimum number of keyword matches required
            target_papers: Stop filtering after finding this many papers (default 1000)
        """
        logger.info("Starting advanced NLP paper filtering with early stopping...")
        original_count = len(df)
        
        # Step 1: Extract dates for sorting
        logger.info("Extracting dates for chronological sorting...")
        if 'update_date' in df.columns:
            df['sort_date'] = pd.to_datetime(df['update_date'], errors='coerce')
        elif 'versions' in df.columns:
            df['sort_date'] = df['versions'].apply(
                lambda x: pd.to_datetime(x[-1]['created'] if x and len(x) > 0 else None, errors='coerce')
                if pd.notna(x) else None
            )
        else:
            logger.warning("No date field available for sorting")
            df['sort_date'] = pd.NaT
        
        # Step 2: Category-based filtering (fast)
        if use_categories:
            logger.info(f"Filtering by categories: {self.nlp_categories}")
            category_mask = df['categories'].apply(
                lambda x: any(cat in x.split() if pd.notna(x) else [] for cat in self.nlp_categories)
            )
            df = df[category_mask].copy()
            logger.info(f"After category filtering: {len(df):,} papers")
        
        # Step 3: Sort by date (newest first)
        df = df.sort_values('sort_date', ascending=False, na_position='last')
        logger.info(f"Sorted papers by date (newest first)")
        
        # Step 4: Apply keyword filtering incrementally until we have enough papers
        filtered_papers = []
        
        if use_keywords:
            logger.info(f"Applying keyword filtering incrementally (target: {target_papers} papers)...")
            
            batch_size = 1000  # Process in batches for efficiency
            total_processed = 0
            
            for start_idx in range(0, len(df), batch_size):
                if len(filtered_papers) >= target_papers:
                    break
                    
                end_idx = min(start_idx + batch_size, len(df))
                batch_df = df.iloc[start_idx:end_idx]
                total_processed = end_idx
                
                # Check keywords for this batch
                for idx, row in batch_df.iterrows():
                    # Check title
                    title_has_kw, title_keywords = self.contains_nlp_keywords(row['title'])
                    # Check abstract
                    abstract_has_kw, abstract_keywords = self.contains_nlp_keywords(row['abstract'])
                    
                    # Combine keywords
                    all_keywords = list(set(title_keywords + abstract_keywords))
                    
                    # Check if paper passes
                    if len(all_keywords) >= min_keyword_matches:
                        paper_dict = row.to_dict()
                        paper_dict['matched_nlp_keywords'] = all_keywords
                        paper_dict['keyword_match_count'] = len(all_keywords)
                        filtered_papers.append(paper_dict)
                        
                        if len(filtered_papers) >= target_papers:
                            logger.info(f"Reached target of {target_papers} papers after processing {total_processed:,} papers")
                            break
                
                if total_processed % 10000 == 0:
                    logger.info(f"Processed {total_processed:,} papers, found {len(filtered_papers)} matching papers so far...")
            
            # Create dataframe from filtered papers
            if filtered_papers:
                filtered_df = pd.DataFrame(filtered_papers)
            else:
                filtered_df = pd.DataFrame()
                
            logger.info(f"After keyword filtering: {len(filtered_df):,} papers (processed {total_processed:,} papers)")
        else:
            # No keyword filtering, just take top papers by date
            filtered_df = df.head(target_papers).copy()
        
        # Add filtering metadata
        if not filtered_df.empty:
            filtered_df['filter_method'] = 'advanced_incremental'
            filtered_df['used_categories'] = use_categories
            filtered_df['used_keywords'] = use_keywords
            
            # Add year extraction
            if 'sort_date' in filtered_df.columns:
                filtered_df['year'] = filtered_df['sort_date'].dt.year
        
        logger.info(f"\nFiltering Summary:")
        logger.info(f"Original papers: {original_count:,}")
        logger.info(f"Papers after category filter: {len(df):,}")
        logger.info(f"Final filtered papers: {len(filtered_df):,}")
        logger.info(f"Early stopping saved processing: {len(df) - total_processed if use_keywords else 0:,} papers")
        
        return filtered_df
    
    def get_latest_papers(self, df: pd.DataFrame, limit: int = 1000) -> pd.DataFrame:
        """
        Get the most recent papers from the filtered dataset
        """
        logger.info(f"Selecting {limit} most recent papers...")
        
        # Sort by update date or version date
        if 'year' in df.columns:
            # First try to sort by year, then by other date fields if available
            sort_columns = ['year']
            if 'update_date' in df.columns:
                sort_columns.append('update_date')
            elif 'versions' in df.columns:
                # Extract the latest version date
                df['latest_version_date'] = df['versions'].apply(
                    lambda x: x[-1]['created'] if x and len(x) > 0 else None
                    if pd.notna(x) else None
                )
                sort_columns.append('latest_version_date')
            
            latest_papers = df.sort_values(sort_columns, ascending=False).head(limit)
        else:
            logger.warning("No date information available, selecting first papers")
            latest_papers = df.head(limit)
        
        logger.info(f"Selected {len(latest_papers)} papers")
        
        if 'year' in latest_papers.columns:
            year_dist = latest_papers['year'].value_counts().sort_index(ascending=False)
            logger.info("Year distribution of selected papers:")
            for year, count in year_dist.head(10).items():
                logger.info(f"  {year}: {count} papers")
        
        return latest_papers
    
    def analyze_nlp_filtering_results(self, df: pd.DataFrame) -> Dict:
        """
        Analyze the results of NLP filtering
        """
        analysis = {}
        
        # Basic statistics
        analysis['total_papers'] = len(df)
        analysis['papers_with_abstract'] = df['abstract'].notna().sum()
        analysis['papers_with_title'] = df['title'].notna().sum()
        
        # Keyword analysis
        if 'matched_nlp_keywords' in df.columns:
            # Count keyword frequency
            all_matched_keywords = []
            for keywords in df['matched_nlp_keywords']:
                all_matched_keywords.extend(keywords)
            
            keyword_counts = Counter(all_matched_keywords)
            analysis['top_keywords'] = dict(keyword_counts.most_common(20))
            analysis['unique_keywords_matched'] = len(keyword_counts)
            
            # Keyword match distribution
            match_dist = df['keyword_match_count'].value_counts().sort_index()
            analysis['keyword_match_distribution'] = dict(match_dist)
        
        # Category analysis
        if 'categories' in df.columns:
            all_categories = []
            for cats in df['categories'].dropna():
                all_categories.extend(cats.split())
            
            category_counts = Counter(all_categories)
            analysis['top_categories'] = dict(category_counts.most_common(15))
        
        # Year distribution
        if 'year' in df.columns:
            year_dist = df['year'].value_counts().sort_index(ascending=False)
            analysis['year_distribution'] = dict(year_dist.head(10))
        
        return analysis
    
    def get_paper_stats(self, df: pd.DataFrame) -> Dict:
        """Get statistics about the filtered papers"""
        stats = {
            "total_papers": len(df),
            "papers_with_abstract": df['abstract'].notna().sum(),
            "unique_authors": len(set([author for authors in df['authors_parsed'].dropna() 
                                      for author in authors])) if 'authors_parsed' in df.columns else 0,
            "year_distribution": df['year'].value_counts().to_dict() if 'year' in df.columns else {},
        }
        
        logger.info(f"\n{'='*60}")
        logger.info("FILTERED DATASET STATISTICS:")
        logger.info(f"{'='*60}")
        for key, value in stats.items():
            if key != 'year_distribution':
                logger.info(f"{key}: {value}")
        
        if stats['year_distribution']:
            logger.info("\nYear Distribution (top 10):")
            for year, count in list(sorted(stats['year_distribution'].items(), reverse=True))[:10]:
                logger.info(f"  {year}: {count} papers")
        
        return stats


def main():
    """Main execution function"""
    pipeline = ArxivDataPipeline()
    
    # Step 1: Download dataset
    try:
        dataset_path = pipeline.download_dataset()
        logger.info(f"Dataset available at: {dataset_path}")
    except Exception as e:
        logger.error(f"Failed to download dataset: {e}")
        return
    
    # Step 2: Load metadata
    try:
        df = pipeline.load_metadata()
        logger.info(f"Loaded {len(df)} papers from metadata")
        logger.info(f"Columns: {df.columns.tolist()}")
    except Exception as e:
        logger.error(f"Failed to load metadata: {e}")
        return
    
    # Step 3: Explore categories
    category_analysis = pipeline.explore_categories(df)
    
    # Step 4: Show sample papers to understand structure
    logger.info(f"\n{'='*60}")
    logger.info("SAMPLE PAPER STRUCTURE:")
    logger.info(f"{'='*60}")
    if len(df) > 0:
        sample_paper = df.iloc[0].to_dict()
        for key, value in sample_paper.items():
            if isinstance(value, str) and len(value) > 200:
                value = value[:200] + "..."
            logger.info(f"{key}: {value}")
    
    logger.info("\n" + "="*60)
    logger.info("NEXT STEPS:")
    logger.info("="*60)
    logger.info("1. Review the categories above to select NLP-related ones")
    logger.info("2. Use filter_nlp_papers() with selected categories")
    logger.info("3. Download PDFs for filtered papers")
    logger.info("4. Convert PDFs to text format")
    logger.info("5. Prepare for Elasticsearch ingestion")


if __name__ == "__main__":
    main()