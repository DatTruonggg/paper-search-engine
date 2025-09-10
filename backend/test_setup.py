#!/usr/bin/env python3

import sys
import os
import json
from pathlib import Path

# Add the app directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

try:
    # Test basic imports
    print("Testing imports...")
    from app.schemas import Paper, SearchRequest
    from app.settings import settings
    from app.services.tokenizer import TokenizerService
    print("✅ Basic imports successful")
    
    # Test data normalization  
    print("\nTesting data normalization...")
    from app.services.ingest import IngestService
    
    ingest_service = IngestService()
    
    # Sample ArXiv paper data
    sample_paper = {
        "id": "0704.0001",
        "title": "Test Paper Title",
        "abstract": "This is a test abstract for verification purposes.",
        "authors": "John Doe, Jane Smith, Alice Johnson",
        "categories": "cs.LG cs.AI",
        "update_date": "2023-01-15",
        "doi": "10.1000/test"
    }
    
    paper = ingest_service.normalize_paper(sample_paper)
    print(f"✅ Paper normalized: {paper.title}")
    print(f"   Authors: {len(paper.authors)} ({', '.join(paper.authors[:2])}...)")
    print(f"   Categories: {paper.categories}")
    print(f"   Year: {paper.year}")
    
    # Test tokenizer
    print("\nTesting tokenizer...")
    tokenizer = TokenizerService()
    tokens = tokenizer.tokenize("machine learning neural networks")
    print(f"✅ Tokenized query: {tokens}")
    
    # Test settings
    print(f"\nTesting settings...")
    print(f"✅ Backend: {settings.data_backend}")
    print(f"✅ Debug mode: {settings.debug}")
    
    print("\n🎉 All tests passed! Backend is ready.")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
