# Two-Week Paper Search & Q&A Chatbot Project Plan

## Project Overview
Build a chatbot for:
1. Finding papers (semantic search)
2. Answering questions via RAG from papers

Timeline: 2 weeks  
Dataset: 1000 papers from ArXiv (NLP or CV domain)  
Reference: Simplified version of Asta PaperFinder

## Strategic Recommendations

### 1. **Simplify the Architecture**
Instead of implementing all PaperFinder agents, focus on:
- **One search agent** (BroadSearchAgent or FastBroadSearchAgent)
- **Query analyzer** (simplified version)
- **RAG pipeline** for Q&A (not in PaperFinder)

### 2. **Dataset Selection**
- **ArXiv Dataset on Kaggle**: [arxiv-dataset](https://www.kaggle.com/Cornell-University/arxiv) - 1.7M papers with abstracts
- **Papers with Code**: Has full-text for many papers
- **S2ORC**: Semantic Scholar corpus (subset with full-text)
- Start with **CS papers only** (filter for cs.* categories)

### 3. **Core Components to Build**

#### Week 1: Paper Search
```python
# Simplified architecture
1. Data ingestion pipeline
2. Vector database (Elasticsearch with dense vectors)
3. Search API with 2-3 methods:
   - Semantic search (embeddings)
   - Keyword search (BM25)
   - Hybrid search
4. Basic relevance scoring
```

#### Week 2: RAG Q&A
```python
1. Document chunking strategy
2. Retrieval pipeline
3. Context-aware answering
4. Citation tracking
```

### 4. **Tech Stack Recommendations**

```python
# Faster alternatives to PaperFinder's stack
- Framework: FastAPI (keep) or LangChain + FastAPI
- Vector DB: Elasticsearch (with vector search)
- Embeddings: OpenAI text-embedding-3-small or Cohere
- LLM: GPT-4o-mini for cost efficiency
- Frontend: Gradio or Streamlit for quick prototype
```

### 5. **Key Simplifications from PaperFinder**

| PaperFinder | Your Version |
|------------|--------------|
| 10+ specialized agents | 2-3 generic search methods |
| Complex query analysis | Simple intent classification |
| Multiple external APIs | Local data + 1-2 APIs |
| Production-grade caching | Simple in-memory cache |
| Elaborate ranking | Basic relevance scoring |

### 6. **Implementation Shortcuts**

```python
# Use existing libraries
from langchain.document_loaders import ArxivLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import ElasticsearchStore
from langchain.chains import RetrievalQA

# Pre-built components save days of work
```

### 7. **Quick Win Features**

Focus on:
- **Semantic search** (most valuable)
- **Abstract + title search** (skip full-text initially)
- **Simple filters** (year, authors, venue)
- **Top-k retrieval** (no complex ranking)

### 8. **Data Pipeline Priority**

```python
# Day 1-2: Get data flowing
1. Download ArXiv dataset (JSON format)
2. Filter to 1000 CS papers
3. Extract: title, abstract, authors, year, arxiv_id
4. Generate embeddings
5. Store in vector DB
```

### 9. **RAG Implementation Tips**

```python
# Efficient RAG setup
- Chunk size: 512 tokens with 50 token overlap
- Retrieve top 5-10 chunks
- Use prompt template:
  "Answer based on these papers: {context}"
- Track source papers for citations
```

### 10. **Project Structure**

```
your-paper-bot/
├── api/
│   ├── main.py          # FastAPI app
│   ├── search.py         # Search endpoints
│   └── qa.py            # Q&A endpoints
├── core/
│   ├── vectordb.py      # Vector store operations
│   ├── retriever.py     # Search logic
│   └── rag.py           # RAG chain
├── data/
│   ├── ingest.py        # Data loading
│   └── preprocess.py    # Chunking, cleaning
└── frontend/
    └── app.py           # Gradio interface
```

### 11. **Time-Saving Resources**

- **LangChain Templates**: [langchain-templates](https://github.com/langchain-ai/langchain/tree/master/templates)
- **OpenAI Cookbook**: RAG examples
- **Pinecone Examples**: [pinecone-examples](https://github.com/pinecone-io/examples)
- **Gradio Templates**: For quick UI

### 12. **MVP Feature Set**

**Week 1 Deliverables:**
- Upload and index 1000 papers
- Semantic search API
- Basic filtering (year, author)
- Simple relevance scoring

**Week 2 Deliverables:**
- RAG-based Q&A
- Citation tracking
- Basic UI with Gradio
- Docker container

### 13. **Avoid These Time Sinks**

- Complex query parsing (use simple regex/rules)
- Multiple agent coordination
- Custom ranking algorithms
- Full-text processing initially
- Complex caching systems

### 14. **Testing Strategy**

```python
# Quick validation
test_queries = [
    "transformer architectures",
    "papers by Hinton",
    "BERT improvements 2023"
]

test_questions = [
    "What are the main contributions of BERT?",
    "How do transformers handle long sequences?"
]
```

### 15. **Deployment Shortcut**

```yaml
# docker-compose.yml for everything
services:
  api:
    build: .
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.11.0
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
  redis:
    image: redis:alpine
```

## Recommended Daily Schedule

**Days 1-2**: Data pipeline, Elasticsearch setup  
**Days 3-4**: Search API implementation  
**Days 5-6**: Frontend, testing search  
**Days 7-8**: RAG pipeline setup  
**Days 9-10**: Q&A integration  
**Days 11-12**: UI completion, testing  
**Days 13-14**: Documentation, deployment, buffer  

## Sample Implementation Code

### Quick Start Data Ingestion

```python
# data/ingest.py
import json
from typing import List, Dict
from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer

class PaperIngester:
    def __init__(self):
        self.es = Elasticsearch(['http://localhost:9200'])
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.index_name = "papers"
        
    def load_arxiv_papers(self, filepath: str, limit: int = 1000) -> List[Dict]:
        """Load ArXiv papers from JSON file"""
        papers = []
        with open(filepath, 'r') as f:
            for i, line in enumerate(f):
                if i >= limit:
                    break
                paper = json.loads(line)
                # Filter for CS papers
                if any(cat.startswith('cs.') for cat in paper.get('categories', '').split()):
                    papers.append({
                        'id': paper['id'],
                        'title': paper['title'],
                        'abstract': paper['abstract'],
                        'authors': paper['authors'],
                        'year': paper['year'],
                        'categories': paper['categories']
                    })
        return papers[:limit]
    
    def create_index(self):
        """Create Elasticsearch index with dense vector mapping"""
        mapping = {
            "mappings": {
                "properties": {
                    "title": {"type": "text"},
                    "abstract": {"type": "text"},
                    "authors": {"type": "keyword"},
                    "year": {"type": "integer"},
                    "categories": {"type": "keyword"},
                    "embedding": {
                        "type": "dense_vector",
                        "dims": 384,  # all-MiniLM-L6-v2 dimension
                        "index": True,
                        "similarity": "cosine"
                    }
                }
            }
        }
        
        # Delete index if exists
        if self.es.indices.exists(index=self.index_name):
            self.es.indices.delete(index=self.index_name)
        
        # Create new index
        self.es.indices.create(index=self.index_name, body=mapping)
    
    def index_papers(self, papers: List[Dict]):
        """Generate embeddings and index papers"""
        # Combine title and abstract for embedding
        texts = [f"{p['title']} {p['abstract']}" for p in papers]
        embeddings = self.encoder.encode(texts)
        
        # Prepare bulk indexing
        actions = []
        for i, (paper, embedding) in enumerate(zip(papers, embeddings)):
            action = {
                "_index": self.index_name,
                "_id": paper['id'],
                "_source": {
                    **paper,
                    "embedding": embedding.tolist()
                }
            }
            actions.append(action)
        
        # Bulk index
        helpers.bulk(self.es, actions)
        self.es.indices.refresh(index=self.index_name)
```

### Simple Search API

```python
# api/search.py
from fastapi import FastAPI, Query
from typing import List, Optional
from pydantic import BaseModel
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

app = FastAPI()
es = Elasticsearch(['http://localhost:9200'])
encoder = SentenceTransformer('all-MiniLM-L6-v2')

class SearchResult(BaseModel):
    id: str
    title: str
    abstract: str
    authors: List[str]
    score: float

@app.post("/search")
async def search_papers(
    query: str,
    limit: int = 10,
    year_min: Optional[int] = None,
    year_max: Optional[int] = None
):
    # Encode query
    query_vector = encoder.encode(query).tolist()
    
    # Build query with filters
    must_conditions = []
    if year_min or year_max:
        range_clause = {"year": {}}
        if year_min:
            range_clause["year"]["gte"] = year_min
        if year_max:
            range_clause["year"]["lte"] = year_max
        must_conditions.append({"range": range_clause})
    
    # Hybrid search: combine vector and text search
    search_body = {
        "size": limit,
        "query": {
            "bool": {
                "must": must_conditions,
                "should": [
                    {
                        "script_score": {
                            "query": {"match_all": {}},
                            "script": {
                                "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                                "params": {"query_vector": query_vector}
                            }
                        }
                    },
                    {
                        "multi_match": {
                            "query": query,
                            "fields": ["title^2", "abstract"],
                            "type": "best_fields"
                        }
                    }
                ]
            }
        }
    }
    
    # Execute search
    response = es.search(index="papers", body=search_body)
    
    return [
        SearchResult(
            id=hit['_source']['id'],
            title=hit['_source']['title'],
            abstract=hit['_source']['abstract'],
            authors=hit['_source']['authors'],
            score=hit['_score']
        )
        for hit in response['hits']['hits']
    ]
```

### Basic RAG Implementation

```python
# core/rag.py
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import ElasticsearchStore
from elasticsearch import Elasticsearch

class PaperQA:
    def __init__(self):
        self.es = Elasticsearch(['http://localhost:9200'])
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
        # Create vector store using Elasticsearch
        self.vectorstore = ElasticsearchStore(
            es_connection=self.es,
            index_name="papers",
            embedding=self.embeddings,
            vector_query_field="embedding"
        )
        
        # Create QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_kwargs={"k": 5}
            ),
            return_source_documents=True
        )
    
    def answer_question(self, question: str):
        """Answer question based on paper corpus"""
        result = self.qa_chain({"query": question})
        
        # Extract sources
        sources = []
        for doc in result['source_documents']:
            sources.append({
                'title': doc.metadata.get('title'),
                'authors': doc.metadata.get('authors'),
                'id': doc.metadata.get('id')
            })
        
        return {
            'answer': result['result'],
            'sources': sources
        }

    def hybrid_search(self, question: str, k: int = 5):
        """Perform hybrid search using both text and vector similarity"""
        from sentence_transformers import SentenceTransformer
        
        encoder = SentenceTransformer('all-MiniLM-L6-v2')
        query_vector = encoder.encode(question).tolist()
        
        # Elasticsearch hybrid query
        search_body = {
            "size": k,
            "query": {
                "bool": {
                    "should": [
                        {
                            "script_score": {
                                "query": {"match_all": {}},
                                "script": {
                                    "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                                    "params": {"query_vector": query_vector}
                                }
                            }
                        },
                        {
                            "multi_match": {
                                "query": question,
                                "fields": ["title^2", "abstract"],
                                "type": "best_fields"
                            }
                        }
                    ]
                }
            }
        }
        
        response = self.es.search(index="papers", body=search_body)
        return response['hits']['hits']
```

### Gradio Frontend

```python
# frontend/app.py
import gradio as gr
import requests

def search_papers(query, year_min, year_max):
    response = requests.post(
        "http://localhost:8000/search",
        json={
            "query": query,
            "limit": 10,
            "year_min": year_min if year_min else None,
            "year_max": year_max if year_max else None
        }
    )
    results = response.json()
    
    output = ""
    for paper in results:
        output += f"**{paper['title']}**\n"
        output += f"Authors: {', '.join(paper['authors'])}\n"
        output += f"Score: {paper['score']:.3f}\n"
        output += f"{paper['abstract'][:200]}...\n\n"
    
    return output

def answer_question(question):
    response = requests.post(
        "http://localhost:8000/qa",
        json={"question": question}
    )
    result = response.json()
    
    output = f"**Answer:** {result['answer']}\n\n"
    output += "**Sources:**\n"
    for source in result['sources']:
        output += f"- {source['title']} by {', '.join(source['authors'])}\n"
    
    return output

# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Paper Search & Q&A System")
    
    with gr.Tab("Search Papers"):
        query_input = gr.Textbox(label="Search Query")
        year_min = gr.Number(label="Min Year", precision=0)
        year_max = gr.Number(label="Max Year", precision=0)
        search_btn = gr.Button("Search")
        search_output = gr.Markdown()
        
        search_btn.click(
            search_papers,
            inputs=[query_input, year_min, year_max],
            outputs=search_output
        )
    
    with gr.Tab("Ask Questions"):
        question_input = gr.Textbox(label="Your Question")
        qa_btn = gr.Button("Get Answer")
        qa_output = gr.Markdown()
        
        qa_btn.click(
            answer_question,
            inputs=question_input,
            outputs=qa_output
        )

demo.launch()
```

## Key Success Factors

1. **Start Simple**: Get basic search working before adding complexity
2. **Use Existing Tools**: LangChain, Gradio, Qdrant save development time
3. **Focus on Core Features**: Search and Q&A are the MVPs
4. **Iterate Quickly**: Deploy early, improve based on testing
5. **Document as You Go**: Keep track of decisions and setup steps

This approach gives you a working system in 2 weeks while leaving room to add PaperFinder's advanced features later.