# Paper Search Engine - Environment Setup Guide

## Service Configuration Options

The Paper Search Engine supports multiple backend configurations. Choose the setup that best fits your needs:

## Option 1: Full Stack (Recommended)

All services for complete functionality:

```bash
# Start all services
./make.sh services

# Configure backend to use Elasticsearch
echo "DATA_BACKEND=es" > backend/.env
echo "ES_URL=http://localhost:9200" >> backend/.env
echo "REDIS_URL=redis://localhost:6379/0" >> backend/.env
echo "MINIO_ENDPOINT=localhost:9000" >> backend/.env
echo "OPENAI_API_KEY=your_api_key" >> backend/.env
```

**Provides:**
- ✅ Fast search with Elasticsearch
- ✅ Session caching with Redis  
- ✅ PDF storage with MinIO
- ✅ AI chat with OpenAI

## Option 2: Minimal Setup (PostgreSQL)

Use PostgreSQL as the only external dependency:

```bash
# Start only PostgreSQL
docker run -d --name paper-search-pg \
  -e POSTGRES_DB=paper_search \
  -e POSTGRES_USER=user \
  -e POSTGRES_PASSWORD=password \
  -p 5432:5432 postgres:15-alpine

# Configure backend
echo "DATA_BACKEND=pg" > backend/.env
echo "PG_DSN=postgresql://user:password@localhost:5432/paper_search" >> backend/.env
echo "OPENAI_API_KEY=your_api_key" >> backend/.env
```

**Provides:**
- ✅ Full-text search with PostgreSQL
- ✅ AI chat with OpenAI
- ❌ No session caching
- ❌ No PDF storage

## Option 3: Search-Only (No External Dependencies)

For development/testing without Docker:

```bash
# No external services needed
echo "DATA_BACKEND=pg" > backend/.env
echo "PG_DSN=sqlite:///papers.db" >> backend/.env  # SQLite fallback
```

**Provides:**
- ⚠️ Limited search capabilities
- ❌ No AI chat
- ❌ No caching or PDF storage

## OpenAI API Key Setup

For chat functionality, you need an OpenAI API key:

1. Go to https://platform.openai.com/api-keys
2. Create a new API key
3. Add to your environment:

```bash
echo "OPENAI_API_KEY=sk-your-actual-key-here" >> backend/.env
```

**Cost Estimate:** ~$0.01-0.10 per chat interaction depending on context length.

## Service Health Checks

Check service availability:

```bash
# Backend health check
curl http://localhost:8000/api/health

# Elasticsearch
curl http://localhost:9200/_cluster/health

# PostgreSQL
docker exec paper-search-pg pg_isready -U user

# Redis
docker exec paper-search-redis redis-cli ping

# MinIO
curl http://localhost:9000/minio/health/live
```

## Troubleshooting Service Issues

### Elasticsearch Issues

```bash
# Check logs
docker logs paper-search-es

# Increase memory if needed
docker-compose down
# Edit docker-compose.yml: ES_JAVA_OPTS=-Xms2g -Xmx2g
docker-compose up -d elasticsearch
```

### PostgreSQL Issues

```bash
# Check logs  
docker logs paper-search-pg

# Reset database
docker-compose down
docker volume rm paper-search-engine_pg_data
docker-compose up -d postgres
```

### Port Conflicts

Default ports used:
- 3000: Next.js frontend
- 8000: FastAPI backend  
- 9200: Elasticsearch
- 5432: PostgreSQL
- 6379: Redis
- 9000: MinIO API
- 9001: MinIO Console

Change ports in `docker-compose.yml` if conflicts occur.

## Performance Tuning

### For Large Datasets (>1M papers)

1. **Elasticsearch Settings:**
```yaml
# In docker-compose.yml
environment:
  - "ES_JAVA_OPTS=-Xms4g -Xmx4g"
  - cluster.routing.allocation.disk.threshold_enabled=false
```

2. **PostgreSQL Settings:**
```yaml
# In docker-compose.yml  
command: >
  postgres
  -c shared_buffers=1GB
  -c work_mem=50MB
  -c maintenance_work_mem=512MB
```

3. **Backend Scaling:**
```bash
# Use more workers in production
gunicorn app.main:app -w 8 -k uvicorn.workers.UvicornWorker
```

### For Development (Fast Startup)

```bash
# Start only essential services
docker-compose up -d postgres redis

# Use smaller batch sizes for ingestion
curl -X POST "http://localhost:8000/api/ingest" \
     -d '{"batchSize": 100}'
```

## Data Sources

### Sample Data (Included)

The repository includes `data/arxiv-metadata-oai-snapshot.json` with a subset of ArXiv papers.

### Full ArXiv Dataset

Download the complete dataset:

```bash
# Warning: 3.5GB+ compressed, 25GB+ uncompressed
wget https://www.kaggle.com/datasets/Cornell-University/arxiv/download
unzip arxiv.zip
mv arxiv-metadata-oai-snapshot.json data/
```

### Custom Data

Follow the ArXiv JSON format:

```json
{"id": "paper_id", "title": "...", "abstract": "...", "authors": "A, B, C", "categories": "cat1 cat2", "update_date": "YYYY-MM-DD"}
```

Each paper should be on a separate line (JSONL format).

## Security Notes

- **Development Only:** Default passwords in `docker-compose.yml` are for development only
- **Production:** Use proper secrets management and secure passwords
- **API Keys:** Never commit API keys to version control
- **Network:** Bind services to localhost in production, use reverse proxy
