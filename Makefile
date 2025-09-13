# Makefile for Paper Search Engine

.PHONY: help install test test-unit test-integration test-all lint format clean docker-up docker-down ingest search

help:  ## Show this help message
	@echo "Paper Search Engine - Available Commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Development Setup
install:  ## Install dependencies
	pip install -r requirements.txt
	pip install -r tests/requirements.txt

install-dev:  ## Install development dependencies
	pip install -r requirements.txt
	pip install -r tests/requirements.txt
	pip install black flake8 mypy pre-commit

# Testing
test:  ## Run all tests
	pytest

test-unit:  ## Run unit tests only
	pytest tests/unit/ -v

test-integration:  ## Run integration tests (requires services)
	pytest tests/integration/ -v -m integration

test-fast:  ## Run fast tests only (exclude slow markers)
	pytest -m "not slow" -v

test-embedder:  ## Run embedder tests
	pytest -m embedder -v

test-chunker:  ## Run chunker tests
	pytest -m chunker -v

test-indexer:  ## Run indexer tests
	pytest -m indexer -v

test-search:  ## Run search tests
	pytest -m search -v

test-pipeline:  ## Run pipeline tests
	pytest -m pipeline -v

test-coverage:  ## Run tests with coverage
	pytest --cov=data_pipeline --cov=backend --cov-report=html --cov-report=term

test-parallel:  ## Run tests in parallel
	pytest -n auto

# Code Quality
lint:  ## Run linting
	flake8 data_pipeline/ backend/ tests/
	black --check data_pipeline/ backend/ tests/

format:  ## Format code
	black data_pipeline/ backend/ tests/

type-check:  ## Run type checking
	mypy data_pipeline/ backend/

quality: lint type-check  ## Run all quality checks

# Docker Services
docker-up:  ## Start all services (ES, MinIO, etc.)
	docker-compose up -d

docker-down:  ## Stop all services
	docker-compose down

docker-logs:  ## Show docker logs
	docker-compose logs -f

docker-restart:  ## Restart services
	docker-compose restart

# Data Pipeline
ingest:  ## Run ingestion pipeline
	python data_pipeline/ingest_papers.py

ingest-sample:  ## Ingest first 10 files for testing
	python data_pipeline/ingest_papers.py --max-files 10

ingest-resume:  ## Resume ingestion from specific paper
	@read -p "Resume from paper ID: " paper_id; \
	python data_pipeline/ingest_papers.py --resume-from $$paper_id

# Search Testing
search-test:  ## Test search functionality
	python backend/services/search_service.py

test-implementation:  ## Run implementation test script
	python test_implementation.py

# Cleanup
clean:  ## Clean up temporary files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -delete
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf dist/
	rm -rf build/
	rm -rf *.egg-info/

clean-docker:  ## Clean up docker volumes and containers
	docker-compose down -v
	docker system prune -f

# Development Workflow
dev-setup: install docker-up  ## Complete development setup
	@echo "Waiting for services to start..."
	sleep 10
	make test-implementation

dev-test: test-unit test-implementation  ## Quick development test cycle

dev-full: quality test docker-up ingest-sample search-test  ## Full development workflow

# Production
build:  ## Build docker images
	docker-compose build

deploy:  ## Deploy services
	docker-compose up -d --build

# Monitoring
check-services:  ## Check service health
	@echo "Checking Elasticsearch..."
	@curl -s http://localhost:9202/_cluster/health | jq '.status' || echo "ES not available"
	@echo "Checking MinIO..."
	@curl -s http://localhost:9002/minio/health/live && echo "MinIO OK" || echo "MinIO not available"

stats:  ## Show index statistics
	@python -c "from data_pipeline.es_indexer import ESIndexer; indexer = ESIndexer(es_host='localhost:9202'); print(indexer.get_index_stats())"

# Documentation
docs:  ## Generate documentation
	@echo "Generating documentation..."
	@echo "See README_SEARCH_ENGINE.md for usage instructions"

# Benchmarks
benchmark:  ## Run performance benchmarks
	pytest tests/ -m "benchmark" --benchmark-only

# Database Management
reset-index:  ## Reset Elasticsearch index
	@python -c "from data_pipeline.es_indexer import ESIndexer; indexer = ESIndexer(es_host='localhost:9202'); indexer.create_index(force=True)"

backup-index:  ## Backup Elasticsearch index
	@echo "Index backup functionality not implemented yet"

# Utilities
count-papers:  ## Count papers in data directory
	@find data/processed/markdown -name "*.md" -type f | wc -l

show-config:  ## Show current configuration
	@echo "=== Current Configuration ==="
	@echo "ES Host: $${ES_HOST:-localhost:9202}"
	@echo "BGE Model: $${BGE_MODEL_NAME:-BAAI/bge-large-en-v1.5}"
	@echo "Index Name: $${ES_INDEX_NAME:-papers}"
	@echo "Chunk Size: $${CHUNK_SIZE:-512}"