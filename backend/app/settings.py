from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # Backend configuration
    data_backend: str = "es"  # 'es' or 'pg' (ElasticSearch or PostgreSQL)
    debug: bool = True
    log_level: str = "info"
    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000"
    
    # Elasticsearch
    es_url: str = "http://localhost:9200"
    es_index: str = "papers"
    
    # PostgreSQL
    pg_dsn: str = "postgresql://user:password@localhost:5432/paper_search"
    
    # Redis
    redis_url: str = "redis://localhost:6379/0"
    redis_session_ttl: int = 3600  # 1 hour
    
    # MinIO
    minio_endpoint: Optional[str] = None
    minio_access_key: Optional[str] = None
    minio_secret_key: Optional[str] = None
    minio_bucket: str = "papers"
    minio_secure: bool = False
    
    # OpenAI
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-3.5-turbo"
    
    # Search settings
    default_page_size: int = 20
    max_page_size: int = 100
    
    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
