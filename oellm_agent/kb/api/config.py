#!/usr/bin/env python3
from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    kb_db_path: str = "kb/data/kb.sqlite3"
    kb_collection: str = "default"
    kb_embed_model: str = "BAAI/bge-small-zh-v1.5"
    kb_rerank_model: str = "BAAI/bge-reranker-base"
    kb_embed_cache_dir: Optional[str] = None  # 模型缓存目录，None 则使用默认
    kb_rerank_cache_dir: Optional[str] = None
    kb_chunk_size: int = 800
    kb_chunk_overlap: int = 120
    kb_upload_dir: str = "kb/uploads"
    kb_dense_k: int = 50
    kb_bm25_k: int = 50
    kb_w_dense: float = 0.6
    kb_w_bm25: float = 0.4

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
