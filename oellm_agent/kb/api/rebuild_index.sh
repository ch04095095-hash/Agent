#!/bin/bash
# Rebuild the FAISS index from existing SQLite metadata.
# Useful when the .index file is corrupted or missing.

cd "$(dirname "$0")"
source .venv/bin/activate
python3 -c "
from pathlib import Path
from service import KBService
from config import settings

kb = KBService(
    db_path=Path(settings.kb_db_path),
    collection=settings.kb_collection,
    embed_model=settings.kb_embed_model,
    rerank_model=settings.kb_rerank_model,
    chunk_size=settings.kb_chunk_size,
    chunk_overlap=settings.kb_chunk_overlap,
    upload_dir=Path(settings.kb_upload_dir),
    dense_k=settings.kb_dense_k,
    bm25_k=settings.kb_bm25_k,
    w_dense=settings.kb_w_dense,
    w_bm25=settings.kb_w_bm25,
)
kb._rebuild_faiss()
print('FAISS index rebuilt.')
print(kb.stats())
"
