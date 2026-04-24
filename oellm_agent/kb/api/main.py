#!/usr/bin/env python3
"""
Knowledge Base API — FastAPI entrypoint.
All endpoints: upload, process, chunk, embed, store, search, rerank.
"""
import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import settings
from schemas import (
    ChunkResult,
    IngestRequest,
    IngestResponse,
    ProcessRequest,
    ProcessResponse,
    SearchRequest,
    SearchResponse,
    UploadResponse,
)
from service import KBService


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Knowledge Base API",
    description="文件上传、文本分块、向量化、存储、检索、重排序的完整知识库接口",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_kb: KBService | None = None


def get_kb() -> KBService:
    global _kb
    if _kb is None:
        _kb = KBService(
            db_path=Path(settings.kb_db_path),
            collection=settings.kb_collection,
            embed_model=settings.kb_embed_model,
            rerank_model=settings.kb_rerank_model,
            embed_cache_dir=settings.kb_embed_cache_dir,
            rerank_cache_dir=settings.kb_rerank_cache_dir,
            chunk_size=settings.kb_chunk_size,
            chunk_overlap=settings.kb_chunk_overlap,
            upload_dir=Path(settings.kb_upload_dir),
            dense_k=settings.kb_dense_k,
            bm25_k=settings.kb_bm25_k,
            w_dense=settings.kb_w_dense,
            w_bm25=settings.kb_w_bm25,
        )
    return _kb


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    kb = get_kb()
    return {"status": "ok", **kb.stats()}


# ---------------------------------------------------------------------------
# Ingestion — upload & process files
# ---------------------------------------------------------------------------

@app.post("/ingest/file/upload", response_model=UploadResponse)
async def ingest_file_upload(
    file: UploadFile = File(...),
    title: str = Form(""),
    source_type: str = Form("manual"),
    tags: str = Form(""),
    chunk_size: int = Form(800),
    chunk_overlap: int = Form(120),
):
    """
    Step 1 of 2 — upload a file and save it to disk.
    Returns a doc_id; call /ingest/file/process with that doc_id to process it.
    """
    suffix = Path(file.filename or "tmp").suffix.lower()
    if suffix not in {".txt", ".md", ".json", ".pdf", ".xlsx", ".xls", ".docx"}:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {suffix}. Supported: .txt .md .json .pdf .xlsx .xls .docx",
        )

    kb = get_kb()
    upload_dir = Path(settings.kb_upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    doc_id = str(uuid.uuid4())
    title = title or Path(file.filename or "").stem
    saved_path = upload_dir / file.filename

    with open(saved_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    kb.upload_file_record(
        doc_id=doc_id,
        file_path=saved_path,
        filename=file.filename,
        title=title,
        source_type=source_type,
        tags=tags.split(",") if tags else [],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    return UploadResponse(
        doc_id=doc_id,
        filename=file.filename,
        suffix=suffix,
        message=f"文件 {file.filename} 已保存，doc_id={doc_id}，请调用 /ingest/file/process 进行处理",
    )


@app.post("/ingest/file/process", response_model=ProcessResponse)
async def ingest_file_process(request: ProcessRequest):
    """
    Step 2 of 2 — read the file saved for this doc_id, extract text,
    chunk, embed, and store.  Optionally override title/tags/chunk_size.
    """
    kb = get_kb()
    try:
        n, _ = kb.process_doc(
            doc_id=request.doc_id,
            title=request.title,
            source_type=request.source_type,
            tags=request.tags,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return ProcessResponse(
        total_chunks=n,
        doc_id=request.doc_id,
        message=f"成功处理 doc_id={request.doc_id}，入库 {n} 个 chunk",
    )


@app.post("/ingest/text", response_model=IngestResponse)
async def ingest_text(request: IngestRequest):
    """
    Ingest raw text(s) directly — no file upload needed.
    """
    if not request.texts:
        raise HTTPException(status_code=400, detail="texts cannot be empty")

    kb = get_kb()
    doc_id = request.meta.doc_id if request.meta else ""
    title = request.meta.title if request.meta else "inline_doc"
    source_type = request.meta.source_type if request.meta else "manual"
    tags = request.meta.tags if request.meta else []

    if not doc_id:
        import hashlib, time
        doc_id = hashlib.sha1(
            (str(time.time()) + "".join(request.texts)).encode()
        ).hexdigest()[:16]

    try:
        n, assigned_id = kb.ingest_texts(
            texts=request.texts,
            doc_id=doc_id,
            title=title,
            source_type=source_type,
            tags=tags,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return IngestResponse(
        total_chunks=n,
        doc_id=assigned_id,
        message=f"成功处理 {n} 个文本 chunk",
    )


# ---------------------------------------------------------------------------
# Search — dense + BM25 hybrid + optional rerank
# ---------------------------------------------------------------------------

@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """
    Hybrid search: FAISS dense vector + BM25 keyword, fused with weighted sum.
    Optional cross-encoder reranking for better relevance.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="query cannot be empty")

    kb = get_kb()
    try:
        results = kb.search(
            query=request.query,
            top_k=request.top_k,
            source_type=request.source_type,
            dense_k=request.dense_k,
            bm25_k=request.bm25_k,
            w_dense=request.w_dense,
            w_bm25=request.w_bm25,
            rerank=request.rerank,
            rerank_top_k=request.rerank_top_k,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    chunks = [
        ChunkResult(
            chunk_id=r["chunk_id"],
            doc_id=r["doc_id"],
            title=r["title"],
            source_type=r["source_type"],
            tags=r["tags"],
            chunk_index=r["chunk_index"],
            content=r["content"],
            source_path=r["source_path"],
            score=r["score"],
            dense_score=r["dense_score"],
            bm25_score=r["bm25_score"],
            rerank_score=r.get("rerank_score"),
        )
        for r in results
    ]

    return SearchResponse(
        query=request.query,
        total=len(chunks),
        results=chunks,
    )


@app.get("/search", response_model=SearchResponse)
async def search_get(
    q: str = Query(..., description="search query"),
    top_k: int = Query(5, ge=1, le=100),
    source_type: str = Query("", description="filter by source_type"),
    rerank: bool = Query(True),
    rerank_top_k: int = Query(20, ge=1, le=100),
):
    """GET version of /search for quick testing."""
    return await search(
        SearchRequest(
            query=q,
            top_k=top_k,
            source_type=source_type,
            rerank=rerank,
            rerank_top_k=rerank_top_k,
        )
    )


# ---------------------------------------------------------------------------
# Management
# ---------------------------------------------------------------------------

@app.get("/stats")
async def stats():
    """KB statistics: total chunks, docs, index size."""
    return get_kb().stats()


@app.delete("/doc/{doc_id}")
async def delete_doc(doc_id: str):
    """
    Delete all chunks belonging to a doc_id.
    Note: FAISS index will be rebuilt without the deleted doc's vectors.
    """
    kb = get_kb()
    count = kb.delete_doc(doc_id)
    if count == 0:
        raise HTTPException(status_code=404, detail=f"doc_id '{doc_id}' not found")
    return {"deleted": count, "doc_id": doc_id}


@app.delete("/kb")
async def clear_kb():
    """Clear all data — USE WITH CAUTION."""
    kb = get_kb()
    kb.clear()
    return {"message": "knowledge base cleared"}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
