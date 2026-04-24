from pydantic import BaseModel, Field
from typing import List, Optional


class DocMeta(BaseModel):
    doc_id: str
    title: str
    source_type: str = "manual"
    tags: List[str] = Field(default_factory=list)


class IngestRequest(BaseModel):
    texts: List[str]
    meta: Optional[DocMeta] = None
    chunk_size: int = 800
    chunk_overlap: int = 120


class IngestFileRequest(BaseModel):
    title: str = ""
    source_type: str = "manual"
    tags: List[str] = Field(default_factory=list)
    chunk_size: int = 800
    chunk_overlap: int = 120


class UploadResponse(BaseModel):
    doc_id: str
    filename: str
    suffix: str
    message: str


class ProcessRequest(BaseModel):
    doc_id: str
    title: str = ""
    source_type: str = "manual"
    tags: List[str] = Field(default_factory=list)
    chunk_size: int = 800
    chunk_overlap: int = 120


class ProcessResponse(BaseModel):
    total_chunks: int
    doc_id: str
    message: str


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    source_type: str = ""
    dense_k: int = 50
    bm25_k: int = 50
    w_dense: float = 0.6
    w_bm25: float = 0.4
    rerank: bool = True
    rerank_top_k: int = 20


class ChunkResult(BaseModel):
    chunk_id: str
    doc_id: str
    title: str
    source_type: str
    tags: List[str]
    chunk_index: int
    content: str
    source_path: str
    score: float
    dense_score: float = 0.0
    bm25_score: float = 0.0
    rerank_score: Optional[float] = None


class IngestResponse(BaseModel):
    total_chunks: int
    doc_id: str
    message: str


class SearchResponse(BaseModel):
    query: str
    total: int
    results: List[ChunkResult]
