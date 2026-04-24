#!/usr/bin/env python3
import hashlib
import json
import pickle
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer


@dataclass
class KBChunk:
    chunk_id: str
    doc_id: str
    title: str
    source_type: str
    tags: List[str]
    content: str
    chunk_index: int
    source_path: str


class RecursiveChunker:
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 120):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str) -> List[str]:
        text = re.sub(r"\n{3,}", "\n\n", text.strip())
        if not text:
            return []

        separators = ["\n\n", "\n", "。", "；", "，", " "]
        chunks = [text]

        for sep in separators:
            new_chunks: List[str] = []
            for c in chunks:
                if len(c) <= self.chunk_size:
                    new_chunks.append(c)
                else:
                    parts = c.split(sep)
                    if len(parts) == 1:
                        new_chunks.append(c)
                    else:
                        buff = ""
                        join_sep = sep if sep not in {" ", "", None} else " "
                        for p in parts:
                            p = p.strip()
                            if not p:
                                continue
                            cand = f"{buff}{join_sep if buff else ''}{p}"
                            if len(cand) <= self.chunk_size:
                                buff = cand
                            else:
                                if buff:
                                    new_chunks.append(buff)
                                buff = p
                        if buff:
                            new_chunks.append(buff)
            chunks = new_chunks

        final_chunks: List[str] = []
        for c in chunks:
            c = c.strip()
            if not c:
                continue
            if len(c) <= self.chunk_size:
                final_chunks.append(c)
            else:
                step = max(1, self.chunk_size - self.chunk_overlap)
                for i in range(0, len(c), step):
                    final_chunks.append(c[i : i + self.chunk_size])
        return final_chunks


class FileReader:
    @staticmethod
    def read(path: Path) -> Tuple[str, Dict]:
        ext = path.suffix.lower()
        if ext in {".txt", ".md"}:
            return path.read_text(encoding="utf-8"), {}
        if ext == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                meta = {
                    "doc_id": data.get("doc_id", path.stem),
                    "title": data.get("title", path.stem),
                    "source_type": data.get("source_type", "manual"),
                    "tags": data.get("tags", []),
                }
                content = data.get("content")
                if content:
                    return str(content), meta
            return json.dumps(data, ensure_ascii=False, indent=2), {}
        if ext == ".pdf":
            from pypdf import PdfReader

            reader = PdfReader(str(path))
            pages = []
            for p in reader.pages:
                pages.append(p.extract_text() or "")
            return "\n\n".join(pages), {}
        raise ValueError(f"unsupported file type: {ext}")


class VectorKB:
    def __init__(
        self,
        db_path: Path,
        collection: str = "coal_truck_kb",
        embed_model: str = "BAAI/bge-small-zh-v1.5",
        rerank_model: str = "BAAI/bge-reranker-base",
    ):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.collection = collection

        # FAISS index file stored next to sqlite
        self.faiss_path = self.db_path.parent / f"{self.collection}.index"
        self.chunk_id_map_path = self.db_path.parent / f"{self.collection}.ids.pkl"

        self.embedder = SentenceTransformer(embed_model)
        self.reranker = CrossEncoder(rerank_model, device="cpu")
        sample_vec = self.embedder.encode(
            ["test"],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0]
        self.vector_size = int(sample_vec.shape[0])

        self._init_sqlite()
        self._init_faiss()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_sqlite(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS kb_chunks (
                    chunk_id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    tags_json TEXT NOT NULL,
                    content TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    source_path TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT (datetime('now'))
                );
                """
            )

    def _init_faiss(self) -> None:
        if self.faiss_path.exists():
            self.index = faiss.read_index(str(self.faiss_path))
            with open(self.chunk_id_map_path, "rb") as f:
                self.chunk_ids: List[str] = pickle.load(f)
        else:
            self.index = faiss.IndexFlatIP(self.vector_size)
            self.chunk_ids = []

    def _save_faiss(self) -> None:
        faiss.write_index(self.index, str(self.faiss_path))
        with open(self.chunk_id_map_path, "wb") as f:
            pickle.dump(self.chunk_ids, f)

    def _embed(self, texts: Sequence[str]) -> np.ndarray:
        arr = self.embedder.encode(
            list(texts),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(arr, dtype=np.float32)

    def ingest_chunks(self, chunks: List[KBChunk], batch_size: int = 64) -> int:
        if not chunks:
            return 0

        with self._connect() as conn:
            for c in chunks:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO kb_chunks(
                        chunk_id, doc_id, title, source_type, tags_json,
                        content, chunk_index, source_path
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        c.chunk_id,
                        c.doc_id,
                        c.title,
                        c.source_type,
                        json.dumps(c.tags, ensure_ascii=False),
                        c.content,
                        c.chunk_index,
                        c.source_path,
                    ),
                )

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            vectors = self._embed([c.content for c in batch])
            self.index.add(vectors)
            self.chunk_ids.extend(c.chunk_id for c in batch)

        self._save_faiss()
        return len(chunks)

    def search(
        self,
        query: str,
        top_k: int = 5,
        source_type: Optional[str] = None,
        dense_k: int = 30,
        bm25_k: int = 30,
        w_dense: float = 0.7,
        w_bm25: float = 0.3,
        rerank: bool = True,
        rerank_top_k: int = 20,
    ) -> List[Dict]:
        dense_hits = self._dense_search(query, dense_k, source_type)
        bm25_hits = self._bm25_search(query, bm25_k, source_type)
        merged = self._hybrid_fuse(
            dense_hits=dense_hits,
            bm25_hits=bm25_hits,
            top_k=rerank_top_k if rerank else top_k,
            w_dense=w_dense,
            w_bm25=w_bm25,
        )
        if rerank and merged:
            return self._rerank(query=query, fused_rows=merged, top_k=top_k)
        return merged[:top_k]

    def _dense_search(self, query: str, top_k: int, source_type: Optional[str]) -> List[Dict]:
        if top_k <= 0 or self.index.ntotal <= 0:
            return []

        qv = self._embed([query])[0:1]
        all_results = self.index.search(qv, min(top_k, self.index.ntotal))

        out = []
        for idx, score in zip(all_results[1][0], all_results[0][0]):
            if idx < 0:
                continue
            chunk_id = self.chunk_ids[int(idx)]
            if source_type:
                with self._connect() as conn:
                    row = conn.execute(
                        "SELECT source_type FROM kb_chunks WHERE chunk_id = ?", (chunk_id,)
                    ).fetchone()
                if not row or row["source_type"] != source_type:
                    continue
            out.append({"chunk_id": chunk_id, "dense_score": float(score)})
        return out

    def _bm25_search(self, query: str, top_k: int, source_type: Optional[str]) -> List[Dict]:
        with self._connect() as conn:
            if source_type:
                rows = conn.execute(
                    "SELECT chunk_id, content FROM kb_chunks WHERE source_type = ?",
                    (source_type,),
                ).fetchall()
            else:
                rows = conn.execute("SELECT chunk_id, content FROM kb_chunks").fetchall()

        if not rows:
            return []

        chunk_ids = [r["chunk_id"] for r in rows]
        corpus = [r["content"] for r in rows]
        tokenized_corpus = [self._tokenize(t) for t in corpus]
        tokenized_query = self._tokenize(query)

        bm25 = BM25Okapi(tokenized_corpus)
        scores = bm25.get_scores(tokenized_query)

        idx = np.argsort(scores)[::-1][:top_k]
        out = []
        for i in idx:
            if scores[i] <= 0:
                continue
            out.append({"chunk_id": chunk_ids[int(i)], "bm25_score": float(scores[int(i)])})
        return out

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        text = text.lower().strip()
        cjk = re.findall(r"[\u4e00-\u9fff]", text)
        cjk_bigrams = ["".join(cjk[i : i + 2]) for i in range(len(cjk) - 1)]
        words = re.findall(r"[a-z0-9_]+", text)
        return cjk_bigrams + words

    def _hybrid_fuse(
        self,
        dense_hits: List[Dict],
        bm25_hits: List[Dict],
        top_k: int,
        w_dense: float,
        w_bm25: float,
    ) -> List[Dict]:
        dense_map = {h["chunk_id"]: h["dense_score"] for h in dense_hits}
        bm25_map = {h["chunk_id"]: h["bm25_score"] for h in bm25_hits}

        d_norm = self._normalize_scores(dense_map)
        b_norm = self._normalize_scores(bm25_map)

        all_ids = set(d_norm) | set(b_norm)
        fused: List[Tuple[str, float, float, float]] = []
        for cid in all_ids:
            ds = d_norm.get(cid, 0.0)
            bs = b_norm.get(cid, 0.0)
            score = w_dense * ds + w_bm25 * bs
            fused.append((cid, score, ds, bs))

        fused.sort(key=lambda x: x[1], reverse=True)
        top_ids = [x[0] for x in fused[:top_k]]

        if not top_ids:
            return []

        qmarks = ",".join(["?"] * len(top_ids))
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM kb_chunks WHERE chunk_id IN ({qmarks})", top_ids
            ).fetchall()

        row_map = {r["chunk_id"]: r for r in rows}

        out: List[Dict] = []
        for cid, score, ds, bs in fused[:top_k]:
            r = row_map.get(cid)
            if not r:
                continue
            out.append(
                {
                    "chunk_id": cid,
                    "doc_id": r["doc_id"],
                    "title": r["title"],
                    "source_type": r["source_type"],
                    "tags": json.loads(r["tags_json"]),
                    "chunk_index": r["chunk_index"],
                    "content": r["content"],
                    "source_path": r["source_path"],
                    "score": float(score),
                    "dense_score": float(ds),
                    "bm25_score": float(bs),
                }
            )
        return out

    @staticmethod
    def _normalize_scores(score_map: Dict[str, float]) -> Dict[str, float]:
        if not score_map:
            return {}
        vals = np.array(list(score_map.values()), dtype=float)
        mn = float(vals.min())
        mx = float(vals.max())
        if abs(mx - mn) < 1e-12:
            return {k: 1.0 for k in score_map}
        return {k: (v - mn) / (mx - mn) for k, v in score_map.items()}

    def _rerank(self, query: str, fused_rows: List[Dict], top_k: int) -> List[Dict]:
        if not fused_rows:
            return []
        pairs = [[query, r["content"]] for r in fused_rows]
        scores = self.reranker.predict(pairs)
        scored = []
        for row, rr in zip(fused_rows, scores):
            x = dict(row)
            x["rerank_score"] = float(rr)
            x["score"] = float(rr)
            scored.append(x)
        scored.sort(key=lambda x: x["rerank_score"], reverse=True)
        return scored[:top_k]


def make_chunk_id(source_path: str, chunk_index: int, content: str) -> str:
    sig = hashlib.sha1(f"{source_path}|{chunk_index}|{content}".encode("utf-8")).hexdigest()
    return sig


def build_chunks_from_file(path: Path, chunker: RecursiveChunker) -> List[KBChunk]:
    text, meta = FileReader.read(path)
    pieces = chunker.split(text)

    doc_id = meta.get("doc_id", path.stem)
    title = meta.get("title", path.stem)
    source_type = meta.get("source_type", "manual")
    tags = list(meta.get("tags", []))

    out: List[KBChunk] = []
    for i, p in enumerate(pieces):
        chunk_id = make_chunk_id(str(path.resolve()), i, p)
        out.append(
            KBChunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                title=title,
                source_type=source_type,
                tags=tags,
                content=p,
                chunk_index=i,
                source_path=str(path.resolve()),
            )
        )
    return out


def iter_files(inputs: Iterable[Path]) -> Iterable[Path]:
    exts = {".txt", ".md", ".json", ".pdf"}
    for p in inputs:
        if p.is_file() and p.suffix.lower() in exts:
            yield p
        elif p.is_dir():
            for f in p.rglob("*"):
                if f.is_file() and f.suffix.lower() in exts:
                    yield f
