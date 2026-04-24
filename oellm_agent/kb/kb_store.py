#!/usr/bin/env python3
import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class Chunk:
    doc_id: str
    title: str
    source_type: str
    tags: List[str]
    content: str
    chunk_index: int


class KBStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def init_schema(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS kb_chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    tags_json TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    UNIQUE(doc_id, chunk_index)
                );

                CREATE VIRTUAL TABLE IF NOT EXISTS kb_chunks_fts USING fts5(
                    doc_id,
                    title,
                    source_type,
                    tags,
                    content,
                    content='kb_chunks',
                    content_rowid='id'
                );

                CREATE TRIGGER IF NOT EXISTS kb_chunks_ai AFTER INSERT ON kb_chunks BEGIN
                    INSERT INTO kb_chunks_fts(rowid, doc_id, title, source_type, tags, content)
                    VALUES (new.id, new.doc_id, new.title, new.source_type, new.tags_json, new.content);
                END;

                CREATE TRIGGER IF NOT EXISTS kb_chunks_ad AFTER DELETE ON kb_chunks BEGIN
                    INSERT INTO kb_chunks_fts(kb_chunks_fts, rowid, doc_id, title, source_type, tags, content)
                    VALUES ('delete', old.id, old.doc_id, old.title, old.source_type, old.tags_json, old.content);
                END;

                CREATE TRIGGER IF NOT EXISTS kb_chunks_au AFTER UPDATE ON kb_chunks BEGIN
                    INSERT INTO kb_chunks_fts(kb_chunks_fts, rowid, doc_id, title, source_type, tags, content)
                    VALUES ('delete', old.id, old.doc_id, old.title, old.source_type, old.tags_json, old.content);
                    INSERT INTO kb_chunks_fts(rowid, doc_id, title, source_type, tags, content)
                    VALUES (new.id, new.doc_id, new.title, new.source_type, new.tags_json, new.content);
                END;
                """
            )

    def upsert_chunks(self, chunks: List[Chunk]) -> int:
        if not chunks:
            return 0

        with self.connect() as conn:
            for ch in chunks:
                conn.execute(
                    """
                    INSERT INTO kb_chunks(doc_id, title, source_type, tags_json, chunk_index, content)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(doc_id, chunk_index) DO UPDATE SET
                        title=excluded.title,
                        source_type=excluded.source_type,
                        tags_json=excluded.tags_json,
                        content=excluded.content
                    """,
                    (
                        ch.doc_id,
                        ch.title,
                        ch.source_type,
                        json.dumps(ch.tags, ensure_ascii=False),
                        ch.chunk_index,
                        ch.content,
                    ),
                )
            return len(chunks)

    def search(self, query: str, top_k: int = 5, source_type: Optional[str] = None) -> List[Dict]:
        q = _sanitize_fts_query(query)
        if not q:
            return []

        where = "WHERE kb_chunks_fts MATCH ?"
        params: List = [q]
        if source_type:
            where += " AND c.source_type = ?"
            params.append(source_type)

        params.append(top_k)

        sql = f"""
            SELECT
                c.id,
                c.doc_id,
                c.title,
                c.source_type,
                c.tags_json,
                c.chunk_index,
                c.content,
                bm25(kb_chunks_fts, 2.0, 1.5, 1.0, 0.8, 1.2) AS score
            FROM kb_chunks_fts
            JOIN kb_chunks c ON c.id = kb_chunks_fts.rowid
            {where}
            ORDER BY score
            LIMIT ?
        """

        with self.connect() as conn:
            rows = conn.execute(sql, params).fetchall()

            # Fallback: if FTS returns no hits (common on some Chinese tokenization setups),
            # degrade to LIKE retrieval for reliability.
            if not rows:
                like_terms = [t for t in re.split(r"\s+", query.strip()) if t]
                if like_terms:
                    conds = []
                    like_params: List = []
                    for t in like_terms:
                        conds.append("(title LIKE ? OR content LIKE ? OR tags_json LIKE ?)")
                        pattern = f"%{t}%"
                        like_params.extend([pattern, pattern, pattern])

                    where_sql = " OR ".join(conds)
                    source_sql = ""
                    if source_type:
                        source_sql = " AND source_type = ?"
                        like_params.append(source_type)

                    like_params.append(top_k)
                    rows = conn.execute(
                        f"""
                        SELECT
                            id,
                            doc_id,
                            title,
                            source_type,
                            tags_json,
                            chunk_index,
                            content,
                            1.0 AS score
                        FROM kb_chunks
                        WHERE ({where_sql}) {source_sql}
                        ORDER BY id DESC
                        LIMIT ?
                        """,
                        like_params,
                    ).fetchall()

        out: List[Dict] = []
        for r in rows:
            out.append(
                {
                    "id": r["id"],
                    "doc_id": r["doc_id"],
                    "title": r["title"],
                    "source_type": r["source_type"],
                    "tags": json.loads(r["tags_json"]),
                    "chunk_index": r["chunk_index"],
                    "content": r["content"],
                    "score": r["score"],
                }
            )
        return out


def chunk_text(text: str, max_chars: int = 800, overlap: int = 120) -> List[str]:
    cleaned = re.sub(r"\n{3,}", "\n\n", text.strip())
    if not cleaned:
        return []

    paragraphs = cleaned.split("\n\n")
    chunks: List[str] = []
    cur = ""
    for p in paragraphs:
        if len(p) > max_chars:
            # hard split very long paragraph
            for i in range(0, len(p), max_chars - overlap):
                part = p[i : i + max_chars]
                if part.strip():
                    chunks.append(part.strip())
            continue

        if not cur:
            cur = p
            continue

        if len(cur) + 2 + len(p) <= max_chars:
            cur += "\n\n" + p
        else:
            chunks.append(cur.strip())
            tail = cur[-overlap:] if overlap > 0 else ""
            cur = (tail + "\n\n" + p).strip() if tail else p

    if cur.strip():
        chunks.append(cur.strip())

    return chunks


def _sanitize_fts_query(query: str) -> str:
    query = query.strip()
    query = query.replace('"', ' ')
    query = re.sub(r"\s+", " ", query)
    if not query:
        return ""

    # FTS5 MATCH defaults to AND; for natural Chinese queries (e.g. "怎么处置") this is too strict.
    # Build a softer OR query and keep both word-level and CJK bigram terms.
    tokens: List[str] = []
    for word in query.split(" "):
        w = word.strip()
        if not w:
            continue
        tokens.append(w)
        if re.search(r"[\u4e00-\u9fff]", w) and len(w) >= 2:
            for i in range(len(w) - 1):
                tokens.append(w[i : i + 2])

    # deduplicate while preserving order
    seen = set()
    uniq = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            uniq.append(t)

    if not uniq:
        return ""
    return " OR ".join(uniq)
