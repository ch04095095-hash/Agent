#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import List

from kb_store import Chunk, KBStore, chunk_text


def load_doc(doc_path: Path) -> dict:
    data = json.loads(doc_path.read_text(encoding="utf-8"))
    required = ["doc_id", "title", "source_type", "tags", "content"]
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"{doc_path} missing fields: {missing}")
    return data


def build(input_dir: Path, db_path: Path, max_chars: int, overlap: int) -> int:
    store = KBStore(db_path)
    store.init_schema()

    total = 0
    for doc_file in sorted(input_dir.glob("*.json")):
        doc = load_doc(doc_file)
        pieces = chunk_text(doc["content"], max_chars=max_chars, overlap=overlap)

        chunks: List[Chunk] = []
        for i, text in enumerate(pieces):
            chunks.append(
                Chunk(
                    doc_id=doc["doc_id"],
                    title=doc["title"],
                    source_type=doc["source_type"],
                    tags=list(doc["tags"]),
                    content=text,
                    chunk_index=i,
                )
            )

        n = store.upsert_chunks(chunks)
        total += n
        print(f"indexed {doc_file.name}: {n} chunks")

    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Build KB sqlite from JSON docs")
    parser.add_argument("--input-dir", default="kb/docs", help="folder with *.json docs")
    parser.add_argument("--db", default="kb/data/kb.sqlite3", help="sqlite db path")
    parser.add_argument("--max-chars", type=int, default=800)
    parser.add_argument("--overlap", type=int, default=120)
    args = parser.parse_args()

    total = build(
        input_dir=Path(args.input_dir),
        db_path=Path(args.db),
        max_chars=args.max_chars,
        overlap=args.overlap,
    )
    print(f"done. total chunks upserted: {total}")


if __name__ == "__main__":
    main()
