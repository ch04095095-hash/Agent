#!/usr/bin/env python3
import argparse
from pathlib import Path

from vector_kb import VectorKB


def main() -> None:
    parser = argparse.ArgumentParser(description="Search vector KB with hybrid ranking")
    parser.add_argument("query", help="query text")
    parser.add_argument("--db", default="kb/data/kb.sqlite3", help="metadata sqlite path")
    parser.add_argument("--collection", default="coal_truck_kb")
    parser.add_argument("--embed-model", default="BAAI/bge-small-zh-v1.5")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--dense-k", type=int, default=30)
    parser.add_argument("--bm25-k", type=int, default=30)
    parser.add_argument("--w-dense", type=float, default=0.7)
    parser.add_argument("--w-bm25", type=float, default=0.3)
    parser.add_argument("--source-type", default="", help="optional filter")
    args = parser.parse_args()

    kb = VectorKB(
        db_path=Path(args.db),
        collection=args.collection,
        embed_model=args.embed_model,
    )

    rows = kb.search(
        query=args.query,
        top_k=args.top_k,
        source_type=args.source_type or None,
        dense_k=args.dense_k,
        bm25_k=args.bm25_k,
        w_dense=args.w_dense,
        w_bm25=args.w_bm25,
    )

    if not rows:
        print("no hits")
        return

    for i, r in enumerate(rows, start=1):
        print(f"\n[{i}] score={r['score']:.4f} dense={r['dense_score']:.4f} bm25={r['bm25_score']:.4f}")
        print(f"doc_id: {r['doc_id']}")
        print(f"title: {r['title']}")
        print(f"source_type: {r['source_type']}")
        print(f"tags: {', '.join(r['tags'])}")
        print(f"chunk_index: {r['chunk_index']}")
        print(f"source_path: {r['source_path']}")
        print("content:")
        print(r["content"])


if __name__ == "__main__":
    main()
