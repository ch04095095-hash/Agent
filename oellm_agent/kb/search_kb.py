#!/usr/bin/env python3
import argparse
from pathlib import Path

from kb_store import KBStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Search KB")
    parser.add_argument("query", help="search query text")
    parser.add_argument("--db", default="kb/data/kb.sqlite3", help="sqlite db path")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--source-type", default="", help="optional filter: safety_manual|case|maintenance")
    args = parser.parse_args()

    store = KBStore(Path(args.db))
    rows = store.search(
        query=args.query,
        top_k=args.top_k,
        source_type=args.source_type or None,
    )

    if not rows:
        print("no hits")
        return

    for i, r in enumerate(rows, start=1):
        print(f"\n[{i}] score={r['score']:.4f}")
        print(f"doc_id: {r['doc_id']}")
        print(f"title: {r['title']}")
        print(f"source_type: {r['source_type']}")
        print(f"tags: {', '.join(r['tags'])}")
        print(f"chunk_index: {r['chunk_index']}")
        print("content:")
        print(r["content"])


if __name__ == "__main__":
    main()
