#!/usr/bin/env python3
import argparse
from pathlib import Path

from vector_kb import RecursiveChunker, VectorKB, build_chunks_from_file, iter_files


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest files into vector KB")
    parser.add_argument("inputs", nargs="+", help="files or directories to ingest")
    parser.add_argument("--db", default="kb/data/kb.sqlite3", help="metadata sqlite path")
    parser.add_argument("--collection", default="coal_truck_kb")
    parser.add_argument("--embed-model", default="BAAI/bge-small-zh-v1.5")
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--chunk-overlap", type=int, default=120)
    args = parser.parse_args()

    kb = VectorKB(
        db_path=Path(args.db),
        collection=args.collection,
        embed_model=args.embed_model,
    )

    chunker = RecursiveChunker(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)

    all_chunks = []
    files = list(iter_files(Path(x) for x in args.inputs))
    if not files:
        print("no supported files found")
        return

    for f in files:
        chunks = build_chunks_from_file(f, chunker)
        all_chunks.extend(chunks)
        print(f"prepared {f}: {len(chunks)} chunks")

    n = kb.ingest_chunks(all_chunks)
    print(f"ingested chunks: {n}")


if __name__ == "__main__":
    main()
