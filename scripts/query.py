# scripts/query.py
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
from rag_local.local_index import LocalDenseIndex
from rag_local.rag import rag_answer

EMBED_MODEL = "all-MiniLM-L6-v2"
INDEX_PATH = "data/local_index.pkl"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("-q", "--query", required=True, help="User query")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    q = args.query.strip()
    if not q:
        print("Empty query.")
        return 1

    index = LocalDenseIndex(embedding_model=EMBED_MODEL, persist_path=INDEX_PATH)
    ok = index.load()
    if not ok:
        print(f"Index not found: {INDEX_PATH}")
        print("Run: python scripts/ingest.py")
        return 1

    hits = index.search(q, k=args.top_k)
    print(f"Loaded index: {INDEX_PATH}")
    print(f"Retrieved hits: {len(hits)} (top_k={args.top_k})")

    if not hits:
        print("No hits retrieved.")
        return 0

    print("\nTop hits preview:")
    for i, h in enumerate(hits, 1):
        preview = h.text[:220].replace("\n", " ")
        print(f"{i}) score={h.score:.3f} id={h.chunk_id} meta={h.meta}")
        print(f"   {preview}")
    print()

    answer = rag_answer(q, index, args.top_k)
    answer = (answer or "").strip()
    if not answer:
        print("Got empty answer from Ollama.")
        return 1

    print("\n=== ANSWER ===\n")
    print(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
