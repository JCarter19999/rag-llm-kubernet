# scripts/ingest.py
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_local.loaders import load_docs
from rag_local.chunking import make_chunks
from rag_local.local_index import LocalDenseIndex

EMBED_MODEL = "all-MiniLM-L6-v2"
INDEX_PATH = "data/local_index.pkl"

CHUNK_SIZE = 1200
OVERLAP = 150


def main() -> int:
    index = LocalDenseIndex(embedding_model=EMBED_MODEL, persist_path=INDEX_PATH)

    docs = load_docs("docs")
    all_chunks = []

    for d in docs:
        chunks = make_chunks(
            doc_id=d.doc_id,
            source=d.source,
            text=d.text,
            chunk_size=CHUNK_SIZE,
            overlap=OVERLAP,
        )
        all_chunks.extend(chunks)

    index.add_documents(all_chunks)
    out = index.save()
    print(f"Ingested {len(all_chunks)} chunks -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
