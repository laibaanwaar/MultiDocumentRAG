from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

DOCUMENTS_DIR = BASE_DIR / "data" / "documents"
QDRANT_PATH = BASE_DIR / "qdrant_storage"

COLLECTION_NAME = "pakistan_legal_knowledge_base"

EMBEDDING_MODEL_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)
EMBEDDING_DIMENSION = 384

# Small chunks are important because MiniLM truncates long input.
CHUNK_SIZE_WORDS = 180
CHUNK_OVERLAP_WORDS = 30

EMBEDDING_BATCH_SIZE = 32
UPSERT_BATCH_SIZE = 64

FETCH_K = 15
FINAL_K = 5
MAX_CONTEXT_CHARACTERS = 9_000