# Central settings - loads the .env file and holds the constants used by every module
import os
from pathlib import Path

from dotenv import load_dotenv

# Project root is one level above src/, so the .env file is found from any working directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TYPESENSE_HOST = os.getenv("TYPESENSE_HOST")
TYPESENSE_API_KEY = os.getenv("TYPESENSE_API_KEY")

# Vector stores the app can use
BACKENDS = ("typesense", "chroma")
DEFAULT_BACKEND = "typesense"

# Default data to ingest for each backend
DEFAULT_TEXT_FILE = PROJECT_ROOT / "data" / "text_files" / "test.txt"
DEFAULT_PDF_DIR = PROJECT_ROOT / "data" / "pdf"

# Typesense Cloud requires https on port 443
TYPESENSE_PORT = "443"
TYPESENSE_PROTOCOL = "https"
COLLECTION_NAME = "lang-chain"

# ChromaDB (local, persisted on disk)
CHROMA_DIR = PROJECT_ROOT / "data" / "vector_store"
CHROMA_COLLECTION_NAME = "pdf_documents_cli"

# Embedding models
TYPESENSE_EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
CHROMA_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Text splitting (text files use CHUNK_SIZE/CHUNK_OVERLAP, PDFs use PDF_CHUNK_SIZE/PDF_CHUNK_OVERLAP)
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
PDF_CHUNK_SIZE = 1000
PDF_CHUNK_OVERLAP = 200

# Groq LLM
LLM_MODEL = "openai/gpt-oss-120b"
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 1024

# Number of chunks passed to the LLM as context
TOP_K = 3


def check_settings(backend: str = DEFAULT_BACKEND):
    """Raise a clear error if a required value is missing from the .env file"""

    required = {"GROQ_API_KEY": GROQ_API_KEY}

    if backend == "typesense":
        required["TYPESENSE_HOST"] = TYPESENSE_HOST
        required["TYPESENSE_API_KEY"] = TYPESENSE_API_KEY

    missing = [name for name, value in required.items() if not value]

    if missing:
        raise ValueError(
            f"Missing in the .env file: {', '.join(missing)}"
        )
