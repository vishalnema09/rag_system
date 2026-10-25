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

# Typesense Cloud requires https on port 443
TYPESENSE_PORT = "443"
TYPESENSE_PROTOCOL = "https"
COLLECTION_NAME = "lang-chain"

# Text splitting
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

# Groq LLM
LLM_MODEL = "openai/gpt-oss-120b"
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 1024

# Number of chunks passed to the LLM as context
TOP_K = 3


def check_settings():
    """Raise a clear error if a required value is missing from the .env file"""

    missing = [
        name for name, value in {
            "GROQ_API_KEY": GROQ_API_KEY,
            "TYPESENSE_HOST": TYPESENSE_HOST,
            "TYPESENSE_API_KEY": TYPESENSE_API_KEY,
        }.items() if not value
    ]

    if missing:
        raise ValueError(
            f"Missing in the .env file: {', '.join(missing)}"
        )
