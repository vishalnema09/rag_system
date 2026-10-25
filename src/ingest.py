# Ingestion - load a text file, split it into chunks, embed them and store them in Typesense
from typing import List, Any

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Typesense
from langchain_text_splitters import CharacterTextSplitter

from src import config
from src.retriever import get_typesense_client, get_embeddings


def load_and_split(file_path: str) -> List[Any]:
    """
    Load a text file and split it into overlapping chunks.

    Args:
        file_path: Path of the text file

    Returns:
        List of chunked documents
    """

    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()

    text_splitter = CharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP
    )

    docs = text_splitter.split_documents(documents)

    print(f"Split {len(documents)} documents into {len(docs)} chunks")

    return docs


def collection_exists() -> bool:
    """Check whether the Typesense collection is already created"""

    client = get_typesense_client()
    names = [c["name"] for c in client.collections.retrieve()]

    return config.COLLECTION_NAME in names


def ingest_file(file_path: str, reindex: bool = False) -> None:
    """
    Store a text file in Typesense without creating duplicate chunks.

    Args:
        file_path: Path of the text file
        reindex: If True, delete the old collection and store the chunks again.
            If False and the collection already exists, ingestion is skipped.
    """

    if collection_exists():
        if not reindex:
            print(
                f"Collection '{config.COLLECTION_NAME}' already exists, "
                f"skipping ingestion (use --reindex to store it again)"
            )
            return

        get_typesense_client().collections[config.COLLECTION_NAME].delete()
        print(f"Deleted old collection: {config.COLLECTION_NAME}")

    docs = load_and_split(file_path)

    Typesense.from_documents(
        docs,
        get_embeddings(),
        typesense_client_params={
            "host": config.TYPESENSE_HOST,
            "port": config.TYPESENSE_PORT,
            "protocol": config.TYPESENSE_PROTOCOL,
            "typesense_api_key": config.TYPESENSE_API_KEY,
            "typesense_collection_name": config.COLLECTION_NAME,
        },
    )

    print(f"Stored {len(docs)} chunks in collection: {config.COLLECTION_NAME}")
