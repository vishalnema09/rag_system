# Typesense connection - builds the client, the embedding model and the retriever
import typesense
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Typesense

from src import config


def get_typesense_client() -> typesense.Client:
    """Connect to Typesense Cloud using the host and API key from the .env file"""

    return typesense.Client({
        "nodes": [{
            "host": config.TYPESENSE_HOST,
            "port": config.TYPESENSE_PORT,
            "protocol": config.TYPESENSE_PROTOCOL
        }],
        "api_key": config.TYPESENSE_API_KEY,
        "connection_timeout_seconds": 2
    })


def get_embeddings() -> HuggingFaceEmbeddings:
    """Load the embedding model (downloads sentence-transformers/all-mpnet-base-v2 on the first run)"""

    return HuggingFaceEmbeddings()


def get_vectorstore(embeddings: HuggingFaceEmbeddings = None) -> Typesense:
    """
    Connect to the existing Typesense collection (nothing is added or deleted).

    Returns:
        The Typesense vector store
    """

    return Typesense(
        typesense_client=get_typesense_client(),
        embedding=embeddings or get_embeddings(),
        typesense_collection_name=config.COLLECTION_NAME,
        text_key="text"
    )


def get_retriever(top_k: int = config.TOP_K):
    """Wrap the Typesense vector store as a retriever returning the top_k chunks"""

    return get_vectorstore().as_retriever(search_kwargs={"k": top_k})
