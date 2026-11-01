# PDF RAG on ChromaDB - load PDFs, split them, embed the chunks and retrieve them from a local vector store
import hashlib
from pathlib import Path
from typing import List, Any

import chromadb
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src import config


class Embedder:
    """Turns text into vectors with a SentenceTransformer model (loaded only when first used)"""

    def __init__(self, model_name: str = config.CHROMA_EMBEDDING_MODEL):
        self.model_name = model_name
        self._model = None

    def embed(self, texts: List[str]) -> List[List[float]]:
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)

        return self._model.encode(texts, show_progress_bar=False).tolist()


def load_pdfs(directory: str) -> List[Any]:
    """
    Load every PDF in a folder into LangChain Document objects (one per page).

    Args:
        directory: Folder that contains the PDF files

    Returns:
        List of page documents
    """

    dir_loader = DirectoryLoader(
        str(directory),
        glob="**/*.pdf",
        loader_cls=PyMuPDFLoader,
        show_progress=False
    )

    return dir_loader.load()


def split_pdf_documents(documents: List[Any]) -> List[Any]:
    """
    Split page documents into overlapping chunks and keep only the useful metadata
    (source file name and page number) so it can be shown as a citation.

    Returns:
        List of chunked documents
    """

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.PDF_CHUNK_SIZE,
        chunk_overlap=config.PDF_CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )

    split_docs = text_splitter.split_documents(documents)

    for doc in split_docs:
        doc.metadata = {
            "source": Path(doc.metadata.get("source", "unknown")).name,
            "page": int(doc.metadata.get("page", 0)) + 1
        }

    print(f"Split {len(documents)} pages into {len(split_docs)} chunks")

    return split_docs


def get_collection(persist_directory: str = None):
    """Get or create the ChromaDB collection stored on disk"""

    path = Path(persist_directory or config.CHROMA_DIR)
    path.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(path))

    return client.get_or_create_collection(name=config.CHROMA_COLLECTION_NAME)


def ingest_pdfs(
    directory: str = None,
    reindex: bool = False,
    embedder: Embedder = None,
    persist_directory: str = None
) -> None:
    """
    Store the PDFs of a folder in ChromaDB without creating duplicate chunks.

    Args:
        directory: Folder that contains the PDF files
        reindex: If True, delete the old collection and store the chunks again.
            If False and the collection already has chunks, ingestion is skipped.
        embedder: Embedding model (a default SentenceTransformer is used if not given)
        persist_directory: Where ChromaDB keeps its files
    """

    collection = get_collection(persist_directory)

    if collection.count() > 0:
        if not reindex:
            print(
                f"Collection '{config.CHROMA_COLLECTION_NAME}' already has "
                f"{collection.count()} chunks, skipping ingestion "
                f"(use --reindex to store it again)"
            )
            return

        client = chromadb.PersistentClient(path=str(persist_directory or config.CHROMA_DIR))
        client.delete_collection(config.CHROMA_COLLECTION_NAME)
        collection = get_collection(persist_directory)
        print(f"Deleted old collection: {config.CHROMA_COLLECTION_NAME}")

    documents = load_pdfs(directory or config.DEFAULT_PDF_DIR)

    if not documents:
        raise ValueError(f"No PDF files found in: {directory or config.DEFAULT_PDF_DIR}")

    chunks = split_pdf_documents(documents)
    texts = [doc.page_content for doc in chunks]

    # The id comes from the content, so the same chunk is never stored twice
    ids = [
        hashlib.md5(
            f"{doc.metadata['source']}|{doc.metadata['page']}|{doc.page_content}".encode("utf-8")
        ).hexdigest()
        for doc in chunks
    ]

    embeddings = (embedder or Embedder()).embed(texts)

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=[doc.metadata for doc in chunks]
    )

    print(f"Stored {collection.count()} chunks in collection: {config.CHROMA_COLLECTION_NAME}")


class ChromaRetriever:
    """Fetches the most similar chunks for a question from the ChromaDB collection"""

    def __init__(
        self,
        top_k: int = config.TOP_K,
        embedder: Embedder = None,
        persist_directory: str = None
    ):
        self.top_k = top_k
        self.embedder = embedder or Embedder()
        self.collection = get_collection(persist_directory)

    def invoke(self, query: str) -> List[Document]:
        """
        Embed the query and return the top_k closest chunks as LangChain Documents
        (same interface as the Typesense retriever, so the RAG chain works with both).
        """

        if self.collection.count() == 0:
            return []

        query_embedding = self.embedder.embed([query])[0]

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=self.top_k
        )

        return [
            Document(page_content=text, metadata=metadata)
            for text, metadata in zip(results["documents"][0], results["metadatas"][0])
        ]
