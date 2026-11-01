# RAG chain - retrieve chunks (Typesense or ChromaDB), build a prompt with that context, and ask the Groq LLM
from typing import Dict, Any

from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

from src import config

PROMPT_TEMPLATE = """Use the following context to answer the question.
If the answer is not in the context, say you don't know.

Context:
{context}

Question: {query}

Answer:"""


def get_llm() -> ChatGroq:
    """Set up the Groq LLM (API key is read from the .env file)"""

    return ChatGroq(
        groq_api_key=config.GROQ_API_KEY,
        model_name=config.LLM_MODEL,
        temperature=config.LLM_TEMPERATURE,
        max_tokens=config.LLM_MAX_TOKENS
    )


def get_backend_retriever(backend: str = config.DEFAULT_BACKEND, top_k: int = config.TOP_K):
    """Build the retriever of the chosen vector store ('typesense' or 'chroma')"""

    if backend == "typesense":
        from src.retriever import get_retriever
        return get_retriever(top_k)

    if backend == "chroma":
        from src.pdf_store import ChromaRetriever
        return ChromaRetriever(top_k=top_k)

    raise ValueError(f"Unknown backend '{backend}'. Choose one of: {', '.join(config.BACKENDS)}")


def rag_answer(
    query: str,
    retriever,
    llm: ChatGroq,
    top_k: int = config.TOP_K
) -> Dict[str, Any]:
    """
    RAG: retrieve relevant chunks, put them in the prompt,
    and let the Groq LLM answer using only that context.

    Returns:
        Dictionary with the 'answer' string and the list of 'sources'
        (file name, page and text of every chunk used as context)
    """

    results = retriever.invoke(query)[:top_k]

    if not results:
        return {"answer": "No relevant context found.", "sources": []}

    context = "\n\n".join(doc.page_content for doc in results)

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "query"]
    )

    response = llm.invoke([prompt.format(context=context, query=query)])

    sources = [
        {
            "source": doc.metadata.get("source", "unknown"),
            "page": doc.metadata.get("page"),
            "content": doc.page_content
        }
        for doc in results
    ]

    return {"answer": response.content, "sources": sources}


def rag_typesense(
    query: str,
    retriever,
    llm: ChatGroq,
    top_k: int = config.TOP_K
):
    """
    Simple RAG: same as rag_answer, but returns only the answer string.

    Returns:
        The answer string
    """

    return rag_answer(query, retriever, llm, top_k)["answer"]


def get_chain(backend: str = config.DEFAULT_BACKEND, top_k: int = config.TOP_K):
    """
    Build the retriever and the LLM once.

    Returns:
        A function that takes a question and returns the answer and its sources
    """

    config.check_settings(backend)

    retriever = get_backend_retriever(backend, top_k)
    llm = get_llm()

    def ask(query: str) -> Dict[str, Any]:
        return rag_answer(query, retriever, llm, top_k)

    return ask
