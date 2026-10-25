# RAG chain - retrieve chunks from Typesense, build a prompt with that context, and ask the Groq LLM
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

from src import config
from src.retriever import get_retriever


def get_llm() -> ChatGroq:
    """Set up the Groq LLM (API key is read from the .env file)"""

    config.check_settings()

    return ChatGroq(
        groq_api_key=config.GROQ_API_KEY,
        model_name=config.LLM_MODEL,
        temperature=config.LLM_TEMPERATURE,
        max_tokens=config.LLM_MAX_TOKENS
    )


def rag_typesense(
    query: str,
    retriever,
    llm: ChatGroq,
    top_k: int = config.TOP_K
):
    """
    Simple RAG on Typesense: retrieve relevant chunks, put them in the prompt,
    and let the Groq LLM answer using only that context.

    Returns:
        The answer string
    """

    results = retriever.invoke(query)[:top_k]
    context = "\n\n".join(doc.page_content for doc in results)

    if not context:
        return "No relevant context found."

    prompt = PromptTemplate(
        template="""Use the following context to answer the question.
If the answer is not in the context, say you don't know.

Context:
{context}

Question: {query}

Answer:""",
        input_variables=["context", "query"]
    )

    response = llm.invoke([prompt.format(context=context, query=query)])
    return response.content


def get_chain():
    """
    Build the retriever and the LLM once.

    Returns:
        A function that takes a question and returns the answer
    """

    retriever = get_retriever()
    llm = get_llm()

    def ask(query: str) -> str:
        return rag_typesense(query, retriever, llm)

    return ask
