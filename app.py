# Streamlit chat UI for the RAG system: pick a vector store, ask questions and see the sources of every answer
import streamlit as st

from src import config
from src.rag_chain import get_chain

BACKEND_LABELS = {
    "chroma": "PDFs (ChromaDB, local)",
    "typesense": "Text file (Typesense Cloud)",
}

st.set_page_config(page_title="RAG System", page_icon="📚")
st.title("📚 RAG Question Answering")
st.caption("Answers come only from the stored documents. If the answer is not there, the assistant says it doesn't know.")


@st.cache_resource(show_spinner="Loading models and connecting to the vector store...")
def load_chain(backend: str, top_k: int):
    """Build the chain once per backend and top_k, so every question reuses the same models"""

    return get_chain(backend, top_k)


with st.sidebar:
    st.header("Settings")

    backend = st.radio(
        "Documents",
        options=config.BACKENDS,
        format_func=lambda name: BACKEND_LABELS[name],
    )
    top_k = st.slider("Chunks used as context", min_value=1, max_value=8, value=config.TOP_K)

    if st.button("Clear chat"):
        st.session_state.messages = []

    st.markdown(
        "Store documents first from the terminal:\n\n"
        "```\npython main.py ingest --backend chroma\npython main.py ingest\n```"
    )

if "messages" not in st.session_state:
    st.session_state.messages = []


def show_sources(sources: list) -> None:
    """Show the chunks an answer was based on inside an expander"""

    with st.expander(f"Sources ({len(sources)})"):
        for source in sources:
            page = f", page {source['page']}" if source["page"] else ""
            st.markdown(f"**{source['source']}{page}**")
            st.text(source["content"][:400])


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            show_sources(message["sources"])

question = st.chat_input("Ask a question about your documents")

if question:
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            ask = load_chain(backend, top_k)
            with st.spinner("Searching and thinking..."):
                result = ask(question)
        except Exception as error:
            st.error(f"Something went wrong: {error}")
            st.stop()

        st.markdown(result["answer"])
        if result["sources"]:
            show_sources(result["sources"])

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"],
    })
