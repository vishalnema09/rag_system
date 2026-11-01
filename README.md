# Dual-Backend RAG Assistant with Real-Time Source Attribution

A production-grade Retrieval-Augmented Generation (RAG) system with a **dual-vector backend**:
- **ChromaDB (Local):** Ingests multi-page PDFs, chunks them with MD5 deduplication, and retrieves answers with exact file and page citations.
- **Typesense Cloud:** Ingests raw text files for fast cloud-native vector search.
- **Groq LLM:** Delivers sub-second, grounded responses using strictly retrieved context (zero hallucination).

## How It Works

```
1. PDF Documents -> PyMuPDF -> Chunks -> all-MiniLM-L6-v2 -> ChromaDB (Local)
2. Text Files    -> Splitter -> Chunks -> all-mpnet-base-v2 -> Typesense (Cloud)
3. Question      -> Similarity Search (Top-K Chunks) -> Groq LLM -> Answer + Page Citations
```

The project has three interfaces:
- **Web App (`app.py`):** Interactive Streamlit chat UI with backend switcher, context slider, and source dropdowns.
- **Command Line App (`main.py`):** Fast terminal interface for automated ingestion and single/interactive Q&A.
- **Notebooks (`notebook/`):** Step-by-step RAG research and experiments.

## Setup

Requires Python 3.13.

```bash
# with uv
uv sync

# or with pip
pip install -r requirements.txt
```

Create a `.env` file in the project root:
```env
GROQ_API_KEY=your_groq_api_key
TYPESENSE_HOST=xxx.a1.typesense.net
TYPESENSE_API_KEY=your_typesense_api_key
```

`.env` is listed in `.gitignore`, so your keys are not committed.

## Usage

### 1. Web Application (Streamlit)
Start the browser chat interface:
```bash
streamlit run app.py
```
Open `http://localhost:8501` to switch backends (PDFs vs. Text) and view answers with page-level citations.

### 2. Command Line Interface (CLI)

**Ingest documents into vector store:**
```bash
# Ingest PDFs into local ChromaDB
python main.py ingest --backend chroma

# Ingest a text file into Typesense Cloud
python main.py ingest data/text_files/test.txt --backend typesense

# Reindex (delete old collection and re-embed from scratch)
python main.py ingest --backend chroma --reindex
```

**Ask questions:**
```bash
# Single question
python main.py ask "What is the company mission?" --backend chroma

# Interactive chat loop (type 'exit' to quit)
python main.py ask --backend chroma
```

If the answer is not in the stored documents, the LLM explicitly states it does not know.

## Project Structure

```
app.py                    Streamlit chat web application with source expander
main.py                   Command-line entry point (ingest, ask)
src/
  config.py               Environment settings, chunk sizes, and model constants
  pdf_store.py            PyMuPDF loading, chunking, MD5 deduplication & ChromaDB retriever
  retriever.py            Typesense client, embeddings & retriever
  ingest.py               Ingestion router for ChromaDB and Typesense
  rag_chain.py            Prompt template + Groq LLM + RAG answering logic
notebook/
  document.ipynb          PDF RAG experiments with ChromaDB
typesense.ipynb           Typesense exploration & search demo
data/
  pdf/                    Sample corporate PDFs
  text_files/             Sample text files
  vector_store/           Persisted local ChromaDB database
  books.jsonl             Sample books dataset
```
