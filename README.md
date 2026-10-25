# RAG System

A Retrieval-Augmented Generation (RAG) system. It splits a text file into chunks, embeds them, stores them in **Typesense Cloud**, and answers questions with a **Groq** LLM using only the retrieved chunks as context.

## How it works

```
text file -> split into chunks -> embed (all-mpnet-base-v2) -> Typesense
question -> embed -> top matching chunks from Typesense -> prompt -> Groq LLM -> answer
```

The project has two parts:

- **Notebooks** - step-by-step experiments.
  - `notebook/document.ipynb`: PDF RAG with ChromaDB (sources, confidence score, streaming, citations, chat history, summarization).
  - `typesense.ipynb`: Typesense basics (filter and facet search) and the LangChain + Typesense + Groq RAG.
- **Command line app** - `main.py` plus the modules in `src/`.

## Setup

Requires Python 3.13.

```bash
# with uv
uv sync

# or with pip
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_api_key
TYPESENSE_HOST=xxx.a1.typesense.net
TYPESENSE_API_KEY=your_typesense_api_key
```

`.env` is listed in `.gitignore`, so your keys are not committed.

## Usage

Store a text file in Typesense (the first run downloads the embedding model):

```bash
python main.py ingest test.txt
```

If the collection already exists, ingestion is skipped so chunks are never duplicated. To delete the old collection and store the file again:

```bash
python main.py ingest test.txt --reindex
```

Ask a single question:

```bash
python main.py ask "What is artificial intelligence?"
```

Or start an interactive chat (type `exit` to quit):

```bash
python main.py ask
```

If the answer is not in the stored file, the LLM says it does not know.

## Project structure

```
main.py              command line entry point (ingest, ask)
src/
  config.py          .env loading and constants (collection name, chunk size, model)
  retriever.py       Typesense client, embeddings and retriever
  ingest.py          load -> split -> embed -> store, with duplicate protection
  rag_chain.py       prompt + Groq LLM + the RAG function
notebook/
  document.ipynb     PDF RAG with ChromaDB
typesense.ipynb      Typesense RAG
data/                sample PDFs and text files
test.txt             sample text used for the Typesense RAG
books.jsonl          sample books data for the Typesense search demo
```
