# Command line entry point: ingest documents into a vector store, or ask questions about them
import argparse

from src import config
from src.ingest import ingest
from src.rag_chain import get_chain


def print_answer(result: dict) -> None:
    """Print the answer and the sources (file and page) it was based on"""

    print(result["answer"])

    if result["sources"]:
        print("\nSources:")
        for source in result["sources"]:
            page = f", page {source['page']}" if source["page"] else ""
            print(f"  - {source['source']}{page}")


def main():
    parser = argparse.ArgumentParser(description="RAG system using Typesense / ChromaDB and Groq")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Store documents in a vector store")
    ingest_parser.add_argument(
        "path",
        nargs="?",
        help="Text file (typesense) or PDF folder (chroma). Defaults: "
             "data/text_files/test.txt and data/pdf"
    )
    ingest_parser.add_argument(
        "--backend",
        choices=config.BACKENDS,
        default=config.DEFAULT_BACKEND,
        help="typesense = text file in Typesense Cloud, chroma = PDF folder in local ChromaDB"
    )
    ingest_parser.add_argument(
        "--reindex",
        action="store_true",
        help="Delete the old data and store it again"
    )

    ask_parser = subparsers.add_parser("ask", help="Ask a question about the stored documents")
    ask_parser.add_argument(
        "question",
        nargs="?",
        help="Your question (leave it out to start an interactive chat)"
    )
    ask_parser.add_argument(
        "--backend",
        choices=config.BACKENDS,
        default=config.DEFAULT_BACKEND,
        help="Vector store to search"
    )

    args = parser.parse_args()

    if args.command == "ingest":
        ingest(args.backend, args.path, reindex=args.reindex)
        return

    ask = get_chain(args.backend)

    if args.question:
        print_answer(ask(args.question))
        return

    # Interactive chat loop
    print("Ask a question (type 'exit' to quit)")
    while True:
        question = input("\nYou: ").strip()

        if question.lower() in ("exit", "quit"):
            break
        if not question:
            continue

        print()
        print_answer(ask(question))


if __name__ == "__main__":
    main()
