# Command line entry point: ingest a text file into Typesense, or ask questions about it
import argparse

from src import config
from src.ingest import ingest_file
from src.rag_chain import get_chain


def main():
    parser = argparse.ArgumentParser(description="RAG system using Typesense and Groq")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Store a text file in Typesense")
    ingest_parser.add_argument("file", help="Path of the text file, e.g. test.txt")
    ingest_parser.add_argument(
        "--reindex",
        action="store_true",
        help="Delete the old collection and store the file again"
    )

    ask_parser = subparsers.add_parser("ask", help="Ask a question about the stored file")
    ask_parser.add_argument(
        "question",
        nargs="?",
        help="Your question (leave it out to start an interactive chat)"
    )

    args = parser.parse_args()
    config.check_settings()

    if args.command == "ingest":
        ingest_file(args.file, reindex=args.reindex)
        return

    ask = get_chain()

    if args.question:
        print(ask(args.question))
        return

    # Interactive chat loop
    print("Ask a question (type 'exit' to quit)")
    while True:
        question = input("\nYou: ").strip()

        if question.lower() in ("exit", "quit"):
            break
        if not question:
            continue

        print(f"\nAnswer: {ask(question)}")


if __name__ == "__main__":
    main()
