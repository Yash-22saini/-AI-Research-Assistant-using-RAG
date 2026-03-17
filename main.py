"""
main.py — CLI entry point for the AI Research Assistant (RAG).

Usage:
    python main.py                        # Uses default sample PDF
    python main.py --pdf path/to/doc.pdf  # Custom PDF path
"""

import argparse
from app.ingest import load_single_pdf
from app.embed import create_vectorstore
from app.retriever import get_relevant_docs
from app.generator import generate_answer


def run(pdf_path: str):
    print("\n📚 AI Research Assistant (RAG)")
    print("=" * 40)

    print(f"📄 Loading and chunking: {pdf_path} ...")
    try:
        docs = load_single_pdf(pdf_path)
    except Exception as e:
        print(f"❌ Error: {e}")
        return

    print(f"✅ {len(docs)} chunks created.")

    print("🔢 Generating embeddings and building FAISS index...")
    create_vectorstore(docs)
    print("✅ Vector store saved to vectorstore/faiss_index.pkl")

    chat_history = []
    print("\n💬 Ready! Type your question or 'exit' to quit.\n")

    while True:
        query = input("❓ Question: ").strip()

        if not query:
            continue
        if query.lower() in ("exit", "quit"):
            print("👋 Goodbye!")
            break

        print("🔍 Retrieving relevant chunks...")
        retrieved_docs = get_relevant_docs(query, k=5)

        print("🤖 Generating answer...\n")
        answer = generate_answer(query, retrieved_docs, chat_history)

        print(f"Answer:\n{answer}\n")
        print("-" * 40)

        chat_history.append(f"Q: {query}")
        chat_history.append(f"A: {answer}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Research Assistant CLI")
    parser.add_argument(
        "--pdf",
        type=str,
        default="data/sample.pdf",
        help="Path to the PDF file to load (default: data/sample.pdf)"
    )
    args = parser.parse_args()
    run(args.pdf)