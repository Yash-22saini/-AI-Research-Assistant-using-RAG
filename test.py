from app.ingest import load_single_pdf
from app.embed import create_vectorstore
from app.retriever import get_relevant_docs
from app.generator import generate_answer

docs = load_single_pdf("data/sample.pdf")
create_vectorstore(docs)

query = "What is this document about?"
retrieved_docs = get_relevant_docs(query)

answer = generate_answer(query, retrieved_docs, [])

print("\nANSWER:\n", answer)