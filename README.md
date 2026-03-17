# 📚 AI Research Assistant using RAG

An intelligent Q&A system that answers questions from your documents using **Retrieval-Augmented Generation (RAG)**. Upload any PDF, ask questions in natural language, and get accurate, context-aware answers powered by Google Gemini 2.5 Flash.

---

## ✨ Features

- 📄 Upload and process multiple PDF documents
- 🔍 Semantic similarity search using FAISS vector database
- 🧠 Context-aware answers using Google Gemini 2.5 Flash
- 💬 Multi-turn conversation with chat history
- 🖥️ Streamlit web UI + CLI + test script
- 📊 Source tracking per answer

## 🔄 How RAG Works

```
PDF Upload
    │
    ▼
┌─────────────┐
│  ingest.py  │  → Parses PDF, splits into 500-token chunks (100 overlap)
└─────────────┘
    │
    ▼
┌─────────────┐
│   embed.py  │  → Converts chunks to 384-dim vectors using MiniLM
└─────────────┘
    │
    ▼
┌─────────────┐
│   FAISS DB  │  → Indexes vectors for fast cosine similarity search
└─────────────┘
    │
    ▼  (at query time)
┌──────────────┐
│ retriever.py │  → Finds top-5 most relevant chunks for the question
└──────────────┘
    │
    ▼
┌──────────────┐
│ generator.py │  → Sends question + chunks + history to Gemini 2.5 Flash
└──────────────┘
    │
    ▼
  Answer
```

---
## 🚀 Demo

```
❓ Question: What is RAG and how does it work?

🤖 Answer: Retrieval-Augmented Generation (RAG) is a technique that combines a
retrieval system with a language model to produce more accurate and grounded
responses. Instead of relying purely on the LLM's parametric knowledge, RAG
fetches relevant documents from an external knowledge base and provides them
as context to the model at inference time. The pipeline consists of three stages:
document ingestion and chunking, embedding and vector indexing, and retrieval
and generation.
```

---


---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.13 |
| LLM | Google Gemini 2.5 Flash |
| Embeddings | HuggingFace `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Database | FAISS (Facebook AI Similarity Search) |
| Framework | LangChain v0.3 |
| Document Parsing | PyPDF + LangChain Text Splitters |
| UI | Streamlit |

---

## 📁 Project Structure

```
rag_assistant/
├── app/
│   ├── __init__.py
│   ├── ingest.py        # PDF loading and text chunking
│   ├── embed.py         # HuggingFace embeddings + FAISS index
│   ├── retriever.py     # Cosine similarity search
│   └── generator.py     # Gemini 2.5 Flash answer generation
├── vectorstore/         # Auto-created after processing docs
├── data/
│   └── sample.pdf       # Place your PDFs here
├── ui.py                # Streamlit web interface
├── main.py              # CLI entry point
├── test.py              # End-to-end pipeline test
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```



## 🧪 Sample Test Questions

Use these with `sample.pdf` to verify your setup:

```
What is RAG and how does it work?
What are the types of machine learning?
Who coined the term Artificial Intelligence?
What is the bias-variance tradeoff?
What is the difference between CNN and RNN?
What ethical concerns exist in AI?
Which LLMs are mentioned in the document?
What is the transformer architecture?
```

---

## 📦 Dependencies

```
langchain>=0.3.0
langchain-community>=0.3.0
langchain-core>=0.3.0
langchain-text-splitters>=0.3.0
langchain-huggingface>=0.1.0
faiss-cpu>=1.8.0
sentence-transformers>=3.0.0
pypdf>=4.2.0
python-dotenv>=1.0.0
google-genai>=1.0.0
streamlit>=1.35.0
tiktoken>=0.7.0
```

---



---
