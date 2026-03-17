import os
import pickle
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

VECTORSTORE_PATH = "vectorstore/faiss_index.pkl"

_embedding_model = None  # Cache the model to avoid reloading on every call


def get_embedding_model():
    """Return cached embedding model (loads once per session)."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}  # cosine similarity
        )
    return _embedding_model


def create_vectorstore(documents):
    """
    Embed documents and save FAISS index to disk.
    Returns the vectorstore object.
    """
    os.makedirs("vectorstore", exist_ok=True)

    embedding_model = get_embedding_model()
    vectordb = FAISS.from_documents(documents, embedding_model)

    with open(VECTORSTORE_PATH, "wb") as f:
        pickle.dump(vectordb, f)

    return vectordb


def vectorstore_exists():
    """Check if a saved vectorstore is available."""
    return os.path.exists(VECTORSTORE_PATH)