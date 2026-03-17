import pickle
from app.embed import VECTORSTORE_PATH


def get_retriever(k: int = 5):
    """
    Load FAISS vectorstore from disk and return a retriever.

    Args:
        k: Number of top similar chunks to retrieve per query.

    Returns:
        A LangChain retriever object.
    """
    with open(VECTORSTORE_PATH, "rb") as f:
        vectordb = pickle.load(f)

    return vectordb.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )


def get_relevant_docs(query: str, k: int = 5):
    """
    Convenience function: retrieve top-k docs for a query string.
    Returns a list of LangChain Document objects.
    """
    retriever = get_retriever(k=k)
    return retriever.invoke(query)