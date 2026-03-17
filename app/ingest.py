import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_multiple_pdfs(uploaded_files):
    """
    Load and chunk multiple PDF files uploaded via Streamlit.
    Returns a list of LangChain Document objects.
    """
    all_docs = []
    temp_files = []

    for file in uploaded_files:
        temp_path = f"temp_{file.name}"
        temp_files.append(temp_path)

        with open(temp_path, "wb") as f:
            f.write(file.getbuffer())

        try:
            loader = PyPDFLoader(temp_path)
            docs = loader.load()

            for doc in docs:
                doc.metadata["source"] = file.name

            all_docs.extend(docs)
        except Exception as e:
            raise RuntimeError(f"Failed to load '{file.name}': {e}")

    # Clean up temp files
    for path in temp_files:
        if os.path.exists(path):
            os.remove(path)

    if not all_docs:
        raise ValueError("No text could be extracted from the uploaded PDFs.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    chunks = splitter.split_documents(all_docs)
    return chunks


def load_single_pdf(file_path: str):
    """
    Load and chunk a single PDF from a filesystem path (used by main.py CLI).
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    loader = PyPDFLoader(file_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    return splitter.split_documents(docs)