# VeriRAG Retrieval Module
#
# Responsible for:
# 1. Loading the embedding model
# 2. Connecting to the permanent ChromaDB
# 3. Creating the retriever
# 4. Retrieving relevant documents
#
# This module does NOT:
# - generate answers
# - grade documents
# - perform web search
# - process user-uploaded PDFs
#
# User-uploaded PDF retrieval is handled by ingestion.py.
# The agent decides whether to use this permanent retriever
# or a session-specific uploaded-document retriever.


from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from app.config import (
    EMBEDDING_MODEL,
    CHROMA_PERSIST_DIRECTORY,
    RETRIEVAL_K,
)


# Load the embedding model once.
#
# The same embedding model must be used when:
# 1. Creating the original ChromaDB
# 2. Querying the ChromaDB
#
# Otherwise, the vector representations would not
# be compatible.

embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
)


# Connect to the existing permanent ChromaDB.
#
# This database contains the original documents
# that were processed during the project setup.
#
# User-uploaded PDFs do NOT get added here.

vectorstore = Chroma(
    persist_directory=CHROMA_PERSIST_DIRECTORY,
    embedding_function=embedding_model,
)


# Create the permanent document retriever.

retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": RETRIEVAL_K,
    }
)


# Retrieve relevant documents

def retrieve_documents(
    question: str,
):
    """
    Retrieve relevant documents from the permanent ChromaDB.

    Parameters
    ----------
    question : str
        User's question.

    Returns
    -------
    list[Document]
        Retrieved documents from the permanent knowledge base.
    """

    # Avoid performing a meaningless vector search
    # when the question is empty.

    if not question or not question.strip():
        return []

    documents = retriever.invoke(
        question.strip()
    )

    return documents