# VeriRAG Document Ingestion
#
# This module processes user-uploaded PDF documents.
#
# Pipeline:
#
# PDF
#  ↓
# Temporary file
#  ↓
# PDF pages
#  ↓
# Text chunks
#  ↓
# Embeddings
#  ↓
# Session Vector Store
#
# Each session gets its own vector store so that
# uploaded documents remain isolated between sessions.


import os
import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import (
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    RETRIEVAL_K,
)


# Load the embedding model once.
#
# The same embedding model is reused for all
# uploaded documents.

embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
)


# Store session-specific vector stores in memory.
#
# Example:
#
# session_1 → Vector store containing User A's PDFs
# session_2 → Vector store containing User B's PDFs
#
# This keeps uploaded documents isolated between sessions.

session_vectorstores = {}


def process_uploaded_pdfs(
    session_id: str,
    files: list[bytes],
    filenames: list[str] | None = None,
) -> dict:
    """
    Process user-uploaded PDF files.

    Pipeline:

    1. Save each PDF temporarily.
    2. Load the PDF pages.
    3. Preserve source filename and page metadata.
    4. Split documents into chunks.
    5. Generate embeddings.
    6. Create a session-specific Chroma vector store.
    7. Store the vector store against the session ID.

    Parameters
    ----------
    session_id:
        Unique identifier for the user's session.

    files:
        PDF contents as bytes.

    filenames:
        Original uploaded PDF filenames.

    Returns
    -------
    dict
        Information about the processed documents.
    """

    if not files:
        raise ValueError(
            "No PDF files were provided."
        )

    # If filenames are not provided, create generic names.
    if filenames is None:
        filenames = [
            f"document_{index}.pdf"
            for index in range(1, len(files) + 1)
        ]

    if len(files) != len(filenames):
        raise ValueError(
            "The number of files and filenames must match."
        )

    documents = []

    # Process every uploaded PDF.
    for file_content, filename in zip(
        files,
        filenames,
    ):

        if not file_content:
            continue

        temp_path = None

        try:
            # Create a temporary PDF file.
            #
            # delete=False allows PyPDFLoader to access
            # the file after the file handle is closed.

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf",
            ) as temp_file:

                temp_file.write(file_content)

                temp_path = temp_file.name

            # Load all pages from the PDF.
            loader = PyPDFLoader(temp_path)

            pdf_documents = loader.load()

            # Preserve useful source metadata.
            #
            # PyPDFLoader already provides page information,
            # but we explicitly add the original filename
            # so that it can later be shown to the user.

            for document in pdf_documents:

                document.metadata["source_type"] = "pdf"

                document.metadata["source_file"] = filename

                # PyPDFLoader uses zero-based page numbers.
                #
                # Convert to a human-friendly one-based page
                # number for displaying to the user.

                if "page" in document.metadata:
                    document.metadata["page"] = (
                        document.metadata["page"] + 1
                    )

            documents.extend(pdf_documents)

        finally:
            # Remove the temporary PDF file.
            #
            # The document contents have already been loaded
            # into memory, so the temporary file is no longer
            # required.

            if (
                temp_path is not None
                and os.path.exists(temp_path)
            ):
                os.remove(temp_path)

    if not documents:
        raise ValueError(
            "No readable content was found in the uploaded PDFs."
        )

    # Split the documents into smaller chunks.
    #
    # These values were tested during the notebook
    # experimentation phase.

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chunks = splitter.split_documents(
        documents
    )

    if not chunks:
        raise ValueError(
            "No text chunks could be created from the uploaded PDFs."
        )

    # Create a Chroma vector store from the chunks.
    #
    # This vector store is NOT the permanent chroma_db.
    # It belongs only to the current session.

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
    )

    # Associate the vector store with the session.

    session_vectorstores[session_id] = vectorstore

    # Return processing information that can be
    # shown by the API and Streamlit frontend.

    return {
        "files_processed": len(files),
        "pages": len(documents),
        "chunks": len(chunks),
    }


def get_session_retriever(
    session_id: str | None,
):
    """
    Get the retriever associated with a session.

    Returns None if:
    - No session ID was provided.
    - The session does not have uploaded documents.
    """

    if not session_id:
        return None

    vectorstore = session_vectorstores.get(
        session_id
    )

    if vectorstore is None:
        return None

    # Convert the session vector store into
    # a retriever.

    return vectorstore.as_retriever(
        search_kwargs={
            "k": RETRIEVAL_K,
        }
    )