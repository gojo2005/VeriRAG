# VeriRAG FastAPI Backend
#
# This module exposes the VeriRAG system through REST API endpoints.
#
# Endpoints:
#
# GET  /health
#      Check whether the API is running.
#
# POST /upload
#      Upload one or more PDF documents.
#
# POST /ask
#      Ask a question using either:
#      - user-uploaded PDFs, or
#      - the permanent ChromaDB knowledge base.


import uuid
from typing import Annotated

from fastapi import (
    FastAPI,
    File,
    UploadFile,
    HTTPException,
)

from app.agent import run_verirag

from app.ingestion import (
    process_uploaded_pdfs,
    get_session_retriever,
)

from app.schemas import (
    VeriRAGRequest,
    VeriRAGResponse,
)


# Create the FastAPI application

app = FastAPI(
    title="VeriRAG API",
    description=(
        "Self-evaluating Agentic RAG system "
        "with user PDF ingestion, web fallback, "
        "and grounding verification."
    ),
    version="1.0.0",
    openapi_version="3.0.3",
)


# Health check endpoint

@app.get("/health")
def health_check():
    """
    Check whether the VeriRAG API is running.
    """

    return {
        "status": "ok",
    }


# PDF upload endpoint

@app.post("/upload")
async def upload_documents(
    files: Annotated[
        list[UploadFile],
        File(
            description="Upload one or more PDF files",
        ),
    ],
):
    """
    Upload one or more PDF documents.

    The uploaded documents are processed as:

    PDF
      ↓
    Text extraction
      ↓
    Chunking
      ↓
    Embeddings
      ↓
    Session-specific Chroma vector store

    Returns a session_id that must be supplied with
    subsequent /ask requests.
    """

    # Make sure at least one file was uploaded.

    if not files:
        raise HTTPException(
            status_code=400,
            detail="Please upload at least one PDF file.",
        )

    # Create a unique session ID.

    session_id = str(
        uuid.uuid4()
    )

    pdf_contents = []
    filenames = []

    # Process every uploaded file.

    for file in files:

        # Check the uploaded file type.

        if file.content_type != "application/pdf":
            raise HTTPException(
                status_code=400,
                detail=(
                    f"{file.filename} is not a PDF file."
                ),
            )

        # Read the PDF into memory.

        content = await file.read()

        # Reject empty files.

        if not content:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"{file.filename} is empty."
                ),
            )

        # Store both the file content and original filename.

        pdf_contents.append(
            content
        )

        filenames.append(
            file.filename or "uploaded_document.pdf"
        )

    try:

        # Process the uploaded PDFs.
        #
        # PDF
        # ↓
        # Documents
        # ↓
        # Chunks
        # ↓
        # Embeddings
        # ↓
        # Session Vector Store

        result = process_uploaded_pdfs(
            session_id=session_id,
            files=pdf_contents,
            filenames=filenames,
        )

    except ValueError as exc:

        # Handle expected document-processing errors.

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except Exception as exc:

        # Convert unexpected ingestion errors into
        # a clean API response instead of exposing
        # internal application details.

        raise HTTPException(
            status_code=500,
            detail="Failed to process the uploaded PDF documents.",
        ) from exc

    return {
        "session_id": session_id,
        "message": "Documents processed successfully.",
        **result,
    }


# Question-answering endpoint

@app.post(
    "/ask",
    response_model=VeriRAGResponse,
)
def ask_question(
    request: VeriRAGRequest,
):
    """
    Ask a question to the VeriRAG agent.

    If session_id is provided:
        Search the user's uploaded PDFs first.

    If session_id is not provided:
        Search the permanent ChromaDB.

    If the retrieved documents are insufficient:
        Rewrite the query and use web fallback.

    The final answer is checked for grounding.
    """

    # Get the retriever associated with the
    # user's uploaded-document session.

    retriever = get_session_retriever(
        request.session_id
    )

    # Run the complete VeriRAG workflow.

    response = run_verirag(
        request.question,
        retriever=retriever,
    )

    return response