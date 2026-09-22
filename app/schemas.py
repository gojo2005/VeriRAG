from pydantic import BaseModel, Field


class VeriRAGRequest(BaseModel):
    question: str = Field(
        min_length=1,
        description="The question asked by the user.",
    )

    session_id: str | None = Field(
        default=None,
        description=(
            "Optional session ID associated with "
            "user-uploaded PDF documents."
        ),
    )


class VeriRAGResponse(BaseModel):
    question: str
    answer: str
    sources: list[str]
    grounded: bool

    # Step-by-step execution trace of the RAG agent
    trace: list[str]