# VeriRAG Graders
#
# This module contains:
# 1. Retrieval relevance grader
# 2. Answer grounding grader
#
# Both graders use structured output so that the LLM
# returns a predictable True/False result.


from pydantic import BaseModel, Field
from langchain_groq import ChatGroq

from app.config import (
    GROQ_MODEL,
    LLM_TEMPERATURE,
)


# Create the LLM used by the graders

llm = ChatGroq(
    model=GROQ_MODEL,
    temperature=LLM_TEMPERATURE,
)


# Retrieval grading schema

class RetrievalGrade(BaseModel):
    relevant: bool = Field(
        description=(
            "True only if the retrieved documents contain "
            "enough information to meaningfully answer the "
            "user's question. False if the documents are "
            "irrelevant, insufficient, ambiguous, or do not "
            "contain the information needed to answer the question."
        )
    )


# Create structured retrieval grader

retrieval_grader = llm.with_structured_output(
    RetrievalGrade
)


# Grounding grading schema

class GroundingGrade(BaseModel):
    grounded: bool = Field(
        description=(
            "True only if every factual claim in the generated "
            "answer is supported by the provided source documents. "
            "False if any factual claim is unsupported or contradicts "
            "the provided sources."
        )
    )


# Create structured grounding grader

grounding_grader = llm.with_structured_output(
    GroundingGrade
)


# Retrieval relevance grader

def grade_retrieval(
    question: str,
    documents,
) -> bool:
    """
    Check whether the retrieved documents contain enough
    information to answer the user's question.

    Returns:
        True  -> documents are sufficiently relevant.
        False -> documents are irrelevant or insufficient.
    """

    if not documents:
        return False

    documents_text = "\n\n".join(
        document.page_content
        for document in documents
        if document.page_content.strip()
    )

    if not documents_text.strip():
        return False

    grading_prompt = f"""
You are a strict retrieval relevance grader for a
Retrieval-Augmented Generation (RAG) system.

Your task is to determine whether the retrieved documents
contain enough useful information to answer the user's question.

Question:
{question}

Retrieved Documents:
--------------------
{documents_text}
--------------------

Evaluation rules:

1. Return True only if the documents contain information
   that can meaningfully be used to answer the question.

2. Return False if the documents are only vaguely related
   to the question.

3. Return False if the documents discuss the general topic
   but do not contain enough information to answer the
   specific question.

4. If the question asks for a specific fact and the documents
   do not contain that fact, return False.

5. If the question requires information that is clearly
   outside the retrieved documents, return False.

6. Do not use outside knowledge to decide whether the
   documents are sufficient.

Return the result using the RetrievalGrade schema.
"""

    grade = retrieval_grader.invoke(
        grading_prompt
    )

    return bool(grade.relevant)


# Grounding grader

def grade_grounding(
    question: str,
    answer: str,
    documents,
) -> bool:
    """
    Check whether the generated answer is fully supported
    by the provided source documents.

    Returns:
        True  -> answer is fully grounded.
        False -> answer contains unsupported or contradictory claims.
    """

    if not documents:
        return False

    documents_text = "\n\n".join(
        document.page_content
        for document in documents
        if document.page_content.strip()
    )

    if not documents_text.strip():
        return False

    grounding_prompt = f"""
You are a strict grounding grader for a
Retrieval-Augmented Generation (RAG) system.

Your task is to determine whether the generated answer
is fully supported by the provided source documents.

Question:
{question}

Source Documents:
--------------------
{documents_text}
--------------------

Generated Answer:
--------------------
{answer}
--------------------

Evaluation rules:

1. Check every factual claim in the generated answer
   against the source documents.

2. Return False if any factual claim is not supported
   by the source documents.

3. Return False if any claim contradicts the source documents.

4. Return False if the answer introduces outside information
   that cannot be verified from the provided sources.

5. Return True only when the answer is fully supported
   by the provided source documents.

6. Do not use outside knowledge when evaluating the answer.

Return the result using the GroundingGrade schema.
"""

    grade = grounding_grader.invoke(
        grounding_prompt
    )

    return bool(grade.grounded)