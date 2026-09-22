from typing import TypedDict

from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END

from app.config import (
    GROQ_MODEL,
    MAX_CORRECTION_ATTEMPTS,
    LLM_TEMPERATURE,
)
from app.retrieval import retrieve_documents
from app.graders import (
    grade_retrieval,
    grade_grounding as grade_grounding_llm,
)
from app.tools import web_search
from app.schemas import VeriRAGResponse


llm = ChatGroq(
    model=GROQ_MODEL,
    temperature=LLM_TEMPERATURE,
)


class GraphState(TypedDict):
    question: str
    documents: list[Document]
    relevant: bool
    answer: str
    rewritten_query: str
    grounded: bool
    correction_attempts: int
    sources: list[str]
    retriever: object | None

    # Human-readable execution trace
    trace: list[str]


def grade_documents(state: GraphState):
    """
    Retrieve documents and determine whether they are
    relevant enough to answer the question.
    """

    question = state["question"]

    trace = list(state.get("trace", []))

    uploaded_retriever = state.get("retriever")

    if uploaded_retriever is not None:
        documents = uploaded_retriever.invoke(question)
        retrieval_source = "uploaded PDF documents"
    else:
        documents = retrieve_documents(question)
        retrieval_source = "permanent ChromaDB"

    trace.append(
        f"✓ Retrieved {len(documents)} document chunks "
        f"from {retrieval_source}"
    )

    relevant = grade_retrieval(question, documents)

    if relevant:
        trace.append("✓ Retrieval Grader: Relevant")
    else:
        trace.append("✗ Retrieval Grader: Insufficient")

    return {
        "documents": documents,
        "relevant": relevant,
        "trace": trace,
    }


def route_after_grading(state: GraphState):
    """
    Decide whether to generate an answer directly
    or perform web fallback.
    """

    if state["relevant"]:
        return "generate"

    return "rewrite"


def generate_answer(state: GraphState):
    """
    Generate an answer using only the retrieved documents.
    """

    question = state["question"]
    documents = state["documents"]

    trace = list(state.get("trace", []))

    if not documents:
        answer = (
            "I could not find enough information in the "
            "available sources to answer this question."
        )

        trace.append("⚠ No source documents available for generation")

        return {
            "answer": answer,
            "sources": [],
            "trace": trace,
        }

    context = "\n\n".join(
        document.page_content
        for document in documents
        if document.page_content.strip()
    )

    if not context.strip():
        answer = (
            "I could not find enough information in the "
            "available sources to answer this question."
        )

        trace.append("⚠ Retrieved documents contained no usable text")

        return {
            "answer": answer,
            "sources": [],
            "trace": trace,
        }

    prompt = f"""
You are a factual question-answering assistant.

Answer the user's question using ONLY the information
contained in the provided source documents.

Do not use outside knowledge.

If the documents do not contain enough information,
clearly say that the information is insufficient.

User Question:
{question}

Source Documents:
--------------------
{context}
--------------------

Provide a concise and factual answer.
"""

    response = llm.invoke(prompt)
    answer = response.content

    sources = format_sources(documents)

    trace.append("✓ Answer generated from retrieved sources")

    return {
        "answer": answer,
        "sources": sources,
        "trace": trace,
    }


def rewrite_query(state: GraphState):
    """
    Rewrite the user's question into a better web-search query.
    """

    question = state["question"]

    trace = list(state.get("trace", []))

    prompt = f"""
Rewrite the following user question into a concise
search query suitable for a web search engine.

Do not answer the question.

User Question:
{question}

Return only the rewritten search query.
"""

    response = llm.invoke(prompt)

    rewritten_query = response.content.strip()

    trace.append(
        f"✓ Query rewritten for web search: {rewritten_query}"
    )

    return {
        "rewritten_query": rewritten_query,
        "trace": trace,
    }


def web_search_node(state: GraphState):
    """
    Search the web when the uploaded/permanent documents
    are insufficient.
    """

    query = state["rewritten_query"]

    trace = list(state.get("trace", []))

    results = web_search(query)

    trace.append(
        f"✓ Web search completed: {len(results)} results found"
    )

    documents = []

    for result in results:
        content = result.get("content", "")

        if not content:
            continue

        documents.append(
            Document(
                page_content=content,
                metadata={
                    "source_type": "web",
                    "source": "web",
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                },
            )
        )

    if not documents:
        trace.append("⚠ Web search returned no usable source content")

    return {
        "documents": documents,
        "trace": trace,
    }


def grade_grounding(state: GraphState):
    """
    Check whether the generated answer is fully supported
    by the retrieved source documents.
    """

    question = state["question"]
    answer = state["answer"]
    documents = state["documents"]

    trace = list(state.get("trace", []))

    grounded = grade_grounding_llm(
        question,
        answer,
        documents,
    )

    if grounded:
        trace.append("✓ Grounding Grader: Answer is grounded")
    else:
        trace.append("✗ Grounding Grader: Answer is NOT fully grounded")

    return {
        "grounded": grounded,
        "trace": trace,
    }


def route_after_grounding(state: GraphState):
    """
    Decide whether to finish or self-correct the answer.
    """

    if state["grounded"]:
        return "end"

    if state["correction_attempts"] >= MAX_CORRECTION_ATTEMPTS:
        return "end"

    return "self_correct"


def self_correct(state: GraphState):
    """
    Regenerate the answer using the same source documents
    after a grounding failure.
    """

    question = state["question"]
    documents = state["documents"]

    attempts = state["correction_attempts"] + 1

    trace = list(state.get("trace", []))

    trace.append(
        f"↻ Self-correction attempt {attempts}"
    )

    context = "\n\n".join(
        document.page_content
        for document in documents
        if document.page_content.strip()
    )

    prompt = f"""
You are correcting a RAG answer.

The previous answer was found to contain information
that was not fully supported by the source documents.

Answer the question again using ONLY the provided
source documents.

Do not introduce outside information.

Question:
{question}

Source Documents:
--------------------
{context}
--------------------

Return only the corrected answer.
"""

    response = llm.invoke(prompt)

    answer = response.content

    trace.append(
        f"✓ Corrected answer generated "
        f"(attempt {attempts})"
    )

    return {
        "answer": answer,
        "correction_attempts": attempts,
        "trace": trace,
    }


def format_sources(documents: list[Document]) -> list[str]:
    """
    Convert document metadata into human-readable source labels.
    """

    sources = []

    for document in documents:
        metadata = document.metadata

        # Web source
        url = metadata.get("url")

        if url:
            title = metadata.get("title") or "Web Source"
            sources.append(f"{title} — {url}")
            continue

        # PDF or local source
        source_file = (
            metadata.get("source_file")
            or metadata.get("source")
            or "Unknown source"
        )

        page = (
            metadata.get("page")
            or metadata.get("page_number")
        )

        if page is not None:
            sources.append(
                f"{source_file} — Page {page}"
            )
        else:
            sources.append(
                f"{source_file} — Page Unknown"
            )

    # Remove duplicate sources while preserving order
    return list(dict.fromkeys(sources))


workflow = StateGraph(GraphState)

workflow.add_node(
    "grade_documents",
    grade_documents,
)

workflow.add_node(
    "generate",
    generate_answer,
)

workflow.add_node(
    "rewrite",
    rewrite_query,
)

workflow.add_node(
    "web_search",
    web_search_node,
)

workflow.add_node(
    "grade_grounding",
    grade_grounding,
)

workflow.add_node(
    "self_correct",
    self_correct,
)


workflow.add_edge(
    START,
    "grade_documents",
)


workflow.add_conditional_edges(
    "grade_documents",
    route_after_grading,
    {
        "generate": "generate",
        "rewrite": "rewrite",
    },
)


workflow.add_edge(
    "rewrite",
    "web_search",
)


workflow.add_edge(
    "web_search",
    "generate",
)


workflow.add_edge(
    "generate",
    "grade_grounding",
)


workflow.add_conditional_edges(
    "grade_grounding",
    route_after_grounding,
    {
        "end": END,
        "self_correct": "self_correct",
    },
)


workflow.add_edge(
    "self_correct",
    "grade_grounding",
)


verirag_agent = workflow.compile()


def run_verirag(
    question: str,
    retriever=None,
) -> VeriRAGResponse:

    initial_state: GraphState = {
        "question": question,
        "documents": [],
        "relevant": False,
        "answer": "",
        "rewritten_query": "",
        "grounded": False,
        "correction_attempts": 0,
        "sources": [],
        "retriever": retriever,
        "trace": [],
    }

    result = verirag_agent.invoke(initial_state)

    return VeriRAGResponse(
        question=question,
        answer=result["answer"],
        sources=result["sources"],
        grounded=result["grounded"],
        trace=result["trace"],
    )