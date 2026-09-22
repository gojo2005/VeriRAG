# VeriRAG Agent Test
#
# This test verifies the self-correction workflow.
#
# Test flow:
#
# Correct source
#      ↓
# Incorrect generated answer
#      ↓
# Grounding check → False
#      ↓
# Self-correction
#      ↓
# Corrected answer
#      ↓
# Grounding check → True


from app.agent import (
    self_correct,
    grade_grounding,
)

from langchain_core.documents import Document


# Controlled source document
#
# This document contains the correct information
# that the answer must be grounded in.

test_document = Document(
    page_content=(
        "Mahatma Gandhi was born on October 2, 1869, "
        "in Porbandar, India."
    ),
    metadata={
        "source_type": "test",
        "source": "controlled_test",
    },
)


# Intentionally incorrect answer
#
# We deliberately use October 5 instead of October 2
# so that the grounding grader should reject it.

state = {
    "question": "When was Mahatma Gandhi born?",

    "documents": [
        test_document,
    ],

    "relevant": True,

    "answer": (
        "Mahatma Gandhi was born on October 5, 1869, "
        "in Porbandar, India."
    ),

    "rewritten_query": "",

    "grounded": False,

    "correction_attempts": 0,

    "sources": [],

    # No uploaded-document retriever is required
    # for this controlled test.
    "retriever": None,
}


print("INITIAL ANSWER:")
print(state["answer"])


# First grounding check

state = grade_grounding(state)

print("\nINITIAL GROUNDING:")
print(state["grounded"])


# Self-correction

if not state["grounded"]:

    state = self_correct(state)

    print("\nCORRECTED ANSWER:")
    print(state["answer"])

    print("\nCORRECTION ATTEMPTS:")
    print(state["correction_attempts"])


# Grounding check after correction

state = grade_grounding(state)

print("\nFINAL GROUNDING:")
print(state["grounded"])