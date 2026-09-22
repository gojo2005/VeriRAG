# VeriRAG LLM Test
#
# This file is a simple test to verify that:
# 1. The Groq API key is loaded correctly.
# 2. The configured Groq model is accessible.
# 3. The LLM can generate a response.


from langchain_groq import ChatGroq

from app.config import (
    GROQ_MODEL,
    LLM_TEMPERATURE,
)


# Create the LLM using the same configuration
# used by the VeriRAG application.

llm = ChatGroq(
    model=GROQ_MODEL,
    temperature=LLM_TEMPERATURE,
)


# Send a simple test prompt.

response = llm.invoke(
    "Hello! Introduce yourself in one sentence."
)


# Display the generated response.

print("LLM RESPONSE:")
print(response.content)