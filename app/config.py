# VeriRAG Configuration
#
# Centralized configuration for the VeriRAG application.
# Keeping configuration here prevents model names, retrieval
# settings, and other constants from being scattered
# throughout the project.


import os

from dotenv import load_dotenv


# Load environment variables from .env

load_dotenv()


# API Keys

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")


# LLM Configuration

GROQ_MODEL = "qwen/qwen3.8-27b"

LLM_TEMPERATURE = 0


# Embedding Configuration

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# Document Chunking Configuration

CHUNK_SIZE = 1000

CHUNK_OVERLAP = 200


# Retrieval Configuration

RETRIEVAL_K = 3


# Web Search Configuration

TAVILY_MAX_RESULTS = 3


# Self-Correction Configuration

MAX_CORRECTION_ATTEMPTS = 2


# Vector Database Configuration

CHROMA_PERSIST_DIRECTORY = "chroma_db"