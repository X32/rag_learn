# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RAG (Retrieval-Augmented Generation) learning project implementing document retrieval and question answering systems.

## Architecture

Two RAG implementations:

1. **naive_rag.py** - From-scratch implementation using:
   - `sentence-transformers` for embeddings
   - `chromadb` for vector storage
   - `openrouter` for LLM access
   - Custom text chunking with sentence-aware splitting

2. **langchain_rag.py** - LangChain-based implementation using:
   - LangChain abstractions for document loading, splitting, and chaining
   - `HuggingFaceEmbeddings` for embeddings
   - `Chroma` vectorstore
   - Built-in retrieval and QA chains

## Common Commands

```bash
# Run naive RAG
python naive_rag.py

# Run LangChain RAG
python langchain_rag.py

# Install dependencies (check requirements.txt if exists)
pip install sentence-transformers chromadb openrouter python-dotenv langchain langchain-community langchain-core
```

## Configuration

- Environment variables via `.env` file (OPENROUTER_API_KEY required)
- Chunk size/overlap and top_k configurable in both scripts
- Vector DB persists to `./chroma_db*` directories

## Data Source

- `chatWithNavar-2.md` - Naval Ravikant interview transcript (Chinese), used as the knowledge base for RAG queries
