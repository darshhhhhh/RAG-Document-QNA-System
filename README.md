# RAG-Based Document Q&A System

A production-style Retrieval-Augmented Generation system that allows users to upload PDF documents and ask source-grounded questions from the uploaded content.

This project demonstrates practical skills in Python engineering, document processing, text cleaning, metadata-aware chunking, sentence embeddings, vector databases, semantic retrieval, re-ranking, FastAPI backend development, Streamlit frontend development, testing, and deployment preparation.

---

## Features

- Upload and process PDF documents
- Extract page-level text from PDFs
- Clean noisy extracted text
- Split text into metadata-aware chunks
- Generate dense embeddings using Sentence Transformers
- Store document chunks in ChromaDB
- Retrieve relevant chunks using vector search
- Re-rank retrieved chunks using a Cross-Encoder model
- Generate grounded answers from retrieved context
- Return confidence score and source citations
- Fallback response when answer is not found
- Streamlit-based chat interface
- FastAPI backend with upload and ask endpoints
- Pytest test suite
- Docker-ready backend
- GitHub Actions CI workflow

---

## Tech Stack

| Area | Tools |
|---|---|
| Language | Python |
| PDF Parsing | pypdf |
| Embeddings | Sentence Transformers |
| Vector Database | ChromaDB |
| Re-ranking | Cross-Encoder |
| Backend | FastAPI |
| Frontend | Streamlit |
| Testing | pytest |
| Deployment | Docker |
| CI/CD | GitHub Actions |

---

## Project Architecture

```text
PDF Upload
   ↓
Document Loader
   ↓
Text Cleaner
   ↓
Metadata-Aware Chunking
   ↓
Embedding Generation
   ↓
ChromaDB Vector Store
   ↓
Retriever
   ↓
Cross-Encoder Re-ranker
   ↓
Answer Generator
   ↓
Answer + Confidence + Sources