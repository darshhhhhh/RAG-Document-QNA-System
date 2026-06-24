import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from app.routes import router


app = FastAPI(
    title="RAG-Based Document Q&A System",
    description=(
        "A production-style Retrieval-Augmented Generation API for "
        "uploading PDF documents and asking source-grounded questions."
    ),
    version="1.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(router)


@app.get("/")
def root():
    return {
        "message": "Welcome to the RAG-Based Document Q&A System API.",
        "docs": "/docs",
        "health": "/health"
    }