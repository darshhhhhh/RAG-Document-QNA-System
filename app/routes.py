import shutil
import sys
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from app.schemas import AskRequest, AskResponse, HealthResponse, ResetResponse, UploadResponse
from src.config import UPLOADED_DOCS_DIR
from src.ingestion_pipeline import DocumentIngestionPipeline
from src.rag_pipeline import RAGPipeline
from src.vector_store import VectorStore


router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health_check():
    return {
        "status": "success",
        "message": "RAG Document Q&A API is running."
    }


@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported."
        )

    try:
        UPLOADED_DOCS_DIR.mkdir(parents=True, exist_ok=True)

        file_path = UPLOADED_DOCS_DIR / file.filename

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        ingestion_pipeline = DocumentIngestionPipeline()

        summary = ingestion_pipeline.ingest_pdf(
            pdf_path=file_path,
            reset_collection=True
        )

        return summary

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Document upload failed: {error}"
        )


@router.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest):
    try:
        vector_store = VectorStore()

        if vector_store.count() == 0:
            raise HTTPException(
                status_code=400,
                detail="No document has been processed yet. Upload a PDF first."
            )

        rag_pipeline = RAGPipeline()
        response = rag_pipeline.ask(request.question)

        return {
            "question": response["question"],
            "answer": response["answer"],
            "confidence": response["confidence"],
            "best_score": response["best_score"],
            "answer_type": response["answer_type"],
            "sources": response["sources"],
            "retrieved_context": response["retrieved_context"]
        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Question answering failed: {error}"
        )


@router.delete("/reset", response_model=ResetResponse)
def reset_vector_database():
    try:
        vector_store = VectorStore()
        vector_store.reset_collection()

        return {
            "status": "success",
            "message": "Vector database reset successfully.",
            "vector_db_count": vector_store.count()
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Reset failed: {error}"
        )