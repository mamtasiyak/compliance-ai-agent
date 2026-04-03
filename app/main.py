"""
FastAPI application entrypoint for the compliance obligation extraction service.
"""

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI

from app.schemas.document import DocumentInput
from app.schemas.extraction import ExtractionResult
from app.services.Extraction_service import ExtractionService
from app.services.openai_provider import OpenAIProvider

app = FastAPI(
    title="Compliance Obligation Extractor",
    description="AI-powered API for extracting compliance obligations from documents.",
    version="0.1.0",
)

@app.post("/extract",
          response_model=ExtractionResult,
          status_code=200,
          summary="Extract compliance obligations from a document",
          tags=["Extraction"],
          )
async def extract_obligations(payload: DocumentInput) -> ExtractionResult:
    """
    Accept a raw document and return extracted compliance obligations.

    Delegates all extraction logic to ExtractionService.

    - **document_text**: Full text of the document to analyse.
    """
    # service = ExtractionService()
    service = ExtractionService(provider=OpenAIProvider())
    return await service.extract(payload.document_text)