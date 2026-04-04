"""
FastAPI application entrypoint for the compliance obligation extraction service.
"""

from dotenv import load_dotenv
load_dotenv()

import logging

from fastapi import FastAPI

from app.schemas.document import DocumentInput
from app.schemas.extraction import ExtractionResult
from app.services.extraction_service import ExtractionService
from app.services.openai_provider import OpenAIProvider

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
)
logger = logging.getLogger(__name__)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

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

    try:
        return await service.extract(payload.document_text)
    except Exception as e:
        logger.error("Extraction failed: %s", str(e))
        return ExtractionResult(
        document_summary="Extraction failed due to an internal error.",
        total_obligations=0,
        obligations=[],
    )