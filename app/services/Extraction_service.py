"""
Service layer for compliance obligation Extraction.

ExtractionService encapsulates all business logic related to analysing documents
and producing structured ExtractionResult objects. Keeping this logic here — rather
than inside route handlers — ensures the extraction pipeline remains independently
testable and reusable across transports (HTTP, CLI, background tasks, etc.).
"""

import json
import logging
from abc import ABC, abstractmethod
from typing import Any

from pydantic import ValidationError

from app.schemas.extraction import ExtractionResult
from app.schemas.compliance import ComplianceObligation

logger = logging.getLogger(__name__)


class BaseLLMProvider(ABC):
    """
    Minimal interface every LLM provider adapter must implement.
 
    Swap providers by passing a different concrete subclass to ExtractionService.
    """

    @abstractmethod
    async def complete(self, system_prompt:str, user_prompt:str) -> str:
        """
        Send a chat-completion request and return the raw text response.
 
        Args:
            system_prompt: High-level instructions / persona for the model.
            user_prompt:   The user-turn content (document text, task description).
 
        Returns:
            The model's raw text output.
        """

class PlaceholderLLMProvider(BaseLLMProvider):
    """
    Stand-in LLM provider used during development and unit tests.
 
    Replace with OpenAIProvider, AnthropicProvider, etc. for production.
    """
 
    async def complete(self, system_prompt: str, user_prompt: str) -> str:  
        """
        Return a minimal but schema-valid JSON stub.
 
        The stub demonstrates the expected response envelope so that the
        parsing pipeline can be exercised end-to-end without live API calls.
        """
        stub: dict[str, Any] = {
            "document_summary": (
                "This document establishes internal data-handling and security "
                "obligations for all staff processing personal data."
            ),
            "obligations": [
                {
                    "obligation": "Annual Data Retention Review",
                    "description": (
                        "All personally identifiable data stores must be audited "
                        "annually and records purged when the retention period expires."
                    ),
                    "risk_level": "High",
                    "deadline": "Annually by 31 March",
                    "responsible_entity": "Data Protection Officer",
                    "source_text": (
                        "Personal data shall be kept no longer than necessary for "
                        "the purpose for which it was collected."
                    ),
                    "section_reference": "Section 4.2",
                    "confidence": 0.91,
                },
                {
                    "obligation": "72-Hour Breach Notification",
                    "description": (
                        "The organisation must notify the relevant supervisory authority "
                        "within 72 hours of becoming aware of a personal data breach."
                    ),
                    "risk_level": "Critical",
                    "deadline": "Within 72 hours of breach discovery",
                    "responsible_entity": "Information Security Team",
                    "source_text": (
                        "The controller shall notify the supervisory authority of a "
                        "personal data breach without undue delay."
                    ),
                    "section_reference": "Clause 8(b)",
                    "confidence": 0.97,
                },
            ],
        }
        return json.dumps(stub)
    
_SYSTEM_PROMPT = """\
You are a specialist compliance analyst. Your task is to read document 
provided by the user and extract every distinct compliance obligation it contains.
Return only a valid JSON object - no markdown, no commentary - with this exact shape:
{
    "document_summary": "<one to three sentence summary of the document>",
    "obligations":[
    {   
        "obligation":           "<short title of the rule>",
        "description":          "<detailed explanation of the obligation>",
        "risk_level":           "<low | medium | high | critical>",
        "deadline":             "<time requirement, or null>",
        "responsible_entity":   "<accountable team or role, or null>",
        "source_text":          "<verbatim excerpt from the document>",
        "section_reference":    "<section or page reference, or null>",
        "confidence":           <float 0.0-1.0>
        }
    ]
}

Rules:
- Extract every obligation; do not omit any.
- Set confidence to reflect how clearly the text implies the obligation.
- Use null (not an empty string) for optional fields that are absent.
- Do not invent obligations that are not supported by the document text.
- Even if the document is short, extract any implied or explicit obligations.
- Do not return an empty list unless absolutely no obligations exist.
- Treat sentences with "must", "shall", "required", or "need to" as obligations.
""" 

def _build_user_prompt(document_text: str) -> str:
    return f"Document to analyse: \n\n{document_text}"
 
class LLMResponseParseError(Exception):
    """Raised when the LLM response cannot be parsed into a valid ExtractionResult."""

def _parse_llm_response(raw:str) -> ExtractionResult:
    """
    Safely deserialise the LLM's raw text output into an ExtractionResult.
 
    Steps:
      1. Strip accidental markdown fences.
      2. Parse as JSON.
      3. Validate each obligation via the ComplianceObligation schema.
      4. Assemble and return an ExtractionResult.
 
    Raises:
        LLMResponseParseError: On invalid JSON or schema validation failure.
    """
    # 1. Strip markdown code fences if the model wrapped its output
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        cleaned = "\n".join(
            line for line in lines if not line.strip().startswith("```")
        ).strip()
 
    # 2. Parse JSON
    try:
        payload: dict[str, Any] = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise LLMResponseParseError(
            f"LLM returned non-JSON content: {e}\n\nRaw output:\n{raw[:500]}"
        ) from e
 
    # 3. Validate individual obligations, skipping invalid ones with a warning
    raw_obligations: list[dict[str, Any]] = payload.get("obligations", [])
    valid_obligations: list[ComplianceObligation] = []
 
    for idx, raw_ob in enumerate(raw_obligations):
        try:
            valid_obligations.append(ComplianceObligation.model_validate(raw_ob))
        except ValidationError as e:
            logger.warning(
                "Obligation at index %d failed validation and was skipped: %s",
                idx,
                e,
            )
 
    # 4. Assemble ExtractionResult (model_validator enforces total_obligations consistency)
    try:
        return ExtractionResult(
            document_summary=payload.get("document_summary", "No summary provided."),
            total_obligations=len(valid_obligations),
            obligations=valid_obligations,
        )
    except ValidationError as e:
        raise LLMResponseParseError(
            f"Failed to construct ExtractionResult: {e}"
        ) from e
 
 


class ExtractionService:
    """
    Orchestrates the end-to-end compliance obligation extraction pipeline.
    
    Responsibilities:
    - Accept raw document text as input.
    - Coordinate preprocessing, LLM inference, and postprocessing steps.
    - Return a validated ExtractionResult to the caller.

    Usage::
 
        service = ExtractionService()                          # default (placeholder)
        service = ExtractionService(provider=OpenAIProvider()) # production
 
    The service is stateless; a single instance can safely handle concurrent requests.
    """
    def __init__ (self, provider: BaseLLMProvider | None = None) -> None:
        """
        Initialise the service with an LLM provider.
 
        Args:
            provider: A concrete BaseLLMProvider implementation.
                      Defaults to PlaceholderLLMProvider when not supplied.
        """
        self._provider = provider or PlaceholderLLMProvider()

    async def extract(self, document_text:str) -> ExtractionResult:
        """
        Analyse *document_text* and return all detected compliance obligations.
 
        Args:
            document_text: The full plain-text content of the document to analyse.
 
        Returns:
            A validated ExtractionResult containing the document summary,
            obligation count, and list of ComplianceObligation objects.
 
        Raises:
            LLMResponseParseError: If the LLM output cannot be parsed.
            RuntimeError: If the LLM provider raises an unexpected error.
        """

        logger.info("Starting extraction for document of length (%d chars).", len(document_text))

        system_prompt = _SYSTEM_PROMPT
        user_prompt = _build_user_prompt(document_text)

        try:
            raw_response = await self._provider.complete(system_prompt, user_prompt)
        except Exception as e:
            logger.exception("LLM provider raised an unexpected error.")
            raise RuntimeError(
                "The LLM provider failed to return a response. "
                "Check provider configuration and network connectivity."
            ) from e
        
        result = _parse_llm_response(raw_response)
        logger.info("Extraction complete: %d obligation(s) found.", result.total_obligations)
        return result