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
from unittest import result

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
    
_SYSTEM_PROMPT = """
You are a compliance extraction specialist. Your task is to extract distinct compliance obligations from the given document.

DEFINITION OF A COMPLIANCE OBLIGATION:
A compliance obligation is a single, distinct requirement that an organisation or individual must fulfil. It may span multiple sentences if those sentences together describe ONE requirement (e.g. what must be done, who is responsible, and by when).

DO NOT create a separate obligation for:
- Sentences that describe HOW to fulfil an existing obligation
- Sentences that assign responsibility for the same obligation
- Sentences that set the timeline for the same obligation
These should be captured within the SAME obligation entry.

WHEN TO CREATE A NEW OBLIGATION:
Only create a new obligation when the document introduces a NEW and DISTINCT requirement — a different action, a different subject, or a different regulatory rule.

EXAMPLES:
BAD (over-splitting):
- Obligation 1: "Employees must complete cybersecurity training within 14 days."
- Obligation 2: "HR shall track completion status."
→ These are ONE obligation. HR tracking is the enforcement mechanism, not a separate requirement.

GOOD (correct grouping):
- Obligation 1: "Cybersecurity Awareness Training — Employees must complete training within 14 days of joining. HR shall track completion and send reminders."

ADDITIONAL RULES:
- NEVER return an empty obligations list if at least one obligation exists.
- Do NOT summarise instead of extracting.
- obligation titles must be short (5-8 words max).
- risk_level must be exactly one of: Low, Medium, High, Critical.
- confidence reflects how clearly the obligation is stated (0.0–1.0).

Return ONLY valid JSON in this exact format — no markdown, no explanation:

{
  "document_summary": "<short summary of the document>",
  "obligations": [
    {
      "obligation": "<short title>",
      "description": "<full explanation of what must be done, who is responsible, and by when>",
      "risk_level": "<Low | Medium | High | Critical>",
      "deadline": "<deadline or null>",
      "responsible_entity": "<responsible party or null>",
      "source_text": "<exact text from the document>",
      "section_reference": "<section reference or null>",
      "confidence": <0.0-1.0>
    }
  ]
}
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

    logger.info(f"Raw LLM response: {raw[:500]}")
    logger.info("Parsing LLM response.")

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
        logger.error("Failed to parse LLM response as JSON.")
        raise LLMResponseParseError(
            f"Invalid JSON from LLM content: {e}\n\nRaw output:\n{raw[:500]}"
        ) from e
 
    # 3. Validate individual obligations, skipping invalid ones with a warning
    raw_obligations: list[dict[str, Any]] = payload.get("obligations", [])
    valid_obligations: list[ComplianceObligation] = []
 
    for idx, raw_ob in enumerate(raw_obligations):
        # --- normalization layer ---
        if "risk_level" in raw_ob and isinstance(raw_ob["risk_level"], str):
            raw_ob["risk_level"] = raw_ob["risk_level"].strip().capitalize()
        # normalize strings (common cleanup)
        for key in ["obligation", "description", "responsible_entity", "source_text"]:
            if key in raw_ob and isinstance(raw_ob[key], str):
                raw_ob[key] = raw_ob[key].strip()
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
 
def _merge_results(results: list[ExtractionResult]) -> ExtractionResult:
    """
    Merge multiple per-chunk ExtractionResults into a single result.

    Uses the first chunk's summary as the document summary, concatenates
    all obligations, and deduplicates by obligation title — keeping the
    instance with the highest confidence score.
    """
    seen: dict[str, Any] = {}
    for r in results:
        for ob in r.obligations:
            key = ob.obligation.strip().lower()
            if key not in seen or ob.confidence > seen[key].confidence:
                seen[key] = ob

    unique_obligations = list(seen.values())

    return ExtractionResult(
        document_summary=results[0].document_summary,
        total_obligations=len(unique_obligations),
        obligations=unique_obligations,
    )

def _chunk_text(text: str, chunk_size: int = 3000, overlap: int = 200) -> list[str]:
    """
    Split text into overlapping chunks to avoid cutting obligations in half.

    Args:
        text: The full document text.
        chunk_size: Maximum characters per chunk.
        overlap: How many characters to repeat at the start of each new chunk.

    Returns:
        List of text chunks.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


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

    async def _extract_chunk(self, chunk: str) -> ExtractionResult:
        """Run extraction pipeline on a single chunk of text."""
        system_prompt = _SYSTEM_PROMPT
        user_prompt = _build_user_prompt(chunk)

        logger.info("Sending prompt to LLM provider.")

        try:
            raw_response = await self._provider.complete(system_prompt, user_prompt)
            if not raw_response or not raw_response.strip():
                logger.error("LLM returned empty response.")
                raise RuntimeError("LLM returned empty response.")

            logger.info("Received raw response from LLM (%d chars).", len(raw_response))

        except Exception as e:
            logger.exception("LLM provider raised an unexpected error.")
            raise RuntimeError(
                "The LLM provider failed to return a response. "
                "Check provider configuration and network connectivity."
            ) from e

        return _parse_llm_response(raw_response)

    async def extract(self, document_text: str) -> ExtractionResult:
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

        if len(document_text) <= 3000:
            result = await self._extract_chunk(document_text)
        else:
            chunks = _chunk_text(document_text)
            logger.info("Document split into %d chunk(s).", len(chunks))
            results = []
            for i, chunk in enumerate(chunks):
                logger.info("Extracting chunk %d/%d.", i + 1, len(chunks))
                results.append(await self._extract_chunk(chunk))
            result = _merge_results(results)

        logger.info("Extraction complete: %d obligation(s) found.", result.total_obligations)
        if result.total_obligations == 0:
            logger.warning("No obligations extracted from document.")

        return result