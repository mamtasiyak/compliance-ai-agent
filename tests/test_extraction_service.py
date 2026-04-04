import pytest

import json

from app.schemas.compliance import RiskLevel
from app.services.extraction_service import ExtractionService, BaseLLMProvider


class FakeLLMProvider(BaseLLMProvider):
    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        return json.dumps({
            "document_summary": "A test document with two obligations.",
            "obligations": [
                {
                    "obligation": "Obligation One",
                    "description": "First test obligation description.",
                    "risk_level": "High",
                    "deadline": None,
                    "responsible_entity": None,
                    "source_text": "Must do obligation one.",
                    "section_reference": None,
                    "confidence": 0.9,
                },
                {
                    "obligation": "Obligation Two",
                    "description": "Second test obligation description.",
                    "risk_level": "Low",
                    "deadline": None,
                    "responsible_entity": None,
                    "source_text": "Shall do obligation two.",
                    "section_reference": None,
                    "confidence": 0.8,
                },
            ],
        })
    
@pytest.mark.asyncio
async def test_extraction_returns_valid_result():
    # Arrange: creating the service with fake provider
    service = ExtractionService(provider=FakeLLMProvider())

    # Act: calling the method under test
    result = await service.extract("Some sample document text.")

    # Assert: checking the results
    assert result.total_obligations == 2
    assert len(result.obligations) == 2


@pytest.mark.asyncio
async def test_risk_level_is_normalized():
    class LowercaseRiskProvider(BaseLLMProvider):
        async def complete(self, system_prompt: str, user_prompt: str) -> str:
            return json.dumps({
                "document_summary": "A document testing risk level normalization.",
                "obligations": [
                    {
                        "obligation": "Test Obligation",
                        "description": "Testing that lowercase risk level is normalized.",
                        "risk_level": "high",       # <-- lowercase, as LLMs often return
                        "deadline": None,
                        "responsible_entity": None,
                        "source_text": "Must normalize risk levels.",
                        "section_reference": None,
                        "confidence": 0.9,
                    }
                ],
            })

    service = ExtractionService(provider=LowercaseRiskProvider())
    result = await service.extract("Some document text.")

    assert result.obligations[0].risk_level == RiskLevel.HIGH

@pytest.mark.asyncio
async def test_invalid_obligation_is_skipped():
    class OneInvalidProvider(BaseLLMProvider):
        async def complete(self, system_prompt: str, user_prompt: str) -> str:
            return json.dumps({
                "document_summary": "A document with one valid and one invalid obligation.",
                "obligations": [
                    {
                        "obligation": "Valid Obligation",
                        "description": "This obligation is perfectly valid.",
                        "risk_level": "High",
                        "deadline": None,
                        "responsible_entity": None,
                        "source_text": "Must comply with valid obligation.",
                        "section_reference": None,
                        "confidence": 0.9,
                    },
                    {
                        "obligation": "Invalid Obligation",
                        "description": "This obligation has an out-of-range confidence.",
                        "risk_level": "Low",
                        "deadline": None,
                        "responsible_entity": None,
                        "source_text": "Shall comply with invalid obligation.",
                        "section_reference": None,
                        "confidence": 1.5,  # <-- invalid: must be <= 1.0
                    },
                ],
            })

    service = ExtractionService(provider=OneInvalidProvider())
    result = await service.extract("Some document text.")

    assert len(result.obligations) == 1
    assert result.total_obligations == 1

