"""
OpenAI LLM provider adapter for the compliance extraction service.
 
Reads OPENAI_API_KEY from the environment.
Drop-in replacement for PlaceholderLLMProvider; pass an instance to ExtractionService(provider=OpenAIProvider()).
"""

from __future__ import annotations

import logging
import os

from app.services.Extraction_service import BaseLLMProvider
from openai import AsyncOpenAI, OpenAIError

logger = logging.getLogger(__name__)

# Default model; override via the constructor or the OPENAI_MODEL env var.
_DEFAULT_MODEL = "gpt-4o"

class OpenAIProvider(BaseLLMProvider):
    """
    Async OpenAI chat-completion adapter.
 
    Attributes:
        model:  The OpenAI model identifier to use for completions.
        client: An AsyncOpenAI client scoped to this provider instance.
    """

    def __init__(self, *, model:str | None = None, api_key:str | None = None) -> None:
        """
        Initialise the provider.
 
        Args:
            model:   OpenAI model identifier.
                     Falls back to the ``OPENAI_MODEL`` env var, then ``gpt-4o``.
            api_key: OpenAI API key.
                     Falls back to the ``OPENAI_API_KEY`` env var.
        """

        self.model: str = (model or os.environ.get("OPENAI_MODEL", _DEFAULT_MODEL))

        self.client: AsyncOpenAI = AsyncOpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        """
        Send a chat-completion request to OpenAI and return the model's text.

        Args:
            system_prompt: Instruction passed as the ``system`` role.
            user_prompt: Document content and task passed as the ``user`` role.

        Returns:
            The raw text content of the model's first response choice.

        Raises:
            RuntimeError: Wraps any OpenAIError so callers receive a consistent
                        exception type regardless of provider.
        """

        logger.debug("Sending completion request to OpenAI (model=%s).", self.model)

        try:
            response = await self.client.chat.completions.create(
                model = self. model,
                messages =[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature = 0.0,     # deterministic output for structured extraction
                response_format = {"type": "json_object"},  # ensure output is always JSON
            )

        except OpenAIError as e:
            logger.exception("OpenAI API request failed.")
            raise RuntimeError(f"OpenAI completion request failed: {e}") from e
        
        result = response.choices[0].message.content

        logger.debug("Received response from OpenAI (%d chars, finish_reason=%s).", len(result), response.choices[0].finish_reason,)

        return result