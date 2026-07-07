"""Anthropic Claude provider (optional, higher-quality reasoning). SDK imported
lazily so the app runs without `anthropic` installed. Only usable once
ANTHROPIC_API_KEY is set in .env; otherwise the toggle shows it locked."""
from __future__ import annotations

from app.config import settings
from app.llm.base import LLMProvider, LLMResult


class ClaudeProvider(LLMProvider):
    name = "claude"

    def __init__(self) -> None:
        self._client = None
        self.model_override: str | None = None   # runtime pick from the UI toggle

    def available(self) -> bool:
        return bool(settings.anthropic_api_key)

    def model_id(self) -> str:
        return self.model_override or settings.claude_model

    def _get_client(self):
        if self._client is None:
            import anthropic  # lazy import

            self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        return self._client

    def generate(
        self, system: str, user: str, max_tokens: int = 700, temperature: float = 0.3
    ) -> LLMResult:
        client = self._get_client()
        # No temperature: current Claude models (Opus 4.8 / Sonnet 5 / Fable 5)
        # reject sampling params with a 400 — steering is prompt-only now.
        msg = client.messages.create(
            model=self.model_id(),
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(
            block.text for block in msg.content if getattr(block, "type", "") == "text"
        ).strip()
        return LLMResult(
            text=text,
            input_tokens=msg.usage.input_tokens,
            output_tokens=msg.usage.output_tokens,
            provider=self.name,
            model=self.model_id(),
        )
