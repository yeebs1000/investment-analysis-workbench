"""Google Gemini provider — the app's single LLM. SDK imported lazily so the
app runs without google-genai configured. Uses the configured model
(settings.gemini_model), with a one-shot retry on a lighter model when the
primary hits a transient server-side error (e.g. 504 DEADLINE_EXCEEDED)."""
from __future__ import annotations

from app.config import settings
from app.llm.base import LLMProvider, LLMResult


class GeminiProvider(LLMProvider):
    name = "gemini"

    def __init__(self) -> None:
        self._client = None
        self.model_override: str | None = None   # runtime pick from the UI toggle

    def available(self) -> bool:
        return bool(settings.gemini_api_key)

    def model_id(self) -> str:
        return self.model_override or settings.gemini_model

    def _get_client(self):
        if self._client is None:
            from google import genai  # lazy import
            from google.genai import types

            # Bound every request so a slow/overloaded model can't hang the call;
            # on timeout we fall straight to the fallback model. With a lean payload
            # gemini-3.x answers in ~5-10s, so a 45s ceiling gives healthy headroom
            # while still failing fast to the stabler fallback when it's overloaded.
            self._client = genai.Client(
                api_key=settings.gemini_api_key,
                http_options=types.HttpOptions(timeout=45000),  # ms
            )
        return self._client

    def _config_for(self, model: str, system: str, max_tokens: int, temperature: float):
        """Build the per-model request config.

        Gemini 3.x are *thinking* models: reasoning tokens are drawn from the same
        output budget, so a small cap can leave NO visible answer. We cap thinking
        to 'low' and give a generous output budget so the brief always comes back.
        """
        from google.genai import types

        kwargs = dict(
            system_instruction=system,
            temperature=temperature,
            max_output_tokens=max(max_tokens, 2048),
        )
        if model.startswith("gemini-3"):
            # Thinking tokens draw from the output budget; give generous headroom
            # and cap reasoning to 'low' so a short answer always survives.
            kwargs["max_output_tokens"] = max(max_tokens, 4096)
            try:
                kwargs["thinking_config"] = types.ThinkingConfig(thinking_level="low")
            except Exception:  # noqa: BLE001 - older SDK without thinking_level
                kwargs.pop("thinking_config", None)
        try:
            return types.GenerateContentConfig(**kwargs)
        except Exception:  # noqa: BLE001 - drop unsupported keys and retry
            kwargs.pop("thinking_config", None)
            return types.GenerateContentConfig(**kwargs)

    # Thinking models occasionally exceed Gemini's server-side deadline under
    # load (surfaces as "504 DEADLINE_EXCEEDED"). This lighter model reasons
    # less and answers faster, so it's the retry target on a transient failure.
    _FALLBACK_MODEL = "gemini-3.1-flash-lite"

    @staticmethod
    def _is_transient(exc: Exception) -> bool:
        s = str(exc).upper()
        return any(t in s for t in (
            "DEADLINE_EXCEEDED", "UNAVAILABLE", "504", "503", "OVERLOADED",
            "RESOURCE_EXHAUSTED", "TIMEOUT", "TIMED OUT",
        ))

    def _call(self, model: str, system: str, user: str, max_tokens: int, temperature: float) -> LLMResult:
        client = self._get_client()
        cfg = self._config_for(model, system, max_tokens, temperature)
        resp = client.models.generate_content(model=model, contents=user, config=cfg)
        text = (resp.text or "").strip()
        if not text:
            raise RuntimeError(f"{model} returned empty text")
        usage = getattr(resp, "usage_metadata", None)
        return LLMResult(
            text=text,
            input_tokens=getattr(usage, "prompt_token_count", 0) or 0,
            output_tokens=getattr(usage, "candidates_token_count", 0) or 0,
            provider=self.name,
            model=model,
        )

    def generate(
        self, system: str, user: str, max_tokens: int = 700, temperature: float = 0.3
    ) -> LLMResult:
        model = self.model_id()
        try:
            return self._call(model, system, user, max_tokens, temperature)
        except Exception as exc:  # noqa: BLE001
            # On a transient server-side failure (e.g. the DEADLINE_EXCEEDED the
            # options Ask box was hitting), retry once on the lighter model
            # rather than surfacing the raw 504. Non-transient errors (bad key,
            # blocked content) propagate unchanged.
            if self._is_transient(exc) and model != self._FALLBACK_MODEL:
                return self._call(self._FALLBACK_MODEL, system, user, max_tokens, temperature)
            raise
