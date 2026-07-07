"""LLM router: provider toggle, response cache, and a session cost meter.

Cache key = hash(provider | model | system | user). Identical inputs return the
cached narrative with ZERO new spend, so re-opening the same unchanged analysis
never re-bills. The cost meter accumulates only on real (non-cached) calls.
"""
from __future__ import annotations

import hashlib
import threading

from app.config import settings
from app.llm.base import LLMResult
from app.llm.claude import ClaudeProvider
from app.llm.gemini import GeminiProvider

# Selectable models per provider — current generation only, no legacy/dated
# variants. Gemini list verified live against the models API on the user's key
# (2026-07-06); Claude list/pricing from Anthropic's current catalog.
MODEL_OPTIONS: dict[str, list[str]] = {
    "gemini": ["gemini-3.5-flash", "gemini-3.1-flash-lite"],
    "claude": ["claude-opus-4-8", "claude-sonnet-5", "claude-haiku-4-5", "claude-fable-5"],
}

# Approximate USD pricing per 1M tokens (input, output). Estimates — clearly
# labelled as such in the UI; adjust here if rates change.
PRICING: dict[str, tuple[float, float]] = {
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-3.5-flash": (0.50, 3.0),
    "gemini-3.1-flash-lite": (0.10, 0.40),
    "claude-fable-5": (10.0, 50.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-sonnet-5": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
}
_DEFAULT_PRICE = (1.0, 5.0)


def _est_cost(model: str, in_tok: int, out_tok: int) -> float:
    pin, pout = PRICING.get(model, _DEFAULT_PRICE)
    return in_tok / 1e6 * pin + out_tok / 1e6 * pout


class LLMRouter:
    def __init__(self) -> None:
        self._providers = {"gemini": GeminiProvider(), "claude": ClaudeProvider()}
        self._cache: dict[str, LLMResult] = {}
        self._lock = threading.Lock()
        self._usage: dict[str, dict] = {}

    # --- status --------------------------------------------------------
    def providers_status(self) -> dict:
        return {
            "default": settings.default_llm,
            "available": {name: p.available() for name, p in self._providers.items()},
            "models": {name: p.model_id() for name, p in self._providers.items()},
            "options": MODEL_OPTIONS,
        }

    def set_model(self, provider: str, model: str) -> dict:
        """Switch a provider's model at runtime (session-scoped; .env holds the
        startup default). Only current-generation models are accepted."""
        if provider not in self._providers:
            raise ValueError(f"Unknown provider '{provider}'.")
        if model not in MODEL_OPTIONS.get(provider, []):
            raise ValueError(f"'{model}' is not a selectable {provider} model.")
        self._providers[provider].model_override = model
        return self.providers_status()

    def resolve(self, requested: str | None) -> str:
        """Pick the effective provider: requested -> default -> 'none'."""
        choice = (requested or settings.default_llm or "none").lower()
        if choice in ("none", "deterministic"):
            return "none"
        if choice in self._providers and self._providers[choice].available():
            return choice
        return "none"

    # --- generation ----------------------------------------------------
    def narrate(
        self, provider_name: str, system: str, user: str, max_tokens: int = 700
    ) -> LLMResult | None:
        """Return a narrative, or None if no usable provider was selected."""
        name = self.resolve(provider_name)
        if name == "none":
            return None
        provider = self._providers[name]
        key = hashlib.sha256(
            f"{name}|{provider.model_id()}|{system}|{user}".encode()
        ).hexdigest()

        with self._lock:
            cached = self._cache.get(key)
        if cached is not None:
            return LLMResult(**{**cached.__dict__, "cached": True})

        result = provider.generate(system, user, max_tokens=max_tokens)
        with self._lock:
            self._cache[key] = result
            self._record(result)
        return result

    # --- cost meter ----------------------------------------------------
    def _record(self, r: LLMResult) -> None:
        u = self._usage.setdefault(
            r.provider,
            {"calls": 0, "input_tokens": 0, "output_tokens": 0, "est_cost_usd": 0.0},
        )
        u["calls"] += 1
        u["input_tokens"] += r.input_tokens
        u["output_tokens"] += r.output_tokens
        u["est_cost_usd"] += _est_cost(r.model, r.input_tokens, r.output_tokens)

    def usage(self) -> dict:
        with self._lock:
            by_provider = {k: dict(v) for k, v in self._usage.items()}
        total_cost = round(sum(v["est_cost_usd"] for v in by_provider.values()), 4)
        total_calls = sum(v["calls"] for v in by_provider.values())
        for v in by_provider.values():
            v["est_cost_usd"] = round(v["est_cost_usd"], 4)
        return {
            "by_provider": by_provider,
            "total_est_cost_usd": total_cost,
            "total_calls": total_calls,
            "cached_entries": len(self._cache),
            "note": "Costs are estimates from approximate per-token pricing.",
        }

    def reset_usage(self) -> None:
        with self._lock:
            self._usage.clear()


router = LLMRouter()
