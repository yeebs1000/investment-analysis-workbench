"""LLM narration layer. The LLM only interprets pre-computed deterministic
numbers (it never produces indicator values); a router toggles providers and
caches responses so unchanged inputs cost nothing."""
