"""Application configuration loaded from environment / `.env`.

All settings have safe defaults so the app can at least start without a `.env`.
Secrets (API keys) default to empty strings; the LLM router degrades to
deterministic-only mode when a provider's key is missing.
"""
from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- Moomoo OpenD gateway ---
    opend_host: str = "127.0.0.1"
    opend_port: int = 11111
    trd_env: str = "REAL"            # REAL | SIMULATE
    trd_market: str = "US"           # US | HK | CN
    security_firm: str = "FUTUINC"   # FUTUINC (Moomoo US) | FUTUSECURITIES (Futu HK) | FUTUAU | FUTUSG

    # --- IBKR (Interactive Brokers) via TWS / IB Gateway, read-only ---
    ibkr_enabled: bool = False       # set true to merge IBKR holdings into the book
    ibkr_host: str = "127.0.0.1"
    ibkr_port: int = 4001            # IB Gateway live 4001 / paper 4002; TWS live 7496 / paper 7497
    ibkr_client_id: int = 1107       # any unused client id for the API connection

    # --- Tiger Brokers via Tiger Open API, read-only ---
    tiger_enabled: bool = False      # set true to merge Tiger holdings into the book
    tiger_id: str = ""               # your Tiger developer/API id
    tiger_account: str = ""          # the trading account number to read
    tiger_private_key_path: str = "" # path to your RSA private key file (never committed)

    # --- External market data (Finnhub: symbol search + analyst recommendations) ---
    finnhub_api_key: str = ""

    # --- Macro data (FRED: yield curve, credit spreads, VIX) — optional ---
    # Free key from https://fred.stlouisfed.org/docs/api/api_key.html; absent
    # -> the macro regime panel simply doesn't populate (degrades cleanly).
    fred_api_key: str = ""

    # --- LLM (single model each; no backdated/fallback variants) ---
    # Gemini is the default. Claude is optional — it only becomes selectable once
    # ANTHROPIC_API_KEY is set; until then the toggle shows it locked.
    gemini_api_key: str = ""
    anthropic_api_key: str = ""
    default_llm: str = "gemini"      # gemini | claude | none
    gemini_model: str = "gemini-2.5-flash"
    claude_model: str = "claude-opus-4-8"

    # --- API server ---
    api_host: str = "127.0.0.1"
    api_port: int = 8010


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
