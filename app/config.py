"""Settings, read from environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


@dataclass(frozen=True)
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(_env("DATA_DIR", str(ROOT / "data"))))
    # The demo clock. Every response deadline in the fixtures is still open on this date.
    # DEMO_TODAY=real uses the wall clock, and the deadlines have then passed.
    demo_today: str = field(default_factory=lambda: _env("DEMO_TODAY", "2026-03-10"))
    min_pii_matches: int = field(default_factory=lambda: int(_env("MIN_PII_MATCHES", "3")))
    max_verify_attempts: int = field(default_factory=lambda: int(_env("MAX_VERIFY_ATTEMPTS", "3")))
    max_off_topic: int = field(default_factory=lambda: int(_env("MAX_OFF_TOPIC", "3")))
    max_frustrated_turns: int = field(default_factory=lambda: int(_env("MAX_FRUSTRATED_TURNS", "3")))
    # Lockout across sessions: failed attempts against one cardholder within the window.
    max_party_failures: int = field(default_factory=lambda: int(_env("MAX_PARTY_FAILURES", "5")))
    lockout_window_min: int = field(default_factory=lambda: int(_env("LOCKOUT_WINDOW_MIN", "30")))

    llm_provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "sglang"))
    base_url: str = field(default_factory=lambda: _env("SGLANG_BASE_URL", "http://localhost:30000/v1"))
    # auto asks the server which model it serves.
    model: str = field(default_factory=lambda: _env("MODEL", "auto"))
    # json_schema, json_object, or prompt. Empty means try them in that order.
    extract_mode: str = field(default_factory=lambda: _env("EXTRACT_MODE", ""))

    def today(self) -> date:
        if self.demo_today.lower() == "real":
            return date.today()
        return date.fromisoformat(self.demo_today)


settings = Settings()
