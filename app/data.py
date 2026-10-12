"""Load the fixtures and normalize identity fields.

The model never reads these files; code passes it only what a phase allows.
"""
from __future__ import annotations

import json
import re
from datetime import date, datetime
from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from .config import settings


class Cardholder(BaseModel):
    party_id: str
    name: str
    name_aliases: list[str] = Field(default_factory=list)
    account_number: str
    dob: str
    card_last4: str
    phone: str
    phone_aliases: list[str] = Field(default_factory=list)
    email: str
    email_aliases: list[str] = Field(default_factory=list)


class Dispute(BaseModel):
    dispute_id: str
    party_id: str
    merchant: str
    reason: str
    transaction_date: str
    opened_at: str
    status: str               # needs_info | under_review | resolved_credited | resolved_denied
    disputed_amount: str
    provisional_credit: str
    final_credit: str
    summary: str
    decision_reason: Optional[str] = None
    documents_needed: list[str] = Field(default_factory=list)
    response_deadline: Optional[str] = None


class AuthorizedUser(BaseModel):
    name: str
    relationship: str
    cardholder_name: str
    cardholder_party_id: str


class Store(BaseModel):
    cardholders: list[Cardholder]
    disputes: list[Dispute]
    authorized_users: list[AuthorizedUser]
    consent_scenarios: dict[str, list[str]]
    guidance: dict
    dispute_schema: dict

    def cardholder(self, party_id: str) -> Optional[Cardholder]:
        return next((p for p in self.cardholders if p.party_id == party_id), None)

    def disputes_for(self, party_id: str) -> list[Dispute]:
        return [d for d in self.disputes if d.party_id == party_id]

    def dispute(self, dispute_id: str) -> Optional[Dispute]:
        did = normalize_dispute_id(dispute_id)
        return next((d for d in self.disputes if d.dispute_id == did), None)


def _load(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=4)
def load_store(data_dir: Optional[str] = None) -> Store:
    d = Path(data_dir) if data_dir else settings.data_dir
    consent = {k: v["status_sequence"] for k, v in _load(d / "consent_scenarios.json").items()}
    return Store(
        cardholders=[Cardholder(**p) for p in _load(d / "cardholders.json")],
        disputes=[Dispute(**x) for x in _load(d / "disputes.json")],
        authorized_users=[AuthorizedUser(**a) for a in _load(d / "authorized_users.json")],
        consent_scenarios=consent,
        guidance=_load(d / "guidance.json"),
        dispute_schema=_load(d / "dispute_schema.json"),
    )


# ---------------------------------------------------------------- normalization

def norm_name(s: str) -> str:
    """Lowercase, drop punctuation, and sort words so order doesn't matter."""
    tokens = re.sub(r"[^a-z\s]", " ", s.lower()).split()
    return " ".join(sorted(tokens))


def norm_name_compact(s: str) -> str:
    """Remove spaces so Lena Hoffmann matches LenaHoffmann."""
    return re.sub(r"[^a-z]", "", s.lower())


def names_match(given: str, record: Cardholder) -> bool:
    if not given:
        return False
    g_sorted, g_compact = norm_name(given), norm_name_compact(given)
    return any(g_sorted == norm_name(c) or g_compact == norm_name_compact(c)
               for c in [record.name, *record.name_aliases])


def same_person(given: str, known_names: list[str]) -> bool:
    """True if a spoken name could be one of the known names.

    A partial name (Grace, Mr. Okafor) counts as the same person. Only a word
    outside every known name means someone new.
    """
    if not given:
        return True
    g = set(norm_name(given).split()) - {"mr", "mrs", "ms", "miss", "dr"}
    if not g:
        return True
    for k in known_names:
        if not k:
            continue
        if g <= set(norm_name(k).split()) or norm_name_compact(given) == norm_name_compact(k):
            return True
    return False


_DOB_FORMATS = ["%Y-%m-%d", "%m/%d/%Y", "%m-%d-%Y", "%B %d, %Y", "%B %d %Y", "%b %d, %Y",
                "%b %d %Y", "%d %B %Y", "%d %b %Y", "%Y/%m/%d", "%m/%d/%y"]


def norm_dob(s: str) -> Optional[str]:
    if not s:
        return None
    s = re.sub(r"(\d)(st|nd|rd|th)\b", r"\1", s.strip(), flags=re.I)
    s = re.sub(r"\s+", " ", s)
    for fmt in _DOB_FORMATS:
        try:
            return datetime.strptime(s, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def norm_phone(s: str) -> Optional[str]:
    if not s:
        return None
    digits = re.sub(r"\D", "", s)
    if len(digits) == 10:
        digits = "1" + digits
    return digits if len(digits) == 11 else None


def norm_email(s: str) -> Optional[str]:
    if not s:
        return None
    s = s.strip().lower().replace(" at ", "@").replace(" dot ", ".")
    s = re.sub(r"\s+", "", s)
    return s if re.fullmatch(r"[^@]+@[^@]+\.[a-z]{2,}", s) else None


def norm_last4(s: str) -> Optional[str]:
    if not s:
        return None
    digits = re.sub(r"\D", "", s)
    return digits[-4:] if len(digits) >= 4 else None


def norm_account(s: str) -> Optional[str]:
    if not s:
        return None
    m = re.search(r"(\d{4,})", s)
    return f"ACC-{m.group(1)}" if m else None


def normalize_dispute_id(s: str) -> str:
    m = re.search(r"(\d{3,})", s or "")
    return f"DSP-{m.group(1)}" if m else (s or "").upper()


def merchant_tokens(s: str) -> set[str]:
    stop = {"the", "inc", "llc", "co", "store", "shop", "a", "an", "my", "at", "from", "charge", "dispute"}
    return set(re.findall(r"[a-z]+", (s or "").lower())) - stop


def merchant_match(hint: str, merchant: str) -> bool:
    """A spoken merchant matches when any meaningful word lines up: bluewater finds Bluewater Electronics."""
    h = merchant_tokens(hint)
    return bool(h) and bool(h & merchant_tokens(merchant))


def money(s: str) -> Decimal:
    return Decimal(s)


def parse_date(s: str) -> date:
    return date.fromisoformat(s)
