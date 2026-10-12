"""Load a domain's fixtures and normalize identity fields.

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

from pydantic import BaseModel, ConfigDict, Field

from .config import settings


class Person(BaseModel):
    """The account holder: a driver or a patient."""
    party_id: str
    name: str
    name_aliases: list[str] = Field(default_factory=list)
    account_number: str
    dob: str
    id_last4: str
    phone: str
    phone_aliases: list[str] = Field(default_factory=list)
    email: str
    email_aliases: list[str] = Field(default_factory=list)


class Contact(BaseModel):
    """Someone allowed to call for the account holder, with their consent."""
    name: str
    relationship: str
    party_name: str
    party_id: str


class Record(BaseModel):
    """A load or a visit. Fields every domain shares; the rest are domain specific."""
    model_config = ConfigDict(extra="allow")

    record_id: str
    party_id: str
    counterparty: str              # facility or provider
    location: str = ""
    category: str                  # equipment type or visit type
    date: str
    status: str
    summary: str = ""
    documents_needed: list[str] = Field(default_factory=list)
    deadline: Optional[str] = None
    live: list[dict] = Field(default_factory=list)   # status updates, read one per poll


class Store(BaseModel):
    people: list[Person]
    records: list[Record]
    contacts: list[Contact]
    consent_scenarios: dict[str, list[str]]
    guidance: dict
    id_prefix: str

    def person(self, party_id: str) -> Optional[Person]:
        return next((p for p in self.people if p.party_id == party_id), None)

    def records_for(self, party_id: str) -> list[Record]:
        return [r for r in self.records if r.party_id == party_id]

    def record(self, record_id: str) -> Optional[Record]:
        rid = normalize_id(record_id, self.id_prefix)
        return next((r for r in self.records if r.record_id == rid), None)


def _load(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=8)
def load_store(domain: str, id_prefix: str, data_dir: Optional[str] = None) -> Store:
    d = (Path(data_dir) if data_dir else settings.data_dir) / domain
    consent = {k: v["status_sequence"] for k, v in _load(d / "consent_scenarios.json").items()}
    return Store(
        people=[Person(**p) for p in _load(d / "people.json")],
        records=[Record(**r) for r in _load(d / "records.json")],
        contacts=[Contact(**c) for c in _load(d / "contacts.json")],
        consent_scenarios=consent,
        guidance=_load(d / "guidance.json"),
        id_prefix=id_prefix,
    )


# ---------------------------------------------------------------- normalization

def norm_name(s: str) -> str:
    """Lowercase, drop punctuation, and sort words so order doesn't matter."""
    tokens = re.sub(r"[^a-z\s]", " ", s.lower()).split()
    return " ".join(sorted(tokens))


def norm_name_compact(s: str) -> str:
    """Remove spaces so Hannah Kim matches HannahKim."""
    return re.sub(r"[^a-z]", "", s.lower())


def names_match(given: str, record: Person) -> bool:
    if not given:
        return False
    g_sorted, g_compact = norm_name(given), norm_name_compact(given)
    return any(g_sorted == norm_name(c) or g_compact == norm_name_compact(c)
               for c in [record.name, *record.name_aliases])


def same_person(given: str, known_names: list[str]) -> bool:
    """True if a spoken name could be one of the known names.

    A partial name (Maria, Mr. Alvarez) counts as the same person. Only a word
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
    """DRV-2201 and MRN-10482 both reduce to their digits; prefixes vary in speech."""
    if not s:
        return None
    m = re.search(r"(\d{4,})", s)
    return m.group(1) if m else None


def normalize_id(s: str, prefix: str) -> str:
    m = re.search(r"(\d{3,})", s or "")
    return f"{prefix}-{m.group(1)}" if m else (s or "").upper()


_STOP = {"the", "inc", "llc", "co", "dc", "a", "an", "my", "at", "from", "to", "in", "load", "visit", "dr",
         "doctor", "appointment", "with", "one", "ga", "nc", "al"}


def name_tokens(s: str) -> set[str]:
    return set(re.findall(r"[a-z]+", (s or "").lower())) - _STOP


def fuzzy_match(hint: str, *targets: str) -> bool:
    """A spoken name matches when a meaningful word lines up: peachtree finds Peachtree Foods DC."""
    h = name_tokens(hint)
    return bool(h) and any(h & name_tokens(t) for t in targets)


def money(s: str) -> Decimal:
    return Decimal(s)


def usd(s) -> str:
    return f"${Decimal(str(s)):,.2f}"


def parse_date(s: str) -> date:
    return date.fromisoformat(s)


def clock(hhmm: Optional[str]) -> str:
    """13:00 becomes 1:00 PM."""
    if not hhmm:
        return ""
    h, m = (int(x) for x in hhmm.split(":"))
    return f"{(h % 12) or 12}:{m:02d} {'AM' if h < 12 else 'PM'}"


def long_date(s: str) -> str:
    d = parse_date(s)
    return f"{d.strftime('%B')} {d.day}, {d.year}"
