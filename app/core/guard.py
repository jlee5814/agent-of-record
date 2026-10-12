"""Step 4: check a reply against the record before the caller sees it.

Before verification, nothing from a record may appear. After, every record ID,
dollar amount, clock time, and duration must be in the approved facts or in
what the caller said. Anything else is a break, and the reply is replaced.
"""
from __future__ import annotations

import json
import re
from decimal import Decimal, InvalidOperation
from typing import Optional

from .data import Store, norm_phone
from .domain import Domain
from .state import SessionState

_MONEY = re.compile(r"\$\s?\d[\d,]*(?:\.\d{2})?")
_CLOCK = re.compile(r"\b(\d{1,2}):(\d{2})\s*([ap])?\.?\s*m?\.?\b", re.I)
_DURATION = re.compile(r"\b(\d+(?:\.\d+)?)\s*(?:more\s+|extra\s+)?(minutes?|mins?|hours?|hrs?)\b", re.I)


def _norm_money(s: str) -> str:
    """Normalize an amount to cents so $45 and $45.00 match."""
    digits = re.sub(r"[^\d.]", "", s).rstrip(".")
    try:
        return str(Decimal(digits).quantize(Decimal("0.01")))
    except InvalidOperation:
        return digits


def _clocks(text: str) -> set[tuple[int, int]]:
    """2:30 PM, 2:30pm, and 14:30 all become (2, 30)."""
    return {(int(h) % 12, int(m)) for h, m, _ in _CLOCK.findall(text)}


def _minutes(text: str) -> set[float]:
    """Durations in minutes, so 2 hours and 120 minutes match."""
    out = set()
    for n, unit in _DURATION.findall(text):
        v = float(n)
        out.add(v * 60 if unit.lower().startswith("h") else v)
    return out


def guard(reply: str, state: SessionState, store: Store, domain: Domain, grounded: Optional[dict],
          user_texts: list[str]) -> list[str]:
    """List what a reply says that the phase doesn't allow. Empty means send it."""
    problems: list[str] = []
    said = " ".join(user_texts)
    said_low = said.lower()
    said_digits = re.sub(r"\D", "", said)
    low = reply.lower()

    if not state.verified:
        if domain.id_regex.search(reply):
            problems.append(f"{domain.record_noun} id before verification")
        if _MONEY.search(reply):
            problems.append("amount before verification")
        for r in store.records:
            for term in [*domain.private_terms(r), *r.documents_needed]:
                if term and term.lower() in low and term.lower() not in said_low:
                    problems.append(f"record detail before verification: {term}")
        for p in store.people:
            if p.email.lower() in low and p.email.lower() not in said_low:
                problems.append("record email echoed")
            ph = norm_phone(p.phone) or ""
            if ph and ph[-10:] in re.sub(r"\D", "", reply) and ph[-10:] not in said_digits:
                problems.append("record phone echoed")
            if p.dob in reply and p.dob not in said:
                problems.append("record dob echoed")
        return sorted(set(problems))

    # Verified: every ID, amount, clock time, and duration must be grounded or said by the caller.
    own = {r.record_id for r in store.records_for(state.verified_party_id or "")}
    for m in domain.id_regex.findall(reply):
        rid = f"{domain.id_prefix}-" + re.sub(r"\D", "", m)
        if rid not in own:
            problems.append(f"ungrounded {domain.record_noun} id {rid}")
    blob = json.dumps(grounded or {}) + " " + said
    allowed_money = {_norm_money(m) for m in _MONEY.findall(blob)}
    for m in _MONEY.findall(reply):
        if _norm_money(m) not in allowed_money:
            problems.append(f"ungrounded amount {m}")
    allowed_clocks, allowed_minutes = _clocks(blob), _minutes(blob)
    for c in _clocks(reply) - allowed_clocks:
        problems.append(f"ungrounded time {c[0] or 12}:{c[1]:02d}")
    for v in _minutes(reply) - allowed_minutes:
        problems.append(f"ungrounded duration {v:g} minutes")
    return sorted(set(problems))
