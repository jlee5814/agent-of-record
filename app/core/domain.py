"""What changes between domains, behind one interface.

The procedure, the gates, and the guard are the same for a driver and a
patient. A domain supplies the words, the record fields, how a record's fields
become plain sentences, and a few regexes for the offline extractor.
"""
from __future__ import annotations

import re
from datetime import date, timedelta
from functools import cached_property
from typing import Optional

from .data import Record, Store, load_store, name_tokens, parse_date
from .state import SessionState


class Domain:
    key = ""
    org = ""                       # Redline Freight driver support
    opening = ""
    record_noun = "record"         # load, visit
    id_prefix = ""                 # LD, APT
    person_noun = "account holder" # driver, patient
    id_last4_label = ""            # last four digits of your CDL number
    account_label = ""             # driver ID, patient number
    account_example = ""
    consent_channel = ""           # the Redline driver app
    team = "a person"              # who a handoff goes to
    privacy_reason = ""            # why verification comes first, in one clause
    categories: dict[str, str] = {}
    status_groups: dict[str, set[str]] = {}
    status_words: dict[str, str] = {}
    intents: dict[str, str] = {}
    live_intents: set[str] = set()
    summary_sections: list[str] = []
    money_rule = ""
    hint_notes = ""
    restricted_note = ""
    restricted_reply = ""
    emergency_note = ""
    emergency_reply = ""
    in_scope_words = ""
    examples: list[tuple[str, str]] = []
    # Offline extractor patterns.
    re_restricted = r"$^"
    re_emergency = r"$^"

    @cached_property
    def store(self) -> Store:
        return load_store(self.key, self.id_prefix)

    @cached_property
    def id_regex(self) -> re.Pattern:
        return re.compile(rf"\b{self.id_prefix}-?\s?\d{{3,}}\b", re.I)

    # ------------------------------------------------------------ hooks

    def label(self, r: Record) -> str:
        raise NotImplementedError

    def sections(self, r: Record, today: date, live: Optional[dict]) -> dict[str, list[str]]:
        """Record fields as plain sentences, grouped so templates can pick a section."""
        raise NotImplementedError

    def sections_for(self, intent: Optional[str], status: str) -> list[str]:
        """Which sections answer an intent, when no follow up rule applies."""
        return ["overview"]

    def offline_hints(self, t: str, tl: str, phase: str, today: date) -> dict:
        return {}

    def private_terms(self, r: Record) -> list[str]:
        """Strings that must not appear before verification unless the caller said them."""
        return [r.counterparty]

    # ------------------------------------------------------------ shared

    def live_entry(self, r: Record, s: SessionState) -> Optional[dict]:
        if not r.live:
            return None
        return r.live[min(s.live_polls.get(r.record_id, 0), len(r.live) - 1)]

    def facts(self, r: Record, today: date, live: Optional[dict]) -> dict:
        secs = self.sections(r, today, live)
        deadline = None
        if r.deadline:
            days = (parse_date(r.deadline) - today).days
            deadline = {"date": r.deadline, "days_left": days, "expired": days < 0}
            secs.setdefault("deadline", []).append(
                f"Paperwork is due by {r.deadline} ({days} days from today, {today.isoformat()})." if days >= 0 else
                f"The paperwork deadline was {r.deadline}, which has passed. A late submission needs {self.team}; "
                "do not promise it will be accepted.")
        return {
            "record_id": r.record_id,
            "status": (live or {}).get("status") or r.status,
            "counterparty": r.counterparty,
            "documents_needed": list(r.documents_needed),
            "deadline": deadline,
            "sections": secs,
            "facts": [f for v in secs.values() for f in v],
        }

    def answer(self, record_facts: dict, intent: Optional[str]) -> str:
        secs = record_facts.get("sections", {})
        out = []
        for name in self.sections_for(intent, record_facts.get("status", "")):
            out += secs.get(name, [])
        return " ".join(out) or " ".join(record_facts.get("facts", [])[:3])

    def party_hint(self, tl: str) -> Optional[str]:
        """Offline: the first facility or provider word the caller said."""
        words = set(re.findall(r"[a-z]+", tl))
        for r in self.store.records:
            for tok in sorted(name_tokens(f"{r.counterparty} {r.location}")):
                if len(tok) >= 3 and tok in words:
                    return tok
        return None

    @staticmethod
    def day_hint(tl: str, today: date) -> Optional[str]:
        if re.search(r"\btoday\b|\bthis (morning|afternoon)\b|\bright now\b", tl):
            return today.isoformat()
        if re.search(r"\byesterday\b", tl):
            return (today - timedelta(days=1)).isoformat()
        return None


_REGISTRY: dict[str, Domain] = {}


def get_domain(key: str) -> Domain:
    if key not in _REGISTRY:
        if key == "freight":
            from .domains.freight import Freight
            _REGISTRY[key] = Freight()
        elif key == "clinic":
            from .domains.clinic import Clinic
            _REGISTRY[key] = Clinic()
        elif key == "portfolio":
            from .domains.portfolio import Portfolio
            _REGISTRY[key] = Portfolio()
        else:
            raise ValueError(f"unknown domain {key!r}; use {', '.join(DOMAINS)}")
    return _REGISTRY[key]


DOMAINS = ("freight", "clinic", "portfolio")
