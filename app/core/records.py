"""Record lookup and document guidance, shared by every domain.

Everything here returns plain data. The reply model words it but can't add to it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from .data import Record, fuzzy_match, normalize_id, parse_date
from .state import IntentMemory

if TYPE_CHECKING:
    from .domain import Domain

# Weakest first: the model is likeliest to invent these, and they identify a record least.
RELAX_ORDER = ("category", "status", "year", "month", "date")

FOLLOWUP_TOPICS = ["missing_document_alternatives", "submission_timing", "processing_time",
                   "submission_method", "file_format"]


@dataclass
class Resolution:
    status: str                    # selected | ambiguous | none | no_records
    record_id: Optional[str] = None
    candidates: list[str] = field(default_factory=list)
    used_hints: dict = field(default_factory=dict)


def resolve(memory: IntentMemory, records: list[Record], domain: "Domain") -> Resolution:
    """Filter the caller's records by every hint they gave.

    One match is selected; several mean the agent asks. If the hints match
    nothing, every record is listed so the caller can choose.
    """
    if not records:
        return Resolution(status="no_records")

    hints = {k: v for k, v in {
        "record_id": normalize_id(memory.record_id, domain.id_prefix) if memory.record_id else None,
        "party": memory.party, "category": memory.category, "status": memory.status,
        "date": memory.date, "month": memory.month, "year": memory.year,
    }.items() if v}

    if "record_id" in hints:
        hit = [r for r in records if r.record_id == hints["record_id"]]
        if hit:
            return Resolution(status="selected", record_id=hit[0].record_id, used_hints={"record_id": hints["record_id"]})
        hints.pop("record_id")  # an ID that isn't this caller's; fall through to other hints

    def keep(r: Record) -> bool:
        d = parse_date(r.date)
        if "party" in hints and not fuzzy_match(hints["party"], r.counterparty, r.location):
            return False
        if "category" in hints and r.category != hints["category"]:
            return False
        if "status" in hints and r.status not in domain.status_groups.get(hints["status"], set()):
            return False
        if "date" in hints and r.date != hints["date"]:
            return False
        if "month" in hints and d.month != hints["month"]:
            return False
        if "year" in hints and d.year != hints["year"]:
            return False
        return True

    if not hints:
        if len(records) == 1:
            return Resolution(status="selected", record_id=records[0].record_id)
        return Resolution(status="ambiguous", candidates=[r.record_id for r in records])

    survivors = [r for r in records if keep(r)]
    # A hint the model invented (a trailer type, a status) can exclude the right record.
    # When everything together matches nothing, drop the weakest hints one at a time.
    for weak in RELAX_ORDER:
        if survivors or len(hints) <= 1:
            break
        if weak in hints:
            hints.pop(weak)
            survivors = [r for r in records if keep(r)]
    if len(survivors) == 1:
        return Resolution(status="selected", record_id=survivors[0].record_id, used_hints=hints)
    if len(survivors) > 1:
        return Resolution(status="ambiguous", candidates=[r.record_id for r in survivors], used_hints=hints)
    return Resolution(status="none", candidates=[r.record_id for r in records], used_hints=hints)


# ------------------------------------------------------------ document guidance

def _tokens(s: str) -> set[str]:
    return set(re.findall(r"[a-z]+", s.lower()))


def map_document(name: str, guidance: dict) -> Optional[str]:
    """Find the guidance entry for a record's document name: exact, then the shortest entry holding every word."""
    keys = list(guidance.get("document_guidance", {}).keys())
    n = name.strip().lower()
    if n in keys:
        return n
    nt = _tokens(n)
    hits = [k for k in keys if nt and nt <= _tokens(k)]
    return min(hits, key=len) if hits else None


def document_guidance(r: Record, guidance: dict) -> dict:
    alts = guidance["document_alternatives"]
    docs = []
    for name in r.documents_needed:
        key = map_document(name, guidance)
        docs.append({
            "name": name,
            "guidance_key": key,
            "requirements": guidance["document_guidance"].get(key) if key else
            "No document specific rule on file; the general guidance applies.",
            "if_unavailable": alts.get(key) or alts["default"],
        })
    return {
        "general": guidance["default_guidance"],
        "for_category": guidance.get("category_guidance", {}).get(r.category),
        "documents": docs,
        "processing_time": guidance["followup_settings"]["processing_time"],
        "human_review_rule": guidance["followup_settings"]["human_review_rule"],
    }


def _join(items: list[str]) -> str:
    if not items:
        return "the requested documents"
    if len(items) == 1:
        return f"the {items[0]}"
    return "the " + ", the ".join(items[:-1]) + f" and the {items[-1]}"


def followup_answer(r: Record, guidance: dict, utterance: str = "", topic: Optional[str] = None) -> dict:
    """Pick a follow up rule: by topic, then by phrase, else the fallback.

    Rules about documents are skipped when a record needs none, so the agent
    never asks for documents that aren't required.
    """
    eligible = [x for x in guidance.get("followup_rules", [])
                if r.documents_needed or not x.get("requires_documents")]
    u = utterance.lower()
    chosen = None
    if topic:
        chosen = next((x for x in eligible if x["topic"] == topic), None)
    if chosen is None and u:
        best = (0, None)                 # longest matching phrase wins
        for x in eligible:
            for p in x.get("match_any", []):
                if p in u and len(p) > best[0]:
                    best = (len(p), x)
        chosen = best[1]
    if chosen is None:
        return {"topic": "fallback", "text": guidance["followup_fallback"]}
    text = chosen["text"].format(record_id=r.record_id, documents=_join(r.documents_needed),
                                 processing_time=guidance["followup_settings"]["processing_time"])
    return {"topic": chosen["topic"], "text": text}
