"""Dispute lookup, facts from the record, and document guidance.

Everything here returns plain data. The reply model words it but can't add to it.
"""
from __future__ import annotations

import calendar
import re
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import Optional

from .data import Dispute, merchant_match, money, normalize_dispute_id, parse_date
from .state import IntentMemory

# What a caller means by open, credited, or denied, in record terms.
STATUS_GROUPS = {
    "open": {"needs_info", "under_review"},
    "credited": {"resolved_credited"},
    "denied": {"resolved_denied"},
}
STATUS_WORDS = {
    "needs_info": "waiting on documents from the cardholder",
    "under_review": "under review",
    "resolved_credited": "resolved in the cardholder's favor",
    "resolved_denied": "resolved in the merchant's favor",
}
REASON_WORDS = {
    "item_not_received": "item not received",
    "service_not_provided": "service not provided",
    "duplicate_charge": "duplicate charge",
    "wrong_amount": "wrong amount charged",
    "fraud": "unauthorized charge",
}


# ------------------------------------------------------------------ resolution

@dataclass
class Resolution:
    status: str                    # selected | ambiguous | none | no_disputes
    dispute_id: Optional[str] = None
    candidates: list[str] = field(default_factory=list)
    used_hints: dict = field(default_factory=dict)


def resolve_dispute(memory: IntentMemory, disputes: list[Dispute]) -> Resolution:
    """Filter the caller's disputes by every hint they gave.

    One match is selected; several mean the agent asks. If the hints match
    nothing, every dispute is listed so the caller can choose.
    """
    if not disputes:
        return Resolution(status="no_disputes")

    hints = {k: v for k, v in {
        "dispute_id": normalize_dispute_id(memory.dispute_id) if memory.dispute_id else None,
        "merchant": memory.merchant, "reason": memory.reason, "status": memory.status,
        "month": memory.month, "year": memory.year,
    }.items() if v}

    if "dispute_id" in hints:
        hit = [d for d in disputes if d.dispute_id == hints["dispute_id"]]
        if hit:
            return Resolution(status="selected", dispute_id=hit[0].dispute_id,
                              used_hints={"dispute_id": hints["dispute_id"]})
        hints.pop("dispute_id")  # an ID that isn't this caller's; fall through to other hints

    def keep(d: Dispute) -> bool:
        t = parse_date(d.transaction_date)
        if "merchant" in hints and not merchant_match(hints["merchant"], d.merchant):
            return False
        if "reason" in hints and d.reason != hints["reason"]:
            return False
        if "status" in hints and d.status not in STATUS_GROUPS.get(hints["status"], set()):
            return False
        if "month" in hints and t.month != hints["month"]:
            return False
        if "year" in hints and t.year != hints["year"]:
            return False
        return True

    if not hints:
        if len(disputes) == 1:
            return Resolution(status="selected", dispute_id=disputes[0].dispute_id)
        return Resolution(status="ambiguous", candidates=[d.dispute_id for d in disputes])

    survivors = [d for d in disputes if keep(d)]
    if len(survivors) == 1:
        return Resolution(status="selected", dispute_id=survivors[0].dispute_id, used_hints=hints)
    if len(survivors) > 1:
        return Resolution(status="ambiguous", candidates=[d.dispute_id for d in survivors], used_hints=hints)
    return Resolution(status="none", candidates=[d.dispute_id for d in disputes], used_hints=hints)


def _usd(x: Decimal) -> str:
    return f"${x:,.2f}"


def _long_date(s: str) -> str:
    d = parse_date(s)
    return f"{calendar.month_name[d.month]} {d.day}, {d.year}"


def dispute_label(d: Dispute) -> str:
    """Short label naming a dispute, for use after verification."""
    return (f"{d.dispute_id}: {d.merchant}, {_usd(money(d.disputed_amount))} charge on "
            f"{_long_date(d.transaction_date)}, {STATUS_WORDS[d.status]}")


# --------------------------------------------------------------- dispute facts

def dispute_facts(d: Dispute, today: date) -> dict:
    """Turn a dispute's fields into plain sentences the model can't misread.

    The disputed amount is what is contested, not a refund. A provisional credit
    is temporary. Only a final credit stays.
    """
    disputed, provisional, final = money(d.disputed_amount), money(d.provisional_credit), money(d.final_credit)
    facts: list[str] = [
        f"Dispute {d.dispute_id} is about a {_usd(disputed)} charge from {d.merchant} on {d.transaction_date} "
        f"({REASON_WORDS.get(d.reason, d.reason)}).",
        f"Current status: {STATUS_WORDS[d.status]}.",
    ]
    if d.status == "needs_info":
        facts.append("The review is paused until the cardholder sends the documents listed.")
    if d.status in STATUS_GROUPS["open"]:
        if provisional > 0:
            facts.append(f"A provisional credit of {_usd(provisional)} is on the account. It is temporary and the bank "
                         "can reverse it if the dispute is decided for the merchant. It is not a refund.")
        else:
            facts.append(f"No credit has been issued. The disputed amount, {_usd(disputed)}, is what is being "
                         "contested, not an amount the bank has agreed to return.")
    elif d.status == "resolved_credited":
        facts.append(f"The dispute was decided in the cardholder's favor ({d.decision_reason}). "
                     f"A final credit of {_usd(final)} stays on the account.")
    elif d.status == "resolved_denied":
        facts.append(f"The dispute was decided for the merchant because {d.decision_reason}. No credit was issued.")

    deadline = None
    if d.response_deadline:
        days = (parse_date(d.response_deadline) - today).days
        deadline = {"date": d.response_deadline, "days_left": days, "expired": days < 0}
        if days >= 0:
            facts.append(f"Documents are due by {d.response_deadline} ({days} days from today, {today.isoformat()}).")
        else:
            facts.append(f"The document deadline was {d.response_deadline}, which has passed. A late submission needs "
                         "a dispute specialist; do not promise the dispute will reopen.")

    return {
        "dispute_id": d.dispute_id,
        "merchant": d.merchant,
        "reason": d.reason,
        "status": d.status,
        "transaction_date": d.transaction_date,
        "summary": d.summary,
        "decision_reason": d.decision_reason,
        "documents_needed": list(d.documents_needed),
        "deadline": deadline,
        "amounts": {"disputed": _usd(disputed), "provisional_credit": _usd(provisional), "final_credit": _usd(final)},
        "facts": facts,
    }


# ------------------------------------------------------------ document guidance

def _tokens(s: str) -> set[str]:
    return set(re.findall(r"[a-z]+", s.lower()))


def map_document(name: str, guidance: dict) -> Optional[str]:
    """Find the guidance entry for a dispute's document name.

    Disputes say proof of order; the guidance says proof of order or delivery.
    Try an exact match, then the shortest entry containing every word of the name.
    """
    keys = list(guidance.get("document_guidance", {}).keys())
    n = name.strip().lower()
    if n in keys:
        return n
    nt = _tokens(n)
    hits = [k for k in keys if nt and nt <= _tokens(k)]
    return min(hits, key=len) if hits else None


def document_guidance(d: Dispute, guidance: dict) -> dict:
    alts = guidance["document_alternatives"]
    docs = []
    for name in d.documents_needed:
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
        "for_reason": guidance.get("reason_guidance", {}).get(d.reason),
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


def followup_answer(d: Dispute, guidance: dict, utterance: str = "", topic: Optional[str] = None) -> dict:
    """Pick a follow up rule: by topic, then by phrase, else the fallback.

    Rules about documents are skipped when a dispute needs none, so the agent
    never asks for documents that aren't required.
    """
    eligible = [r for r in guidance.get("followup_rules", [])
                if d.documents_needed or not r.get("requires_documents")]
    u = utterance.lower()

    chosen = None
    if topic:
        chosen = next((r for r in eligible if r["topic"] == topic), None)
    if chosen is None and u:
        # Longest matching phrase wins, so how long does it take beats how long do i have.
        best = (0, None)
        for r in eligible:
            for p in r.get("match_any", []):
                if p in u and len(p) > best[0]:
                    best = (len(p), r)
        chosen = best[1]

    if chosen is None:
        return {"topic": "fallback", "text": guidance["followup_fallback"]}
    text = chosen["text"].format(dispute_id=d.dispute_id, documents=_join(d.documents_needed),
                                 processing_time=guidance["followup_settings"]["processing_time"])
    return {"topic": chosen["topic"], "text": text}


FOLLOWUP_TOPICS = ["missing_document_alternatives", "submission_timing", "processing_time",
                   "submission_method", "file_format", "receipt_confirmation"]
