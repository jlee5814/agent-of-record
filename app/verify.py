"""Identity matching, representative checks, and consent polling. No model involved.

A caller is verified when at least three of five fields (name, date of birth,
phone, email, ID last four) match one account holder and no field given
contradicts the record. The account number helps find the record but doesn't
count. Results never say which field failed, so no one can probe fields one
at a time.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Optional

from .data import (Person, Store, names_match, norm_account, norm_dob, norm_email,
                   norm_last4, norm_name, norm_phone)
from .state import IDENTITY_FIELDS, Identity


@dataclass
class FieldCheck:
    matched: list[str] = field(default_factory=list)
    conflicted: list[str] = field(default_factory=list)


@dataclass
class VerifyResult:
    status: str                         # verified | need_more | mismatch
    party_id: Optional[str] = None
    provided: int = 0
    needed: int = 0
    # Internal audit detail. Never shown to the caller or the response model.
    detail: dict = field(default_factory=dict)


def check_fields(identity: Identity, rec: Person) -> FieldCheck:
    fc = FieldCheck()

    def mark(name: str, ok: bool) -> None:
        (fc.matched if ok else fc.conflicted).append(name)

    if identity.full_name:
        mark("full_name", names_match(identity.full_name, rec))
    if identity.dob:
        mark("dob", norm_dob(identity.dob) == rec.dob)
    if identity.phone:
        p = norm_phone(identity.phone)
        mark("phone", p is not None and p in {norm_phone(x) for x in [rec.phone, *rec.phone_aliases]})
    if identity.email:
        e = norm_email(identity.email)
        mark("email", e is not None and e in {norm_email(x) for x in [rec.email, *rec.email_aliases]})
    if identity.id_last4:
        mark("id_last4", norm_last4(identity.id_last4) == rec.id_last4)
    return fc


def verify(identity: Identity, store: Store, min_matches: int = 3) -> VerifyResult:
    provided = identity.provided()
    n = len(provided)
    acct = norm_account(identity.account_number) if identity.account_number else None

    pool = store.people
    if acct:
        narrowed = [p for p in pool if norm_account(p.account_number) == acct]
        # An account number that matches nobody does not block; PII still decides.
        pool = narrowed or pool

    scored = []
    for rec in pool:
        fc = check_fields(identity, rec)
        scored.append((len(fc.matched), -len(fc.conflicted), rec, fc))
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)

    if n < min_matches:
        best = scored[0] if scored else None
        return VerifyResult(status="need_more", provided=n, needed=min_matches - n,
                            detail={"best_party": best[2].party_id if best else None,
                                    "matched": best[3].matched if best else []})

    winners = [s for s in scored if s[0] >= min_matches and s[1] == 0]
    if len(winners) == 1:
        _, _, rec, fc = winners[0]
        return VerifyResult(status="verified", party_id=rec.party_id, provided=n,
                            detail={"matched": fc.matched})

    best = scored[0] if scored else None
    return VerifyResult(status="mismatch", provided=n, needed=0,
                        detail={"best_party": best[2].party_id if best else None,
                                "matched": best[3].matched if best else [],
                                "conflicted": best[3].conflicted if best else [],
                                "ambiguous": len(winners) > 1})


class FailureRegistry:
    """Count failed attempts per account holder across all sessions.

    Without this, an attacker could open a new chat after three misses. Too many
    recent failures lock the account holder out of self service, even with correct
    details, and send the caller to a person.
    """

    def __init__(self, max_failures: int = 5, window_s: float = 1800.0):
        self.max_failures = max_failures
        self.window_s = window_s
        self._fails: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def _recent(self, party_id: str, now: float) -> list[float]:
        return [t for t in self._fails.get(party_id, []) if now - t < self.window_s]

    def record(self, party_id: Optional[str]) -> None:
        if not party_id:
            return
        now = time.time()
        with self._lock:
            self._fails[party_id] = self._recent(party_id, now) + [now]

    def locked(self, party_id: Optional[str]) -> bool:
        if not party_id:
            return False
        with self._lock:
            return len(self._recent(party_id, time.time())) >= self.max_failures


def rep_authorized(rep_name: Optional[str], party_id: str, store: Store) -> bool:
    """True if the caller is on file as an authorized contact for this account holder."""
    if not rep_name:
        return False
    target = norm_name(rep_name)
    return any(c.party_id == party_id and norm_name(c.name) == target for c in store.contacts)


def poll_consent(scenario: str, polls_so_far: int, store: Store) -> tuple[str, int]:
    """Poll the simulated consent service once; return (status, polls).

    Each poll reads the next status in the scenario. A sequence that ends without
    approval times out.
    """
    seq = store.consent_scenarios.get(scenario) or store.consent_scenarios["default"]
    if polls_so_far >= len(seq):
        return ("approved" if "approved" in seq else "timed_out"), polls_so_far
    status = seq[polls_so_far]
    polls = polls_so_far + 1
    if status != "approved" and polls >= len(seq):
        return "timed_out", polls
    return status, polls


__all__ = ["verify", "VerifyResult", "rep_authorized", "poll_consent", "IDENTITY_FIELDS"]
