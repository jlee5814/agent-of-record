"""Session state. The controller writes it; the model only proposes."""
from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class Phase(str, Enum):
    VERIFY_ID = "VERIFY_ID"
    RESOLVE_INTENT = "RESOLVE_INTENT"
    PROCESS_CASE = "PROCESS_CASE"
    POST_PROCESS = "POST_PROCESS"
    HUMAN_HANDOFF = "HUMAN_HANDOFF"
    ENDED = "ENDED"


IDENTITY_FIELDS = ("full_name", "dob", "phone", "email", "card_last4")


class Identity(BaseModel):
    """Values the caller has given. Never repeated back in full."""
    full_name: Optional[str] = None
    dob: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    card_last4: Optional[str] = None
    account_number: Optional[str] = None  # lookup key only; not one of the five PII fields

    def provided(self) -> list[str]:
        return [f for f in IDENTITY_FIELDS if getattr(self, f)]


HINT_FIELDS = ("dispute_id", "merchant", "reason", "status", "month", "year")


class IntentMemory(BaseModel):
    """Dispute hints saved whenever the caller gives them, even before verification."""
    dispute_id: Optional[str] = None
    merchant: Optional[str] = None
    reason: Optional[str] = None         # item_not_received | duplicate_charge | wrong_amount | ...
    status: Optional[str] = None         # open | credited | denied
    month: Optional[int] = None          # month of the charge, 1..12
    year: Optional[int] = None
    intent: Optional[str] = None         # decision_question | status_inquiry | document_submission | ...

    def any(self) -> bool:
        return any(getattr(self, f) for f in (*HINT_FIELDS, "intent"))

    def has_record_hint(self) -> bool:
        return any(getattr(self, f) for f in HINT_FIELDS)


class ConsentState(BaseModel):
    scenario: str = "default"
    status: str = "not_requested"        # not_requested | pending | approved | timed_out
    polls: int = 0


class Event(BaseModel):
    t: float = Field(default_factory=time.time)
    kind: str
    data: dict[str, Any] = Field(default_factory=dict)


class SessionState(BaseModel):
    session_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    phase: Phase = Phase.VERIFY_ID

    identity: Identity = Field(default_factory=Identity)
    identity_dirty: bool = False          # changed since the last verification attempt
    verify_attempts: int = 0
    verified_party_id: Optional[str] = None

    caller_role: str = "unknown"          # unknown | cardholder | representative
    rep_name: Optional[str] = None
    rep_relationship: Optional[str] = None
    rep_authorized: Optional[bool] = None
    consent: ConsentState = Field(default_factory=ConsentState)

    memory: IntentMemory = Field(default_factory=IntentMemory)
    candidate_ids: list[str] = Field(default_factory=list)
    selected_id: Optional[str] = None
    topics_discussed: list[str] = Field(default_factory=list)

    off_topic_strikes: int = 0
    frustration_streak: int = 0
    last_emotion: str = "neutral"
    handoff_reason: Optional[str] = None
    handoff_offered: bool = False

    email_decision: Optional[str] = None  # send | skip
    email_sent: bool = False

    transcript: list[dict[str, str]] = Field(default_factory=list)
    # Models see the transcript from here on. Moves forward when a new speaker takes over.
    context_start: int = 0
    events: list[Event] = Field(default_factory=list)

    def recent(self, n: int = 6) -> list[dict[str, str]]:
        return self.transcript[max(self.context_start, len(self.transcript) - n):]

    def log(self, kind: str, **data: Any) -> None:
        self.events.append(Event(kind=kind, data=data))

    @property
    def verified(self) -> bool:
        """The disclosure gate: identity matched, and for a representative, listed and consented."""
        if self.verified_party_id is None:
            return False
        if self.caller_role == "representative":
            return bool(self.rep_authorized) and self.consent.status == "approved"
        return True
