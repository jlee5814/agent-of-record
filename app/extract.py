"""Step 1: turn the caller's words into structured fields.

The extractor sees only the conversation, never the records, so it can't leak
one. Its output is a proposal; the controller decides.
"""
from __future__ import annotations

import json
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

from .disputes import FOLLOWUP_TOPICS
from .state import SessionState

Emotion = Literal["neutral", "frustrated", "angry", "anxious", "confused", "refusing", "positive"]
Intent = Literal["decision_question", "status_inquiry", "document_submission", "next_steps", "credit_question",
                 "general_dispute_question", "other"]
Reason = Literal["item_not_received", "service_not_provided", "duplicate_charge", "wrong_amount", "fraud"]


class TurnExtraction(BaseModel):
    full_name: Optional[str] = None
    dob: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    card_last4: Optional[str] = None
    account_number: Optional[str] = None

    caller_role: Optional[Literal["cardholder", "representative"]] = None
    rep_name: Optional[str] = None
    relationship: Optional[str] = None

    dispute_id: Optional[str] = None
    merchant: Optional[str] = None
    reason: Optional[Reason] = None
    dispute_status: Optional[Literal["open", "credited", "denied"]] = None
    month: Optional[int] = Field(default=None, ge=1, le=12)
    year: Optional[int] = None
    intent: Optional[Intent] = None
    followup_topic: Optional[str] = None
    question: Optional[str] = None

    scope: Literal["in_scope", "out_of_scope", "smalltalk"] = "in_scope"
    emotion: Emotion = "neutral"
    wants_human: bool = False
    accepts_human: bool = False
    declines_human: bool = False
    refuses_verification: bool = False
    email_choice: Optional[Literal["send", "skip"]] = None
    done: bool = False

    @field_validator("followup_topic")
    @classmethod
    def _topic(cls, v):
        return v if v in FOLLOWUP_TOPICS else None

    @field_validator("*", mode="before")
    @classmethod
    def _blank_to_none(cls, v):
        if isinstance(v, str) and v.strip().lower() in {"", "null", "none", "unknown", "n/a"}:
            return None
        return v


def json_schema() -> dict:
    s = TurnExtraction.model_json_schema()
    s.pop("title", None)
    return s


SYSTEM = f"""You extract structured fields from one caller turn in a bank card dispute support chat.

Return values only for things the caller said in their LATEST message, using earlier turns only to
interpret short answers (for example a bare date after the agent asked for date of birth).
Leave a field out when the caller did not give it. Never guess or fill in values.

Identity
- full_name: the cardholder's full name as spoken. If the caller is someone else calling for the
  cardholder, this is the cardholder's name, and rep_name is the caller's own name.
- dob: ISO format YYYY-MM-DD when the date is unambiguous; otherwise copy what they said.
- phone, email: as given. A spelled out email (jane dot doe at mail dot com) becomes jane.doe@mail.com.
- card_last4: the last four digits of the card.
- account_number: for example ACC-30417.
- caller_role: representative when the caller says they call for someone else (my husband, my mother).

Dispute hints (capture them even during identity verification)
- dispute_id like DSP-4102; merchant as named (Bluewater, the parking authority); month and year of
  the charge.
- reason, only when clear: item_not_received (never arrived, didn't show up), service_not_provided
  (cancelled flight, class never happened), duplicate_charge (charged twice), wrong_amount (charged
  more than agreed), fraud (not my charge, didn't make it).
- dispute_status, only when the caller's word clearly means one of these:
  open: open, pending, in progress, under review, still waiting
  credited: won, credited, refunded, approved
  denied: denied, rejected, lost, declined, turned down
  Any other word (cancelled, closed, on hold) leaves dispute_status out, so the agent asks.
- intent: what they want. decision_question when they ask why a dispute was decided as it was.
  credit_question when they ask about money back or a credit.
- followup_topic, only for questions about sending documents: one of {FOLLOWUP_TOPICS}.
- question: the caller's question restated in a few plain words.

Conversation signals
- scope: out_of_scope for requests unrelated to their card, disputes, or this chat (trivia, coding,
  general knowledge). smalltalk for greetings and thanks. Everything about their identity, account,
  disputes, documents, credits, the email summary, or talking to a person is in_scope.
- emotion: the caller's tone this turn. refusing when they decline to give requested information.
- wants_human: they ask for a person. accepts_human or declines_human: their answer when the agent
  just offered a transfer.
- refuses_verification: they decline or object to giving identity details.
- email_choice: send or skip, only when answering the agent's offer of an email summary.
- done: they say they have no more questions, including a plain no when the agent asked whether
  there is anything else.
"""


def build_user_message(state: SessionState, text: str, candidates: Optional[list[str]] = None) -> str:
    recent = state.recent(6)
    lines = [f"{m['role'].upper()}: {m['content']}" for m in recent]
    ctx = {"current_phase": state.phase.value}
    if candidates:
        # After verification only: lets the caller pick between their own disputes.
        ctx["disputes_the_agent_listed"] = candidates
    return (
        "Context: " + json.dumps(ctx) + "\n\n"
        "Recent conversation:\n" + ("\n".join(lines) if lines else "(start of chat)") + "\n\n"
        f"LATEST CALLER MESSAGE:\n{text}"
    )


def parse(raw: dict) -> TurnExtraction:
    clean = {k: v for k, v in (raw or {}).items() if k in TurnExtraction.model_fields}
    try:
        return TurnExtraction(**clean)
    except Exception:
        # Drop fields one at a time until it validates; a bad field never kills the turn.
        ok = {}
        for k, v in clean.items():
            try:
                TurnExtraction(**{**ok, k: v})
                ok[k] = v
            except Exception:
                continue
        return TurnExtraction(**ok)
