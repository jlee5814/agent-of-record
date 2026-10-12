"""Step 1: turn the caller's words into structured fields.

The extractor sees only the conversation, never the records, so it can't leak
one. Its output is a proposal; the controller decides. The schema is the same
shape for every domain; only the allowed values and the descriptions change.
"""
from __future__ import annotations

import json
from datetime import date
from functools import lru_cache
from typing import Literal, Optional

from pydantic import BaseModel, Field, create_model, field_validator

from .domain import Domain
from .records import FOLLOWUP_TOPICS
from .state import SessionState

Emotion = Literal["neutral", "frustrated", "angry", "anxious", "confused", "refusing", "positive"]


class _Base(BaseModel):
    full_name: Optional[str] = None
    dob: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    id_last4: Optional[str] = None
    account_number: Optional[str] = None

    caller_role: Optional[Literal["holder", "representative"]] = None
    rep_name: Optional[str] = None
    relationship: Optional[str] = None

    record_id: Optional[str] = None
    party: Optional[str] = None
    date: Optional[str] = None
    month: Optional[int] = Field(default=None, ge=1, le=12)
    year: Optional[int] = None
    followup_topic: Optional[str] = None
    question: Optional[str] = None

    scope: Literal["in_scope", "out_of_scope", "smalltalk"] = "in_scope"
    emotion: Emotion = "neutral"
    wants_human: bool = False
    accepts_human: bool = False
    declines_human: bool = False
    refuses_verification: bool = False
    restricted_request: bool = False
    emergency: bool = False
    email_choice: Optional[Literal["send", "skip"]] = None
    done: bool = False

    @field_validator("followup_topic")
    @classmethod
    def _topic(cls, v):
        return v if v in FOLLOWUP_TOPICS else None

    @field_validator("date")
    @classmethod
    def _iso(cls, v):
        try:
            return date.fromisoformat(v).isoformat() if v else None
        except ValueError:
            return None

    @field_validator("*", mode="before")
    @classmethod
    def _blank_to_none(cls, v):
        if isinstance(v, str) and v.strip().lower() in {"", "null", "none", "unknown", "n/a"}:
            return None
        return v


@lru_cache(maxsize=8)
def model_for(domain_key: str):
    from .domain import get_domain
    d = get_domain(domain_key)
    return create_model(
        f"TurnExtraction_{domain_key}", __base__=_Base,
        category=(Optional[Literal[tuple(d.categories)]], None),
        record_status=(Optional[Literal[tuple(d.status_groups)]], None),
        intent=(Optional[Literal[tuple([*d.intents, "other"])]], None),
    )


def json_schema(domain: Domain) -> dict:
    s = model_for(domain.key).model_json_schema()
    s.pop("title", None)
    return s


def _options(d: dict) -> str:
    return "\n".join(f"  {k}: {v}" for k, v in d.items())


@lru_cache(maxsize=8)
def _system(domain_key: str) -> str:
    from .domain import get_domain
    d = get_domain(domain_key)
    return f"""You extract structured fields from one caller turn in a chat with {d.org}.

Return values only for things the caller said in their LATEST message, using earlier turns only to
interpret short answers (for example a bare date after the agent asked for date of birth).
Leave a field out when the caller did not give it. Never guess or fill in values.

Identity
- full_name: the {d.person_noun}'s full name as spoken. If someone calls for the {d.person_noun}, this is the
  {d.person_noun}'s name, and rep_name is the caller's own name.
- dob: ISO format YYYY-MM-DD when the date is unambiguous; otherwise copy what they said.
- phone, email: as given. A spelled out email (jane dot doe at mail dot com) becomes jane.doe@mail.com.
- id_last4: the {d.id_last4_label}.
- account_number: their {d.account_label}, for example {d.account_example}.
- caller_role: representative when the caller says they call for someone else (my husband, my father).

Which {d.record_noun} (capture these even during identity verification)
- record_id like {d.id_prefix}-1234.
- {d.hint_notes}
- category, one of:
{_options(d.categories)}
- date: the {d.record_noun}'s day in YYYY-MM-DD when they name one (today, yesterday, Tuesday), using today's
  date from the context. month and year when they name only those.
- intent, one of:
{_options(d.intents)}
  other: anything else
- followup_topic, only for questions about sending documents: one of {FOLLOWUP_TOPICS}.
- question: the caller's question restated in a few plain words.

Conversation signals
- scope: out_of_scope for requests unrelated to this service (trivia, coding, general knowledge).
  smalltalk for greetings and thanks. Their identity, their {d.record_noun}s, waits, money, documents, the email
  summary, and talking to a person are in_scope.
- emotion: the caller's tone this turn. refusing when they decline to give requested information.
- wants_human: they ask for a person. accepts_human or declines_human: their answer when the agent
  just offered a transfer.
- refuses_verification: they decline or object to giving identity details.
- restricted_request: {d.restricted_note}
- emergency: {d.emergency_note}.
- email_choice: send or skip, only when answering the agent's offer of an email summary.
- done: they say they have no more questions, including a plain no when the agent asked whether
  there is anything else.
"""


def system_prompt(domain: Domain) -> str:
    return _system(domain.key)


def build_user_message(state: SessionState, text: str, today: date, candidates: Optional[list[str]] = None) -> str:
    recent = state.recent(6)
    lines = [f"{m['role'].upper()}: {m['content']}" for m in recent]
    ctx = {"current_phase": state.phase.value, "today": f"{today.isoformat()} ({today.strftime('%A')})"}
    if candidates:
        # After verification only: lets the caller pick between their own records.
        ctx["records_the_agent_listed"] = candidates
    return (
        "Context: " + json.dumps(ctx) + "\n\n"
        "Recent conversation:\n" + ("\n".join(lines) if lines else "(start of chat)") + "\n\n"
        f"LATEST CALLER MESSAGE:\n{text}"
    )


def parse(raw: dict, domain: Domain):
    M = model_for(domain.key)
    clean = {k: v for k, v in (raw or {}).items() if k in M.model_fields}
    try:
        return M(**clean)
    except Exception:
        # Drop fields one at a time until it validates; a bad field never kills the turn.
        ok = {}
        for k, v in clean.items():
            try:
                M(**{**ok, k: v})
                ok[k] = v
            except Exception:
                continue
        return M(**ok)
