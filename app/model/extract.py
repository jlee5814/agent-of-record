"""Step 1, the model side: the prompt that turns the caller's words into fields.

The extractor sees only the conversation, never the records, so it can't leak
one. What it returns is a proposal; core/extraction.py validates it and the
procedure decides.
"""
from __future__ import annotations

import json
from datetime import date
from functools import lru_cache
from typing import Optional

from ..core.domain import Domain
from ..core.extraction import json_schema
from ..core.records import FOLLOWUP_TOPICS
from ..core.state import SessionState

__all__ = ["system_prompt", "build_user_message", "json_schema"]


def _options(d: dict) -> str:
    return "\n".join(f"  {k}: {v}" for k, v in d.items())


@lru_cache(maxsize=8)
def _system(domain_key: str) -> str:
    from ..core.domain import get_domain
    d = get_domain(domain_key)
    return f"""You extract structured fields from one caller turn in a chat with {d.org}.

Return values only for things the caller said in their LATEST message, using earlier turns only to
interpret short answers (for example a bare date after the agent asked for date of birth).
Leave a field out when the caller did not give it. Never guess or fill in values. Most fields are
empty on most turns; a field with a fixed list of values gets one only when the caller's words name it.

Identity
- full_name: the {d.person_noun}'s full name as spoken. If someone calls for the {d.person_noun}, this is the
  {d.person_noun}'s name, and rep_name is the caller's own name.
- dob: ISO format YYYY-MM-DD when the date is unambiguous; otherwise copy what they said.
- phone, email: as given. A spelled out email (jane dot doe at mail dot com) becomes jane.doe@mail.com.
- id_last4: the {d.id_last4_label}.
- account_number: their {d.account_label}, for example {d.account_example}.
- caller_role: representative only when the caller says they call for someone else (my husband, my
  father); then also fill rep_name and relationship. Someone giving their own details is the holder.

Which {d.record_noun} (capture these even during identity verification)
- record_id: the {d.record_noun} number, only if the caller says one.
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
