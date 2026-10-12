"""Step 3, the model side: rewrite the procedure's draft so it sounds like a person.

Code renders the correct reply from fixed templates; the model rewrites that
draft in a natural voice and must keep every fact. It never sees the record
itself. core/guard.py checks what comes back.

Found in live runs: an 8B model composing from directives and raw facts asked
for identity after verification and picked the wrong reason for a deduction.
Rewriting a correct draft is a task it does well.
"""
from __future__ import annotations

import json
from functools import lru_cache
from typing import Optional

from ..core.domain import Domain, get_domain
from ..core.state import SessionState


@lru_cache(maxsize=8)
def _system(domain_key: str) -> str:
    d = get_domain(domain_key)
    return f"""You are the voice of {d.org}, writing to a caller in a text chat.

The payload holds DRAFT: the correct reply for this turn, written from fixed templates. It is accurate
but stiff. Rewrite it so it reads like a calm, competent person at the desk.

Rules
1. Keep every fact in the draft: each number, amount, time, date, name, ID, document, and question.
   A list of options stays a list.
2. Add nothing. No new facts, promises, steps, or offers. If the caller asked something the draft
   does not answer, do not answer it.
3. Keep the draft's parts in order. Two to six sentences, plus any list. Plain text, no headings.
4. If caller_emotion is not neutral, you may open with one short, sincere acknowledgement.
5. {d.money_rule}
6. Never mention drafts, directives, controllers, templates, or tools.
7. Reply in the language the caller writes in. Keep numbers, IDs, dates, and amounts exactly as in
   the draft.
"""


def system_prompt(domain: Domain) -> str:
    return _system(domain.key)


def build_payload(state: SessionState, domain: Domain, directives: list[dict], grounded: Optional[dict],
                  caller_question: Optional[str], emotion: str) -> dict:
    return {
        "domain": domain.key,
        "phase": state.phase.value,
        "verified": state.verified,
        "caller_emotion": emotion,
        "caller_question": caller_question,
        "directives": directives,
        "grounded": grounded or {},
    }


def build_user_message(state: SessionState, payload: dict) -> str:
    """What the model sees: the recent chat, the caller's tone, and the draft. Not the record."""
    convo = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in state.recent(6))
    view = {"caller_emotion": payload.get("caller_emotion"), "draft": payload["draft"]}
    return ("Recent conversation:\n" + convo + "\n\n"
            "Rewrite the draft as the agent's next message.\n\nPAYLOAD_JSON:\n" + json.dumps(view, indent=1))
