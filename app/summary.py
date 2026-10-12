"""The email summary and a mock outbox.

The summary is built from the record and the session, never from model text.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional

from .config import settings
from .domain import Domain
from .records import document_guidance
from .state import SessionState


@dataclass
class Email:
    to: str
    subject: str
    body: str
    session_id: str
    sent_at: float = field(default_factory=time.time)


class Outbox:
    """Saves each message to disk and keeps it in memory for the UI."""

    def __init__(self, directory: Optional[Path] = None):
        self.dir = directory or Path(settings.data_dir).parent / "outbox"
        self.messages: list[Email] = []

    def send(self, email: Email) -> Email:
        self.messages.append(email)
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            (self.dir / f"{email.session_id}.json").write_text(json.dumps(asdict(email), indent=2))
        except OSError:
            pass
        return email

    def for_session(self, session_id: str) -> list[Email]:
        return [m for m in self.messages if m.session_id == session_id]


def build_summary(state: SessionState, domain: Domain, today: date) -> Email:
    store = domain.store
    p = store.person(state.verified_party_id or "")
    assert p is not None, "summary requires a verified account holder"
    noun = domain.record_noun
    lines = [f"Hello {p.name.split()[0]},", "", f"Here is a summary of your chat with {domain.org}.", ""]

    if state.caller_role == "representative" and state.rep_name:
        rel = f" ({state.rep_relationship})" if state.rep_relationship else ""
        lines += [f"{state.rep_name}{rel} contacted us on your behalf, with your consent.", ""]

    ids = [i for i in dict.fromkeys(state.topics_discussed) if i.startswith(domain.id_prefix + "-")]
    if not ids and state.selected_id:
        ids = [state.selected_id]
    records = [r for r in (store.record(i) for i in ids) if r]

    if records:
        lines.append("What we discussed")
        for r in records:
            lines.append(f"- {domain.label(r)}.")
            secs = domain.facts(r, today, domain.live_entry(r, state))["sections"]
            for name in domain.summary_sections:
                lines += [f"  {f}" for f in secs.get(name, [])]
        lines += ["", "Next steps"]
        for r in records:
            if r.documents_needed:
                g = document_guidance(r, store.guidance)
                for doc in g["documents"]:
                    lines.append(f"- Send the {doc['name']} for {noun} {r.record_id}. {doc['requirements']}")
                if r.deadline:
                    late = f" This date has passed; {domain.team} will review a late submission." \
                        if r.deadline < today.isoformat() else ""
                    lines.append(f"- Deadline: {r.deadline}.{late}")
                lines.append(f"- How to send: {g['general']}")
            else:
                lines.append(f"- Nothing is needed from you on {noun} {r.record_id} right now.")
        lines.append("")
    else:
        lines += [f"We did not review a specific {noun} in this chat.", ""]

    if state.handoff_reason:
        lines += [f"Your chat was transferred to {domain.team}, who has a copy of this summary.", ""]

    lines += ["If you have questions, reply to this email.", "", domain.org]
    return Email(to=p.email, subject=f"Summary of your chat with {domain.org}", body="\n".join(lines),
                 session_id=state.session_id)
