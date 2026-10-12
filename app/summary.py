"""The email summary and a mock outbox.

The summary is built from the record and the session, never from model text.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from .disputes import STATUS_WORDS, document_guidance
from .config import settings
from .data import Store, money
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


def build_summary(state: SessionState, store: Store) -> Email:
    p = store.cardholder(state.verified_party_id or "")
    assert p is not None, "summary requires a verified cardholder"
    today = settings.today().isoformat()
    lines = [f"Hello {p.name.split()[0]},", "", "Here is a summary of your chat with Acme Bank card disputes.", ""]

    if state.caller_role == "representative" and state.rep_name:
        rel = f" ({state.rep_relationship})" if state.rep_relationship else ""
        lines += [f"{state.rep_name}{rel} contacted us on your behalf, with your consent.", ""]

    discussed = [i for i in dict.fromkeys(state.topics_discussed) if i.startswith("DSP-")]
    if not discussed and state.selected_id:
        discussed = [state.selected_id]
    disputes = [d for d in (store.dispute(i) for i in discussed) if d]

    if disputes:
        lines.append("What we discussed")
        for d in disputes:
            lines.append(f"- Dispute {d.dispute_id}: ${money(d.disputed_amount):,.2f} charge from {d.merchant} on "
                         f"{d.transaction_date}. Status: {STATUS_WORDS[d.status]}.")
            if d.decision_reason and d.status == "resolved_denied":
                lines.append(f"  Reason: {d.decision_reason}.")
            if d.status in ("needs_info", "under_review") and d.provisional_credit != "0.00":
                lines.append(f"  A provisional credit of ${money(d.provisional_credit):,.2f} is on your account. "
                             "It is temporary until the dispute is decided.")
        lines += ["", "Next steps"]
        for d in disputes:
            if d.documents_needed:
                g = document_guidance(d, store.guidance)
                for doc in g["documents"]:
                    lines.append(f"- Send the {doc['name']} for dispute {d.dispute_id}. {doc['requirements']}")
                if d.response_deadline:
                    late = " This date has passed; a dispute specialist will review a late submission." \
                        if d.response_deadline < today else ""
                    lines.append(f"- Deadline: {d.response_deadline}.{late}")
                lines.append(f"- How to send: {g['general']}")
                lines.append(f"- After we have everything, a decision usually takes {g['processing_time']}.")
            elif d.status == "under_review":
                lines.append(f"- Dispute {d.dispute_id} is under review. Nothing is needed from you right now.")
            else:
                lines.append(f"- Dispute {d.dispute_id} is closed. No further action is needed.")
        lines.append("")
    else:
        lines += ["We did not review a specific dispute in this chat.", ""]

    if state.handoff_reason:
        lines += ["Your chat was transferred to a person, who has a copy of this summary.", ""]

    lines += ["If you have questions, reply to this email or call the number on the back of your card.", "",
              "Acme Bank card disputes"]
    return Email(to=p.email, subject="Summary of your card dispute chat", body="\n".join(lines),
                 session_id=state.session_id)
