"""One turn: extract, decide, reply, guard. The only place the model and the code meet.

The model reads the caller and writes the sentence. The procedure decides
everything else, and the guard reconciles the sentence against the record.
"""
from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass, field
from typing import Optional

from .config import Settings, settings as default_settings
from .core import guard as G
from .core import templates as T
from .core.domain import Domain, get_domain
from .core.extraction import parse
from .core.procedure import Procedure
from .core.state import SessionState
from .core.summary import Outbox
from .core.verify import FailureRegistry
from .model import extract as X
from .model import reply as R
from .model.llm import CallStats

PII_FIELDS = ("full_name", "dob", "phone", "email", "id_last4", "account_number")


def mask_pii(d: dict) -> dict:
    """Replace identity values with their length for traces and the panel.

    Free text fields, like the restated question, can quote the caller's
    details too, so those values and any long digit runs are scrubbed there.
    """
    secrets = [str(d[k]) for k in PII_FIELDS if d.get(k)]
    out = {}
    for k, v in d.items():
        if k in PII_FIELDS and v:
            out[k] = f"<{len(str(v))} chars>"
        elif isinstance(v, str):
            for sec in secrets:
                v = v.replace(sec, "***")
            out[k] = re.sub(r"\d[\d/.-]{3,}\d", "***", v)
        else:
            out[k] = v
    return out


@dataclass
class TurnResult:
    reply: str
    state: SessionState
    trace: dict = field(default_factory=dict)


class Agent:
    def __init__(self, llm, domain: Domain | str = "freight", cfg: Settings = default_settings,
                 outbox: Optional[Outbox] = None, failures: Optional[FailureRegistry] = None):
        self.llm = llm
        self.domain = get_domain(domain) if isinstance(domain, str) else domain
        self.store = self.domain.store
        self.procedure = Procedure(self.domain, cfg, outbox, failures)

    @property
    def cfg(self) -> Settings:
        return self.procedure.cfg

    @cfg.setter
    def cfg(self, value: Settings) -> None:
        self.procedure.cfg = value

    @property
    def outbox(self) -> Outbox:
        return self.procedure.outbox

    def opening(self, state: SessionState) -> str:
        if not state.transcript:
            state.transcript.append({"role": "assistant", "content": self.domain.opening})
        return self.domain.opening

    def handle(self, state: SessionState, text: str) -> TurnResult:
        t_turn = time.perf_counter()
        calls: list[CallStats] = []
        phase_before = state.phase

        # 1. Extract. After verification, the extractor also sees the caller's record labels,
        #    so it can map the paper mill one to an ID.
        D = self.domain
        candidates = None
        if state.verified:
            candidates = [D.label(r) for r in self.store.records_for(state.verified_party_id)]
        ex_res = self.llm.extract(X.system_prompt(D), X.build_user_message(state, text, self.cfg.today(), candidates),
                                  X.json_schema(D), "record_turn")
        calls.append(ex_res.stats)
        ex = parse(ex_res.value, D)
        if not ex_res.stats.parse_ok:
            # The turn still runs; an empty extraction just asks again.
            state.log("extract_parse_failed", mode=ex_res.stats.mode)
        state.transcript.append({"role": "user", "content": text})

        # 2. Decide. Deterministic; no model call.
        t_ctl = time.perf_counter()
        directives, grounded = self.procedure.run(state, ex, text)
        ctl_ms = (time.perf_counter() - t_ctl) * 1000

        # 3. Reply, then 4. guard.
        payload = R.build_payload(state, D, directives, grounded, ex.question, ex.emotion)
        gen = self.llm.generate(R.system_prompt(D), R.build_user_message(state, payload))
        calls.append(gen.stats)
        reply = gen.value
        reply_source = "offline_template" if getattr(self.llm, "name", "").startswith("offline") else "model"
        if not reply:
            # Never silent: the trace says the model came back empty and why.
            reply = T.render_template(payload)
            reply_source = "template_empty_model_reply"
            state.log("empty_model_reply", stop_reason=gen.stats.stop_reason,
                      thinking_chars=gen.stats.thinking_chars)

        t_guard = time.perf_counter()
        user_texts = [m["content"] for m in state.transcript if m["role"] == "user"]
        problems = G.guard(reply, state, self.store, D, grounded, user_texts)
        if problems:
            state.log("guard_blocked", problems=problems, blocked=reply)
            reply = T.render_template(payload)
            reply_source = "template_guard"
        guard_ms = (time.perf_counter() - t_guard) * 1000

        state.transcript.append({"role": "assistant", "content": reply})
        total_ms = (time.perf_counter() - t_turn) * 1000
        trace = {
            "phase_before": phase_before.value,
            "phase_after": state.phase.value,
            "extraction": mask_pii(ex.model_dump(exclude_defaults=True)),
            "directives": directives,
            "grounded_keys": sorted((grounded or {}).keys()),
            "guard": problems,
            "reply_source": reply_source,
            "timing_ms": {
                "extract": round(ex_res.stats.latency_ms, 1),
                "control": round(ctl_ms, 2),
                "respond": round(gen.stats.latency_ms, 1),
                "guard": round(guard_ms, 2),
                "total": round(total_ms, 1),
            },
            "calls": [asdict(c) for c in calls],
        }
        state.log("turn", phase_before=phase_before.value, phase_after=state.phase.value,
                  directives=[d["action"] for d in directives], guard=problems)
        return TurnResult(reply, state, trace)
