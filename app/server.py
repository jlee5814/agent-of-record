"""HTTP API and the chat page. Sessions live in memory."""
from __future__ import annotations

import copy
import logging
import os
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .config import ROOT, settings
from .controller import Agent
from .domain import DOMAINS, get_domain
from .llm import make_llm
from .state import IDENTITY_FIELDS, Phase, SessionState
from .summary import Outbox
from .verify import FailureRegistry

app = FastAPI(title="agent-of-record")
log = logging.getLogger("agent_of_record")


def _error_detail(e: Exception) -> str:
    """The server's own error message, which usually names the cause (schema, model, context length)."""
    return (getattr(e, "message", None) or str(e) or "no detail")[:400]


OUTBOX = Outbox()
# One registry per domain for every session, so a lockout cannot be dodged by opening a new chat.
FAILURES: dict[str, FailureRegistry] = {}


class Session:
    def __init__(self, agent: Agent, state: SessionState, provider: str, model: str):
        self.agent, self.state, self.provider, self.model = agent, state, provider, model
        self.lock = threading.Lock()
        self.last_trace: Optional[dict] = None


SESSIONS: dict[str, Session] = {}


class NewSession(BaseModel):
    provider: Optional[str] = None          # sglang | offline
    model: Optional[str] = None
    base_url: Optional[str] = None
    extract_mode: Optional[str] = None      # json_schema | json_object | prompt; empty tries each
    domain: Optional[str] = None            # freight | clinic
    consent_scenario: str = "default"


class ChatIn(BaseModel):
    session_id: str
    message: str


def _mask(field: str, v: Optional[str]) -> Optional[str]:
    if not v:
        return None
    if field == "full_name":
        return " ".join(w[0] + "***" for w in v.split())
    return "***" + v[-2:] if len(v) > 2 else "***"


def view(sess: Session) -> dict:
    s = sess.state
    return {
        "session_id": s.session_id,
        "provider": sess.provider,
        "model": sess.model,
        "phase": s.phase.value,
        "gates": {
            "pii_provided": len(s.identity.provided()),
            "pii_required": settings.min_pii_matches,
            "identity_matched": s.verified_party_id is not None,
            "verify_attempts": s.verify_attempts,
            "max_attempts": settings.max_verify_attempts,
            "disclosure_allowed": s.verified,
        },
        "identity": {f: _mask(f, getattr(s.identity, f)) for f in (*IDENTITY_FIELDS, "account_number")},
        "caller": {"role": s.caller_role, "rep_name": s.rep_name, "relationship": s.rep_relationship,
                   "rep_authorized": s.rep_authorized},
        "consent": s.consent.model_dump(),
        "memory": s.memory.model_dump(exclude_none=True),
        "domain": {"key": sess.agent.domain.key, "org": sess.agent.domain.org, "noun": sess.agent.domain.record_noun,
                   "person": sess.agent.domain.person_noun, "id_last4": sess.agent.domain.id_last4_label,
                   "account": sess.agent.domain.account_label},
        "record": {"selected": s.selected_id, "candidates": s.candidate_ids, "discussed": list(s.topics_discussed),
                   "live_polls": s.live_polls},
        "extract_mode": getattr(sess.agent.llm, "extract_mode", None) or "rules",
        "signals": {"emotion": s.last_emotion, "frustration_streak": s.frustration_streak,
                    "off_topic_strikes": s.off_topic_strikes, "handoff_offered": s.handoff_offered,
                    "handoff_reason": s.handoff_reason},
        "email": {"decision": s.email_decision, "sent": s.email_sent,
                  "messages": [asdict(m) for m in sess.agent.outbox.for_session(s.session_id)]},
        # Which fields matched stays on the server, so the panel can't be used to probe them.
        "events": [{**e.model_dump(), "data": {k: v for k, v in e.data.items() if k != "detail"}}
                   for e in s.events[-40:] if e.kind != "turn"],
        "last_trace": sess.last_trace,
        "today": settings.today().isoformat(),
    }


@app.get("/api/config")
def config():
    return {
        "default_provider": settings.llm_provider,
        "default_model": settings.model,
        "default_base_url": settings.base_url,
        "default_domain": settings.domain,
        "domains": {k: {"org": get_domain(k).org, "examples": get_domain(k).examples} for k in DOMAINS},
        "demo_today": settings.today().isoformat(),
    }


@app.post("/api/session")
def new_session(body: NewSession):
    provider = body.provider or settings.llm_provider
    try:
        llm = make_llm(provider, body.model or settings.model, body.base_url or settings.base_url,
                       body.extract_mode or settings.extract_mode, os.getenv("OPENAI_API_KEY"),
                       os.getenv("QWEN_THINKING") == "1")
    except Exception as e:  # server down, bad URL, unknown mode
        raise HTTPException(400, f"Could not reach the model server: {_error_detail(e)}")
    try:
        domain = get_domain(body.domain or settings.domain)
    except ValueError as e:
        raise HTTPException(400, str(e))
    state = SessionState()
    if body.consent_scenario in domain.store.consent_scenarios:
        state.consent.scenario = body.consent_scenario
    agent = Agent(llm, domain, settings, OUTBOX, FAILURES.setdefault(domain.key, FailureRegistry(
        settings.max_party_failures, settings.lockout_window_min * 60)))
    sess = Session(agent, state, provider, getattr(llm, "model", "rules"))
    opening = agent.opening(state)
    SESSIONS[state.session_id] = sess
    return {"session_id": state.session_id, "reply": opening, "state": view(sess)}


@app.post("/api/chat")
def chat(body: ChatIn):
    sess = SESSIONS.get(body.session_id)
    if not sess:
        raise HTTPException(404, "Session not found. Start a new session.")
    msg = body.message.strip()
    if not msg:
        raise HTTPException(400, "Empty message.")
    if len(msg) > 2000:
        raise HTTPException(400, "Message too long.")
    with sess.lock:
        snapshot = copy.deepcopy(sess.state)
        try:
            r = sess.agent.handle(sess.state, msg)
        except Exception as e:
            sess.state = snapshot  # a failed turn leaves no partial state behind
            name = type(e).__name__
            detail = _error_detail(e)
            log.error("model call failed: %s: %s", name, detail)
            raise HTTPException(502, f"Model call failed ({name}): {detail}")
        sess.last_trace = r.trace
        return {"reply": r.reply, "state": view(sess)}


@app.get("/api/session/{sid}")
def get_session(sid: str):
    sess = SESSIONS.get(sid)
    if not sess:
        raise HTTPException(404, "Session not found.")
    return {"state": view(sess), "transcript": sess.state.transcript}


@app.get("/api/health")
def health():
    return {"ok": True, "sessions": len(SESSIONS)}


@app.get("/")
def index():
    return FileResponse(Path(ROOT) / "web" / "index.html")
