"""A stand-in for the model: regex extraction and template replies.

It fills the model's seat in the tests and the demo without a server. It's
deliberately simple: its job is to exercise the procedure, not to compete
with a model.
"""
from __future__ import annotations

import calendar
import json
import re
import time
from datetime import date

from .llm import CallStats, LLMResult

MONTHS = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}
MONTHS.update({m.lower(): i for i, m in enumerate(calendar.month_abbr) if m})

IN_SCOPE_COMMON = (r"email|verif\w*|dob|birth\w*|name|phone|human|person|agent|supervisor|status|submit\w*|"
                   r"upload\w*|summary|send|skip|calling|consent|money|paperwork|document\w*|wife|husband|father|"
                   r"mother|daughter|son")
GREETING = re.compile(r"^\s*(hi|hello|hey|good (morning|afternoon|evening)|thanks|thank you|ok|okay)\b[\s!.]*$", re.I)
NAME = r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})"


def _first(pattern: str, text: str, flags=0):
    m = re.search(pattern, text, flags)
    return m.group(1) if m else None


def heuristic_extract(text: str, phase: str, last_agent: str, domain, today: date) -> dict:
    t = text.strip()
    tl = t.lower()
    out: dict = {}

    # ---- identity
    out["email"] = _first(r"([\w.+-]+@[\w-]+\.[\w.]+)", t)
    out["account_number"] = _first(r"\b((?:DRV|MRN)-?\s?\d{4,})\b", t, re.I)
    iso = _first(r"\b((?:19|20)\d{2}-\d{2}-\d{2})\b", t)
    slash = _first(r"\b(\d{1,2}/\d{1,2}/(?:19|20)\d{2})\b", t)
    words = _first(r"\b((?:%s)\s+\d{1,2}(?:st|nd|rd|th)?,?\s+(?:19|20)\d{2})\b" % "|".join(
        m for m in MONTHS if len(m) > 3), t, re.I)
    out["dob"] = iso or slash or words
    out["id_last4"] = _first(r"(?:cdl|member id|insurance id|insurance|custodian account|last\s*(?:four|4)(?:\s*digits)?|ends? in|"
                             r"ending(?:\s+in)?)[^0-9]{0,25}(\d{4})\b", t, re.I)
    phone = _first(r"(\+?1?[\s.(-]*\d{3}[\s.)-]*\d{3}[\s.-]*\d{4})\b", t)
    if phone and len(re.sub(r"\D", "", phone)) in (10, 11):
        out["phone"] = phone.strip()

    rep = re.search(r"(?i:i'?m|i am|my name is|this is)\s+" + NAME + r".{0,40}?(?i:calling|call|writing)\s+"
                    r"(?i:for|on behalf of)\s+(?:(?i:my|her|his)\s+(\w+)\s*,?\s*)?" + NAME, t)
    if rep:
        out["caller_role"] = "representative"
        out["rep_name"], out["relationship"], out["full_name"] = rep.group(1), rep.group(2), rep.group(3)
    else:
        out["full_name"] = (_first(r"(?i:my name is|name's|this is|i'?m|i am|name is)\s+" + NAME, t)
                            or _first(r"^(?:hi,?\s+|hey,?\s+)?" + NAME + r",\s+(?i:dob|born|date of birth)", t))

    # ---- which record, and what they want
    # The caller's own name can echo an account name (Linda Park, Park Family Trust), so hide it from the hints.
    tl_hints = tl
    for nm in (out.get("full_name"), out.get("rep_name")):
        for w in (nm or "").lower().split():
            tl_hints = re.sub(rf"\b{re.escape(w)}\b", " ", tl_hints)
    out.update(domain.offline_hints(t, tl_hints, phase, today))
    for name, num in MONTHS.items():
        if re.search(rf"\b(from|in|since|back in|of|on)\s+{name}\b", tl):
            out["month"] = num
            break
    if "?" in t:
        out["question"] = t[:120]
    if re.search(domain.re_restricted, tl):
        out["restricted_request"] = True
    if re.search(domain.re_emergency, tl):
        out["emergency"] = True

    # ---- signals
    in_scope = re.compile(rf"\b({IN_SCOPE_COMMON}|{domain.in_scope_words})\b", re.I)
    if any(out.get(k) for k in ("full_name", "dob", "phone", "email", "id_last4", "account_number", "record_id")):
        out["scope"] = "in_scope"
    elif GREETING.match(t):
        out["scope"] = "smalltalk"
    elif in_scope.search(t):
        out["scope"] = "in_scope"
    elif "?" in t or re.search(r"^(what|who|whom|where|when|how|why|tell me|write|explain|can you|could you|"
                               r"give me|recommend|define)\b", tl):
        out["scope"] = "out_of_scope"
    else:
        out["scope"] = "in_scope"

    if re.search(r"\b(ridiculous|frustrat|annoy|already told|ugh|waste of time|seriously)\b", tl):
        out["emotion"] = "frustrated"
    elif re.search(r"\b(angry|furious|unacceptable|outrageous)\b", tl) or t.count("!") >= 2:
        out["emotion"] = "angry"
    elif re.search(r"\b(worried|anxious|scared|nervous|stressed)\b", tl):
        out["emotion"] = "anxious"
    elif re.search(r"\b(confused|don'?t understand|what do you mean)\b", tl):
        out["emotion"] = "confused"
    if re.search(r"\b(won'?t give|not giving|not going to give|refuse|none of your business|why do you need)\b", tl):
        out["emotion"] = out.get("emotion") or "refusing"
        out["refuses_verification"] = True

    offered = bool(re.search(r"\b(transfer|human representative|connect you)\b", last_agent.lower()))
    if re.search(r"\b(human|real person|representative|supervisor|someone else|live agent)\b", tl) and not offered:
        out["wants_human"] = True
    if offered:
        if re.search(r"^(yes|yeah|sure|please|ok|okay|transfer)\b", tl):
            out["accepts_human"] = True
        elif re.search(r"^(no|nope|not yet|nah)\b", tl):
            out["declines_human"] = True

    if phase == "POST_PROCESS":
        if re.search(r"\b(skip|no thanks|don'?t send|no need|nope)\b|^no\b", tl):
            out["email_choice"] = "skip"
        elif re.search(r"\b(yes|send|sure|please|ok|okay)\b", tl):
            out["email_choice"] = "send"
    if re.search(r"\b(that'?s all|that is all|no more questions|that'?s it|nothing else|i'?m done|bye|goodbye)\b", tl):
        out["done"] = True
    elif phase == "PROCESS_CASE" and re.search(r"^(no|nope|nah)\b[\s,.!]*(thanks|thank you)?[\s.!]*$", tl):
        out["done"] = True
    return {k: v for k, v in out.items() if v not in (None, "", False)}


class OfflineLLM:
    name = "offline:rules"

    def __init__(self, domain: str | None = None):
        self.domain_key = domain

    def _domain(self, system: str):
        from ..core.domain import DOMAINS, get_domain
        if self.domain_key:
            return get_domain(self.domain_key)
        # No domain given: the extraction prompt names the organization.
        return next((get_domain(k) for k in DOMAINS if get_domain(k).org in system), get_domain(DOMAINS[0]))

    def extract(self, system: str, user: str, schema: dict, tool_name: str) -> LLMResult:
        t0 = time.perf_counter()
        phase = _first(r'"current_phase":\s*"(\w+)"', user) or "VERIFY_ID"
        today = date.fromisoformat(_first(r'"today":\s*"(\d{4}-\d{2}-\d{2})', user) or "2026-03-10")
        latest = user.split("LATEST CALLER MESSAGE:\n", 1)[-1]
        convo = user.split("Recent conversation:\n", 1)[-1].split("\n\nLATEST CALLER MESSAGE", 1)[0]
        agent_lines = [ln for ln in convo.splitlines() if ln.startswith("ASSISTANT:")]
        value = heuristic_extract(latest, phase, agent_lines[-1] if agent_lines else "", self._domain(system), today)
        return LLMResult(value, CallStats("extract", "offline", (time.perf_counter() - t0) * 1000, mode="rules"))

    def generate(self, system: str, user: str, max_tokens: int = 400) -> LLMResult:
        from ..core.templates import render_template
        t0 = time.perf_counter()
        payload = json.loads(user.split("PAYLOAD_JSON:\n", 1)[1])
        text = render_template(payload)
        return LLMResult(text, CallStats("respond", "offline", (time.perf_counter() - t0) * 1000))
