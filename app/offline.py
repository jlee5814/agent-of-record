"""Offline provider: regex extraction and template replies.

It runs the tests and the keyless demo. It's deliberately simple: its job is
to exercise the controller without a network, not to compete with a model.
"""
from __future__ import annotations

import calendar
import json
import re
import time

from .llm import CallStats, LLMResult

MONTHS = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}
MONTHS.update({m.lower(): i for i, m in enumerate(calendar.month_abbr) if m})

IN_SCOPE = re.compile(
    r"\b(disput\w*|charge\w*|card|account|merchant|refund\w*|credit\w*|deni\w*|document\w*|receipt|order|"
    r"email|verif\w*|dob|birth\w*|name|phone|deadline|human|person|agent|supervisor|status|submit\w*|"
    r"upload\w*|app|summary|send|skip|husband|wife|spouse|mother|father|calling|consent|fraud|transaction|"
    r"statement|bluewater|parking|airline|flight|grocery|money|back)\b",
    re.I)
GREETING = re.compile(r"^\s*(hi|hello|hey|good (morning|afternoon|evening)|thanks|thank you|ok|okay)\b[\s!.]*$", re.I)
NAME = r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})"


def _first(pattern: str, text: str, flags=0):
    m = re.search(pattern, text, flags)
    return m.group(1) if m else None


def heuristic_extract(text: str, phase: str, last_agent: str = "") -> dict:
    t = text.strip()
    tl = t.lower()
    out: dict = {}

    # ---- identity
    out["email"] = _first(r"([\w.+-]+@[\w-]+\.[\w.]+)", t)
    out["account_number"] = _first(r"\b(ACC-?\s?\d{4,})\b", t, re.I)
    iso = _first(r"\b((?:19|20)\d{2}-\d{2}-\d{2})\b", t)
    slash = _first(r"\b(\d{1,2}/\d{1,2}/(?:19|20)\d{2})\b", t)
    words = _first(r"\b((?:%s)\s+\d{1,2}(?:st|nd|rd|th)?,?\s+(?:19|20)\d{2})\b" % "|".join(
        m for m in MONTHS if len(m) > 3), t, re.I)
    out["dob"] = iso or slash or words
    out["card_last4"] = _first(r"(?:card|last\s*(?:four|4)(?:\s*digits)?|ending(?:\s+in)?)[^0-9]{0,25}(\d{4})\b", t, re.I)
    phone = _first(r"(\+?1?[\s.(-]*\d{3}[\s.)-]*\d{3}[\s.-]*\d{4})\b", t)
    if phone and len(re.sub(r"\D", "", phone)) in (10, 11):
        out["phone"] = phone.strip()

    rep = re.search(r"(?i:i'?m|i am|my name is|this is)\s+" + NAME + r".{0,40}?(?i:calling|call|writing)\s+"
                    r"(?i:for|on behalf of)\s+(?:(?i:my|her|his)\s+(\w+)\s*,?\s*)?" + NAME, t)
    if rep:
        out["caller_role"] = "representative"
        out["rep_name"], out["relationship"], out["full_name"] = rep.group(1), rep.group(2), rep.group(3)
    else:
        out["full_name"] = _first(r"(?i:my name is|name's|this is|i'?m|i am|name is)\s+" + NAME, t)
        if re.search(r"\b(cardholder|my card|my dispute|my account)\b", tl):
            out["caller_role"] = "cardholder"

    # ---- dispute hints
    out["dispute_id"] = _first(r"\b(DSP-?\s?\d{3,})\b", t, re.I)
    for merchant in ("bluewater", "parking", "northgate", "grocery", "skyline", "airline", "corner cafe"):
        if merchant in tl:
            out["merchant"] = merchant
            break
    if re.search(r"never (arrived|came|showed)|didn'?t (arrive|show up)|not received|never got", tl):
        out["reason"] = "item_not_received"
    elif re.search(r"charged twice|double charged|duplicate", tl):
        out["reason"] = "duplicate_charge"
    elif re.search(r"\b(cancell?ed flight|flight was cancell?ed|never happened)\b", tl):
        out["reason"] = "service_not_provided"
    elif re.search(r"wrong amount|charged more|overcharged", tl):
        out["reason"] = "wrong_amount"
    elif re.search(r"\b(fraud|didn'?t make|not my charge|unauthori[sz]ed)\b", tl):
        out["reason"] = "fraud"
    if re.search(r"\b(denied|rejected|lost)\b", tl):
        out["dispute_status"] = "denied"
    elif re.search(r"\b(won|credited|refunded)\b", tl):
        out["dispute_status"] = "credited"
    elif re.search(r"\b(open|pending|in progress|under review)\b", tl):
        out["dispute_status"] = "open"
    for name, num in MONTHS.items():
        if re.search(rf"\b(from|in|since|back in|of|on)\s+{name}\b", tl):
            out["month"] = num
            break
    yr = _first(r"\b(20\d{2})\b", t)
    if yr and not (out.get("dob") and yr in out["dob"]):
        out["year"] = int(yr)

    if re.search(r"\bwhy\b.*\b(denied|rejected|lost)\b|\b(denied|rejected|lost)\b.*\bwhy\b", tl):
        out["intent"] = "decision_question"
    elif re.search(r"\b(submit|upload|send|format|pdf|screenshot|photo)\b", tl) and phase == "PROCESS_CASE":
        out["intent"] = "document_submission"
    elif re.search(r"\bstatus\b|what'?s happening|where (is|are) (it|things)", tl):
        out["intent"] = "status_inquiry"
    elif re.search(r"\b(money back|refund|credit|keep the|get the)\b", tl):
        out["intent"] = "credit_question"
    elif out.get("dispute_status") == "denied":
        out["intent"] = "decision_question"
    elif "calling about" in tl or "asking about" in tl:
        out["intent"] = "general_dispute_question"
    if "?" in t:
        out["question"] = t[:120]

    # ---- signals
    if any(out.get(k) for k in ("full_name", "dob", "phone", "email", "card_last4", "account_number", "dispute_id")):
        out["scope"] = "in_scope"
    elif GREETING.match(t):
        out["scope"] = "smalltalk"
    elif IN_SCOPE.search(t):
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

    def extract(self, system: str, user: str, schema: dict, tool_name: str) -> LLMResult:
        t0 = time.perf_counter()
        phase = _first(r'"current_phase":\s*"(\w+)"', user) or "VERIFY_ID"
        latest = user.split("LATEST CALLER MESSAGE:\n", 1)[-1]
        convo = user.split("Recent conversation:\n", 1)[-1].split("\n\nLATEST CALLER MESSAGE", 1)[0]
        agent_lines = [ln for ln in convo.splitlines() if ln.startswith("ASSISTANT:")]
        value = heuristic_extract(latest, phase, agent_lines[-1] if agent_lines else "")
        return LLMResult(value, CallStats("extract", "offline", (time.perf_counter() - t0) * 1000, mode="rules"))

    def generate(self, system: str, user: str, max_tokens: int = 400) -> LLMResult:
        from .respond import render_template
        t0 = time.perf_counter()
        payload = json.loads(user.split("PAYLOAD_JSON:\n", 1)[1])
        text = render_template(payload)
        return LLMResult(text, CallStats("respond", "offline", (time.perf_counter() - t0) * 1000))
