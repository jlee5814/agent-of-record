"""Signals code reads from the caller's words without a model.

The model reads fuzzy language better than any regex. A few signals have an
exact shape, though, and some matter too much to leave to a sampler: an injury,
a proxy introducing themselves, a request the agent can't grant, the caller
saying they're done. Found live with Qwen3-8B: it missed each of these at least
once. Code can only add these signals, never remove one the model raised.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from .domain import Domain

NAME = r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})"
PROXY = re.compile(r"(?i:i'?m|i am|my name is|this is)\s+" + NAME + r".{0,40}?(?i:calling|call|writing)\s+"
                   r"(?i:for|on behalf of)\s+(?:(?i:my|her|his)\s+(\w+)\s*,?\s*)?" + NAME)
DONE = re.compile(r"\b(that'?s all|that is all|no more questions|that'?s it|nothing else|i'?m done|bye|goodbye)\b", re.I)
BARE_NO = re.compile(r"^(no|nope|nah)\b[\s,.!]*(thanks|thank you)?[\s.!]*$", re.I)
SEND = re.compile(r"\b(yes|send|sure|please|ok|okay)\b", re.I)
SKIP = re.compile(r"\b(skip|no thanks|don'?t send|no need|nope)\b|^no\b", re.I)
COMMON_SCOPE = (r"email|verif\w*|dob|birth\w*|name|phone|human|person|agent|supervisor|status|submit\w*|"
                r"upload\w*|summary|send|skip|calling|consent|money|paperwork|document\w*|wife|husband|father|"
                r"mother|daughter|son|account")
QUESTION_START = re.compile(r"^(what|who|whom|where|when|how|why|tell me|write|explain|can you|could you|"
                            r"give me|recommend|define)\b", re.I)


MONTHS = "january|february|march|april|may|june|july|august|september|october|november|december"
IDENTITY = {
    "email": re.compile(r"([\w.+-]+@[\w-]+\.[\w.]+)"),
    "dob": re.compile(r"\b((?:19|20)\d{2}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/(?:19|20)\d{2}|"
                      rf"(?:{MONTHS})\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+(?:19|20)\d{{2}})\b", re.I),
    "id_last4": re.compile(r"(?:cdl|member id|insurance id|insurance|custodian account|card|last\s*(?:four|4)"
                           r"(?:\s*digits)?|ends? in|ending(?:\s+in)?)[^0-9]{0,25}(\d{4})\b", re.I),
    "phone": re.compile(r"(\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4})\b"),
}


@dataclass
class Signals:
    emergency: bool = False
    restricted: bool = False
    done: bool = False
    email_choice: Optional[str] = None
    off_topic: bool = False
    rep_name: Optional[str] = None
    relationship: Optional[str] = None
    holder_name: Optional[str] = None
    identity: dict = None


def read(text: str, phase: str, domain: Domain) -> Signals:
    t, tl = text.strip(), text.strip().lower()
    sig = Signals(emergency=bool(re.search(domain.re_emergency, tl)),
                  restricted=bool(re.search(domain.re_restricted, tl)))
    m = PROXY.search(t)
    if m:
        sig.rep_name, sig.relationship, sig.holder_name = m.group(1), m.group(2), m.group(3)
    if phase in ("PROCESS_CASE", "POST_PROCESS"):
        sig.done = bool(DONE.search(tl)) or (phase == "PROCESS_CASE" and bool(BARE_NO.match(tl)))
    if phase == "POST_PROCESS":
        sig.email_choice = "skip" if SKIP.search(tl) else "send" if SEND.search(tl) else None
    sig.identity = {f: m.group(1) for f, rx in IDENTITY.items() if (m := rx.search(t))}
    in_scope = re.search(rf"\b({COMMON_SCOPE}|{domain.in_scope_words})\b", tl)
    sig.off_topic = not in_scope and ("?" in t or bool(QUESTION_START.match(tl)))
    return sig
