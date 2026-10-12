"""Step 3: word the controller's directives as a reply, then check it.

The reply model gets the directives and the approved facts, which are empty
before verification. The guard checks each reply against what the phase allows
and substitutes a template if it fails.
"""
from __future__ import annotations

import json
import re
from decimal import Decimal, InvalidOperation
from functools import lru_cache
from typing import Optional

from .data import Store, norm_phone
from .domain import Domain, get_domain
from .state import SessionState


@lru_cache(maxsize=8)
def _system(domain_key: str) -> str:
    d = get_domain(domain_key)
    return f"""You are the voice of {d.org}, speaking with a caller in a text chat.

A controller has already decided what happens this turn. You receive its decisions as DIRECTIVES and
any data you may use as GROUNDED. Your job is to say it naturally, like a calm, competent person at
the desk.

Hard rules
1. Carry out every directive, in order, and nothing beyond them. Do not advance the process on your own.
2. State facts only from GROUNDED. Never invent amounts, times, wait estimates, {d.record_noun} numbers,
   names, document names, deadlines, or outcomes. If GROUNDED does not answer the question, say you
   do not have that detail and offer what you can do.
3. If GROUNDED is empty, you know nothing about any {d.record_noun} or record. Do not confirm or deny whether
   one exists, its status, or whether any identity detail was right.
4. Never say which identity field failed to match. Never repeat back full identity values.
5. {d.money_rule} Never promise money or an outcome.
6. If the caller's emotion is not neutral, open with one short, sincere acknowledgement, then continue.
   Do not over apologize. Explain the reason for a required step when a directive asks you to.
7. Keep it short: two to five sentences, plain text, no markdown headings, at most one short list.
   End with a clear question or next step when the directives call for input.
8. Do not mention directives, controllers, phases, or tools.
9. If a directive says the speaker changed, treat everything earlier in the chat as unknown to you.
10. Consent comes only from the {d.person_noun}, through {d.consent_channel}. Someone calling for them can never
    grant it. Never ask the caller whether the {d.person_noun} consents, and never treat their word as consent.
11. Reply in the language the caller is writing in. Keep numbers, times, dates, and dollar amounts
    exactly as they appear in GROUNDED.
"""


def system_prompt(domain: Domain) -> str:
    return _system(domain.key)


FIELD_LABELS = {"full_name": "full name", "dob": "date of birth", "phone": "phone number on file",
                "email": "email on file"}


def field_label(f: str, d: Domain) -> str:
    return d.id_last4_label if f == "id_last4" else FIELD_LABELS[f]


def mask_email(email: str) -> str:
    local, _, domain = email.partition("@")
    return f"{local[:1]}{'*' * max(len(local) - 1, 3)}@{domain}"


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
    convo = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in state.recent(6))
    return ("Recent conversation:\n" + convo + "\n\n"
            "Write the agent's next message.\n\nPAYLOAD_JSON:\n" + json.dumps(payload, indent=1))


# -------------------------------------------------------------- templates

def _join(items: list[str]) -> str:
    items = [i for i in items if i]
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + f" and {items[-1]}"


def _answer(g: dict, intent: Optional[str], d: Domain) -> str:
    rec = g.get("record") or {}
    if not rec:
        return ""
    follow = g.get("followup")
    if follow and follow.get("topic") != "fallback":
        return follow["text"]
    return d.answer(rec, intent)


def render_template(payload: dict) -> str:
    """Word the directives with fixed templates. Used offline and when the guard blocks a reply."""
    d = get_domain(payload.get("domain", "freight"))
    g = payload.get("grounded") or {}
    noun, person = d.record_noun, d.person_noun
    out: list[str] = []
    for x in payload.get("directives", []):
        a = x["action"]
        if a == "emergency":
            out.append(d.emergency_reply)
        elif a == "empathize":
            out.append({
                "frustrated": "I understand this is frustrating, and I'm sorry for the hassle.",
                "angry": "I hear you, and I'm sorry this has been so difficult.",
                "anxious": "I understand this is stressful. I'll walk you through it.",
                "confused": "No problem, let me explain.",
                "refusing": "I understand your hesitation.",
            }.get(x.get("emotion"), "I understand."))
        elif a == "identity_changed":
            out.append(f"Since I'm now speaking with someone new, I need to verify the caller again before discussing any {noun}.")
        elif a == "greet_back":
            out.append("Hello, thanks for reaching out.")
        elif a == "off_topic":
            s = f"I'm sorry, I can only help with your {noun}s here."
            if x.get("offer_human"):
                s += f" If you need something else, I can transfer you to {d.team}. Would you like that?"
            out.append(s)
        elif a == "restricted":
            out.append(d.restricted_reply)
        elif a == "noted_hint":
            out.append("I've noted what you're asking about, and I'll pull it up once you're verified.")
        elif a == "ask_identity":
            missing = [field_label(f, d) for f in x.get("missing", [])]
            n = x.get("needed", 0)
            explained = any(y["action"] == "explain_verification" for y in payload.get("directives", []))
            if explained and x.get("have", 0) == 0:
                out.append("Could you share three of those details?")
            elif x.get("have", 0) == 0:
                out.append(f"Before I can discuss any {noun}, I need to verify your identity. Please share at least "
                           f"three of the following: {_join(missing)}.")
            else:
                out.append(f"Thank you. I need {n} more detail{'s' if n != 1 else ''} to verify your identity. "
                           f"You can give me your {_join(missing)}.")
        elif a == "verify_failed":
            left = x.get("attempts_left", 0)
            out.append("I wasn't able to verify your identity with those details. Could you check them and try again? "
                       f"You have {left} more attempt{'s' if left != 1 else ''}, or I can transfer you to {d.team}.")
        elif a == "explain_verification":
            labels = _join([field_label(f, d) for f in ("full_name", "dob", "phone", "email", "id_last4")]).replace(" and ", " or ")
            out.append(f"{d.privacy_reason}, so I have to confirm I'm speaking with the {person} or someone they've "
                       f"authorized before sharing anything. Any three of your {labels} will do.")
        elif a == "ask_rep_identity":
            out.append(f"Since you're calling for someone else, could you tell me your full name and how you're related to the {person}?")
        elif a == "rep_not_authorized":
            out.append(f"I'm sorry, but I don't see you listed as an authorized contact, so I can't share details. "
                       f"The {person} can contact us directly or add you, or I can transfer you to {d.team}.")
        elif a == "consent_requested":
            out.append(f"Thank you. You're listed as an authorized contact. As a last step, I've sent a consent request "
                       f"to the {person} through {d.consent_channel}. Once they approve it, I can share details.")
        elif a == "consent_pending":
            out.append(f"The consent request is still pending. Once the {person} approves it in {d.consent_channel}, I can continue.")
        elif a == "consent_approved":
            out.append(f"The {person} approved the request in {d.consent_channel}. Thanks for waiting.")
        elif a == "consent_timeout":
            out.append(f"The {person} hasn't approved the consent request, so I can't share details today. "
                       f"I'll transfer you to {d.team}, who can help with other options.")
        elif a == "verified":
            out.append("Thank you, you're verified.")
        elif a == "record_selected":
            out.append(f"I found the {noun}: {x['label']}.")
        elif a == "ask_disambiguate":
            out.append(f"I see more than one {noun} that could match. Which one do you mean?\n" +
                       "\n".join(f"- {o}" for o in x.get("options", [])))
        elif a == "no_match":
            out.append(f"I couldn't find a {noun} matching that. Here's what I have on file:\n" +
                       "\n".join(f"- {o}" for o in x.get("options", [])) + "\nWhich one would you like to discuss?")
        elif a == "no_records":
            out.append(f"I don't see any {noun}s on file for you. Is there something else I can help with?")
        elif a == "ask_record_question":
            out.append("What would you like to know about it?")
        elif a == "answer_record":
            ans = _answer(g, x.get("intent"), d)
            out.append(ans or f"I don't have that detail on file, but I can tell you this {noun}'s status and what's next.")
        elif a == "deadline_passed":
            out.append(f"Because the deadline has passed, {d.team} should review a late submission. Would you like me to transfer you?")
        elif a == "anything_else":
            out.append(f"Is there anything else I can help you with on this {noun}?")
        elif a == "offer_email":
            out.append(f"Would you like me to email a summary of today's chat to {x.get('email_masked', 'the email on file')}? "
                       "You can also skip it.")
        elif a == "email_sent":
            out.append(f"Done. I've sent the summary to {x.get('email_masked')}. Thanks for reaching out, and take care.")
        elif a == "email_skipped":
            out.append("No problem, I won't send an email. Thanks for reaching out, and take care.")
        elif a == "offer_human":
            out.append(f"Would you like me to transfer you to {d.team}?")
        elif a == "handoff":
            out.append(f"I'm transferring you to {d.team} now. I've passed along a summary so you won't have to repeat yourself.")
        elif a == "handoff_declined":
            out.append("Okay, we'll keep going here.")
        elif a == "ended":
            out.append("This chat has ended. Please start a new one if you need anything else.")
    return " ".join(y for y in out if y).replace(" \n", "\n")


# -------------------------------------------------------------- output guard

_MONEY = re.compile(r"\$\s?\d[\d,]*(?:\.\d{2})?")
_CLOCK = re.compile(r"\b(\d{1,2}):(\d{2})\s*([ap])?\.?\s*m?\.?\b", re.I)
_DURATION = re.compile(r"\b(\d+(?:\.\d+)?)\s*(?:more\s+|extra\s+)?(minutes?|mins?|hours?|hrs?)\b", re.I)


def _norm_money(s: str) -> str:
    """Normalize an amount to cents so $45 and $45.00 match."""
    digits = re.sub(r"[^\d.]", "", s).rstrip(".")
    try:
        return str(Decimal(digits).quantize(Decimal("0.01")))
    except InvalidOperation:
        return digits


def _clocks(text: str) -> set[tuple[int, int]]:
    """2:30 PM, 2:30pm, and 14:30 all become (2, 30)."""
    return {(int(h) % 12, int(m)) for h, m, _ in _CLOCK.findall(text)}


def _minutes(text: str) -> set[float]:
    """Durations in minutes, so 2 hours and 120 minutes match."""
    out = set()
    for n, unit in _DURATION.findall(text):
        v = float(n)
        out.add(v * 60 if unit.lower().startswith("h") else v)
    return out


def guard(reply: str, state: SessionState, store: Store, domain: Domain, grounded: Optional[dict],
          user_texts: list[str]) -> list[str]:
    """List what a reply says that the phase doesn't allow. Empty means send it."""
    problems: list[str] = []
    said = " ".join(user_texts)
    said_low = said.lower()
    said_digits = re.sub(r"\D", "", said)
    low = reply.lower()

    if not state.verified:
        if domain.id_regex.search(reply):
            problems.append(f"{domain.record_noun} id before verification")
        if _MONEY.search(reply):
            problems.append("amount before verification")
        for r in store.records:
            for term in [*domain.private_terms(r), *r.documents_needed]:
                if term and term.lower() in low and term.lower() not in said_low:
                    problems.append(f"record detail before verification: {term}")
        for p in store.people:
            if p.email.lower() in low and p.email.lower() not in said_low:
                problems.append("record email echoed")
            ph = norm_phone(p.phone) or ""
            if ph and ph[-10:] in re.sub(r"\D", "", reply) and ph[-10:] not in said_digits:
                problems.append("record phone echoed")
            if p.dob in reply and p.dob not in said:
                problems.append("record dob echoed")
        return sorted(set(problems))

    # Verified: every ID, amount, clock time, and duration must be grounded or said by the caller.
    own = {r.record_id for r in store.records_for(state.verified_party_id or "")}
    for m in domain.id_regex.findall(reply):
        rid = f"{domain.id_prefix}-" + re.sub(r"\D", "", m)
        if rid not in own:
            problems.append(f"ungrounded {domain.record_noun} id {rid}")
    blob = json.dumps(grounded or {}) + " " + said
    allowed_money = {_norm_money(m) for m in _MONEY.findall(blob)}
    for m in _MONEY.findall(reply):
        if _norm_money(m) not in allowed_money:
            problems.append(f"ungrounded amount {m}")
    allowed_clocks, allowed_minutes = _clocks(blob), _minutes(blob)
    for c in _clocks(reply) - allowed_clocks:
        problems.append(f"ungrounded time {c[0] or 12}:{c[1]:02d}")
    for v in _minutes(reply) - allowed_minutes:
        problems.append(f"ungrounded duration {v:g} minutes")
    return sorted(set(problems))
