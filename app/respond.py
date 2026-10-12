"""Step 3: word the controller's directives as a reply, then check it.

The reply model gets the directives and the approved facts, which are empty
before verification. The guard checks each reply against what the phase allows
and substitutes a template if it fails.
"""
from __future__ import annotations

import json
import re
from decimal import Decimal, InvalidOperation
from typing import Optional

from .data import Store, norm_phone
from .state import SessionState

SYSTEM = """You are the voice of a card dispute support agent at Acme Bank, speaking with a caller in a text chat.

A controller has already decided what happens this turn. You receive its decisions as DIRECTIVES and
any data you may use as GROUNDED. Your job is to say it naturally, like a calm, competent human
representative.

Hard rules
1. Carry out every directive, in order, and nothing beyond them. Do not advance the process on your own.
2. State facts only from GROUNDED. Never invent amounts, dates, dispute numbers, document names,
   deadlines, or outcomes. If GROUNDED does not answer the caller's question, say you do not have that
   detail and offer what you can do.
3. If GROUNDED is empty, you know nothing about any dispute or record. Do not confirm or deny whether a
   dispute exists, its status, or whether any identity detail was right.
4. Never say which identity field failed to match. Never repeat back full identity values.
5. The disputed amount is what the caller contests, never an amount they will get back. A provisional
   credit is temporary and can be reversed. Never promise a refund or an outcome.
6. If the caller's emotion is not neutral, open with one short, sincere acknowledgement, then continue.
   Do not over apologize or grovel. Explain the reason for a required step when a directive asks you to.
7. Keep it short: two to five sentences, plain text, no markdown headings, at most one short list.
   End with a clear question or next step when the directives call for input.
8. Do not mention directives, controllers, phases, or tools.
9. If a directive says the speaker changed, treat everything earlier in the call as unknown to you.
10. Consent comes only from the cardholder, through the request sent to their banking app. Someone
    calling for them can never grant it. Never ask the caller whether the cardholder consents, and never
    treat their word as consent. When consent is approved, credit the cardholder's approval.
11. Reply in the language the caller is writing in. Keep dispute numbers, dates, and dollar amounts
    exactly as they appear in GROUNDED.
"""

FIELD_LABELS = {"full_name": "full name", "dob": "date of birth", "phone": "phone number on file",
                "email": "email on file", "card_last4": "last four digits of your card"}


def mask_email(email: str) -> str:
    local, _, domain = email.partition("@")
    return f"{local[:1]}{'*' * max(len(local) - 1, 3)}@{domain}"


def build_payload(state: SessionState, directives: list[dict], grounded: Optional[dict],
                  caller_question: Optional[str], emotion: str) -> dict:
    return {
        "phase": state.phase.value,
        "verified": state.verified,
        "caller_emotion": emotion,
        "caller_question": caller_question,
        "directives": directives,
        "grounded": grounded or {},
    }


def build_user_message(state: SessionState, payload: dict) -> str:
    recent = state.recent(6)
    convo = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in recent)
    return ("Recent conversation:\n" + convo + "\n\n"
            "Write the agent's next message.\n\nPAYLOAD_JSON:\n" + json.dumps(payload, indent=1))


# -------------------------------------------------------------- templates

def _join(items: list[str]) -> str:
    items = [i for i in items if i]
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + f" and {items[-1]}"


def _answer_dispute(g: dict, intent: Optional[str]) -> str:
    d = g.get("dispute") or {}
    if not d:
        return ""
    facts = d.get("facts", [])
    follow = g.get("followup")
    if follow and follow.get("topic") != "fallback":
        return follow["text"]
    if d.get("status") == "resolved_denied" and intent in (None, "decision_question", "general_dispute_question",
                                                           "status_inquiry", "credit_question"):
        return (f"Dispute {d['dispute_id']} with {d['merchant']} was decided for the merchant because "
                f"{d['decision_reason']}. No credit was issued.")
    if d.get("status") == "needs_info" and intent in (None, "status_inquiry", "next_steps", "general_dispute_question",
                                                      "document_submission"):
        parts = [f"Dispute {d['dispute_id']} with {d['merchant']} is waiting on documents from you."]
        docs = _join(d.get("documents_needed", []))
        if docs:
            parts.append(f"To move it forward, please send the {docs}.")
        dl = d.get("deadline")
        if dl and not dl["expired"]:
            parts.append(f"They're due by {dl['date']}.")
        elif dl and dl["expired"]:
            parts.append(f"The deadline was {dl['date']}, so a dispute specialist will need to review a late submission.")
        return " ".join(parts)
    if intent == "credit_question":
        return " ".join(f for f in facts if "$" in f) or " ".join(facts[:2])
    if follow:
        return " ".join(facts[:2]) + " " + follow["text"]
    return " ".join(facts[:3])


def render_template(payload: dict) -> str:
    """Word the directives with fixed templates. Used offline and when the guard blocks a reply."""
    g = payload.get("grounded") or {}
    out: list[str] = []
    for d in payload.get("directives", []):
        a = d["action"]
        if a == "empathize":
            out.append({
                "frustrated": "I understand this is frustrating, and I'm sorry for the hassle.",
                "angry": "I hear you, and I'm sorry this has been so difficult.",
                "anxious": "I understand this is stressful. I'll walk you through it step by step.",
                "confused": "No problem, let me explain.",
                "refusing": "I understand your hesitation.",
            }.get(d.get("emotion"), "I understand."))
        elif a == "identity_changed":
            out.append("Since I'm now speaking with someone new, I need to verify the caller again before discussing any dispute.")
        elif a == "greet_back":
            out.append("Hello, thanks for reaching out.")
        elif a == "off_topic":
            s = "I'm sorry, I can only help with your card disputes."
            if d.get("offer_human"):
                s += " If you need help with something else, I can transfer you to a person. Would you like that?"
            out.append(s)
        elif a == "noted_hint":
            out.append("I've noted what you're calling about, and I'll pull it up once you're verified.")
        elif a == "ask_identity":
            missing = [FIELD_LABELS[f] for f in d.get("missing", [])]
            n = d.get("needed", 0)
            explained = any(x["action"] == "explain_verification" for x in payload.get("directives", []))
            if explained and d.get("have", 0) == 0:
                out.append("Could you share three of those details?")
            elif d.get("have", 0) == 0:
                out.append("Before I can discuss any dispute, I need to verify your identity. Please share at least "
                           f"three of the following: {_join(missing)}.")
            else:
                out.append(f"Thank you. I need {n} more detail{'s' if n != 1 else ''} to verify your identity. "
                           f"You can give me your {_join(missing)}.")
        elif a == "verify_failed":
            left = d.get("attempts_left", 0)
            out.append("I wasn't able to verify your identity with those details. Could you check them and try again? "
                       f"You have {left} more attempt{'s' if left != 1 else ''}, or I can transfer you to a person.")
        elif a == "explain_verification":
            out.append("Dispute details are private account information, so I have to confirm I'm speaking with the "
                       "cardholder or someone they've authorized before sharing anything. Any three of your full name, "
                       "date of birth, phone number, email, or the last four digits of your card will do.")
        elif a == "ask_rep_identity":
            out.append("Since you're calling for someone else, could you tell me your full name and how you're related to the cardholder?")
        elif a == "rep_not_authorized":
            out.append("I'm sorry, but I don't see you listed as an authorized user on this account, so I can't share "
                       "dispute details. The cardholder can contact us directly or add you, or I can transfer you to a person.")
        elif a == "consent_requested":
            out.append("Thank you. You're listed as an authorized user. As a last step, I've sent a consent request to the "
                       "cardholder's banking app. Once they approve it, I can share the dispute details.")
        elif a == "consent_pending":
            out.append("The consent request is still pending. Once the cardholder approves it in their app, I can continue.")
        elif a == "consent_approved":
            out.append("The cardholder approved the request in their app. Thanks for waiting.")
        elif a == "consent_timeout":
            out.append("The cardholder hasn't approved the consent request, so I can't share dispute details today. "
                       "I'll transfer you to a person who can help with other options.")
        elif a == "verified":
            out.append("Thank you, you're verified.")
        elif a == "dispute_selected":
            out.append(f"I found the dispute you mentioned: {d['label']}.")
        elif a == "ask_disambiguate":
            opts = d.get("options", [])
            out.append("I see more than one dispute that could match. Which one are you asking about?\n" +
                       "\n".join(f"- {o}" for o in opts))
        elif a == "no_match":
            opts = d.get("options", [])
            out.append("I couldn't find a dispute matching that. Here's what I have on file:\n" +
                       "\n".join(f"- {o}" for o in opts) + "\nWhich one would you like to discuss?")
        elif a == "no_disputes":
            out.append("I don't see any disputes on your account. Is there something else I can help with?")
        elif a == "ask_dispute_question":
            out.append("What would you like to know about it?")
        elif a == "answer_dispute":
            ans = _answer_dispute(g, d.get("intent"))
            out.append(ans or "I don't have that detail on file, but I can tell you the dispute's status and what's needed next.")
        elif a == "deadline_passed":
            out.append("Because the deadline has passed, a dispute specialist should review a late submission. Would you like me to transfer you?")
        elif a == "anything_else":
            out.append("Is there anything else I can help you with on this dispute?")
        elif a == "offer_email":
            out.append(f"Would you like me to email a summary of today's chat, including the dispute status and next steps, "
                       f"to {d.get('email_masked', 'the email on file')}? You can also skip it.")
        elif a == "email_sent":
            out.append(f"Done. I've sent the summary to {d.get('email_masked')}. Thanks for reaching out, and take care.")
        elif a == "email_skipped":
            out.append("No problem, I won't send an email. Thanks for reaching out, and take care.")
        elif a == "offer_human":
            out.append("Would you like me to transfer you to a person?")
        elif a == "handoff":
            out.append("I'm transferring you to a person now. I've passed along a summary so you won't have to repeat yourself.")
        elif a == "handoff_declined":
            out.append("Okay, we'll keep going here.")
        elif a == "ended":
            out.append("This chat has ended. Please start a new one if you need anything else.")
    return " ".join(x for x in out if x).replace(" \n", "\n")

# -------------------------------------------------------------- output guard

_DISPUTE_ID = re.compile(r"\bDSP-?\s?\d{3,}\b", re.I)
_MONEY = re.compile(r"\$\s?\d[\d,]*(?:\.\d{2})?")


def _norm_money(s: str) -> str:
    """Normalize an amount to cents so $1,249 and $1,249.00 match."""
    digits = re.sub(r"[^\d.]", "", s).rstrip(".")
    try:
        return str(Decimal(digits).quantize(Decimal("0.01")))
    except InvalidOperation:
        return digits


def guard(reply: str, state: SessionState, store: Store, grounded: Optional[dict],
          user_texts: list[str]) -> list[str]:
    """List what a reply says that the phase doesn't allow. Empty means send it."""
    problems: list[str] = []
    said = " ".join(user_texts).lower()
    said_digits = re.sub(r"\D", "", said)

    if not state.verified:
        low = reply.lower()
        if _DISPUTE_ID.search(reply):
            problems.append("dispute id before verification")
        if _MONEY.search(reply):
            problems.append("amount before verification")
        for d in store.disputes:
            if d.decision_reason and d.decision_reason.lower()[:40] in low:
                problems.append("decision reason before verification")
            for doc in d.documents_needed:
                if doc.lower() in low and doc.lower() not in said:
                    problems.append("required document before verification")
            if d.merchant.lower() in low and d.merchant.lower() not in said:
                problems.append("merchant before verification")
        for p in store.cardholders:
            if p.email.lower() in low and p.email.lower() not in said:
                problems.append("record email echoed")
            ph = norm_phone(p.phone) or ""
            if ph and ph[-10:] in re.sub(r"\D", "", reply) and ph[-10:] not in said_digits:
                problems.append("record phone echoed")
            if p.dob in reply and p.dob not in said:
                problems.append("record dob echoed")
        return sorted(set(problems))

    # Verified: every dispute ID and amount must be grounded.
    own = {d.dispute_id for d in store.disputes_for(state.verified_party_id or "")}
    for m in _DISPUTE_ID.findall(reply):
        did = "DSP-" + re.sub(r"\D", "", m)
        if did not in own:
            problems.append(f"ungrounded dispute id {did}")
    allowed = {_norm_money(m) for m in _MONEY.findall(json.dumps(grounded or {}))}
    for m in _MONEY.findall(reply):
        if _norm_money(m) not in allowed:
            problems.append(f"ungrounded amount {m}")
    return sorted(set(problems))
