"""Fixed wording for every directive.

Used offline, and whenever the guard blocks a model reply or the model comes
back empty. Plain string building; no model involved.
"""
from __future__ import annotations

from typing import Optional

from .domain import Domain, get_domain


FIELD_LABELS = {"full_name": "full name", "dob": "date of birth", "phone": "phone number on file",
                "email": "email on file"}


def field_label(f: str, d: Domain) -> str:
    return d.id_last4_label if f == "id_last4" else FIELD_LABELS[f]


def mask_email(email: str) -> str:
    local, _, domain = email.partition("@")
    return f"{local[:1]}{'*' * max(len(local) - 1, 3)}@{domain}"


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
            ans = x.get("say") or _answer(g, x.get("intent"), d)
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
