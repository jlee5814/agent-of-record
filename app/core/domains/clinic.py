"""Clinic: a patient, or family with consent, asking about the wait, their turn, the doctor, or a bill."""
from __future__ import annotations

import re
from datetime import date
from typing import Optional

from ..data import Record, clock, long_date, money, usd
from ..domain import Domain

CATEGORY_WORDS = {"sick_visit": "sick visit", "follow_up": "follow up visit", "physical": "physical",
                  "lab_work": "lab visit"}


class Clinic(Domain):
    key = "clinic"
    org = "AF Clinic front desk"
    opening = ("Thanks for contacting AF Clinic. I can help with your visit, the wait, or a bill. "
               "How can I help?")
    record_noun = "visit"
    id_prefix = "APT"
    person_noun = "patient"
    id_last4_label = "last four digits of your insurance member ID"
    account_label = "patient number"
    account_example = "MRN-10482"
    consent_channel = "the patient portal"
    team = "the front desk"
    privacy_reason = "Visit details are private health information"
    categories = {"sick_visit": "same day sick visit", "follow_up": "follow up", "physical": "annual physical",
                  "lab_work": "lab draw"}
    status_groups = {"waiting": {"checked_in", "roomed"}, "upcoming": {"scheduled"}, "done": {"completed"},
                     "cancelled": {"cancelled"}}
    status_words = {"scheduled": "scheduled", "checked_in": "checked in and waiting", "roomed": "in an exam room",
                    "with_provider": "with the provider", "completed": "completed", "cancelled": "cancelled"}
    intents = {
        "wait_time": "how long the wait is, their place in line, when it is their turn, when the doctor is coming",
        "appointment_time": "when the appointment is",
        "bill_question": "a copay, a bill, or why they owe an amount",
        "document_submission": "sending documents such as an insurance card",
        "general_visit_question": "anything else about a visit",
    }
    live_intents = {"wait_time"}
    summary_sections = ["bill"]
    money_rule = ("A copay estimate is not the final bill. Only a completed visit has a final balance. Wait times and "
                  "places in line are estimates, never promises; urgent patients can be seen first.")
    hint_notes = ("party: the provider as spoken (Dr. Patel, Patel, Dr. Lee). category: sick_visit, follow_up, "
                  "physical, or lab_work when the caller names the kind of visit. status: waiting when they are at "
                  "the clinic now; upcoming for a future appointment; done for a past visit; cancelled.")
    restricted_note = ("they ask for medical advice, test results, a diagnosis, or a prescription. Only their care team "
                       "can answer those.")
    restricted_reply = ("I can't answer medical questions or share results here. Your care team can, through the "
                        "patient portal or during your visit.")
    emergency_note = "chest pain, trouble breathing, fainting, severe bleeding, or any emergency"
    emergency_reply = ("If this is an emergency, call 911 now or tell the front desk right away. I'm alerting the "
                       "front desk.")
    in_scope_words = (r"visit|appointment|wait\w*|turn|line|doctor|dr|patel|lee|nurse|copay|bill\w*|owe|insurance|"
                      r"card|portal|checked|check|room|front desk|daughter|son|father|mother|dad|mom|longer|next|"
                      r"physical|lab|pay|paid|cost|price")
    examples = [
        ("Still waiting", "Hi, this is Hannah Kim, DOB 7/19/1988, insurance ID ends in 5530. I'm in the waiting room "
                          "for my 2:30 today. When's the doctor coming?"),
        ("Bill question", "Hannah Kim, born 1988-07-19, member ID last four 5530. Why do I owe $60 for my "
                          "follow up with Dr. Patel?"),
        ("Daughter calling", "Hi, I'm Grace Ellison, calling for my father Robert Ellison. His date of birth is "
                             "1954-03-08 and his insurance ID ends in 2291."),
        ("Medical question", "Hannah Kim, DOB 1988-07-19, member ID 5530. Can you tell me my lab results?"),
        ("Name order", "This is Lan Nguyen, born October 2nd 1979, email nguyen.lan@example.org"),
        ("Off topic", "What's the weather like today?"),
    ]
    re_restricted = r"\b(results?|diagnos\w*|prescri\w*|medication|should i take|is it serious)\b"
    re_emergency = r"chest pain|can'?t breathe|trouble breathing|faint\w*|bleeding|emergency"

    def label(self, r: Record) -> str:
        return (f"{r.record_id}: {CATEGORY_WORDS[r.category]} with {r.counterparty} on {long_date(r.date)} at "
                f"{clock(r.scheduled_time)}, {self.status_words[r.status]}")

    def sections(self, r: Record, today: date, live: Optional[dict]) -> dict[str, list[str]]:
        status = (live or {}).get("status") or r.status
        overview = [f"Visit {r.record_id} is a {CATEGORY_WORDS[r.category]} with {r.counterparty} on {r.date} at "
                    f"{clock(r.scheduled_time)}.",
                    f"Current status: {self.status_words.get(status, status)}."]
        if r.check_in_time:
            overview.append(f"The patient checked in at {clock(r.check_in_time)}.")

        queue: list[str] = []
        if live:
            pos = live.get("position")
            if status == "roomed":
                queue.append("The patient is in an exam room.")
            elif pos:
                queue.append(f"The patient is number {pos} in line for {r.counterparty}.")
            if live.get("est_minutes"):
                queue.append(f"The front desk estimates about {live['est_minutes']} minutes. This is an estimate, "
                             "not a promise; urgent patients can be seen first.")
            if live.get("note"):
                queue.append(live["note"])

        bill: list[str] = []
        if r.bill:
            b = r.bill
            bill.append(f"The visit was billed {usd(b['billed'])}; insurance paid {usd(b['insurance_paid'])}; "
                        f"the patient balance is {usd(b['patient_owes'])}.")
            bill.append(f"Reason: {b['reason']}.")
        elif money(r.copay_estimate) > 0:
            bill.append(f"The estimated copay is {usd(r.copay_estimate)}. The final amount depends on what is billed.")
        else:
            bill.append("No copay is expected for this visit. Extra services could still be billed.")
        return {"overview": overview, "queue": queue, "bill": bill}

    def sections_for(self, intent: Optional[str], status: str) -> list[str]:
        if intent == "wait_time":
            return ["queue"] if status in ("checked_in", "roomed") else ["overview"]
        if intent == "bill_question":
            return ["bill"]
        if intent == "appointment_time":
            return ["overview"]
        if status in ("checked_in", "roomed"):
            return ["queue"]
        if status == "completed":
            return ["bill"]
        return ["overview"]

    def offline_hints(self, t: str, tl: str, phase: str, today: date) -> dict:
        out: dict = {}
        m = self.id_regex.search(t)
        out["record_id"] = m.group(0) if m else None
        out["party"] = self.party_hint(tl)
        if re.search(r"\bphysical\b", tl):
            out["category"] = "physical"
        elif re.search(r"\bfollow ?up\b", tl):
            out["category"] = "follow_up"
        elif re.search(r"\blab\b", tl) and "results" not in tl:
            out["category"] = "lab_work"
        if re.search(r"\b(waiting room|i'?m here|checked in|in the lobby)\b", tl):
            out["record_status"] = "waiting"
        elif re.search(r"\b(next week|upcoming|coming up)\b", tl):
            out["record_status"] = "upcoming"
        out["date"] = self.day_hint(tl, today)

        if re.search(r"how (much )?longer|when'?s the doctor|when is the doctor|my turn|in line|how long .{0,20}wait|"
                     r"still waiting|am i next|when .{0,20}(seen|called)", tl):
            out["intent"] = "wait_time"
        elif re.search(r"\b(owe|bill\w*|copay|charged|balance|pay|cost)\b", tl):
            out["intent"] = "bill_question"
        elif re.search(r"\b(upload|send|photo|card)\b", tl) and phase == "PROCESS_CASE":
            out["intent"] = "document_submission"
        elif re.search(r"\bwhen is my appointment\b|\bwhat time\b", tl):
            out["intent"] = "appointment_time"
        return out
