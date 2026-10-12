"""Freight: a driver calling the carrier about a load, a dock wait, detention, or settlement pay."""
from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from typing import Optional

from ..data import Record, clock, long_date, usd
from ..domain import Domain

DOCK_WORDS = {"waiting_for_door": "waiting for a door", "door_assigned": "door assigned", "loading": "loading",
              "unloading": "unloading", "loaded": "loaded", "unloaded": "unloaded"}
CATEGORY_WORDS = {"reefer": "reefer", "dry_van": "dry van", "flatbed": "flatbed"}


class Freight(Domain):
    key = "freight"
    org = "AKT driver support"
    opening = ("Thanks for calling AKT driver support. I can help with a load, a wait at the dock, "
               "detention, or your settlement. How can I help?")
    record_noun = "load"
    id_prefix = "LD"
    person_noun = "driver"
    id_last4_label = "last four digits of your CDL number"
    account_label = "driver ID"
    account_example = "DRV-2201"
    consent_channel = "the AKT driver app"
    team = "a dispatcher"
    privacy_reason = "Load and pay details are private to the driver"
    categories = {"reefer": "refrigerated trailer", "dry_van": "dry van", "flatbed": "flatbed"}
    status_groups = {"at_dock": {"at_facility"}, "on_the_road": {"dispatched", "in_transit"},
                     "delivered": {"delivered"}, "paid": {"settled"}}
    status_words = {"at_facility": "at the facility", "dispatched": "dispatched", "in_transit": "in transit",
                    "delivered": "delivered, not yet settled", "settled": "settled and paid",
                    **DOCK_WORDS}
    intents = {
        "dock_status": "how much longer at the dock, which door, when loading or unloading will finish",
        "detention_question": "detention pay for time spent waiting",
        "pay_question": "settlement or net pay, or why money was deducted (docked) from their pay",
        "appointment_question": "the appointment time or check in",
        "document_submission": "sending paperwork such as a bill of lading or a lumper receipt",
        "general_load_question": "anything else about a load",
    }
    live_intents = {"dock_status"}
    summary_sections = ["detention", "pay"]
    money_rule = ("Pay is final only on a settled load. Detention is paid only after it is approved, so never promise "
                  "it; a claim under review is not money owed. Dock wait times are the facility's estimates, never "
                  "promises.")
    hint_notes = ("party: the shipper, receiver, or city as spoken (Peachtree, the paper mill, Savannah). "
                  "category: reefer, dry_van, or flatbed when the caller names the trailer. "
                  "status: at_dock when they are at or waiting at a facility now; on_the_road; delivered; paid for a "
                  "load that already paid out. Docked pay means money deducted, not a loading dock.")
    restricted_note = ("they ask you to approve detention, change their pay or rate, waive a deduction, or give them a "
                       "different load. Only settlements or dispatch can do those.")
    restricted_reply = ("I can't approve pay changes or reassign loads from here. Settlements and dispatch make those "
                        "calls, and I can transfer you.")
    emergency_note = "a crash, an injury, a fire, a spill, or anyone in danger"
    emergency_reply = "If anyone is hurt or in danger, call 911 now. I'm connecting you to a dispatcher right away."
    in_scope_words = (r"load|dock|door|detention|settle\w*|pay|paid|paycheck|deduct\w*|docked|lumper|bol|lading|shipper|"
                      r"receiver|appointment|trailer|reefer|flatbed|dispatch\w*|truck|wait\w*|fuel|advance|paperwork|"
                      r"receipt|longer|loaded|unloaded|check\w*|wife|husband|spouse")
    examples = [
        ("Sitting at the dock", "Hey, this is Tony Alvarez, driver ID DRV-2201. I'm sitting at Peachtree in Savannah "
                                "on today's load. How much longer? DOB 9/14/1981, CDL ends in 4417."),
        ("Pay got docked", "This is Tony Alvarez, born 1981-09-14, CDL last four 4417. Why did you dock $180 from my "
                           "pay on the Lakeside load?"),
        ("Detention promise", "Tony Alvarez, DOB 1981-09-14, CDL 4417. Am I getting the $150 detention on the "
                              "paper mill load?"),
        ("Spouse calling", "Hi, I'm Maria Alvarez, calling for my husband Tony Alvarez. His date of birth is "
                           "1981-09-14 and his CDL ends in 4417."),
        ("Accent misheard", "My name is Andre Nowak, born December 5th 1975, email andre.nowak@example.org"),
        ("Off topic", "Who won the game last night?"),
    ]
    re_restricted = r"approve (my |the )?detention|waive|change my (pay|rate)|different load|give me (a |another )load"
    re_emergency = r"\b(accident|crash\w*|injur\w*|hurt|fire|spill\w*)\b"

    def label(self, r: Record) -> str:
        return (f"{r.record_id}: {CATEGORY_WORDS[r.category]} {r.stop_type} at {r.counterparty}, {r.location} "
                f"on {long_date(r.date)}, {self.status_words[r.status]}")

    def sections(self, r: Record, today: date, live: Optional[dict]) -> dict[str, list[str]]:
        status = (live or {}).get("status") or r.status
        overview = [f"Load {r.record_id} is a {CATEGORY_WORDS[r.category]} {r.stop_type} at {r.counterparty} in "
                    f"{r.location} on {r.date}.",
                    f"Current status: {self.status_words.get(status, status)}."]
        appt = [f"The appointment was {clock(r.appointment_time)} and the driver checked in at {clock(r.check_in_time)}."]

        dock: list[str] = []
        if live:
            s = f"Latest dock update: {DOCK_WORDS.get(live['status'], live['status'])}"
            s += f", door {live['door']}." if live.get("door") else "."
            dock.append(s)
            if live.get("est_minutes"):
                dock.append(f"The facility estimates about {live['est_minutes']} more minutes. "
                            "This is the facility's estimate, not a promise.")
            if live.get("note"):
                dock.append(live["note"])

        det = r.detention or {}
        start = datetime.strptime(r.appointment_time, "%H:%M") + timedelta(hours=r.free_time_hours)
        rule = (f"Free time is {r.free_time_hours} hours from the appointment, so detention would start at "
                f"{clock(start.strftime('%H:%M'))} and pay {usd(r.detention_rate)} per hour after that.")
        detention = {
            "not_started": [rule, "Detention is paid only after the bill of lading shows signed in and out times "
                                  "and settlements approves the claim."],
            "pending_review": [f"A detention claim for {det.get('hours')} hours ({usd(det.get('amount', 0))}) is "
                               "under review. It is not approved, so it is not money owed yet."],
            "approved": [f"Detention of {det.get('hours')} hours was approved for {usd(det.get('amount', 0))} and is "
                         "included in the settlement."],
            "denied": [f"Detention was denied because {det.get('reason')}."],
            "none": ["No detention was claimed on this load."],
        }.get(det.get("status", "not_started"), [rule])

        pay: list[str] = []
        st = r.settlement
        if st:
            pay.append(f"Settlement gross pay was {usd(st['gross'])}.")
            for d in st["deductions"]:
                pay.append(f"{usd(d['amount'])} was deducted for the {d['label']} because {d['reason']}.")
            if not st["deductions"]:
                pay.append("Nothing was deducted.")
            pay.append(f"Net pay was {usd(st['net'])}.")
            if "lumper receipt" in r.documents_needed and any(d["label"] == "lumper fee" for d in st["deductions"]):
                pay.append("Turning in the lumper receipt lets settlements review the lumper fee deduction for "
                           "reimbursement. Reimbursement is not guaranteed.")
        else:
            pay.append("This load hasn't settled, so there is no final pay amount yet.")
        return {"overview": overview, "appointment": appt, "dock": dock, "detention": detention, "pay": pay}

    def sections_for(self, intent: Optional[str], status: str) -> list[str]:
        if intent == "dock_status":
            return ["dock"] if status in DOCK_WORDS or status == "at_facility" else ["overview"]
        if intent == "detention_question":
            return ["detention"]
        if intent == "pay_question":
            return ["pay", "deadline"]
        if intent == "appointment_question":
            return ["appointment"]
        if status in DOCK_WORDS or status == "at_facility":
            return ["dock"]
        if status == "settled":
            return ["pay", "deadline"]
        if status == "delivered":
            return ["detention", "pay"]
        return ["overview"]

    def private_terms(self, r: Record) -> list[str]:
        return [r.counterparty, r.location]

    def offline_hints(self, t: str, tl: str, phase: str, today: date) -> dict:
        out: dict = {}
        m = self.id_regex.search(t)
        out["record_id"] = m.group(0) if m else None
        out["party"] = self.party_hint(tl)
        if "reefer" in tl:
            out["category"] = "reefer"
        elif "flatbed" in tl:
            out["category"] = "flatbed"
        elif "dry van" in tl:
            out["category"] = "dry_van"
        if re.search(r"\b(sitting at|waiting at|at the dock|still here|i'?m at)\b", tl):
            out["record_status"] = "at_dock"
        elif re.search(r"\b(delivered|dropped)\b", tl):
            out["record_status"] = "delivered"
        out["date"] = self.day_hint(tl, today)

        docked_pay = re.search(r"\bdock(ed)?\b[^.?!]{0,30}\b(pay|check)\b|\bdeduct\w*|took [^.?!]{0,20} out of|"
                               r"short on my (pay|check)", tl)
        if re.search(r"how (much )?longer|which door|what door|still waiting|when .{0,20}(loaded|unloaded|done)|"
                     r"how long .{0,20}(wait|dock|here)", tl) and not docked_pay:
            out["intent"] = "dock_status"
        elif "detention" in tl:
            out["intent"] = "detention_question"
        elif docked_pay or re.search(r"\b(settlement|net pay|paycheck|my pay|got paid)\b", tl):
            out["intent"] = "pay_question"
        elif re.search(r"\b(upload|send|photo|scan)\b", tl) and phase == "PROCESS_CASE":
            out["intent"] = "document_submission"
        elif "appointment" in tl:
            out["intent"] = "appointment_question"
        return out
