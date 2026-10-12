"""Portfolio: a client asking why the manager's market value differs from the custodian bank's.

The custodian holds the client's assets and keeps the official book of record.
The manager keeps its own book for trading and reporting. Operations reconciles
the two every day; a difference is a break to explain, not money gone missing.
"""
from __future__ import annotations

import re
from datetime import date
from decimal import Decimal
from typing import Optional

from ..data import Record, long_date, usd
from ..domain import Domain

CATEGORY_WORDS = {"balanced": "balanced", "core_fixed_income": "core fixed income",
                  "large_cap_equity": "large cap equity"}
LIVE_WORDS = {"under_research": "under research", "partially_matched": "partially matched", "matched": "matched"}
MANAGER = "Northbridge"


class Portfolio(Domain):
    key = "portfolio"
    org = "Northbridge Capital client service"
    opening = ("Thanks for contacting Northbridge Capital client service. I can help with your account values and how "
               "they reconcile with your custodian. How can I help?")
    record_noun = "reconciliation"
    id_prefix = "REC"
    person_noun = "client"
    id_last4_label = "last four digits of your custodian account number"
    account_label = "Northbridge account number"
    account_example = "NB-48210"
    consent_channel = "the Northbridge client portal"
    team = "your relationship manager"
    privacy_reason = "Account values and holdings are private financial information"
    categories = {"balanced": "balanced account", "core_fixed_income": "bond account",
                  "large_cap_equity": "stock account"}
    status_groups = {"open": {"open_breaks", "awaiting_statement"}, "matched": {"matched", "resolved"}}
    status_words = {"open_breaks": "open breaks", "awaiting_statement": "waiting on the custodian statement",
                    "matched": "matched", "resolved": "resolved and matched", **LIVE_WORDS}
    intents = {
        "difference_question": "why Northbridge's value and the custodian's value differ, or what a break is",
        "status_update": "whether the difference is fixed yet, or the latest progress",
        "value_question": "the account's market value",
        "document_submission": "sending documents such as a custodian statement",
        "general_account_question": "anything else about the account's reconciliation",
    }
    live_intents = {"status_update"}
    summary_sections = ["values", "breaks"]
    money_rule = ("A difference between Northbridge's value and the custodian's value is a reconciliation break to "
                  "explain, not a loss, a gain, or money owed to or from the client. The custodian holds the assets and "
                  "its records are the official book of record. Never promise when a break will clear, and never say "
                  "which side is right beyond what the record says.")
    hint_notes = ("party: the custodian bank or the account as spoken (Harbor, the trust, my IRA, the joint account). "
                  "status: open when they say it is still off, not fixed, or broken; matched when they mean one that "
                  "already matches. date: the reconciliation day, so yesterday's numbers or month end become a date "
                  "or a month.")
    restricted_note = ("they ask for investment advice, ask you to buy, sell, or rebalance, or ask you to move or wire "
                       "money. Only their relationship manager can act on those, after confirming them directly.")
    restricted_reply = ("I can't give investment advice or take trade or money movement instructions here. Your "
                        "relationship manager can help with those, and I can transfer you.")
    emergency_note = "suspected fraud: a transfer or trade they didn't make, a hacked account, or money moved without permission"
    emergency_reply = ("If you think there's unauthorized activity, call the fraud line on your custodian statement now. "
                       "I'm connecting you to your relationship manager right away.")
    in_scope_words = (r"account|portfolio|value|market|custodian|statement|reconcil\w*|break\w*|differ\w*|match\w*|"
                      r"trust|ira|harbor|summit|northbridge|holding\w*|trade\w*|settle\w*|interest|accru\w*|bond|"
                      r"stock|price\w*|split|advisor|off|gap|report|fixed|update|resolved|yet|credit\w*|lost|refund\w*")
    examples = [
        ("Values don't match", "Hi, this is Linda Park, account NB-48210, DOB 5/11/1963, custodian account ends in 6042. "
                               "Yesterday's value for my trust account doesn't match my Harbor statement. Why?"),
        ("Is it fixed yet", "Linda Park, born 1963-05-11, custodian account last four 6042. Is the difference on my "
                            "trust account fixed yet?"),
        ("Pricing gap", "This is Seun Adeyemi, DOB 1977-02-14, custodian account ends in 7735. Why is my joint "
                        "account off by $3,180?"),
        ("Advisor calling", "Hi, I'm Michael Chen, calling for my client Linda Park. Her date of birth is 1963-05-11 "
                            "and her custodian account ends in 6042."),
        ("Missing statement", "Linda Park, DOB 1963-05-11, custodian account 6042. Why isn't my IRA reconciled for "
                              "February?"),
        ("Trade request", "Linda Park, DOB 1963-05-11, custodian account 6042. Sell my bond fund and wire the cash to me."),
    ]
    re_restricted = r"should i (buy|sell)|\bsell my\b|\bbuy more\b|\bwire\b|transfer (the )?(money|funds|cash)|move (my )?money|rebalance"
    re_emergency = r"unauthori[sz]ed|didn'?t make|\bfraud\b|stolen|hacked|someone (took|moved)"

    def label(self, r: Record) -> str:
        return (f"{r.record_id}: {r.location} ({CATEGORY_WORDS[r.category]}) at {r.counterparty} as of "
                f"{long_date(r.date)}, {self.status_words[r.status]}")

    def sections(self, r: Record, today: date, live: Optional[dict]) -> dict[str, list[str]]:
        status = (live or {}).get("status") or r.status
        acct = r.location if r.location.lower().endswith("account") else f"{r.location} account"
        overview = [f"Reconciliation {r.record_id} compares the {acct} between {MANAGER} and "
                    f"{r.counterparty}, the custodian, as of {r.date}.",
                    f"Current status: {self.status_words.get(status, status)}."]
        custody = (f"{r.counterparty} holds the assets, and its records are the official book of record. "
                   f"{MANAGER} keeps its own book for trading and reporting.")
        values: list[str] = []
        if r.custodian_mv is None:
            values.append(f"{MANAGER}'s market value is {usd(r.manager_mv)}. The custodian's file for this date "
                          "has not arrived, so the two sides can't be compared yet.")
        else:
            diff = Decimal(r.manager_mv) - Decimal(r.custodian_mv)
            if diff == 0:
                values.append(f"Both sides show a market value of {usd(r.manager_mv)}.")
            else:
                values.append(f"{MANAGER}'s market value is {usd(r.manager_mv)} and {r.counterparty}'s is "
                              f"{usd(r.custodian_mv)}, a difference of {usd(abs(diff))}.")
                values.append("A difference is a reconciliation break to explain. It is not missing money, and it "
                              "does not mean the holdings changed.")
        values.append(custody)

        breaks: list[str] = []
        for b in r.breaks:
            verb = "was resolved" if b["state"] == "resolved" else "is open"
            breaks.append(f"The {b['type']} item of {usd(b['amount'])} {verb}: {b['detail']}.")

        progress: list[str] = []
        if live:
            progress.append(f"Latest update: {LIVE_WORDS.get(live['status'], live['status'])}.")
            if Decimal(live.get("remaining", "0")) > 0:
                progress.append(f"{usd(live['remaining'])} remains unmatched. Operations can't promise when it "
                                "will clear.")
            if live.get("note"):
                progress.append(live["note"])
        return {"overview": overview, "values": values, "breaks": breaks, "progress": progress}

    def sections_for(self, intent: Optional[str], status: str) -> list[str]:
        if status == "awaiting_statement" and intent != "document_submission":
            return ["values", "deadline"]
        if intent == "status_update":
            return ["progress"] if status in LIVE_WORDS or status == "open_breaks" else ["overview", "values"]
        if intent == "value_question":
            return ["values"]
        if intent == "difference_question":
            return ["values", "breaks"]
        if status in ("open_breaks", "resolved"):
            return ["values", "breaks"]
        return ["values"]

    def private_terms(self, r: Record) -> list[str]:
        return [r.counterparty, r.location]

    def offline_hints(self, t: str, tl: str, phase: str, today: date) -> dict:
        out: dict = {}
        m = self.id_regex.search(t)
        out["record_id"] = m.group(0) if m else None
        out["party"] = self.party_hint(tl)
        if re.search(r"\b(still off|not fixed|still open|still different|doesn'?t match|don'?t match)\b", tl):
            out["record_status"] = "open"
        out["date"] = self.day_hint(tl, today)
        if re.search(r"\bmonth end\b", tl):
            out["month"] = 2
        if re.search(r"fixed yet|resolved yet|any update|still (off|open|different)|matched yet|update on|"
                     r"be fixed|fixed by|when will it (clear|match|be)", tl):
            out["intent"] = "status_update"
        elif re.search(r"differ\w*|don'?t match|doesn'?t match|\boff by\b|\bgap\b|mismatch|why .{0,30}(lower|higher)|"
                       r"reconciled|\blost\b|\bcredit\b|refund", tl):
            out["intent"] = "difference_question"
        elif re.search(r"market value|how much is|worth", tl):
            out["intent"] = "value_question"
        elif re.search(r"\b(upload|send|statement)\b", tl) and phase == "PROCESS_CASE":
            out["intent"] = "document_submission"
        return out
