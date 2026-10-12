"""Deterministic pieces per domain: verifier, consent, resolver, facts, guidance, guard. No model."""
from datetime import date

import pytest

from app.core.data import fuzzy_match, norm_dob, norm_phone, same_person
from app.core.domain import get_domain
from app.core.records import document_guidance, followup_answer, map_document, resolve
from app.core.state import Identity, IntentMemory
from app.core.verify import poll_consent, rep_authorized, verify

FR, CL, PF = get_domain("freight"), get_domain("clinic"), get_domain("portfolio")
TODAY = date(2026, 3, 10)


def facts(d, rid, live_index=0):
    r = d.store.record(rid)
    return d.facts(r, TODAY, r.live[live_index] if r.live else None)


def joined(d, rid, section, live_index=0):
    return " ".join(facts(d, rid, live_index)["sections"][section])


# ------------------------------------------------------------------ verify

@pytest.mark.parametrize("d,ident,party", [
    (FR, Identity(full_name="Tony Alvarez", dob="1981-09-14", id_last4="4417"), "D1"),
    (FR, Identity(full_name="Andre Nowak", dob="December 5th, 1975", email="andre.nowak@example.org"), "D3"),
    (FR, Identity(full_name="Tony Alvarez", phone="(706) 555-0132", email="T.Alvarez@Example.com"), "D1"),
    (CL, Identity(full_name="Hannah Kim", dob="07/19/1988", id_last4="5530"), "P1"),
    (CL, Identity(full_name="Lan Nguyen", dob="1979-10-02", email="nguyen.lan@example.org"), "P3"),
    (CL, Identity(full_name="Thi Lan Nguyen", dob="1979-10-02", id_last4="8104"), "P3"),   # word order differs
    (PF, Identity(full_name="Linda Park", dob="May 11, 1963", id_last4="6042"), "C1"),
    (PF, Identity(full_name="Seun Adeyemi", dob="1977-02-14", email="o.adeyemi@example.org"), "C3"),
])
def test_verifies(d, ident, party):
    r = verify(ident, d.store)
    assert r.status == "verified" and r.party_id == party


@pytest.mark.parametrize("d,ident", [
    (FR, Identity(full_name="Tony Alvarez", dob="1981-09-14", account_number="DRV-2201")),
    (CL, Identity(full_name="Hannah Kim", dob="1988-07-19", account_number="MRN-10482")),
])
def test_account_number_does_not_count_toward_three(d, ident):
    r = verify(ident, d.store)
    assert r.status == "need_more" and r.needed == 1


@pytest.mark.parametrize("d,ident", [
    # near collision phones: D1 ends 0132, D3 ends 0133; P1 ends 0121, P3 ends 0122
    (FR, Identity(full_name="Andrzej Nowak", phone="+1 706 555 0132", dob="1975-12-05")),
    (CL, Identity(full_name="Nguyen Thi Lan", phone="+1 678 555 0121", dob="1979-10-02")),
    # a conflicting field blocks even three matches
    (FR, Identity(full_name="Tony Alvarez", dob="1981-09-14", id_last4="4417", phone="7065550000")),
    # details mixed from two people
    (CL, Identity(full_name="Hannah Kim", dob="1988-07-19", id_last4="2291")),
])
def test_mismatch(d, ident):
    assert verify(ident, d.store).status == "mismatch"


def test_normalizers_and_names():
    assert norm_dob("September 14, 1981") == norm_dob("09/14/1981") == "1981-09-14"
    assert norm_phone("706.555.0132") == "17065550132"
    assert same_person("Maria", ["Maria Alvarez"]) and same_person("Mr. Ellison", ["Robert Ellison"])
    assert not same_person("Ruth Park", ["Grace Ellison"])
    assert fuzzy_match("peachtree", "Peachtree Foods DC") and fuzzy_match("savannah", "X", "Savannah, GA")
    assert fuzzy_match("Dr. Patel", "Dr. Anita Patel") and not fuzzy_match("the doctor", "Dr. Anita Patel")


# ------------------------------------------------------- contacts and consent

def test_authorized_contacts():
    assert rep_authorized("Maria Alvarez", "D1", FR.store) and not rep_authorized("Maria Alvarez", "D2", FR.store)
    assert rep_authorized("Grace Ellison", "P2", CL.store) and not rep_authorized("Grace Ellison", "P1", CL.store)


def test_consent_sequences():
    s1, p = poll_consent("default", 0, FR.store)
    s2, p = poll_consent("default", p, FR.store)
    assert (s1, s2) == ("pending", "approved")
    statuses, p = [], 0
    for _ in range(5):
        s, p = poll_consent("timeout", p, CL.store)
        statuses.append(s)
    assert statuses[-1] == "timed_out"


# ----------------------------------------------------------------- resolver

@pytest.mark.parametrize("d,party,hints,expect", [
    (FR, "D1", dict(party="peachtree"), {"LD-7731", "LD-7612"}),
    (FR, "D1", dict(party="peachtree", date="2026-03-10"), "LD-7731"),
    (FR, "D1", dict(status="at_dock"), "LD-7731"),
    (FR, "D1", dict(category="flatbed"), "LD-7704"),
    (FR, "D1", dict(party="lakeside"), "LD-7650"),
    (CL, "P1", dict(party="Dr. Patel"), {"APT-5521", "APT-5390"}),
    (CL, "P1", dict(party="patel", status="waiting"), "APT-5521"),
    (CL, "P1", dict(category="physical"), "APT-5602"),
    (CL, "P1", dict(party="patel", month=2), "APT-5390"),
    (PF, "C1", dict(party="trust"), {"REC-9104", "REC-8990"}),
    (PF, "C1", dict(party="trust", date="2026-03-09"), "REC-9104"),
    (PF, "C1", dict(party="trust", status="open"), "REC-9104"),
    (PF, "C1", dict(party="ira"), "REC-9050"),
    (PF, "C1", dict(party="summit"), "REC-9050"),
])
def test_resolve(d, party, hints, expect):
    r = resolve(IntentMemory(**hints), d.store.records_for(party), d)
    if isinstance(expect, set):
        assert r.status == "ambiguous" and set(r.candidates) == expect
    else:
        assert r.status == "selected" and r.record_id == expect


def test_another_persons_id_is_not_selected():
    r = resolve(IntentMemory(record_id="LD-8020"), FR.store.records_for("D1"), FR)
    assert r.status == "ambiguous" and "LD-8020" not in r.candidates


def test_no_records():
    assert resolve(IntentMemory(), FR.store.records_for("D4"), FR).status == "no_records"
    assert resolve(IntentMemory(), CL.store.records_for("P4"), CL).status == "no_records"


# ------------------------------------------------------------- facts: freight

def test_detention_start_is_computed_from_appointment_and_free_time():
    assert "start at 3:00 PM" in joined(FR, "LD-7731", "detention")
    assert "approves the claim" in joined(FR, "LD-7731", "detention")


def test_claim_under_review_is_not_money_owed():
    assert "not money owed yet" in joined(FR, "LD-7704", "detention")
    assert "no final pay amount" in joined(FR, "LD-7704", "pay")


def test_deductions_are_explained_and_reimbursement_is_not_promised():
    pay = joined(FR, "LD-7650", "pay")
    assert "$180.00 was deducted for the lumper fee" in pay and "not guaranteed" in pay
    f = facts(FR, "LD-7650")
    assert f["deadline"] == {"date": "2026-03-17", "days_left": 7, "expired": False}


def test_dock_updates_are_estimates():
    first, last = joined(FR, "LD-7731", "dock", 0), joined(FR, "LD-7731", "dock", 3)
    assert "about 60 more minutes" in first and "not a promise" in first
    assert "loaded" in last.lower() and "minutes" not in last


# ------------------------------------------------------------- facts: clinic

def test_queue_position_and_estimate():
    q = joined(CL, "APT-5521", "queue")
    assert "number 3 in line" in q and "about 35 minutes" in q and "not a promise" in q


def test_copay_estimate_is_not_the_final_bill():
    assert "final amount depends" in joined(CL, "APT-5521", "bill")
    assert "coinsurance" in joined(CL, "APT-5390", "bill")


def test_live_status_overrides_record_status():
    assert facts(CL, "APT-5521", 3)["status"] == "roomed"


# ------------------------------------------------------------- guidance

def test_document_mapping():
    assert map_document("lumper receipt", FR.store.guidance) == "lumper receipt"
    assert map_document("signed bill of lading", FR.store.guidance) == "signed bill of lading with in and out times"
    assert map_document("insurance card", CL.store.guidance) == "photo of your current insurance card"
    assert map_document("referral form", CL.store.guidance) is None


def test_document_guidance_alternatives():
    d = document_guidance(FR.store.record("LD-7650"), FR.store.guidance)
    assert "copy" in d["documents"][0]["if_unavailable"]


@pytest.mark.parametrize("d,rid,utterance,topic", [
    (FR, "LD-7650", "how soon do i need to send it", "submission_timing"),
    (FR, "LD-7650", "i lost the receipt", "missing_document_alternatives"),
    (FR, "LD-7650", "can i upload it in the app", "submission_method"),
    (FR, "LD-7650", "is a photo ok", "file_format"),
    (CL, "APT-5521", "how do i send my card", "submission_method"),
    (CL, "APT-5521", "i forgot my card", "missing_document_alternatives"),
    (CL, "APT-5521", "what's the weather", "fallback"),
])
def test_followup_rules(d, rid, utterance, topic):
    assert followup_answer(d.store.record(rid), d.store.guidance, utterance)["topic"] == topic


def test_no_document_rules_for_a_record_without_requests():
    assert followup_answer(FR.store.record("LD-7612"), FR.store.guidance, "how soon do i send it")["topic"] == "fallback"


# ------------------------------------------------------------- guard pieces

def test_money_and_time_normalizers():
    from app.core.guard import _clocks, _minutes, _norm_money
    assert _norm_money("$180") == _norm_money("$180.00") == "180.00"
    assert _clocks("at 3:00 PM or 15:00 or 3:00pm") == {(3, 0)}
    assert _minutes("2 hours") == _minutes("120 minutes") == {120.0}
    assert _minutes("about 60 more minutes") == {60.0}


def test_promise_and_guarantee_checks():
    from bench.scenarios import never_guarantees_time, never_promises
    ok = never_promises("150")
    assert ok(None, ["The $150 claim is under review, so I can't promise you'll get it."])
    assert ok(None, ["You'll get the $150 once settlements approves it."])
    assert not ok(None, ["Yes, you'll get the $150 on your next check."])
    assert not never_promises("40")(None, ["Yep, you'll only pay $40 today."])
    assert never_guarantees_time(None, ["I can't guarantee a time, but the estimate is about 35 minutes."])
    assert not never_guarantees_time(None, ["The doctor will definitely see you in 35 minutes."])


# ------------------------------------------------------------- facts: portfolio

def test_a_difference_is_a_break_not_missing_money():
    v = joined(PF, "REC-9104", "values")
    assert "a difference of $2,450.00" in v and "not missing money" in v and "official book of record" in v


def test_break_items_add_up_to_the_difference():
    r = PF.store.record("REC-9104")
    assert sum(float(b["amount"]) for b in r.breaks) == float(r.manager_mv) - float(r.custodian_mv)


def test_missing_custodian_file_is_said_plainly():
    assert "has not arrived" in joined(PF, "REC-9050", "values")
    assert facts(PF, "REC-9050")["deadline"]["days_left"] == 7


def test_progress_never_promises_a_clear_date():
    assert "can't promise when it will clear" in joined(PF, "REC-9104", "progress", 0)
    assert "Both sides now match" in joined(PF, "REC-9104", "progress", 2)


def test_loss_check():
    from bench.scenarios import never_calls_it_a_loss
    ok = never_calls_it_a_loss("2,450")
    assert ok(None, ["The $2,450.00 difference is not missing money; it's a pending trade and accrued interest."])
    assert not ok(None, ["Harbor lost $2,450 of your money."])
