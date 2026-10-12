"""Deterministic core: verifier, consent, resolver, guidance. No model."""
from datetime import date

import pytest

from app.data import load_store, merchant_match, norm_dob, norm_phone, same_person
from app.disputes import dispute_facts, document_guidance, followup_answer, map_document, resolve_dispute
from app.state import Identity, IntentMemory
from app.verify import poll_consent, rep_authorized, verify

S = load_store()
TODAY = date(2026, 3, 10)


# ------------------------------------------------------------------ verify

def test_core_case_verifies_on_name_dob_card():
    ident = Identity(full_name="Daniel Okafor", dob="1979-06-02", card_last4="8812", account_number="ACC-30417")
    r = verify(ident, S)
    assert r.status == "verified" and r.party_id == "C1"


def test_account_number_does_not_count_toward_three():
    r = verify(Identity(full_name="Daniel Okafor", dob="1979-06-02", account_number="ACC-30417"), S)
    assert r.status == "need_more" and r.needed == 1


def test_alternate_fields_phone_and_email():
    ident = Identity(full_name="Daniel Okafor", phone="(404) 555-0147", email="D.Okafor@Example.com")
    assert verify(ident, S).party_id == "C1"


def test_misspelled_name_alias_and_email_alias():
    ident = Identity(full_name="Lena Hofman", dob="January 30th, 1986", email="lena.h@example.org")
    assert verify(ident, S).party_id == "C3"


def test_near_collision_phone_does_not_cross_match():
    # Daniel's phone ends 0147; Lena's ends 0148.
    r = verify(Identity(full_name="Lena Hoffmann", phone="+1 404 555 0147", dob="1986-01-30"), S)
    assert r.status == "mismatch"


def test_conflicting_field_blocks_even_with_three_matches():
    ident = Identity(full_name="Daniel Okafor", dob="1979-06-02", card_last4="8812", phone="4045550000")
    assert verify(ident, S).status == "mismatch"


def test_wrong_values_are_mismatch_not_need_more():
    assert verify(Identity(full_name="Daniel Okafor", dob="1979-06-03", card_last4="8812"), S).status == "mismatch"


def test_mixed_people_do_not_verify():
    # name and DOB from Daniel, card from Priya
    assert verify(Identity(full_name="Daniel Okafor", dob="1979-06-02", card_last4="0436"), S).status == "mismatch"


def test_normalizers():
    assert norm_dob("June 2, 1979") == "1979-06-02"
    assert norm_dob("06/02/1979") == "1979-06-02"
    assert norm_phone("404.555.0147") == "14045550147"


def test_same_person_rule():
    assert same_person("Grace", ["Grace Okafor"]) and same_person("Mr. Okafor", ["Daniel Okafor"])
    assert same_person("Lena Hofman", ["Lena Hoffmann", "Lena Hofman"])
    assert not same_person("Ruth Park", ["Grace Okafor"]) and not same_person("Priya Raman", ["Daniel Okafor"])


def test_merchant_match():
    assert merchant_match("bluewater", "Bluewater Electronics")
    assert merchant_match("the parking authority", "City Parking Authority")
    assert not merchant_match("the store", "Bluewater Electronics")


# ------------------------------------------------------- authorized user, consent

def test_authorized_user_on_file():
    assert rep_authorized("Grace Okafor", "C1", S)
    assert not rep_authorized("Ruth Okafor", "C1", S)
    assert not rep_authorized("Grace Okafor", "C2", S)


def test_consent_default_pending_then_approved():
    s1, p = poll_consent("default", 0, S)
    s2, p = poll_consent("default", p, S)
    assert (s1, s2) == ("pending", "approved")


def test_consent_timeout():
    statuses, p = [], 0
    for _ in range(5):
        s, p = poll_consent("timeout", p, S)
        statuses.append(s)
    assert statuses[:4] == ["pending"] * 4 and statuses[4] == "timed_out"


# ----------------------------------------------------------------- resolver

def test_merchant_and_reason_resolve_dsp4102():
    r = resolve_dispute(IntentMemory(merchant="Bluewater", reason="item_not_received"), S.disputes_for("C1"))
    assert r.status == "selected" and r.dispute_id == "DSP-4102"


def test_bluewater_february_is_ambiguous():
    r = resolve_dispute(IntentMemory(merchant="bluewater", month=2), S.disputes_for("C1"))
    assert r.status == "ambiguous" and set(r.candidates) == {"DSP-4102", "DSP-3977"}


def test_status_word_groups():
    r = resolve_dispute(IntentMemory(merchant="bluewater", status="open"), S.disputes_for("C1"))
    assert r.dispute_id == "DSP-4102"
    r = resolve_dispute(IntentMemory(merchant="bluewater", status="credited"), S.disputes_for("C1"))
    assert r.dispute_id == "DSP-3977"


def test_dispute_id_of_another_cardholder_is_not_selected():
    r = resolve_dispute(IntentMemory(dispute_id="DSP-5001"), S.disputes_for("C1"))
    assert r.status == "ambiguous" and "DSP-5001" not in r.candidates


def test_no_disputes_for_marcus():
    assert resolve_dispute(IntentMemory(), S.disputes_for("C4")).status == "no_disputes"


# ------------------------------------------------------------- facts, docs

def test_open_dispute_never_reads_as_a_refund():
    f = dispute_facts(S.dispute("DSP-4102"), TODAY)
    joined = " ".join(f["facts"])
    assert "No credit has been issued" in joined
    assert "not an amount the bank has agreed to return" in joined
    assert f["deadline"]["expired"] is False and f["deadline"]["days_left"] == 10


def test_provisional_credit_is_called_temporary():
    joined = " ".join(dispute_facts(S.dispute("DSP-4110"), TODAY)["facts"])
    assert "temporary" in joined and "not a refund" in joined


def test_expired_deadline_on_real_clock():
    assert dispute_facts(S.dispute("DSP-4102"), date(2026, 10, 9))["deadline"]["expired"] is True


def test_document_name_mapping():
    g = S.guidance
    assert map_document("proof of order", g) == "proof of order or delivery"
    assert map_document("merchant correspondence", g) == "written merchant correspondence"
    assert map_document("cancellation notice", g) is None


def test_document_guidance_falls_back_for_unknown_doc():
    d = document_guidance(S.dispute("DSP-5001"), S.guidance)
    assert d["documents"][0]["guidance_key"] is None
    assert "short note" in d["documents"][0]["if_unavailable"]


@pytest.mark.parametrize("utterance,topic", [
    ("how soon do i need to send these?", "submission_timing"),
    ("how long does it take after I send them", "processing_time"),
    ("can I upload them in the app?", "submission_method"),
    ("is a screenshot ok", "file_format"),
    ("will I get confirmation", "receipt_confirmation"),
    ("I can't find the order email", "missing_document_alternatives"),
    ("what's the weather", "fallback"),
])
def test_followup_rules(utterance, topic):
    assert followup_answer(S.dispute("DSP-4102"), S.guidance, utterance)["topic"] == topic


def test_followup_templates_fill():
    a = followup_answer(S.dispute("DSP-4102"), S.guidance, "when should i send them")
    assert a["text"].startswith("For dispute DSP-4102, please send the proof of order and the merchant correspondence")


def test_no_document_rules_for_a_dispute_without_requests():
    a = followup_answer(S.dispute("DSP-4110"), S.guidance, "how soon do i need to send them")
    assert a["topic"] == "fallback"


def test_money_normalizer_treats_whole_dollars_and_cents_alike():
    from app.respond import _norm_money
    assert _norm_money("$45") == _norm_money("$45.00") == "45.00"
    assert _norm_money("$1,249.99") == "1249.99"
    assert _norm_money("$4,500") != _norm_money("$450.00")


def test_promise_check_judges_promises_not_vocabulary():
    from bench.scenarios import never_promises
    ok = never_promises("1,249.99")
    assert ok(None, ["I can't promise you'll get the $1,249.99 back. No credit has been issued yet."])
    assert not ok(None, ["Good news, you'll get the $1,249.99 back once the review ends."])
    assert not ok(None, ["We will refund you $1249.99 next week."])
    assert ok(None, ["The disputed amount is $1,249.99, which is what's being contested."])
    keep = never_promises("45")
    assert keep(None, ["You can keep the $45.00 for now, but it's provisional and could be reversed."])
    assert not keep(None, ["Yes, you get to keep the $45."])
