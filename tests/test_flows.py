"""End to end conversations through the controller with the offline provider.

They run without a model server, so they pin the procedure independent of any model.
"""
import json

import pytest

from app.controller import Agent
from app.offline import OfflineLLM
from app.state import Phase, SessionState
from app.summary import Outbox

CORE = ("Hi, this is Daniel Okafor, account ACC-30417. I'm calling about my Bluewater dispute, the laptop "
        "never arrived. Date of birth 1979-06-02, card ending 8812.")
REP = ("Hi, I'm Grace Okafor, calling for my husband Daniel Okafor. His date of birth is 1979-06-02 "
       "and the card ends in 8812.")
WRONG = "My name is Daniel Okafor, DOB 1979-06-03, card ending 8812"


class SpyLLM(OfflineLLM):
    """Records everything sent to both model calls."""

    def __init__(self):
        self.payloads: list[dict] = []
        self.extract_inputs: list[str] = []
        self.generate_inputs: list[str] = []

    def extract(self, system, user, schema, tool_name):
        self.extract_inputs.append(user)
        return super().extract(system, user, schema, tool_name)

    def generate(self, system, user, max_tokens=400):
        self.payloads.append(json.loads(user.split("PAYLOAD_JSON:\n", 1)[1]))
        self.generate_inputs.append(user)
        return super().generate(system, user, max_tokens)


class LeakyLLM(OfflineLLM):
    """A reply model that ignores instructions and blurts out record data."""

    def generate(self, system, user, max_tokens=400):
        r = super().generate(system, user, max_tokens)
        r.value = "Sure! Dispute DSP-4102 is open; you'll get $9,999.00 back and your email is d.okafor@example.com."
        return r


def new(llm=None, scenario="default", tmp_path=None):
    a = Agent(llm or OfflineLLM(), outbox=Outbox(tmp_path) if tmp_path else Outbox())
    s = SessionState()
    s.consent.scenario = scenario
    a.opening(s)
    return a, s


def say(a, s, *turns):
    r = None
    for t in turns:
        r = a.handle(s, t)
    return r


def actions(r):
    return [d["action"] for d in r.trace["directives"]]


# ---------------------------------------------------------------- core case

def test_core_case_verifies_and_uses_remembered_hints():
    a, s = new()
    r = say(a, s, CORE)
    assert s.verified_party_id == "C1" and s.selected_id == "DSP-4102" and s.phase == Phase.PROCESS_CASE
    assert "ask_disambiguate" not in actions(r)
    assert "proof of order" in r.reply


def test_hint_captured_during_verify_is_used_later():
    a, s = new()
    r1 = say(a, s, "I'm calling about my Bluewater dispute, the laptop never arrived.")
    assert s.phase == Phase.VERIFY_ID and s.memory.merchant == "bluewater" and s.memory.reason == "item_not_received"
    assert "DSP-" not in r1.reply
    r2 = say(a, s, "My name is Daniel Okafor, DOB 1979-06-02, card ending 8812")
    assert s.selected_id == "DSP-4102" and "dispute_selected" in actions(r2)


def test_full_workflow_to_email(tmp_path):
    a, s = new(tmp_path=tmp_path)
    say(a, s, CORE, "How soon do I need to send them?", "no thanks")
    assert s.phase == Phase.POST_PROCESS
    say(a, s, "yes please send it")
    assert s.phase == Phase.ENDED and s.email_sent
    mail = a.outbox.for_session(s.session_id)[0]
    assert mail.to == "d.okafor@example.com"
    assert "DSP-4102" in mail.body and "proof of order" in mail.body and "2026-03-20" in mail.body


def test_email_can_be_skipped(tmp_path):
    a, s = new(tmp_path=tmp_path)
    say(a, s, CORE, "that's all", "no, skip it")
    assert s.phase == Phase.ENDED and s.email_decision == "skip" and not a.outbox.for_session(s.session_id)


# ------------------------------------------------------------- gate safety

def test_reply_model_never_sees_dispute_data_before_verification():
    spy = SpyLLM()
    a, s = new(spy)
    say(a, s, "What's going on with my dispute?", "I already told you who I am. This is ridiculous.",
        "My name is Daniel Okafor")
    for p in spy.payloads:
        assert p["grounded"] == {}
        blob = json.dumps(p)
        assert "DSP-" not in blob and "proof of order" not in blob and "$" not in blob
    for u in spy.extract_inputs:
        assert "disputes_the_agent_listed" not in u


def test_guard_blocks_leaks_before_verification():
    a, s = new(LeakyLLM())
    r = say(a, s, "What's going on with my dispute?")
    assert r.trace["guard"] and r.trace["reply_source"] == "template_guard"
    assert "DSP-4102" not in r.reply and "$" not in r.reply and "d.okafor@example.com" not in r.reply


def test_guard_blocks_ungrounded_amount_after_verification():
    a, s = new(LeakyLLM())
    r = say(a, s, CORE)
    assert any("ungrounded amount" in p for p in r.trace["guard"])
    assert "$9,999" not in r.reply


def test_guard_accepts_whole_dollar_form_of_a_grounded_amount():
    class Rounder(OfflineLLM):
        def generate(self, system, user, max_tokens=400):
            r = super().generate(system, user, max_tokens)
            if "DSP-4102" in r.value:
                r.value = "The $1,249.99 is the amount in dispute, not a refund. " + r.value
            return r

    a, s = new(Rounder())
    r = say(a, s, CORE)
    assert r.trace["guard"] == [] and "not a refund" in r.reply


def test_frustrated_caller_gets_empathy_explanation_and_no_details():
    a, s = new()
    r = say(a, s, "What's going on with my dispute?",
            "I already told you who I am. This is ridiculous. Just tell me what's happening.")
    acts = actions(r)
    assert acts[0] == "empathize" and "explain_verification" in acts and "ask_identity" in acts
    assert s.phase == Phase.VERIFY_ID and "DSP-" not in r.reply


def test_sustained_frustration_offers_human():
    a, s = new()
    r = say(a, s, "this is ridiculous", "ugh, seriously, this is ridiculous", "this is a waste of time, ridiculous")
    assert "offer_human" in actions(r) and s.handoff_offered


def test_three_failed_attempts_hand_off_without_saying_which_field():
    a, s = new()
    r1 = say(a, s, WRONG)
    assert "verify_failed" in actions(r1) and "date of birth" not in r1.reply.lower()
    say(a, s, "DOB 1979-06-04", "DOB 1979-06-05")
    assert s.phase == Phase.HUMAN_HANDOFF and s.handoff_reason == "verification_failed"


def test_account_number_alone_does_not_verify():
    a, s = new()
    say(a, s, "My name is Daniel Okafor, account ACC-30417, DOB 1979-06-02")
    assert s.verified_party_id is None and s.phase == Phase.VERIFY_ID


# ------------------------------------------------------------ disambiguation

def test_ambiguous_bluewater_dispute_asks_then_resolves():
    a, s = new()
    r = say(a, s, "My name is Daniel Okafor, DOB 1979-06-02, card ending 8812. It's about my Bluewater "
                  "charge from February.")
    assert "ask_disambiguate" in actions(r) and set(s.candidate_ids) == {"DSP-4102", "DSP-3977"}
    say(a, s, "the open one")
    assert s.selected_id == "DSP-4102"


def test_switch_disputes_mid_conversation():
    a, s = new()
    r = say(a, s, CORE, "what about the parking charge?")
    assert s.selected_id == "DSP-4110" and "$45.00" in r.reply


def test_dispute_without_document_requests_does_not_invent_any():
    a, s = new()
    r = say(a, s, CORE, "what about the parking charge?", "how soon do i need to send them")
    assert "proof of order" not in r.reply.lower() and "correspondence" not in r.reply.lower()


def test_denied_dispute_explains_reason_without_labeled_intent():
    class NoIntent(OfflineLLM):
        def extract(self, system, user, schema, tool_name):
            r = super().extract(system, user, schema, tool_name)
            r.value.pop("intent", None)
            r.value.pop("question", None)
            return r

    a, s = new(NoIntent())
    r = say(a, s, "My name is Daniel Okafor, DOB 1979-06-02, card ending 8812. My Northgate dispute was denied.")
    assert s.selected_id == "DSP-3950" and "answer_dispute" in actions(r) and "signed receipt" in r.reply


def test_marcus_has_no_disputes():
    a, s = new()
    r = say(a, s, "My name is Marcus Bell, DOB 2000-04-09, card ending 7009")
    assert "no_disputes" in actions(r) and s.phase == Phase.POST_PROCESS


# -------------------------------------------------------------- scope

def test_off_topic_refused_then_human_offered():
    a, s = new()
    r1 = say(a, s, "what is RL?")
    assert "off_topic" in actions(r1) and "reinforcement" not in r1.reply.lower()
    say(a, s, "who won the world cup?")
    r3 = say(a, s, "tell me a joke")
    assert s.handoff_offered and "transfer" in r3.reply.lower()
    say(a, s, "yes")
    assert s.phase == Phase.HUMAN_HANDOFF


def test_off_topic_after_verification_keeps_phase():
    a, s = new()
    say(a, s, CORE)
    r = say(a, s, "what is RL?")
    assert s.phase == Phase.PROCESS_CASE and "off_topic" in actions(r)


def test_caller_can_ask_for_human_anytime():
    a, s = new()
    say(a, s, "I want to talk to a real person")
    assert s.phase == Phase.HUMAN_HANDOFF and s.handoff_reason == "caller_request"


# ------------------------------------------------- authorized user and consent

def test_authorized_user_with_consent_default():
    a, s = new(scenario="default")
    r1 = say(a, s, REP)
    assert s.caller_role == "representative" and s.rep_authorized and s.consent.status == "pending"
    assert "DSP-" not in r1.reply and not s.verified
    say(a, s, "ok he approved it")
    assert s.verified and s.phase in (Phase.RESOLVE_INTENT, Phase.PROCESS_CASE)


def test_consent_timeout_hands_off_without_disclosure():
    a, s = new(scenario="timeout")
    replies = [say(a, s, REP).reply] + [say(a, s, "ok").reply for _ in range(4)]
    assert s.phase == Phase.HUMAN_HANDOFF and s.handoff_reason == "consent_timeout"
    assert not s.verified and all("DSP-" not in r for r in replies)


def test_unlisted_caller_is_refused():
    a, s = new()
    r = say(a, s, "Hi, I'm Ruth Park, I'm calling for my friend Daniel Okafor. His date of birth is "
                  "1979-06-02 and the card ends in 8812")
    assert s.rep_authorized is False and not s.verified and "rep_not_authorized" in actions(r)
    assert "DSP-" not in r.reply


# ------------------------------------------------------------- alternate IDs

@pytest.mark.parametrize("utterance,party", [
    ("My name is Lena Hofman, DOB 1986-01-30, email lena.h@example.org", "C3"),
    ("This is Priya Raman, born 1992-11-23, card ending 0436", "C2"),
    ("My name is Daniel Okafor, phone (404) 555-0147, email d.okafor@example.com", "C1"),
])
def test_alternate_identity_fields(utterance, party):
    a, s = new()
    say(a, s, utterance)
    assert s.verified_party_id == party


def test_expired_deadline_offers_human():
    from app.config import Settings
    a, s = new()
    a.cfg = Settings(demo_today="2026-10-09")
    r = say(a, s, CORE)
    assert "deadline_passed" in actions(r) and s.handoff_offered


# ------------------------------------------------- lockout and PII hygiene

def test_lockout_survives_a_new_session():
    from app.verify import FailureRegistry
    shared = FailureRegistry(max_failures=5, window_s=600)

    a1 = Agent(OfflineLLM(), failures=shared)
    s1 = SessionState(); a1.opening(s1)
    say(a1, s1, WRONG, "DOB 1979-06-04", "DOB 1979-06-05")
    assert s1.phase == Phase.HUMAN_HANDOFF            # per session limit

    a2 = Agent(OfflineLLM(), failures=shared)          # attacker opens a new chat
    s2 = SessionState(); a2.opening(s2)
    say(a2, s2, WRONG, "DOB 1979-06-06")
    assert s2.phase == Phase.HUMAN_HANDOFF and s2.handoff_reason == "identity_locked"

    a3 = Agent(OfflineLLM(), failures=shared)          # even correct details are refused now
    s3 = SessionState(); a3.opening(s3)
    r = say(a3, s3, "My name is Daniel Okafor, DOB 1979-06-02, card ending 8812")
    assert not s3.verified and s3.handoff_reason == "identity_locked" and "DSP-" not in r.reply


def test_failures_with_no_matching_field_do_not_lock_anyone():
    from app.verify import FailureRegistry
    shared = FailureRegistry(max_failures=2, window_s=600)
    for _ in range(3):
        a = Agent(OfflineLLM(), failures=shared); s = SessionState(); a.opening(s)
        say(a, s, "My name is Nobody Here, DOB 1970-01-01, card ending 0000")
    a = Agent(OfflineLLM(), failures=shared); s = SessionState(); a.opening(s)
    say(a, s, CORE)
    assert s.verified_party_id == "C1"


def test_trace_never_contains_raw_identity():
    a, s = new()
    blob = json.dumps(say(a, s, CORE).trace)
    for raw in ("Daniel Okafor", "1979-06-02", "8812", "ACC-30417"):
        assert raw not in blob


# ------------------------------------------------------- speaker change mid call

def test_new_speaker_after_verified_cardholder_relocks_gate():
    a, s = new()
    say(a, s, CORE)
    assert s.verified
    r = say(a, s, REP)
    assert "identity_changed" in actions(r)
    assert s.caller_role == "representative" and s.rep_authorized and s.consent.status == "pending"
    assert not s.verified and s.phase == Phase.VERIFY_ID
    assert "DSP-" not in r.reply and "proof of order" not in r.reply.lower()


def test_models_do_not_see_the_earlier_callers_turns_after_a_switch():
    spy = SpyLLM()
    a, s = new(spy)
    say(a, s, CORE, REP)
    reply_input = spy.generate_inputs[-1].lower()
    assert "dsp-4102" not in reply_input and "proof of order" not in reply_input
    assert spy.payloads[-1]["grounded"] == {}


def test_same_cardholder_repeating_name_does_not_relock():
    a, s = new()
    say(a, s, CORE)
    r = say(a, s, "My name is Daniel Okafor. How soon do I need to send them?")
    assert s.verified and "identity_changed" not in actions(r)


def test_different_name_after_verification_relocks():
    a, s = new()
    say(a, s, CORE)
    r = say(a, s, "Actually this is Priya Raman now, what disputes do I have?")
    assert "identity_changed" in actions(r) and not s.verified and "DSP-" not in r.reply


def test_partial_names_after_verification_do_not_reset_consent():
    class PartialNames(OfflineLLM):
        def extract(self, system, user, schema, tool_name):
            r = super().extract(system, user, schema, tool_name)
            if "LATEST CALLER MESSAGE:\nok" in user:
                r.value.update({"rep_name": "Grace", "full_name": "Daniel"})
            return r

    a, s = new(PartialNames(), scenario="timeout")
    say(a, s, REP, "ok", "ok", "ok", "ok")
    assert s.phase == Phase.HUMAN_HANDOFF and s.handoff_reason == "consent_timeout"
    assert not any(e.kind == "identity_changed" for e in s.events)


# ------------------------------------------------------- model failure modes

def test_empty_model_reply_is_recorded_not_silent():
    class Empty(OfflineLLM):
        name = "test:empty"

        def generate(self, system, user, max_tokens=400):
            r = super().generate(system, user, max_tokens)
            r.value, r.stats.stop_reason = "", "length"
            return r

    a, s = new(Empty())
    r = say(a, s, CORE)
    assert r.trace["reply_source"] == "template_empty_model_reply"
    assert any(e.kind == "empty_model_reply" and e.data["stop_reason"] == "length" for e in s.events)
    assert "proof of order" in r.reply


def test_unparseable_extraction_is_logged_and_the_turn_still_runs():
    class Garbled(OfflineLLM):
        def extract(self, system, user, schema, tool_name):
            r = super().extract(system, user, schema, tool_name)
            r.value, r.stats.parse_ok = {}, False
            return r

    a, s = new(Garbled())
    r = say(a, s, CORE)
    assert any(e.kind == "extract_parse_failed" for e in s.events)
    assert s.phase == Phase.VERIFY_ID and "ask_identity" in actions(r)
