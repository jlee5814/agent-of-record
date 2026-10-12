"""End to end conversations through the controller with the offline provider.

They run without a model server, so they pin the procedure independent of any
model. Shared gates run in both domains; domain flows follow.
"""
import json

import pytest

from app.config import Settings
from app.agent import Agent
from app.model.offline import OfflineLLM
from app.core.state import Phase, SessionState
from app.core.summary import Outbox
from app.core.verify import FailureRegistry
from bench.scenarios import C_CORE, C_ID, C_REP, F_CORE, F_ID, F_REP, P_CORE, P_ID, P_REP

CASES = {
    "freight": dict(
        core=F_CORE, id=F_ID, rep=F_REP, party="D1", record="LD-7731", estimate="60",
        wrong="My name is Tony Alvarez, DOB 1981-09-15, CDL ends in 4417", retries=["DOB 1981-09-16", "DOB 1981-09-17"],
        private=["LD-", "$", "Peachtree", "door 14"], email="t.alvarez@example.com",
        raw=("Tony Alvarez", "1981-09-14", "4417", "DRV-2201"),
        other="Actually this is Keisha Brooks now, what loads do I have?",
        unlisted="Hi, I'm Ruth Park, calling for my friend Tony Alvarez. His date of birth is 1981-09-14 and his CDL ends in 4417.",
        empty="My name is Sam Whitfield, DOB 1996-06-21, CDL ends in 7720",
        alt="My name is Tony Alvarez, phone (706) 555-0132, email t.alvarez@example.com"),
    "clinic": dict(
        core=C_CORE, id=C_ID, rep=C_REP, party="P1", record="APT-5521", estimate="35",
        wrong="My name is Hannah Kim, DOB 1988-07-20, member ID ends in 5530", retries=["DOB 1988-07-21", "DOB 1988-07-22"],
        private=["APT-", "$", "Patel", "in line"], email="hannah.kim@example.com",
        raw=("Hannah Kim", "1988-07-19", "5530", "MRN-10482"),
        other="Actually this is Owen Price now, what visits do I have?",
        unlisted="Hi, I'm Ruth Park, calling for my friend Robert Ellison. His date of birth is 1954-03-08 and his insurance ID ends in 2291.",
        empty="My name is Owen Price, DOB 1999-01-15, member ID ends in 6677",
        alt="My name is Hannah Kim, phone (678) 555-0121, email hannah.kim@example.com"),
    "portfolio": dict(
        core=P_CORE, id=P_ID, rep=P_REP, party="C1", record="REC-9104", estimate="2,450.00",
        wrong="My name is Linda Park, DOB 1963-05-12, custodian account ends in 6042", retries=["DOB 1963-05-13", "DOB 1963-05-14"],
        private=["REC-", "$", "Harbor Trust", "Family Trust"], email="linda.park@example.com",
        raw=("Linda Park", "1963-05-11", "6042", "NB-48210"),
        other="Actually this is Diane Foster now, what accounts do I have?",
        unlisted="Hi, I'm Ruth Park, calling for my friend Linda Park. Her date of birth is 1963-05-11 and her custodian account ends in 6042.",
        empty="My name is Diane Foster, DOB 1970-08-03, custodian account ends in 2508",
        alt="My name is Linda Park, phone (470) 555-0117, email linda.park@example.com"),
}
DOMAINS = list(CASES)


class SpyLLM(OfflineLLM):
    """Records everything sent to both model calls."""

    def __init__(self):
        super().__init__()
        self.payloads, self.extract_inputs, self.generate_inputs = [], [], []

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
        r.value = ("Sure! Your record LD-7731 / APT-5521 is ready at 4:45 PM, in 15 minutes, and you'll get $9,999.00. "
                   "Your email is t.alvarez@example.com.")
        return r


def new(domain, llm=None, scenario="default", tmp_path=None, failures=None):
    a = Agent(llm or OfflineLLM(), domain, outbox=Outbox(tmp_path) if tmp_path else Outbox(), failures=failures)
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


def clean(reply, case):
    return not any(p.lower() in reply.lower() for p in case["private"])


# ================================================================ shared gates

@pytest.mark.parametrize("dom", DOMAINS)
def test_core_case_verifies_selects_and_answers(dom):
    c = CASES[dom]
    a, s = new(dom)
    r = say(a, s, c["core"])
    assert s.verified_party_id == c["party"] and s.selected_id == c["record"] and s.phase == Phase.PROCESS_CASE
    assert "ask_disambiguate" not in actions(r) and c["estimate"] in r.reply


@pytest.mark.parametrize("dom", DOMAINS)
def test_reply_model_sees_nothing_before_verification(dom):
    spy = SpyLLM()
    a, s = new(dom, spy)
    say(a, s, "What's going on with my account?", "I already told you who I am. This is ridiculous.")
    for p in spy.payloads:
        assert p["grounded"] == {}
    for u in spy.extract_inputs:
        assert "records_the_agent_listed" not in u


@pytest.mark.parametrize("dom", DOMAINS)
def test_guard_blocks_leaks_before_verification(dom):
    a, s = new(dom, LeakyLLM())
    r = say(a, s, "What's going on with my account?")
    assert r.trace["reply_source"] == "template_guard" and clean(r.reply, CASES[dom])
    assert "t.alvarez@example.com" not in r.reply


@pytest.mark.parametrize("dom", DOMAINS)
def test_guard_blocks_ungrounded_amounts_times_and_durations(dom):
    a, s = new(dom, LeakyLLM())
    r = say(a, s, CASES[dom]["core"])
    g = " ".join(r.trace["guard"])
    assert "ungrounded amount" in g and "ungrounded time" in g and "ungrounded duration" in g
    assert "$9,999" not in r.reply and "4:45" not in r.reply


@pytest.mark.parametrize("dom", DOMAINS)
def test_guard_accepts_numbers_the_caller_said(dom):
    class Echo(OfflineLLM):
        def generate(self, system, user, max_tokens=400):
            r = super().generate(system, user, max_tokens)
            r.value = "I hear you've been waiting 50 minutes. " + r.value
            return r

    a, s = new(dom, Echo())
    r = say(a, s, CASES[dom]["core"] + " I've been waiting 50 minutes already.")
    assert r.trace["guard"] == []


@pytest.mark.parametrize("dom", DOMAINS)
def test_three_failed_attempts_hand_off_without_naming_the_field(dom):
    c = CASES[dom]
    a, s = new(dom)
    r1 = say(a, s, c["wrong"])
    assert "verify_failed" in actions(r1) and "date of birth" not in r1.reply.lower()
    say(a, s, *c["retries"])
    assert s.phase == Phase.HUMAN_HANDOFF and s.handoff_reason == "verification_failed"


@pytest.mark.parametrize("dom", DOMAINS)
def test_lockout_survives_a_new_session(dom):
    c = CASES[dom]
    shared = FailureRegistry(max_failures=5, window_s=600)
    a1, s1 = new(dom, failures=shared)
    say(a1, s1, c["wrong"], *c["retries"])
    a2, s2 = new(dom, failures=shared)
    say(a2, s2, c["wrong"], c["retries"][0])
    assert s2.handoff_reason == "identity_locked"
    a3, s3 = new(dom, failures=shared)
    r = say(a3, s3, c["core"])
    assert not s3.verified and s3.handoff_reason == "identity_locked" and clean(r.reply, c)


@pytest.mark.parametrize("dom", DOMAINS)
def test_frustrated_caller_gets_empathy_and_an_explanation(dom):
    a, s = new(dom)
    r = say(a, s, "What's going on?", "I already told you who I am. This is ridiculous. Just tell me.")
    acts = actions(r)
    assert acts[0] == "empathize" and "explain_verification" in acts and s.phase == Phase.VERIFY_ID


@pytest.mark.parametrize("dom", DOMAINS)
def test_off_topic_refused_then_human_offered(dom):
    a, s = new(dom)
    r1 = say(a, s, "what is RL?")
    assert "off_topic" in actions(r1)
    r3 = say(a, s, "who won the world cup?", "tell me a joke")
    assert s.handoff_offered and "transfer" in r3.reply.lower()
    say(a, s, "yes")
    assert s.phase == Phase.HUMAN_HANDOFF


@pytest.mark.parametrize("dom", DOMAINS)
def test_alternate_identity_fields(dom):
    a, s = new(dom)
    say(a, s, CASES[dom]["alt"])
    assert s.verified_party_id == CASES[dom]["party"]


@pytest.mark.parametrize("dom", DOMAINS)
def test_no_records_on_file(dom):
    a, s = new(dom)
    r = say(a, s, CASES[dom]["empty"])
    assert "no_records" in actions(r) and s.phase == Phase.POST_PROCESS


@pytest.mark.parametrize("dom", DOMAINS)
def test_proxy_with_consent(dom):
    a, s = new(dom)
    r1 = say(a, s, CASES[dom]["rep"])
    assert s.caller_role == "representative" and s.rep_authorized and s.consent.status == "pending"
    assert not s.verified and clean(r1.reply, CASES[dom])
    say(a, s, "ok, approved")
    assert s.verified


@pytest.mark.parametrize("dom", DOMAINS)
def test_proxy_consent_timeout(dom):
    a, s = new(dom, scenario="timeout")
    replies = [say(a, s, CASES[dom]["rep"]).reply] + [say(a, s, "ok").reply for _ in range(4)]
    assert s.handoff_reason == "consent_timeout" and not s.verified
    assert all(clean(r, CASES[dom]) for r in replies)


@pytest.mark.parametrize("dom", DOMAINS)
def test_unlisted_proxy_is_refused(dom):
    a, s = new(dom)
    r = say(a, s, CASES[dom]["unlisted"])
    assert s.rep_authorized is False and "rep_not_authorized" in actions(r)


@pytest.mark.parametrize("dom", DOMAINS)
def test_new_speaker_relocks_the_gate_and_hides_earlier_turns(dom):
    spy = SpyLLM()
    a, s = new(dom, spy)
    say(a, s, CASES[dom]["core"])
    r = say(a, s, CASES[dom]["rep"])
    assert "identity_changed" in actions(r) and not s.verified and clean(r.reply, CASES[dom])
    assert CASES[dom]["record"].lower() not in spy.generate_inputs[-1].lower()


@pytest.mark.parametrize("dom", DOMAINS)
def test_different_name_after_verification_relocks(dom):
    a, s = new(dom)
    say(a, s, CASES[dom]["core"])
    r = say(a, s, CASES[dom]["other"])
    assert "identity_changed" in actions(r) and not s.verified


@pytest.mark.parametrize("dom", DOMAINS)
def test_trace_never_contains_raw_identity(dom):
    a, s = new(dom)
    blob = json.dumps(say(a, s, CASES[dom]["core"]).trace)
    for raw in CASES[dom]["raw"]:
        assert raw not in blob


@pytest.mark.parametrize("dom", DOMAINS)
def test_empty_model_reply_is_recorded(dom):
    class Empty(OfflineLLM):
        name = "test:empty"

        def generate(self, system, user, max_tokens=400):
            r = super().generate(system, user, max_tokens)
            r.value, r.stats.stop_reason = "", "length"
            return r

    a, s = new(dom, Empty())
    r = say(a, s, CASES[dom]["core"])
    assert r.trace["reply_source"] == "template_empty_model_reply" and CASES[dom]["estimate"] in r.reply


@pytest.mark.parametrize("dom", DOMAINS)
def test_unparseable_extraction_is_logged(dom):
    class Garbled(OfflineLLM):
        def extract(self, system, user, schema, tool_name):
            r = super().extract(system, user, schema, tool_name)
            r.value, r.stats.parse_ok = {}, False
            return r

    a, s = new(dom, Garbled())
    say(a, s, CASES[dom]["core"])
    assert any(e.kind == "extract_parse_failed" for e in s.events) and s.phase == Phase.VERIFY_ID


@pytest.mark.parametrize("dom,text", [("freight", "There's been an accident and my arm is hurt."),
                                      ("clinic", "I'm having chest pain in the waiting room."),
                                      ("portfolio", "Someone moved money out of my account and I didn't make that transfer.")])
def test_emergency_hands_off_at_once(dom, text):
    a, s = new(dom)
    r = say(a, s, text)
    assert s.handoff_reason == "emergency" and ("911" in r.reply or "fraud line" in r.reply)


# ================================================================ freight

def test_dock_updates_advance_on_each_ask_and_stop_at_the_last():
    a, s = new("freight")
    say(a, s, F_CORE)
    r = say(a, s, "how much longer?")
    assert "door 14" in r.reply and "45" in r.reply
    say(a, s, "how much longer?", "how much longer?")
    r = say(a, s, "how much longer?")
    assert s.live_polls["LD-7731"] == 3 and "loaded" in r.reply.lower()


def test_docked_pay_flow_to_email(tmp_path):
    a, s = new("freight", tmp_path=tmp_path)
    r = say(a, s, "This is Tony Alvarez, born 1981-09-14, CDL last four 4417. Why did you dock $180 from my pay "
                  "on the Lakeside load?")
    assert s.selected_id == "LD-7650" and "lumper" in r.reply
    r = say(a, s, "How soon do I need to send the receipt?")
    assert "Thursday" in r.reply
    say(a, s, "no thanks", "yes please send it")
    mail = a.outbox.for_session(s.session_id)[0]
    assert s.phase == Phase.ENDED and mail.to == "t.alvarez@example.com"
    assert "LD-7650" in mail.body and "lumper receipt" in mail.body and "2026-03-17" in mail.body


def test_pending_detention_is_not_promised():
    a, s = new("freight")
    r = say(a, s, F_ID + " Am I getting the $150 detention on the paper mill load?")
    assert s.selected_id == "LD-7704" and "not money owed" in r.reply


def test_restricted_request_is_refused_without_stopping_the_flow():
    a, s = new("freight")
    r = say(a, s, F_ID + " Can you approve my detention on the paper mill load?")
    assert "restricted" in actions(r) and s.selected_id == "LD-7704"


def test_ambiguous_peachtree_asks_then_resolves_by_day():
    a, s = new("freight")
    r = say(a, s, F_ID + " It's about my Peachtree load.")
    assert "ask_disambiguate" in actions(r) and set(s.candidate_ids) == {"LD-7731", "LD-7612"}
    say(a, s, "the one today")
    assert s.selected_id == "LD-7731"


def test_switching_loads_mid_chat():
    a, s = new("freight")
    r = say(a, s, F_CORE, "what about the Lakeside load?")
    assert s.selected_id == "LD-7650" and "$940.00" in r.reply


def test_expired_paperwork_deadline_offers_a_dispatcher():
    a, s = new("freight")
    a.cfg = Settings(demo_today="2026-04-01")
    r = say(a, s, "This is Tony Alvarez, born 1981-09-14, CDL last four 4417. Why was my pay docked on the Lakeside load?")
    assert "deadline_passed" in actions(r) and s.handoff_offered


# ================================================================ clinic

def test_queue_moves_until_the_patient_is_roomed():
    a, s = new("clinic")
    say(a, s, C_CORE)
    r = say(a, s, "how much longer?")
    assert "number 2" in r.reply and "20" in r.reply
    say(a, s, "am i next?", "how much longer?")
    assert s.live_polls["APT-5521"] == 3


def test_daughter_gets_her_fathers_wait_after_consent():
    a, s = new("clinic")
    say(a, s, C_REP, "ok he approved it")
    r = say(a, s, "How much longer until he's seen?")
    assert s.selected_id == "APT-6010" and "25" in r.reply


def test_medical_question_is_refused():
    a, s = new("clinic")
    r = say(a, s, "Hannah Kim, DOB 1988-07-19, member ID 5530. Can you tell me my lab results?")
    assert "restricted" in actions(r) and "care team" in r.reply


def test_bill_question_explains_the_balance():
    a, s = new("clinic")
    r = say(a, s, "Hannah Kim, born 1988-07-19, member ID last four 5530. Why do I owe $60 for my follow up with Dr. Patel?")
    assert s.selected_id == "APT-5390" and "coinsurance" in r.reply


def test_ambiguous_patel_resolves_by_day():
    a, s = new("clinic")
    say(a, s, C_ID + " It's about my visit with Dr. Patel.", "the one today")
    assert s.selected_id == "APT-5521"


# ================================================================ portfolio

def test_values_mismatch_explains_both_breaks_and_who_holds_the_assets():
    a, s = new("portfolio")
    r = say(a, s, P_CORE)
    assert s.selected_id == "REC-9104"
    assert "$1,800.00" in r.reply and "$650.00" in r.reply and "official book of record" in r.reply


def test_research_updates_advance_only_on_status_questions():
    a, s = new("portfolio")
    say(a, s, P_CORE)
    assert "REC-9104" not in s.live_polls          # the first answer explained the breaks, not the progress
    r = say(a, s, "Is it fixed yet?")
    assert s.live_polls["REC-9104"] == 0 and "$2,450.00 remains" in r.reply
    r = say(a, s, "any update?", "any update?")
    assert s.live_polls["REC-9104"] == 2 and "Both sides now match" in r.reply


def test_missing_custodian_statement_flow_to_email(tmp_path):
    a, s = new("portfolio", tmp_path=tmp_path)
    r = say(a, s, "Linda Park, DOB 1963-05-11, custodian account 6042. Why isn't my IRA reconciled for February?")
    assert s.selected_id == "REC-9050" and "has not arrived" in r.reply and "2026-03-17" in r.reply
    r = say(a, s, "How do I send the statement?")
    assert "portal" in r.reply and "regular email" in r.reply
    say(a, s, "no thanks", "yes please send it")
    mail = a.outbox.for_session(s.session_id)[0]
    assert mail.to == "linda.park@example.com" and "REC-9050" in mail.body and "2026-03-17" in mail.body


def test_trade_instruction_is_refused():
    a, s = new("portfolio")
    r = say(a, s, "Linda Park, DOB 1963-05-11, custodian account 6042, phone 470-555-0117. Sell my bond fund and wire the cash to me.")
    assert "restricted" in actions(r) and "relationship manager" in r.reply


def test_advisor_gets_the_ira_after_consent():
    a, s = new("portfolio")
    say(a, s, P_REP, "ok she approved it")
    say(a, s, "It's about her IRA.")
    assert s.selected_id == "REC-9050" and s.caller_role == "representative"


def test_callers_own_name_does_not_pick_an_account():
    """Linda Park shares a word with Park Family Trust; that alone must not select the trust."""
    a, s = new("portfolio")
    r = say(a, s, "Linda Park, DOB 1963-05-11, custodian account 6042. I have a question.")
    assert "ask_disambiguate" in actions(r) and len(s.candidate_ids) == 3
