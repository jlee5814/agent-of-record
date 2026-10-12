"""The procedure. Each turn runs extract, control, respond, and guard.

Only this module changes session state or phase during a turn. The model
proposes fields and words the reply; code decides everything else.
"""
from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass, field
from typing import Optional

from . import extract as X
from . import respond as R
from .config import Settings, settings as default_settings
from .data import norm_account, norm_dob, norm_email, norm_last4, norm_phone, same_person
from .domain import Domain, get_domain
from .records import document_guidance, followup_answer, resolve
from .llm import CallStats
from .state import HINT_FIELDS, IDENTITY_FIELDS, ConsentState, Identity, IntentMemory, Phase, SessionState
from .summary import Outbox, build_summary
from .verify import FailureRegistry, poll_consent, rep_authorized, verify

NEGATIVE = {"frustrated", "angry", "anxious", "confused", "refusing"}
ESCALATING = {"frustrated", "angry"}
PII_FIELDS = ("full_name", "dob", "phone", "email", "id_last4", "account_number")


def _same_value(field: str, new: str, old: Optional[str]) -> bool:
    """True if a value only restates what the caller already gave.

    Models repeat earlier values, sometimes in part (a first name) or in a new
    format (06/02/1979). Treating those as new input would force a pointless
    verification that a partial name then fails.
    """
    if not old:
        return False
    if field == "full_name":
        return same_person(new, [old])
    norm = {"dob": norm_dob, "phone": norm_phone, "email": norm_email,
            "id_last4": norm_last4, "account_number": norm_account}.get(field)
    if norm:
        a, b = norm(new), norm(old)
        return a is not None and a == b
    return new == old


def mask_pii(d: dict) -> dict:
    """Replace identity values with their length for traces and the panel.

    Free text fields, like the restated question, can quote the caller's
    details too, so those values and any long digit runs are scrubbed there.
    """
    secrets = [str(d[k]) for k in PII_FIELDS if d.get(k)]
    out = {}
    for k, v in d.items():
        if k in PII_FIELDS and v:
            out[k] = f"<{len(str(v))} chars>"
        elif isinstance(v, str):
            for sec in secrets:
                v = v.replace(sec, "***")
            out[k] = re.sub(r"\d[\d/.-]{3,}\d", "***", v)
        else:
            out[k] = v
    return out


@dataclass
class TurnResult:
    reply: str
    state: SessionState
    trace: dict = field(default_factory=dict)


class Agent:
    def __init__(self, llm, domain: Domain | str = "freight", cfg: Settings = default_settings,
                 outbox: Optional[Outbox] = None, failures: Optional[FailureRegistry] = None):
        self.llm = llm
        self.domain = get_domain(domain) if isinstance(domain, str) else domain
        self.store = self.domain.store
        self.cfg = cfg
        self.outbox = outbox or Outbox()
        # Share one registry across sessions (the server does) so lockouts survive a new chat.
        self.failures = failures or FailureRegistry(cfg.max_party_failures, cfg.lockout_window_min * 60)

    # ------------------------------------------------------------ public

    def opening(self, state: SessionState) -> str:
        if not state.transcript:
            state.transcript.append({"role": "assistant", "content": self.domain.opening})
        return self.domain.opening

    def handle(self, state: SessionState, text: str) -> TurnResult:
        t_turn = time.perf_counter()
        calls: list[CallStats] = []
        phase_before = state.phase

        # 1. Extract. After verification, the extractor also sees the caller's record labels,
        #    so it can map the paper mill one to an ID.
        D = self.domain
        candidates = None
        if state.verified:
            candidates = [D.label(r) for r in self.store.records_for(state.verified_party_id)]
        ex_res = self.llm.extract(X.system_prompt(D), X.build_user_message(state, text, self.cfg.today(), candidates),
                                  X.json_schema(D), "record_turn")
        calls.append(ex_res.stats)
        ex = X.parse(ex_res.value, D)
        if not ex_res.stats.parse_ok:
            # The turn still runs; an empty extraction just asks again.
            state.log("extract_parse_failed", mode=ex_res.stats.mode)
        state.transcript.append({"role": "user", "content": text})

        # 2. Control.
        t_ctl = time.perf_counter()
        directives: list[dict] = []
        ctx = {"grounded": None, "utterance": text}
        self._control(state, ex, directives, ctx)
        ctl_ms = (time.perf_counter() - t_ctl) * 1000

        # 3. Respond, then guard.
        grounded = ctx["grounded"] if state.verified else None
        payload = R.build_payload(state, D, directives, grounded, ex.question, ex.emotion)
        gen = self.llm.generate(R.system_prompt(D), R.build_user_message(state, payload))
        calls.append(gen.stats)
        reply = gen.value
        reply_source = "offline_template" if getattr(self.llm, "name", "").startswith("offline") else "model"
        if not reply:
            # Never silent: the trace says the model came back empty and why.
            reply = R.render_template(payload)
            reply_source = "template_empty_model_reply"
            state.log("empty_model_reply", stop_reason=gen.stats.stop_reason,
                      thinking_chars=gen.stats.thinking_chars)

        t_guard = time.perf_counter()
        user_texts = [m["content"] for m in state.transcript if m["role"] == "user"]
        problems = R.guard(reply, state, self.store, D, grounded, user_texts)
        if problems:
            state.log("guard_blocked", problems=problems, blocked=reply)
            reply = R.render_template(payload)
            reply_source = "template_guard"
        guard_ms = (time.perf_counter() - t_guard) * 1000

        state.transcript.append({"role": "assistant", "content": reply})
        total_ms = (time.perf_counter() - t_turn) * 1000
        trace = {
            "phase_before": phase_before.value,
            "phase_after": state.phase.value,
            "extraction": mask_pii(ex.model_dump(exclude_defaults=True)),
            "directives": directives,
            "grounded_keys": sorted((grounded or {}).keys()),
            "guard": problems,
            "reply_source": reply_source,
            "timing_ms": {
                "extract": round(ex_res.stats.latency_ms, 1),
                "control": round(ctl_ms, 2),
                "respond": round(gen.stats.latency_ms, 1),
                "guard": round(guard_ms, 2),
                "total": round(total_ms, 1),
            },
            "calls": [asdict(c) for c in calls],
        }
        state.log("turn", phase_before=phase_before.value, phase_after=state.phase.value,
                  directives=[d["action"] for d in directives], guard=problems)
        return TurnResult(reply, state, trace)

    # ------------------------------------------------------------ control

    def _control(self, s: SessionState, ex: X._Base, out: list[dict], ctx: dict) -> None:
        if s.phase in (Phase.HUMAN_HANDOFF, Phase.ENDED):
            out.append({"action": "handoff" if s.phase == Phase.HUMAN_HANDOFF else "ended"})
            return

        # A new speaker after verification relocks the gate before anything else runs.
        if self._identity_switched(s, ex):
            out.append({"action": "identity_changed"})
        self._merge(s, ex)

        # A safety emergency outranks everything, verified or not.
        if ex.emergency:
            out.append({"action": "emergency"})
            self._handoff(s, out, "emergency", announce=False)
            return

        # Caller asked for a person, or accepted an offer of one.
        if ex.wants_human or (s.handoff_offered and ex.accepts_human):
            if ex.emotion in NEGATIVE:
                out.append({"action": "empathize", "emotion": ex.emotion})
            self._handoff(s, out, "caller_request" if ex.wants_human else s.handoff_reason or "offered")
            return
        if s.handoff_offered and ex.declines_human:
            s.handoff_offered = False
            out.append({"action": "handoff_declined"})

        if ex.emotion in NEGATIVE:
            out.append({"action": "empathize", "emotion": ex.emotion})

        off_topic = ex.scope == "out_of_scope"
        if off_topic:
            offer = s.off_topic_strikes >= self.cfg.max_off_topic
            out.append({"action": "off_topic", "strikes": s.off_topic_strikes, "offer_human": offer})
            if offer:
                s.handoff_offered = True
                s.handoff_reason = "repeated_off_topic"
                return
        elif ex.scope == "smalltalk" and len(s.transcript) <= 2:
            out.append({"action": "greet_back"})

        if ex.restricted_request:
            out.append({"action": "restricted"})

        if s.frustration_streak >= self.cfg.max_frustrated_turns and not s.handoff_offered:
            s.handoff_offered = True
            s.handoff_reason = "sustained_frustration"
            out.append({"action": "offer_human"})

        if s.phase == Phase.VERIFY_ID:
            self._verify_phase(s, ex, out, ctx)
        elif s.phase == Phase.RESOLVE_INTENT:
            self._resolve_phase(s, ex, out, ctx, turn_hints=self._turn_hints(ex))
        elif s.phase == Phase.PROCESS_CASE:
            if off_topic or (ex.restricted_request and not ex.intent):
                out.append({"action": "anything_else"})
                self._ground(s, ctx)
            else:
                self._process_phase(s, ex, out, ctx)
        elif s.phase == Phase.POST_PROCESS:
            self._post_phase(s, ex, out, ctx, off_topic)

    # ------------------------------------------------------------ speaker change

    def _identity_switched(self, s: SessionState, ex: X._Base) -> bool:
        """Reset to VERIFY_ID if someone new speaks after verification.

        A new speaker is a representative joining, a different representative, or a
        name that doesn't match the record. The same person restating a name changes
        nothing. Failed attempt counts carry over.
        """
        if not s.verified_party_id:
            return False
        rec = self.store.person(s.verified_party_id)
        new_rep = bool(ex.caller_role == "representative" or ex.rep_name) and s.caller_role != "representative"
        other_rep = bool(ex.rep_name and s.rep_name and not same_person(ex.rep_name, [s.rep_name]))
        other_name = bool(ex.full_name and rec and not same_person(ex.full_name, [rec.name, *rec.name_aliases]))
        if not (new_rep or other_rep or other_name):
            return False
        s.log("identity_changed", reason="representative" if new_rep or other_rep else "different_name")
        s.verified_party_id = None
        s.identity = Identity()
        s.identity_dirty = False
        s.caller_role = "unknown"
        s.rep_name = s.rep_relationship = None
        s.rep_authorized = None
        s.consent = ConsentState(scenario=s.consent.scenario)
        s.selected_id = None
        s.candidate_ids = []
        s.topics_discussed = []
        s.phase = Phase.VERIFY_ID
        s.context_start = len(s.transcript) - 1      # models now see only the new speaker's turn
        return True

    # ------------------------------------------------------------ merge

    def _merge(self, s: SessionState, ex: X._Base) -> None:
        if s.phase == Phase.VERIFY_ID:
            for f in (*IDENTITY_FIELDS, "account_number"):
                v = getattr(ex, f)
                if v and not _same_value(f, v, getattr(s.identity, f)):
                    setattr(s.identity, f, v)
                    if f != "account_number":
                        s.identity_dirty = True
            if ex.caller_role == "representative":
                s.caller_role = "representative"
            elif ex.caller_role == "holder" and s.caller_role == "unknown":
                s.caller_role = "holder"
            if ex.rep_name:
                s.rep_name = ex.rep_name
                if s.caller_role == "unknown":
                    s.caller_role = "representative"
            if ex.relationship:
                s.rep_relationship = ex.relationship

        # Remember intent hints whenever they appear, even before verification.
        hints = self._turn_hints(ex)
        if hints.any():
            for k in (*HINT_FIELDS, "intent"):
                v = getattr(hints, k)
                if v:
                    setattr(s.memory, k, v)
            if s.phase == Phase.VERIFY_ID:
                s.log("memory_captured", hints=hints.model_dump(exclude_none=True))

        # Conversation counters.
        if ex.scope == "out_of_scope":
            s.off_topic_strikes += 1
        elif ex.scope == "in_scope":
            s.off_topic_strikes = 0
        if ex.emotion in ESCALATING or ex.refuses_verification:
            s.frustration_streak += 1
        elif ex.emotion in ("neutral", "positive"):
            s.frustration_streak = 0
        s.last_emotion = ex.emotion

    @staticmethod
    def _turn_hints(ex: X._Base) -> IntentMemory:
        return IntentMemory(record_id=ex.record_id, party=ex.party, category=ex.category, status=ex.record_status,
                            date=ex.date, month=ex.month, year=ex.year, intent=ex.intent)

    # ------------------------------------------------------------ VERIFY_ID

    def _verify_phase(self, s: SessionState, ex: X._Base, out: list[dict], ctx: dict) -> None:
        start = len(out)
        self._verify_inner(s, ex, out, ctx)
        # Acknowledge a remembered hint only while the caller is still unverified.
        if self._turn_hints(ex).any() and not s.verified and s.phase == Phase.VERIFY_ID:
            out.insert(start, {"action": "noted_hint"})

    def _verify_inner(self, s: SessionState, ex: X._Base, out: list[dict], ctx: dict) -> None:
        if ex.refuses_verification or (ex.emotion in ESCALATING and not s.identity_dirty):
            out.append({"action": "explain_verification"})

        rep = s.caller_role == "representative"
        if rep and not s.rep_name:
            out.append({"action": "ask_rep_identity"})

        # A representative whose account holder already matched is waiting on authorization or consent.
        if rep and s.verified_party_id and not s.identity_dirty:
            self._rep_gate(s, out, ctx)
            return

        if s.identity_dirty:
            s.identity_dirty = False
            r = verify(s.identity, self.store, self.cfg.min_pii_matches)
            s.log("verify", status=r.status, provided=r.provided, detail=r.detail)
            if r.status == "verified" and self.failures.locked(r.party_id):
                # Correct details, but this account holder is locked from earlier failures.
                # Same reply as any handoff, so the lockout reveals nothing about the match.
                s.log("verify_locked")
                self._handoff(s, out, "identity_locked")
                return
            if r.status == "verified":
                s.verified_party_id = r.party_id
                if rep:
                    self._rep_gate(s, out, ctx)
                else:
                    if s.caller_role == "unknown":
                        s.caller_role = "holder"
                    out.append({"action": "verified"})
                    self._enter_resolve(s, out, ctx)
                return
            if r.status == "mismatch":
                s.verify_attempts += 1
                if r.detail.get("matched"):
                    self.failures.record(r.detail.get("best_party"))
                left = self.cfg.max_verify_attempts - s.verify_attempts
                if self.failures.locked(r.detail.get("best_party")) and r.detail.get("matched"):
                    self._handoff(s, out, "identity_locked")
                    return
                if left <= 0:
                    self._handoff(s, out, "verification_failed")
                    return
                out.append({"action": "verify_failed", "attempts_left": left})
                return
            # need_more falls through to ask.
        missing = [f for f in IDENTITY_FIELDS if not getattr(s.identity, f)]
        have = len(s.identity.provided())
        out.append({"action": "ask_identity", "needed": max(self.cfg.min_pii_matches - have, 1),
                    "missing": missing, "have": have, "for_holder": rep})

    def _rep_gate(self, s: SessionState, out: list[dict], ctx: dict) -> None:
        if s.rep_authorized is None:
            s.rep_authorized = rep_authorized(s.rep_name, s.verified_party_id, self.store)
            s.log("rep_check", rep=s.rep_name, authorized=s.rep_authorized)
        if not s.rep_name:
            out.append({"action": "ask_rep_identity"})
            return
        if not s.rep_authorized:
            out.append({"action": "rep_not_authorized"})
            s.handoff_offered = True
            s.handoff_reason = "rep_not_authorized"
            return
        if s.consent.status == "not_requested":
            s.consent.status, s.consent.polls = poll_consent(s.consent.scenario, 0, self.store)
            s.log("consent_requested", scenario=s.consent.scenario, status=s.consent.status)
            out.append({"action": "consent_requested", "sent_to": self.domain.consent_channel,
                        "note": f"the caller cannot grant consent; wait for the {self.domain.person_noun}'s approval"})
            if s.consent.status != "approved":
                return
        elif s.consent.status == "pending":
            s.consent.status, s.consent.polls = poll_consent(s.consent.scenario, s.consent.polls, self.store)
            s.log("consent_poll", status=s.consent.status, polls=s.consent.polls)
        if s.consent.status == "pending":
            out.append({"action": "consent_pending", "note": f"approval must come through {self.domain.consent_channel}"})
        elif s.consent.status == "timed_out":
            out.append({"action": "consent_timeout"})
            self._handoff(s, out, "consent_timeout", announce=False)
        elif s.consent.status == "approved":
            out.append({"action": "consent_approved", "approved_by": f"the {self.domain.person_noun}, in {self.domain.consent_channel}"})
            out.append({"action": "verified"})
            self._enter_resolve(s, out, ctx)

    # ------------------------------------------------------------ RESOLVE_INTENT

    def _enter_resolve(self, s: SessionState, out: list[dict], ctx: dict) -> None:
        s.phase = Phase.RESOLVE_INTENT
        s.log("phase", to=s.phase.value)
        self._resolve_phase(s, None, out, ctx, turn_hints=IntentMemory())

    def _resolve_phase(self, s: SessionState, ex: Optional[X._Base], out: list[dict], ctx: dict,
                       turn_hints: IntentMemory) -> None:
        records = self.store.records_for(s.verified_party_id)
        r = resolve(s.memory, records, self.domain)
        if r.status == "none" and turn_hints.any():
            r = resolve(turn_hints, records, self.domain)  # newest words win over stale hints
        s.log("resolve", status=r.status, record_id=r.record_id, candidates=r.candidates, hints=r.used_hints)
        labels = {x.record_id: self.domain.label(x) for x in records}
        self._ground(s, ctx, on_file=list(labels.values()))

        if r.status == "selected":
            s.selected_id = r.record_id
            s.candidate_ids = []
            s.phase = Phase.PROCESS_CASE
            s.log("phase", to=s.phase.value, record_id=r.record_id)
            out.append({"action": "record_selected", "label": labels[r.record_id]})
            if s.memory.intent or (ex and (ex.question or ex.intent)):
                self._answer(s, ex, out, ctx)
            else:
                out.append({"action": "ask_record_question"})
        elif r.status == "ambiguous":
            s.candidate_ids = r.candidates
            out.append({"action": "ask_disambiguate", "options": [labels[c] for c in r.candidates]})
        elif r.status == "none":
            s.candidate_ids = [x.record_id for x in records]
            out.append({"action": "no_match", "options": list(labels.values())})
        else:
            out.append({"action": "no_records"})
            s.phase = Phase.POST_PROCESS
            out.append({"action": "offer_email", "email_masked": self._masked_email(s)})

    # ------------------------------------------------------------ PROCESS_CASE

    def _process_phase(self, s: SessionState, ex: X._Base, out: list[dict], ctx: dict) -> None:
        # Switching records mid conversation.
        th = self._turn_hints(ex)
        if th.has_record_hint():
            records = self.store.records_for(s.verified_party_id)
            r = resolve(IntentMemory(**th.model_dump(exclude={"intent"})), records, self.domain)
            if r.status == "selected" and r.record_id != s.selected_id:
                s.selected_id = r.record_id
                s.log("record_switch", record_id=r.record_id)
                out.append({"action": "record_selected", "label": self.domain.label(self.store.record(r.record_id))})
            elif r.status == "ambiguous" and s.selected_id not in r.candidates:
                s.phase = Phase.RESOLVE_INTENT
                s.memory = IntentMemory(**th.model_dump())
                self._resolve_phase(s, ex, out, ctx, turn_hints=th)
                return

        if ex.done and not ex.question:
            s.phase = Phase.POST_PROCESS
            s.log("phase", to=s.phase.value)
            self._ground(s, ctx)
            out.append({"action": "offer_email", "email_masked": self._masked_email(s)})
            return
        self._answer(s, ex, out, ctx)

    def _answer(self, s: SessionState, ex: Optional[X._Base], out: list[dict], ctx: dict) -> None:
        rec = self.store.record(s.selected_id)
        intent = (ex.intent if ex and ex.intent else None) or s.memory.intent
        # Live status: the first look shows the current update; asking again reads the next one.
        if rec.live:
            if rec.record_id not in s.live_polls:
                s.live_polls[rec.record_id] = 0
            elif intent in self.domain.live_intents:
                s.live_polls[rec.record_id] = min(s.live_polls[rec.record_id] + 1, len(rec.live) - 1)
                s.log("live_poll", record_id=rec.record_id, index=s.live_polls[rec.record_id])
        topic = ex.followup_topic if ex else None
        follow = None
        if rec.documents_needed or topic:
            f = followup_answer(rec, self.store.guidance, ctx.get("utterance", ""), topic)
            follow = f if f["topic"] != "fallback" or intent == "document_submission" else None
        self._ground(s, ctx, followup=follow)
        if rec.record_id not in s.topics_discussed:
            s.topics_discussed.append(rec.record_id)
        if follow:
            s.topics_discussed.append(follow["topic"])
        out.append({"action": "answer_record", "intent": intent})
        deadline = ctx["grounded"]["record"].get("deadline")
        if deadline and deadline["expired"] and not s.handoff_offered:
            out.append({"action": "deadline_passed"})
            s.handoff_offered = True
            s.handoff_reason = "deadline_passed"
        else:
            out.append({"action": "anything_else"})
        # Consume the remembered intent once it has been answered.
        s.memory.intent = None

    # ------------------------------------------------------------ POST_PROCESS

    def _post_phase(self, s: SessionState, ex: X._Base, out: list[dict], ctx: dict, off_topic: bool) -> None:
        self._ground(s, ctx)
        if ex.email_choice == "send":
            p = self.store.person(s.verified_party_id)
            email = self.outbox.send(build_summary(s, self.domain, self.cfg.today()))
            s.email_decision, s.email_sent = "send", True
            s.log("email_sent", to=R.mask_email(p.email), chars=len(email.body))
            out.append({"action": "email_sent", "email_masked": R.mask_email(p.email)})
            s.phase = Phase.ENDED
            return
        if ex.email_choice == "skip":
            s.email_decision = "skip"
            s.log("email_skipped")
            out.append({"action": "email_skipped"})
            s.phase = Phase.ENDED
            return
        if not off_topic and (ex.question or ex.intent) and s.selected_id:
            s.phase = Phase.PROCESS_CASE
            self._process_phase(s, ex, out, ctx)
            return
        out.append({"action": "offer_email", "email_masked": self._masked_email(s)})

    # ------------------------------------------------------------ helpers

    def _handoff(self, s: SessionState, out: list[dict], reason: str, announce: bool = True) -> None:
        s.phase = Phase.HUMAN_HANDOFF
        s.handoff_reason = reason
        s.log("handoff", reason=reason, verified=s.verified, record_id=s.selected_id,
              memory=s.memory.model_dump(exclude_none=True))
        if announce:
            out.append({"action": "handoff", "reason": reason})

    def _masked_email(self, s: SessionState) -> Optional[str]:
        p = self.store.person(s.verified_party_id or "")
        return R.mask_email(p.email) if p else None

    def _ground(self, s: SessionState, ctx: dict, on_file: Optional[list[str]] = None,
                followup: Optional[dict] = None) -> None:
        """Build the only data the reply model may see. Empty until verified."""
        if not s.verified:
            ctx["grounded"] = None
            return
        D = self.domain
        g = ctx.get("grounded") or {}
        p = self.store.person(s.verified_party_id)
        g["caller"] = {"first_name": p.name.split()[0], "role": s.caller_role,
                       "rep_name": s.rep_name if s.caller_role == "representative" else None}
        g["on_file"] = on_file or [D.label(x) for x in self.store.records_for(p.party_id)]
        if s.selected_id:
            rec = self.store.record(s.selected_id)
            g["record"] = D.facts(rec, self.cfg.today(), D.live_entry(rec, s))
            g["documents"] = document_guidance(rec, self.store.guidance) if rec.documents_needed else None
        if followup is not None:
            g["followup"] = followup
        ctx["grounded"] = g
