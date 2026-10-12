"""The SGLang provider against a mock server: wire format, fallbacks, and metrics.

No real server runs here. httpx.MockTransport plays SGLang, so these tests
pin what goes over the wire and how odd responses are handled.
"""
import json

import httpx
import pytest

from app.agent import Agent
from app.core.domain import get_domain
from app.core.extraction import json_schema
from app.model.extract import system_prompt
from app.model.llm import ServerError, SGLangLLM, grammar_schema, strip_think
from app.core.state import Phase, SessionState


def sse(text: str = "", pieces: list[str] | None = None, reasoning: str = "", usage: dict | None = None,
        finish: str = "stop") -> str:
    chunks = []
    if reasoning:
        chunks.append({"choices": [{"index": 0, "delta": {"reasoning_content": reasoning}, "finish_reason": None}]})
    for p in pieces if pieces is not None else ([text] if text else []):
        chunks.append({"choices": [{"index": 0, "delta": {"content": p}, "finish_reason": None}]})
    chunks.append({"choices": [{"index": 0, "delta": {}, "finish_reason": finish}]})
    if usage:
        chunks.append({"choices": [], "usage": usage})
    return "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"


class FakeServer:
    """Records requests and answers each from a handler."""

    def __init__(self, handler, models=("Qwen/Qwen3-8B",)):
        self.handler, self.models, self.bodies = handler, models, []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"object": "list", "data": [{"id": m} for m in self.models]})
        body = json.loads(request.content)
        self.bodies.append(body)
        status, payload = self.handler(body, len(self.bodies))
        if status != 200:
            return httpx.Response(status, json={"object": "error", "message": payload, "code": status})
        return httpx.Response(200, text=payload, headers={"content-type": "text/event-stream"})


def client(handler, **kw) -> tuple[SGLangLLM, FakeServer]:
    fake = FakeServer(handler)
    return SGLangLLM("http://sglang.test/v1", transport=httpx.MockTransport(fake), **kw), fake


FREIGHT = get_domain("freight")
SCHEMA = json_schema(FREIGHT)
X_SYSTEM = system_prompt(FREIGHT)


# ------------------------------------------------------------------ wire format

def test_model_name_comes_from_the_server_when_auto():
    llm, _ = client(lambda b, n: (200, sse("{}")))
    assert llm.model == "Qwen/Qwen3-8B" and llm.name == "sglang:Qwen/Qwen3-8B"


def test_extract_request_is_greedy_grammar_constrained_and_never_thinks():
    llm, fake = client(lambda b, n: (200, sse('{"full_name": "Tony Alvarez"}')))
    r = llm.extract(X_SYSTEM, "LATEST CALLER MESSAGE:\nhi", SCHEMA)
    body = fake.bodies[0]
    assert body["stream"] is True and body["stream_options"] == {"include_usage": True}
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["temperature"] == 0.0 and "top_p" not in body
    rf = body["response_format"]
    assert rf["type"] == "json_schema" and rf["json_schema"]["strict"] is True
    assert r.value == {"full_name": "Tony Alvarez"} and r.stats.mode == "json_schema" and r.stats.parse_ok


def test_grammar_schema_drops_null_branches_and_bounds():
    g = grammar_schema(SCHEMA)
    blob = json.dumps(g)
    assert '"null"' not in blob and "maximum" not in blob and "title" not in blob
    assert g["required"] == [] and g["additionalProperties"] is False
    assert g["properties"]["month"] == {"type": "integer"}
    assert set(g["properties"]["category"]["enum"]) == {"reefer", "dry_van", "flatbed"}
    assert "dock_status" in g["properties"]["intent"]["enum"]
    clinic = grammar_schema(json_schema(get_domain("clinic")))
    assert "wait_time" in clinic["properties"]["intent"]["enum"]
    portfolio = grammar_schema(json_schema(get_domain("portfolio")))
    assert "status_update" in portfolio["properties"]["intent"]["enum"]


def test_generate_uses_qwen_sampling_without_thinking():
    llm, fake = client(lambda b, n: (200, sse("Hello there.")))
    r = llm.generate("system", "user")
    body = fake.bodies[0]
    assert body["temperature"] == 0.7 and body["top_p"] == 0.8 and body["top_k"] == 20
    assert body["chat_template_kwargs"] == {"enable_thinking": False} and "response_format" not in body
    assert r.value == "Hello there."


def test_thinking_flag_reaches_replies_only():
    llm, fake = client(lambda b, n: (200, sse("{}")), thinking=True)
    llm.extract(X_SYSTEM, "hi", SCHEMA)
    llm.generate("system", "user")
    assert fake.bodies[0]["chat_template_kwargs"]["enable_thinking"] is False
    assert fake.bodies[1]["chat_template_kwargs"]["enable_thinking"] is True


# ------------------------------------------------------------------ fallbacks

def test_rejected_json_schema_falls_back_and_the_mode_sticks():
    def handler(body, n):
        if body.get("response_format", {}).get("type") == "json_schema":
            return 400, "response_format json_schema is not supported by this backend"
        return 200, sse('{"dob": "1981-09-14"}')

    llm, fake = client(handler)
    r = llm.extract(X_SYSTEM, "hi", SCHEMA)
    assert r.value == {"dob": "1981-09-14"} and r.stats.mode == "json_object" and llm.extract_mode == "json_object"
    llm.extract(X_SYSTEM, "again", SCHEMA)
    assert [b.get("response_format", {}).get("type") for b in fake.bodies] == ["json_schema", "json_object", "json_object"]
    assert "matching this schema" in fake.bodies[1]["messages"][0]["content"]


def test_accepted_but_ignored_constraint_falls_back():
    """Some backends take response_format and decode freely anyway."""
    def handler(body, n):
        if body.get("response_format", {}).get("type") == "json_schema":
            return 200, sse("Sure! The caller's name is Tony.")
        return 200, sse('{"full_name": "Tony Alvarez"}')

    llm, _ = client(handler)
    r = llm.extract(X_SYSTEM, "hi", SCHEMA)
    assert r.stats.mode == "json_object" and r.value["full_name"] == "Tony Alvarez"


def test_once_a_mode_works_a_later_rejection_is_raised_not_hidden():
    def handler(body, n):
        return (200, sse("{}")) if n == 1 else (400, "grammar compilation failed")

    llm, _ = client(handler)
    llm.extract(X_SYSTEM, "hi", SCHEMA)
    with pytest.raises(ServerError) as e:
        llm.extract(X_SYSTEM, "again", SCHEMA)
    assert e.value.status == 400 and "grammar compilation failed" in e.value.message


def test_forced_prompt_mode_sends_no_response_format():
    llm, fake = client(lambda b, n: (200, sse('```json\n{"id_last4": "4417"}\n```')), extract_mode="prompt")
    r = llm.extract(X_SYSTEM, "hi", SCHEMA)
    assert "response_format" not in fake.bodies[0] and r.value == {"id_last4": "4417"} and r.stats.mode == "prompt"


def test_unknown_mode_is_refused():
    with pytest.raises(ValueError):
        client(lambda b, n: (200, sse("{}")), extract_mode="regex")


def test_prose_in_the_last_mode_is_a_counted_parse_failure():
    llm, _ = client(lambda b, n: (200, sse("I think the caller is Tony.")), extract_mode="prompt")
    r = llm.extract(X_SYSTEM, "hi", SCHEMA)
    assert r.value == {} and r.stats.parse_ok is False


# ------------------------------------------------------------------ output cleanup and metrics

@pytest.mark.parametrize("raw,clean", [
    ("<think>plan the reply</think>\n\nHello.", "Hello."),
    ("leftover reasoning</think>Hello.", "Hello."),
    ("Hello.", "Hello."),
])
def test_strip_think(raw, clean):
    assert strip_think(raw) == clean


def test_reasoning_content_is_counted_and_kept_out_of_the_reply():
    llm, _ = client(lambda b, n: (200, sse("Hello.", reasoning="The caller wants a status update.")))
    r = llm.generate("system", "user")
    assert r.value == "Hello." and r.stats.thinking_chars == len("The caller wants a status update.")


def test_stats_read_usage_cache_hits_and_timing():
    usage = {"prompt_tokens": 1200, "completion_tokens": 40, "prompt_tokens_details": {"cached_tokens": 1024}}
    llm, _ = client(lambda b, n: (200, sse(pieces=["Your ", "load ", "is ", "ready."], usage=usage)))
    s = llm.generate("system", "user").stats
    assert (s.input_tokens, s.output_tokens, s.cached_tokens) == (1200, 40, 1024)
    assert s.ttft_ms > 0 and s.latency_ms >= s.ttft_ms and s.decode_tps > 0 and s.stop_reason == "stop"


def test_missing_usage_falls_back_to_counting_stream_pieces():
    llm, _ = client(lambda b, n: (200, sse(pieces=["a", "b", "c"])))
    s = llm.generate("system", "user").stats
    assert s.output_tokens == 3 and s.cached_tokens == 0


def test_cache_hits_reported_by_the_sglang_extension_field():
    from app.model.llm import cached_tokens
    assert cached_tokens({"sglext": {"cached_tokens": 77}}) == 77
    assert cached_tokens({"cached_tokens": 5}) == 5
    assert cached_tokens({}) == 0


# ------------------------------------------------------------------ end to end

def test_full_turn_through_the_agent_with_a_mock_qwen():
    """Extraction and reply both come over the wire; the gate and guard still hold."""
    extraction = {"full_name": "Tony Alvarez", "dob": "1981-09-14", "id_last4": "4417", "party": "Peachtree",
                  "date": "2026-03-10", "intent": "dock_status"}

    def handler(body, n):
        if "response_format" in body:
            return 200, sse(json.dumps(extraction))
        return 200, sse("You're verified, Tony. Load LD-7731 at Peachtree is waiting for a door; the facility "
                        "estimates about 60 minutes.")

    llm, _ = client(handler)
    a = Agent(llm, "freight")
    s = SessionState()
    a.opening(s)
    r = a.handle(s, "Hi, it's Tony ...")
    assert s.verified_party_id == "D1" and s.selected_id == "LD-7731" and s.phase == Phase.PROCESS_CASE
    assert r.trace["reply_source"] == "model" and r.trace["guard"] == []
    assert [c["stage"] for c in r.trace["calls"]] == ["extract", "respond"]
    assert r.trace["calls"][0]["mode"] == "json_schema"


def test_an_invented_wait_time_from_the_model_is_replaced():
    def handler(body, n):
        if "response_format" in body:
            return 200, sse(json.dumps({"full_name": "Hannah Kim", "dob": "1988-07-19", "id_last4": "5530",
                                        "record_status": "waiting", "intent": "wait_time"}))
        return 200, sse("Good news, the doctor will see you in 5 minutes.")

    llm, _ = client(handler)
    a = Agent(llm, "clinic")
    s = SessionState()
    a.opening(s)
    r = a.handle(s, "hi")
    assert r.trace["reply_source"] == "template_guard" and "in 5 minutes" not in r.reply and "35" in r.reply


def test_a_server_abort_is_an_error_not_a_parse_failure():
    """Found live on MLX: a prompt longer than the token pool came back 200 with finish_reason abort."""
    llm, _ = client(lambda b, n: (200, sse("", finish="abort")))
    with pytest.raises(ServerError) as e:
        llm.extract(X_SYSTEM, "hi", SCHEMA)
    assert e.value.status == 503 and "max-total-tokens" in e.value.message
