"""Model providers behind one small interface.

Each turn makes two calls: extract() returns structured fields and generate()
returns the reply. Every call records latency, time to first token, decode
speed, tokens, and cache hits for the bench. The offline provider needs no
server and drives the tests.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any, Optional, Protocol


@dataclass
class CallStats:
    stage: str
    model: str
    latency_ms: float
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0         # prompt tokens served from the prefix cache
    ttft_ms: float = 0.0           # time to first token
    decode_tps: float = 0.0        # output tokens per second after the first
    stop_reason: str = ""          # stop, length, or empty
    mode: str = ""                 # extraction method: json_schema, json_object, or prompt
    parse_ok: bool = True          # False when extraction output wasn't a JSON object
    thinking_chars: int = 0        # reasoning text the server returned, if any


@dataclass
class LLMResult:
    value: Any
    stats: CallStats


class LLM(Protocol):
    name: str

    def extract(self, system: str, user: str, schema: dict, tool_name: str) -> LLMResult: ...

    def generate(self, system: str, user: str, max_tokens: int = 400) -> LLMResult: ...


# ---------------------------------------------------------------- helpers

_DROP_KEYS = {"title", "default", "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
              "minLength", "maxLength", "pattern", "multipleOf"}


def grammar_schema(schema: dict) -> dict:
    """Simplify a Pydantic schema for grammar constrained decoding.

    Every field is optional, so a missing field already means the caller didn't
    say it. Null branches and numeric bounds only make the grammar bigger; the
    validator in extract.py checks bounds after decoding.
    """
    def clean(node):
        if isinstance(node, dict):
            if "anyOf" in node:
                branches = [b for b in node["anyOf"] if b.get("type") != "null"]
                if len(branches) == 1:
                    return clean({**{k: v for k, v in node.items() if k != "anyOf"}, **branches[0]})
            out = {k: clean(v) for k, v in node.items() if k not in _DROP_KEYS}
            if out.get("type") == "object":
                out["additionalProperties"] = False
            return out
        if isinstance(node, list):
            return [clean(x) for x in node]
        return node

    s = clean(schema)
    s["required"] = []
    return s


def _loads(text: str) -> Optional[dict]:
    """Parse a JSON object from model text, ignoring code fences and stray words. None if there isn't one."""
    text = strip_think(text or "").strip()
    if text.startswith("```"):
        text = text.strip("`").split("\n", 1)[-1]
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        value = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


_THINK = re.compile(r"<think>.*?</think>", re.S)


def strip_think(text: str) -> str:
    """Remove Qwen3 reasoning blocks that leak into content when the server doesn't split them out."""
    text = _THINK.sub("", text)
    if "</think>" in text:                 # opening tag was in the prompt, closing tag in the output
        text = text.split("</think>", 1)[1]
    return text.replace("<think>", "").strip()


def cached_tokens(usage: dict) -> int:
    """Prefix cache hits. Servers report them in different places, or not at all."""
    details = usage.get("prompt_tokens_details") or {}
    for v in (details.get("cached_tokens"), usage.get("cached_tokens"),
              (usage.get("sglext") or {}).get("cached_tokens")):
        if isinstance(v, int):
            return v
    return 0


class ServerError(Exception):
    """An error response from the model server, with its message."""

    def __init__(self, status: int, message: str):
        super().__init__(f"HTTP {status}: {message}")
        self.status = status
        self.message = message


@dataclass
class _Stream:
    text: str
    reasoning: str
    usage: dict
    ttft_ms: float
    latency_ms: float
    decode_tps: float
    finish_reason: str
    pieces: int


# ---------------------------------------------------------------- SGLang

class SGLangLLM:
    """Qwen3 on SGLang through its OpenAI compatible API. Works with vLLM too.

    Extraction tries grammar constrained JSON first (json_schema), then JSON
    mode, then a plain prompt, and keeps the first one the server honors.
    Every call streams, so time to first token and decode speed are measured,
    not guessed.
    """

    MODES = ("json_schema", "json_object", "prompt")

    def __init__(self, base_url: str, model: str = "auto", api_key: Optional[str] = None,
                 extract_mode: str = "", thinking: bool = False, timeout: float = 120.0, transport=None):
        import httpx
        self.http = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout, transport=transport,
                                 headers={"Authorization": f"Bearer {api_key or 'none'}"})
        if extract_mode and extract_mode not in self.MODES:
            raise ValueError(f"extract_mode must be one of {self.MODES}")
        self.model = self._served_model() if model in ("", "auto") else model
        self.name = f"sglang:{self.model}"
        self.thinking = thinking            # applies to replies only; extraction never thinks
        self._forced = bool(extract_mode)
        self._extract_mode: Optional[str] = extract_mode or None

    @property
    def extract_mode(self) -> Optional[str]:
        return self._extract_mode

    def _served_model(self) -> str:
        r = self.http.get("/models")
        r.raise_for_status()
        data = r.json().get("data") or []
        return data[0]["id"] if data else "default"

    # ------------------------------------------------------------ transport

    def _body(self, system: str, user: str, max_tokens: int, temperature: float, thinking: bool) -> dict:
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
            # Qwen3 thinks by default. A support reply doesn't need it, and it costs seconds per turn.
            "chat_template_kwargs": {"enable_thinking": thinking},
        }
        if temperature > 0:
            body.update(top_p=0.8, top_k=20)   # Qwen3's suggested sampling without thinking
        return body

    def _stream(self, body: dict) -> _Stream:
        t0 = time.perf_counter()
        first: Optional[float] = None
        text, reasoning, usage, finish, pieces = [], [], {}, "", 0
        with self.http.stream("POST", "/chat/completions", json=body) as r:
            if r.status_code >= 400:
                r.read()
                try:
                    err = r.json()
                    msg = (err.get("error") or {}).get("message") if isinstance(err.get("error"), dict) else None
                    msg = msg or err.get("message") or r.text
                except ValueError:
                    msg = r.text
                raise ServerError(r.status_code, str(msg)[:400])
            for line in r.iter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if chunk.get("usage"):
                    usage = chunk["usage"]
                for ch in chunk.get("choices") or []:
                    delta = ch.get("delta") or {}
                    piece, think = delta.get("content") or "", delta.get("reasoning_content") or ""
                    if (piece or think) and first is None:
                        first = time.perf_counter()
                    if piece:
                        text.append(piece)
                        pieces += 1
                    if think:
                        reasoning.append(think)
                    if ch.get("finish_reason"):
                        finish = ch["finish_reason"]
        end = time.perf_counter()
        out_tokens = usage.get("completion_tokens") or pieces
        decode_s = end - first if first else 0.0
        tps = (out_tokens - 1) / decode_s if decode_s > 0 and out_tokens > 1 else 0.0
        return _Stream("".join(text), "".join(reasoning), usage,
                       ((first or end) - t0) * 1000, (end - t0) * 1000, tps, finish, pieces)

    def _stats(self, stage: str, s: _Stream, latency_ms: float, mode: str = "", parse_ok: bool = True) -> CallStats:
        u = s.usage
        return CallStats(stage=stage, model=self.model, latency_ms=latency_ms,
                         input_tokens=u.get("prompt_tokens", 0) or 0,
                         output_tokens=u.get("completion_tokens", 0) or s.pieces,
                         cached_tokens=cached_tokens(u), ttft_ms=round(s.ttft_ms, 3),
                         decode_tps=round(s.decode_tps, 2), stop_reason=s.finish_reason or "",
                         mode=mode, parse_ok=parse_ok, thinking_chars=len(s.reasoning))

    # ------------------------------------------------------------ calls

    def _extract_body(self, mode: str, system: str, user: str, schema: dict) -> dict:
        g = grammar_schema(schema)
        if mode == "json_schema":
            sys = system + "\n\nReply with only a JSON object. Omit fields the caller did not give."
        else:
            sys = (system + "\n\nReply with only a JSON object matching this schema, omitting fields the "
                   "caller did not give:\n" + json.dumps(g))
        body = self._body(sys, user, 512, 0.0, thinking=False)
        if mode == "json_schema":
            body["response_format"] = {"type": "json_schema",
                                       "json_schema": {"name": "turn", "schema": g, "strict": True}}
        elif mode == "json_object":
            body["response_format"] = {"type": "json_object"}
        return body

    def extract(self, system: str, user: str, schema: dict, tool_name: str = "turn") -> LLMResult:
        """Extract fields, falling back until a method works, then keep it.

        A server can reject a mode outright (HTTP 400) or accept it and ignore
        it, returning prose. Both move to the next mode, but only until one
        works; after that a failure is reported, not hidden.
        """
        t0 = time.perf_counter()
        modes = [self._extract_mode] if self._extract_mode else list(self.MODES)
        last_err: Optional[Exception] = None
        for i, mode in enumerate(modes):
            try:
                s = self._stream(self._extract_body(mode, system, user, schema))
            except ServerError as e:
                if e.status in (400, 404, 422) and self._extract_mode is None:
                    last_err = e
                    continue
                raise
            value = _loads(s.text)
            ok = value is not None
            if not ok and self._extract_mode is None and i < len(modes) - 1:
                continue                     # accepted but ignored: try the next mode
            if ok and self._extract_mode is None:
                self._extract_mode = mode
            stats = self._stats("extract", s, (time.perf_counter() - t0) * 1000, mode, ok)
            return LLMResult(value or {}, stats)
        raise last_err or RuntimeError("no extraction mode worked")

    def generate(self, system: str, user: str, max_tokens: int = 400) -> LLMResult:
        t0 = time.perf_counter()
        s = self._stream(self._body(system, user, max_tokens, 0.7, thinking=self.thinking))
        return LLMResult(strip_think(s.text), self._stats("respond", s, (time.perf_counter() - t0) * 1000))


# ---------------------------------------------------------------- factory

def make_llm(provider: str, model: str = "auto", base_url: Optional[str] = None,
             extract_mode: str = "", api_key: Optional[str] = None, thinking: bool = False):
    if provider == "offline":
        from .offline import OfflineLLM
        return OfflineLLM()
    if provider in ("sglang", "openai-compat"):
        from ..config import settings
        return SGLangLLM(base_url or settings.base_url, model, api_key, extract_mode, thinking)
    raise ValueError(f"unknown provider {provider!r}; use sglang or offline")
