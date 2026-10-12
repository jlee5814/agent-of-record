# gated-agent

A support agent for bank card disputes. Code decides what the agent may say. A local Qwen3-8B, served by SGLang on Apple Silicon through its MLX backend, only reads the caller's words and writes the reply.

The bank, the cardholders, and the disputes are all made up. Acme Bank doesn't exist.

## Why I built it

I wanted to know how far a small local model can go in a job with real rules: verify the caller, never leak an account, never promise money. My answer was to stop asking the model to follow those rules and take them away from it. The model proposes; code decides. An 8B model is plenty for reading messy language and writing a polite sentence. It isn't the thing I want guarding someone's account.

Running it locally also made the serving side worth studying. Each turn makes two calls with long fixed prompts, and one of them must return valid JSON. SGLang is built for both.

## How a turn works

1. **Extract.** Qwen reads the latest message and returns fields: identity details, hints about which dispute, tone, whether the caller wants a person. It sees the conversation only, never the records.
2. **Control.** Plain Python takes those fields and runs the procedure. It is the only code that changes state.
3. **Respond.** Qwen gets the controller's decisions and the facts it may use, and writes the reply. Before verification, the facts are empty.
4. **Guard.** Code checks the reply. Before verification, any dispute number, amount, merchant, or record value blocks it. After, any number or amount not in the approved facts blocks it. A blocked reply becomes a fixed template, and the trace says so.

## What code decides, not the model

- **Verification.** Three of five details (name, date of birth, phone, email, card last four) must match one cardholder, with no detail contradicting the record. The account number helps find the record but doesn't count. A miss never says which detail was wrong.
- **Lockout.** Three misses end the chat. Five misses against one cardholder within thirty minutes lock that cardholder out of self service across every session, so opening a new chat doesn't reset the count.
- **Someone calling for the cardholder.** They must be on file as an authorized user, and the cardholder must approve a request in their banking app. The caller's word is never consent.
- **A new speaker.** If someone new speaks after verification, the gate relocks and both model calls lose sight of everything said before.
- **Which dispute.** Hints the caller gives early (Bluewater, the laptop, February) are saved before verification and used after it. Hard filters pick the dispute; two matches mean the agent asks.
- **Money.** The record's amounts reach the model as sentences, not fields. The disputed amount is what's contested, not a refund. A provisional credit is temporary. I added this after seeing how easily a model reads a number next to a dispute as money coming back.

## Why SGLang

- **Structured output.** Extraction asks for grammar constrained JSON through `response_format`, so a small model can't return broken JSON. Some backends reject that, and some accept it and ignore it. The provider tries the JSON schema, then JSON mode, then a plain prompt, keeps the first that works, and records which one it used.
- **Prefix caching.** Both system prompts are fixed and long. RadixAttention reuses them every turn, and the bench reports how much of the input came from cache.
- **Streaming.** Every call streams, so I measure time to first token and decode speed instead of guessing at them from total latency.
- **Thinking off.** Qwen3 thinks by default. For a two sentence support reply, that costs seconds a turn for little gain, so I turn it off through `chat_template_kwargs`. `--thinking` turns it back on for comparison. Extraction never thinks.

## Run it

**1. Serve Qwen3 with SGLang on MLX.** Install SGLang for Apple Silicon by following [its Metal guide](https://docs.sglang.io/docs/hardware-platforms/apple_metal), then:

```bash
SGLANG_USE_MLX=1 python -m sglang.launch_server \
  --model-path Qwen/Qwen3-8B \
  --disable-cuda-graph \
  --enable-cache-report \
  --port 30000
```

`--enable-cache-report` makes SGLang return cached token counts. For less memory, swap in a 4 bit MLX build such as `mlx-community/Qwen3-8B-4bit`.

**2. Start the agent.**

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
python -m uvicorn app.server:app --port 8000
```

Open http://localhost:8000. The side panel shows the procedure, the gate, what the agent remembered, and the timing of each step, including time to first token and cache hits. Settings switch the extraction method between chats.

With no SGLang server running, the page falls back to offline mode: regex extraction and template replies. It exists so the tests and the demo run anywhere.

## Bench

```bash
python -m bench.run                                    # every scenario once
python -m bench.run --repeats 3 --concurrency 4        # more samples, parallel chats
python -m bench.run --extract-mode json_schema --extract-mode json_object --extract-mode prompt
python -m bench.run --thinking                         # what thinking costs and buys
python -m bench.run --provider offline                 # no server
```

Seventeen scripted callers cover the gate, prompt injection, wrong details, a spouse calling with and without consent, a speaker switch mid chat, an ambiguous dispute, a refund trap, a provisional credit trap, and callers writing in broken English and in Spanish. Every check is code, not a judge model.

The report gives pass rate, server errors (kept apart from failures so a timeout never reads as a leak), guard interventions, extraction parse failures, latency per step, time to first token, decode speed, cache hit share, and chats per minute. Rollouts are saved with every caller line and reply in `bench/results/`.

## Tests

```bash
pytest
```

92 tests, no server needed. A mock transport plays SGLang, so the tests pin what goes over the wire: the request shape, the extraction fallbacks, stripping stray think tags, reading cache hits from wherever the server puts them, and a full turn through the agent.

## Tradeoffs

- **The guard checks numbers, not meaning.** It catches an invented amount but not a sentence promising a real amount back. The bench catches those. A second model pass could catch them live, at the cost of another call every turn.
- **Templates are the floor.** When the guard blocks a reply, the caller gets a correct but stiff answer. I would rather be stiff than wrong.
- **The offline extractor is English only.** The language scenarios run against Qwen alone.
- **One machine.** MLX on a laptop serves one model for one person well. The concurrency flag shows where batching stops helping on this hardware.

## Layout

```
app/
  controller.py   the procedure: extract, control, respond, guard
  verify.py       identity matching, lockout, consent
  disputes.py     dispute lookup, facts as sentences, document guidance
  extract.py      extraction schema and prompt
  respond.py      reply prompt, templates, guard
  llm.py          SGLang provider: streaming, fallbacks, metrics
  offline.py      regex extraction for tests and the demo without a server
data/             synthetic Acme Bank fixtures
bench/            scenarios and the runner
tests/
web/index.html    chat page and state panel
```
