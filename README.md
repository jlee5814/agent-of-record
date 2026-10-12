# agent-of-record

An agent for any system of record.

In every job I've had, two records had to agree while someone waited:

- a driver's clock against the signed times at the dock
- a patient's appointment against the real queue
- an investment manager's book against the custodian bank's

So I built an agent that checks every reply against the record before the caller sees it. A local Qwen3-8B, served by SGLang on Apple Silicon through its MLX backend, reads the caller's words and writes the reply.

Code decides everything else:

- who the caller is
- which record they mean
- what may be said

Then a guard reconciles the reply against the record. Every amount, time, and ID must match, or the reply is replaced. The model is the working copy; the record is the book of record.

## The three desks

**Freight.** A driver calls AKT from a dock. How much longer until I get a door? Am I getting detention for this? Why did you dock $180 from my pay?

**Clinic.** A patient messages AF Clinic from the waiting room. When's the doctor coming? Am I next? Why do I owe $60?

**Portfolio.** A client of EP Capital asks why yesterday's value for their trust account doesn't match their Harbor Trust Bank statement. Is my money missing? When will it match?

The desks look different but share a shape: a private record, money that's easy to promise by accident, and a status that changes while the person waits. The records, the banks, and every person in the data are made up.

## Why I built it

I wanted to see how far a small local model can go in a job with real rules: verify the caller, never leak a record, never promise money or a time. My answer was to take the rules away from the model. An 8B model reads messy language and writes a polite sentence well. Code does the rest.

## The split: model and code

The code lives in two packages with one seam between them.

| | `app/model/` | `app/core/` |
|---|---|---|
| Job | Read the caller's words; write the reply | Decide everything else |
| Behavior | Probabilistic | Deterministic |
| Sees | The conversation, then the directives and approved facts | The records, the session, the rules |
| Changes state | Never | Only in the procedure |
| Files | `extract.py`, `reply.py`, `llm.py`, `offline.py` | `procedure.py`, `verify.py`, `guard.py`, `extraction.py`, `records.py`, `templates.py`, domains |

`app/agent.py` is the seam; it alone imports both. A test reads every import and fails if core imports the model, or if the model imports the code that decides: the procedure, the verifier, or the guard. The rule lives in code, not in a prompt.

## How a turn works

1. **Extract** (model). Qwen reads the latest message and proposes fields: identity details, hints about the record, what the caller wants, tone, and any emergency. It sees the conversation, never the records. `core/extraction.py` validates the proposal and drops any bad field.
2. **Decide** (code). The procedure runs the gates, picks the record, and emits directives plus the facts the reply may use. No other code changes state.
3. **Reply** (model). Qwen turns the directives and approved facts into a sentence. Before verification, it gets no facts.
4. **Guard** (code). Before verification, the reply may contain nothing from a record. After, every record number, dollar amount, clock time, and duration must appear in the approved facts or in the caller's own words. A reply that fails becomes a fixed template from `core/templates.py`, and the trace says so.

## The rules

- **Verification.** Three of five details must match one person, and none may contradict the record: name, date of birth, phone, email, and the last four digits of a CDL, an insurance member ID, or a custodian account. An account or patient number helps find the record but doesn't count. A miss never says which detail was wrong.
- **Lockout.** Three misses end the chat. Five misses against one person within thirty minutes lock that person out of self service in every session.
- **Proxies.** A spouse calling for a driver, a daughter for her father, or an advisor for a client must be on file, and the account holder must approve in their app or portal. The caller's word is never consent.
- **A new speaker.** When someone new speaks after verification, the gate relocks and both model calls forget everything said before.
- **Which record.** Hints given early (Peachtree, today, Dr. Patel, my trust account) are saved before verification and used after. Hard filters pick the record; two matches mean the agent asks.
- **Status.** Dock, queue, and research status come from a feed, and each ask reads the next update: door assigned, loading, loaded; third in line, second, next; under research, partially matched, matched. Every estimate says it's an estimate, and nobody promises when a break will clear.
- **Money.** Record fields reach the model as plain sentences. A detention claim under review is not money owed. A copay estimate is not the bill. A deduction comes with its reason. A reconciliation break is a difference with named causes, not a loss, and the custodian's book is the official one.
- **Limits.** Approving detention, reading lab results, giving investment advice, or taking a trade or wire instruction gets a polite no and the right contact. An injury, chest pain, or a transfer the client didn't make skips everything: 911 or the custodian's fraud line first, then a person.

## Why SGLang

- **Structured output.** Extraction asks for grammar constrained JSON through `response_format`, so a small model can't return broken JSON. Some backends reject that, and some accept it and ignore it. The provider tries the JSON schema, then JSON mode, then a plain prompt, keeps the first that works, and records which one it used.
- **Prefix caching.** Each domain's two system prompts are fixed and long. RadixAttention reuses them every turn, and the bench reports how much input came from cache.
- **Streaming.** Every call streams, so time to first token and decode speed are measured, not inferred.
- **Thinking off.** Qwen3 thinks by default, which costs seconds a turn for a two sentence reply. I turn it off through `chat_template_kwargs`; `--thinking` turns it back on for comparison. Extraction never thinks.

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

Open http://localhost:8000 and pick a desk in Settings. The buttons under the chat fill in example callers. The side panel shows the procedure, the gate, what the agent remembered, the last status update, and the time each step took.

Without an SGLang server, the page falls back to offline mode: regex extraction and template replies, so the tests and the demo run anywhere.

## Bench

```bash
python -m bench.run                                    # every domain, every scenario once
python -m bench.run --domain clinic --repeats 3        # one domain, more samples
python -m bench.run --extract-mode json_schema --extract-mode json_object --extract-mode prompt
python -m bench.run --thinking                         # what thinking costs and buys
python -m bench.run --provider offline                 # no server
```

61 scripted callers cover the three desks: the gate, prompt injection, wrong details, proxies with and without consent, a speaker switch, an ambiguous record, repeated status checks, money and time traps, a request the agent can't grant, an emergency, and callers writing in broken English, Spanish, or Korean. Every check is code, not a judge model.

The report gives pass rate, server errors, guard interventions, parse failures, latency per step, time to first token, decode speed, cache hit share, and chats per minute. Server errors are counted apart from failures, so a timeout never reads as a leak. Every rollout, with each caller line and reply, lands in `bench/results/`.

## Adding a domain

A new domain is one Python file and one data folder. Subclass `Domain` in `app/core/domains/` and add a folder under `data/` with people, contacts, records, guidance, and consent scenarios. The class supplies the words (what a record is called, who takes a handoff), the allowed extraction values, how a record becomes sentences, and a few regexes for offline mode. The procedure, the gates, and the guard stay the same.

## Tests

```bash
pytest
```

159 tests, no server needed. The gate tests run in every domain, and a boundary test keeps the model and the code apart. A mock transport stands in for SGLang, so the tests pin the request shape, the extraction fallbacks, stray think tags, cache reporting, and a full turn where the model invents a wait time and the guard replaces it.

## Tradeoffs

- **The guard checks numbers, not meaning.** It catches an invented wait time but not a promise of a real amount. The bench catches those. A second model pass could catch them live, at the cost of another call each turn.
- **Templates are the floor.** When the guard blocks a reply, the caller gets a correct but stiff answer. I would rather be stiff than wrong.
- **The status feed is scripted.** Each ask reads the next update, not a clock, which keeps the bench repeatable. A real deployment would read live dock and queue systems.
- **The offline extractor is English only.** The language scenarios run against Qwen alone.

## Layout

```
app/
  agent.py          the seam: extract, decide, reply, guard
  core/             deterministic
    procedure.py    the gates and the state machine; the only state writer
    verify.py       identity matching, lockout, consent
    extraction.py   the contract: fields the model may propose, and their validation
    records.py      record lookup and document guidance
    guard.py        reconciles each reply against the record
    templates.py    fixed wording for every directive
    domain.py       what a domain supplies, and the registry
    domains/        freight.py, clinic.py, portfolio.py
  model/            probabilistic
    extract.py      the extraction prompt
    reply.py        the reply prompt
    llm.py          SGLang provider: streaming, fallbacks, metrics
    offline.py      a regex stand in for the model, for tests and the demo
data/               synthetic fixtures, one folder per domain
bench/              scenarios and the runner
tests/
web/index.html      chat page and state panel
```
