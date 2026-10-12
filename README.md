# gated-agent

A support agent for three jobs I've done: driving a truck, working in a clinic, and portfolio operations at an investment manager. Code decides what the agent may say. A local Qwen3-8B, served by SGLang on Apple Silicon through its MLX backend, only reads the caller's words and writes the reply.

The carrier, the clinic, the investment manager, the banks, and every person in the data are made up.

## The three desks

**Freight.** A driver calls Redline Freight from a dock. How much longer until I get a door? Am I getting detention for this? Why did you dock $180 from my pay?

**Clinic.** A patient messages Maple Grove Family Clinic from the waiting room. When's the doctor coming? Am I next? Why do I owe $60?

**Portfolio.** A client of Northbridge Capital asks why yesterday's value for their trust account doesn't match the statement from Harbor Trust Bank. The bank is the custodian: it holds the client's money and keeps the official book of record. The manager keeps its own book for trading and reporting, and portfolio operations reconciles the two every day. A gap is a break to explain, such as a trade the manager books on trade date and the custodian on settlement date, interest one side accrues daily and the other posts on the coupon date, or two pricing vendors for the same bond. It is never money gone missing.

I picked these because I've worked all three. They look different but share a shape: a private record, money that's easy to promise or misstate by accident, and a status that keeps changing while the person is still asking. Same engine, three domains. A new domain is one Python file and one data folder.

## Why I built it

I wanted to see how far a small local model can go in a job with real rules: verify the caller, never leak a record, never promise money or a time. My answer was to take those rules away from the model. The model proposes; code decides. An 8B model is plenty for reading messy language and writing a polite sentence. It isn't the thing I want guarding someone's pay, their health record, or their retirement account.

## How a turn works

1. **Extract.** Qwen reads the latest message and returns fields: identity details, hints about which record, what the caller wants, tone, and whether it's an emergency. It sees the conversation only, never the records.
2. **Control.** Plain Python runs the procedure. It is the only code that changes state.
3. **Respond.** Qwen gets the controller's decisions and the facts it may use, and writes the reply. Before verification, the facts are empty.
4. **Guard.** Code checks the reply. Before verification, any record number, amount, facility, provider, bank, account name, or record value blocks it. After, any record number, dollar amount, clock time, or number of minutes not in the approved facts (or said by the caller) blocks it. A blocked reply becomes a fixed template, and the trace says so.

## What code decides, not the model

- **Verification.** Three of five details (name, date of birth, phone, email, and the last four of a CDL, an insurance member ID, or a custodian account) must match one person, with nothing contradicting the record. The driver ID, patient number, or Northbridge account number helps find the record but doesn't count. A miss never says which detail was wrong.
- **Lockout.** Three misses end the chat. Five misses against one person within thirty minutes lock that person out of self service across every session.
- **Someone calling for them.** A spouse calling for a driver, a daughter for her father, or an advisor for a client must be on file, and the account holder must approve a request in their app or portal. The caller's word is never consent.
- **A new speaker.** If someone new speaks after verification, the gate relocks and both model calls lose sight of everything said before.
- **Which record.** Hints the caller gives early (Peachtree, today, Dr. Patel, my trust account) are saved before verification and used after. Hard filters pick the record; two matches mean the agent asks.
- **Status that changes.** Dock, queue, and break research status come from a feed. Asking again reads the next update: door assigned, then loading, then loaded; third in line, then second, then next; under research, then partially matched, then matched. Every estimate is labeled as one, and no one promises when a break will clear.
- **Money as sentences.** Record fields reach the model as plain statements. A detention claim under review is not money owed. A copay estimate is not the bill. A deduction comes with its reason. A reconciliation break is a difference with named causes, not a loss, and the custodian's book is the official one.
- **Things the agent can't do.** Approving detention, reading lab results, giving investment advice, or taking a trade or wire instruction gets a polite no and a pointer to the right people. An injury, chest pain, or a transfer the client didn't make skips everything and hands off at once, with 911 or the custodian's fraud line first.

## Why SGLang

- **Structured output.** Extraction asks for grammar constrained JSON through `response_format`, so a small model can't return broken JSON. Some backends reject that, and some accept it and ignore it. The provider tries the JSON schema, then JSON mode, then a plain prompt, keeps the first that works, and records which one it used.
- **Prefix caching.** Each domain's two system prompts are fixed and long. RadixAttention reuses them every turn, and the bench reports how much input came from cache.
- **Streaming.** Every call streams, so I measure time to first token and decode speed directly.
- **Thinking off.** Qwen3 thinks by default. For a two sentence reply that costs seconds a turn for little gain, so I turn it off through `chat_template_kwargs`. `--thinking` turns it back on for comparison. Extraction never thinks.

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

Open http://localhost:8000 and pick freight, clinic, or portfolio in Settings. The buttons under the chat fill in example callers. The side panel shows the procedure, the gate, what the agent remembered, which status update it last read, and the timing of each step.

With no SGLang server running, the page falls back to offline mode: regex extraction and template replies. It exists so the tests and the demo run anywhere.

## Bench

```bash
python -m bench.run                                    # every domain, every scenario once
python -m bench.run --domain clinic --repeats 3        # one domain, more samples
python -m bench.run --extract-mode json_schema --extract-mode json_object --extract-mode prompt
python -m bench.run --thinking                         # what thinking costs and buys
python -m bench.run --provider offline                 # no server
```

About twenty scripted callers per domain, 61 in all. Each domain covers the gate, prompt injection, wrong details, a proxy with and without consent, a speaker switch, an ambiguous record, repeated status checks, money traps (detention, copay, a break called a loss, a credit promise), a time trap, a request the agent can't grant, an emergency, and callers writing in broken English and in Spanish or Korean. Every check is code, not a judge model.

The report gives pass rate, server errors (kept apart from failures so a timeout never reads as a leak), guard interventions, extraction parse failures, latency per step, time to first token, decode speed, cache hit share, and chats per minute. Rollouts are saved with every caller line and reply in `bench/results/`.

## Adding a domain

Subclass `Domain` in `app/domains/`, then add a folder under `data/` with people, contacts, records, guidance, and consent scenarios. The class supplies the words (what a record is called, who a handoff goes to), the allowed values for extraction, how a record becomes sentences, and a few regexes for offline mode. The procedure, the gates, and the guard don't change.

## Tests

```bash
pytest
```

155 tests, no server needed. Gate tests run in every domain. A mock transport plays SGLang, so the tests pin the request shape, the extraction fallbacks, stray think tags, cache hit reporting, and full turns where the model invents a wait time and the guard replaces it.

## Tradeoffs

- **The guard checks numbers, not meaning.** It catches an invented wait time but not a sentence promising a real amount. The bench catches those. A second model pass could catch them live, at the cost of another call every turn.
- **Templates are the floor.** When the guard blocks a reply, the caller gets a correct but stiff answer. I would rather be stiff than wrong.
- **The status feed is scripted.** Each ask reads the next update rather than a clock. It keeps the bench repeatable; a real deployment would read live dock and queue systems.
- **The offline extractor is English only.** The language scenarios run against Qwen alone.

## Layout

```
app/
  controller.py   the procedure: extract, control, respond, guard
  domain.py       what a domain supplies, and the registry
  domains/        freight.py, clinic.py, portfolio.py
  verify.py       identity matching, lockout, consent
  records.py      record lookup and document guidance
  extract.py      extraction schema and prompt, built per domain
  respond.py      reply prompt, templates, guard
  llm.py          SGLang provider: streaming, fallbacks, metrics
  offline.py      regex extraction for tests and the demo without a server
data/             synthetic fixtures, one folder per domain
bench/            scenarios and the runner
tests/
web/index.html    chat page and state panel
```
