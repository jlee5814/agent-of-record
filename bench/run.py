"""Run every scenario against a model server, concurrently.

    python -m bench.run --provider offline
    python -m bench.run --domain clinic --repeats 3 --concurrency 4
    python -m bench.run --extract-mode json_schema --extract-mode prompt

Reports pass rate, server errors, guard interventions, latency per step, time
to first token, decode speed, tokens, prefix cache hits, and extraction parse
failures. Writes rollouts, JSON, and Markdown to bench/results/.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402
from app.agent import Agent  # noqa: E402
from app.core.domain import DOMAINS  # noqa: E402
from app.model.llm import make_llm  # noqa: E402
from app.core.state import SessionState  # noqa: E402
from app.core.summary import Outbox  # noqa: E402
from bench.scenarios import SCENARIOS, Scenario  # noqa: E402


# Event kinds that mark code correcting or backing up the model.
CORRECTIONS = ("identity_not_typed", "hints_from_text", "signals_from_text", "guard_blocked", "empty_model_reply",
               "extract_parse_failed")


@dataclass
class Rollout:
    label: str
    scenario: str
    repeat: int
    checks: dict[str, bool] = field(default_factory=dict)
    guard_blocks: int = 0
    turns: list[dict] = field(default_factory=list)
    error: str | None = None
    wall_ms: float = 0.0
    corrections: dict = field(default_factory=dict)    # times code corrected or backed up the model


def run_one(llm_factory, domain: str, label: str, sc: Scenario, rep: int, outbox_dir: Path) -> Rollout:
    ro = Rollout(label=label, scenario=sc.name, repeat=rep)
    t0 = time.perf_counter()
    try:
        agent = Agent(llm_factory(), domain, outbox=Outbox(outbox_dir))
        s = SessionState()
        s.consent.scenario = sc.consent
        agent.opening(s)
        replies = []
        for text in sc.turns:
            r = agent.handle(s, text)
            replies.append(r.reply)
            ro.turns.append({"timing_ms": r.trace["timing_ms"], "calls": r.trace["calls"],
                             "guard": r.trace["guard"], "phase_after": r.trace["phase_after"],
                             "reply_source": r.trace.get("reply_source", "model"),
                             "caller": text[:300], "reply": r.reply[:1200]})
            ro.guard_blocks += bool(r.trace["guard"])
        ro.checks = {name: bool(fn(s, replies)) for name, fn in sc.checks.items()}
        for e in s.events:
            if e.kind in CORRECTIONS:
                ro.corrections[e.kind] = ro.corrections.get(e.kind, 0) + 1
    except Exception as e:  # a failed rollout is data, not a crash
        ro.error = f"{type(e).__name__}: {e}"[:300]
        ro.checks = {name: False for name in sc.checks}
    ro.wall_ms = (time.perf_counter() - t0) * 1000
    return ro


def pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = (len(xs) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def _p(xs: list[float]) -> dict:
    return {"p50": round(pct(xs, .5), 1), "p95": round(pct(xs, .95), 1)}


def summarize(label: str, rollouts: list[Rollout], wall_s: float) -> dict:
    # Judge behavior only on completed rollouts; count server failures separately,
    # so a timeout can never read as a safety failure.
    completed = [r for r in rollouts if not r.error]
    checks = [v for r in completed for v in r.checks.values()]
    turns = [t for r in rollouts for t in r.turns]
    calls = [c for t in turns for c in t["calls"]]
    extract = [c for c in calls if c["stage"] == "extract"]
    respond = [c for c in calls if c["stage"] == "respond"]
    stage = {k: [t["timing_ms"][k] for t in turns] for k in ("extract", "control", "respond", "guard", "total")}
    tin = sum(c["input_tokens"] for c in calls)
    cached = sum(c.get("cached_tokens", 0) for c in calls)
    n = max(len(rollouts), 1)
    modes: dict[str, int] = {}
    for c in extract:
        modes[c.get("mode") or "?"] = modes.get(c.get("mode") or "?", 0) + 1
    return {
        "label": label,
        "rollouts": len(rollouts),
        "errors": sum(1 for r in rollouts if r.error),
        "check_pass_rate": round(sum(checks) / max(len(checks), 1), 3),
        "scenarios_all_passed": sum(1 for r in completed if r.checks and all(r.checks.values())),
        "completed": len(completed),
        "guard_interventions": sum(r.guard_blocks for r in rollouts),
        "empty_model_replies": sum(1 for t in turns if t.get("reply_source") == "template_empty_model_reply"),
        "extract_parse_failures": sum(1 for c in extract if not c.get("parse_ok", True)),
        "extract_modes": modes,
        "turns": len(turns),
        "latency_ms": {k: _p(v) for k, v in stage.items()},
        "ttft_ms": {"extract": _p([c["ttft_ms"] for c in extract if c.get("ttft_ms")]),
                    "respond": _p([c["ttft_ms"] for c in respond if c.get("ttft_ms")])},
        "decode_tps": {"extract": _p([c["decode_tps"] for c in extract if c.get("decode_tps")]),
                       "respond": _p([c["decode_tps"] for c in respond if c.get("decode_tps")])},
        "tokens_per_conversation": {"input": round(tin / n), "output": round(sum(c["output_tokens"] for c in calls) / n)},
        "prefix_cache_hit_share": round(cached / max(tin, 1), 3),
        "throughput_conversations_per_min": round(len(rollouts) / max(wall_s, 1e-9) * 60, 1),
        "corrections": {k: sum(r.corrections.get(k, 0) for r in rollouts) for k in CORRECTIONS},
        "failed_checks": sorted({f"{r.scenario}:{k}" for r in completed for k, v in r.checks.items() if not v}),
        "error_samples": [r.error for r in rollouts if r.error][:3],
    }


def to_markdown(results: list[dict]) -> str:
    head = ("| Run | Checks passed | Scenarios fully passed | Errors | Guard | Parse failures | Turn p50 / p95 ms | "
            "Extract p50 | Respond p50 | Respond TTFT p50 | Decode tok/s p50 | Cache hit share | Conv per min |\n"
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    rows = []
    for r in results:
        L = r["latency_ms"]
        rows.append(f"| {r['label']} | {r['check_pass_rate']:.0%} | {r['scenarios_all_passed']}/{r['completed']} | "
                    f"{r['errors']} | {r['guard_interventions']} | {r['extract_parse_failures']} | "
                    f"{L['total']['p50']:.0f} / {L['total']['p95']:.0f} | {L['extract']['p50']:.0f} | "
                    f"{L['respond']['p50']:.0f} | {r['ttft_ms']['respond']['p50']:.0f} | "
                    f"{r['decode_tps']['respond']['p50']:.1f} | {r['prefix_cache_hit_share']:.0%} | "
                    f"{r['throughput_conversations_per_min']} |")
    notes = "\n".join(f"- **{r['label']}** extraction modes {r['extract_modes']}; code corrections "
                      f"{ {k: v for k, v in r['corrections'].items() if v} }; failed: "
                      f"{', '.join(r['failed_checks']) or 'none'}" for r in results)
    return head + "\n".join(rows) + "\n\n" + notes + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default=os.getenv("LLM_PROVIDER", "sglang"))
    ap.add_argument("--base-url", default=settings.base_url)
    ap.add_argument("--model", default=settings.model)
    ap.add_argument("--domain", action="append", choices=DOMAINS, help="freight or clinic; repeat for both (default)")
    ap.add_argument("--extract-mode", action="append", default=[],
                    help="json_schema, json_object, or prompt. Repeat to compare. Default tries each in turn.")
    ap.add_argument("--thinking", action="store_true", help="let Qwen3 think before each reply")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--only", help="comma separated scenario names or tags")
    args = ap.parse_args()

    def pick(domain: str) -> list[Scenario]:
        scenarios = SCENARIOS[domain]
        if args.provider == "offline":
            scenarios = [sc for sc in scenarios if "live_only" not in sc.tags]
        if args.only:
            want = set(args.only.split(","))
            scenarios = [sc for sc in scenarios if sc.name in want or want & set(sc.tags)]
        return scenarios

    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(exist_ok=True)
    results = []
    runs = [(d, m) for d in (args.domain or list(DOMAINS)) for m in (args.extract_mode or [""])]
    for domain, mode in runs:
        # Build one client up front: it fails fast if the server is down and resolves the model name.
        probe = make_llm(args.provider, args.model, args.base_url, mode, os.getenv("OPENAI_API_KEY"), args.thinking)
        model = getattr(probe, "model", "rules")
        label = f"{domain} {model} [{mode or 'auto'}]" if args.provider != "offline" else f"{domain} rules"
        scenarios = pick(domain)

        def factory(mode=mode, model=model):
            return make_llm(args.provider, model, args.base_url, mode, os.getenv("OPENAI_API_KEY"), args.thinking)

        jobs = [(sc, rep) for sc in scenarios for rep in range(args.repeats)]
        t0 = time.perf_counter()
        rollouts: list[Rollout] = []
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futs = [pool.submit(run_one, factory, domain, label, sc, rep, out_dir / "outbox") for sc, rep in jobs]
            for f in as_completed(futs):
                rollouts.append(f.result())
        summary = summarize(label, rollouts, time.perf_counter() - t0)
        results.append(summary)
        slug = label.replace("/", "_").replace(" ", "").replace("[", "-").replace("]", "")
        (out_dir / f"rollouts-{slug}.jsonl").write_text("\n".join(json.dumps(r.__dict__) for r in rollouts))
        print(f"{label}: {summary['check_pass_rate']:.0%} checks, {summary['errors']} errors, "
              f"{summary['guard_interventions']} guard, {summary['extract_parse_failures']} parse failures, "
              f"turn p50 {summary['latency_ms']['total']['p50']} ms, "
              f"reply TTFT p50 {summary['ttft_ms']['respond']['p50']} ms", flush=True)

    (out_dir / "summary.json").write_text(json.dumps(results, indent=2))
    md = to_markdown(results)
    (out_dir / "summary.md").write_text(md)
    print("\n" + md)


if __name__ == "__main__":
    main()
