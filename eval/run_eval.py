"""Eval harness for the v2 multi-agent pipeline.

Two suites:

  factual   — 22 questions whose ground truth is recomputed from the database.
              Measures whether the pipeline states the right number, and
              whether the Critic's confidence tracks correctness.

  injection — fixed draft notes containing deliberately fabricated figures.
              Measures the Critic's catch rate on hallucinated numbers and,
              just as importantly, its false-positive rate on clean ones.

The critic-off baseline on the injection suite is 0% by construction: with no
verifier in the loop, the draft is exactly what the user sees, so every
fabricated figure reaches them unflagged.

Usage:
    python -m eval.run_eval                    # both suites
    python -m eval.run_eval --suite injection
    python -m eval.run_eval --limit 5
"""

import argparse
import json
import sqlite3
import time
from pathlib import Path

from agents.critic import criticize
from agents.utils import matches_any, matches_exact, significant_numbers, to_float
from data_agent import DB_PATH, format_sql_evidence
from graph.research_graph import run_research

EVAL_DIR = Path(__file__).resolve().parent
QUESTIONS_PATH = EVAL_DIR / "questions.json"
RESULTS_PATH = EVAL_DIR / "results.json"

TOLERANCE = 0.01  # 1% relative — allows for sensible rounding in prose


def _load_spec() -> dict:
    return json.loads(QUESTIONS_PATH.read_text())


def _scalar(sql: str):
    with sqlite3.connect(DB_PATH) as db:
        row = db.execute(sql).fetchone()
    return row[0] if row else None


def _row(sql: str) -> dict:
    with sqlite3.connect(DB_PATH) as db:
        cur = db.execute(sql)
        columns = [d[0] for d in cur.description]
        values = cur.fetchone()
    return dict(zip(columns, values)) if values else {}


# --- suite 1: factual accuracy -----------------------------------------------

def run_factual(spec: dict, limit: int | None = None) -> dict:
    cases = spec["factual"][:limit] if limit else spec["factual"]
    results = []

    print(f"\n{'=' * 78}\nFACTUAL ACCURACY — {len(cases)} questions\n{'=' * 78}")

    for case in cases:
        expected = _scalar(case["ground_truth_sql"])

        try:
            outcome = run_research(case["question"])
        except Exception as e:
            results.append({**case, "expected": expected, "error": str(e), "correct": False})
            print(f"  {case['id']}  ERROR  {e}")
            continue

        answer = outcome["answer"]
        draft = outcome["draft"]

        correct = matches_any(float(expected), significant_numbers(answer), rel_tol=TOLERANCE)
        correct_draft = matches_any(float(expected), significant_numbers(draft), rel_tol=TOLERANCE)

        results.append(
            {
                "id": case["id"],
                "question": case["question"],
                "expected": expected,
                "correct": bool(correct),
                "correct_before_critic": bool(correct_draft),
                "confidence": outcome["confidence"],
                "verified_count": outcome["verification"]["verified_count"],
                "unverified_count": outcome["verification"]["unverified_count"],
                "sql": (outcome.get("sql") or {}).get("query"),
                "elapsed_s": outcome["elapsed_s"],
            }
        )

        mark = "PASS" if correct else "FAIL"
        print(f"  {case['id']}  {mark}  conf={outcome['confidence']:<6} "
              f"expected={expected:<22.6g} ({outcome['elapsed_s']}s)")

    total = len(results)
    correct = sum(1 for r in results if r.get("correct"))

    # Calibration: when the pipeline is wrong, does the critic warn?
    wrong = [r for r in results if not r.get("correct")]
    wrong_flagged = sum(1 for r in wrong if r.get("confidence") in {"low", "medium"})

    summary = {
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total, 4) if total else 0.0,
        "wrong_answers": len(wrong),
        "wrong_answers_flagged_not_high_confidence": wrong_flagged,
        "calibration": round(wrong_flagged / len(wrong), 4) if wrong else None,
    }

    print(f"\n  Accuracy: {correct}/{total} = {summary['accuracy']:.1%}")
    if wrong:
        print(f"  Of {len(wrong)} wrong answers, {wrong_flagged} were NOT marked high confidence "
              f"({summary['calibration']:.0%} calibration)")

    return {"summary": summary, "results": results}


# --- suite 2: critic catch rate ----------------------------------------------

def run_injection(spec: dict, limit: int | None = None) -> dict:
    cases = spec["injection"][:limit] if limit else spec["injection"]
    results = []

    print(f"\n{'=' * 78}\nCRITIC CATCH RATE — {len(cases)} injected-error cases\n{'=' * 78}")

    total_injected = 0
    total_caught = 0
    total_legit = 0
    total_false_positives = 0

    for case in cases:
        evidence_row = _row(case["evidence_sql"])

        sql_result = {
            "ok": True,
            "sql": case["evidence_sql"],
            "rows": [evidence_row],
            "row_count": 1,
            "truncated": False,
        }
        evidence_text = format_sql_evidence(sql_result)
        draft = case["draft"].format(**evidence_row)

        injected = [float(v) for v in case["injected"]]

        legitimate = case["legitimate"]
        if legitimate == "__all_from_evidence__":
            legitimate = [to_float(v) for v in evidence_row.values()]
            legitimate = [v for v in legitimate if v is not None]
        legitimate = [float(v) for v in legitimate]

        critique = criticize(
            draft=draft,
            sql_evidence=evidence_text,
            sql_result=sql_result,
        )

        flagged_values = [
            to_float(c.get("value"))
            for c in critique["claims"]
            if c["verdict"] not in {"supported", "contextual"}
        ]
        flagged_values = [v for v in flagged_values if v is not None]

        # Exact matching: "was this specific figure flagged?" is an identity
        # question, so unit rescaling would mis-attribute credit here.
        caught = [v for v in injected if matches_exact(v, flagged_values)]
        missed = [v for v in injected if v not in caught]
        false_positives = [v for v in legitimate if matches_exact(v, flagged_values)]

        total_injected += len(injected)
        total_caught += len(caught)
        total_legit += len(legitimate)
        total_false_positives += len(false_positives)

        results.append(
            {
                "id": case["id"],
                "description": case["description"],
                "injected": injected,
                "caught": caught,
                "missed": missed,
                "legitimate": legitimate,
                "false_positives": false_positives,
                "confidence": critique["confidence"],
            }
        )

        detail = f"caught {len(caught)}/{len(injected)} injected"
        if legitimate:
            detail += f", {len(false_positives)}/{len(legitimate)} false positives"
        status = "OK" if not missed and not false_positives else "!!"
        print(f"  {case['id']}  {status}  {detail:<48} conf={critique['confidence']}")
        if missed:
            print(f"        MISSED: {missed}")
        if false_positives:
            print(f"        FALSE POSITIVE: {false_positives}")

    catch_rate = total_caught / total_injected if total_injected else 0.0
    fp_rate = total_false_positives / total_legit if total_legit else 0.0

    summary = {
        "injected_total": total_injected,
        "caught": total_caught,
        "catch_rate": round(catch_rate, 4),
        "legitimate_total": total_legit,
        "false_positives": total_false_positives,
        "false_positive_rate": round(fp_rate, 4),
        # With no critic the draft is shown as-is, so nothing is ever flagged.
        "unflagged_rate_critic_off": 1.0,
        "unflagged_rate_critic_on": round(1 - catch_rate, 4),
    }

    print(f"\n  Catch rate:          {total_caught}/{total_injected} = {catch_rate:.1%}")
    print(f"  False positive rate: {total_false_positives}/{total_legit} = {fp_rate:.1%}")
    print(f"\n  Fabricated figures reaching the user unflagged:")
    print(f"    critic OFF: {total_injected}/{total_injected} = 100.0%")
    print(f"    critic ON:  {total_injected - total_caught}/{total_injected} = {1 - catch_rate:.1%}")

    return {"summary": summary, "results": results}


# --- entrypoint ---------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=["all", "factual", "injection"], default="all")
    parser.add_argument("--limit", type=int, default=None, help="cap cases per suite")
    parser.add_argument("--output", default=str(RESULTS_PATH))
    args = parser.parse_args()

    spec = _load_spec()
    started = time.time()
    output_path = Path(args.output)

    # Merge into any existing results rather than replacing the file. Running
    # one suite (e.g. the cheap injection pass when the token budget is tight)
    # must not discard the other suite's last measurement.
    report = {}
    if output_path.exists():
        try:
            report = json.loads(output_path.read_text())
        except (json.JSONDecodeError, OSError):
            report = {}

    now = time.strftime("%Y-%m-%d %H:%M:%S")

    if args.suite in {"all", "factual"}:
        report["factual"] = run_factual(spec, args.limit)
        report["factual"]["measured_at"] = now
        report["factual"]["limit"] = args.limit

    if args.suite in {"all", "injection"}:
        report["injection"] = run_injection(spec, args.limit)
        report["injection"]["measured_at"] = now
        report["injection"]["limit"] = args.limit

    report["generated_at"] = now
    report["last_suite_run"] = args.suite
    report["elapsed_s"] = round(time.time() - started, 1)

    print(f"\n{'=' * 78}")
    for suite, label, fmt in (
        ("factual", "Answer accuracy", lambda s: f"{s['accuracy']:.1%}"),
        ("injection", "Critic catch rate", lambda s:
            f"{s['catch_rate']:.1%}  (false positives {s['false_positive_rate']:.1%})"),
    ):
        if suite not in report:
            continue
        stale = "" if suite in {args.suite, "all"} or args.suite == "all" else \
            f"   [carried over from {report[suite].get('measured_at', 'an earlier run')}]"
        print(f"  {label}: {fmt(report[suite]['summary'])}{stale}")
    print(f"  Total time: {report['elapsed_s']}s\n{'=' * 78}")

    output_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
