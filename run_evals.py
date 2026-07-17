"""
FarmerChat Eval Runner
======================
Usage:
  python run_evals.py                        # run all categories
  python run_evals.py --categories PEST NUTR # run specific categories
  python run_evals.py --dry-run              # parse only, no API calls
  python run_evals.py --delay 1.5            # seconds between turns (default 1.0)
"""

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime

from api_client import FarmerChatClient
from parse_cases import load_all_cases, TestCase
from config import RESULTS_DIR
import langfuse_client


def make_device_id() -> str:
    return f"eval_{int(time.time() * 1000)}"


def run_test_case(tc: TestCase, delay: float = 1.0, with_langfuse: bool = False) -> dict:
    """Run a single test case. Returns a result dict."""
    device_id = make_device_id()
    client = FarmerChatClient(device_id)

    result = {
        "test_code": tc.test_code,
        "category": tc.category,
        "scenario": tc.scenario,
        "difficulty": tc.difficulty,
        "expected_action": tc.expected_action,
        "clarification_slots": tc.clarification_slots,
        "resolution_goal": tc.resolution_goal,
        "persona": tc.persona,
        "device_id": device_id,
        "user_id": None,
        "conversation_id": None,
        "status": "pending",
        "error": None,
        "turns": [],
    }

    try:
        setup = client.setup()
        result["user_id"] = client.user_id
        result["conversation_id"] = client.conversation_id

        # Walk through turns; send Farmer turns, record actual vs expected
        pending_expected: dict = {}

        for turn in tc.turns:
            if turn.role == "farmer":
                time.sleep(delay)
                api_resp = client.send_message(turn.text)
                clarify_message = (
                    api_resp.get("agentic_trace", {}).get("surface", {}).get("payload", {}).get("message")
                )
                actual_text = (
                    api_resp.get("response")
                    or clarify_message
                    or api_resp.get("answer")
                    or api_resp.get("message")
                    or json.dumps(api_resp)
                )
                message_id = api_resp.get("message_id") or api_resp.get("section_message_id")
                turn_result = {
                    "role": "farmer",
                    "query": turn.text,
                    "actual_response": actual_text,
                    "expected_response": pending_expected.get("text"),
                    "expected_chips": pending_expected.get("chips"),
                    "correct_chip": pending_expected.get("correct_chip"),
                    "raw_api_response": api_resp,
                    "message_id": message_id,
                    "needed_retry": api_resp.get("_needed_retry", False),
                    "langfuse_trace": None,
                }

                if with_langfuse and message_id:
                    try:
                        turn_result["langfuse_trace"] = langfuse_client.find_turn_by_message_id(
                            client.user_id, message_id
                        )
                    except Exception as e:
                        turn_result["langfuse_trace_error"] = str(e)

                result["turns"].append(turn_result)
                pending_expected = {}

            elif turn.role == "agent":
                # Store expected so next farmer turn can reference it
                pending_expected = {
                    "text": turn.text,
                    "chips": turn.expected_chips,
                    "correct_chip": turn.correct_chip,
                }

            elif turn.role == "tool_call":
                result["turns"].append({
                    "role": "tool_call",
                    "expected_tool": turn.text,
                })

        result["status"] = "pass"

    except Exception as e:
        result["status"] = "error"
        result["error"] = traceback.format_exc()
        print(f"    ERROR: {e}")

    return result


def run_category(category: str, cases: list, delay: float, dry_run: bool, with_langfuse: bool = False) -> list:
    results = []
    print(f"\n{'='*60}")
    print(f"Category: {category}  ({len(cases)} cases)")
    print(f"{'='*60}")

    for i, tc in enumerate(cases, 1):
        print(f"  [{i}/{len(cases)}] {tc.test_code} | {tc.scenario} | {tc.difficulty}", end="")

        if dry_run:
            farmer_turns = [t for t in tc.turns if t.role == "farmer"]
            print(f"  → DRY RUN ({len(farmer_turns)} farmer turns)")
            results.append({"test_code": tc.test_code, "status": "dry_run"})
            continue

        print(" ...", end="", flush=True)
        result = run_test_case(tc, delay=delay, with_langfuse=with_langfuse)
        status_icon = "✓" if result["status"] == "pass" else "✗"
        print(f" {status_icon} {result['status']}")
        results.append(result)
        time.sleep(0.5)

    return results


def save_results(all_results: dict, run_id: str):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # One JSON per category
    for category, results in all_results.items():
        path = os.path.join(RESULTS_DIR, f"{run_id}_{category}.json")
        with open(path, "w") as f:
            json.dump(results, f, indent=2)

    # Summary CSV
    summary_path = os.path.join(RESULTS_DIR, f"{run_id}_summary.csv")
    with open(summary_path, "w") as f:
        f.write("test_code,category,scenario,difficulty,expected_action,status,user_id,conversation_id,error\n")
        for category, results in all_results.items():
            for r in results:
                error_short = (r.get("error") or "").split("\n")[0].replace(",", ";")
                f.write(",".join([
                    r.get("test_code", ""),
                    r.get("category", category),
                    r.get("scenario", ""),
                    r.get("difficulty", ""),
                    r.get("expected_action", ""),
                    r.get("status", ""),
                    r.get("user_id") or "",
                    r.get("conversation_id") or "",
                    error_short,
                ]) + "\n")

    print(f"\nResults saved to: {RESULTS_DIR}/")
    print(f"  Summary: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="FarmerChat Eval Runner")
    parser.add_argument("--categories", nargs="+", help="Run only these categories (e.g. PEST NUTR)")
    parser.add_argument("--dry-run", action="store_true", help="Parse CSVs only, no API calls")
    parser.add_argument("--delay", type=float, default=1.0, help="Seconds between API turns (default 1.0)")
    parser.add_argument("--with-langfuse", action="store_true",
                         help="Fetch the Langfuse trace (tool calls, retrieval, generations) for each turn via message_id")
    args = parser.parse_args()

    if args.with_langfuse:
        from config import LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY
        if not LANGFUSE_PUBLIC_KEY or not LANGFUSE_SECRET_KEY:
            sys.exit("ERROR: --with-langfuse requires LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY env vars to be set.")

    print("Loading eval cases...")
    all_cases = load_all_cases()

    categories = args.categories if args.categories else list(all_cases.keys())
    total = sum(len(all_cases[c]) for c in categories if c in all_cases)
    print(f"\nRunning {total} test cases across {len(categories)} categories")
    if args.dry_run:
        print("Mode: DRY RUN (no API calls)")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results = {}

    for category in categories:
        if category not in all_cases:
            print(f"WARNING: Unknown category '{category}', skipping")
            continue
        results = run_category(category, all_cases[category], args.delay, args.dry_run, with_langfuse=args.with_langfuse)
        all_results[category] = results

    if not args.dry_run:
        save_results(all_results, run_id)

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    total_pass = total_error = total_dry = 0
    for category, results in all_results.items():
        passed = sum(1 for r in results if r.get("status") == "pass")
        errors = sum(1 for r in results if r.get("status") == "error")
        dry    = sum(1 for r in results if r.get("status") == "dry_run")
        total_pass += passed; total_error += errors; total_dry += dry
        print(f"  {category:6s}: {len(results)} cases | pass={passed} error={errors} dry={dry}")
    print(f"  {'TOTAL':6s}: {total_pass + total_error + total_dry} | pass={total_pass} error={total_error} dry={total_dry}")


if __name__ == "__main__":
    main()
