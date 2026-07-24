"""
Parallel FarmerChat eval runner.

Creates a fresh, unique account (device_id) per test-case-repeat instead of
sharing one login across workers. A field-name bug (latitude/longitude vs.
the API's actual lat/long) previously made fresh accounts return empty
bodies, which is why earlier versions of this script pooled one shared,
heavily-reused account — that reuse caused its own problem (conversation
history pollution degrading response quality over thousands of calls). With
the bug fixed, fresh-per-call accounts avoid both issues and parallelize
without any shared-state coordination.

Usage:
  python run_parallel.py                                  # all categories, 3 repeats, 6 workers
  python run_parallel.py --categories NUTR PEST            # only these categories
  python run_parallel.py --repeats 1                       # single pass, no consistency repeats
  python run_parallel.py --workers 12                      # concurrency (no shared-account bottleneck now)
  python run_parallel.py --resume-from failed.json          # only re-run [category, test_code, repeat] tuples in this JSON file
  python run_parallel.py --out results/my_run.jsonl
"""

import argparse
import json
import random
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from parse_cases import load_all_cases
from api_client import FarmerChatClient, EmptyResponseError
from config import CSV_FILES_BY_LANG, LANGUAGE_IDS
import langfuse_client

NETWORK_ERRORS = (requests.exceptions.ConnectionError, requests.exceptions.Timeout)
RETRIABLE_ERRORS = NETWORK_ERRORS + (EmptyResponseError,)


def make_device_id():
    return f"device_eval_{uuid.uuid4().hex[:16]}"


def run_repeat(category, tc, repeat_idx, language_id, with_langfuse=True, max_attempts=3):
    """Retries the whole repeat on transient network errors AND empty-body responses — each
    retry calls _run_repeat_once again, which creates an entirely new device_id -> user_id ->
    conversation_id, not just a same-identity retry (that's handled once, cheaply, inside
    api_client itself; this is the fresh-everything fallback for when that's not enough)."""
    last_result = None
    for attempt in range(max_attempts):
        try:
            last_result = _run_repeat_once(category, tc, repeat_idx, language_id, with_langfuse)
        except RETRIABLE_ERRORS as e:
            last_result = {
                "category": category, "test_code": tc.test_code, "scenario": tc.scenario,
                "difficulty": tc.difficulty, "expected_action": tc.expected_action,
                "clarification_slots": tc.clarification_slots, "resolution_goal": tc.resolution_goal,
                "notes": tc.notes, "repeat": repeat_idx, "status": "error",
                "error": str(e), "turns": [],
            }
        if last_result["status"] == "pass" or attempt == max_attempts - 1:
            return last_result
        time.sleep(3.0)
    return last_result


def _run_repeat_once(category, tc, repeat_idx, language_id, with_langfuse=True):
    time.sleep(random.uniform(0, 2.0))  # jitter to avoid a thundering herd of simultaneous initialize_user calls
    client = FarmerChatClient(make_device_id())
    client.initialize_user()
    client.set_language(language_id)

    base = {
        "category": category, "test_code": tc.test_code, "scenario": tc.scenario,
        "difficulty": tc.difficulty, "expected_action": tc.expected_action,
        "clarification_slots": tc.clarification_slots, "resolution_goal": tc.resolution_goal,
        "notes": tc.notes, "repeat": repeat_idx,
    }

    try:
        client.new_conversation()
    except RETRIABLE_ERRORS:
        raise  # let the outer retry-with-fresh-identity wrapper handle it
    except Exception as e:
        return {**base, "status": "error", "error": f"new_conversation failed: {e}", "turns": []}

    turns_out = []
    try:
        for turn in tc.turns:
            if turn.role != "farmer":
                continue
            time.sleep(0.5)
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
            needed_retry = api_resp.get("_needed_retry", False)

            lt = None
            if with_langfuse and message_id:
                try:
                    lt = langfuse_client.find_turn_by_message_id(client.user_id, message_id)
                except Exception:
                    lt = None
            md = lt["observation"]["metadata"] if lt else {}

            turns_out.append({
                "query": turn.text,
                "response": actual_text,
                "message_id": message_id,
                "user_id": client.user_id,
                "needed_retry": needed_retry,
                "langfuse_found": lt is not None,
                "span_name": lt["observation"]["name"] if lt else None,
                "used_tools": md.get("used_tools"),
                "resolution_type": md.get("resolution_type") or md.get("decision"),
                "detected_commodities": md.get("detected_commodities"),
                "alignment_decision": md.get("alignment_decision"),
            })
    except RETRIABLE_ERRORS:
        raise  # let the outer retry-with-fresh-identity wrapper handle it
    except Exception as e:
        return {**base, "status": "error", "error": str(e), "turns": turns_out}

    return {**base, "status": "pass", "error": None, "turns": turns_out}


def build_work_items(all_cases, categories, repeats, resume_from):
    if resume_from:
        with open(resume_from) as f:
            items = json.load(f)  # list of [category, test_code, repeat]
        tc_by_code = {tc.test_code: tc for cases in all_cases.values() for tc in cases}
        return [(cat, tc_by_code[code], rep) for cat, code, rep in items]

    work = []
    for category, cases in all_cases.items():
        if categories and category not in categories:
            continue
        for tc in cases:
            for i in range(1, repeats + 1):
                work.append((category, tc, i))
    return work


def main():
    parser = argparse.ArgumentParser(description="Parallel FarmerChat eval runner")
    parser.add_argument("--lang", default="en", choices=list(CSV_FILES_BY_LANG.keys()), help="Language of test cases (default en)")
    parser.add_argument("--categories", nargs="+", help="Only these categories (default: all)")
    parser.add_argument("--repeats", type=int, default=3, help="Repeats per test case (default 3)")
    parser.add_argument("--workers", type=int, default=12, help="Concurrent worker threads (default 12; no shared-account bottleneck)")
    parser.add_argument("--no-langfuse", action="store_true", help="Skip Langfuse trace lookup")
    parser.add_argument("--resume-from", help="JSON file of [category, test_code, repeat] tuples to re-run only")
    parser.add_argument("--out", default="/tmp/parallel_results.jsonl", help="Output JSONL path")
    args = parser.parse_args()

    all_cases = load_all_cases(CSV_FILES_BY_LANG[args.lang])
    work_items = build_work_items(all_cases, args.categories, args.repeats, args.resume_from)
    total = len(work_items)
    language_id = LANGUAGE_IDS[args.lang]
    print(f"Total work items: {total}  (max_workers={args.workers}, lang={args.lang})", flush=True)

    lock = threading.Lock()
    done_count = 0
    t0 = time.time()

    with open(args.out, "w") as out, ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(run_repeat, cat, tc, i, language_id, not args.no_langfuse): (cat, tc, i)
            for cat, tc, i in work_items
        }
        for future in as_completed(futures):
            cat, tc, i = futures[future]
            try:
                record = future.result()
            except Exception as e:
                record = {
                    "category": cat, "test_code": tc.test_code, "scenario": tc.scenario,
                    "difficulty": tc.difficulty, "expected_action": tc.expected_action,
                    "clarification_slots": tc.clarification_slots, "resolution_goal": tc.resolution_goal,
                    "notes": tc.notes, "repeat": i, "status": "error", "error": str(e), "turns": [],
                }
            with lock:
                out.write(json.dumps(record) + "\n")
                out.flush()
                done_count += 1
                elapsed = time.time() - t0
                eta = (elapsed / done_count) * (total - done_count)
                print(f"[{done_count}/{total}] {record['test_code']} repeat={record['repeat']} "
                      f"status={record['status']}  elapsed={elapsed/60:.1f}m  eta={eta/60:.1f}m", flush=True)

    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
