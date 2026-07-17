"""
Parallel FarmerChat eval runner.

Logs in once and shares that token across worker threads (avoids concurrent
logins to the same account racing/invalidating each other's tokens), with
coordinated re-authentication if the token expires mid-run.

Usage:
  python run_parallel.py                                  # all categories, 3 repeats, 6 workers
  python run_parallel.py --categories NUTR PEST            # only these categories
  python run_parallel.py --repeats 1                       # single pass, no consistency repeats
  python run_parallel.py --workers 4                       # lower concurrency
  python run_parallel.py --resume-from failed.json          # only re-run [category, test_code, repeat] tuples in this JSON file
  python run_parallel.py --out results/my_run.jsonl
"""

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from parse_cases import load_all_cases
from api_client import FarmerChatClient
import langfuse_client

DEVICE_ID = "device_0001234545"


class SharedToken:
    """One shared login, refreshed under a lock so concurrent 401s don't cause a thundering herd of re-logins."""

    def __init__(self, device_id):
        self.device_id = device_id
        self.token = None
        self.user_id = None
        self.lock = threading.Lock()
        self._login()

    def _login(self):
        c = FarmerChatClient(self.device_id)
        c.initialize_user()
        c.set_language()
        self.token = c.access_token
        self.user_id = c.user_id
        print(f"(re)logged in, user_id={self.user_id}", flush=True)

    def get(self):
        with self.lock:
            return self.token, self.user_id

    def refresh_if_stale(self, stale_token):
        """Only re-logs in if no other thread already refreshed past stale_token."""
        with self.lock:
            if self.token == stale_token:
                self._login()
            return self.token, self.user_id


def run_repeat(category, tc, repeat_idx, shared, with_langfuse=True):
    token, uid = shared.get()
    client = FarmerChatClient(DEVICE_ID)
    client.attach_session(token, uid)

    def coordinated_reauth(c):
        new_token, new_uid = shared.refresh_if_stale(c.access_token)
        c.attach_session(new_token, new_uid)

    client.on_reauth = coordinated_reauth

    base = {
        "category": category, "test_code": tc.test_code, "scenario": tc.scenario,
        "difficulty": tc.difficulty, "expected_action": tc.expected_action,
        "clarification_slots": tc.clarification_slots, "resolution_goal": tc.resolution_goal,
        "notes": tc.notes, "repeat": repeat_idx,
    }

    try:
        client.new_conversation()
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
                "needed_retry": needed_retry,
                "langfuse_found": lt is not None,
                "span_name": lt["observation"]["name"] if lt else None,
                "used_tools": md.get("used_tools"),
                "resolution_type": md.get("resolution_type") or md.get("decision"),
                "detected_commodities": md.get("detected_commodities"),
                "alignment_decision": md.get("alignment_decision"),
            })
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
    parser.add_argument("--categories", nargs="+", help="Only these categories (default: all)")
    parser.add_argument("--repeats", type=int, default=3, help="Repeats per test case (default 3)")
    parser.add_argument("--workers", type=int, default=6, help="Concurrent worker threads (default 6)")
    parser.add_argument("--no-langfuse", action="store_true", help="Skip Langfuse trace lookup")
    parser.add_argument("--resume-from", help="JSON file of [category, test_code, repeat] tuples to re-run only")
    parser.add_argument("--out", default="/tmp/parallel_results.jsonl", help="Output JSONL path")
    args = parser.parse_args()

    all_cases = load_all_cases()
    work_items = build_work_items(all_cases, args.categories, args.repeats, args.resume_from)
    total = len(work_items)
    print(f"Total work items: {total}  (max_workers={args.workers})", flush=True)

    shared = SharedToken(DEVICE_ID)

    lock = threading.Lock()
    done_count = 0
    t0 = time.time()

    with open(args.out, "w") as out, ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(run_repeat, cat, tc, i, shared, not args.no_langfuse): (cat, tc, i)
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
