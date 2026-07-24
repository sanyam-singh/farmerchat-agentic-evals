"""
LLM-judge review: does the agent's actual response semantically satisfy the
test case's expected_action / clarification_slots / expected tool call,
and is that judgment consistent across the 3 repeats?

Naive string-matching against Langfuse's resolution_type undercounts real
compliance (e.g. the agent correctly refutes bad advice, but the trace
labels it "direct_action" not "confirm") — this reads actual response
content instead, using GPT-5.5 as judge.

Usage:
  python judge_expected_behavior.py --lang en --workers 8
  python judge_expected_behavior.py --lang hi --workers 8
"""

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import openai
import requests

from config import _secret

JUDGE_MODEL = "gpt-5.5"
NETWORK_ERRORS = (requests.exceptions.ConnectionError, requests.exceptions.Timeout)


def build_prompt(tc, ann):
    passed = [r for r in tc["repeats"] if r["status"] == "pass"]
    repeats_block = []
    for r in passed:
        turns_text = "\n".join(f'  Farmer: {t["query"]}\n  Agent: {t["response"]}' for t in r["turns"])
        repeats_block.append(f"--- Repeat {r['repeat']} ---\n{turns_text}")

    return f"""You are reviewing an agricultural chatbot's responses against what a test case expected, by reading actual content — not by matching internal classifier labels.

TEST CASE: {tc["test_code"]} ({tc["category"]}, scenario: {tc["scenario"]})
EXPECTED ACTION: {ann.get("expected_action")}
CLARIFICATION NEEDED: {"Yes — slots: " + ann["clarification_slots"] if ann.get("needs_clarification") else "No"}
EXPECTED TOOL / DATA USE: {"; ".join(ann.get("expected_tool_calls", [])) or "none specified"}
RESOLUTION GOAL (what this test checks): {tc.get("resolution_goal") or "not specified"}

CONVERSATION REPEATS (same query/queries, run 3 independent times):
{chr(10).join(repeats_block)}

For EACH repeat, judge from the actual text:
1. Does the response semantically satisfy the EXPECTED ACTION? (e.g. if expected action is "confirm", does the agent clearly validate or refute what the farmer described, even if phrased differently than a literal "confirm"? If "ask_clarification", does it actually ask before giving specifics?)
2. If clarification was needed, did the agent ask for it (in the first relevant turn) rather than assuming details?
3. Does the response reflect use of the expected data/tool (e.g. weather-specific info if a weather tool was expected), based on content alone?

Then assess whether the verdict is CONSISTENT across all repeats — do they all handle the expected action the same way, or does behavior vary run-to-run?

Respond with ONLY this JSON:
{{
  "per_repeat": [
    {{"repeat": 1, "satisfies_expected_action": true/false, "clarification_handled_correctly": true/false/null, "tool_data_reflected": true/false/null, "brief_reason": "one sentence"}}
  ],
  "consistent_across_repeats": true/false,
  "consistency_note": "one sentence on what varies, or 'all repeats handled it the same way'",
  "overall_verdict": "meets_expectation" | "partially_meets" | "does_not_meet"
}}
"""


def judge_test_case(tc, ann, openai_key, max_attempts=3):
    client = openai.OpenAI(api_key=openai_key)
    prompt = build_prompt(tc, ann)
    last_exc = None
    for attempt in range(max_attempts):
        try:
            resp = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": "You are a meticulous QA reviewer for an agricultural chatbot. Respond ONLY with valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
            )
            return json.loads(resp.choices[0].message.content.strip())
        except openai.RateLimitError as e:
            last_exc = e
            if attempt < max_attempts - 1:
                time.sleep(5.0 * (attempt + 1))
        except NETWORK_ERRORS as e:
            last_exc = e
            if attempt < max_attempts - 1:
                time.sleep(3.0)
        except Exception as e:
            return {"error": str(e)}
    return {"error": f"exhausted retries: {last_exc}"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", required=True, choices=["en", "hi"])
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    openai_key = _secret("OPENAI_API_KEY", "OPENAI_API_KEY")

    analysis_path = "/tmp/full_analysis_complete.json" if args.lang == "en" else "/tmp/hindi_analysis.json"
    annotations_path = f"/tmp/expected_annotations_{args.lang}.json"
    out_path = f"/tmp/judged_{args.lang}.jsonl"

    with open(analysis_path) as f:
        test_cases = json.load(f)
    with open(annotations_path) as f:
        annotations = json.load(f)

    # Only judge cases that have at least one passing repeat
    to_judge = [tc for tc in test_cases if any(r["status"] == "pass" for r in tc["repeats"])]
    print(f"Judging {len(to_judge)} test cases ({args.lang})", flush=True)

    lock = threading.Lock()
    done = 0
    t0 = time.time()

    with open(out_path, "w") as out, ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(judge_test_case, tc, annotations.get(tc["test_code"], {}), openai_key): tc
            for tc in to_judge
        }
        for future in as_completed(futures):
            tc = futures[future]
            try:
                verdict = future.result()
            except Exception as e:
                verdict = {"error": str(e)}
            record = {"test_code": tc["test_code"], "category": tc["category"], **verdict}
            with lock:
                out.write(json.dumps(record) + "\n")
                out.flush()
                done += 1
                elapsed = time.time() - t0
                eta = (elapsed / done) * (len(to_judge) - done)
                print(f"[{done}/{len(to_judge)}] elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m", flush=True)

    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
