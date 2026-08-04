import json
import sys
sys.path.insert(0, ".")
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from judge_expected_behavior import judge_test_case
from config import _secret

openai_key = _secret("OPENAI_API_KEY", "OPENAI_API_KEY")

analysis = {tc["test_code"]: tc for tc in json.load(open("/tmp/hindi_analysis.json"))}
annotations = json.load(open("/tmp/expected_annotations_hi.json"))
affected = json.load(open("/tmp/requery_affected_hi.json"))

to_judge = [analysis[tc] for tc in affected if tc in analysis and any(r["status"] == "pass" for r in analysis[tc]["repeats"])]
print(f"Re-judging {len(to_judge)} affected Hindi test cases", flush=True)

lock = threading.Lock()
results = {}
done = 0
with ThreadPoolExecutor(max_workers=10) as ex:
    futures = {ex.submit(judge_test_case, tc, annotations.get(tc["test_code"], {}), openai_key): tc for tc in to_judge}
    for future in as_completed(futures):
        tc = futures[future]
        try:
            verdict = future.result()
        except Exception as e:
            verdict = {"error": str(e)}
        with lock:
            results[tc["test_code"]] = {"test_code": tc["test_code"], "category": tc["category"], **verdict}
            done += 1
            print(f"PROGRESS [{done}/{len(to_judge)}]", flush=True)

existing = [json.loads(l) for l in open("/tmp/judged_hi.jsonl")]
existing_map = {j["test_code"]: j for j in existing}
existing_map.update(results)

with open("/tmp/judged_hi.jsonl", "w") as f:
    for j in existing_map.values():
        f.write(json.dumps(j) + "\n")

print(f"ALL DONE. Wrote {len(existing_map)} total judged entries ({len(results)} re-judged)", flush=True)
