import json, time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from requests.auth import HTTPBasicAuth
from config import LANGFUSE_HOST, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY
from langfuse_client import get_trace

auth = HTTPBasicAuth(LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY)

def list_turn_observations_with_retry(user_id, limit=30, max_attempts=6):
    for attempt in range(max_attempts):
        resp = requests.get(f"{LANGFUSE_HOST}/api/public/observations", auth=auth,
                             params={"userId": user_id, "type": "SPAN", "limit": limit}, timeout=30)
        if resp.status_code == 429:
            retry_after = float(resp.headers.get("Retry-After", 3.0 * (attempt + 1)))
            time.sleep(retry_after)
            continue
        resp.raise_for_status()
        return resp.json().get("data", [])
    return []

analysis = json.load(open("/tmp/full_analysis_complete.json"))

missing_by_user = defaultdict(list)
for tci, tc in enumerate(analysis):
    for ri, r in enumerate(tc["repeats"]):
        if r["status"] != "pass":
            continue
        for ti, t in enumerate(r["turns"]):
            if not t.get("langfuse_found") and t.get("user_id"):
                missing_by_user[t["user_id"]].append((tci, ri, ti, t["message_id"]))

print(f"Distinct users to re-check: {len(missing_by_user)}", flush=True)

def check_user(user_id, entries):
    obs_list = list_turn_observations_with_retry(user_id, limit=30)
    qid_map = {}
    for obs in obs_list:
        inp = obs.get("input")
        if isinstance(inp, dict) and inp.get("query_id"):
            qid_map[inp["query_id"]] = obs
    results = []
    for tci, ri, ti, message_id in entries:
        if message_id in qid_map:
            results.append((tci, ri, ti, qid_map[message_id]))
    return results

recovered = []
done = 0
with ThreadPoolExecutor(max_workers=4) as ex:
    futures = {ex.submit(check_user, uid, entries): uid for uid, entries in missing_by_user.items()}
    for fut in as_completed(futures):
        try:
            recovered.extend(fut.result())
        except Exception as e:
            print("error:", e, flush=True)
        done += 1
        if done % 20 == 0 or done == len(missing_by_user):
            print(f"PROGRESS {done}/{len(missing_by_user)} users checked, recovered so far: {len(recovered)}", flush=True)

print(f"DONE recovered {len(recovered)} / 367 missing English turns via direct re-query", flush=True)

for tci, ri, ti, obs in recovered:
    try:
        trace = get_trace(obs["traceId"])
    except Exception:
        trace = None
    md = obs.get("metadata") or {}
    t = analysis[tci]["repeats"][ri]["turns"][ti]
    t["langfuse_found"] = True
    t["span_name"] = obs["name"]
    t["used_tools"] = md.get("used_tools")
    t["resolution_type"] = md.get("resolution_type") or md.get("decision")
    t["detected_commodities"] = md.get("detected_commodities")
    t["alignment_decision"] = md.get("alignment_decision")

json.dump(analysis, open("/tmp/full_analysis_complete.json", "w"))
affected = sorted(set(analysis[tci]["test_code"] for tci, ri, ti, obs in recovered))
json.dump(affected, open("/tmp/requery_affected_en.json", "w"))
print(f"ALL DONE. Merged. {len(affected)} test cases affected.", flush=True)
