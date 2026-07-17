"""
Looks up the Langfuse record for a FarmerChat turn by message_id.

message_id is NOT the Langfuse trace ID (traces use OTel-style hex IDs).
It's the query_id field nested inside the `input` of the per-turn SPAN
observation. The SPAN name varies by which path the backend took for that
turn — e.g. "farmer_chat_mobile_app:chat:mobile:en" for a full agent
answer, "farmer_chat_mobile_app:alignment_intercept" when the alignment
layer short-circuits into a clarifying question. So we don't filter by
name — just list recent SPANs for the user and match input.query_id.
"""

import time
from typing import Optional
import requests
from requests.auth import HTTPBasicAuth
from config import LANGFUSE_HOST, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY

_auth = HTTPBasicAuth(LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY)


def _list_turn_observations(user_id: str, limit: int = 30) -> list:
    resp = requests.get(
        f"{LANGFUSE_HOST}/api/public/observations",
        auth=_auth,
        params={"userId": user_id, "type": "SPAN", "limit": limit},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json().get("data", [])


def get_trace(trace_id: str) -> dict:
    resp = requests.get(f"{LANGFUSE_HOST}/api/public/traces/{trace_id}", auth=_auth, timeout=30)
    resp.raise_for_status()
    return resp.json()


def find_turn_by_message_id(
    user_id: str, message_id: str, max_attempts: int = 12, delay: float = 3.0, limit: int = 30
) -> Optional[dict]:
    """
    Poll Langfuse until the SPAN for this turn shows up (ingestion is async),
    then attach its parent trace for turn-level scores. Returns None if not
    found within max_attempts.
    """
    for attempt in range(1, max_attempts + 1):
        for obs in _list_turn_observations(user_id, limit=limit):
            inp = obs.get("input")
            if isinstance(inp, dict) and inp.get("query_id") == message_id:
                return {"observation": obs, "trace": get_trace(obs["traceId"])}
        if attempt < max_attempts:
            time.sleep(delay)
    return None


def score_trace(trace_id: str, name: str, value, comment: str = "") -> dict:
    """Push an eval score back to Langfuse, attached to the trace."""
    resp = requests.post(
        f"{LANGFUSE_HOST}/api/public/scores",
        auth=_auth,
        json={"traceId": trace_id, "name": name, "value": value, "comment": comment},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()
