"""
FarmerChat API client for eval runs.
Creates a fresh user per test case and drives multi-turn conversations.
"""

import json
import time
import requests
from config import BASE_URL, API_KEY, LANGUAGE_ID, BUILD_VERSION, DEVICE_INFO_TEMPLATE, DEFAULT_LAT, DEFAULT_LON


class EmptyResponseError(RuntimeError):
    pass


def _parse_sse(raw: str) -> dict:
    """
    The agentic endpoint responds as SSE regardless of streaming_required.
    Each block is "event: <name>\\ndata: <json>". Returns {event_name: parsed_json}.
    """
    events = {}
    for block in raw.strip().split("\n\n"):
        event_name, data_line = None, None
        for line in block.splitlines():
            if line.startswith("event:"):
                event_name = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data_line = line[len("data:"):].strip()
        if event_name and data_line:
            events[event_name] = json.loads(data_line)
    return events


class FarmerChatClient:
    def __init__(self, device_id: str):
        self.device_id = device_id
        self.access_token: str = ""
        self.user_id: str = ""
        self.conversation_id: str = ""
        self.session = requests.Session()

    # ------------------------------------------------------------------ #

    def attach_session(self, access_token: str, user_id: str) -> None:
        """
        Reuse an already-issued token/user instead of calling initialize_user again.
        For parallel runs: logging in once and sharing the token avoids concurrent
        logins to the same account racing/invalidating each other's tokens.
        """
        self.access_token = access_token
        self.user_id = user_id

    def initialize_user(self, lat: float = DEFAULT_LAT, lon: float = DEFAULT_LON) -> dict:
        device_info = DEVICE_INFO_TEMPLATE.format(device_id=self.device_id)
        resp = self.session.post(
            f"{BASE_URL}/api/user/initialize_user/",
            headers={
                "Content-Type": "application/json",
                "API-Key": API_KEY,
                "Device-Info": device_info,
                "Build-Version": BUILD_VERSION,
            },
            json={
                "device_id": self.device_id,
                "latitude": lat,
                "longitude": lon,
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        self.access_token = data.get("access_token") or data.get("token", {}).get("access", "")
        self.user_id = data.get("user_id") or data.get("id", "")
        return data

    def set_language(self) -> dict:
        resp = self.session.post(
            f"{BASE_URL}/api/user/set_preferred_language/",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.access_token}",
            },
            json={"user_id": self.user_id, "language_id": LANGUAGE_ID},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def new_conversation(self) -> dict:
        resp = self.session.post(
            f"{BASE_URL}/api/chat/new_conversation/",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.access_token}",
            },
            json={"user_id": self.user_id},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        self.conversation_id = data.get("conversation_id") or data.get("id", "")
        return data

    def send_message(self, query: str, use_latest_prompt: bool = True, retry_on_empty: int = 1) -> dict:
        try:
            result = self._send_message_once(query, use_latest_prompt)
            result["_needed_retry"] = False
            return result
        except EmptyResponseError:
            if retry_on_empty <= 0:
                raise
            time.sleep(2.0)
            result = self.send_message(query, use_latest_prompt, retry_on_empty=retry_on_empty - 1)
            result["_needed_retry"] = True
            return result

    def _send_message_once(self, query: str, use_latest_prompt: bool) -> dict:
        resp = self.session.post(
            f"{BASE_URL}/api/chat/get_answer_for_text_query_agentic/",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.access_token}",
                "Build-Version": BUILD_VERSION,
            },
            json={
                "conversation_id": self.conversation_id,
                "query": query,
                "input_type": "text",
                "use_latest_prompt": use_latest_prompt,
                "triggered_input_type": "text",
                "streaming_required": False,
                "retry": False,
            },
            timeout=120,
        )
        resp.raise_for_status()

        if not resp.content.strip():
            raise EmptyResponseError(
                "Agentic endpoint returned an empty body (200 OK). Seen intermittently even on "
                "an allow-listed account — likely staging-side flakiness/rate-limiting rather "
                "than a client bug."
            )

        events = _parse_sse(resp.content.decode("utf-8"))
        metadata = events.get("metadata", {})
        done = events.get("done", {})

        # `metadata` carries the scoring-friendly shape (message_id, response, ...);
        # `done` carries the raw agent trace (query_id, tool calls, latency, tokens).
        # done.query_id == metadata.message_id — both are the Langfuse trace ID.
        result = dict(metadata)
        result["agentic_trace"] = done
        return result

    # ------------------------------------------------------------------ #

    def setup(self, lat: float = DEFAULT_LAT, lon: float = DEFAULT_LON) -> dict:
        """Initialize user → set language → new conversation. Returns setup metadata."""
        init_data = self.initialize_user(lat, lon)
        self.set_language()
        conv_data = self.new_conversation()
        return {
            "device_id": self.device_id,
            "user_id": self.user_id,
            "conversation_id": self.conversation_id,
            "init_response": init_data,
            "conv_response": conv_data,
        }
