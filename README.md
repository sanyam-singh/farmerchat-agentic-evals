# FarmerChat Agentic Evals

Evaluation harness for FarmerChat's agentic chat endpoint (`get_answer_for_text_query_agentic`) — accuracy/consistency testing across English and Hindi test suites, Langfuse trace correlation, LLM-judge review of expected-behavior compliance, and a benchmark comparing a fine-tuned fact-generation model against the live agent.

## Setup

```bash
pip install -r requirements.txt   # requests, openai
cp secrets_local.py.example secrets_local.py   # fill in real keys, or set env vars instead
```

Required credentials (env var or `secrets_local.py`):

| Env var | Purpose |
|---|---|
| `FARMERCHAT_API_KEY` | FarmerChat staging API |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` | Trace lookup (project `fc-agent-eagle`) |
| `OPENAI_API_KEY` | LLM judge calls (GPT-5.5) and fact extraction |

Test case CSVs (`English_Cases - <CATEGORY>.csv`, `Hindi_Cases - <CATEGORY>.csv`) are expected at the path set by `CSV_DIR` in `config.py`.

**Standing constraint: never exceed 5 concurrent workers against the FarmerChat agentic endpoint.** The backend has a real concurrency ceiling at that level — going higher causes empty-body failures. This cap doesn't apply to pure-OpenAI work (judge calls, fact evaluation), which can run at much higher concurrency (15–40 workers).

## Scripts

### `run_parallel.py` — main eval runner
Runs the full test suite (all categories × repeats) against the agentic endpoint, with a fresh account per test-case-repeat (no shared-login pollution) and Langfuse trace correlation per turn.

```bash
python run_parallel.py --lang en --workers 5 --repeats 3 --out results/full_en.jsonl
python run_parallel.py --lang hi --workers 5 --categories NUTR PEST
python run_parallel.py --resume-from failed.json   # re-run only [category, test_code, repeat] tuples
```

### `judge_expected_behavior.py` — LLM-judge compliance review
Reads actual conversation content (not just Langfuse's `resolution_type` label) against each test case's `Expected Action` / `Clarification Slots` / `Tool Call` annotations, using GPT-5.5, and checks consistency across the 3 repeats.

```bash
python judge_expected_behavior.py --lang en --workers 8
```

### `benchmark_finetuned_vs_agent.py` — FT model vs. agent benchmark
Compares a fine-tuned fact-generation model against the live agent, both scored against gold facts from `finalSFT/test.jsonl` using GPT-5.5 (semantic match → contradiction check → relevance check).

```bash
python benchmark_finetuned_vs_agent.py --sample 100 --workers 8      # both sides, sampled
python benchmark_finetuned_vs_agent.py --ft-only --workers 20        # FT side only (no FarmerChat calls, higher concurrency OK)
python benchmark_finetuned_vs_agent.py --agent-only --workers 5      # agent side only (5-worker cap applies)
python benchmark_finetuned_vs_agent.py --sample-ids-file ids.json    # exact sample_ids, for resuming
```

### `requery_en.py` / `requery_hi.py` — Langfuse trace recovery
Re-checks Langfuse directly for turns whose original trace lookup missed (async ingestion lag), without re-running anything against the FarmerChat API. Requires `user_id` to have been persisted per turn by `run_parallel.py`.

### `rejudge_hi.py`
Re-runs the LLM judge for Hindi test cases whose repeats changed after trace recovery.

### `parse_cases.py`
Parses the eval CSVs into structured test cases (`Turn`/`TestCase` dataclasses). Imported by the other scripts, not usually run directly.

### `api_client.py` / `langfuse_client.py` / `config.py`
Supporting modules: `FarmerChatClient` (auth, conversation lifecycle, SSE parsing, retry-on-empty), `find_turn_by_message_id` (Langfuse trace lookup — see note below), and shared config/secrets loading.

## Notes on Langfuse trace correlation

The API's `message_id` is **not** the Langfuse trace ID — it's a `query_id` field nested inside `input` on a SPAN-type observation. Lookup works by listing recent SPAN observations for the `userId` and matching `input.query_id` (no name filter, since the span name varies by code path — e.g. `org.farmerchat:chat:mobile:en` vs `farmer_chat_mobile_app:alignment_intercept`).

Trace ingestion is async and sometimes exceeds the polling window under concurrent load — a turn showing `langfuse_found: false` right after a run often does have a trace, it just hadn't landed yet. `requery_en.py`/`requery_hi.py` re-check later without re-running the eval.

## Data note

Regenerated intermediate outputs (analysis JSON, judge results, benchmark JSONL) live under `backups/` locally and in `/tmp` during runs — both are gitignored. `/tmp` in particular gets cleared periodically by the OS; treat anything only in `/tmp` as at-risk and back up before it matters.
