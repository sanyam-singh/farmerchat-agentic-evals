"""
Benchmark: fine-tuned fact-generator model vs. the FarmerChat agent, both
scored against the gold fact lists in finalSFT/test.jsonl using GPT-5.5 as
judge. Methodology ported from AWS_model_eval.ipynb's
EnhancedFactEvaluatorProduction (semantic matching -> contradiction check ->
relevance check), with the judge model swapped from gpt-4o to gpt-5.5.

Usage:
  python benchmark_finetuned_vs_agent.py --sample 5     # pilot run
  python benchmark_finetuned_vs_agent.py --sample 100 --workers 8
"""

import argparse
import ast
import json
import re
import threading
import time
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import openai

from config import _secret
from api_client import FarmerChatClient

TEST_PATH = "/Users/sanyaamsingh/Desktop/finalSFT/test.jsonl"
FT_MODEL = "ft:gpt-4o-mini-2024-07-18:dgf-prod-dev-account:e4:Czgtay51"
JUDGE_MODEL = "gpt-5.5"

SYSTEM_PROMPT = """
You are an agricultural fact generator specialized in farming practices. Your task is to extract atomic, verifiable agricultural facts from the provided text and return them as a single, valid JSON object.

GENERATION SCOPE (extract only):
- Extract ONLY facts related to agriculture, farming, crops, livestock, pests/diseases, soil, irrigation, inputs, seasons, or agricultural practices.
- Ignore greetings, conversational elements, follow-up questions, response metadata, disclaimers, and any non-agricultural content.
- Do NOT invent new facts. Only extract facts explicitly present in the input text.

BIHAR AGRICULTURAL CONTEXT (for relevance scoring only; do not add facts):
- Common Bihar districts: Patna, Darbhanga, Madhubani, Champaran, Gopalganj, Gaya, Aurangabad, Muzaffarpur, Begusarai, Bhagalpur
- Primary crops: rice, wheat, maize, sugarcane, potato, onion, arhar (pigeon pea), masur (lentil), gram (chickpea), jute, tobacco
- Key challenges: flooding, drought, pest management, soil salinity, waterlogging
- Agricultural seasons: Kharif (June-October), Rabi (November-April), Zaid (April-June)

FACT ATOMICITY REQUIREMENTS:
- Each fact must contain exactly ONE verifiable claim.
- Break complex statements into multiple atomic facts.
- Preserve original phrasing when possible, including measurements, timing, dose, and units.

STRICT EXCLUSION CRITERIA (do not output these as facts):
- Greetings/pleasantries, acknowledgments, filler words
- Follow-up suggestions or questions
- Meta statements about the conversation or context
- Opinions or advice phrased as opinion ("I think", "you should consider", etc.)
- Disclaimers ("consult an expert", "results may vary", etc.)

OUTPUT SIZE LIMIT (to avoid truncation):
- Return AT MOST 25 facts. If there are more, choose the 25 most specific and actionable.

OUTPUT FORMAT (MUST FOLLOW EXACTLY):
- Output MUST be JSON ONLY: a single JSON object.
- Do NOT output markdown, code fences, explanations, or any extra text before or after the JSON.
- Use ASCII double quotes only (") for all JSON strings.
- No trailing commas. JSON must be parseable by json.loads().

JSON SCHEMA:
{
  "facts": [
    {
      "fact": "One atomic factual statement",
      "category": "crop_variety | pest_disease | soil_management | irrigation | seasonal_practice | input_management",
      "location_dependency": "bihar_specific | universal | region_adaptable",
      "bihar_relevance": "high | medium | low",
      "confidence": 0.0 to 1.0
    }
  ]
}

FIELD RULES:
- category must be exactly one of the 6 allowed values.
- location_dependency must be exactly one of: bihar_specific, universal, region_adaptable.
- bihar_relevance must be exactly one of: high, medium, low.
- confidence must be a number between 0.0 and 1.0 (no quotes).
- If the input contains ZERO extractable agricultural facts, return: {"facts": []}

Return JSON only.
"""

QUESTION_RE = re.compile(
    r"Generate agricultural facts from this question:\s*(.*?)\s*for the location\s*:\s*(.*?)\s*and for this season:\s*(.*)"
)


def load_test_set(path, sample=None):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            msgs = obj["messages"]
            user_msg = next(m["content"] for m in msgs if m["role"] == "user")
            assistant_msg = next(m["content"] for m in msgs if m["role"] == "assistant")
            try:
                gt_facts = json.loads(assistant_msg)
            except Exception:
                gt_facts = assistant_msg
            m = QUESTION_RE.match(user_msg)
            rows.append({
                "meta_question": user_msg,
                "question": m.group(1) if m else user_msg,
                "location": m.group(2) if m else "",
                "season": m.group(3) if m else "",
                "gt_facts": gt_facts,
            })
            if sample and len(rows) >= sample:
                break
    return rows


class FactEvaluator:
    """Ported from AWS_model_eval.ipynb's EnhancedFactEvaluatorProduction, judge model swapped to gpt-5.5."""

    def __init__(self, openai_api_key):
        self._client = openai.OpenAI(api_key=openai_api_key)

    def check_valid_json(self, predicted_facts):
        was_repaired = False
        try:
            if isinstance(predicted_facts, str):
                try:
                    parsed = json.loads(predicted_facts)
                except json.JSONDecodeError:
                    try:
                        parsed = ast.literal_eval(predicted_facts)
                    except Exception:
                        repaired = self._repair_json(predicted_facts)
                        if repaired is None:
                            return {"is_valid": False, "was_repaired": False, "parsed_data": {"facts": []}, "error": "unparseable, repair failed"}
                        parsed = repaired
                        was_repaired = True
            elif isinstance(predicted_facts, list):
                parsed = predicted_facts
            elif isinstance(predicted_facts, dict):
                parsed = predicted_facts.get("facts", predicted_facts)
            else:
                return {"is_valid": False, "was_repaired": False, "parsed_data": {"facts": []}, "error": f"unsupported type {type(predicted_facts)}"}

            if isinstance(parsed, list):
                facts_list = parsed
            elif isinstance(parsed, dict) and "facts" in parsed:
                facts_list = parsed["facts"]
            else:
                facts_list = [parsed] if parsed else []

            return {"is_valid": True, "was_repaired": was_repaired, "parsed_data": {"facts": facts_list}, "error": None}
        except Exception as e:
            return {"is_valid": False, "was_repaired": False, "parsed_data": {"facts": []}, "error": str(e)}

    def _repair_json(self, raw_text):
        """Hacky fallback: ask a cheap model to reformat malformed fact-list text into valid JSON,
        preserving content rather than discarding it. Returns a parsed list/dict, or None on failure."""
        prompt = f"""The following text is supposed to be a JSON list of agricultural facts, but it has formatting errors (duplicate keys, missing colons, stray tokens, etc). Fix ONLY the JSON syntax — do not change, add, or remove any fact content or wording. Return ONLY the corrected JSON, no commentary.

MALFORMED TEXT:
{raw_text}
"""
        try:
            resp = self._client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You fix malformed JSON without altering its content. Respond ONLY with valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
            )
            content = resp.choices[0].message.content.strip()
            content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip())
            return json.loads(content)
        except Exception:
            return None

    def extract_facts_by_category(self, facts_data):
        category_facts = defaultdict(list)
        if isinstance(facts_data, str):
            try:
                facts_data = json.loads(facts_data)
            except Exception:
                try:
                    facts_data = ast.literal_eval(facts_data)
                except Exception:
                    return category_facts
        if isinstance(facts_data, list):
            facts_list = facts_data
        elif isinstance(facts_data, dict) and "facts" in facts_data:
            facts_list = facts_data.get("facts", [])
        else:
            return category_facts
        for fact in facts_list:
            if isinstance(fact, dict):
                category_facts[fact.get("category", "unknown")].append(fact)
            elif isinstance(fact, str):
                category_facts["unknown"].append({"fact": fact, "category": "unknown"})
        return dict(category_facts)

    def _extract_fact_text(self, fact_item):
        if isinstance(fact_item, str):
            return fact_item
        if isinstance(fact_item, dict):
            return fact_item.get("fact", str(fact_item))
        return str(fact_item)

    def find_best_semantic_match(self, gold_fact, pred_facts, category):
        if not pred_facts:
            return {"best_match": None, "reason": "No predicted facts available", "confidence": 0.0}
        prompt = f"""You are an agricultural fact comparison expert. Compare the reference fact with the candidate facts to find the best semantic match based on agricultural meaning and context.

REFERENCE FACT (Category: {category}):
{gold_fact}

CANDIDATE FACTS:
{json.dumps(pred_facts, indent=2)}

INSTRUCTIONS:
1. Find the candidate fact that conveys the most similar agricultural meaning to the reference fact
2. Prioritize matches that share the same crop/plant type, agricultural practice, measurements/dosages/timing, expected outcomes
3. Consider facts as matching even with different wording if they convey equivalent agricultural advice
4. If no candidate fact is semantically similar enough (confidence < 0.7), return null for best_match

RESPOND WITH ONLY JSON:
{{"best_match": "exact text of best matching candidate fact or null", "reason": "explanation", "confidence": 0.0-1.0}}
"""
        try:
            resp = self._client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": "You are an expert agricultural fact comparison specialist. Respond ONLY with valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
            )
            data = json.loads(resp.choices[0].message.content.strip())
            best_match = data.get("best_match")
            confidence = data.get("confidence", 0.0)
            if best_match and best_match in pred_facts:
                return {"best_match": best_match, "reason": data.get("reason", ""), "confidence": confidence}
            return {"best_match": None, "reason": data.get("reason", ""), "confidence": 0.0}
        except Exception as e:
            return {"best_match": None, "reason": f"error: {e}", "confidence": 0.0}

    def check_contradictions(self, pred_fact, gold_facts, category):
        if not gold_facts:
            return []
        prompt = f"""You are an agricultural contradiction-detection expert. Identify ONLY genuine contradictions between the CANDIDATE FACT and the REFERENCE FACTS.

CANDIDATE FACT (Category: {category}):
{pred_fact}

REFERENCE FACTS:
{json.dumps(gold_facts, indent=2)}

A genuine contradiction = opposite or conflicting claims about the SAME agricultural aspect (same subject and property).

RESPOND WITH ONLY THIS JSON:
{{"contradictions": [{{"reference_fact": "text", "reason": "short explanation", "confidence": "High|Med|Low"}}]}}
If none, return {{"contradictions": []}}.
"""
        try:
            resp = self._client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": "You are an expert agricultural contradiction detection specialist. Respond ONLY with valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
            )
            data = json.loads(resp.choices[0].message.content.strip())
            return data.get("contradictions", [])
        except Exception:
            return []

    def evaluate_unmatched_relevance(self, question, gold_facts, pred_fact):
        prompt = f"""You are an agricultural expert. Score how relevant, ground-truth-aligned, and practically useful this predicted fact is (1-10 overall_score).

QUESTION: {question}
GOLD_FACTS: {json.dumps(gold_facts)}
PREDICTED_FACT: {pred_fact}

RESPOND WITH ONLY THIS JSON:
{{"overall_score": 0, "explanation": "short reason"}}
"""
        try:
            resp = self._client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": "You are an expert agricultural fact evaluation specialist. Respond ONLY with valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
            )
            data = json.loads(resp.choices[0].message.content.strip())
            return data.get("overall_score", 0), data.get("explanation", "")
        except Exception as e:
            return 0, f"error: {e}"

    def evaluate(self, predicted_facts, golden_facts, question):
        json_result = self.check_valid_json(predicted_facts)
        pred_data = json_result["parsed_data"] if json_result["is_valid"] else predicted_facts

        pred_by_cat = self.extract_facts_by_category(pred_data)
        gold_by_cat = self.extract_facts_by_category(golden_facts)

        total_matches = total_gold = total_contradictions = 0
        total_relevant_unmatched = total_irrelevant_unmatched = 0
        detail = []

        for category, gold_items in gold_by_cat.items():
            pred_items = pred_by_cat.get(category, [])
            total_gold += len(gold_items)
            gold_texts = [self._extract_fact_text(f) for f in gold_items]
            pred_texts = [self._extract_fact_text(f) for f in pred_items]
            used = set()

            for gold_text in gold_texts:
                available = [f for f in pred_texts if f not in used]
                if not available:
                    detail.append({"gold_fact": gold_text, "status": "unmatched_gold"})
                    continue
                match = self.find_best_semantic_match(gold_text, available, category)
                if match["best_match"] and match["confidence"] >= 0.7:
                    used.add(match["best_match"])
                    total_matches += 1
                    detail.append({"gold_fact": gold_text, "matched_pred_fact": match["best_match"], "status": "matched"})
                else:
                    detail.append({"gold_fact": gold_text, "status": "unmatched_gold"})

            for pred_text in [f for f in pred_texts if f not in used]:
                contradictions = self.check_contradictions(pred_text, gold_texts, category)
                if contradictions:
                    total_contradictions += len(contradictions)
                    detail.append({"pred_fact": pred_text, "status": "contradictory", "contradictions": contradictions})
                else:
                    score, reason = self.evaluate_unmatched_relevance(question, gold_texts, pred_text)
                    if score >= 6:
                        total_relevant_unmatched += 1
                        detail.append({"pred_fact": pred_text, "status": "unmatched_relevant", "score": score})
                    else:
                        total_irrelevant_unmatched += 1
                        detail.append({"pred_fact": pred_text, "status": "unmatched_irrelevant", "score": score})

        total_predicted = total_matches + total_relevant_unmatched + total_irrelevant_unmatched
        recall = total_matches / total_gold if total_gold else 0
        precision = total_matches / total_predicted if total_predicted else 0
        relevance = (total_matches + total_relevant_unmatched) / total_predicted if total_predicted else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

        return {
            "json_validity": json_result["is_valid"],
            "json_was_repaired": json_result.get("was_repaired", False),
            "cleaned_facts": pred_data.get("facts", []) if json_result["is_valid"] else [],
            "total_gold_facts": total_gold,
            "total_matches": total_matches,
            "total_contradictions": total_contradictions,
            "total_relevant_unmatched": total_relevant_unmatched,
            "total_irrelevant_unmatched": total_irrelevant_unmatched,
            "fact_recall_iogt": recall,
            "fact_precision": precision,
            "relevance": relevance,
            "f1_score": f1,
            "detail": detail,
        }

    def extract_facts_from_text(self, question, response_text):
        """Run the same fact-extractor system prompt against a free-text response (the agent's answer)."""
        user_msg = (
            f"Extract atomic agricultural facts from this response text, which was given in answer to "
            f"the question: {question}\n\nRESPONSE TEXT:\n{response_text}"
        )
        try:
            resp = self._client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_msg}],
                response_format={"type": "json_object"},
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            return json.dumps({"facts": [], "error": str(e)})


def call_ft_model(client, meta_question):
    resp = client.chat.completions.create(
        model=FT_MODEL,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": meta_question}],
        temperature=0.0,
    )
    return resp.choices[0].message.content


def call_agent(question, language_id=1):
    """Fresh account per call — no shared-token bottleneck, and no cross-question
    conversation-history pollution (each account is used exactly once)."""
    client = FarmerChatClient(f"device_bench_{uuid.uuid4().hex[:16]}")
    client.initialize_user()
    client.set_language(language_id)
    client.new_conversation()
    resp = client.send_message(question)
    clarify = resp.get("agentic_trace", {}).get("surface", {}).get("payload", {}).get("message")
    return resp.get("response") or clarify or resp.get("message") or ""


def process_row(idx, row, openai_key, ft_only=False, agent_only=False):
    evaluator = FactEvaluator(openai_key)
    oai_client = openai.OpenAI(api_key=openai_key)

    result = {"sample_id": idx, "question": row["question"], "gt_facts": row["gt_facts"]}

    if not agent_only:
        try:
            ft_raw = call_ft_model(oai_client, row["meta_question"])
            result["ft_raw"] = ft_raw
            result["ft_eval"] = evaluator.evaluate(ft_raw, row["gt_facts"], row["question"])
        except Exception as e:
            result["ft_error"] = str(e)

    if ft_only:
        return result

    try:
        agent_raw = call_agent(row["question"])
        result["agent_raw"] = agent_raw
        agent_facts_json = evaluator.extract_facts_from_text(row["question"], agent_raw)
        result["agent_facts_json"] = agent_facts_json
        result["agent_eval"] = evaluator.evaluate(agent_facts_json, row["gt_facts"], row["question"])
    except Exception as e:
        result["agent_error"] = str(e)

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--ft-only", action="store_true", help="Skip the agent entirely (e.g. while FarmerChat staging is unreliable)")
    parser.add_argument("--agent-only", action="store_true", help="Skip the FT model call, only run the agent side")
    parser.add_argument("--out", default="/tmp/benchmark_results.jsonl")
    args = parser.parse_args()

    openai_key = _secret("OPENAI_API_KEY", "OPENAI_API_KEY")
    rows = load_test_set(TEST_PATH, sample=args.sample)
    print(f"Loaded {len(rows)} samples", flush=True)

    lock = threading.Lock()
    done = 0
    t0 = time.time()

    with open(args.out, "w") as out, ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(process_row, i, row, openai_key, args.ft_only, args.agent_only): i for i, row in enumerate(rows)}
        for future in as_completed(futures):
            idx = futures[future]
            try:
                result = future.result()
            except Exception as e:
                result = {"sample_id": idx, "error": str(e)}
            with lock:
                out.write(json.dumps(result) + "\n")
                out.flush()
                done += 1
                elapsed = time.time() - t0
                eta = (elapsed / done) * (len(rows) - done)
                print(f"[{done}/{len(rows)}] elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m", flush=True)

    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
