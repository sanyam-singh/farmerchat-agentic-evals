"""
Parse eval CSV files into structured test cases.

Each CSV row either starts a new test case (Test Code is non-empty)
or is a continuation turn (Test Code is empty).

Turn types:
  Farmer    → user message to send to the API
  FarmerChat → expected agent response
  Tool Call  → expected tool invocation (metadata only, not sent)
  (empty)    → blank separator row, skip
"""

import csv
from dataclasses import dataclass, field
from typing import Optional
from config import CSV_FILES


@dataclass
class Turn:
    role: str           # "farmer", "agent", "tool_call"
    text: str
    expected_chips: Optional[str] = None
    correct_chip: Optional[str] = None


@dataclass
class TestCase:
    test_code: str
    category: str       # PEST, NUTR, etc.
    scenario: str
    difficulty: str
    expected_action: str
    clarification_slots: str
    resolution_goal: str
    notes: str
    persona: str
    turns: list = field(default_factory=list)   # list of Turn


def _strip_chip_brackets(text: str) -> str:
    """Convert [Tomato] → Tomato for sending as query text."""
    text = text.strip()
    if text.startswith("[") and text.endswith("]"):
        return text[1:-1].strip()
    return text


def parse_csv(category: str, filepath: str) -> list:
    cases = []
    current: Optional[TestCase] = None

    with open(filepath, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            test_code = row["Test Code"].strip()
            ua = row["User/Agent"].strip()
            conv = row["Conversation"].strip()

            # Blank separator row
            if not ua and not test_code and not conv:
                continue

            # New test case starts
            if test_code:
                if current:
                    cases.append(current)
                current = TestCase(
                    test_code=test_code,
                    category=category,
                    scenario=row["Scenario"].strip(),
                    difficulty=row["Difficulty"].strip(),
                    expected_action=row["Expected Action"].strip(),
                    clarification_slots=row["Clarification Slots"].strip(),
                    resolution_goal=row["Resolution Goal (what we test)"].strip(),
                    notes=row["Notes"].strip(),
                    persona=row["Persona details"].strip(),
                )

            if not current:
                continue

            if ua == "Farmer" and conv:
                current.turns.append(Turn(
                    role="farmer",
                    text=_strip_chip_brackets(conv),
                ))
            elif ua == "FarmerChat" and conv:
                current.turns.append(Turn(
                    role="agent",
                    text=conv,
                    expected_chips=row["Chips Name"].strip() or None,
                    correct_chip=row["Correct Tap (which chip)"].strip() or None,
                ))
            elif ua == "Tool Call" and conv:
                current.turns.append(Turn(
                    role="tool_call",
                    text=conv,
                ))

    if current:
        cases.append(current)

    return cases


def load_all_cases(csv_files: dict = None) -> dict:
    """Returns {category: [TestCase, ...]}. Defaults to the English CSV set."""
    csv_files = csv_files if csv_files is not None else CSV_FILES
    all_cases = {}
    for category, filepath in csv_files.items():
        all_cases[category] = parse_csv(category, filepath)
        print(f"  Loaded {len(all_cases[category])} cases from {category}")
    return all_cases


if __name__ == "__main__":
    cases = load_all_cases()
    total = sum(len(v) for v in cases.values())
    print(f"\nTotal: {total} test cases across {len(cases)} categories")

    # Show a sample
    sample = cases["AMBG"][0]
    print(f"\nSample: {sample.test_code} | {sample.scenario} | {sample.difficulty}")
    for t in sample.turns:
        print(f"  [{t.role}] {t.text[:80]}")
