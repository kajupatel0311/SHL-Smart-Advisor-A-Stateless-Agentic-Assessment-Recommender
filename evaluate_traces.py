import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import List

import requests

TRACES_DIR = Path("conversations/GenAI_SampleConversations")
API_URL = "https://shl-smart-advisor-a-stateless-agentic.onrender.com/chat"
ALIASES = {
    "opq": ["occupational personality questionnaire", "opq32r", "opq32"],
    "gsa": ["global skills assessment"],
}


def normalize_name(name: str) -> str:
    name = (name or "").lower()
    name = re.sub(r"[^\w\s]", "", name)
    for filler in ["assessment", "test", "product", "solution", "the", "an", "a"]:
        name = re.sub(rf"\b{filler}\b", "", name)
    return re.sub(r"\s+", " ", name).strip()


def expand_aliases(name: str) -> list[str]:
    normalized = normalize_name(name)
    if not normalized:
        return []
    variants = {normalized}
    for alias, expansions in ALIASES.items():
        if alias in normalized:
            variants.update(normalize_name(expansion) for expansion in expansions)
        for expansion in expansions:
            if normalize_name(expansion) in normalized:
                variants.add(alias)
    return list(variants)


def is_match(expected: str, recommended: str, threshold: float = 0.7) -> bool:
    norm_expected = normalize_name(expected)
    norm_recommended = normalize_name(recommended)
    if not norm_expected or not norm_recommended:
        return False

    expected_variants = expand_aliases(expected)
    recommended_variants = expand_aliases(recommended)
    if any(v in expected_variants for v in recommended_variants) or any(v in recommended_variants for v in expected_variants):
        return True

    if norm_expected in norm_recommended or norm_recommended in norm_expected:
        return True
    similarity = SequenceMatcher(None, norm_expected, norm_recommended).ratio()
    return similarity >= threshold


def calculate_recall_at_k(recommended_names: List[str], expected_names: List[str], k: int = 10) -> float:
    if not expected_names:
        return 0.0
    top_k = [name for name in recommended_names[:k] if name]
    unique_expected = list(dict.fromkeys(expected_names))
    unique_recommended = list(dict.fromkeys(top_k))
    hits = 0
    for target in unique_expected:
        if any(is_match(target, rec) for rec in unique_recommended):
            hits += 1
    return hits / len(unique_expected)


def parse_trace_file(path: Path) -> tuple[list[dict], list[str], dict]:
    text = path.read_text(encoding="utf-8")
    turns = []
    for block in text.split("### Turn"):
        if not block.strip() or "**User**" not in block:
            continue
        user_match = re.search(r"\*\*User\*\*\s*(?:\n|\r\n)?>(.*?)\n\n\*\*Agent\*\*", block, re.S)
        if not user_match:
            continue
        user_text = " ".join(user_match.group(1).replace(">", "").split())
        turns.append({"role": "user", "content": user_text})

    expected_names = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [cell.strip() for cell in stripped.split("|")[1:-1]]
        if len(cells) < 2:
            continue
        if not cells[1] or cells[1].lower() in {"name", "test type", "keys", "duration", "languages", "url"}:
            continue
        if cells[0] in {"#", "---", "------"}:
            continue
        name = re.sub(r"<.*?>", "", cells[1])
        name = re.sub(r"[`*_]", "", name)
        name = name.replace("(New)", "").strip()
        if not name or name in {"---", "------"}:
            continue
        if name.startswith(("Occupational", "SVAR", "Contact", "Entry", "Customer", "Smart", "Linux", "Networking", "SHL", "OPQ")) or "Questionnaire" in name or "Report" in name or "Simulation" in name or "Programming" in name:
            expected_names.append(name)

    expected_end = True if "_`end_of_conversation`: **true**_" in text.lower() else False
    behavior = {"expected_end_of_conversation": expected_end}
    return turns, expected_names, behavior


def run_evaluation() -> None:
    print("Starting SHL Automated Evaluation Simulator...\n")

    total_recall = 0.0
    trace_count = 0
    schema_passes = 0
    behavior_passes = 0

    for path in sorted(TRACES_DIR.glob("*.md")):
        print(f"Evaluating trace: {path.name}")
        messages, expected_names, behavior = parse_trace_file(path)
        payload = {"messages": messages}

        try:
            response = requests.post(API_URL, json=payload, timeout=120)
            if response.status_code != 200:
                print(f"  [FAIL] Server Error {response.status_code}: {response.text}")
                continue

            data = response.json()
            assert "reply" in data
            assert "recommendations" in data
            assert "end_of_conversation" in data
            schema_passes += 1

            recommended_tests = [rec.get("name", "") for rec in data.get("recommendations", [])]
            recall = calculate_recall_at_k(recommended_tests, expected_names, k=10)
            total_recall += recall
            trace_count += 1

            if len(messages) < 3:
                expected_state = False
            else:
                expected_state = behavior["expected_end_of_conversation"]
            if data.get("end_of_conversation") is expected_state and (expected_state or data.get("recommendations")):
                behavior_passes += 1

            print(f"  [PASS] Schema Valid | Recall@10: {recall:.2f} | Behavior: {data.get('end_of_conversation')}")
            print(f"  Agent Reply: {data.get('reply', '')[:80]}...\n")
        except Exception as exc:
            print(f"  [FAIL] Test Failed: {exc}")
        
        # Add a sleep to prevent hitting Gemini's free tier rate limits (429 errors)
        import time
        time.sleep(5)

    print("=== FINAL LOCAL EVALUATION REPORT ===")
    print(f"Total Traces Processed: {trace_count}")
    print(f"Hard Evals (Schema Pass Rate): {schema_passes}/{trace_count}")
    print(f"Behavior Probes Passed: {behavior_passes}/{trace_count}")
    if trace_count > 0:
        print(f"Mean Recall@10: {(total_recall / trace_count):.2f}")


if __name__ == "__main__":
    run_evaluation()
