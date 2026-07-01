import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from bs4 import BeautifulSoup


DEFAULT_PRODUCTS: List[Dict[str, Any]] = [
    {
        "id": "shl-101",
        "title": "Leadership Coaching Essentials",
        "category": "Leadership",
        "summary": "A structured coaching program for first-time managers and team leads.",
        "tags": ["leadership", "coaching", "management"],
        "suitability": "Best for emerging leaders and supervisors.",
        "difficulty": "Beginner",
        "source": "generated",
    },
    {
        "id": "shl-102",
        "title": "High-Impact Communication Workshop",
        "category": "Communication",
        "summary": "Practice feedback, meeting facilitation, and stakeholder communication.",
        "tags": ["communication", "feedback", "facilitation"],
        "suitability": "Ideal for customer-facing and cross-functional roles.",
        "difficulty": "Intermediate",
        "source": "generated",
    },
    {
        "id": "shl-103",
        "title": "Data-Driven Decision Making",
        "category": "Analytics",
        "summary": "Teaches practical methods to interpret metrics and improve team decisions.",
        "tags": ["analytics", "decision-making", "metrics"],
        "suitability": "Useful for managers building an evidence-based culture.",
        "difficulty": "Intermediate",
        "source": "generated",
    },
    {
        "id": "shl-104",
        "title": "Conflict Resolution for Teams",
        "category": "People Management",
        "summary": "A playbook for managing tension, misalignment, and difficult conversations.",
        "tags": ["conflict", "people", "resolution"],
        "suitability": "Great for team leaders and HR partners.",
        "difficulty": "Intermediate",
        "source": "generated",
    },
    {
        "id": "shl-105",
        "title": "Strategic Thinking Accelerator",
        "category": "Strategy",
        "summary": "Helps organizations shape priorities and align work to long-term goals.",
        "tags": ["strategy", "planning", "alignment"],
        "suitability": "Best for senior managers and directors.",
        "difficulty": "Advanced",
        "source": "generated",
    },
]


def build_catalog() -> List[Dict[str, Any]]:
    return [dict(item) for item in DEFAULT_PRODUCTS]


def load_catalog(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    catalog_path = path or Path(__file__).resolve().parents[1] / "data" / "shl_catalog.json"
    if not catalog_path.exists():
        return build_catalog()

    try:
        with catalog_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        raw_text = catalog_path.read_text(encoding="utf-8")
        cleaned = re.sub(r"\\x[0-9a-fA-F]{2}", "", raw_text)
        cleaned = cleaned.replace("\x00", "")
        cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)

        objects: List[Dict[str, Any]] = []
        for match in re.finditer(r"\{.*?\}", cleaned, flags=re.S):
            candidate = match.group(0)
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                objects.append(parsed)

        if objects:
            return objects
        return build_catalog()


def parse_conversation_markdown(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    turn_blocks = re.split(r"### Turn\s+\d+", text)
    turns: List[Dict[str, str]] = []
    for block in turn_blocks[1:]:
        block = block.strip()
        if not block:
            continue
        user_match = re.search(r"\*\*User\*\*\s*(?:\n|\r\n)?>(.*?)\n\n\*\*Agent\*\*", block, re.S)
        agent_match = re.search(r"\*\*Agent\*\*\s*(.*?)(?:\n\n_`end_of_conversation`|$)", block, re.S)
        if not user_match and not agent_match:
            continue
        turns.append(
            {
                "user_text": re.sub(r"\s+", " ", user_match.group(1)).strip() if user_match else "",
                "agent_text": re.sub(r"\s+", " ", agent_match.group(1)).strip() if agent_match else "",
            }
        )
    return {"path": str(path), "turn_count": len(turns), "turns": turns}


def fetch_source_catalog(source_url: Optional[str]) -> Optional[List[Dict[str, Any]]]:
    if not source_url:
        return None
    try:
        response = requests.get(source_url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        cards = soup.select("article, .card, .product")
        if not cards:
            return None
        items: List[Dict[str, Any]] = []
        for index, card in enumerate(cards[:8], start=1):
            title = card.get_text(" ", strip=True)[:80]
            items.append(
                {
                    "id": f"scraped-{index}",
                    "title": title or f"Catalog item {index}",
                    "category": "General",
                    "summary": "Imported from a remote catalog source.",
                    "tags": ["scraped"],
                    "suitability": "General recommendation",
                    "difficulty": "Intermediate",
                    "source": source_url,
                }
            )
        return items
    except Exception:
        return None


def write_catalog(path: Path, catalog: Optional[List[Dict[str, Any]]] = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    items = catalog or build_catalog()
    with path.open("w", encoding="utf-8") as handle:
        json.dump(items, handle, indent=2)
    return path


def scrape_catalog(output_path: Path, source_url: Optional[str] = None) -> List[Dict[str, Any]]:
    catalog = fetch_source_catalog(source_url) or load_catalog(output_path)
    write_catalog(output_path, catalog)
    return catalog


if __name__ == "__main__":
    output = Path(__file__).resolve().parents[1] / "data" / "shl_catalog.json"
    scrape_catalog(output)
    print(f"Wrote catalog to {output}")
