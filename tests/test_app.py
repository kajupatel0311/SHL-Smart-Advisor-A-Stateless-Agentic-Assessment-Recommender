from pathlib import Path

from fastapi.testclient import TestClient

from evaluate_traces import is_match, parse_trace_file
from src.main import app
from src.scraper import load_catalog, parse_conversation_markdown


client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_chat_endpoint():
    response = client.post(
        "/chat",
        json={
            "messages": [
                {"role": "user", "content": "We need a solution for senior leadership."}
            ]
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["reply"]
    assert isinstance(data["recommendations"], list)
    assert "end_of_conversation" in data


def test_catalog_loader_reads_existing_json():
    catalog = load_catalog(Path("data/shl_catalog.json"))
    assert catalog
    assert any(item.get("entity_id") == "4302" for item in catalog)


def test_parse_conversation_markdown():
    conversation = parse_conversation_markdown(Path("conversations/GenAI_SampleConversations/C1.md"))
    assert conversation["turn_count"] >= 4
    assert conversation["turns"][0]["user_text"].startswith("We need a solution for senior leadership")
    assert "agent_text" in conversation["turns"][0]


def test_parse_trace_file_extracts_ground_truth_and_state():
    messages, expected_names, behavior = parse_trace_file(Path("conversations/GenAI_SampleConversations/C1.md"))
    assert messages
    assert expected_names
    assert "Occupational Personality Questionnaire OPQ32r" in expected_names
    assert behavior["expected_end_of_conversation"] is True


def test_alias_matching_for_common_shl_acronyms():
    assert is_match("Occupational Personality Questionnaire OPQ32r", "OPQ")
    assert is_match("Occupational Personality Questionnaire OPQ32r", "OPQ32r")
    assert is_match("Global Skills Assessment", "GSA")


def test_chat_endpoint_returns_schema_error_payload_on_internal_failure(monkeypatch):
    class BrokenAgent:
        def process_chat(self, messages):
            raise RuntimeError("boom")

    monkeypatch.setattr("src.main.agent", BrokenAgent())
    monkeypatch.setattr("src.main.retriever", object())

    response = client.post(
        "/chat",
        json={"messages": [{"role": "user", "content": "Please help me"}]},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["reply"]
    assert data["recommendations"] == []
    assert data["end_of_conversation"] is False
