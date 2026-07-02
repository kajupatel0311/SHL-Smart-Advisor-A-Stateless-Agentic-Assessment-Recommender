import os
import json
from openai import OpenAI
from typing import List
from src.retriever import SHLRetriever
from dotenv import load_dotenv

load_dotenv()


class RecommendationAgent:
    """
    Agent responsible for processing user queries and generating recommendations
    based on retrieved catalog data using Groq (free) or OpenAI as fallback.
    """

    def __init__(self, catalog: List[dict], retriever: SHLRetriever):
        """
        Initialize the agent with the catalog and retriever.
        """
        self.catalog = catalog
        self.retriever = retriever

        # Support Groq as a free alternative, fallback to OpenAI
        if os.getenv("GROQ_API_KEY"):
            self.client = OpenAI(
                api_key=os.getenv("GROQ_API_KEY"),
                base_url="https://api.groq.com/openai/v1"
            )
            self.model_name = "llama-3.3-70b-versatile"
        else:
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.model_name = "gpt-4o-mini"

        self.system_instruction = (
            "You are an expert SHL Assessment Recommender. "
            "You help hiring managers select the right psychometric assessments from SHL's catalog. "
            "Rules: "
            "1. If the user is vague about role, level, or requirements, ask ONE clarifying question. "
            "2. ONLY recommend assessments that exist in the CATALOG DATA provided in the context. "
            "3. For each recommendation, use the EXACT 'name', 'link' as 'url', and first 'keys' item as 'test_type' from the catalog. "
            "4. Recommend 1-10 assessments maximum. "
            "5. If a user edits constraints, update your recommendations accordingly. "
            "6. Refuse to provide general hiring or legal advice. "
            "7. Set end_of_conversation to true ONLY when the user explicitly confirms they are done or satisfied. "
            "8. ALWAYS return ONLY a valid JSON object with EXACTLY this schema, no extra fields: "
            '{"reply": "string", "recommendations": [{"name": "string", "url": "https://www.shl.com/...", "test_type": "string"}], "end_of_conversation": false}'
        )

    def route_query(self, query: str) -> str:
        """Determines the theme of the query based on keyword matching."""
        lowered = query.lower()
        if any(keyword in lowered for keyword in ["lead", "manager", "coach", "team", "people"]):
            return "leadership"
        if any(keyword in lowered for keyword in ["communicat", "speak", "feedback", "present"]):
            return "communication"
        if any(keyword in lowered for keyword in ["data", "metric", "analytics", "analysis"]):
            return "analytics"
        return "general"

    def _sanitize_response(self, raw: dict, retrieved_data: list) -> dict:
        """Sanitizes raw LLM JSON output to strictly comply with the ChatResponse schema."""
        # Build a lookup from catalog data by name for fallback field resolution
        catalog_lookup = {}
        for item, _ in retrieved_data:
            name_key = (item.get("name") or item.get("title") or "").strip().lower()
            if name_key:
                catalog_lookup[name_key] = item

        clean_recs = []
        for rec in raw.get("recommendations", []):
            name = (rec.get("name") or rec.get("title") or "").strip()
            url = (rec.get("url") or rec.get("link") or "").strip()
            test_type = (rec.get("test_type") or rec.get("category") or "").strip()

            # Fill missing url/test_type from catalog lookup by name
            if not url or not test_type:
                catalog_item = catalog_lookup.get(name.lower(), {})
                # 'link' is the field name in our catalog
                url = url or catalog_item.get("link") or catalog_item.get("url") or ""
                # 'keys' is a list in our catalog; take first item as test_type
                keys = catalog_item.get("keys") or []
                test_type = test_type or (keys[0] if keys else "") or catalog_item.get("test_type") or "General"

            # Ensure SHL URL
            if url and not url.startswith("https://www.shl.com"):
                url = ""

            if name:
                clean_recs.append({
                    "name": name,
                    "url": url,
                    "test_type": test_type
                })

        return {
            "reply": raw.get("reply", ""),
            "recommendations": clean_recs,
            "end_of_conversation": bool(raw.get("end_of_conversation", False))
        }

    def process_chat(self, messages: list) -> dict:
        """Processes a multi-turn conversation and returns the agent's response."""
        # Combine ALL user messages for retrieval context to avoid losing early context
        user_messages = [msg['content'] for msg in messages if msg['role'] == 'user']
        combined_query = " ".join(user_messages) if user_messages else ""
        retrieved_data = self.retriever.search(combined_query, top_k=10)

        # Build the catalog context showing all relevant fields explicitly
        catalog_items_for_prompt = []
        for item, score in retrieved_data:
            keys_list = item.get("keys") or []
            catalog_items_for_prompt.append({
                "name": item.get("name") or item.get("title") or "",
                "url": item.get("link") or item.get("url") or "",
                "test_type": keys_list[0] if keys_list else "General",
                "description": (item.get("description") or "")[:300],
                "job_levels": item.get("job_levels") or [],
                "duration": item.get("duration") or ""
            })

        context_str = json.dumps(catalog_items_for_prompt, indent=2)

        # Build OpenAI-compatible message list
        api_messages = [{
            "role": "system",
            "content": self.system_instruction + f"\n\nCATALOG DATA (use ONLY these items):\n{context_str}"
        }]
        for msg in messages:
            api_messages.append({"role": msg["role"], "content": msg["content"]})

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=api_messages,
                response_format={"type": "json_object"}
            )
            raw = json.loads(response.choices[0].message.content)
            return self._sanitize_response(raw, retrieved_data)
        except Exception as e:
            return {
                "reply": f"I'm sorry, I encountered an error processing that request: {str(e)}",
                "recommendations": [],
                "end_of_conversation": False
            }
