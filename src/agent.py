import os
import json
from openai import OpenAI
from typing import List, Tuple
from src.retriever import SHLRetriever
from dotenv import load_dotenv

load_dotenv()

class RecommendationAgent:
    """
    Agent responsible for processing user queries and generating recommendations
    based on retrieved catalog data using OpenAI's models.
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
            self.model_name = "llama-3.1-8b-instant" # Fast, free model on Groq
        else:
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.model_name = "gpt-4o-mini"
            
        self.system_instruction = (
            "You are an expert SHL Assessment Recommender. "
            "1. If the user is vague, ask a clarifying question. "
            "2. Recommend 1-10 assessments ONLY from the provided catalog data. "
            "3. If the user edits constraints, update the list. "
            "4. Compare tests accurately using catalog data. "
            "5. Refuse general hiring/legal advice. "
            "6. ALWAYS return a JSON object with EXACTLY this schema, with no extra fields: "
            '{"reply": "string", "recommendations": [{"name": "string", "url": "https://www.shl.com/...", "test_type": "string"}], "end_of_conversation": true/false}'
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
        """Sanitizes the raw JSON output to strictly comply with the ChatResponse schema."""
        catalog_lookup = {}
        for item, _ in retrieved_data:
            name_key = (item.get("name") or item.get("title") or "").strip().lower()
            if name_key:
                catalog_lookup[name_key] = item

        clean_recs = []
        for rec in raw.get("recommendations", []):
            name = rec.get("name") or rec.get("title") or ""
            url = rec.get("url") or rec.get("link") or ""
            test_type = rec.get("test_type") or rec.get("category") or ""

            if not url or not test_type:
                catalog_item = catalog_lookup.get(name.strip().lower(), {})
                url = url or catalog_item.get("url") or catalog_item.get("link") or ""
                test_type = test_type or catalog_item.get("test_type") or catalog_item.get("category") or "General"

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
        """Processes a chat conversation and returns the agent's response."""
        latest_query = messages[-1]['content'] if messages else ""
        retrieved_data = self.retriever.search(latest_query, top_k=10)
        
        context_str = f"CATALOG DATA: {json.dumps(retrieved_data)}"
        
        api_messages = [{"role": "system", "content": self.system_instruction + f"\n\nContext: {context_str}"}]
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

