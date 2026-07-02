import os
import json
import google.generativeai as genai
from typing import List, Tuple
from src.retriever import SHLRetriever
from dotenv import load_dotenv

load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

class RecommendationAgent:
    """
    Agent responsible for processing user queries and generating recommendations
    based on retrieved catalog data using Google's Gemini models.
    """
    def __init__(self, catalog: List[dict], retriever: SHLRetriever):
        """
        Initialize the agent with the catalog and retriever.
        
        Args:
            catalog: List of dictionary items representing the SHL catalog.
            retriever: The SHLRetriever instance used to fetch relevant items.
        """
        self.catalog = catalog
        self.retriever = retriever
        # Initialize the model with the system instruction
        self.model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            system_instruction=(
                "You are an expert SHL Assessment Recommender. "
                "1. If the user is vague, ask a clarifying question. "
                "2. Recommend 1-10 assessments ONLY from the provided catalog data. "
                "3. If the user edits constraints, update the list. "
                "4. Compare tests accurately using catalog data. "
                "5. Refuse general hiring/legal advice. "
                "6. ALWAYS return a JSON object with EXACTLY this schema, with no extra fields: "
                '{"reply": "string", "recommendations": [{"name": "string", "url": "https://www.shl.com/...", "test_type": "string"}], "end_of_conversation": true/false}'
            )
        )

    def route_query(self, query: str) -> str:
        """
        Determines the theme of the query based on keyword matching.
        Preserved for logging compatibility in main.py.
        """
        lowered = query.lower()
        if any(keyword in lowered for keyword in ["lead", "manager", "coach", "team", "people"]):
            return "leadership"
        if any(keyword in lowered for keyword in ["communicat", "speak", "feedback", "present"]):
            return "communication"
        if any(keyword in lowered for keyword in ["data", "metric", "analytics", "analysis"]):
            return "analytics"
        return "general"

    def _sanitize_response(self, raw: dict, retrieved_data: list) -> dict:
        """
        Sanitizes the raw Gemini JSON output to strictly comply with the ChatResponse schema.
        Strips extra fields like entity_id, description, reasoning.
        Maps catalog data to fill missing 'url' and 'test_type' fields if needed.
        """
        # Build a quick lookup from catalog data by name for fallback field resolution
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

            # Try to fill missing url/test_type from catalog lookup by name
            if not url or not test_type:
                catalog_item = catalog_lookup.get(name.strip().lower(), {})
                url = url or catalog_item.get("url") or catalog_item.get("link") or ""
                test_type = test_type or catalog_item.get("test_type") or catalog_item.get("category") or "General"

            # Only include recommendations that have at minimum a name
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
        """
        Processes a chat conversation and returns the agent's response.
        Retrieves context based on the latest message and formats the prompt for Gemini.
        """
        # Get the latest query for RAG
        latest_query = messages[-1]['content'] if messages else ""
        retrieved_data = self.retriever.search(latest_query, top_k=10)
        
        # Build the prompt with retrieved context
        context_str = f"CATALOG DATA: {json.dumps(retrieved_data)}"
        prompt = f"Context: {context_str}\n\nHistory: {json.dumps(messages)}\n\nResponse:"

        try:
            # Call Gemini with JSON constrained output
            response = self.model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json",
                )
            )
            # Parse the JSON string from Gemini then sanitize to match schema
            raw = json.loads(response.text)
            return self._sanitize_response(raw, retrieved_data)
        except Exception as e:
            # Fallback if JSON parsing or generation fails
            return {
                "reply": f"I'm sorry, I encountered an error processing that request: {str(e)}",
                "recommendations": [],
                "end_of_conversation": False
            }

