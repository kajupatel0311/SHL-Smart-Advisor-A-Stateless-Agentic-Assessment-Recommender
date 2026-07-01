import json
import os
from typing import List, Tuple

from dotenv import load_dotenv

load_dotenv()

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - optional dependency
    OpenAI = None


class RecommendationAgent:
    """
    Agent responsible for processing user queries and generating recommendations
    based on retrieved catalog data using OpenAI's models.
    """
    def __init__(self, catalog: List[dict], retriever) -> None:
        """
        Initialize the agent with the catalog and retriever.
        
        Args:
            catalog: List of dictionary items representing the SHL catalog.
            retriever: The SHLRetriever instance used to fetch relevant items.
        """
        self.catalog = catalog
        self.retriever = retriever
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) if OpenAI and os.getenv("OPENAI_API_KEY") else None
        self.system_prompt = """
        You are a Consultative SHL Advisor. Your goal is to guide users to the best SHL assessments.
        Follow these strict rules:
        - Clarify: If the conversation turn count is < 3 and the user's intent is vague, ask exactly one targeted follow-up question.
        - Refine: If a user mentions a new constraint (e.g., "only personality"), use the retrieved search results to filter current recommendations.
        - Compare: When comparing products, use a bulleted list based ONLY on the retrieved 'description' fields.
        - Strict Grounding: Never mention a product or URL not present in the provided context. Every URL must start with 'https://www.shl.com/'.
        - Set end_of_conversation to true only after a recommendation has been made AND the user has no further clarifying requests (i.e. they are satisfied).
        - Do not provide general hiring advice outside the scope of the catalog.
        """

    def build_system_prompt(self) -> str:
        """Returns the base system prompt."""
        return self.system_prompt

    def route_query(self, query: str) -> str:
        """
        Determines the theme of the query based on keyword matching.
        
        Args:
            query: User's query string.
            
        Returns:
            The theme of the query (e.g. leadership, communication, analytics, general).
        """
        lowered = query.lower()
        if any(keyword in lowered for keyword in ["lead", "manager", "coach", "team", "people"]):
            return "leadership"
        if any(keyword in lowered for keyword in ["communicat", "speak", "feedback", "present"]):
            return "communication"
        if any(keyword in lowered for keyword in ["data", "metric", "analytics", "analysis"]):
            return "analytics"
        return "general"

    def process_chat(self, messages: List[dict]) -> dict:
        """
        Processes a chat conversation and returns the agent's response.
        Retrieves context based on the latest message and formats the prompt for the OpenAI model.
        
        Args:
            messages: List of message dictionaries representing the conversation history.
            
        Returns:
            A dictionary matching the expected schema for the API response.
        """
        latest_message = messages[-1]["content"] if messages else ""
        results = self.retriever.search(latest_message, top_k=10)
        retrieved_context = {
            "role": "system",
            "content": f"Retrieved Catalog Data: {json.dumps(results)}. Only recommend from this list.",
        }
        prompt_messages = [{"role": "system", "content": self.system_prompt}] + messages + [retrieved_context]

        if self.client is not None:
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=prompt_messages,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "shl_recommender_response",
                            "strict": True,
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "reply": {"type": "string"},
                                    "recommendations": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "name": {"type": "string"},
                                                "url": {"type": "string"},
                                                "test_type": {"type": "string"},
                                            },
                                            "required": ["name", "url", "test_type"],
                                            "additionalProperties": False,
                                        },
                                    },
                                    "end_of_conversation": {"type": "boolean"},
                                },
                                "required": ["reply", "recommendations", "end_of_conversation"],
                                "additionalProperties": False,
                            },
                        },
                    },
                )
                return json.loads(response.choices[0].message.content)
            except Exception:
                pass

        reply = f"I can help narrow this down. Based on your latest message, I would focus on the most relevant SHL options from the catalog."
        recommendations = []
        for item, _ in results[:3]:
            recommendations.append(
                {
                    "name": item.get("name") or item.get("title") or "Unnamed item",
                    "url": item.get("link") or item.get("url") or "",
                    "test_type": item.get("test_type") or item.get("keys", ["General"])[0] or "General",
                }
            )

        return {
            "reply": reply,
            "recommendations": recommendations,
            "end_of_conversation": False,
        }

    def respond(self, query: str, results: List[Tuple[dict, float]]) -> str:
        """
        Fallback response generator when OpenAI is unavailable.
        
        Args:
            query: The user's query string.
            results: List of retrieved catalog items and their scores.
            
        Returns:
            A simple text response based on the top result.
        """
        mode = self.route_query(query)
        if not results:
            return f"I could not find a strong match for '{query}'."
        top_item = results[0][0]
        title = top_item.get("title") or top_item.get("name") or "the top matching product"
        return (
            f"For your request, I would prioritize {title} because it aligns with the "
            f"{mode} theme of '{query}'."
        )
