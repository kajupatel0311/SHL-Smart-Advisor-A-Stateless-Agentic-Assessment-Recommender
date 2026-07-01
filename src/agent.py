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
            model_name="gemini-1.5-pro",
            system_instruction=(
                "You are an expert SHL Assessment Recommender. "
                "1. If the user is vague, ask a clarifying question. "
                "2. Recommend 1-10 assessments ONLY from the provided catalog data. "
                "3. If the user edits constraints, update the list. "
                "4. Compare tests accurately using catalog data. "
                "5. Refuse general hiring/legal advice. "
                "6. ALWAYS return a JSON object with keys: 'reply', 'recommendations', and 'end_of_conversation'."
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
            # Parse the JSON string from Gemini
            return json.loads(response.text)
        except Exception as e:
            # Fallback if JSON parsing or generation fails
            return {
                "reply": f"I'm sorry, I encountered an error processing that request: {str(e)}",
                "recommendations": [],
                "end_of_conversation": False
            }
