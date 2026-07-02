import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from src.agent import RecommendationAgent
from src.retriever import SHLRetriever
from src.schemas import ChatRequest, ChatResponse, Message
from src.scraper import load_catalog, scrape_catalog

load_dotenv()

retriever: Optional[SHLRetriever] = None
agent: Optional[RecommendationAgent] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager to initialize the retriever and agent 
    before the application starts serving requests.
    """
    global retriever, agent
    catalog_path = Path(os.getenv("SHL_CATALOG_PATH", "data/shl_catalog.json"))
    if not catalog_path.exists():
        scrape_catalog(catalog_path)
    catalog = load_catalog(catalog_path)
    retriever = SHLRetriever(catalog)
    agent = RecommendationAgent(catalog, retriever)
    yield


app = FastAPI(title="SHL Recommender", version="1.0.0", lifespan=lifespan)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Global exception handler to ensure API always returns a valid JSON 
    matching the ChatResponse schema, even on internal errors.
    """
    return JSONResponse(
        status_code=200,
        content={
            "reply": f"An internal error occurred: {str(exc)}",
            "recommendations": [],
            "end_of_conversation": False
        }
    )


@app.get("/health")
def health_check() -> dict:
    """Simple health check endpoint."""
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest) -> ChatResponse:
    """
    Processes chat requests using the RecommendationAgent.
    Logs the turn count and a basic heuristic for user intent.
    """
    global retriever, agent

    if agent is None or retriever is None:
        catalog_path = Path(os.getenv("SHL_CATALOG_PATH", "data/shl_catalog.json"))
        if not catalog_path.exists():
            scrape_catalog(catalog_path)
        catalog = load_catalog(catalog_path)
        retriever = SHLRetriever(catalog)
        agent = RecommendationAgent(catalog, retriever)

    # Log turn count and user intent
    turn_count = len(request.messages)
    last_message = request.messages[-1].content if request.messages else ""
    intent = agent.route_query(last_message) if agent else "unknown"
    print(f"Turn Count: {turn_count} | User Intent: {intent}")

    try:
        messages = [{"role": message.role, "content": message.content} for message in request.messages]
        response_data = agent.process_chat(messages)
        return ChatResponse(**response_data)
    except Exception as exc:
        # We also raise HTTP exceptions which will be caught by the global handler or FastAPI
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/catalog")
def list_catalog() -> List[dict]:
    """Returns the current loaded catalog data."""
    catalog_path = Path(os.getenv("SHL_CATALOG_PATH", "data/shl_catalog.json"))
    if not catalog_path.exists():
        scrape_catalog(catalog_path)
    return load_catalog(catalog_path)


