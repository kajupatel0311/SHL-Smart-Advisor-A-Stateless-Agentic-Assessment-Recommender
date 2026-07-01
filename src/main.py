import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

from src.agent import RecommendationAgent
from src.retriever import SHLRetriever
from src.schemas import ChatRequest, ChatResponse, Message
from src.scraper import load_catalog, scrape_catalog

load_dotenv()

retriever: Optional[SHLRetriever] = None
agent: Optional[RecommendationAgent] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global retriever, agent
    catalog_path = Path(os.getenv("SHL_CATALOG_PATH", "data/shl_catalog.json"))
    if not catalog_path.exists():
        scrape_catalog(catalog_path)
    catalog = load_catalog(catalog_path)
    retriever = SHLRetriever(catalog)
    agent = RecommendationAgent(catalog, retriever)
    yield


app = FastAPI(title="SHL Recommender", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health_check() -> dict:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest) -> ChatResponse:
    global retriever, agent

    if agent is None or retriever is None:
        catalog_path = Path(os.getenv("SHL_CATALOG_PATH", "data/shl_catalog.json"))
        if not catalog_path.exists():
            scrape_catalog(catalog_path)
        catalog = load_catalog(catalog_path)
        retriever = SHLRetriever(catalog)
        agent = RecommendationAgent(catalog, retriever)

    try:
        messages = [{"role": message.role, "content": message.content} for message in request.messages]
        response_data = agent.process_chat(messages)
        return ChatResponse(**response_data)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/catalog")
def list_catalog() -> List[dict]:
    catalog_path = Path(os.getenv("SHL_CATALOG_PATH", "data/shl_catalog.json"))
    if not catalog_path.exists():
        scrape_catalog(catalog_path)
    return load_catalog(catalog_path)
