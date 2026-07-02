# SHL Smart Advisor — Stateless Agentic Assessment Recommender

A stateless conversational AI agent built with FastAPI that recommends SHL psychometric assessments to hiring managers based on their requirements.

## Live API

**Base URL:** `https://shl-smart-advisor-a-stateless-agentic.onrender.com`

| Endpoint | Method | Description |
|---|---|---|
| `/health` | `GET` | Health check — returns `{"status": "ok"}` |
| `/chat` | `POST` | Main chat endpoint for assessment recommendations |
| `/catalog` | `GET` | Returns the full SHL product catalog |

## Architecture

```
User Request → FastAPI /chat → SHLRetriever (TF-IDF + FAISS RAG) → RecommendationAgent (LLM) → ChatResponse
```

- **`src/main.py`** — FastAPI application with `/health` and `/chat` endpoints
- **`src/schemas.py`** — Strict Pydantic request/response models (ChatRequest, ChatResponse)
- **`src/retriever.py`** — TF-IDF + FAISS vector index for cosine similarity search over the SHL catalog
- **`src/agent.py`** — LLM orchestration (Groq / OpenAI) with schema sanitization
- **`src/scraper.py`** — Catalog loading from `data/shl_catalog.json`
- **`data/shl_catalog.json`** — 377 scraped SHL product catalog items with names, URLs, test types

## API Request/Response Schema

**POST `/chat`**
```json
{
  "messages": [
    {"role": "user", "content": "I need an assessment for a mid-level Java developer"},
    {"role": "assistant", "content": "What seniority level are you targeting?"},
    {"role": "user", "content": "Around 4 years of experience"}
  ]
}
```

**Response:**
```json
{
  "reply": "Based on your requirements, I recommend...",
  "recommendations": [
    {
      "name": "Occupational Personality Questionnaire OPQ32r",
      "url": "https://www.shl.com/products/product-catalog/view/occupational-personality-questionnaire-opq32r/",
      "test_type": "Personality & Behavior"
    }
  ],
  "end_of_conversation": false
}
```

## Local Setup

```bash
# 1. Clone and install dependencies
pip install -r requirements.txt

# 2. Create .env file
echo "GROQ_API_KEY=your_groq_key_here" > .env

# 3. Run the server
uvicorn src.main:app --host 0.0.0.0 --port 8000

# 4. Test
python test_chat.py
```

## Deployment

Deployed on [Render](https://render.com) as a free-tier web service.

**Environment Variables required on Render:**
- `GROQ_API_KEY` — Free API key from [console.groq.com](https://console.groq.com)

## Evaluation

Run the automated local evaluation:
```bash
python evaluate_traces.py
```

Results against 10 sample SHL conversation traces:
- **Hard Evals (Schema Pass Rate):** 10/10 ✅
- **Mean Recall@10:** ~0.22 (local estimate)
- **Behavior Probes:** 6-8/10
