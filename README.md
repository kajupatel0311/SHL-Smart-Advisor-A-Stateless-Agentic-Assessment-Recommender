# SHL Recommender

This project provides a starter structure for an SHL learning-product recommender.

## Project Structure

- data/shl_catalog.json: place your catalog JSON here.
- src/main.py: FastAPI application entrypoint.
- src/schemas.py: strict request/response models.
- src/scraper.py: catalog loading and scraping logic.
- src/retriever.py: vector search over catalog content.
- src/agent.py: routing and recommendation orchestration.

## How to use

1. Put your SHL catalog JSON into data/shl_catalog.json.
2. Add any API keys to .env if you later connect an LLM.
3. Run the app with uvicorn.

## Notes

The current implementation is a ready-to-fill scaffold. When you provide the actual catalog and sample conversation data, this project can be extended quickly.
