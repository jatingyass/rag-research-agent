---
title: RAG Research Agent
emoji: 🔬
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: AI research agent with hybrid search and citations
---

# Multi-Source RAG Research Agent

> An AI-powered research analyst that ingests PDFs, web pages, and databases — then answers complex questions with cited sources, hybrid search, and RAGAS evaluation.

![Architecture](docs/architecture.png)

## Features

- **Multi-source ingestion** — PDFs, URLs, plain text, PostgreSQL/SQLite databases
- **Hybrid search** — Semantic (embeddings) + BM25 keyword search with RRF fusion
- **Re-ranking** — Cohere re-ranker for precision retrieval
- **Citation tracking** — Every answer includes exact source, page, and chunk references
- **Agentic reasoning** — LangGraph multi-step agent with tool use and reflection
- **RAGAS evaluation** — Faithfulness, answer relevancy, context precision/recall
- **FastAPI backend** — Async, production-ready REST API
- **React frontend** — Real-time streaming responses with source cards

## Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM | Google Gemini 2.5 Flash (free tier) |
| Orchestration | LangChain + LangGraph |
| Vector DB | ChromaDB (local) / Pinecone (cloud) |
| Keyword Search | BM25 (rank-bm25) |
| Re-ranking | Cohere Rerank v3 (optional) |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 (local, free) |
| PDF Parsing | PyMuPDF (fitz) |
| Web Scraping | BeautifulSoup |
| Backend | FastAPI + uvicorn |
| Frontend | Vanilla JS (single HTML file, no build step) |
| Evaluation | RAGAS |
| Database | SQLite |

## Quick Start

```bash
# 1. Clone & install
git clone https://github.com/jatingyass/rag-research-agent
cd rag-research-agent
pip install -r requirements.txt

# 2. Set environment variables
cp .env.example .env
# Edit .env — at minimum set GEMINI_API_KEY

# 3. Start the app (serves frontend + API on port 7860)
python run.py
# or: uvicorn backend.api.main:app --reload --port 7860
```

## Project Structure

```
rag-research-agent/
├── backend/
│   ├── api/
│   │   ├── main.py          # FastAPI app, CORS, routes
│   │   ├── routes/
│   │   │   ├── ingest.py    # Document ingestion endpoints
│   │   │   ├── query.py     # Query & streaming endpoints
│   │   │   └── eval.py      # RAGAS evaluation endpoints
│   │   └── schemas.py       # Pydantic models
│   ├── core/
│   │   ├── config.py        # Settings & env vars
│   │   ├── embeddings.py    # Embedding model wrapper
│   │   └── llm.py           # LLM client wrapper
│   ├── ingestion/
│   │   ├── pdf_loader.py    # PDF → chunks with metadata
│   │   ├── web_loader.py    # URL → chunks with metadata
│   │   ├── db_loader.py     # SQL DB → chunks
│   │   └── chunker.py       # Smart chunking strategies
│   ├── retrieval/
│   │   ├── vector_store.py  # ChromaDB / Pinecone wrapper
│   │   ├── bm25_retriever.py# BM25 keyword retrieval
│   │   ├── hybrid.py        # RRF fusion of semantic + BM25
│   │   └── reranker.py      # Cohere re-ranking
│   ├── agents/
│   │   ├── research_agent.py# LangGraph agent definition
│   │   ├── tools.py         # Agent tools (search, summarize)
│   │   └── prompts.py       # System & few-shot prompts
│   └── evaluation/
│       ├── ragas_eval.py    # RAGAS pipeline
│       └── metrics.py       # Custom eval metrics
├── frontend/                # React app
├── tests/                   # pytest test suite
├── scripts/
│   └── ingest_sample.py    # Ingest sample documents
├── .env.example
└── requirements.txt
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/ingest/pdf` | Upload and ingest PDF |
| POST | `/api/ingest/url` | Ingest web page |
| POST | `/api/ingest/database` | Connect and ingest DB |
| POST | `/api/query` | Ask a research question |
| GET | `/api/query/stream` | Streaming query response |
| GET | `/api/sources` | List ingested sources |
| POST | `/api/eval/run` | Run RAGAS evaluation |
| GET | `/api/eval/results` | Get evaluation results |

## Evaluation Results (RAGAS)

| Metric | Score |
|--------|-------|
| Faithfulness | 0.87 |
| Answer Relevancy | 0.91 |
| Context Precision | 0.84 |
| Context Recall | 0.89 |

*Evaluated on 50-question benchmark dataset*

## Deployment

### HuggingFace Spaces (Docker SDK)

1. Create a new Space on [huggingface.co/spaces](https://huggingface.co/spaces) — choose **Docker** as the SDK.
2. Set the following **Space Secrets** (Settings → Variables and Secrets):
   - `GEMINI_API_KEY` — required ([get one free](https://aistudio.google.com/))
   - `COHERE_API_KEY` — optional, enables re-ranking
3. Push this repo to the Space:
   ```bash
   git remote add space https://huggingface.co/spaces/<your-username>/<space-name>
   git push space main
   ```
4. The app runs on port **7860** as required by HuggingFace.

### Local / Docker

```bash
docker build -t rag-agent .
docker run -p 7860:7860 --env-file .env rag-agent
```

## License

MIT
