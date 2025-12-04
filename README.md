# Agentic Shopping Voice Assistant — Backend

This repository contains the **full Python backend** for the voice-to-voice shopping assistant:

- LangGraph router → planner → retriever → answerer pipeline
- Private FAISS RAG index + Groq-powered filter extraction
- MCP tools for live web search (`web.search`) and private RAG (`rag.search`)
- FastAPI voice service (`voice/`) with OpenAI TTS integration
- Tests, scripts, and sample data (`data/`, `chroma_db/`)

The React UI lives separately in `../frontend_TTS`. Point that frontend at this API via `VITE_API_URL`.

## Project Structure

```
agentic-shopping-voice-assistant/
├── backend/                    # 🆕 Unified API Gateway (FastAPI)
│   ├── __init__.py
│   └── api_gateway.py          # Main FastAPI app - routes all requests
├── graph/                      # LangGraph pipeline (router, planner, retriever, answerer)
├── mcp_server/                 # MCP tool shims + Serper web search client
├── voice/                      # Voice processing modules (TTS/ASR utilities)
│   ├── __init__.py
│   └── tts.py                  # OpenAI TTS wrapper
├── scripts/                    # Data prep utilities (extract_metadata, index_data)
├── data/                       # Product datasets (parquet + CSV)
├── chroma_db/                  # Vector store artifacts
├── tests/                      # Pytest suites (router, planner, retrievers, answerer)
├── tts_output/                 # Generated audio clips (MP3)
├── start_api.sh                # Convenience launcher for the API gateway
├── requirements.txt            # Python dependencies
├── SETUP_GUIDE.md              # 🆕 Complete setup instructions
└── README.md                   # This file
```

## Setup

```bash
python -m venv .venv
. .venv/Scripts/activate   # or source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
```

Populate `.env.local` (or export env vars) with at least:

```
OPENAI_API_KEY=...
GROQ_API_KEY=...
SERPER_API_KEY=...
DATA_DRIVE_ID=...
EMB_DRIVE_ID=...
```

## Running Services

1. **LangGraph / MCP stack**  
   Use the provided demo entry points or integrate via `graph/graph.py`.  
   Tests demonstrate typical usage: `pytest tests`.

2. **Backend API Gateway** (unified FastAPI service)
   ```bash
   uvicorn backend.api_gateway:app --reload --port 8000
   ```
   or run `./start_api.sh` (sets up env + uvicorn). This exposes:
   - `GET /health`
   - `POST /api/tts`  
   - `POST /api/asr`  
   - `POST /api/query` (complete LangGraph pipeline)
   - `GET /api/tts/audio/{audio_id}`

3. **Web + RAG MCP tools**  
   `mcp_server/server_stdio.py` can be launched if you need a dedicated MCP host,
   but LangGraph nodes already call the synchronous shims (`rag_search_sync`, `web_search_sync`).

## Frontend Integration

- Start this backend first (LangGraph workflow + FastAPI API gateway).
- In `frontend_TTS`, set `VITE_API_URL` to `http://localhost:8000`.
- The frontend calls:
  - `POST /api/query` for complete voice-to-voice pipeline
  - `POST /api/tts` for text-to-speech
  - `POST /api/asr` for speech-to-text
  - `GET /api/tts/audio/{id}` for audio playback

**For complete setup instructions including frontend configuration, see [SETUP_GUIDE.md](./SETUP_GUIDE.md)**

Keeping backend and frontend in separate repositories avoids duplicated Python assets in the UI project.
