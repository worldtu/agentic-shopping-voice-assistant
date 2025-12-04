#!/bin/bash
# Quick start script for TTS API server

echo "========================================="
echo "Voice Shopping Assistant - API Server"
echo "========================================="
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo "⚠️  .env file not found!"
    echo "Creating .env from .env.example..."
    cp .env.example .env
    echo ""
    echo "❌ Please edit .env and add your OPENAI_API_KEY"
    echo "   Then run this script again."
    exit 1
fi

# Check if OPENAI_API_KEY is set
set -a
source .env
set +a
if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ OPENAI_API_KEY not set in .env file"
    echo "   Please edit .env and add: OPENAI_API_KEY=sk-your-key-here"
    exit 1
fi

echo "✅ Environment configured"
echo "✅ OPENAI_API_KEY found"
echo ""

# Check if dependencies installed
if ! python -c "import fastapi" 2>/dev/null; then
    echo "⚠️  FastAPI not installed"
    echo "Installing dependencies..."
    pip install -r requirements.txt
    echo ""
fi

echo "🚀 Starting API server on http://localhost:8000"
echo ""
echo "Available endpoints:"
echo "  - GET  /health              Health check"
echo "  - POST /api/tts             Generate TTS"
echo "  - GET  /api/tts/audio/{id}  Get audio file"
echo "  - POST /api/query           Complete query pipeline"
echo ""
echo "API docs available at:"
echo "  - http://localhost:8000/docs (Swagger UI)"
echo "  - http://localhost:8000/redoc (ReDoc)"
echo ""
echo "Press Ctrl+C to stop the server"
echo "========================================="
echo ""

# Start the server
uvicorn backend.api_gateway:app --reload --port 8000
