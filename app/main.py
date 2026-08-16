import os
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from app.agent import stream_chat_response
from app.memory import memory_manager
from app.safety import safety_guard

# Load environment variables from .env file
load_dotenv()

app = FastAPI(
    title="MetroCity Smart Infrastructure GPT Chatbot API",
    version="1.0.0",
    description="Production-grade task-oriented chatbot supporting memory, function calling, streaming, and telemetry."
)

# Configure CORS for local web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static directory path for Web UI
STATIC_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static"
)

if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# Pydantic Input Data Models
class ChatRequest(BaseModel):
    conversation_id: str = Field(..., example="session-101", description="Unique session identifier")
    user_message: str = Field(..., example="What is the weather in Metrocity?", description="User query text")
    model: str = Field("llama-3.1-8b-instant", description="Groq model identifier")
    temperature: float = Field(0.2, ge=0.0, le=1.0, description="Sampling temperature")


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serves the minimal web UI HTML page."""
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        with open(index_file, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h2>MetroCity Smart Infrastructure Chatbot API Running</h2>")


@app.get("/health")
async def health_check():
    """System status check endpoint."""
    return {
        "status": "online",
        "system": "MetroCity GPT Engine",
        "api_key_configured": bool(os.getenv("GROQ_API_KEY"))
    }


@app.post("/chat")
async def chat_endpoint(request: Request, body: ChatRequest):
    """
    Main Streaming REST Endpoint.
    Accepts conversation_id and user_message, returns real-time SSE token stream.
    """
    # Rate limit check per IP/Client
    client_ip = request.client.host if request.client else "unknown"
    if not safety_guard.check_rate_limit(client_ip):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please wait before sending more messages."
        )

    return StreamingResponse(
        stream_chat_response(
            conversation_id=body.conversation_id,
            user_message=body.user_message,
            model=body.model,
            temperature=body.temperature
        ),
        media_type="text/event-stream"
    )


@app.delete("/history/{conversation_id}")
async def clear_history_endpoint(conversation_id: str):
    """Resets memory state for a specific conversation session."""
    memory_manager.clear_history(conversation_id)
    return {"status": "success", "message": f"History cleared for session '{conversation_id}'."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)