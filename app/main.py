"""
FastAPI backend.

Purpose: exposes the agent as a real web API with a /chat endpoint, so
a website, app, or any other program can send it a question over the
internet and get an answer back — instead of only running via a Python
script in the terminal.

Run this with: uvicorn app.main:app --reload
(run from the project root — the folder containing "app")

Then test it by opening http://127.0.0.1:8000/docs in a browser —
FastAPI auto-generates an interactive page where you can try the
/chat endpoint directly, no separate frontend needed yet.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.agent.orchestrator import run_agent

app = FastAPI(title="Bio Research Agent")

# CORS lets a frontend running on a different address (e.g. a local
# React/HTML page, or later your deployed site) call this API from the
# browser. "*" is fine for development; you'd narrow this to your real
# frontend's address before a public launch.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Very simple in-memory session store: {session_id: [conversation history]}
# This is fine for development and a small soft launch. It resets every
# time the server restarts, and doesn't scale across multiple server
# instances — a real database would replace this later (Phase: "add
# persistence"), but this is enough to prove multi-turn conversation works.
sessions: dict[str, list] = {}


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"


class ChatResponse(BaseModel):
    answer: str
    session_id: str


@app.get("/")
def root():
    """Simple health check — confirms the server is running at all."""
    return {"status": "Bio Research Agent is running"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    history = sessions.get(request.session_id, [])

    answer, updated_history = run_agent(request.message, conversation_history=history)

    sessions[request.session_id] = updated_history

    return ChatResponse(answer=answer, session_id=request.session_id)