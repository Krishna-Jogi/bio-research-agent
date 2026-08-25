"""
FastAPI backend.

Purpose: exposes the agent as a real web API with a /chat endpoint, so
a website, app, or any other program can send it a question over the
internet and get an answer back — instead of only running via a Python
script in the terminal.

"""

from datetime import date
from fastapi import FastAPI, HTTPException
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

# --- Rate limiting ---------------------------------------------------
# Purpose: without this, one student (or a bug, or someone finding the
# URL) could send unlimited requests, exhausting the free OpenRouter
# quota (200/day total) for every other student. This is the single
# most important protection for a free, shared tool.
#
# Tracked as: {session_id: {"date": "2026-08-17", "count": 5}}
# The count resets automatically whenever the stored date is not today.
DAILY_MESSAGE_LIMIT = 20

usage_tracker: dict[str, dict] = {}


def check_and_record_usage(session_id: str):
    """
    Raises an HTTPException (429 Too Many Requests) if this session has
    hit today's message limit. Otherwise, records one more message used.
    """
    today = str(date.today())
    record = usage_tracker.get(session_id)

    if record is None or record["date"] != today:
        # First message today for this session — start a fresh count.
        usage_tracker[session_id] = {"date": today, "count": 1}
        return

    if record["count"] >= DAILY_MESSAGE_LIMIT:
        raise HTTPException(
            status_code=429,
            detail=(
                f"Daily limit of {DAILY_MESSAGE_LIMIT} messages reached for this session. "
                "This resets at midnight. Thanks for using Field Notes — see you tomorrow!"
            ),
        )

    record["count"] += 1


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
    check_and_record_usage(request.session_id)

    history = sessions.get(request.session_id, [])

    try:
        answer, updated_history = run_agent(request.message, conversation_history=history)
    except Exception as e:
        # Catch-all safety net: any unexpected error (network hiccup, a
        # malformed response from the free model, etc.) becomes a clear
        # message to the student instead of a raw 500 crash.
        print(f"  [ERROR in run_agent: {e}]")
        return ChatResponse(
            answer="Something went wrong answering that question. Please try again.",
            session_id=request.session_id,
        )

    if not answer:
        answer = "The agent didn't return a usable answer. Please try asking again."

    sessions[request.session_id] = updated_history

    return ChatResponse(answer=answer, session_id=request.session_id)