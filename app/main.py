"""
FastAPI backend.

Purpose: exposes the agent as a real web API with a /chat endpoint, so
a website, app, or any other program can send it a question over the
internet and get an answer back — instead of only running via a Python
script in the terminal.
"""

import os
from datetime import date

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.agent.orchestrator import run_agent

load_dotenv()

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

# --- Auth: verifying who's actually asking -----------------------------
# The frontend sends each student's Supabase login token in the
# Authorization header. Rather than manually verifying that token's
# cryptographic signature ourselves (Supabase's newer projects sign
# tokens with public-key cryptography, which is more complex to handle
# correctly), we ask Supabase directly: "is this token valid, and who
# does it belong to?" One small extra network call per request, but
# much simpler and more robust against Supabase changing internals.
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")


def get_current_user(authorization: str | None) -> dict | None:
    """
    Given the raw Authorization header value (e.g. "Bearer eyJ..."),
    asks Supabase to verify it and returns the user's info if valid.

    Returns None if there's no header, the header is malformed, the
    token is invalid/expired, or Supabase's auth setup isn't configured
    — in all those cases, the request is simply treated as anonymous
    rather than rejected outright. This keeps the chat endpoint working
    even for a request with no login (useful during rollout), while
    still identifying logged-in students when possible.
    """
    if not authorization or not authorization.startswith("Bearer "):
        return None

    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        # Auth isn't configured on this deployment yet — treat everyone
        # as anonymous rather than erroring out.
        return None

    token = authorization[len("Bearer "):]

    try:
        response = requests.get(
            f"{SUPABASE_URL}/auth/v1/user",
            headers={
                "Authorization": f"Bearer {token}",
                "apikey": SUPABASE_ANON_KEY,
            },
            timeout=5,
        )
    except requests.exceptions.RequestException as e:
        print(f"  [Auth check failed (network error), treating as anonymous: {e}]")
        return None

    if response.status_code != 200:
        # Invalid or expired token — treat as anonymous rather than
        # blocking the request outright.
        return None

    user = response.json()
    return {"id": user.get("id"), "email": user.get("email")}


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
def chat(request: ChatRequest, authorization: str | None = Header(None)):
    check_and_record_usage(request.session_id)

    user = get_current_user(authorization)
    if user:
        print(f"  [Request from logged-in user: {user['email']}]")
    else:
        print("  [Request from anonymous/unauthenticated session]")

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

    # NOTE: this is where the next step (saving chat history to Supabase,
    # tied to user["id"]) will hook in, once the database table exists.

    return ChatResponse(answer=answer, session_id=request.session_id)