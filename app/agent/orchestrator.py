"""
Agent orchestration loop.

Purpose: this is the actual "agent" part of the project. Up to now,
we've built three standalone tools that work on their own. This file
gives the LLM the ABILITY to call those tools, and implements the loop
that:
  1. sends the student's question + tool descriptions to the LLM
  2. checks whether the LLM wants to call a tool
  3. if yes: runs the real Python function, sends the result back
  4. repeats until the LLM has enough information to give a final answer

This is called "tool calling" or "function calling." The LLM never
executes code itself — it only ever REQUESTS a tool by name with
arguments; our code is what actually runs it.

Run this with: python app/agent/orchestrator.py
(run from the project root, so the .env file is found correctly)
"""

import os
import json
import time
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError

# These are "absolute imports" from the app package, which works both
# when this file is run as part of the package (e.g. imported by main.py)
# and when run directly with `python -m app.agent.orchestrator` from the
# project root.
from app.agent.tools.research_search import search_papers
from app.agent.tools.pharmacovigilance_search import search_adverse_events
from app.agent.tools.math_engine import solve_expression, solve_equation, first_order_half_life, calculate_dosage
from app.rag.retrieve import search_government_sources

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise ValueError("OPENROUTER_API_KEY not found. Check your .env file.")

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

MODEL = "openai/gpt-oss-20b:free"  # free tier, confirmed working in test_connection.py

# This instruction is sent as a "system" message — a special message role
# that sets ground rules for how the model should behave, separate from
# the actual conversation. It isn't shown to the student; it just steers
# the model's behavior on every turn.
SYSTEM_PROMPT = """You are an academic research assistant for biology, \
bioscience, and pharmacology students. You have access to tools for \
searching research papers, checking drug safety/adverse-event data, and \
performing calculations.

Rules you must always follow:
1. If you use the research paper search tool's result in your answer, you \
MUST include the exact source link(s) it returned, so the student can \
independently verify the paper themselves. Never omit a paper's link.
2. If you use the drug safety/pharmacovigilance tool's result, mention \
that the data comes from the FDA's FAERS adverse event database, but do \
NOT include a raw link — just cite it by name.
3. If you use the government source search tool's result, cite the \
document title AND include the source URL it returned. Trust the tool's \
top results — if they mention the topic you asked about, even briefly \
or as part of a broader passage, that counts as relevant and you should \
use it. Only say the ingested documents don't cover the topic if NONE \
of the returned excerpts mention it at all. Do NOT call the same tool \
again with a reworded query more than once per question.
4. If you answer a question WITHOUT using any tool (i.e. from your own \
general knowledge), you MUST clearly say so at the end of your answer — \
for example: "Note: this answer was not verified against a live source. \
For research papers or drug safety data, ask me to look it up." Do not \
present unverified answers the same way as sourced ones.
5. Never fabricate a citation, link, or statistic. If a tool returns no \
results, say so plainly rather than filling in a plausible-sounding answer.
"""


# ---------------------------------------------------------------------------
# STEP 1: Describe each tool to the LLM.
# This is a JSON schema — name, description, and what parameters it takes.
# The LLM reads these descriptions (not our actual code) to decide which
# tool fits a given question. Clear descriptions here directly affect how
# well the agent routes questions correctly.
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_research_papers",
            "description": (
                "Search for real, published research papers on a biology, "
                "bioscience, pharmacology, or clinical research topic. Use "
                "this when the student asks for research papers, studies, "
                "literature, or scientific evidence on a topic."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The topic or keywords to search for, e.g. 'malaria drug resistance'",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "How many papers to return (default 5)",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_drug_safety",
            "description": (
                "Search FDA's adverse event database for real-world reported "
                "side effects of a specific drug. Use this for any "
                "pharmacovigilance question — drug safety, side effects, "
                "adverse reactions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "drug_name": {
                        "type": "string",
                        "description": "The name of the drug, e.g. 'metformin'",
                    },
                },
                "required": ["drug_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "do_math",
            "description": (
                "Perform a precise mathematical calculation. Use this instead "
                "of calculating by hand for any equation, simplification, "
                "drug half-life, or dosage calculation — this guarantees "
                "correct results rather than an estimated answer."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["simplify", "solve_equation", "half_life", "dosage"],
                        "description": "Which kind of calculation to perform",
                    },
                    "expression": {
                        "type": "string",
                        "description": "For 'simplify': the math expression, e.g. '2*x + 3*x'",
                    },
                    "equation": {
                        "type": "string",
                        "description": "For 'solve_equation': the equation set to 0, e.g. '2*x + 4 - 10'",
                    },
                    "variable": {
                        "type": "string",
                        "description": "For 'solve_equation': the variable to solve for (default 'x')",
                    },
                    "k": {
                        "type": "number",
                        "description": "For 'half_life': the first-order rate constant",
                    },
                    "mg_per_kg": {
                        "type": "number",
                        "description": "For 'dosage': desired dose in mg per kg of body weight",
                    },
                    "weight_kg": {
                        "type": "number",
                        "description": "For 'dosage': patient's weight in kg",
                    },
                },
                "required": ["operation"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_government_documents",
            "description": (
                "Search official government guidance documents for topics "
                "that don't have a live API — pharmacovigilance procedures, "
                "PTC (plant tissue culture), nutraceuticals, water quality, "
                "environment, chemical engineering standards, and clinical "
                "research guidelines. Use this for procedural or regulatory "
                "questions, e.g. 'how do I report an adverse drug reaction' "
                "or 'what is an Individual Case Safety Report'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The question or topic to search for",
                    },
                    "domain": {
                        "type": "string",
                        "enum": ["pharmacovigilance", "nutraceuticals", "water"],
                        "description": (
                            "Pick the domain that matches the question's TOPIC, not habit — "
                            "check this list carefully every time: "
                            "'pharmacovigilance' = adverse drug reactions (ADR), ICSRs, "
                            "safety reporting, PSUR. "
                            "'nutraceuticals' = health supplements, functional foods, FSSAI "
                            "food regulations, nutrient/vitamin/mineral limits. "
                            "'water' = water quality, water for injection (WFI), purified "
                            "water, water systems, water testing/GMP. "
                            "If genuinely uncertain which domain fits, OMIT this parameter "
                            "entirely to search across all domains — do not guess."
                        ),
                    },
                },
                "required": ["query"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# STEP 2: The dispatcher — takes a tool name + arguments the LLM requested,
# and actually calls the matching real Python function.
# ---------------------------------------------------------------------------

def call_tool(name: str, args: dict):
    if name == "search_research_papers":
        return search_papers(args["query"], args.get("max_results", 5))

    if name == "search_drug_safety":
        return search_adverse_events(args["drug_name"])

    if name == "search_government_documents":
        return search_government_sources(args["query"], args.get("domain"))

    if name == "do_math":
        op = args["operation"]
        if op == "simplify":
            return {"result": solve_expression(args["expression"])}
        if op == "solve_equation":
            return {"result": solve_equation(args["equation"], args.get("variable", "x"))}
        if op == "half_life":
            return {"result": first_order_half_life(args["k"])}
        if op == "dosage":
            return {"result": calculate_dosage(args["mg_per_kg"], args["weight_kg"])}
        return {"error": f"Unknown math operation: {op}"}

    return {"error": f"Unknown tool: {name}"}


# ---------------------------------------------------------------------------
# STEP 3: The orchestration loop itself.
# ---------------------------------------------------------------------------

def is_garbage_output(text: str) -> bool:
    """
    Detects degenerate output — a free-tier model occasionally gets stuck
    repeating the same character or token over and over instead of giving
    a real answer (e.g. hundreds of '!' characters). A real answer, even
    a short one, uses a healthy variety of characters; a stuck repetition
    loop does not.
    """
    if not text:
        return False
    stripped = text.strip()
    if len(stripped) < 40:
        return False
    unique_chars = set(stripped.replace(" ", "").replace("\n", ""))
    return len(unique_chars) <= 4


def run_agent(user_question: str, conversation_history: list = None, max_turns: int = 5):
    """
    Runs the agent loop for one question.

    Args:
        user_question: the student's question
        conversation_history: optional list of prior messages, so the
            backend can maintain a multi-turn conversation per student
            session instead of treating every message as brand new
        max_turns: safety limit on tool-call rounds

    Returns:
        (answer_text, updated_history) — the updated_history should be
        passed back in on the NEXT call for this same conversation, so
        the agent remembers what was already discussed.
    """
    messages = list(conversation_history) if conversation_history else []

    # Only add the system prompt once, at the very start of a conversation
    # — not on every turn, since it should already be in the history for
    # any conversation that's already underway.
    if not messages:
        messages.append({"role": "system", "content": SYSTEM_PROMPT})

    messages.append({"role": "user", "content": user_question})

    # Track whether any tool was actually called during THIS question, so
    # we can guarantee the "unverified answer" disclosure ourselves in code
    # — rather than relying on the model to remember to add it every time,
    # which a free-tier model won't always do reliably.
    tool_used_this_turn = False

    for turn in range(max_turns):
        # Free-tier models share a limited request pool across everyone
        # using OpenRouter at once, so occasional rate-limit errors are
        # expected, not a sign anything is broken. We retry a few times
        # with a short wait, rather than failing the whole conversation
        # over a temporary, shared-resource hiccup.
        response = None
        last_error = None
        for attempt in range(4):
            try:
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    tools=TOOLS,
                    max_tokens=800,
                )
                break
            except RateLimitError as e:
                last_error = e
                wait_seconds = 25
                print(f"  [Rate-limited, waiting {wait_seconds}s before retry {attempt + 1}/4...]")
                time.sleep(wait_seconds)

        if response is None:
            return f"The free model is currently rate-limited and retries were exhausted. Try again in a minute. ({last_error})", messages

        if not response.choices:
            # Occasionally the free tier returns a technically-successful
            # response with no usable content at all (malformed under load).
            return "The free model returned an unusable response. Please try asking again.", messages

        message = response.choices[0].message

        # Free-tier models occasionally return a completely empty response,
        # or get stuck in a degenerate repetition loop (e.g. hundreds of
        # the same character), under load. Rather than give up immediately,
        # retry the same turn a couple of times first.
        needs_retry = not message.tool_calls and (not message.content or is_garbage_output(message.content))
        if needs_retry:
            print("  [Empty or garbage response from model, retrying...]")
            retried = False
            for _ in range(2):
                time.sleep(3)
                retry_response = client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    tools=TOOLS,
                    max_tokens=800,
                )
                message = retry_response.choices[0].message
                if message.tool_calls or (message.content and not is_garbage_output(message.content)):
                    retried = True
                    break
            if not retried:
                return (
                    "The agent didn't return a usable answer after retrying — this can happen "
                    "occasionally with the free model under high load. Please try asking again.",
                    messages,
                )

        messages.append(message.model_dump())

        # If the model didn't ask for a tool, it's giving its final answer.
        if not message.tool_calls:
            if not message.content:
                return (
                    "The agent didn't return a usable answer this time — please try asking again.",
                    messages,
                )

            answer = message.content

            # Guarantee the unverified-answer disclosure in CODE, not just
            # by asking the model nicely — a free-tier model won't always
            # remember to add it, and this matters too much for trust to
            # leave to chance. Only add it if no tool was used this turn
            # and the model hasn't already included an equivalent note.
            if not tool_used_this_turn and "not verified" not in answer.lower() and "not been verified" not in answer.lower():
                answer += (
                    "\n\n---\n*Note: this answer was not verified against a live source "
                    "or the ingested government documents — it comes from the model's "
                    "general knowledge. Ask me to look up research papers, drug safety "
                    "data, or government guidance for a sourced answer.*"
                )

            return answer, messages

        for tool_call in message.tool_calls:
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            print(f"  [Agent is calling tool: {name}  with args: {args}]")

            tool_used_this_turn = True
            result = call_tool(name, args)

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result, default=str),
            })

    return "The agent couldn't reach a final answer within the turn limit.", messages


if __name__ == "__main__":
    # A few test questions covering each tool, plus one that needs no tool.
    test_questions = [
        "What is the half-life of a drug with a first-order rate constant of 0.05 per hour?",
        "What side effects have been reported for metformin?",
        "Find me recent research papers on malaria drug resistance.",
        "How should a suspected adverse drug reaction be reported in India?",
    ]

    for q in test_questions:
        print("=" * 70)
        print(f"Question: {q}\n")
        answer, _ = run_agent(q)
        print(f"\nAnswer: {answer}\n")