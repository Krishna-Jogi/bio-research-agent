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

MODEL = "openrouter/free"  # primary — an auto-router that picks from
# whichever free models are currently live and support tool calling.
# Switched to this as primary after TWO different named free models
# (gpt-oss-20b:free, then llama-3.3-70b-instruct:free) were silently
# retired by OpenRouter mid-project. A specific model name is not a
# stable thing to depend on for a free-tier project — this auto-router
# is maintained by OpenRouter itself, so it adapts as models are
# added/removed without needing a code change here.
FALLBACK_MODEL = "meta-llama/llama-3.3-70b-instruct:free"  # secondary —
# tried only if the primary somehow fails outright. Nice to have for
# slightly more predictable behavior when it happens to work, but the
# project no longer depends on it staying available.

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
                        "enum": ["pharmacovigilance", "nutraceuticals", "water", "environment", "chemical_engineering", "ptc"],
                        "description": (
                            "Pick the domain that matches the question's TOPIC, not habit — "
                            "check this list carefully every time: "
                            "'pharmacovigilance' = adverse drug reactions (ADR), ICSRs, "
                            "safety reporting, PSUR. "
                            "'nutraceuticals' = health supplements, functional foods, FSSAI "
                            "food regulations, nutrient/vitamin/mineral limits. "
                            "'water' = water quality, water for injection (WFI), purified "
                            "water, water systems, water testing/GMP. "
                            "'environment' = pharmaceutical effluent/emission standards, "
                            "waste disposal, pollution limits, hazardous waste. "
                            "'chemical_engineering' = pharmaceutical process/manufacturing "
                            "development, Quality by Design (QbD), critical quality attributes, "
                            "design space, process parameters, ICH Q8. "
                            "'ptc' = Plant Tissue Culture — biosafety for transgenic plants, "
                            "plant/cell culture containment, DBT biosafety levels for plant work. "
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


# Phrases that reliably show up when a reasoning model's internal
# chain-of-thought leaks into what should be the final answer, instead of
# staying hidden. This is a narrow, defense-in-depth check — the real fix
# is asking OpenRouter to exclude reasoning tokens — but this catches it
# if that doesn't fully work for a given model.
_REASONING_LEAK_MARKERS = (
    "the user asks",
    "we need to provide",
    "thus we need to",
    "let's provide",
    "potential answer:",
)


def looks_like_leaked_reasoning(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _REASONING_LEAK_MARKERS)


def get_agent_response(messages: list, models=(MODEL, FALLBACK_MODEL), attempts_per_model: int = 2):
    """
    Requests a completion, trying each model in `models` in order. For
    each model, retries a couple of times if it hits a rate limit, an
    empty response, or garbage output — all known free-tier quirks.
    Only moves to the next model in the list once the current one has
    genuinely failed multiple times, not on the first hiccup.

    Returns (message, model_used) on success, or (None, None) if every
    model in the list failed.
    """
    for model_name in models:
        for attempt in range(attempts_per_model):
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    tools=TOOLS,
                    max_tokens=1200,
                    # gpt-oss-20b is a reasoning model — without this, it can
                    # spend its whole token budget "thinking out loud" (e.g.
                    # "The user asks... Thus we need to...") as if it were
                    # the actual answer, sometimes leaking that reasoning
                    # into the visible response and running out of room
                    # before writing the real answer. This tells OpenRouter
                    # to keep reasoning internal and only return the final
                    # answer text.
                    extra_body={"reasoning": {"exclude": True}},
                )
            except RateLimitError:
                print(f"  [{model_name} rate-limited, waiting 20s (attempt {attempt + 1}/{attempts_per_model})...]")
                time.sleep(20)
                continue
            except Exception as e:
                # A 404 here means the model itself is gone/renamed (as
                # happened when OpenRouter retired gpt-oss-20b:free) — no
                # amount of retrying will fix that, so don't waste time
                # sleeping and trying again; move straight to the next
                # model in the list.
                error_text = str(e)
                if "404" in error_text or "no longer" in error_text.lower() or "unavailable" in error_text.lower():
                    print(f"  [{model_name} appears to be permanently unavailable: {e}]")
                    break
                print(f"  [{model_name} request failed: {e}]")
                time.sleep(3)
                continue

            if not response.choices:
                print(f"  [{model_name} returned no choices, retrying...]")
                continue

            message = response.choices[0].message

            # Defense in depth: even with reasoning excluded, occasionally
            # a leaked chain-of-thought preamble can still slip through
            # (e.g. "The user asks...", "Thus we need to..."). Treat this
            # the same as garbage output and retry, rather than showing a
            # confusing internal monologue to the person asking a question.
            if message.content and looks_like_leaked_reasoning(message.content):
                print(f"  [{model_name} leaked internal reasoning instead of an answer, retrying...]")
                time.sleep(3)
                continue

            if message.tool_calls or (message.content and not is_garbage_output(message.content)):
                return message, model_name

            print(f"  [{model_name} returned empty/garbage output, retrying...]")
            time.sleep(3)

        print(f"  [{model_name} failed after {attempts_per_model} attempts — trying next model if available]")

    return None, None


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
    # we can guarantee the "unverified answer" disclosure ourselves in code.
    tool_used_this_turn = False

    # Track the REAL sources returned by tools during this turn — not what
    # the model claims it cited, but the actual title/URL pairs the tools
    # gave back. We use this to append a guaranteed-correct source list in
    # code, rather than trusting the model to accurately transcribe a URL
    # into its prose (which, as observed, it doesn't always do correctly).
    verified_sources = []  # list of (title, url) tuples, deduplicated

    def track_sources(tool_name: str, result):
        if tool_name == "search_research_papers" and isinstance(result, list):
            for paper in result:
                title = paper.get("title")
                link = paper.get("link")
                if title and link:
                    entry = (title, link)
                    if entry not in verified_sources:
                        verified_sources.append(entry)

        if tool_name == "search_government_documents" and isinstance(result, dict):
            for r in result.get("results", []):
                title = r.get("title")
                url = r.get("source_url")
                if title and url:
                    entry = (title, url)
                    if entry not in verified_sources:
                        verified_sources.append(entry)

    for turn in range(max_turns):
        turn_start = time.time()
        message, model_used = get_agent_response(messages)
        print(f"  [Turn {turn + 1}: model call took {time.time() - turn_start:.1f}s, used {model_used}]")

        if message is None:
            return (
                "The agent couldn't get a usable response from either the primary or backup "
                "free model right now — this happens occasionally under high load. "
                "Please try again shortly.",
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

            # Append a code-guaranteed, verified source list — this is
            # ALWAYS accurate, regardless of whether the model correctly
            # copied links into its own prose. If it duplicates a link
            # the model already cited correctly, that's harmless; if the
            # model cited something wrong or generic, this is the
            # trustworthy version.
            if verified_sources:
                answer += "\n\n**Verified sources consulted:**\n"
                for title, url in verified_sources:
                    answer += f"- {title}: {url}\n"

            return answer, messages

        for tool_call in message.tool_calls:
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            print(f"  [Agent is calling tool: {name}  with args: {args}]")

            tool_start = time.time()
            tool_used_this_turn = True
            result = call_tool(name, args)
            print(f"  [Tool {name} took {time.time() - tool_start:.1f}s]")
            track_sources(name, result)

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