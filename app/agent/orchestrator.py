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


"""

import os
import json
from dotenv import load_dotenv
from openai import OpenAI

# These imports work because this script's own folder (app/agent) is
# automatically added to Python's search path when you run it directly,
# and "tools" is a subfolder right there with an __init__.py in it.
from tools.research_search import search_papers
from tools.pharmacovigilance_search import search_adverse_events
from tools.math_engine import solve_expression, solve_equation, first_order_half_life, calculate_dosage

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise ValueError("OPENROUTER_API_KEY not found. Check your .env file.")

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

MODEL = "openai/gpt-oss-20b:free"  # free tier, confirmed working in test_connection.py


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

def run_agent(user_question: str, max_turns: int = 5) -> str:
    messages = [{"role": "user", "content": user_question}]

    for turn in range(max_turns):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            max_tokens=800,
        )

        message = response.choices[0].message

        # If the model didn't ask for a tool, it's giving its final answer.
        if not message.tool_calls:
            return message.content

        # The model wants to call one or more tools. Save its request in
        # the conversation, then run each tool and add the results back in,
        # so the next call has everything it needs to respond properly.
        messages.append(message.model_dump())

        for tool_call in message.tool_calls:
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            print(f"  [Agent is calling tool: {name}  with args: {args}]")

            result = call_tool(name, args)

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result, default=str),
            })

    return "The agent couldn't reach a final answer within the turn limit."


if __name__ == "__main__":
    # A few test questions covering each tool, plus one that needs no tool.
    test_questions = [
        "What is the half-life of a drug with a first-order rate constant of 0.05 per hour?",
        "What side effects have been reported for metformin?",
        "Find me recent research papers on malaria drug resistance.",
    ]

    for q in test_questions:
        print("=" * 70)
        print(f"Question: {q}\n")
        answer = run_agent(q)
        print(f"\nAnswer: {answer}\n")