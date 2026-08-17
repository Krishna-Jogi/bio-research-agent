"""
Pharmacovigilance tool using openFDA's free, public REST API (FAERS data).

Purpose: given a drug name, return real reported adverse events (side
effects) from the FDA's Adverse Event Reporting System, so students get
real-world drug safety data — directly covering the compulsory
pharmacovigilance requirement.

This is a STANDALONE function right now, same pattern as
research_search.py — no LLM involved yet. We test the data-fetching
logic on its own before wiring it into the agent's tool-calling loop.

openFDA is free and requires no API key for reasonable use, so this
works even before OpenRouter credits are added.
"""

import requests
from collections import Counter

OPENFDA_URL = "https://api.fda.gov/drug/event.json"


def search_adverse_events(drug_name: str, max_results: int = 100) -> dict:
    """
    Search openFDA for reported adverse events for a given drug.

    Args:
        drug_name: the drug's name, e.g. "metformin", "ibuprofen"
        max_results: how many recent reports to pull and summarize from
                     (default 100 — we summarize across these, rather
                     than returning 100 raw reports)

    Returns:
        A dict with the drug name, total matching reports found, and
        the most commonly reported adverse reactions with their counts.
    """
    params = {
        "search": f'patient.drug.medicinalproduct:"{drug_name}"',
        "limit": max_results,
    }

    response = requests.get(OPENFDA_URL, params=params, timeout=10)
    response.raise_for_status()

    data = response.json()
    total_found = data.get("meta", {}).get("results", {}).get("total", 0)
    reports = data.get("results", [])

    # Each report can list multiple reactions. We count how often each
    # reaction appears across all pulled reports, to surface the most
    # commonly reported side effects rather than dumping raw data.
    reaction_counter = Counter()
    for report in reports:
        reactions = report.get("patient", {}).get("reaction", [])
        for reaction in reactions:
            term = reaction.get("reactionmeddrapt")
            if term:
                reaction_counter[term.lower()] += 1

    top_reactions = reaction_counter.most_common(10)

    return {
        "drug_name": drug_name,
        "total_reports_found": total_found,
        "reports_analyzed": len(reports),
        "top_reported_reactions": top_reactions,
    }


if __name__ == "__main__":
    # Quick manual test — run this file directly to confirm it works
    test_drug = "metformin"
    print(f"Searching adverse event reports for: {test_drug}\n")

    try:
        result = search_adverse_events(test_drug)

        print(f"Drug: {result['drug_name']}")
        print(f"Total reports in FAERS database: {result['total_reports_found']}")
        print(f"Reports analyzed for this summary: {result['reports_analyzed']}\n")
        print("Top 10 most commonly reported reactions:")
        for reaction, count in result["top_reported_reactions"]:
            print(f"  - {reaction}: {count} reports")

        print("\nIf you see real reactions and counts above, this tool is working correctly.")
    except requests.exceptions.HTTPError as e:
        print(f"Request failed: {e}")
        print("This can happen if the drug name doesn't match any records — try a different name.")