"""
Pharmacovigilance tool using openFDA's free, public REST API (FAERS data).
"""

import requests
from collections import Counter

OPENFDA_URL = "https://api.fda.gov/drug/event.json"


def search_adverse_events(drug_name: str, max_results: int = 100) -> dict:
    params = {
        "search": f'patient.drug.medicinalproduct:"{drug_name}"',
        "limit": max_results,
    }

    response = requests.get(OPENFDA_URL, params=params, timeout=10)
    response.raise_for_status()

    data = response.json()
    total_found = data.get("meta", {}).get("results", {}).get("total", 0)
    reports = data.get("results", [])

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
        print("This can happen if the drug name doesn't match any records - try a different name.")
