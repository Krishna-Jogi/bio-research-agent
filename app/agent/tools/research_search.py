"""
Research paper search tool using Europe PMC's free, public REST API.
"""

import requests

EUROPE_PMC_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"


def search_papers(query: str, max_results: int = 5) -> list[dict]:
    params = {
        "query": query,
        "format": "json",
        "pageSize": max_results,
        "resultType": "core",
    }

    response = requests.get(EUROPE_PMC_URL, params=params, timeout=10)
    response.raise_for_status()

    data = response.json()
    results = data.get("resultList", {}).get("result", [])

    papers = []
    for r in results:
        papers.append({
            "title": r.get("title", "No title available"),
            "authors": r.get("authorString", "Authors not listed"),
            "journal": r.get("journalTitle", "Journal not listed"),
            "year": r.get("pubYear", "Year not listed"),
            "abstract": r.get("abstractText", "No abstract available"),
            "link": f"https://europepmc.org/article/{r.get('source', 'MED')}/{r.get('id', '')}",
        })

    return papers


if __name__ == "__main__":
    test_query = "pharmacovigilance adverse drug reactions"
    print(f"Searching for: {test_query}\n")

    papers = search_papers(test_query, max_results=3)

    if not papers:
        print("No papers found - check your internet connection or query.")
    else:
        for i, paper in enumerate(papers, 1):
            print(f"--- Paper {i} ---")
            print(f"Title: {paper['title']}")
            print(f"Authors: {paper['authors']}")
            print(f"Journal: {paper['journal']} ({paper['year']})")
            print(f"Abstract: {paper['abstract'][:200]}...")
            print(f"Link: {paper['link']}")
            print()

        print("If you see real papers above, this tool is working correctly.")
