"""
Research paper search tool using Europe PMC's free, public REST API.

Purpose: given a topic, return real research papers (title, authors,
journal, year, abstract snippet, and a link) so students can find
relevant literature for their research.

This is a STANDALONE function right now — no LLM involved. We're
testing that the data-fetching logic works correctly on its own,
before wiring it into the agent's tool-calling loop later.

Europe PMC is free and requires no API key, so this works even
before OpenRouter credits are added.
"""

import requests

EUROPE_PMC_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"


def search_papers(query: str, max_results: int = 5) -> list[dict]:
    """
    Search Europe PMC for papers matching the query.

    Args:
        query: topic or keywords to search for, e.g. "malaria drug resistance"
        max_results: how many papers to return (default 5)

    Returns:
        A list of dicts, each with title, authors, journal, year,
        abstract, and a link to the paper.
    """
    params = {
        "query": query,
        "format": "json",
        "pageSize": max_results,
        "resultType": "core",  # gives us abstract text, not just metadata
    }

    # A network hiccup here (timeout, DNS issue, Europe PMC being briefly
    # down) shouldn't crash the whole question — that wastes the person's
    # turn and shows a generic "something went wrong" error even when
    # other tools in the same turn found something useful. Instead, fail
    # gracefully and let the agent explain research search wasn't
    # available, same as it already does for "no results found".
    try:
        response = requests.get(EUROPE_PMC_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        print(f"  [Europe PMC search failed: {e}]")
        return []

    results = data.get("resultList", {}).get("result", [])

    papers = []
    for r in results:
        abstract = r.get("abstractText", "No abstract available")
        # Truncate long abstracts before sending to the LLM — full-length
        # abstracts across 5 papers adds up to a lot of tokens, which can
        # overwhelm smaller/free models and increase cost with paid ones.
        # A concise summary is enough for the LLM to synthesize an answer;
        # the full paper is always available via the link.
        if len(abstract) > 500:
            abstract = abstract[:500].rsplit(" ", 1)[0] + "..."

        papers.append({
            "title": r.get("title", "No title available"),
            "authors": r.get("authorString", "Authors not listed"),
            "journal": r.get("journalInfo", {}).get("journal", {}).get("title", "Journal not listed"),
            "year": r.get("pubYear", "Year not listed"),
            "abstract": abstract,
            "link": f"https://europepmc.org/article/{r.get('source', 'MED')}/{r.get('id', '')}",
        })

    return papers


if __name__ == "__main__":
    # Quick manual test — run this file directly to confirm it works
    test_query = "pharmacovigilance adverse drug reactions"
    print(f"Searching for: {test_query}\n")

    papers = search_papers(test_query, max_results=3)

    if not papers:
        print("No papers found — check your internet connection or query.")
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