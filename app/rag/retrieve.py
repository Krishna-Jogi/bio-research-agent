"""
RAG retrieval.

Purpose: given a student's question, find the most relevant chunks
from the government documents we ingested — this becomes a fourth
tool for the agent, alongside research search, pharmacovigilance
search, and the math engine.

"""

import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_PATH = "app/rag/chroma_db"
COLLECTION_NAME = "govt_sources"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Loaded once, reused for every call — loading the model is the slow part,
# so we don't want to repeat it per question.
_embedding_model = None
_collection = None


def _get_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


def _get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        _collection = client.get_or_create_collection(COLLECTION_NAME)
    return _collection


def search_government_sources(query: str, domain: str = None, n_results: int = 4) -> dict:
    """
    Searches ingested government documents for chunks relevant to the query.

    Args:
        query: the student's question or topic
        domain: optional filter, e.g. "pharmacovigilance" — restricts
                results to just that domain if provided
        n_results: how many chunks to return

    Returns:
        A dict with the results — each result includes the chunk text,
        its source title, source URL, and domain, so the agent can cite
        exactly where the information came from.
    """
    model = _get_model()
    collection = _get_collection()

    query_embedding = model.encode([query]).tolist()

    where_filter = {"domain": domain} if domain else None

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results,
        where=where_filter,
    )

    if not results["documents"] or not results["documents"][0]:
        return {"found": False, "results": []}

    formatted = []
    for text, metadata in zip(results["documents"][0], results["metadatas"][0]):
        formatted.append({
            "text": text,
            "title": metadata.get("title"),
            "source_url": metadata.get("source_url"),
            "domain": metadata.get("domain"),
        })

    return {"found": True, "results": formatted}


if __name__ == "__main__":
    test_queries = [
        "How should a suspected adverse drug reaction be reported?",
        "What is an Individual Case Safety Report?",
    ]

    for q in test_queries:
        print("=" * 70)
        print(f"Query: {q}\n")

        result = search_government_sources(q, domain="pharmacovigilance", n_results=2)

        if not result["found"]:
            print("No results found — has ingest.py been run yet?")
            continue

        for i, r in enumerate(result["results"], 1):
            print(f"--- Result {i} ---")
            print(f"Source: {r['title']}")
            print(f"URL: {r['source_url']}")
            print(f"Excerpt: {r['text'][:300]}...\n")