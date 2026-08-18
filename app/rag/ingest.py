"""
RAG ingestion pipeline.

Purpose: for government sources that have NO live API (unlike PubMed or
openFDA), this is how we make their content searchable by the agent.

The process, step by step:
  1. Download a PDF from a government source URL
  2. Extract its raw text
  3. Split ("chunk") that text into small overlapping pieces — small
     enough that each chunk is specific, with overlap so we don't cut
     a sentence's meaning in half at a chunk boundary
  4. Convert each chunk into an "embedding" — a list of numbers that
     represents the chunk's MEANING, using a free, local model
     (nothing sent to any API for this step — it runs on your machine)
  5. Store the chunk text + its embedding + where it came from (source
     URL, domain, title) in Chroma, a local vector database

Later, retrieve.py uses the same embedding model to turn a student's
QUESTION into numbers the same way, then asks Chroma "which stored
chunks have embeddings closest to this question's embedding?" — that's
how it finds relevant content by meaning, not just keyword matching.

Run this with: python -m app.rag.ingest
(run from the project root)
"""

import hashlib
import io

import chromadb
import requests
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

CHROMA_PATH = "app/rag/chroma_db"
COLLECTION_NAME = "govt_sources"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # small, fast, free, runs locally

CHUNK_SIZE_WORDS = 350
CHUNK_OVERLAP_WORDS = 50


print("Loading embedding model (first run downloads it, ~90MB, one-time)...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(COLLECTION_NAME)


def download_pdf_text(url: str) -> str:
    """Downloads a PDF from a URL and extracts its raw text."""
    response = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()

    reader = PdfReader(io.BytesIO(response.content))
    text_parts = []
    for page in reader.pages:
        text_parts.append(page.extract_text() or "")

    return "\n".join(text_parts)


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE_WORDS, overlap: int = CHUNK_OVERLAP_WORDS) -> list[str]:
    """
    Splits text into overlapping word chunks.
    Overlap means the end of one chunk repeats at the start of the next,
    so a sentence that would otherwise get cut in half at a boundary
    still appears whole in at least one chunk.
    """
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


def make_chunk_id(url: str, chunk_index: int) -> str:
    """
    A stable, unique ID per chunk, based on its source URL and position.
    Using a stable ID (rather than a random one) means re-running
    ingestion on the same document updates its existing entries instead
    of creating duplicates.
    """
    raw = f"{url}::{chunk_index}"
    return hashlib.sha256(raw.encode()).hexdigest()


def ingest_document(url: str, domain: str, title: str):
    """
    Downloads, chunks, embeds, and stores one document.

    Args:
        url: direct PDF URL
        domain: which of the 9 domains this belongs to, e.g. "pharmacovigilance"
        title: a human-readable name for this document
    """
    print(f"\nIngesting: {title}")
    print(f"  URL: {url}")

    try:
        text = download_pdf_text(url)
    except Exception as e:
        print(f"  FAILED to download/extract: {e}")
        return 0

    if not text.strip():
        print("  No extractable text found (may be a scanned/image PDF) — skipping.")
        return 0

    chunks = chunk_text(text)
    print(f"  Extracted {len(text.split())} words -> {len(chunks)} chunks")

    if not chunks:
        return 0

    embeddings = embedding_model.encode(chunks).tolist()
    ids = [make_chunk_id(url, i) for i in range(len(chunks))]
    metadatas = [
        {"source_url": url, "domain": domain, "title": title, "chunk_index": i}
        for i in range(len(chunks))
    ]

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas,
    )

    print(f"  Stored {len(chunks)} chunks in the database.")
    return len(chunks)


def ingest_documents(document_list: list[dict]):
    """
    Ingests a list of documents in one go.
    document_list: [{"url": ..., "domain": ..., "title": ...}, ...]
    """
    total_chunks = 0
    for doc in document_list:
        total_chunks += ingest_document(doc["url"], doc["domain"], doc["title"])

    print(f"\n{'=' * 60}")
    print(f"Ingestion complete. Total chunks stored: {total_chunks}")
    print(f"Database location: {CHROMA_PATH}")


if __name__ == "__main__":
    # Test batch: the 3 confirmed-working Pharmacovigilance PDFs from IPC/PvPI
    pharmacovigilance_docs = [
        {
            "url": "https://ipc.gov.in/images/Version_1.0.pdf",
            "domain": "pharmacovigilance",
            "title": "Pharmacovigilance Guidance Document for Marketing Authorization Holders (v1.0)",
        },
        {
            "url": "https://ipc.gov.in/images/pdf/File268.pdf",
            "domain": "pharmacovigilance",
            "title": "Guidance Document for Spontaneous Adverse Drug Reaction Reporting",
        },
        {
            "url": "https://ipc.gov.in/images/1_ICH_ICSR_Implementation_Guide_v5_02.pdf",
            "domain": "pharmacovigilance",
            "title": "Implementation Guide for Electronic Transmission of ICSRs",
        },
    ]

    ingest_documents(pharmacovigilance_docs)