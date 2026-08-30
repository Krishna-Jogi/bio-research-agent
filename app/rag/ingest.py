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
  4. Store each chunk in Chroma, a local vector database. Chroma embeds
     the text itself (turning it into numbers representing its meaning)
     using its built-in ONNX-based model — this used to be done manually
     with sentence-transformers/PyTorch, but that combination caused
     out-of-memory crashes on Render's free tier (512MB); ONNX Runtime
     runs the same underlying model with a much smaller memory footprint.

Later, retrieve.py uses the SAME embedding function to turn a student's
QUESTION into numbers the same way, then asks Chroma "which stored
chunks have embeddings closest to this question's embedding?" — that's
how it finds relevant content by meaning, not just keyword matching.
"""

import hashlib
import io

import chromadb
import requests
from chromadb.utils import embedding_functions
from pypdf import PdfReader

CHROMA_PATH = "app/rag/chroma_db"
COLLECTION_NAME = "govt_sources"

CHUNK_SIZE_WORDS = 350
CHUNK_OVERLAP_WORDS = 50


print("Setting up vector database (ONNX embedding — lightweight, no PyTorch needed)...")
_embedding_function = embedding_functions.ONNXMiniLM_L6_V2()

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(
    COLLECTION_NAME,
    embedding_function=_embedding_function,
)


def download_pdf_text(url: str) -> str:
    """Downloads a PDF from a URL and extracts its raw text."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "application/pdf,application/octet-stream,*/*",
        "Accept-Language": "en-US,en;q=0.9",
    }
    response = requests.get(url, timeout=30, headers=headers)
    response.raise_for_status()

    # Some government servers return an HTML error/redirect page instead
    # of the actual PDF when a request doesn't look enough like a real
    # browser — this check catches that clearly instead of letting the
    # PDF parser fail with a confusing low-level error.
    if not response.content.startswith(b"%PDF"):
        preview = response.content[:200]
        raise ValueError(
            f"Response is not a PDF (got {response.headers.get('Content-Type', 'unknown type')}). "
            f"First bytes: {preview}"
        )

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
    Downloads, chunks, and stores one document. Chroma handles embedding
    automatically using the collection's attached embedding function.

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

    ids = [make_chunk_id(url, i) for i in range(len(chunks))]
    metadatas = [
        {"source_url": url, "domain": domain, "title": title, "chunk_index": i}
        for i in range(len(chunks))
    ]

    # No embeddings passed explicitly — Chroma computes them itself using
    # the collection's attached ONNX embedding function.
    collection.upsert(
        ids=ids,
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

    nutraceuticals_docs = [
        {
            "url": "https://faolex.fao.org/docs/pdf/ind168163.pdf",
            "domain": "nutraceuticals",
            "title": "FSSAI Health Supplements, Nutraceuticals, and Novel Food Regulations, 2016 (FAO Legal Database mirror)",
        },
    ]

    water_docs = [
        {
            "url": "https://cdn.who.int/media/docs/default-source/medicines/norms-and-standards/guidelines/inspections/trs1033-annex3-gmp-water-for-pharmaceuticals-use.pdf?sfvrsn=aaa46ae5_4&download=true",
            "domain": "water",
            "title": "WHO Good Manufacturing Practices: Water for Pharmaceutical Use (TRS 1033, Annex 3)",
        },
    ]

    environment_docs = [
        {
            "url": "https://cpcb.nic.in/uploads/Industry-Specific-Standards/Effluent/73-pharmaceuticals.pdf",
            "domain": "environment",
            "title": "CPCB Pharmaceutical Industry Effluent and Emission Standards (Gazette Notification, 2021)",
        },
    ]

    chemical_engineering_docs = [
        {
            "url": "https://database.ich.org/sites/default/files/Q8_R2_Guideline.pdf",
            "domain": "chemical_engineering",
            "title": "ICH Q8(R2): Pharmaceutical Development (Quality by Design, Critical Quality Attributes, Process Design Space)",
        },
    ]

    ptc_docs = [
        {
            "url": "https://thsti.res.in/pdf/THSTI-BSG.pdf",
            "domain": "ptc",
            "title": "THSTI Biosafety Guidelines (DBT biosafety framework covering transgenic plants and plant tissue/cell culture)",
        },
    ]

    all_docs = (
        pharmacovigilance_docs
        + nutraceuticals_docs
        + water_docs
        + environment_docs
        + chemical_engineering_docs
        + ptc_docs
    )
    ingest_documents(all_docs)