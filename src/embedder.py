# src/embedder.py — Sentence Transformer Embedding Engine
"""
Efficient embedding module using the challenge-imposed model:
    all-MiniLM-L6-v2 (384 dimensions, sentence-transformers)

Key design decisions:
    - Model loaded ONCE at module level (singleton pattern)
    - normalize_embeddings=True → cosine similarity = dot product (faster)
    - Batch processing for ingestion efficiency
    - Single-query mode for search-time latency
"""

import logging
from typing import List

from sentence_transformers import SentenceTransformer

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import EMBEDDING_MODEL, BATCH_SIZE

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Singleton Model Loading — loaded ONCE when module is imported
# ──────────────────────────────────────────────────────────────

logger.info("Loading embedding model: %s ...", EMBEDDING_MODEL)
_model = SentenceTransformer(EMBEDDING_MODEL)
logger.info("✓ Embedding model loaded successfully: %s", EMBEDDING_MODEL)


# ──────────────────────────────────────────────────────────────
# Batch Embedding — for ingestion pipeline
# ──────────────────────────────────────────────────────────────

def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for a batch of text chunks.

    Used during the ingestion pipeline to embed all chunks from a PDF
    in a single efficient batch operation.

    Args:
        texts: List of text strings to embed.

    Returns:
        List of embedding vectors, each a list of 384 floats.
        Embeddings are L2-normalized (normalize_embeddings=True),
        which means cosine similarity = dot product.

    Raises:
        No exceptions — logs errors and returns empty list on failure.
    """
    if not texts:
        logger.warning("embed_texts called with empty list")
        return []

    try:
        embeddings = _model.encode(
            texts,
            batch_size=BATCH_SIZE,
            normalize_embeddings=True,   # REQUIRED by challenge
            show_progress_bar=False,     # caller controls UI
            convert_to_numpy=True
        )
        # Convert numpy arrays to plain Python lists for psycopg2 compatibility
        return [emb.tolist() for emb in embeddings]
    except Exception as e:
        logger.error("Failed to embed batch of %d texts: %s", len(texts), e)
        return []


# ──────────────────────────────────────────────────────────────
# Single Query Embedding — for search time
# ──────────────────────────────────────────────────────────────

def embed_query(question: str) -> List[float]:
    """
    Generate an embedding for a single search query.

    Used at search time to convert a natural language question
    into a vector for similarity comparison against stored embeddings.

    Args:
        question: The natural language question to embed.

    Returns:
        Embedding vector as a list of 384 floats.
        L2-normalized for cosine similarity computation.

    Raises:
        No exceptions — logs errors and returns empty list on failure.
    """
    if not question or not question.strip():
        logger.warning("embed_query called with empty question")
        return []

    try:
        embedding = _model.encode(
            question,
            normalize_embeddings=True,   # REQUIRED by challenge
            show_progress_bar=False,
            convert_to_numpy=True
        )
        return embedding.tolist()
    except Exception as e:
        logger.error("Failed to embed query '%s': %s", question[:50], e)
        return []
