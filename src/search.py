# src/search.py — Core Semantic Search Module
"""
The heart of the RAG system — this is what gets evaluated in the challenge.

Provides:
    - semantic_search(): Returns top 3 most similar fragments as dicts
    - format_results(): Formats results in the exact challenge output format

All results use cosine similarity scores between 0 and 1.
Method: Pure Cosine Similarity (challenge-imposed)
Model: all-MiniLM-L6-v2 (challenge-imposed)
Top K: 3 (challenge-imposed)
"""

import logging
from typing import Dict, List

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TOP_K

from src.embedder import embed_query
from src.db import search_similar

logger = logging.getLogger(__name__)


def semantic_search(question: str, top_k: int = TOP_K) -> List[Dict]:
    """
    Perform semantic search: find the top K most similar text fragments
    to the given natural language question.

    Pipeline:
        1. Validate input
        2. Generate embedding of the question using all-MiniLM-L6-v2
        3. Compare against stored embeddings using Cosine Similarity (pgvector <=>)
        4. Return top 3 results ranked by descending similarity score

    Args:
        question: Natural language question (French or English).
        top_k:    Number of results to return. Default: 3 (challenge-imposed).

    Returns:
        List of result dicts, each containing:
            - rank: int (1, 2, 3)
            - doc_id: int
            - fragment: str (the matched text fragment)
            - score: float (cosine similarity, rounded to 4 decimal places)

        Returns empty list if question is empty or search fails.
    """
    if not question or not question.strip():
        logger.warning("semantic_search called with empty question")
        return []

    # Step 1: Generate embedding for the question
    logger.info("Searching for: '%s'", question[:80])
    query_vector = embed_query(question)

    if not query_vector:
        logger.error("Failed to embed question: '%s'", question[:80])
        return []

    # Step 2: Search database using pure cosine similarity
    raw_results = search_similar(query_vector, top_k)

    if not raw_results:
        logger.info("No results found for: '%s'", question[:80])
        return []

    # Step 3: Format results as ranked dictionaries
    results = []
    for rank, (row_id, doc_id, fragment, score) in enumerate(raw_results, start=1):
        results.append({
            'rank': rank,
            'doc_id': doc_id,
            'fragment': fragment,
            'score': round(float(score), 4)
        })

    logger.info("Returned %d results (top score: %.4f)", len(results),
                results[0]['score'] if results else 0.0)

    return results


def format_results(results: List[Dict]) -> str:
    """
    Format search results in the exact format expected by the challenge.

    Output format (as specified by jury):
        Résultat 1
        Texte : "..."
        Score : 0.91

        Résultat 2
        Texte : "..."
        Score : 0.87

        Résultat 3
        Texte : "..."
        Score : 0.82

    Args:
        results: List of result dicts from semantic_search().

    Returns:
        Formatted string ready for display or challenge submission.
    """
    if not results:
        return "Aucun résultat trouvé."

    output_parts = []
    for result in results:
        # Truncate very long fragments for display (keep first 500 chars)
        fragment_display = result['fragment']
        if len(fragment_display) > 500:
            fragment_display = fragment_display[:497] + "..."

        part = (
            f"Résultat {result['rank']}\n"
            f"Texte : \"{fragment_display}\"\n"
            f"Score : {result['score']:.2f}"
        )
        output_parts.append(part)

    return '\n\n'.join(output_parts)
