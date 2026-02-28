# src/db.py — PostgreSQL + pgvector Database Operations
"""
All database operations for the RAG Bakery system.

Manages:
    - Connection handling
    - Schema initialization (pgvector extension, embeddings table, HNSW index)
    - Batch chunk insertion
    - Cosine similarity search via pgvector <=> operator
"""

import logging
from typing import List, Tuple

import psycopg

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD, EMBEDDING_DIM, TOP_K

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Connection
# ──────────────────────────────────────────────────────────────

def get_connection():
    """
    Create and return a new PostgreSQL connection using config values.

    Returns:
        psycopg connection object.

    Raises:
        psycopg.Error: If connection cannot be established.
    """
    try:
        conn = psycopg.connect(
            host=DB_HOST,
            port=int(DB_PORT),
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        logger.debug("Database connection established to %s:%s/%s", DB_HOST, DB_PORT, DB_NAME)
        return conn
    except psycopg.Error as e:
        logger.error("Failed to connect to database: %s", e)
        raise


# ──────────────────────────────────────────────────────────────
# Schema Initialization
# ──────────────────────────────────────────────────────────────

def init_db():
    """
    Initialize the database schema for the RAG system.

    Operations:
        1. Enable pgvector extension
        2. Create 'embeddings' table with exact challenge schema:
           - id: SERIAL PRIMARY KEY
           - id_document: INTEGER NOT NULL
           - texte_fragment: TEXT NOT NULL
           - vecteur: VECTOR(384)
        3. Create HNSW index on vecteur column for fast cosine similarity search
           - m=16, ef_construction=64

    Idempotent: safe to call multiple times (uses IF NOT EXISTS).
    """
    conn = get_connection()
    cur = conn.cursor()

    try:
        # Step 1: Enable pgvector extension
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        logger.info("✓ pgvector extension enabled")

        # Step 2: Create embeddings table with exact challenge schema
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS embeddings (
                id           SERIAL PRIMARY KEY,
                id_document  INTEGER NOT NULL,
                texte_fragment TEXT NOT NULL,
                vecteur      VECTOR({EMBEDDING_DIM})
            );
        """)
        logger.info("✓ Table 'embeddings' created (or already exists)")

        # Step 3: Create HNSW index for fast cosine similarity search
        cur.execute("""
            CREATE INDEX IF NOT EXISTS embeddings_hnsw_idx
            ON embeddings
            USING hnsw (vecteur vector_cosine_ops)
            WITH (m = 16, ef_construction = 64);
        """)
        logger.info("✓ HNSW index created (or already exists)")

        conn.commit()
        logger.info("✓ Database initialization complete")
        print("✓ Database initialized: pgvector extension, 'embeddings' table, HNSW index")

    except psycopg.Error as e:
        conn.rollback()
        logger.error("Database initialization failed: %s", e)
        raise
    finally:
        cur.close()
        conn.close()


# ──────────────────────────────────────────────────────────────
# Chunk Insertion
# ──────────────────────────────────────────────────────────────

def insert_chunks(
    id_document: int,
    chunks: List[str],
    embeddings: List[List[float]]
):
    """
    Batch insert text chunks with their embedding vectors into the database.

    Args:
        id_document: Integer document ID for this PDF.
        chunks:      List of text chunk strings.
        embeddings:  List of embedding vectors (each a list of 384 floats).

    Raises:
        ValueError: If chunks and embeddings have different lengths.
        psycopg.Error: If database operation fails.
    """
    if len(chunks) != len(embeddings):
        raise ValueError(
            f"Mismatch: {len(chunks)} chunks but {len(embeddings)} embeddings"
        )

    if not chunks:
        logger.warning("insert_chunks called with empty data for document %d", id_document)
        return

    conn = get_connection()
    cur = conn.cursor()

    try:
        # Prepare data tuples: (id_document, texte_fragment, vecteur_as_string)
        data = []
        for chunk, embedding in zip(chunks, embeddings):
            # pgvector expects the vector as a string representation: '[0.1, 0.2, ...]'
            vector_str = '[' + ','.join(str(v) for v in embedding) + ']'
            data.append((id_document, chunk, vector_str))

        # Batch insert using executemany for efficiency
        cur.executemany(
            """
            INSERT INTO embeddings (id_document, texte_fragment, vecteur)
            VALUES (%s, %s, %s::vector)
            """,
            data
        )

        conn.commit()
        logger.info("✓ Inserted %d chunks for document %d", len(chunks), id_document)

    except psycopg.Error as e:
        conn.rollback()
        logger.error("Failed to insert chunks for document %d: %s", id_document, e)
        raise
    finally:
        cur.close()
        conn.close()


# ──────────────────────────────────────────────────────────────
# Cosine Similarity Search — PURE SEMANTIC (challenge requirement)
# ──────────────────────────────────────────────────────────────

def search_similar(
    query_vector: List[float],
    top_k: int = TOP_K
) -> List[Tuple]:
    """
    Search for the most similar text fragments using pure Cosine Similarity.

    Uses pgvector's <=> operator which computes cosine distance.
    Similarity score = 1 - cosine_distance.

    This is the EXACT method required by the challenge:
        - Method: Cosine Similarity
        - Top K: 3
        - No hybrid/lexical filtering — pure semantic search

    Args:
        query_vector: Query embedding vector (list of 384 floats).
        top_k:        Number of results to return. Default: 3.

    Returns:
        List of tuples: (id, id_document, texte_fragment, similarity_score)
        Sorted by similarity score descending.
    """
    if not query_vector:
        logger.warning("search_similar called with empty query vector")
        return []

    conn = get_connection()
    cur = conn.cursor()

    try:
        # Convert query vector to pgvector string format
        vector_str = '[' + ','.join(str(v) for v in query_vector) + ']'

        # Pure cosine similarity search — exactly what the challenge requires
        # <=> computes cosine distance, so similarity = 1 - distance
        # ORDER BY distance ASC = ORDER BY similarity DESC
        cur.execute(
            """
            SELECT
                id,
                id_document,
                texte_fragment,
                1 - (vecteur <=> %s::vector) AS similarity_score
            FROM embeddings
            ORDER BY vecteur <=> %s::vector ASC
            LIMIT %s
            """,
            (vector_str, vector_str, top_k)
        )

        results = cur.fetchall()
        logger.info("Cosine similarity search returned %d results", len(results))
        return results

    except psycopg.Error as e:
        logger.error("Similarity search failed: %s", e)
        return []
    finally:
        cur.close()
        conn.close()
