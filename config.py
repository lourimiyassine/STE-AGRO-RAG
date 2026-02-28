# config.py — ALL constants here, never hardcode in logic files
"""
Central configuration for the RAG Bakery Semantic Search Module.
All constants, paths, and environment variables are managed here.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ──────────────────────────────────────────────────────────────
# Database Configuration
# ──────────────────────────────────────────────────────────────
DB_HOST     = os.getenv('DB_HOST', 'localhost')
DB_PORT     = os.getenv('DB_PORT', '5433')
DB_NAME     = os.getenv('DB_NAME', 'bakery_rag')
DB_USER     = os.getenv('DB_USER', 'postgres')
DB_PASSWORD = os.getenv('DB_PASSWORD', 'secret')

# ──────────────────────────────────────────────────────────────
# Embedding Configuration — IMPOSED by challenge — DO NOT CHANGE
# ──────────────────────────────────────────────────────────────
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
EMBEDDING_DIM   = 384
TOP_K           = 3

# ──────────────────────────────────────────────────────────────
# Ingestion Tuning
# ──────────────────────────────────────────────────────────────
BATCH_SIZE      = 64    # chunks per embedding batch
CHUNK_SIZE      = 300   # target words per chunk (challenge spec)
CHUNK_OVERLAP   = 50    # words overlap between consecutive chunks (challenge spec)
MIN_CHUNK_WORDS = 30    # discard chunks shorter than this (challenge spec)

# ──────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────
PDF_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'pdfs')
