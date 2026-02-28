# src/ingest.py — Master Ingestion Orchestration Pipeline
"""
Orchestrates the complete PDF-to-embedding pipeline:
    1. Initialize database schema
    2. Discover all PDFs recursively
    3. For each PDF: extract → chunk → embed → store
    4. Report results and failures

Designed for resilience: individual PDF failures never crash the pipeline.
"""

import os
import glob
import logging
from pathlib import Path
from typing import List

from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PDF_DIR

from src.extractor import extract_pdf
from src.chunker import chunk_text
from src.embedder import embed_texts
from src.db import init_db, insert_chunks

logger = logging.getLogger(__name__)


def discover_pdfs(pdf_dir: str = PDF_DIR) -> List[str]:
    """
    Discover all PDF files in the given directory recursively.

    Searches for both *.pdf and **/*.pdf patterns and deduplicates.

    Args:
        pdf_dir: Root directory to search for PDFs.

    Returns:
        Sorted, deduplicated list of absolute PDF paths.
    """
    # Search for PDFs at all levels
    pattern_flat = os.path.join(pdf_dir, '*.pdf')
    pattern_deep = os.path.join(pdf_dir, '**', '*.pdf')

    pdf_files = set()
    pdf_files.update(glob.glob(pattern_flat))
    pdf_files.update(glob.glob(pattern_deep, recursive=True))

    # Normalize paths and sort
    pdf_files = sorted(set(os.path.abspath(p) for p in pdf_files))

    logger.info("Discovered %d PDF files in '%s'", len(pdf_files), pdf_dir)
    return pdf_files


def ingest_all_pdfs(pdf_dir: str = PDF_DIR):
    """
    Run the complete ingestion pipeline: discover, extract, chunk, embed, store.

    Pipeline steps for each PDF:
        1. Extract text (3-layer fallback)
        2. Skip if empty
        3. Chunk text (sliding window, sentence boundaries)
        4. Skip if no chunks
        5. Prepend source filename to each chunk
        6. Embed chunks in batch
        7. Insert into database

    Error handling:
        - Individual PDF failures are logged and tracked
        - Pipeline continues to next PDF on failure
        - Failed PDFs are written to 'failed_pdfs.txt'
        - Final summary printed with success/failure counts

    Args:
        pdf_dir: Root directory containing PDFs. Default from config.
    """
    print("\n" + "=" * 60)
    print("  RAG BAKERY — INGESTION PIPELINE")
    print("=" * 60 + "\n")

    # Step 1: Initialize database
    logger.info("Step 1: Initializing database...")
    print("→ Initializing database schema...")
    init_db()

    # Step 2: Discover PDFs
    logger.info("Step 2: Discovering PDFs...")
    pdf_files = discover_pdfs(pdf_dir)

    if not pdf_files:
        print("⚠ No PDF files found in '%s'. Nothing to ingest." % pdf_dir)
        logger.warning("No PDF files found in '%s'", pdf_dir)
        return {'processed': 0, 'skipped': 0, 'failed': 0, 'total_chunks': 0}

    print(f"→ Found {len(pdf_files)} PDF files to process\n")

    # Tracking
    total_chunks_inserted = 0
    pdfs_processed = 0
    pdfs_skipped = 0
    failed_pdfs = []

    # Step 3-4: Process each PDF
    for doc_id, pdf_path in enumerate(tqdm(pdf_files, desc="Processing PDFs", unit="pdf"), start=1):
        filename = Path(pdf_path).stem  # filename without extension

        try:
            # 3a. Extract text
            text = extract_pdf(pdf_path)
            if not text or len(text.strip()) < 50:
                logger.warning("Skipping '%s': insufficient text extracted (%d chars)",
                             filename, len(text) if text else 0)
                pdfs_skipped += 1
                continue

            # 3b. Chunk text
            chunks = chunk_text(text)
            if not chunks:
                logger.warning("Skipping '%s': no valid chunks generated", filename)
                pdfs_skipped += 1
                continue

            # 3c. Prepend source filename to each chunk for traceability
            chunks_with_source = [
                f"[Source: {filename}] {chunk}" for chunk in chunks
            ]

            # 3d. Embed chunks in batch
            embeddings = embed_texts(chunks_with_source)
            if not embeddings or len(embeddings) != len(chunks_with_source):
                logger.error("Embedding failed for '%s': got %d embeddings for %d chunks",
                           filename,
                           len(embeddings) if embeddings else 0,
                           len(chunks_with_source))
                failed_pdfs.append(pdf_path)
                continue

            # 3e. Insert into database
            insert_chunks(doc_id, chunks_with_source, embeddings)

            total_chunks_inserted += len(chunks_with_source)
            pdfs_processed += 1
            logger.info("✓ '%s': %d chunks inserted (doc_id=%d)",
                       filename, len(chunks_with_source), doc_id)

        except Exception as e:
            logger.error("✗ Failed to process '%s': %s", filename, e, exc_info=True)
            failed_pdfs.append(pdf_path)
            continue

    # Step 5: Print summary
    print("\n" + "=" * 60)
    print("  INGESTION COMPLETE — SUMMARY")
    print("=" * 60)
    print(f"  ✓ PDFs processed successfully : {pdfs_processed}")
    print(f"  ⊘ PDFs skipped (no content)   : {pdfs_skipped}")
    print(f"  ✗ PDFs failed                 : {len(failed_pdfs)}")
    print(f"  ◉ Total chunks inserted       : {total_chunks_inserted}")
    print("=" * 60 + "\n")

    # Step 6: Write failed PDFs list
    if failed_pdfs:
        failed_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'failed_pdfs.txt')
        with open(failed_file, 'w', encoding='utf-8') as f:
            f.write("# Failed PDFs — Ingestion Pipeline\n")
            f.write(f"# Total failures: {len(failed_pdfs)}\n\n")
            for pdf in failed_pdfs:
                f.write(f"{pdf}\n")
        print(f"⚠ Failed PDFs written to: {failed_file}")
        logger.warning("Wrote %d failed PDFs to '%s'", len(failed_pdfs), failed_file)

    return {
        'processed': pdfs_processed,
        'skipped': pdfs_skipped,
        'failed': len(failed_pdfs),
        'total_chunks': total_chunks_inserted
    }
