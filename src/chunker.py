# src/chunker.py — Intelligent Sentence-Boundary Sliding Window Chunker
"""
Splits extracted text into semantically meaningful chunks using a sliding
window approach that respects sentence boundaries.

Chunking strategy:
    1. Split text into sentences (handling French punctuation)
    2. Build chunks sentence-by-sentence until reaching target word count
    3. Carry over last N words as overlap for context continuity
    4. Discard chunks shorter than minimum word threshold
"""

import re
import logging
from typing import List

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_WORDS

logger = logging.getLogger(__name__)


def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences, handling French and English punctuation.

    Splits on:
        - Period followed by space and uppercase ('. ')
        - Exclamation mark followed by space ('! ')
        - Question mark followed by space ('? ')
        - Double newlines (paragraph breaks)
        - Semicolons in technical documents ('; ')

    Keeps only sentences with > 20 characters to filter out noise.

    Args:
        text: The raw text to split into sentences.

    Returns:
        List of clean, non-trivial sentences.
    """
    if not text or not text.strip():
        return []

    # First split on paragraph breaks (double newlines)
    paragraphs = re.split(r'\n\n+', text)

    sentences = []
    for paragraph in paragraphs:
        # Split on sentence-ending punctuation followed by space
        # Uses lookbehind to keep the punctuation with the sentence
        parts = re.split(r'(?<=[.!?])\s+', paragraph)

        for part in parts:
            # Further split on semicolons for technical docs
            sub_parts = re.split(r';\s+', part)
            for sub in sub_parts:
                cleaned = sub.strip()
                # Keep only meaningful sentences (> 20 chars)
                if len(cleaned) > 20:
                    sentences.append(cleaned)

    return sentences


def word_count(text: str) -> int:
    """
    Count the number of words in a text string.

    Args:
        text: Input text.

    Returns:
        Number of words (split on whitespace).
    """
    return len(text.split())


def split_into_sections(text: str) -> List[str]:
    """
    Split text into distinct thematic sections based on common headers
    found in technical pastry/bakery datasheets.
    
    This prevents mixing e.g. "Dosage" with "Stockage" in the same chunk.
    """
    if not text:
        return []
        
    # Common headers in the BVZyme and Acide Ascorbique PDFs
    # Using regex to match lines that look like titles
    section_patterns = [
        r"(?im)^Résumé Général\s*$",
        r"(?im)^Propriétés Principales\s*$",
        r"(?im)^Points Importants\s*$",
        r"(?im)^Dosages? Recommandés?.*$",
        r"(?im)^Spécifications Techniques\s*$",
        r"(?im)^Conditionnement.*$",
        r"(?im)^Mode d'Emploi.*$",
        r"(?im)^Avantages et Limitations\s*$",
        r"(?im)^Réglementation\s*$",
        r"(?im)^Stockage et Sécurité\s*$",
        r"(?im)^Product Description\s*$",
        r"(?im)^Application\s*$",
        r"(?im)^Usage\s*$",
        r"(?im)^Packaging.*$",
        r"(?im)^Storage.*$",
        r"(?im)^Technical Data.*$"
    ]
    
    # Combine patterns into one OR regex
    combined_pattern = "|".join(section_patterns)
    
    # Split text keeping the delimiters (headers) attached to the subsequent text
    # We use a capturing group for the split, which returns [text1, sep1, text2, sep2, ...]
    parts = re.split(f"({combined_pattern})", text)
    
    sections = []
    current_section = ""
    
    for i, part in enumerate(parts):
        # Even indices are content, odd indices are the matched headers
        if i % 2 == 1:
            # We hit a header, save the previous accumulated content if any
            if current_section.strip():
                sections.append(current_section.strip())
            # Start a new section starting with this header
            current_section = part
        else:
            current_section += part
            
    # Add the last section
    if current_section.strip():
        sections.append(current_section.strip())
        
    return sections if sections else [text]


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
    min_words: int = MIN_CHUNK_WORDS
) -> List[str]:
    """
    Chunk text intelligently:
        1. First splits document by thematic sections (Dosage, Storage, etc.)
        2. Then applies sliding-window sentence chunking WITHIN each section.
    
    This ensures chunks are semantically pure (no theme bleeding) while 
    respecting word limits (300 words max, 50 word overlap).
    """
    if not text:
        return []

    sections = split_into_sections(text)
    all_chunks = []
    
    for section_text in sections:
        sentences = split_into_sentences(section_text)
        if not sentences:
            continue

        section_chunks = []
        current_chunk_sentences = []
        current_word_count = 0

        for sentence in sentences:
            sentence_words = word_count(sentence)

            # If adding this sentence exceeds chunk_size
            if current_word_count + sentence_words > chunk_size and current_chunk_sentences:
                chunk_str = ' '.join(current_chunk_sentences)
                if word_count(chunk_str) >= min_words:
                    section_chunks.append(chunk_str)

                # Overlap logic
                all_words = chunk_str.split()
                overlap_words = all_words[-overlap:] if len(all_words) > overlap else all_words
                overlap_text = ' '.join(overlap_words)

                current_chunk_sentences = [overlap_text, sentence]
                current_word_count = len(overlap_words) + sentence_words
            else:
                current_chunk_sentences.append(sentence)
                current_word_count += sentence_words

        # Add the remaining sentences as the last chunk for this section
        if current_chunk_sentences:
            chunk_str = ' '.join(current_chunk_sentences)
            # We are slightly more lenient on min_words for section ends
            if word_count(chunk_str) >= min_words - 10: 
                section_chunks.append(chunk_str)
                
        all_chunks.extend(section_chunks)

    logger.info("Section-aware chunking generated %d chunks", len(all_chunks))
    return all_chunks
