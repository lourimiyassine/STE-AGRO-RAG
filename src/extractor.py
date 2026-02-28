# src/extractor.py — 3-Layer PDF Text Extraction Engine
"""
Bulletproof PDF text extraction with 3-layer fallback:
    Layer 1: pdfplumber  — best for digital/structured PDFs
    Layer 2: PyMuPDF     — handles embedded fonts, complex layouts
    Layer 3: Tesseract   — OCR for scanned/image-only PDFs

Every PDF will be processed. No PDF left behind.
"""

import re
import logging
import unicodedata

import pdfplumber
try:
    import fitz  # PyMuPDF (older versions)
except ImportError:
    import pymupdf as fitz  # PyMuPDF >= 1.24.3
from PIL import Image
import pytesseract

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Text Cleaning
# ──────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Deep-clean extracted PDF text for optimal chunking and embedding.

    Operations:
        - Remove control characters (keep newlines and tabs)
        - Normalize Unicode (NFC form for French accents)
        - Remove page number patterns (Page 3, - 3 -, 3/10, etc.)
        - Remove repeated header/footer artifacts
        - Collapse multiple whitespace into single space
        - Collapse multiple newlines into double newline (paragraph break)
        - Strip leading/trailing whitespace
    """
    if not text:
        return ""

    # Normalize Unicode (important for French accented characters)
    text = unicodedata.normalize('NFC', text)

    # Remove control characters except newline (\n), tab (\t), carriage return (\r)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)

    # Remove page number patterns:
    # "Page 3", "page 3", "PAGE 3"
    text = re.sub(r'(?i)\bpage\s+\d+\b', '', text)
    # "- 3 -", "— 3 —"
    text = re.sub(r'[-—–]\s*\d+\s*[-—–]', '', text)
    # "3/10", "3 / 10" (page X of Y)
    text = re.sub(r'\b\d+\s*/\s*\d+\b', '', text)
    # Standalone page numbers at line start/end
    text = re.sub(r'(?m)^\s*\d{1,3}\s*$', '', text)

    # Remove common header/footer artifacts
    text = re.sub(r'(?i)(confidential|proprietary|all rights reserved|©.*?\d{4})', '', text)

    # Collapse multiple spaces into single space
    text = re.sub(r'[ \t]+', ' ', text)

    # Collapse 3+ newlines into double newline (preserve paragraph breaks)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Strip each line
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)

    # Final strip
    text = text.strip()

    return text


# ──────────────────────────────────────────────────────────────
# Layer 1: pdfplumber extraction
# ──────────────────────────────────────────────────────────────

def _extract_with_pdfplumber(pdf_path: str) -> str:
    """
    Extract text using pdfplumber — best for digitally-created PDFs
    with clear text layers. Handles tables and structured layouts well.

    Args:
        pdf_path: Absolute or relative path to the PDF file.

    Returns:
        Extracted and cleaned text, or empty string on failure.
    """
    try:
        text_parts = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
        full_text = '\n\n'.join(text_parts)
        return clean_text(full_text)
    except Exception as e:
        logger.warning("pdfplumber failed on '%s': %s", pdf_path, e)
        return ""


# ──────────────────────────────────────────────────────────────
# Layer 2: PyMuPDF (fitz) extraction
# ──────────────────────────────────────────────────────────────

def _extract_with_pymupdf(pdf_path: str) -> str:
    """
    Extract text using PyMuPDF (fitz) — handles embedded fonts,
    complex layouts, and some PDFs that pdfplumber cannot parse.

    Args:
        pdf_path: Absolute or relative path to the PDF file.

    Returns:
        Extracted and cleaned text, or empty string on failure.
    """
    try:
        text_parts = []
        doc = fitz.open(pdf_path)
        for page in doc:
            page_text = page.get_text("text")
            if page_text:
                text_parts.append(page_text)
        doc.close()
        full_text = '\n\n'.join(text_parts)
        return clean_text(full_text)
    except Exception as e:
        logger.warning("PyMuPDF failed on '%s': %s", pdf_path, e)
        return ""


# ──────────────────────────────────────────────────────────────
# Layer 3: Tesseract OCR extraction
# ──────────────────────────────────────────────────────────────

def _extract_with_ocr(pdf_path: str, dpi: int = 144) -> str:
    """
    Extract text using Tesseract OCR — last resort for scanned/image-only PDFs.
    Renders each page as an image at 2x zoom (144 DPI) then runs OCR.

    Uses 'fra+eng' language pack for French + English technical documents.

    Args:
        pdf_path: Absolute or relative path to the PDF file.
        dpi: Resolution for rendering pages. Default 144 (2x zoom).

    Returns:
        Extracted and cleaned text, or empty string on failure.
    """
    try:
        text_parts = []
        doc = fitz.open(pdf_path)
        zoom = dpi / 72  # 72 is default DPI, so 144/72 = 2x zoom
        matrix = fitz.Matrix(zoom, zoom)

        for page_num, page in enumerate(doc):
            try:
                # Render page to image
                pix = page.get_pixmap(matrix=matrix)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

                # Run OCR with French + English
                page_text = pytesseract.image_to_string(img, lang='fra+eng')
                if page_text and page_text.strip():
                    text_parts.append(page_text)
            except Exception as e:
                logger.warning("OCR failed on page %d of '%s': %s", page_num + 1, pdf_path, e)

        doc.close()
        full_text = '\n\n'.join(text_parts)
        return clean_text(full_text)
    except Exception as e:
        logger.warning("Tesseract OCR failed entirely on '%s': %s", pdf_path, e)
        return ""


# ──────────────────────────────────────────────────────────────
# Main Extraction Function — 3-Layer Fallback
# ──────────────────────────────────────────────────────────────

MIN_CHARS_THRESHOLD = 100  # minimum characters to consider extraction successful

def extract_pdf(pdf_path: str) -> str:
    """
    Extract text from a PDF using a 3-layer fallback strategy.

    Strategy:
        1. Try pdfplumber (fastest, best for digital PDFs)
        2. If result < 100 chars, try PyMuPDF
        3. If still < 100 chars, fall back to Tesseract OCR (slowest, but handles scans)

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Extracted and cleaned text. Returns empty string on total failure — never crashes.

    Logs:
        - Which extraction layer was used
        - Any failures encountered at each layer
    """
    # Layer 1: pdfplumber
    logger.info("Extracting '%s' — trying pdfplumber (Layer 1)...", pdf_path)
    text = _extract_with_pdfplumber(pdf_path)
    if len(text) >= MIN_CHARS_THRESHOLD:
        logger.info("✓ pdfplumber succeeded for '%s' (%d chars)", pdf_path, len(text))
        return text

    # Layer 2: PyMuPDF
    logger.info("pdfplumber insufficient (%d chars) — trying PyMuPDF (Layer 2)...", len(text))
    text = _extract_with_pymupdf(pdf_path)
    if len(text) >= MIN_CHARS_THRESHOLD:
        logger.info("✓ PyMuPDF succeeded for '%s' (%d chars)", pdf_path, len(text))
        return text

    # Layer 3: Tesseract OCR
    logger.info("PyMuPDF insufficient (%d chars) — trying Tesseract OCR (Layer 3)...", len(text))
    text = _extract_with_ocr(pdf_path)
    if len(text) >= MIN_CHARS_THRESHOLD:
        logger.info("✓ Tesseract OCR succeeded for '%s' (%d chars)", pdf_path, len(text))
        return text

    # Total failure
    logger.error("✗ All 3 extraction layers failed for '%s'. Returning empty string.", pdf_path)
    return text if text else ""
