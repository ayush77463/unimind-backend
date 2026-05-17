"""Document upload endpoint — server-side text extraction for PDF, DOCX, TXT."""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Document"])

_MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
_MAX_TEXT_CHARS = 50_000


@router.post("/document/upload")
async def upload_document(
    file: UploadFile = File(...),
    user_id: str = Form(default="anonymous"),
    question: str = Form(default=""),
):
    """Extract text from an uploaded document (PDF, DOCX, TXT, etc.)."""
    filename = file.filename or "unknown"
    ext = Path(filename).suffix.lower()
    logger.info("Document upload: %s (%s) from user %s", filename, ext, user_id)

    try:
        raw_bytes = await file.read()
    except Exception as exc:
        logger.error("Failed to read uploaded file: %s", exc)
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": f"Failed to read file: {exc}"},
        )

    if len(raw_bytes) > _MAX_FILE_SIZE:
        return JSONResponse(
            status_code=413,
            content={
                "success": False,
                "error": f"File too large ({len(raw_bytes)} bytes). Max {_MAX_FILE_SIZE // (1024*1024)} MB.",
            },
        )

    if len(raw_bytes) == 0:
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": "Empty file uploaded."},
        )

    extracted_text = ""
    extraction_method = "unknown"

    try:
        if ext == ".pdf":
            extracted_text, extraction_method = _extract_pdf(raw_bytes, filename)
        elif ext in {".docx", ".doc"}:
            extracted_text, extraction_method = _extract_docx(raw_bytes, filename)
        elif ext in {".txt", ".md", ".csv", ".json", ".py", ".js", ".html", ".css", ".xml", ".yaml", ".yml"}:
            extracted_text, extraction_method = _extract_text(raw_bytes, filename)
        else:
            extracted_text, extraction_method = _extract_text(raw_bytes, filename)
    except Exception as exc:
        logger.error("Text extraction failed for %s: %s", filename, exc)
        return JSONResponse(
            status_code=422,
            content={
                "success": False,
                "error": f"Failed to extract text from {filename}: {exc}",
            },
        )

    if not extracted_text.strip():
        return JSONResponse(
            status_code=422,
            content={
                "success": False,
                "error": f"No readable text found in {filename}.",
            },
        )

    # Truncate extremely long documents
    if len(extracted_text) > _MAX_TEXT_CHARS:
        extracted_text = extracted_text[:_MAX_TEXT_CHARS] + "\n...[truncated]"

    logger.info(
        "Extracted %d chars from %s using %s",
        len(extracted_text), filename, extraction_method,
    )

    return {
        "success": True,
        "filename": filename,
        "extracted_text": extracted_text,
        "char_count": len(extracted_text),
        "extraction_method": extraction_method,
        "file_size_bytes": len(raw_bytes),
    }


def _extract_pdf(raw_bytes: bytes, filename: str) -> tuple[str, str]:
    """Extract text from PDF using PyPDF2."""
    try:
        from PyPDF2 import PdfReader
    except ImportError:
        logger.warning("PyPDF2 not installed — falling back to raw text extraction")
        return _extract_text_raw(raw_bytes, filename), "raw_fallback"

    tmp = None
    try:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        tmp.write(raw_bytes)
        tmp.close()

        reader = PdfReader(tmp.name)
        pages: list[str] = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                pages.append(f"[Page {i + 1}]\n{text.strip()}")

        if not pages:
            logger.info("PyPDF2 found no text in %s — trying raw extraction", filename)
            return _extract_text_raw(raw_bytes, filename), "raw_fallback"

        return "\n\n".join(pages), "pypdf2"
    finally:
        if tmp:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass


def _extract_docx(raw_bytes: bytes, filename: str) -> tuple[str, str]:
    """Extract text from DOCX using python-docx."""
    try:
        import docx
    except ImportError:
        logger.warning("python-docx not installed — falling back to raw text extraction")
        return _extract_text_raw(raw_bytes, filename), "raw_fallback"

    import io
    try:
        doc = docx.Document(io.BytesIO(raw_bytes))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        if not paragraphs:
            return _extract_text_raw(raw_bytes, filename), "raw_fallback"
        return "\n\n".join(paragraphs), "python_docx"
    except Exception as exc:
        logger.warning("python-docx failed on %s: %s", filename, exc)
        return _extract_text_raw(raw_bytes, filename), "raw_fallback"


def _extract_text(raw_bytes: bytes, filename: str) -> tuple[str, str]:
    """Extract text from plain text files."""
    for encoding in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            return raw_bytes.decode(encoding), f"text_{encoding}"
        except (UnicodeDecodeError, ValueError):
            continue
    return _extract_text_raw(raw_bytes, filename), "raw_fallback"


def _extract_text_raw(raw_bytes: bytes, filename: str) -> str:
    """Last-resort: filter printable ASCII chars."""
    chars = bytes(b for b in raw_bytes if 32 <= b < 127 or b in (10, 13, 9))
    text = chars.decode("ascii", errors="ignore")
    # Clean up excessive whitespace
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)
