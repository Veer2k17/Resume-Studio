"""Resume text extraction (PDF, DOCX, TXT).

Main entry point: extract_text(file, filename=None, ocr=False)

Always returns a dict:
    {"text": str, "status": "ok" | "error", "message": str}

On success, status is "ok" and text holds the raw extracted text.
On failure, status is "error", text is "" and message explains why.

The text is RAW. Pass it through utils.preprocessing.clean_text() before
giving it to the model.

ocr=True turns on OCR for scanned PDFs. It needs the Tesseract program
installed on the machine; if it is missing, scanned PDFs are rejected
with a clear message instead.
"""

import io
from pathlib import Path

import pymupdf  # PyMuPDF (needs version 1.24.3 or newer)
from docx import Document
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

MAX_SIZE_MB = 10
MIN_TEXT_CHARS = 100   # fewer characters than this = treated as empty/unreadable
MIN_LETTER_RATIO = 0.5  # below this share of letters, text is garbled
OCR_MAX_PAGES = 5      # resumes are short; do not OCR a whole book


def _result(text="", status="ok", message=""):
    return {"text": text, "status": status, "message": message}


def _error(message):
    return _result("", "error", message)


def _read_input(file, filename=None):
    """Return (bytes, extension) from a file path or a file-like object
    (e.g. a Streamlit UploadedFile)."""
    if isinstance(file, (str, Path)):
        path = Path(file)
        if not path.exists():
            raise FileNotFoundError("File not found.")
        return path.read_bytes(), path.suffix.lower()

    name = filename or getattr(file, "name", "")
    data = file.getvalue() if hasattr(file, "getvalue") else file.read()
    return data, Path(name).suffix.lower()


def _check_text(text, scanned_hint=False):
    stripped = text.strip()
    if len(stripped) < MIN_TEXT_CHARS:
        if scanned_hint:
            return _error(
                "No readable text found. This looks like a scanned or "
                "image-based PDF."
            )
        return _error("The file contains no readable text.")

    # Broken font encodings give text full of symbols or "(cid:12)" codes.
    chars = [c for c in stripped if not c.isspace()]
    letters = sum(c.isalpha() for c in chars)
    if "(cid:" in stripped or letters / max(len(chars), 1) < MIN_LETTER_RATIO:
        return _error(
            "The text could not be decoded properly (broken font encoding). "
            "Try exporting the resume to PDF again, or upload a DOCX."
        )
    return _result(text)


# ---------------------------------------------------------------- PDF

def _ocr_pdf(doc):
    """OCR the first few pages. Returns text, or None if OCR is unavailable."""
    try:
        pages = []
        for page in list(doc)[:OCR_MAX_PAGES]:
            textpage = page.get_textpage_ocr(language="eng", dpi=200, full=True)
            pages.append(page.get_text("text", textpage=textpage, sort=True))
        return "\n".join(pages)
    except Exception:
        return None


def _extract_pdf(data, ocr=False):
    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception:
        return _error("Could not open the PDF. The file may be corrupted.")

    with doc:
        if doc.needs_pass:
            return _error("The PDF is password-protected.")
        if doc.page_count == 0:
            return _error("The PDF has no pages.")
        # sort=True orders text blocks top-to-bottom, left-to-right,
        # which helps with multi-column resumes.
        text = "\n".join(page.get_text("text", sort=True) for page in doc)

        if len(text.strip()) < MIN_TEXT_CHARS:
            if not ocr:
                return _error(
                    "No readable text found. This looks like a scanned or "
                    "image-based PDF (OCR is turned off)."
                )
            text = _ocr_pdf(doc)
            if text is None:
                return _error(
                    "This looks like a scanned PDF, and OCR is not available "
                    "(Tesseract is not installed)."
                )

    return _check_text(text, scanned_hint=True)


# --------------------------------------------------------------- DOCX

def _table_lines(table):
    """Turn a DOCX table into text lines, skipping repeated merged cells
    and including tables nested inside cells."""
    lines = []
    for row in table.rows:
        seen, cells, nested = set(), [], []
        for cell in row.cells:
            if cell._tc in seen:
                continue
            seen.add(cell._tc)
            value = cell.text.strip()
            if value:
                cells.append(value)
            for inner in cell.tables:
                nested.extend(_table_lines(inner))
        if cells:
            lines.append(" | ".join(cells))
        lines.extend(nested)
    return lines


def _textbox_lines(element):
    """Text inside text boxes (common in designed resume templates).
    Word stores each box twice (Choice + Fallback); skip the fallback copy."""
    lines = []
    for box in element.iter(qn("w:txbxContent")):
        if any(a.tag.endswith("}Fallback") for a in box.iterancestors()):
            continue
        for p in box.iter(qn("w:p")):
            line = "".join(t.text or "" for t in p.iter(qn("w:t"))).strip()
            if line:
                lines.append(line)
    return lines


def _extract_docx(data):
    try:
        doc = Document(io.BytesIO(data))
    except Exception:
        return _error(
            "Could not open the DOCX. The file may be corrupted or not a real .docx."
        )

    parts, seen_headers = [], set()
    for section in doc.sections:  # headers often hold name and contact details
        for p in section.header.paragraphs:
            line = p.text.strip()
            if line and line not in seen_headers:
                seen_headers.add(line)
                parts.append(line)

    # Walk the body in order so tables stay in their original position.
    for child in doc.element.body.iterchildren():
        if child.tag.endswith("}p"):
            parts.append(Paragraph(child, doc).text)
            parts.extend(_textbox_lines(child))
        elif child.tag.endswith("}tbl"):
            parts.extend(_table_lines(Table(child, doc)))

    return _check_text("\n".join(parts))


# ---------------------------------------------------------------- TXT

def _extract_txt(data):
    for encoding in ("utf-8-sig", "utf-16", "cp1252", "latin-1"):
        try:
            return _check_text(data.decode(encoding))
        except (UnicodeDecodeError, UnicodeError):
            continue
    return _error("Could not decode the text file.")


# ------------------------------------------------------------- public

def extract_text(file, filename=None, ocr=False):
    """Extract text from a PDF, DOCX or TXT resume.

    file:     a path (str/Path) or a file-like object (Streamlit upload).
    filename: optional, used to detect the type if the object has no .name.
    ocr:      try OCR on scanned PDFs (needs Tesseract installed).
    """
    try:
        data, ext = _read_input(file, filename)
    except FileNotFoundError as e:
        return _error(str(e))
    except Exception:
        return _error("Could not read the file.")

    if not data:
        return _error("The file is empty.")
    if len(data) > MAX_SIZE_MB * 1024 * 1024:
        return _error(f"The file is larger than {MAX_SIZE_MB} MB.")

    try:
        if ext == ".pdf":
            return _extract_pdf(data, ocr=ocr)
        if ext == ".docx":
            return _extract_docx(data)
        if ext == ".txt":
            return _extract_txt(data)
    except Exception:
        return _error("Something went wrong while reading the file.")

    return _error(
        f"Unsupported file type '{ext or 'unknown'}'. Upload a PDF, DOCX or TXT."
    )
