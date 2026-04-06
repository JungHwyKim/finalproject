from __future__ import annotations

import re
from typing import Dict, List, Tuple

import fitz
import pdfplumber

from .models import FigureItem, PaperArtifacts, TableItem


SECTION_PATTERN = re.compile(
    r"^\s*(abstract|introduction|related work|method|methods|experiment|experiments|result|results|conclusion|limitations)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def _split_sections(text: str) -> Dict[str, str]:
    matches = list(SECTION_PATTERN.finditer(text))
    if not matches:
        return {"full_text": text}
    sections: Dict[str, str] = {}
    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        title = match.group(1).strip().lower()
        sections[title] = text[start:end].strip()
    return sections


def _extract_references(text: str) -> List[str]:
    ref_start = re.search(
        r"^\s*(references|bibliography)\s*$", text, re.IGNORECASE | re.MULTILINE
    )
    if not ref_start:
        return []
    tail = text[ref_start.end() :].strip()
    lines = [ln.strip() for ln in tail.splitlines() if ln.strip()]
    refs: List[str] = []
    bucket: List[str] = []
    for ln in lines:
        if re.match(r"^(\[\d+\]|\d+\.)\s+", ln):
            if bucket:
                refs.append(" ".join(bucket))
                bucket = []
            bucket.append(ln)
        else:
            bucket.append(ln)
    if bucket:
        refs.append(" ".join(bucket))
    return refs[:100]


def _extract_figures_and_captions(pdf_path: str) -> List[FigureItem]:
    doc = fitz.open(pdf_path)
    figures: List[FigureItem] = []
    for page_idx in range(len(doc)):
        page = doc.load_page(page_idx)
        page_text = page.get_text("text")
        for line in page_text.splitlines():
            line = line.strip()
            if re.match(r"^(figure|fig\.)\s*\d+", line, re.IGNORECASE):
                figures.append(FigureItem(page=page_idx + 1, caption=line))
    doc.close()
    return figures


def _extract_tables(pdf_path: str) -> List[TableItem]:
    tables: List[TableItem] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_idx, page in enumerate(pdf.pages, start=1):
            extracted = page.extract_tables()
            for table in extracted:
                normalized = [
                    [cell.strip() if cell else "" for cell in row] for row in table
                ]
                tables.append(TableItem(page=page_idx, rows=normalized))
    return tables


def extract_paper_artifacts(pdf_path: str, filename: str) -> PaperArtifacts:
    all_pages: List[str] = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            all_pages.append(page.get_text("text"))
    full_text = "\n".join(all_pages).strip()
    sections = _split_sections(full_text)
    references = _extract_references(full_text)
    figures = _extract_figures_and_captions(pdf_path)
    tables = _extract_tables(pdf_path)

    return PaperArtifacts(
        filename=filename,
        full_text=full_text,
        section_texts=sections,
        references=references,
        tables=tables,
        figures=figures,
    )


def make_simple_chunks(text: str, chunk_size: int = 800) -> List[str]:
    words = text.split()
    chunks: List[str] = []
    cur: List[str] = []
    cur_len = 0
    for w in words:
        cur.append(w)
        cur_len += len(w) + 1
        if cur_len >= chunk_size:
            chunks.append(" ".join(cur))
            cur, cur_len = [], 0
    if cur:
        chunks.append(" ".join(cur))
    return chunks[:30]


def top_k_chunks_for_query(chunks: List[str], query: str, k: int = 3) -> List[str]:
    q_tokens = {t.lower() for t in re.findall(r"\w+", query)}
    if not q_tokens:
        return chunks[:k]
    scored: List[Tuple[int, str]] = []
    for chunk in chunks:
        c_tokens = {t.lower() for t in re.findall(r"\w+", chunk)}
        score = len(q_tokens & c_tokens)
        scored.append((score, chunk))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for s, c in scored[:k] if s > 0] or chunks[:k]
