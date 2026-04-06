from __future__ import annotations

import base64
import io
import os
from typing import List

import fitz
from PIL import Image

from .llm_utils import get_optional_llm
from .models import FigureItem

try:
    import pytesseract
except ImportError:  # pragma: no cover
    pytesseract = None


def _page_image_bytes(page: fitz.Page, zoom: float = 1.8) -> bytes:
    matrix = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=matrix, alpha=False)
    return pix.tobytes("png")


def _run_ocr(image_bytes: bytes) -> str:
    if pytesseract is None:
        return ""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        text = pytesseract.image_to_string(image, lang="eng+kor")
        return text.strip()
    except Exception:
        return ""


def _run_vision_llm(image_bytes: bytes, caption: str) -> str:
    # Optional: to avoid high cost in class demos, set ENABLE_VISION=false.
    if os.getenv("ENABLE_VISION", "true").lower() not in {"1", "true", "yes"}:
        return ""
    llm = get_optional_llm()
    if not llm:
        return ""
    try:
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        message = [
            {
                "type": "text",
                "text": (
                    "이 논문 페이지 이미지에서 해당 figure 내용을 간단히 설명해줘. "
                    f"캡션: {caption}\n"
                    "출력: 3문장 이내, 한국어."
                ),
            },
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        ]
        return llm.invoke([("human", message)]).content.strip()
    except Exception:
        return ""


def enrich_figures_with_ocr_and_vision(
    pdf_path: str, figures: List[FigureItem]
) -> List[FigureItem]:
    if not figures:
        return figures

    doc = fitz.open(pdf_path)
    enriched: List[FigureItem] = []
    for fig in figures:
        page_idx = max(fig.page - 1, 0)
        if page_idx >= len(doc):
            enriched.append(fig)
            continue
        image_bytes = _page_image_bytes(doc.load_page(page_idx))
        ocr_text = _run_ocr(image_bytes)
        vision_summary = _run_vision_llm(image_bytes, fig.caption)
        enriched.append(
            FigureItem(
                page=fig.page,
                caption=fig.caption,
                ocr_text=ocr_text[:2000],
                vision_summary=vision_summary[:700],
            )
        )
    doc.close()
    return enriched
