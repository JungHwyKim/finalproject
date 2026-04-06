from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class FigureItem:
    page: int
    caption: str
    ocr_text: str = ""
    vision_summary: str = ""


@dataclass
class TableItem:
    page: int
    rows: List[List[str]]


@dataclass
class PaperArtifacts:
    filename: str
    full_text: str
    section_texts: Dict[str, str]
    references: List[str]
    tables: List[TableItem]
    figures: List[FigureItem]


@dataclass
class AnalysisResult:
    summary: str = ""
    key_contributions: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    reproducibility: str = ""
    comparative_analysis: str = ""
    qa_seed_chunks: List[str] = field(default_factory=list)
    raw_notes: Dict[str, str] = field(default_factory=dict)
    error: Optional[str] = None
