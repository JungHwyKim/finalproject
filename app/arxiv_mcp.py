from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Tuple

import arxiv

logger = logging.getLogger(__name__)


def _safe_keywords(raw: str, max_terms: int = 6) -> List[str]:
    """
    Keep only short ASCII-ish keyword tokens for arXiv query stability.
    """
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-_.]{1,40}", raw.lower())
    # Drop overly generic filler tokens.
    stop = {"and", "the", "this", "that", "with", "from", "using", "paper"}
    uniq: List[str] = []
    for t in tokens:
        if t in stop:
            continue
        if t not in uniq:
            uniq.append(t)
        if len(uniq) >= max_terms:
            break
    return uniq


def _build_arxiv_query(raw: str) -> str:
    """
    arXiv API는 필드 접두어가 없으면 기대와 다르게 매칭이 약할 수 있어,
    단순 키워드는 all: 접두어를 붙입니다. 사용자가 au:/ti:/abs: 등을 쓰면 그대로 둡니다.
    """
    q = " ".join(raw.split())
    if not q:
        return q
    if ":" in q:
        return q
    parts = _safe_keywords(q)
    if len(parts) == 1:
        return f"all:{parts[0]}"
    if not parts:
        return ""
    return " AND ".join(f"all:{p}" for p in parts)


def _result_to_dict(result: arxiv.Result) -> Dict[str, str]:
    title = (getattr(result, "title", None) or "").strip()
    summary = (getattr(result, "summary", None) or "").strip().replace("\n", " ")[:400]
    published = ""
    p = getattr(result, "published", None)
    if p is not None:
        try:
            published = str(p.date())
        except Exception:
            published = str(p)
    url = getattr(result, "entry_id", None) or getattr(result, "pdf_url", None) or ""
    return {
        "title": title,
        "published": published,
        "summary": summary,
        "url": str(url),
    }


def search_related_arxiv(
    query: str, max_results: int = 5
) -> Tuple[List[Dict[str, str]], Optional[str]]:
    """
    arXiv에서 관련 논문을 검색합니다.
    반환: (논문 목록, 오류/안내 메시지). 네트워크·파싱 오류 시 목록은 비고 메시지에 이유를 담습니다.
    """
    if not query.strip():
        return [], None

    api_query = _build_arxiv_query(query)
    if not api_query:
        return (
            [],
            "검색어가 너무 일반적이거나 유효한 키워드가 없습니다. 영어 키워드 2~4개로 입력해보세요.",
        )
    papers: List[Dict[str, str]] = []

    def _run_search(q: str) -> List[Dict[str, str]]:
        found: List[Dict[str, str]] = []
        search = arxiv.Search(
            query=q,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        client = arxiv.Client(
            page_size=min(50, max_results * 2),
            delay_seconds=1.0,
            num_retries=3,
        )
        for result in client.results(search):
            found.append(_result_to_dict(result))
            if len(found) >= max_results:
                break
        return found

    try:
        papers = _run_search(api_query)
    except Exception as e:
        # Retry once with simplified OR query on HTTP 5xx style failures.
        terms = _safe_keywords(query, max_terms=4)
        fallback_query = " OR ".join(f"all:{t}" for t in terms) if terms else ""
        if fallback_query and fallback_query != api_query:
            try:
                papers = _run_search(fallback_query)
                if papers:
                    return (
                        papers,
                        f"원본 쿼리가 길어 단순화된 쿼리({fallback_query})로 재시도했습니다.",
                    )
            except Exception as e2:
                logger.warning(
                    "arXiv fallback failed query=%r api_query=%r fallback=%r: %s",
                    query,
                    api_query,
                    fallback_query,
                    e2,
                )
        logger.warning(
            "arXiv search failed query=%r api_query=%r: %s", query, api_query, e
        )
        return [], f"arXiv 검색 실패: {e} (검색어를 영어 키워드 2~4개로 줄여보세요.)"

    if not papers:
        return [], (
            f"arXiv에서 결과가 없습니다. 쿼리: {api_query!r} "
            "(네트워크/방화벽이면 위쪽 오류가 로그에 남을 수 있습니다. "
            "또는 ti:키워드, abs:키워드 형식으로 시도해 보세요.)"
        )

    return papers, None
