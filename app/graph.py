from __future__ import annotations

from typing import Any, Dict, List, TypedDict

from langgraph.graph import END, StateGraph

from .llm_utils import get_optional_llm
from .models import AnalysisResult, PaperArtifacts
from .pdf_utils import make_simple_chunks


class GraphState(TypedDict, total=False):
    artifacts: PaperArtifacts
    result: AnalysisResult


def _clip(text: str, n: int = 2200) -> str:
    return text[:n] if len(text) > n else text


def _rule_based_summary(artifacts: PaperArtifacts) -> str:
    intro = artifacts.section_texts.get("introduction", "")
    abstract = artifacts.section_texts.get("abstract", "")
    base = abstract or intro or artifacts.full_text[:2000]
    return f"이 논문은 다음 문제를 다룹니다: {_clip(base, 550)}"


def _call_llm(prompt: str) -> str:
    llm = get_optional_llm()
    if not llm:
        return ""
    return llm.invoke(prompt).content.strip()


def summarize_agent(state: GraphState) -> GraphState:
    artifacts = state["artifacts"]
    prompt = (
        "논문을 5~7문장으로 한국어 요약해줘.\n"
        f"제목(파일명): {artifacts.filename}\n"
        f"본문 일부:\n{_clip(artifacts.full_text, 4500)}"
    )
    summary = _call_llm(prompt) or _rule_based_summary(artifacts)
    result = state.get("result", AnalysisResult())
    result.summary = summary
    return {"artifacts": artifacts, "result": result}


def contribution_agent(state: GraphState) -> GraphState:
    artifacts = state["artifacts"]
    result = state["result"]
    prompt = (
        "다음 논문의 핵심 기여점 3개를 한국어 bullet 스타일로 짧게 작성해줘.\n"
        f"본문 일부:\n{_clip(artifacts.full_text, 4200)}"
    )
    raw = _call_llm(prompt)
    if raw:
        lines = [ln.strip("- ").strip() for ln in raw.splitlines() if ln.strip()]
        result.key_contributions = lines[:3]
    else:
        result.key_contributions = [
            "새로운 방법론 또는 파이프라인 제안",
            "실험을 통한 성능 개선 근거 제시",
            "기존 접근 대비 장단점 분석 제공",
        ]
    return {"artifacts": artifacts, "result": result}


def limitation_agent(state: GraphState) -> GraphState:
    artifacts = state["artifacts"]
    result = state["result"]
    prompt = (
        "논문의 한계점을 3개로 정리해줘. 실험 범위/일반화/비용 관점 포함.\n"
        f"본문 일부:\n{_clip(artifacts.full_text, 4000)}"
    )
    raw = _call_llm(prompt)
    if raw:
        lines = [ln.strip("- ").strip() for ln in raw.splitlines() if ln.strip()]
        result.limitations = lines[:3]
    else:
        result.limitations = [
            "데이터셋 또는 도메인 편향 가능성",
            "추론/학습 비용에 대한 상세 보고 부족",
            "재현을 위한 하이퍼파라미터 정보가 제한적",
        ]
    return {"artifacts": artifacts, "result": result}


def reproducibility_agent(state: GraphState) -> GraphState:
    artifacts = state["artifacts"]
    result = state["result"]
    table_count = len(artifacts.tables)
    fig_count = len(artifacts.figures)
    ref_count = len(artifacts.references)
    prompt = (
        "재현성 관점에서 체크리스트 형태로 평가해줘. (데이터, 코드, 실험설정, 평가지표)\n"
        f"본문 일부:\n{_clip(artifacts.full_text, 3000)}"
    )
    raw = _call_llm(prompt)
    if raw:
        result.reproducibility = raw
    else:
        result.reproducibility = (
            "체크리스트:\n"
            "- 데이터 접근성: 본문에서 확인 필요\n"
            "- 코드 공개 여부: 본문 또는 참고문헌 링크 확인 필요\n"
            "- 실험 설정: 표/그림/섹션 내 하이퍼파라미터 기술 수준 점검\n"
            f"- 보조 단서: 표 {table_count}개, 그림/캡션 {fig_count}개, 참고문헌 {ref_count}개"
        )
    return {"artifacts": artifacts, "result": result}


def comparative_agent(state: GraphState) -> GraphState:
    artifacts = state["artifacts"]
    result = state["result"]
    table_preview = ""
    if artifacts.tables:
        first = artifacts.tables[0]
        rows = [" | ".join(r) for r in first.rows[:4]]
        table_preview = "\n".join(rows)
    prompt = (
        "비교 분석 관점(기준방법 대비 개선/손해, 공정성, 비교 설정)을 1단락으로 작성해줘.\n"
        f"표 예시:\n{table_preview}\n"
        f"본문 일부:\n{_clip(artifacts.full_text, 2500)}"
    )
    raw = _call_llm(prompt)
    result.comparative_analysis = (
        raw
        or "비교 표에서 제안 방법의 성능 우위를 주장하지만, 동일 조건 비교 여부와 통계적 유의성은 추가 확인이 필요합니다."
    )
    return {"artifacts": artifacts, "result": result}


def qa_seed_agent(state: GraphState) -> GraphState:
    artifacts = state["artifacts"]
    result = state["result"]
    result.qa_seed_chunks = make_simple_chunks(artifacts.full_text)
    result.raw_notes["references_preview"] = "\n".join(artifacts.references[:10])
    return {"artifacts": artifacts, "result": result}


def build_analysis_graph():
    graph = StateGraph(GraphState)
    graph.add_node("summary", summarize_agent)
    graph.add_node("contribution", contribution_agent)
    graph.add_node("limitation", limitation_agent)
    graph.add_node("reproducibility", reproducibility_agent)
    graph.add_node("comparative", comparative_agent)
    graph.add_node("qa_seed", qa_seed_agent)

    graph.set_entry_point("summary")
    graph.add_edge("summary", "contribution")
    graph.add_edge("contribution", "limitation")
    graph.add_edge("limitation", "reproducibility")
    graph.add_edge("reproducibility", "comparative")
    graph.add_edge("comparative", "qa_seed")
    graph.add_edge("qa_seed", END)
    return graph.compile()
