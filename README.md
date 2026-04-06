# Multimodal Paper Analyzer
논문 PDF를 업로드하면 텍스트/표/그림/캡션/참고문헌을 함께 분석하고,
LangGraph 멀티 에이전틀 요약부터 Q&A까지 제공하는 프로젝트입니다.

### 이 프로젝트로 할 수 있는 것
1. 논문 요약
2. 핵심 기여점 정리
3. 한계점 정리
4. 재현성 체크
5. 비교 분석
6. RAG 기반 논문 Q&A
7. 그림에 대한 OCR/비전 요약
8. arXiv MCP 기반 유사 논문 자동 추천 + 키워드 재검색

---

### 기술 스택
- Backend: `FastAPI`
- Orchestration: `LangGraph`
- Frontend: `HTML + CSS + Vanilla JS`
- PDF 처리:
  - 텍스트/그림 캡션: `PyMuPDF`
  - 표 추출: `pdfplumber`
- 벡터 DB(RAG): `Chroma` (`langchain-chroma`)
- OCR: `pytesseract` (로컬 Tesseract 엔진 필요)
- Vision(선택): OpenAI Vision 모델
- MCP 연동: `arXiv` 검색

---

### 프로젝트 구조
```text
finalproject/
  app/
    main.py         # FastAPI 엔트리포인트, API 라우트
    graph.py        # LangGraph 멀티 에이전트 오케스트레이션
    pdf_utils.py    # PDF 텍스트/표/캡션/참고문헌 추출
    rag.py          # Chroma 인덱싱 + RAG 질의응답
    vision_utils.py # 그림 OCR + Vision 요약
    arxiv_mcp.py    # arXiv 검색 유틸(유사 논문 추천)
    llm_utils.py    # .env 로드 + LLM 클라이언트 생성
    models.py       # 데이터 모델(dataclass)
  templates/
    index.html      # 웹 화면
  static/
    style.css       # 스타일
    app.js          # 프론트 로직(API 호출/렌더링)
  uploads/          # 업로드된 PDF 저장
  vector_store/     # 문서별 Chroma 인덱스 저장
  requirements.txt
  .env
```

---

### 가상환경 만들기
'''Terminal
python -m venv venv
'''

'''Terminal
.\venv\Scripts\Activate
'''

'''Terminal
pip install -r requirements.txt
'''