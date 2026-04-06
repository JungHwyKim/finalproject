let currentDocId = null;

const uploadForm = document.getElementById("uploadForm");
const statusEl = document.getElementById("status");
const resultEl = document.getElementById("result");
const askBtn = document.getElementById("askBtn");
const answerEl = document.getElementById("answer");
const relatedBtn = document.getElementById("relatedBtn");
const relatedQueryEl = document.getElementById("relatedQuery");
const relatedStatusEl = document.getElementById("relatedStatus");
const relatedResultEl = document.getElementById("relatedResult");

function renderList(title, items) {
  if (!items || items.length === 0) return `<h3>${title}</h3><p>없음</p>`;
  const lis = items.map((x) => `<li>${x}</li>`).join("");
  return `<h3>${title}</h3><ul class="list">${lis}</ul>`;
}

function renderFigureInsights(figures) {
  if (!figures || figures.length === 0) return `<h3>그림 멀티모달 분석 (일부)</h3><p>없음</p>`;
  const blocks = figures
    .map((f) => {
      const ocr = f.ocr_text ? f.ocr_text : "(OCR 텍스트 없음)";
      const vision = f.vision_summary ? f.vision_summary : "(비전 요약 없음)";
      return `
        <li>
          <strong>p.${f.page}</strong> - ${f.caption}<br/>
          <small>OCR: ${ocr}</small><br/>
          <small>Vision: ${vision}</small>
        </li>
      `;
    })
    .join("");
  return `<h3>그림 멀티모달 분석 (일부)</h3><ul class="list">${blocks}</ul>`;
}

function renderRelatedPapers(papers) {
  if (!papers || papers.length === 0) return `<h3>arXiv MCP 유사 논문</h3><p>검색 결과 없음</p>`;
  const items = papers
    .map(
      (p) =>
        `<li><a href="${p.url}" target="_blank" rel="noopener noreferrer">${p.title}</a> (${p.published})<br/><small>${p.summary}</small></li>`,
    )
    .join("");
  return `<h3>arXiv MCP 유사 논문</h3><ul class="list">${items}</ul>`;
}

uploadForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const fileInput = document.getElementById("pdfFile");
  const file = fileInput.files[0];
  if (!file) return;

  statusEl.textContent = "분석 중입니다... PDF 크기에 따라 시간이 걸릴 수 있습니다.";
  resultEl.innerHTML = "";
  answerEl.textContent = "";
  relatedStatusEl.textContent = "";
  relatedResultEl.innerHTML = "";
  currentDocId = null;

  const formData = new FormData();
  formData.append("file", file);

  try {
    const res = await fetch("/analyze", { method: "POST", body: formData });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "분석 실패");
    }
    const data = await res.json();
    currentDocId = data.doc_id;
    statusEl.textContent = `분석 완료: ${data.filename}`;

    resultEl.innerHTML =
      `<h3>요약</h3><p>${data.summary}</p>` +
      renderList("핵심 기여점", data.key_contributions) +
      renderList("한계", data.limitations) +
      `<h3>재현성</h3><pre>${data.reproducibility}</pre>` +
      `<h3>비교 분석</h3><p>${data.comparative_analysis}</p>` +
      renderList("참고문헌 (일부)", data.references) +
      renderFigureInsights(data.figures);
    relatedResultEl.innerHTML = renderRelatedPapers(data.related_papers);
    relatedStatusEl.textContent = data.search_warning || "";
  } catch (err) {
    statusEl.textContent = `오류: ${err.message}`;
  }
});

askBtn.addEventListener("click", async () => {
  const question = document.getElementById("question").value.trim();
  if (!currentDocId) {
    answerEl.textContent = "먼저 PDF를 분석해주세요.";
    return;
  }
  if (!question) return;

  answerEl.textContent = "응답 생성 중...";
  try {
    const res = await fetch("/qa", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ doc_id: currentDocId, question }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Q&A 실패");
    answerEl.textContent = data.answer;
  } catch (err) {
    answerEl.textContent = `오류: ${err.message}`;
  }
});

relatedBtn.addEventListener("click", async () => {
  const query = relatedQueryEl.value.trim();
  if (!currentDocId) {
    relatedStatusEl.textContent = "먼저 PDF를 분석해주세요.";
    return;
  }

  relatedStatusEl.textContent = query ? "arXiv 재검색 중..." : "기본 추천 불러오는 중...";
  try {
    const res = await fetch("/related", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ doc_id: currentDocId, query: query || null }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "관련 논문 검색 실패");
    relatedResultEl.innerHTML = renderRelatedPapers(data.related_papers);
    relatedStatusEl.textContent =
      data.search_warning || (data.related_papers && data.related_papers.length ? "완료" : "완료 (결과 없음)");
  } catch (err) {
    relatedStatusEl.textContent = `오류: ${err.message}`;
  }
});