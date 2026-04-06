from __future__ import annotations

import hashlib
from pathlib import Path
from typing import List

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

from .llm_utils import get_optional_llm
from .models import PaperArtifacts
from .pdf_utils import make_simple_chunks


VECTOR_DIR = Path(__file__).resolve().parent.parent / "vector_store"
VECTOR_DIR.mkdir(exist_ok=True)


class HashEmbeddings(Embeddings):
    """
    Lightweight deterministic embedding fallback.
    Keeps the toy project runnable without external API keys.
    """

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def _embed(self, text: str) -> List[float]:
        vec = np.zeros(self.dim, dtype=np.float32)
        for token in text.lower().split():
            h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dim
            vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)


def get_embeddings() -> Embeddings:
    if get_optional_llm() is not None:
        return OpenAIEmbeddings(model="text-embedding-3-small")
    return HashEmbeddings()


def build_documents(artifacts: PaperArtifacts) -> List[Document]:
    docs: List[Document] = []
    chunks = make_simple_chunks(artifacts.full_text, chunk_size=700)
    for i, chunk in enumerate(chunks):
        docs.append(
            Document(
                page_content=chunk,
                metadata={"type": "text", "chunk": i, "filename": artifacts.filename},
            )
        )

    for idx, fig in enumerate(artifacts.figures):
        vision_part = (
            f"\nVision summary: {fig.vision_summary}" if fig.vision_summary else ""
        )
        ocr_part = f"\nOCR text: {fig.ocr_text[:1200]}" if fig.ocr_text else ""
        docs.append(
            Document(
                page_content=f"Figure caption: {fig.caption}{vision_part}{ocr_part}",
                metadata={"type": "figure", "index": idx, "page": fig.page},
            )
        )

    for idx, ref in enumerate(artifacts.references[:120]):
        docs.append(
            Document(
                page_content=f"Reference item: {ref}",
                metadata={"type": "reference", "index": idx},
            )
        )

    for idx, table in enumerate(artifacts.tables[:50]):
        rows = [" | ".join(row) for row in table.rows[:8]]
        docs.append(
            Document(
                page_content="Table content:\n" + "\n".join(rows),
                metadata={"type": "table", "index": idx, "page": table.page},
            )
        )
    return docs


def build_or_replace_doc_index(doc_id: str, artifacts: PaperArtifacts) -> None:
    persist_dir = VECTOR_DIR / doc_id
    embedding = get_embeddings()
    vectorstore = Chroma(
        collection_name=f"paper_{doc_id}",
        embedding_function=embedding,
        persist_directory=str(persist_dir),
    )
    existing = vectorstore.get()
    if existing.get("ids"):
        vectorstore.delete(ids=existing["ids"])
    docs = build_documents(artifacts)
    if docs:
        vectorstore.add_documents(docs)


def answer_with_rag(doc_id: str, question: str) -> str:
    persist_dir = VECTOR_DIR / doc_id
    if not persist_dir.exists():
        raise ValueError("인덱스를 찾을 수 없습니다. 먼저 analyze를 실행하세요.")
    vectorstore = Chroma(
        collection_name=f"paper_{doc_id}",
        embedding_function=get_embeddings(),
        persist_directory=str(persist_dir),
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    docs = retriever.invoke(question)
    if not docs:
        return "검색된 근거가 없어 답변을 생성하지 못했습니다."

    context = "\n\n".join(
        [f"[근거{i+1}] {d.page_content[:700]}" for i, d in enumerate(docs)]
    )
    llm = get_optional_llm()
    if llm:
        prompt = (
            "아래는 논문에서 검색된 근거입니다. 근거 기반으로만 한국어로 답변하세요.\n"
            "근거가 불충분하면 불충분하다고 명시하세요.\n\n"
            f"질문: {question}\n\n"
            f"{context}"
        )
        return llm.invoke(prompt).content.strip()

    return (
        "LLM 키가 없어 추출형 RAG 응답을 제공합니다.\n\n"
        f"질문: {question}\n\n"
        "검색 근거:\n" + "\n".join([f"- {d.page_content[:260]}" for d in docs])
    )
