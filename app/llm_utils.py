from __future__ import annotations

import os
from typing import Optional

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv


# Load .env in project root once at import time.
load_dotenv()


def get_optional_llm() -> Optional[ChatOpenAI]:
    """
    Return an LLM client only when OPENAI_API_KEY is configured.
    This keeps the toy project runnable without external credentials.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    return ChatOpenAI(model=model, temperature=0.2)
