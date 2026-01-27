from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from outbreaks.cag.engine import OutbreaksCAG

app = FastAPI(title="Outbreaks CAG")

cag = OutbreaksCAG()
_startup_error: Optional[str] = None


class AskRequest(BaseModel):
    question: str
    region_key: Optional[str] = None


class AskResponse(BaseModel):
    answer: str
    used_region: Optional[str]
    cache_type: str


@app.on_event("startup")
def _startup() -> None:
    global _startup_error
    try:
        base_path = cag.knowledge_dir / "playbooks" / "general.md"
        if not base_path.exists():
            raise FileNotFoundError(f"Missing base knowledge file: {base_path}")
        cag.load_model()
        cag.build_base_cache(base_path.read_text(encoding="utf-8"))
    except Exception as exc:  # pylint: disable=broad-except
        _startup_error = str(exc)


@app.post("/cag/ask", response_model=AskResponse)
def cag_ask(payload: AskRequest) -> AskResponse:
    if _startup_error:
        raise HTTPException(status_code=500, detail=_startup_error)
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question must be non-empty.")

    try:
        answer = cag.ask(payload.question, region_key=payload.region_key)
    except Exception as exc:  # pylint: disable=broad-except
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return AskResponse(
        answer=answer,
        used_region=cag.last_used_region,
        cache_type=cag.last_cache_type,
    )
