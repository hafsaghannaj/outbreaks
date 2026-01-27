from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from outbreaks.cag.engine import OutbreaksCAG
import requests

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


class DonItem(BaseModel):
    title: str
    publication_date: Optional[str] = None
    url: Optional[str] = None


class DonResponse(BaseModel):
    items: list[DonItem]


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


@app.get("/hotspots/don", response_model=DonResponse)
def don_hotspots(limit: int = 8) -> DonResponse:
    if limit < 1 or limit > 25:
        raise HTTPException(status_code=400, detail="Limit must be between 1 and 25.")

    base_url = "https://www.who.int/api/news/diseaseoutbreaknews"
    params = {
        "$top": str(limit),
        "$orderby": "PublicationDate desc",
    }

    try:
        res = requests.get(base_url, params=params, timeout=12)
        res.raise_for_status()
        data = res.json()
        raw_items = data.get("value", []) if isinstance(data, dict) else data
    except Exception as exc:  # pylint: disable=broad-except
        raise HTTPException(status_code=502, detail=f"WHO DON fetch failed: {exc}") from exc

    items: list[DonItem] = []
    for item in raw_items[:limit]:
        title = item.get("Title") or item.get("OverrideTitle") or "Untitled"
        pub_date = item.get("PublicationDate") or item.get("PublicationDateAndTime")
        url = item.get("ItemDefaultUrl")
        items.append(DonItem(title=title, publication_date=pub_date, url=url))

    return DonResponse(items=items)
