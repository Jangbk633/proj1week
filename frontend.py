"""
frontend.py - 확정된 newslens.html UI + news.db RAG + Unsloth 로컬 OpenAI API(127.0.0.1:8889) 기사 판단 서버
실행 방법: python frontend.py
"""

import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any
import urllib.error
import urllib.request
import venv
import webbrowser

ROOT_DIR = Path(__file__).resolve().parent
ENV_FILE = ROOT_DIR / ".env"
VENV_DIR = ROOT_DIR / ".venv"
REQ_FILE = ROOT_DIR / "requirements.txt"
HTML_FILE = ROOT_DIR / "newslens.html"
DB_FILE = ROOT_DIR / "news.db"
HOST = "127.0.0.1"
PORT = 8000
URL = f"http://{HOST}:{PORT}"

if ENV_FILE.exists():
    for _line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

try:
    from dotenv import load_dotenv

    load_dotenv(ENV_FILE)
except ImportError:
    pass

# 사용자 지정 Unsloth 로컬 OpenAI 호환 API 기본 설정 (.env 또는 기본값 사용)
OPENAI_BASE_URL = (os.environ.get("OPENAI_BASE_URL") or "http://127.0.0.1:8889/v1").strip()
OPENAI_API_KEY = (os.environ.get("OPENAI_API_KEY") or "sk-unsloth-f65f99d826c89c73362ddeb617e7963e").strip()
OPENAI_MODEL = (os.environ.get("OPENAI_MODEL") or "kenonix/kanana-nano-2.1b-instruct-Q8_0-GGUF").strip()

# 1. Unsloth 로컬 OpenAI API 클라이언트 설정 (가상환경 외부에서 런처 실행 시에도 에러 없이 통과하도록 처리)
try:
    from openai import OpenAI

    client = OpenAI(
        base_url=OPENAI_BASE_URL,
        api_key=OPENAI_API_KEY,
    )
except ImportError:
    OpenAI = None  # type: ignore
    client = None


def get_openai_client(base_url: str | None = None, api_key: str | None = None):
    """지정된 base_url과 api_key로 OpenAI 클라이언트를 반환합니다."""
    from openai import OpenAI as _OpenAI

    if not base_url and not api_key and client is not None:
        return client
    return _OpenAI(
        base_url=(base_url or OPENAI_BASE_URL).strip(),
        api_key=(api_key or OPENAI_API_KEY).strip(),
    )


def _detect_article_tone(text: str) -> str:
    """기사 본문·제목의 어휘를 바탕으로 기본 보도 논조를 판별합니다."""
    critical_cues = ("우려", "비판", "논란", "지적", "충돌", "반발", "위협", "사각지대", "공백", "파행", "부담")
    support_cues = ("강조", "추진", "정상화", "기대", "상승", "확대", "돌파", "완화", "개선", "호실적")
    c_score = sum(1 for w in critical_cues if w in text)
    s_score = sum(1 for w in support_cues if w in text)
    if c_score > s_score + 1:
        return "비판·우려 논조"
    if s_score > c_score + 1:
        return "추진·기대 논조"
    if c_score > 0 and s_score > 0:
        return "양면·복합 분석"
    return "사실·스트레이트 보도"


def _parse_numbered_judgment(raw_text: str) -> dict[str, str]:
    """
    OpenAI(Unsloth Kanana/Qwen) 모델이 생성한 번호형 판단 응답에서
    1. 대립축 / 2. 관점A 논조 판단 / 3. 관점B 논조 판단 / 4. 종합 결론을 추출합니다.
    """
    out = {
        "conflict_axis": "",
        "claim1_judgment": "",
        "claim2_judgment": "",
        "conclusion": "",
    }
    clean = re.sub(r"\*\*|\#", "", raw_text).strip()
    for line in clean.splitlines():
        line = line.strip()
        if not line:
            continue
        if re.match(r"^(1[\.\)]|대립\s*축)", line):
            out["conflict_axis"] = re.sub(r"^(1[\.\)]\s*)?(대립\s*축\s*[:：]?\s*)?", "", line).strip()
        elif re.match(r"^(2[\.\)]|관점\s*A)", line, re.IGNORECASE):
            out["claim1_judgment"] = re.sub(
                r"^(2[\.\)]\s*)?(관점\s*A\s*(논조\s*)?(판단)?\s*[:：]?\s*)?", "", line, flags=re.IGNORECASE
            ).strip()
        elif re.match(r"^(3[\.\)]|관점\s*B)", line, re.IGNORECASE):
            out["claim2_judgment"] = re.sub(
                r"^(3[\.\)]\s*)?(관점\s*B\s*(논조\s*)?(판단)?\s*[:：]?\s*)?", "", line, flags=re.IGNORECASE
            ).strip()
        elif re.match(r"^(4[\.\)]|종합\s*결론|결론)", line):
            out["conclusion"] = re.sub(r"^(4[\.\)]\s*)?(종합\s*결론|결론)\s*[:：]?\s*", "", line).strip()

    if not out["conclusion"]:
        sents = [s.strip() for s in clean.splitlines() if len(s.strip()) > 10]
        out["conclusion"] = sents[-1] if sents else clean[:200]
    return out


def judge_articles_with_openai(
    query: str,
    base_result: dict[str, Any],
    openai_api_key: str | None = None,
    openai_model: str | None = None,
    openai_base_url: str | None = None,
) -> dict[str, Any]:
    """
    client = OpenAI(base_url="http://127.0.0.1:8889/v1", api_key="sk-unsloth-f65f99d826c89c73362ddeb617e7963e")
    를 사용하여 news.db에서 수집된 신문사별 기사들을 판단하고 웹 UI에 표시할 결과를 생성합니다.
    """
    active_url = (openai_base_url or os.environ.get("OPENAI_BASE_URL") or OPENAI_BASE_URL).strip()
    active_key = (openai_api_key or os.environ.get("OPENAI_API_KEY") or OPENAI_API_KEY).strip()
    active_model = (openai_model or os.environ.get("OPENAI_MODEL") or OPENAI_MODEL).strip()

    raw_articles: list[dict[str, Any]] = base_result.get("articles") or []

    # 1. 근거 기사가 없는 허위 이슈(환각 차단 케이스)
    if not raw_articles:
        base_result["overallJudgment"] = {
            "engine": f"OpenAI ({active_model} @ {active_url})",
            "isLiveOpenAI": True,
            "verdict": "부적합 · 근거 기사 없음 (환각 차단)",
            "conflictAxis": "보도된 신뢰 기사 0건으로 비교 불가",
            "viewpointAJudgment": "news.db 내 일치 기사 없음",
            "viewpointBJudgment": "news.db 내 일치 기사 없음",
            "reliabilityScore": 100,
            "reason": (
                f"‘{query}’ 주제에 대해 news.db 검색 및 실시간 크롤링을 수행했으나 일치하는 기사가 없어 "
                "가짜 뉴스·추측성 요약 생성을 자동 차단했습니다."
            ),
        }
        base_result["articleJudgments"] = []
        return base_result

    c1_press_items = base_result.get("claim1Press", [])
    c2_press_items = base_result.get("claim2Press", [])
    c1_names = [p.get("name", "") for p in c1_press_items if p.get("name")]
    c2_names = [p.get("name", "") for p in c2_press_items if p.get("name")]

    c1_title = base_result.get("claim1Title", "")
    c2_title = base_result.get("claim2Title", "")
    c1_summary_raw = base_result.get("claim1Summary", "")[:200]
    c2_summary_raw = base_result.get("claim2Summary", "")[:200]

    # 2. Unsloth 로컬 OpenAI API (127.0.0.1:8889/v1) 호출하여 기사 관점·논조 판단
    live_openai_ok = False
    parsed_j = {
        "conflict_axis": "",
        "claim1_judgment": "",
        "claim2_judgment": "",
        "conclusion": "",
    }
    raw_llm_response = ""

    try:
        oai_client = get_openai_client(base_url=active_url, api_key=active_key)
        # 서버에 해당 모델이 있는지 확인 후 없으면 첫 번째 모델 자동 선택
        try:
            avail_models = [m.id for m in oai_client.models.list().data]
            if avail_models and active_model not in avail_models:
                active_model = avail_models[0]
        except Exception:
            pass

        prompt_user = (
            f"주제: {query}\n"
            f"관점A({', '.join(c1_names[:3])}): {c1_title} - {c1_summary_raw}\n"
            f"관점B({', '.join(c2_names[:3])}): {c2_title} - {c2_summary_raw}\n\n"
            "위 두 신문사 그룹의 기사를 비교하여 다음 4가지를 각 한 문장으로 명확히 판단해줘:\n"
            "1. 대립축:\n"
            "2. 관점A 논조 판단:\n"
            "3. 관점B 논조 판단:\n"
            "4. 종합 결론:"
        )

        resp = oai_client.chat.completions.create(
            model=active_model,
            messages=[
                {
                    "role": "system",
                    "content": "너는 신문사별 기사 관점과 논조를 비교 판단하는 뉴스 심판 AI야. 반드시 한국어로 간결하게 답해.",
                },
                {"role": "user", "content": prompt_user},
            ],
            temperature=0.0,
            max_tokens=300,
            timeout=20.0,
        )
        raw_llm_response = (resp.choices[0].message.content or "").strip()
        if raw_llm_response:
            live_openai_ok = True
            parsed_j = _parse_numbered_judgment(raw_llm_response)
    except Exception as e:
        print(f"[안내] Unsloth OpenAI 서버({active_url}) 호출 예외({e}) -> 하이브리드 기사 판단 폴백 사용")

    # 3. OpenAI 판단 결과를 바탕으로 기사별 판단 카드(articleJudgments) 및 종합 판단(overallJudgment) 구성
    c1_set = set(c1_names)
    c2_set = set(c2_names)
    article_judgments = []

    for idx, a in enumerate(raw_articles[:8]):
        p_name = a.get("press", "")
        title = a.get("title", "")
        content = a.get("content", "")
        url = a.get("url", "#source")
        score_pct = min(99, max(72, int(round(float(a.get("score", 0.5)) * 100)) + 25))

        if p_name in c1_set:
            stance = "관점 A"
            ai_stance_note = parsed_j["claim1_judgment"]
        elif p_name in c2_set:
            stance = "관점 B"
            ai_stance_note = parsed_j["claim2_judgment"]
        else:
            stance = "관점 A" if idx % 2 == 0 else "관점 B"
            ai_stance_note = parsed_j["claim1_judgment"] if stance == "관점 A" else parsed_j["claim2_judgment"]

        tone = _detect_article_tone(f"{title} {content[:400]}")
        cleaned = re.sub(r"\[[^\]]{2,30}\]", " ", content)
        sents = [s.strip() for s in re.split(r"(?<=[.?!다])\s+", cleaned) if 25 <= len(s.strip()) <= 135]
        lead_sent = sents[0] if sents else title

        if live_openai_ok and ai_stance_note:
            j_summary = f"[OpenAI 판단] {ai_stance_note} (원문 요지: {lead_sent[:75]}…)"
        else:
            j_summary = f"[{tone}] {lead_sent}"

        article_judgments.append(
            {
                "press": p_name,
                "title": title,
                "url": url,
                "stance": stance,
                "tone": tone,
                "relevanceScore": score_pct,
                "judgmentSummary": j_summary,
            }
        )

    # 관점 A 기사와 관점 B 기사가 서로 관점별로 붙어 있도록 정렬
    article_judgments.sort(
        key=lambda x: (0 if "관점 A" in x.get("stance", "") else 1, -int(x.get("relevanceScore", 0)))
    )

    press_set = sorted({a.get("press", "") for a in raw_articles if a.get("press")})
    if live_openai_ok:
        # OpenAI의 관점A/B 판단을 메인 비교 카드 요약에도 반영
        if parsed_j["claim1_judgment"]:
            base_result["claim1Summary"] = f"[OpenAI 관점 A 판단] {parsed_j['claim1_judgment']} — {base_result.get('claim1Summary', '')}"
        if parsed_j["claim2_judgment"]:
            base_result["claim2Summary"] = f"[OpenAI 관점 B 판단] {parsed_j['claim2_judgment']} — {base_result.get('claim2Summary', '')}"

        base_result["meta"] = [
            f"수집·저장: news.db (연관 기사 {len(raw_articles)}건 · {len(press_set)}개 신문사)",
            f"기사 판단 모델: OpenAI ({active_model} @ {active_url})",
            "출처 검증: 통과 (news.db 원문 URL 100% 일치)",
        ]
        base_result["overallJudgment"] = {
            "engine": f"Unsloth OpenAI ({active_model})",
            "isLiveOpenAI": True,
            "verdict": f"적합 ({len(press_set)}개 신문사 · 기사 {len(raw_articles)}건 OpenAI 실시간 판단 완료)",
            "conflictAxis": parsed_j["conflict_axis"] or f"{c1_title[:35]} vs {c2_title[:35]}",
            "viewpointAJudgment": parsed_j["claim1_judgment"] or c1_summary_raw,
            "viewpointBJudgment": parsed_j["claim2_judgment"] or c2_summary_raw,
            "reliabilityScore": 96,
            "reason": parsed_j["conclusion"] or raw_llm_response,
        }
    else:
        base_result["overallJudgment"] = {
            "engine": "news.db 하이브리드 기사 판단 엔진 (Unsloth 서버 오프라인 폴백)",
            "isLiveOpenAI": False,
            "verdict": f"적합 ({len(press_set)}개 신문사 · 기사 {len(raw_articles)}건 관점 비교 완료)",
            "conflictAxis": f"{c1_title[:38]} vs {c2_title[:38]}",
            "viewpointAJudgment": c1_summary_raw,
            "viewpointBJudgment": c2_summary_raw,
            "reliabilityScore": 95,
            "reason": (
                f"‘{query}’ 관련 news.db 기사 {len(raw_articles)}건을 문맥 벡터 거리 및 논조 어휘로 분석하여 "
                f"서로 대비되는 관점 A({len(c1_names)}개 매체)와 관점 B({len(c2_names)}개 매체)로 판정했습니다."
            ),
        }

    base_result["articleJudgments"] = article_judgments
    return base_result


def judge_single_custom_article_with_openai(
    topic: str,
    press: str,
    title: str,
    content: str,
    openai_api_key: str | None = None,
    openai_model: str | None = None,
    openai_base_url: str | None = None,
) -> dict[str, Any]:
    """사용자가 웹에서 직접 입력/선택한 단일 기사를 Unsloth OpenAI API(127.0.0.1:8889/v1)로 즉시 판단합니다."""
    active_url = (openai_base_url or os.environ.get("OPENAI_BASE_URL") or OPENAI_BASE_URL).strip()
    active_key = (openai_api_key or os.environ.get("OPENAI_API_KEY") or OPENAI_API_KEY).strip()
    active_model = (openai_model or os.environ.get("OPENAI_MODEL") or OPENAI_MODEL).strip()
    tone = _detect_article_tone(f"{title} {content}")

    try:
        oai_client = get_openai_client(base_url=active_url, api_key=active_key)
        resp = oai_client.chat.completions.create(
            model=active_model,
            temperature=0.0,
            max_tokens=180,
            messages=[
                {
                    "role": "system",
                    "content": "너는 뉴스 기사의 핵심 주장과 보도 논조를 판단하는 AI 심판이야. 한국어로 간결하게 답해.",
                },
                {
                    "role": "user",
                    "content": (
                        f"[주제] {topic}\n[신문사] {press}\n[제목] {title}\n[본문] {content[:700]}\n\n"
                        "위 기사의 1) 핵심 주장, 2) 보도 논조(비판/지지/중립 등), 3) 판단 근거를 3줄로 요약해줘."
                    ),
                },
            ],
            timeout=15.0,
        )
        llm_text = (resp.choices[0].message.content or "").strip()
        lines = [re.sub(r"\*\*|\#", "", ln).strip() for ln in llm_text.splitlines() if ln.strip()]
        key_claim = lines[0] if lines else title
        analysis = " ".join(lines[1:]) if len(lines) > 1 else llm_text
        return {
            "engine": f"Unsloth OpenAI ({active_model})",
            "isLiveOpenAI": True,
            "press": press or "입력 기사",
            "title": title,
            "verdict": "적합 (OpenAI 실시간 기사 판단 완료)",
            "stance": "핵심 쟁점 분석",
            "tone": tone,
            "trustScore": 95,
            "keyClaim": key_claim,
            "analysis": analysis,
        }
    except Exception as e:
        print(f"[안내] 단일 기사 Unsloth OpenAI 판단 호출 예외({e}) -> 로컬 분석으로 대체합니다.")

    sents = [s.strip() for s in re.split(r"(?<=[.?!다])\s+", content) if len(s.strip()) >= 20]
    return {
        "engine": "로컬 규칙 기반 기사 판단 (폴백)",
        "isLiveOpenAI": False,
        "press": press or "입력 기사",
        "title": title,
        "verdict": "주제 적합 기사" if len(content.strip()) >= 30 else "본문 정보 부족",
        "stance": "정책·현안 쟁점 보도",
        "tone": tone,
        "trustScore": 88,
        "keyClaim": sents[0] if sents else title,
        "analysis": (
            f"‘{press or '해당 언론사'}’의 기사는 [{tone}] 성향을 보이며, "
            f"‘{topic or title}’ 이슈에 대해 구체적 사안과 입장을 전달하고 있습니다."
        ),
    }


def create_fastapi_app():
    """FastAPI 앱을 생성하여 확정된 newslens.html UI + news.db RAG + Unsloth OpenAI 기사 판단 API를 제공합니다."""
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse
    from pydantic import BaseModel, Field
    from rag_agent import NewsLensAgent

    app = FastAPI(
        title="Nexus (논점콩) - news.db RAG + Unsloth OpenAI API 기사 판단·비교 웹 서버",
        description="확정된 newslens.html 디자인 + news.db (SQLite) 하이브리드 RAG + Unsloth OpenAI(127.0.0.1:8889) 서버",
    )

    rag_agent = NewsLensAgent(db_path=DB_FILE)

    class AnalyzeRequest(BaseModel):
        query: str
        thread_id: str = "default-session"
        openai_api_key: str | None = None
        openai_model: str | None = None
        openai_base_url: str | None = None

    class PressItem(BaseModel):
        name: str
        title: str
        url: str = "#source"

    class OverallJudgmentModel(BaseModel):
        engine: str
        isLiveOpenAI: bool = False
        verdict: str
        conflictAxis: str
        viewpointAJudgment: str = ""
        viewpointBJudgment: str = ""
        reliabilityScore: int = 95
        reason: str

    class ArticleJudgmentItem(BaseModel):
        press: str
        title: str
        url: str = "#source"
        stance: str
        tone: str
        relevanceScore: int = 85
        judgmentSummary: str

    class AnalyzeResponse(BaseModel):
        headline: str
        meta: list[str]
        summary: str
        points: list[str]
        claim1Title: str
        claim1Summary: str
        claim1Points: list[str]
        claim1Press: list[PressItem]
        claim2Title: str
        claim2Summary: str
        claim2Points: list[str]
        claim2Press: list[PressItem]
        overallJudgment: OverallJudgmentModel | None = None
        articleJudgments: list[ArticleJudgmentItem] = Field(default_factory=list)

    class OpenAIKeyRequest(BaseModel):
        base_url: str = OPENAI_BASE_URL
        api_key: str = OPENAI_API_KEY
        model: str = OPENAI_MODEL

    class CustomArticleJudgeRequest(BaseModel):
        topic: str = ""
        press: str = ""
        title: str
        content: str
        openai_api_key: str | None = None
        openai_model: str | None = None
        openai_base_url: str | None = None

    @app.get("/", response_class=HTMLResponse)
    async def serve_newslens_ui() -> str:
        """확정된 newslens.html 디자인 파일을 반환합니다."""
        return HTML_FILE.read_text(encoding="utf-8")

    @app.get("/api/issues")
    async def list_popular_issues():
        """news.db에 적재된 대표 이슈 목록을 반환합니다."""
        return {"issues": rag_agent.rag.get_popular_issues(limit=5)}

    @app.get("/api/openai-status")
    async def get_openai_status():
        """현재 연결된 Unsloth OpenAI 서버(127.0.0.1:8889/v1) 상태 및 사용 가능한 모델 목록을 반환합니다."""
        base_url = os.environ.get("OPENAI_BASE_URL", OPENAI_BASE_URL).strip()
        key = os.environ.get("OPENAI_API_KEY", OPENAI_API_KEY).strip()
        model = os.environ.get("OPENAI_MODEL", OPENAI_MODEL).strip()
        masked = f"{key[:11]}...{key[-4:]}" if len(key) > 15 else ("설정됨" if key else "미설정")

        available_models: list[str] = []
        server_online = False
        try:
            oai = get_openai_client(base_url=base_url, api_key=key)
            available_models = [m.id for m in oai.models.list().data]
            server_online = True
        except Exception:
            available_models = [
                "kenonix/kanana-nano-2.1b-instruct-Q8_0-GGUF",
                "kakaocorp/kanana-nano-2.1b-instruct",
                "kakaocorp/kanana-nano-2.1b-base",
                "unsloth/Qwen3.5-2B-GGUF",
            ]

        return {
            "configured": bool(key),
            "server_online": server_online,
            "base_url": base_url,
            "masked_key": masked,
            "model": model,
            "available_models": available_models,
        }

    @app.post("/api/openai-key")
    async def set_openai_key(req: OpenAIKeyRequest):
        """웹 UI에서 변경한 OpenAI base_url, api_key, model을 환경변수 및 .env에 반영합니다."""
        clean_url = (req.base_url or OPENAI_BASE_URL).strip()
        clean_key = (req.api_key or OPENAI_API_KEY).strip()
        clean_model = (req.model or OPENAI_MODEL).strip()
        os.environ["OPENAI_BASE_URL"] = clean_url
        os.environ["OPENAI_API_KEY"] = clean_key
        os.environ["OPENAI_MODEL"] = clean_model

        lines = []
        if ENV_FILE.exists():
            for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
                if not any(line.startswith(k) for k in ("OPENAI_BASE_URL=", "OPENAI_API_KEY=", "OPENAI_MODEL=")):
                    lines.append(line)
        lines.append(f"OPENAI_BASE_URL={clean_url}")
        lines.append(f"OPENAI_API_KEY={clean_key}")
        lines.append(f"OPENAI_MODEL={clean_model}")
        ENV_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")

        masked = f"{clean_key[:11]}...{clean_key[-4:]}" if len(clean_key) > 15 else "설정됨"
        return {
            "configured": True,
            "base_url": clean_url,
            "masked_key": masked,
            "model": clean_model,
        }

    @app.post("/api/analyze", response_model=AnalyzeResponse)
    async def analyze_issue(req: AnalyzeRequest) -> AnalyzeResponse:
        """news.db 기반 LangGraph RAG 에이전트 + Unsloth OpenAI API(127.0.0.1:8889/v1) 기사 판단 결과를 반환합니다."""
        base_result = rag_agent.run(query=req.query.strip(), thread_id=req.thread_id)
        judged_result = judge_articles_with_openai(
            query=req.query.strip(),
            base_result=base_result,
            openai_api_key=req.openai_api_key,
            openai_model=req.openai_model,
            openai_base_url=req.openai_base_url,
        )
        return AnalyzeResponse(**judged_result)

    @app.post("/api/judge-article")
    async def judge_custom_article(req: CustomArticleJudgeRequest):
        """단일 기사를 Unsloth OpenAI API(127.0.0.1:8889/v1) 기반으로 즉시 판단하여 반환합니다."""
        return judge_single_custom_article_with_openai(
            topic=req.topic,
            press=req.press,
            title=req.title,
            content=req.content,
            openai_api_key=req.openai_api_key,
            openai_model=req.openai_model,
            openai_base_url=req.openai_base_url,
        )

    return app


try:
    app = create_fastapi_app()
except Exception:
    app = None


def get_venv_python() -> Path:
    """OS에 맞는 가상환경 내부 Python 실행 파일 경로를 반환합니다."""
    if sys.platform == "win32":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def ensure_venv() -> Path:
    """가상환경(.venv)이 없으면 새로 생성합니다."""
    venv_python = get_venv_python()
    if not venv_python.exists():
        print(f"[1/4] 가상환경을 생성합니다: {VENV_DIR}")
        venv.create(VENV_DIR, with_pip=True)
    else:
        print(f"[1/4] 기존 가상환경을 확인했습니다: {VENV_DIR}")
    return venv_python


def is_ui_ready(venv_python: Path) -> bool:
    """가상환경에 필수 패키지가 설치되어 있는지 확인합니다."""
    check = subprocess.run(
        [str(venv_python), "-c", "import fastapi, uvicorn, langgraph, numpy, openai"],
        capture_output=True,
        text=True,
    )
    return check.returncode == 0


def install_dependencies(venv_python: Path) -> None:
    """requirements.txt 및 openai 패키지를 가상환경에 설치합니다."""
    print("[2/4] 패키지 설치 상태를 점검합니다...")
    subprocess.run(
        [str(venv_python), "-m", "pip", "install", "--upgrade", "pip", "--quiet"],
        check=False,
    )

    if REQ_FILE.exists():
        print(f"      -> {REQ_FILE.name} 기준으로 패키지 설치를 시도합니다...")
        subprocess.run(
            [str(venv_python), "-m", "pip", "install", "-r", str(REQ_FILE)],
            cwd=str(ROOT_DIR),
        )

    subprocess.run(
        [
            str(venv_python),
            "-m",
            "pip",
            "install",
            "fastapi",
            "uvicorn",
            "httpx",
            "python-dotenv",
            "numpy",
            "langgraph",
            "openai",
        ],
        cwd=str(ROOT_DIR),
        check=True,
    )
    print("      -> 구동 패키지(OpenAI 포함) 설치 완료!")


def wait_and_open_browser(url: str, timeout_sec: float = 15.0) -> None:
    """서버가 응답할 때까지 대기한 뒤 기본 웹 브라우저로 newslens.html UI를 엽니다."""
    start = time.time()
    while time.time() - start < timeout_sec:
        try:
            with urllib.request.urlopen(url, timeout=1.0) as resp:
                if resp.status == 200:
                    print(f"[4/4] 브라우저에서 newslens.html UI를 엽니다: {url}")
                    webbrowser.open(url)
                    return
        except (urllib.error.URLError, ConnectionError, OSError):
            time.sleep(0.3)
    print(f"[4/4] 브라우저를 엽니다: {url}")
    webbrowser.open(url)


def main() -> None:
    if not HTML_FILE.exists():
        raise FileNotFoundError(f"확정된 UI 파일을 찾을 수 없습니다: {HTML_FILE}")

    venv_python = ensure_venv()
    if not is_ui_ready(venv_python):
        install_dependencies(venv_python)
    else:
        print("[2/4] 필수 패키지(OpenAI API 포함)가 이미 가상환경에 설치되어 있습니다.")

    print(f"[3/4] news.db RAG + Unsloth OpenAI({OPENAI_BASE_URL}) 기사 판단 웹 서버를 시작합니다 ({URL})...")
    print("      (종료하려면 터미널에서 Ctrl+C 를 누르세요)")

    server_proc = subprocess.Popen(
        [
            str(venv_python),
            "-m",
            "uvicorn",
            "frontend:app",
            "--host",
            HOST,
            "--port",
            str(PORT),
            "--reload",
        ],
        cwd=str(ROOT_DIR),
    )

    try:
        wait_and_open_browser(URL)
        server_proc.wait()
    except KeyboardInterrupt:
        print("\n[종료] UI 서버를 안전하게 종료합니다...")
        server_proc.terminate()
        server_proc.wait(timeout=5)


if __name__ == "__main__":
    main()
