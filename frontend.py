"""
frontend.py - 확정된 userinterface.html UI 서빙, 12개 REST API 엔드포인트(create_fastapi_app) 및
원클릭 가상환경·브라우저 런처(main) 모듈
실행 방법: python frontend.py
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.error
import urllib.request
import venv
import webbrowser

from backend import (
    CLASSIFIER_MODEL_FILE,
    DB_FILE,
    ENV_FILE,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OPENAI_MODEL,
    ROOT_DIR,
    SUMMARIZER_MODEL_DIR,
    SUMMARIZER_SAFETENSORS_FILE,
    MultiAgentNewsOrchestrator,
    OpenAISupervisorHead,
    OrchestrationState,
    _detect_article_tone,
    _parse_numbered_judgment,
    classify_news_with_exported_model,
    get_finetuned_summarizer_info,
    get_openai_client,
    judge_articles_with_openai,
    judge_single_custom_article_with_openai,
    load_exported_news_classifier,
    preprocess_korean_news_for_classifier,
    resolve_openai_config,
)

VENV_DIR = ROOT_DIR / ".venv"
REQ_FILE = ROOT_DIR / "requirements.txt"
HTML_FILE = ROOT_DIR / "userinterface.html"
HOST = "127.0.0.1"
PORT = 8000
URL = f"http://{HOST}:{PORT}"


def create_fastapi_app():
    """FastAPI 앱을 생성하여 확정된 userinterface.html UI + OpenAI Head LLM 기반 Multi-Agent Orchestration API를 제공합니다."""
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse
    from pydantic import BaseModel, Field
    from mylggraph import NJKAgent

    app = FastAPI(
        title="Nexus (논점콩) - OpenAI Head LLM 기반 Multi-Agent Orchestration 뉴스 분석 서버",
        description="확정된 userinterface.html 디자인 + OpenAI Supervisor Head LLM + LangGraph Multi-Agent Orchestration (news.db RAG·관점비교·심판·환각차단)",
    )

    rag_agent = NJKAgent(db_path=DB_FILE)
    orchestrator = MultiAgentNewsOrchestrator(rag_agent=rag_agent)

    # 외부 모듈/테스트에서 orchestrator 및 rag_agent에 접근할 수 있도록 state에 바인딩
    app.state.rag_agent = rag_agent
    app.state.orchestrator = orchestrator

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
        cleanTitle: str = ""
        category: str = "시사·종합"
        clusterLabel: str = ""
        url: str = "#source"
        stance: str
        tone: str
        relevanceScore: int = 85
        judgmentSummary: str

    class OrchestrationStepItem(BaseModel):
        step: int
        agent: str
        role: str
        status: str
        detail: str
        model: str | None = None
        elapsedMs: float = 0.0

    class ClassifyNewsRequest(BaseModel):
        title: str
        content: str = ""
        top_k: int = 5

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
        orchestrationTrace: list[OrchestrationStepItem] = Field(default_factory=list)

    class OpenAIKeyRequest(BaseModel):
        base_url: str | None = None
        api_key: str | None = None
        model: str | None = None

    class CustomArticleJudgeRequest(BaseModel):
        topic: str = ""
        press: str = ""
        title: str
        content: str
        openai_api_key: str | None = None
        openai_model: str | None = None
        openai_base_url: str | None = None

    @app.get("/", response_class=HTMLResponse)
    async def serve_NJK_ui() -> str:
        """확정된 userinterface.html 디자인 파일을 반환합니다."""
        return HTML_FILE.read_text(encoding="utf-8")

    @app.get("/api/issues")
    async def list_popular_issues():
        """news.db에 적재된 대표 이슈 목록을 반환합니다."""
        return {"issues": rag_agent.rag.get_popular_issues(limit=5)}

    @app.get("/api/orchestration-info")
    async def get_orchestration_info():
        """현재 구성된 Multi-Agent Orchestration 아키텍처 및 Head LLM 정보를 반환합니다."""
        base_url, _, model = resolve_openai_config()
        return {
            "architecture": "LangGraph Supervisor Multi-Agent Orchestration",
            "head_llm": {
                "provider": "OpenAI Compatible API",
                "base_url": base_url,
                "model": model,
                "roles": [
                    "1. 질의 의도 분석·키워드 확장 및 실행 계획 수립 (supervisor_plan)",
                    "2. 클러스터 센트로이드 & 코사인 유사도 임계값 기반 시맨틱 라우팅 (SemanticCentroidRouter)",
                    "3. 관점 A/B 대립축 심판 및 최종 결과 감수 (judge_verifier_agent)",
                ],
            },
            "sub_agents": [
                {
                    "id": "RAGRetrievalSubAgent",
                    "description": "news.db 기반 sqlite-vec(title_vec) + TF-IDF + 키워드 하이브리드 검색 및 언론사 다양성 리랭킹",
                },
                {
                    "id": "SemanticCentroidRouter",
                    "description": "54개 KMeans 이슈 클러스터 센트로이드 및 검색 기사군 센트로이드 코사인 유사도 임계값 기반 초고속 시맨틱 라우터",
                },
                {
                    "id": "ComparisonSynthesisSubAgent",
                    "description": "문맥 벡터 코사인 거리 기반 관점 A vs 관점 B 클러스터링 및 양면 요약 합성",
                },
                {
                    "id": "JudgeVerifierSubAgent",
                    "description": "OpenAI API 연동 기사별 논조·관점 판정 및 원문 URL 100% 일치 검증",
                },
                {
                    "id": "HallucinationGuardSubAgent",
                    "description": "근거 기사 부재 시 허위 요약 및 가짜 링크 생성을 원천 차단하는 가드레일",
                },
            ],
        }

    @app.get("/api/openai-status")
    async def get_openai_status():
        """현재 연결된 OpenAI Head LLM 서버 상태 및 사용 가능한 모델 목록을 반환합니다."""
        base_url, key, model = resolve_openai_config()
        masked = f"{key[:11]}...{key[-4:]}" if len(key) > 15 else ("설정됨" if key else "미설정")

        available_models: list[str] = []
        server_online = False
        try:
            oai = get_openai_client(base_url=base_url, api_key=key)
            available_models = [m.id for m in oai.models.list().data]
            server_online = True
        except Exception:
            available_models = [
                "kanana_summarizer_full_ft_model",
                "kakaocorp/kanana-nano-2.1b-instruct",
                "kakaocorp/kanana-nano-2.1b-base",
                "unsloth/Qwen3.5-2B-GGUF",
                "gpt-4o-mini",
                "gpt-4o",
            ]

        return {
            "configured": bool(key),
            "server_online": server_online,
            "base_url": base_url,
            "masked_key": masked,
            "model": model,
            "available_models": available_models,
            "orchestration_mode": "Multi-Agent Supervisor (OpenAI Head LLM)",
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
        """
        OpenAI API를 Head LLM(Supervisor)으로 사용하는 Multi-Agent Orchestration 파이프라인을 실행하여
        질의 분석 -> news.db 하이브리드 RAG 검색 -> 관점 A/B 비교 합성 -> 기사 심판 및 검증 결과를 반환합니다.
        """
        orchestrated_result = orchestrator.run(
            query=req.query.strip(),
            thread_id=req.thread_id,
            openai_api_key=req.openai_api_key,
            openai_model=req.openai_model,
            openai_base_url=req.openai_base_url,
        )
        return AnalyzeResponse(**orchestrated_result)

    @app.post("/api/judge-article")
    async def judge_custom_article(req: CustomArticleJudgeRequest):
        """단일 기사를 OpenAI Head LLM, sqlite-vec(title_vec) 및 Export된 최고 성능 분류기로 즉시 판단·분류하여 반환합니다."""
        res = judge_single_custom_article_with_openai(
            topic=req.topic,
            press=req.press,
            title=req.title,
            content=req.content,
            openai_api_key=req.openai_api_key,
            openai_model=req.openai_model,
            openai_base_url=req.openai_base_url,
        )
        vec_cls = rag_agent.rag.classify_news_by_title_vec(title=req.title, content=req.content, top_k=3)
        ml_cls = classify_news_with_exported_model(title=req.title, content=req.content, top_k=3)
        if ml_cls:
            vec_cls["ml_classifier"] = ml_cls
            vec_cls["cluster_id"] = ml_cls["predicted_cluster_id"]
            vec_cls["cluster_label"] = ml_cls["predicted_cluster_label"]
            res["mlClassification"] = ml_cls
        res["vecClassification"] = vec_cls
        return res

    @app.get("/api/vec-classifications")
    async def get_vec_classifications(
        category: str | None = None,
        cluster_id: int | None = None,
        limit: int = 18,
    ):
        """news.db의 title_vec + sqlite-vec 기반 뉴스 분류 현황(카테고리·이슈 클러스터·기사 목록)을 반환합니다."""
        return rag_agent.rag.get_classification_overview(
            category=category,
            cluster_id=cluster_id,
            limit=limit,
        )

    @app.post("/api/classify-news")
    async def classify_news_endpoint(req: ClassifyNewsRequest):
        """입력된 뉴스를 Export된 최고 점수 분류기(Calibrated LinearSVC) 및 sqlite-vec으로 즉시 분류하고 유사 기사를 검색합니다."""
        vec_res = rag_agent.rag.classify_news_by_title_vec(
            title=req.title,
            content=req.content,
            top_k=req.top_k,
        )
        ml_cls = classify_news_with_exported_model(
            title=req.title,
            content=req.content,
            top_k=req.top_k,
        )
        if ml_cls:
            vec_res["ml_classifier"] = ml_cls
            vec_res["cluster_id"] = ml_cls["predicted_cluster_id"]
            vec_res["cluster_label"] = ml_cls["predicted_cluster_label"]
        return vec_res

    @app.get("/api/classifier-info")
    async def get_exported_classifier_info():
        """파인튜닝 완료된 분류기(best_news_classifier.joblib) 및 요약기(kanana_summarizer_full_ft_model/model.safetensors) 메타데이터를 반환합니다."""
        bundle = load_exported_news_classifier()
        summarizer_info = get_finetuned_summarizer_info()
        if not bundle:
            return {
                "loaded": False,
                "model_file": str(CLASSIFIER_MODEL_FILE),
                "summarizer_model": summarizer_info,
            }
        return {
            "loaded": True,
            "model_file": str(CLASSIFIER_MODEL_FILE),
            "model_name": bundle.get("model_name"),
            "optimal_k": bundle.get("optimal_k"),
            "metrics": bundle.get("metrics", {}),
            "all_model_metrics": bundle.get("all_model_metrics", []),
            "cluster_label_map": bundle.get("cluster_label_map", {}),
            "summarizer_model": summarizer_info,
        }

    @app.get("/api/summarizer-info")
    async def get_finetuned_summarizer_endpoint():
        """파인튜닝 완료된 요약기(kanana_summarizer_full_ft_model/model.safetensors) 정보를 반환합니다."""
        return get_finetuned_summarizer_info()

    @app.post("/api/ml-classify")
    async def ml_classify_endpoint(req: ClassifyNewsRequest):
        """Export된 최고 성능 머신러닝 분류기(best_news_classifier.joblib)를 단독 호출하여 뉴스를 분류합니다."""
        res = classify_news_with_exported_model(
            title=req.title,
            content=req.content,
            top_k=req.top_k,
        )
        if not res:
            return {"error": "best_news_classifier.joblib 모델을 찾을 수 없습니다."}
        return res

    @app.post("/api/rebuild-vec-index")
    async def rebuild_vec_index_endpoint():
        """news.db의 CLEAN_TITLE을 벡터화하여 NEWS.title_vec 및 VEC_NEWS_TITLE(sqlite-vec) 색인을 재구축합니다."""
        count = rag_agent.rag.ensure_rag_index(force_rebuild=True)
        return {
            "status": "ok",
            "indexed_count": count,
            "elapsed_ms": rag_agent.rag.last_index_elapsed_ms,
            "sqlite_vec_version": rag_agent.rag.sqlite_vec_version,
        }

    class SyncedSummaryItem(BaseModel):
        news_pk: int
        title: str = ""
        summary: str
        model: str = "kanana_summarizer_full_ft_model"
        summary_at: str = ""

    class ReceiveSummariesRequest(BaseModel):
        summaries: list[SyncedSummaryItem]

    @app.post("/api/receive-summaries")
    async def receive_summaries_from_laptop(req: ReceiveSummariesRequest):
        """노트북(summary.py)에서 4GB GPU SLM으로 요약한 NEWS.SUMMARY 데이터를 수신하여 Render 웹서비스 DB에 반영합니다."""
        import sqlite3

        updated = 0
        with sqlite3.connect(str(DB_FILE)) as con:
            cur = con.cursor()
            news_cols = {c[1] for c in cur.execute("PRAGMA table_info(NEWS)").fetchall()}
            if "SUMMARY" not in news_cols:
                cur.execute("ALTER TABLE NEWS ADD COLUMN SUMMARY TEXT DEFAULT ''")
            if "SUMMARY_MODEL" not in news_cols:
                cur.execute("ALTER TABLE NEWS ADD COLUMN SUMMARY_MODEL TEXT DEFAULT ''")
            if "SUMMARY_AT" not in news_cols:
                cur.execute("ALTER TABLE NEWS ADD COLUMN SUMMARY_AT TEXT DEFAULT ''")

            for item in req.summaries:
                cur.execute(
                    "UPDATE NEWS SET SUMMARY = ?, SUMMARY_MODEL = ?, SUMMARY_AT = ? WHERE PK = ?",
                    (item.summary, item.model, item.summary_at, item.news_pk),
                )
                try:
                    cur.execute(
                        "UPDATE RAG_INDEX SET SUMMARY = ? WHERE NEWS_PK = ?",
                        (item.summary, item.news_pk),
                    )
                except sqlite3.OperationalError:
                    pass
                updated += 1
            con.commit()
        return {"status": "ok", "updated_count": updated}

    @app.get("/api/summaries")
    async def get_synced_summaries(limit: int = 30, only_summarized: bool = True):
        """summary.py가 요약하여 news.db에 저장한 기사 요약 목록을 웹서비스에서 조회합니다."""
        import sqlite3

        with sqlite3.connect(str(DB_FILE)) as con:
            cur = con.cursor()
            news_cols = {c[1] for c in cur.execute("PRAGMA table_info(NEWS)").fetchall()}
            if "SUMMARY" not in news_cols:
                return {"count": 0, "summaries": []}

            where_sql = "WHERE N.SUMMARY IS NOT NULL AND TRIM(N.SUMMARY) != ''" if only_summarized else ""
            rows = cur.execute(
                f"""
                SELECT N.PK, N.TITLE, COALESCE(N.CATEGORY, '시사·종합'), COALESCE(N.SUMMARY, ''),
                       COALESCE(N.SUMMARY_MODEL, ''), COALESCE(N.SUMMARY_AT, ''), COALESCE(L.URL, '')
                FROM NEWS N
                LEFT JOIN LINK L ON N.FK = L.PK
                {where_sql}
                ORDER BY N.PK DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        items = [
            {
                "news_pk": r[0],
                "title": r[1],
                "category": r[2],
                "summary": r[3],
                "summary_model": r[4],
                "summary_at": r[5],
                "url": r[6],
            }
            for r in rows
        ]
        return {"count": len(items), "summaries": items}

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
    """가상환경에 필수 패키지(sqlite_vec, sklearn 포함)가 설치되어 있는지 확인합니다."""
    check = subprocess.run(
        [str(venv_python), "-c", "import fastapi, uvicorn, langgraph, numpy, openai, sqlite_vec, sklearn"],
        capture_output=True,
        text=True,
    )
    return check.returncode == 0


def install_dependencies(venv_python: Path) -> None:
    """requirements.txt, sqlite-vec, scikit-learn, openai 패키지를 가상환경에 설치합니다."""
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
            "sqlite-vec",
            "scikit-learn",
        ],
        cwd=str(ROOT_DIR),
        check=True,
    )
    print("      -> 구동 패키지(sqlite-vec, scikit-learn, OpenAI 포함) 설치 완료!")


def wait_and_open_browser(url: str, timeout_sec: float = 15.0) -> None:
    """서버가 응답할 때까지 대기한 뒤 기본 웹 브라우저로 userinterface.html UI를 엽니다."""
    start = time.time()
    while time.time() - start < timeout_sec:
        try:
            with urllib.request.urlopen(url, timeout=1.0) as resp:
                if resp.status == 200:
                    print(f"[4/4] 브라우저에서 userinterface.html UI를 엽니다: {url}")
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
        print("[2/4] 필수 패키지(OpenAI API 및 LangGraph 포함)가 이미 가상환경에 설치되어 있습니다.")

    print(
        f"[3/4] OpenAI Head LLM({OPENAI_BASE_URL}) 기반 Multi-Agent Orchestration 웹 서버를 시작합니다 ({URL})..."
    )
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


__all__ = [
    "create_fastapi_app",
    "app",
    "get_venv_python",
    "ensure_venv",
    "is_ui_ready",
    "install_dependencies",
    "wait_and_open_browser",
    "main",
]


if __name__ == "__main__":
    main()
