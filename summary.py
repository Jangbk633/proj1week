"""
summary.py - 노트북 4GB VRAM GPU 기반 로컬 SLM 뉴스 요약기 대시보드, REST API, OpenAI 프록시 서버 & CLI 런처

[모듈 분리 및 리팩토링 내역]
1. preprocessing.py  : clean_article_content() (이메일·전화번호·저작권 문구 제거 및 컨텍스트 길이 조절)
2. slm_server.py     : find_local_gguf_path(), find_llama_server_exe(), start_optimized_llama_server()
3. summary_engine.py : connect_summary_db() (timeout=15.0 + WAL 모드), ensure_summary_columns(),
                       LocalKananaSummarizer (summarize_text, summarize_batch, push_summaries_to_render)
4. summary.py        : create_summary_app() (웹 대시보드, 요약 API, OpenAI 호환 프록시) 및 main() CLI 진입점
   - 모듈 임포트 시 전역 `app = create_summary_app()`이 즉시 실행되어 `LocalKananaSummarizer`가 두 번 생성되던 문제를
     지연 초기화(`get_summary_app()` / 모듈 `__getattr__`)로 해결했습니다.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

from preprocessing import clean_article_content
from rag import DEFAULT_DB_PATH, ROOT_DIR
from slm_server import (
    DEFAULT_API_KEY,
    DEFAULT_GGUF_FILENAME,
    DEFAULT_LLAMA_BASE_URL,
    DEFAULT_LLAMA_SERVER_PORT,
    DEFAULT_MODEL_REPO,
    FINETUNED_SAFETENSORS_FILE,
    FINETUNED_SUMMARIZER_DIR,
    find_finetuned_safetensors_model,
    find_free_port,
    find_llama_server_exe,
    find_local_gguf_path,
    get_finetuned_summarizer_info,
    probe_openai_models_url,
    start_optimized_llama_server,
)
from summary_engine import (
    DEFAULT_RENDER_URL,
    SUMMARY_SYNC_TOKEN,
    LocalKananaSummarizer,
    connect_summary_db,
    ensure_summary_columns,
)

DEFAULT_SUMMARY_PORT = int(os.environ.get("SUMMARY_PORT", "8001"))


def create_summary_app(summarizer: LocalKananaSummarizer | None = None):
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import HTMLResponse
    from pydantic import BaseModel, Field

    engine = summarizer or LocalKananaSummarizer()

    app = FastAPI(
        title="Kanana-Nano 2.1B 파인튜닝 뉴스 요약 서버 (summary.py)",
        description=(
            "파인튜닝된 요약기(kanana_summarizer_full_ft_model/model.safetensors)를 사용하여 "
            "news.db의 NEWS.CONTENT를 요약하고 웹서비스에 제공하는 API 서버"
        ),
    )
    app.state.summarizer = engine

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    class SummarizeArticleRequest(BaseModel):
        news_pk: int | None = None
        title: str = ""
        content: str = ""
        force: bool = False

    class BatchSummarizeRequest(BaseModel):
        limit: int = Field(default=10, ge=1, le=200)
        force: bool = False
        category: str | None = None
        render_url: str | None = None

    class PushRenderRequest(BaseModel):
        render_url: str
        limit: int = Field(default=100, ge=1, le=500)

    class ChatCompletionRequest(BaseModel):
        model: str = DEFAULT_MODEL_REPO
        messages: list[dict[str, str]]
        temperature: float = 0.1
        max_tokens: int = 256

    @app.get("/", response_class=HTMLResponse)
    async def summary_dashboard() -> str:
        """노트북 SLM 요약 상태를 확인하고 즉시 배치 요약·Render 동기화를 실행할 수 있는 웹 UI"""
        return """<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Kanana-Nano 2.1B 로컬 GPU 요약 서버 (summary.py)</title>
  <style>
    :root {
      --bg: #0f172a; --card: #1e293b; --border: #334155;
      --text: #f8fafc; --muted: #94a3b8; --accent: #38bdf8; --green: #22c55e;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0; padding: 28px; font-family: 'Pretendard', -apple-system, BlinkMacSystemFont, sans-serif;
      background: var(--bg); color: var(--text); line-height: 1.5;
    }
    .container { max-width: 1080px; margin: 0 auto; }
    .header {
      display: flex; justify-content: space-between; align-items: center;
      padding: 20px 24px; background: var(--card); border: 1px solid var(--border);
      border-radius: 14px; margin-bottom: 20px;
    }
    .badge {
      display: inline-block; padding: 5px 12px; border-radius: 999px;
      font-size: 13px; font-weight: 600; background: rgba(56,189,248,0.15); color: var(--accent);
    }
    .badge.online { background: rgba(34,197,94,0.18); color: var(--green); }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 14px; margin-bottom: 20px; }
    .stat-card {
      background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 16px 18px;
    }
    .stat-label { font-size: 13px; color: var(--muted); }
    .stat-val { font-size: 26px; font-weight: 700; margin-top: 4px; }
    .controls {
      background: var(--card); border: 1px solid var(--border); border-radius: 12px;
      padding: 18px 20px; margin-bottom: 20px; display: flex; flex-wrap: wrap; gap: 10px; align-items: center;
    }
    input, select, button {
      padding: 9px 14px; border-radius: 8px; border: 1px solid var(--border);
      background: #0b1120; color: var(--text); font-size: 14px;
    }
    button {
      background: var(--accent); color: #0f172a; font-weight: 700; border: none; cursor: pointer;
    }
    button:hover { opacity: 0.9; }
    button.secondary { background: #334155; color: var(--text); }
    .list { display: flex; flex-direction: column; gap: 12px; }
    .item {
      background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 16px 20px;
    }
    .item-meta { font-size: 12px; color: var(--muted); margin-bottom: 6px; display: flex; gap: 10px; }
    .item-title { font-size: 16px; font-weight: 700; margin-bottom: 8px; }
    .item-summary {
      background: rgba(15,23,42,0.65); border-left: 3px solid var(--accent);
      padding: 10px 14px; border-radius: 6px; font-size: 14px; color: #e2e8f0;
    }
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <div>
        <h2 style="margin:0 0 6px 0;">파인튜닝 뉴스 요약 서버 (summary.py)</h2>
        <div style="font-size:13px;color:var(--muted);">
          요약기: <code>kanana_summarizer_full_ft_model/model.safetensors</code> · 분류기: <code>best_news_classifier.joblib</code> · DB: <code>news.db</code>
        </div>
      </div>
      <span id="serverBadge" class="badge">상태 확인 중...</span>
    </div>

    <div class="grid">
      <div class="stat-card"><div class="stat-label">전체 뉴스 (NEWS)</div><div class="stat-val" id="stTotal">-</div></div>
      <div class="stat-card"><div class="stat-label">요약 완료 기사</div><div class="stat-val" id="stDone">-</div></div>
      <div class="stat-card"><div class="stat-label">미요약 기사</div><div class="stat-val" id="stRemain">-</div></div>
      <div class="stat-card"><div class="stat-label">진행률</div><div class="stat-val" id="stPct">-</div></div>
    </div>

    <div class="controls">
      <label style="font-size:13px;color:var(--muted);">배치 건수:</label>
      <input type="number" id="batchLimit" value="5" min="1" max="100" style="width:80px;" />
      <button onclick="runBatch()" id="btnBatch">노트북 GPU로 미요약 기사 요약하기</button>
      <input type="text" id="renderUrl" placeholder="Render 웹서비스 주소 (예: https://my-app.onrender.com)" style="flex:1;min-width:260px;" />
      <button class="secondary" onclick="pushToRender()">Render 웹서비스로 요약 전송(Push)</button>
      <span id="statusMsg" style="font-size:13px;color:var(--accent);"></span>
    </div>

    <div class="list" id="summaryList"></div>
  </div>

  <script>
    async function loadDashboard() {
      const stRes = await fetch('/api/status');
      const st = await stRes.json();
      document.getElementById('stTotal').textContent = st.total_news + '건';
      document.getElementById('stDone').textContent = st.summarized_news + '건';
      document.getElementById('stRemain').textContent = st.remaining_news + '건';
      document.getElementById('stPct').textContent = st.progress_pct + '%';
      const badge = document.getElementById('serverBadge');
      badge.textContent = st.slm_server_online ? 'SLM GPU 서버 온라인 (4GB VRAM)' : 'SLM 대기 중 (요약 시 자동 기동)';
      badge.className = 'badge ' + (st.slm_server_online ? 'online' : '');

      const listRes = await fetch('/api/summaries?limit=25&only_summarized=false');
      const data = await listRes.json();
      const box = document.getElementById('summaryList');
      box.innerHTML = data.summaries.map(item => `
        <div class="item">
          <div class="item-meta">
            <span>PK #${item.news_pk}</span>
            <span>분류: ${item.category}</span>
            <span>모델: ${item.summary_model || '미요약'}</span>
            <span>${item.summary_at || ''}</span>
          </div>
          <div class="item-title">${item.title}</div>
          ${item.summary
            ? `<div class="item-summary">${item.summary}</div>`
            : `<button class="secondary" onclick="summarizeOne(${item.news_pk})">이 기사 즉시 요약 (Kanana 2.1B)</button>`
          }
        </div>
      `).join('');
    }

    async function summarizeOne(pk) {
      document.getElementById('statusMsg').textContent = `PK #${pk} 기사 요약 중...`;
      await fetch('/api/summarize', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({news_pk: pk, force: true})
      });
      document.getElementById('statusMsg').textContent = `PK #${pk} 요약 완료!`;
      await loadDashboard();
    }

    async function runBatch() {
      const limit = parseInt(document.getElementById('batchLimit').value || '5', 10);
      const renderUrl = document.getElementById('renderUrl').value.trim();
      const btn = document.getElementById('btnBatch');
      btn.disabled = true;
      document.getElementById('statusMsg').textContent = `${limit}건 GPU 요약 실행 중...`;
      try {
        const res = await fetch('/api/summarize-batch', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({limit, render_url: renderUrl || null})
        });
        const out = await res.json();
        document.getElementById('statusMsg').textContent = `${out.processed_count}건 요약 완료!`;
      } finally {
        btn.disabled = false;
        await loadDashboard();
      }
    }

    async function pushToRender() {
      const renderUrl = document.getElementById('renderUrl').value.trim();
      if (!renderUrl) { alert('Render 웹서비스 URL을 입력하세요.'); return; }
      document.getElementById('statusMsg').textContent = 'Render 서버로 요약본 동기화 중...';
      const res = await fetch('/api/push-to-render', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({render_url: renderUrl, limit: 200})
      });
      const out = await res.json();
      document.getElementById('statusMsg').textContent =
        out.status === 'ok' ? `Render 전송 완료 (${out.pushed_count}건)` : `전송 오류: ${out.error || out.status}`;
    }

    loadDashboard();
  </script>
</body>
</html>"""

    @app.get("/api/status")
    async def get_status():
        """SLM 서버 상태 및 news.db 요약 통계 반환"""
        return engine.get_summary_stats()

    @app.get("/api/summaries")
    async def list_summaries_api(
        limit: int = Query(default=30, ge=1, le=200),
        offset: int = Query(default=0, ge=0),
        category: str | None = None,
        keyword: str | None = None,
        only_summarized: bool = True,
    ):
        """웹서비스(Render 또는 프론트엔드)에 요약된 뉴스 목록을 제공합니다."""
        items = engine.list_summaries(
            limit=limit,
            offset=offset,
            category=category,
            keyword=keyword,
            only_summarized=only_summarized,
        )
        return {
            "count": len(items),
            "model": engine.model_id,
            "summaries": items,
        }

    @app.get("/api/summary/{news_pk}")
    async def get_or_create_summary(news_pk: int, force: bool = False):
        """특정 NEWS.PK의 요약을 반환하며, 아직 요약이 없으면 즉시 로컬 GPU SLM으로 요약 후 반환합니다."""
        res = engine.summarize_news_by_pk(news_pk=news_pk, force=force)
        if not res:
            raise HTTPException(status_code=404, detail=f"NEWS.PK={news_pk} 기사를 찾을 수 없습니다.")
        return res

    @app.post("/api/summarize")
    async def summarize_article_api(req: SummarizeArticleRequest):
        """
        웹서비스에서 특정 news_pk 또는 {title, content} 본문을 보내면
        노트북 4GB GPU의 Kanana-Nano-2.1B 모델로 요약하여 반환합니다.
        """
        if req.news_pk is not None:
            res = engine.summarize_news_by_pk(news_pk=req.news_pk, force=req.force)
            if not res:
                raise HTTPException(status_code=404, detail=f"NEWS.PK={req.news_pk} 기사를 찾을 수 없습니다.")
            return res

        if not req.content.strip():
            raise HTTPException(status_code=400, detail="news_pk 또는 content 중 하나는 반드시 입력해야 합니다.")

        res = engine.summarize_text(title=req.title, content=req.content)
        return {
            "title": req.title,
            "summary": res["summary"],
            "model": res["model"],
            "used_slm": res["used_slm"],
            "elapsed_ms": res["elapsed_ms"],
        }

    @app.post("/api/summarize-batch")
    async def summarize_batch_api(req: BatchSummarizeRequest):
        """news.db 내 미요약 기사들을 일괄 요약하고 선택적으로 Render 웹서비스에 Push합니다."""
        return engine.summarize_batch(
            limit=req.limit,
            force=req.force,
            category=req.category,
            render_url=req.render_url,
        )

    @app.post("/api/push-to-render")
    async def push_to_render_api(req: PushRenderRequest):
        """저장된 요약 데이터를 Render($0) 웹서비스로 즉시 전송합니다."""
        return engine.push_summaries_to_render(render_base_url=req.render_url, limit=req.limit)

    @app.get("/v1/models")
    async def openai_models_proxy():
        """Render 웹서비스(frontend.py)가 OPENAI_BASE_URL로 연결할 때 사용하는 OpenAI 호환 모델 목록 엔드포인트"""
        return {
            "object": "list",
            "data": [
                {
                    "id": engine.model_id,
                    "object": "model",
                    "owned_by": "local-laptop-4gb-gpu",
                }
            ],
        }

    @app.post("/v1/chat/completions")
    async def openai_chat_completions_proxy(req: ChatCompletionRequest):
        """
        Render 웹서비스(frontend.py)에서 OPENAI_BASE_URL을 노트북 터널 주소로 설정했을 때
        요청을 로컬 Kanana-Nano-2.1B GPU 서버로 전달하는 OpenAI 호환 엔드포인트
        """
        if engine.ensure_local_slm_server():
            try:
                from openai import OpenAI

                client = OpenAI(base_url=engine.base_url, api_key=engine.api_key or "local-slm")
                resp = client.chat.completions.create(
                    model=engine.model_id,
                    messages=req.messages,  # type: ignore[arg-type]
                    temperature=req.temperature,
                    max_tokens=req.max_tokens,
                    timeout=45.0,
                )
                return resp.model_dump()
            except Exception as e:
                raise HTTPException(status_code=502, detail=f"로컬 SLM 호출 실패: {e}")
        raise HTTPException(status_code=503, detail="로컬 Kanana SLM 서버가 오프라인입니다.")

    return app


_summary_app_instance: Any = None


def get_summary_app(summarizer: LocalKananaSummarizer | None = None) -> Any:
    """
    [리팩토링] 모듈 임포트 시점에 LocalKananaSummarizer가 중복 생성되는 현상을 막기 위한
    지연 초기화(Lazy Singleton) 팩토리 함수입니다.
    """
    global _summary_app_instance
    if summarizer is not None:
        _summary_app_instance = create_summary_app(summarizer)
        return _summary_app_instance
    if _summary_app_instance is None:
        _summary_app_instance = create_summary_app()
    return _summary_app_instance


def __getattr__(name: str) -> Any:
    """`uvicorn summary:app` 또는 `from summary import app` 호출 시에만 지연 생성"""
    if name == "app":
        return get_summary_app()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def main() -> None:
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser = argparse.ArgumentParser(
        description="news.db NEWS.CONTENT 로컬 SLM(kenonix/kanana-nano-2.1b-instruct-Q8_0-GGUF) 요약 및 웹서비스 제공 프로그램"
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "batch", "serve"],
        default="auto",
        help="auto: 초기 배치 요약 후 API 서버 실행 | batch: 배치 요약만 수행 후 종료 | serve: 요약 API 서버만 즉시 실행",
    )
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH), help="news.db 파일 경로")
    parser.add_argument("--limit", type=int, default=10, help="배치 요약할 기사 수 (기본: 10)")
    parser.add_argument("--pk", type=int, default=None, help="특정 NEWS.PK 기사 1건만 즉시 요약")
    parser.add_argument("--force", action="store_true", help="이미 요약된 기사도 다시 요약")
    parser.add_argument("--host", default="0.0.0.0", help="요약 제공 웹서버 바인딩 호스트 (기본: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=DEFAULT_SUMMARY_PORT, help="요약 제공 웹서버 포트 (기본: 8001)")
    parser.add_argument(
        "--render-url",
        default=DEFAULT_RENDER_URL,
        help="요약 결과를 Push할 Render($0) 웹서비스 URL (예: https://my-app.onrender.com)",
    )
    parser.add_argument(
        "--ngl",
        type=int,
        default=99,
        help="GPU에 오프로딩할 레이어 수 (4GB VRAM에서 Q8_0 2.1B 모델은 99로 전체 GPU 적재 가능)",
    )
    args = parser.parse_args()

    summarizer = LocalKananaSummarizer(
        db_path=args.db,
        n_gpu_layers=args.ngl,
    )

    if args.pk is not None:
        res = summarizer.summarize_news_by_pk(args.pk, force=True)
        print(json.dumps(res, ensure_ascii=False, indent=2))
        if args.render_url and res:
            summarizer.push_summaries_to_render(args.render_url, items=[res])
        return

    if args.mode == "batch":
        out = summarizer.summarize_batch(
            limit=args.limit,
            force=args.force,
            render_url=args.render_url,
        )
        print(
            f"\n[요약 통계] 전체 {out['stats']['total_news']}건 중 "
            f"{out['stats']['summarized_news']}건 요약 완료 ({out['stats']['progress_pct']}%)"
        )
        return

    import threading
    import webbrowser
    import uvicorn

    summarizer.ensure_local_slm_server()

    if args.mode == "auto" and args.limit > 0:
        def _bg_worker() -> None:
            time.sleep(1.5)
            summarizer.summarize_batch(
                limit=args.limit,
                force=args.force,
                render_url=args.render_url,
            )

        threading.Thread(target=_bg_worker, daemon=True).start()

    url = f"http://127.0.0.1:{args.port}"
    print(
        f"\n[요약 제공 웹서버 시작] {url} "
        f"(모델: {summarizer.model_id}, DB: {args.db})"
    )
    threading.Thread(
        target=lambda: (time.sleep(1.2), webbrowser.open(url)),
        daemon=True,
    ).start()

    server_app = get_summary_app(summarizer)
    uvicorn.run(server_app, host=args.host, port=args.port)


__all__ = [
    "DEFAULT_MODEL_REPO",
    "DEFAULT_GGUF_FILENAME",
    "DEFAULT_LLAMA_SERVER_PORT",
    "DEFAULT_LLAMA_BASE_URL",
    "DEFAULT_API_KEY",
    "DEFAULT_SUMMARY_PORT",
    "DEFAULT_RENDER_URL",
    "SUMMARY_SYNC_TOKEN",
    "clean_article_content",
    "connect_summary_db",
    "ensure_summary_columns",
    "find_local_gguf_path",
    "find_llama_server_exe",
    "probe_openai_models_url",
    "find_free_port",
    "start_optimized_llama_server",
    "LocalKananaSummarizer",
    "create_summary_app",
    "get_summary_app",
    "app",
    "main",
]


if __name__ == "__main__":
    main()
