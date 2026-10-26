"""
summary_engine.py - LocalKananaSummarizer 기반 뉴스 요약 생성, DB(NEWS/RAG_INDEX) 저장 및 Render Push 동기화 모듈

[리팩토링 반영 사항]
1. SQLite 동시 쓰기 잠금 완화: 모든 DB 연결에 `connect_summary_db()`(timeout=15.0 + `PRAGMA journal_mode=WAL`)를 적용하여
   백그라운드 배치 요약 스레드와 웹 대시보드 즉시 요약 요청이 동시에 실행될 때 `database is locked` 예외 방지
2. 중복 제거: `clean_article_content`를 `preprocessing.py`에서 임포트하고, GGUF/llama-server 탐색·실행 로직은
   `slm_server.py`에서 임포트하여 모듈화
"""

from __future__ import annotations

import atexit
from datetime import datetime
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import time
from typing import Any

import httpx

from preprocessing import clean_article_content
from rag import DEFAULT_DB_PATH
from slm_server import (
    DEFAULT_API_KEY,
    DEFAULT_GGUF_FILENAME,
    DEFAULT_LLAMA_BASE_URL,
    DEFAULT_LLAMA_SERVER_PORT,
    DEFAULT_MODEL_REPO,
    find_free_port,
    find_llama_server_exe,
    find_local_gguf_path,
    probe_openai_models_url,
    start_optimized_llama_server,
)

DEFAULT_RENDER_URL = os.environ.get("RENDER_SERVICE_URL", "").strip()
SUMMARY_SYNC_TOKEN = os.environ.get("SUMMARY_SYNC_TOKEN", "").strip()


def connect_summary_db(db_path: str | Path = DEFAULT_DB_PATH, timeout: float = 15.0) -> sqlite3.Connection:
    """
    SQLite 동시 쓰기 잠금(`database is locked`)을 방지하기 위해
    timeout=15.0 및 WAL(Write-Ahead Logging) 모드를 활성화한 연결을 반환합니다.
    """
    con = sqlite3.connect(str(db_path), timeout=timeout)
    try:
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA busy_timeout=15000")
    except sqlite3.OperationalError:
        pass
    return con


def ensure_summary_columns(db_path: str | Path = DEFAULT_DB_PATH) -> None:
    """
    news.db의 NEWS 테이블과 RAG_INDEX 테이블에 SUMMARY, SUMMARY_MODEL, SUMMARY_AT 컬럼을 보장합니다.
    """
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"news.db 파일을 찾을 수 없습니다: {db_path}")

    with connect_summary_db(db_path) as con:
        cur = con.cursor()
        news_cols = {c[1] for c in cur.execute("PRAGMA table_info(NEWS)").fetchall()}
        if "SUMMARY" not in news_cols:
            cur.execute("ALTER TABLE NEWS ADD COLUMN SUMMARY TEXT DEFAULT ''")
        if "SUMMARY_MODEL" not in news_cols:
            cur.execute("ALTER TABLE NEWS ADD COLUMN SUMMARY_MODEL TEXT DEFAULT ''")
        if "SUMMARY_AT" not in news_cols:
            cur.execute("ALTER TABLE NEWS ADD COLUMN SUMMARY_AT TEXT DEFAULT ''")

        tables = {
            r[0]
            for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        if "RAG_INDEX" in tables:
            rag_cols = {c[1] for c in cur.execute("PRAGMA table_info(RAG_INDEX)").fetchall()}
            if "SUMMARY" not in rag_cols:
                cur.execute("ALTER TABLE RAG_INDEX ADD COLUMN SUMMARY TEXT NOT NULL DEFAULT ''")
        con.commit()


class LocalKananaSummarizer:
    """
    VRAM 4GB GPU 환경에서 kenonix/kanana-nano-2.1b-instruct-Q8_0-GGUF 모델을 구동하여
    news.db의 기사를 요약하는 로컬 SLM 엔진입니다.
    """

    def __init__(
        self,
        db_path: str | Path = DEFAULT_DB_PATH,
        model_id: str = DEFAULT_MODEL_REPO,
        base_url: str = DEFAULT_LLAMA_BASE_URL,
        api_key: str = DEFAULT_API_KEY,
        n_gpu_layers: int = 99,
        ctx_size: int = 2048,
        auto_start_server: bool = True,
    ) -> None:
        self.db_path = Path(db_path)
        self.model_id = model_id
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.n_gpu_layers = n_gpu_layers
        self.ctx_size = ctx_size
        self.auto_start_server = auto_start_server
        self._llama_proc: subprocess.Popen | None = None

        ensure_summary_columns(self.db_path)
        atexit.register(self.stop_local_server)

    @staticmethod
    def _probe_openai_models_url(base_url: str, api_key: str) -> bool:
        return probe_openai_models_url(base_url, api_key)

    def is_llm_server_online(self) -> bool:
        """
        로컬 OpenAI 호환 SLM 추론 서버(8891, 8892 등)가 응답하는지 확인하고,
        응답하는 로컬 포트가 있으면 self.base_url을 자동 갱신합니다.
        """
        candidates = [
            self.base_url,
            "http://127.0.0.1:8891/v1",
            "http://127.0.0.1:8892/v1",
        ]
        seen: set[str] = set()
        for cand in candidates:
            c = cand.rstrip("/")
            if c in seen:
                continue
            seen.add(c)
            if self._probe_openai_models_url(c, self.api_key):
                self.base_url = c
                return True
        return False

    @staticmethod
    def _find_free_port(preferred_ports: tuple[int, ...] = (8891, 8892, 11435)) -> int:
        return find_free_port(preferred_ports)

    def ensure_local_slm_server(self) -> bool:
        """
        로컬 SLM 추론 서버가 켜져 있지 않으면 비어 있는 포트를 찾아
        4GB VRAM 최적화 파라미터로 llama-server.exe를 자동 실행합니다.
        """
        if self.is_llm_server_online():
            return True

        if not self.auto_start_server:
            return False

        proc, active_url = start_optimized_llama_server(
            model_id=self.model_id,
            api_key=self.api_key,
            n_gpu_layers=self.n_gpu_layers,
            ctx_size=self.ctx_size,
        )
        if proc and active_url:
            self._llama_proc = proc
            self.base_url = active_url
            return True
        return False

    def stop_local_server(self) -> None:
        """직접 띄운 llama-server 프로세스가 있으면 종료합니다."""
        if self._llama_proc and self._llama_proc.poll() is None:
            try:
                self._llama_proc.terminate()
                self._llama_proc.wait(timeout=5)
            except Exception:
                pass
            self._llama_proc = None

    @staticmethod
    def fallback_extractive_summary(title: str, content: str) -> str:
        """SLM 서버가 오프라인일 때 사용하는 규칙 기반 핵심 문장 추출 폴백입니다."""
        cleaned = clean_article_content(content, max_chars=1200)
        cleaned = re.sub(r"\[[^\]]{2,30}\]", " ", cleaned)
        sents = [
            s.strip()
            for s in re.split(r"(?<=[.?!다])\s+", cleaned)
            if 25 <= len(s.strip()) <= 200 and not s.strip().endswith("기자")
        ]
        if len(sents) >= 2:
            return f"{sents[0]} {sents[1]}"
        if sents:
            return sents[0]
        return cleaned[:220] if cleaned else title

    def summarize_text(self, title: str, content: str, max_tokens: int = 220) -> dict[str, Any]:
        """
        단일 기사의 TITLE과 CONTENT를 입력받아 kenonix/kanana-nano-2.1b-instruct-Q8_0-GGUF 모델로
        한국어 2~3문장 핵심 요약을 생성합니다.
        """
        t0 = time.perf_counter()
        cleaned_body = clean_article_content(content, max_chars=1500)
        if not cleaned_body:
            return {
                "summary": title.strip(),
                "model": "empty-content",
                "elapsed_ms": 0.0,
                "used_slm": False,
            }

        server_ok = self.ensure_local_slm_server()
        if server_ok:
            try:
                from openai import OpenAI

                client = OpenAI(base_url=self.base_url, api_key=self.api_key or "local-slm")
                active_model = self.model_id
                try:
                    models = [m.id for m in client.models.list().data]
                    if models and active_model not in models:
                        kanana_matches = [m for m in models if "kanana" in m.lower()]
                        active_model = kanana_matches[0] if kanana_matches else models[0]
                except Exception:
                    pass

                system_prompt = (
                    "너는 한국어 뉴스 기사를 정확하고 간결하게 요약하는 AI 요약기야. "
                    "기사에 없는 내용은 절대 지어내지 말고, 핵심 사실과 요지를 한국어 2~3문장으로만 요약해."
                )
                user_prompt = (
                    f"[기사 제목]\n{title.strip()}\n\n"
                    f"[기사 본문]\n{cleaned_body}\n\n"
                    "위 기사의 핵심 내용을 군더더기 없이 2~3문장으로 요약해줘:"
                )

                resp = client.chat.completions.create(
                    model=active_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.1,
                    max_tokens=max_tokens,
                    timeout=45.0,
                )
                raw_summary = (resp.choices[0].message.content or "").strip()
                summary_clean = re.sub(r"^(\*\*요약\*\*|요약\s*[:：]|#+\s*요약)\s*", "", raw_summary).strip()
                summary_clean = re.sub(r"\s+", " ", summary_clean)

                if summary_clean:
                    elapsed = round((time.perf_counter() - t0) * 1000, 1)
                    return {
                        "summary": summary_clean,
                        "model": active_model,
                        "elapsed_ms": elapsed,
                        "used_slm": True,
                    }
            except Exception as e:
                print(f"[경고] Kanana SLM 요약 호출 오류 ({e}) -> 추출형 요약 폴백 사용")

        fallback = self.fallback_extractive_summary(title, content)
        elapsed = round((time.perf_counter() - t0) * 1000, 1)
        return {
            "summary": fallback,
            "model": "extractive-fallback",
            "elapsed_ms": elapsed,
            "used_slm": False,
        }

    def summarize_news_by_pk(self, news_pk: int, force: bool = False) -> dict[str, Any] | None:
        """
        news.db의 NEWS 테이블에서 특정 PK의 CONTENT를 읽어 요약하고 DB(NEWS 및 RAG_INDEX)에 저장합니다.
        """
        with connect_summary_db(self.db_path) as con:
            cur = con.cursor()
            row = cur.execute(
                """
                SELECT N.PK, N.TITLE, N.CONTENT, COALESCE(N.SUMMARY, ''), COALESCE(N.SUMMARY_MODEL, ''),
                       COALESCE(N.SUMMARY_AT, ''), COALESCE(N.CATEGORY, '시사·종합'), COALESCE(L.URL, '')
                FROM NEWS N
                LEFT JOIN LINK L ON N.FK = L.PK
                WHERE N.PK = ?
                """,
                (news_pk,),
            ).fetchone()

        if not row:
            return None

        pk, title, content, existing_summary, existing_model, existing_at, category, url = row
        if existing_summary.strip() and not force:
            return {
                "news_pk": pk,
                "title": title,
                "category": category,
                "url": url,
                "summary": existing_summary,
                "model": existing_model or self.model_id,
                "summary_at": existing_at,
                "cached": True,
                "elapsed_ms": 0.0,
            }

        res = self.summarize_text(title=title, content=content)
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with connect_summary_db(self.db_path) as con:
            cur = con.cursor()
            cur.execute(
                """
                UPDATE NEWS
                SET SUMMARY = ?, SUMMARY_MODEL = ?, SUMMARY_AT = ?
                WHERE PK = ?
                """,
                (res["summary"], res["model"], now_str, pk),
            )
            try:
                cur.execute(
                    "UPDATE RAG_INDEX SET SUMMARY = ? WHERE NEWS_PK = ?",
                    (res["summary"], pk),
                )
            except sqlite3.OperationalError:
                pass
            con.commit()

        return {
            "news_pk": pk,
            "title": title,
            "category": category,
            "url": url,
            "summary": res["summary"],
            "model": res["model"],
            "summary_at": now_str,
            "cached": False,
            "used_slm": res["used_slm"],
            "elapsed_ms": res["elapsed_ms"],
        }

    def summarize_batch(
        self,
        limit: int = 20,
        force: bool = False,
        category: str | None = None,
        render_url: str | None = None,
    ) -> dict[str, Any]:
        """
        news.db의 NEWS 테이블에서 아직 요약되지 않은 기사들을 읽어 순차적으로 요약하고 DB에 저장합니다.
        render_url이 지정된 경우 요약 완료된 데이터를 Render($0) 웹서비스로 즉시 Push 동기화합니다.
        """
        where_clauses = ["length(TRIM(CONTENT)) >= 120", "TITLE NOT LIKE '[포토]%'"]
        params: list[Any] = []

        if not force:
            where_clauses.append("(SUMMARY IS NULL OR TRIM(SUMMARY) = '')")
        if category and category != "전체":
            where_clauses.append("CATEGORY = ?")
            params.append(category)

        where_sql = " AND ".join(where_clauses)
        with connect_summary_db(self.db_path) as con:
            cur = con.cursor()
            target_pks = [
                r[0]
                for r in cur.execute(
                    f"SELECT PK FROM NEWS WHERE {where_sql} ORDER BY PK DESC LIMIT ?",
                    (*params, limit),
                ).fetchall()
            ]

        results: list[dict[str, Any]] = []
        print(f"[배치 요약 시작] 대상 기사 {len(target_pks)}건 (모델: {self.model_id})")
        for idx, pk in enumerate(target_pks, 1):
            item = self.summarize_news_by_pk(pk, force=force)
            if item:
                results.append(item)
                short_title = item["title"][:32] + ("..." if len(item["title"]) > 32 else "")
                print(
                    f"  [{idx}/{len(target_pks)}] PK={pk} ({item['elapsed_ms']}ms) | "
                    f"{short_title} -> {item['summary'][:55]}..."
                )

        push_result = None
        target_render = (render_url or DEFAULT_RENDER_URL).strip()
        if target_render and results:
            push_result = self.push_summaries_to_render(target_render, items=results)

        stats = self.get_summary_stats()
        return {
            "processed_count": len(results),
            "items": results,
            "stats": stats,
            "render_push": push_result,
        }

    def get_summary_stats(self) -> dict[str, Any]:
        """news.db 내 전체 기사 수 및 요약 완료 현황 통계를 반환합니다."""
        with connect_summary_db(self.db_path) as con:
            cur = con.cursor()
            total = cur.execute("SELECT COUNT(*) FROM NEWS").fetchone()[0]
            summarized = cur.execute(
                "SELECT COUNT(*) FROM NEWS WHERE SUMMARY IS NOT NULL AND TRIM(SUMMARY) != ''"
            ).fetchone()[0]
            slm_count = cur.execute(
                "SELECT COUNT(*) FROM NEWS WHERE SUMMARY_MODEL LIKE '%kanana%' OR SUMMARY_MODEL LIKE '%GGUF%'"
            ).fetchone()[0]
        return {
            "total_news": total,
            "summarized_news": summarized,
            "slm_summarized_news": slm_count,
            "remaining_news": max(0, total - summarized),
            "progress_pct": round(summarized * 100.0 / max(1, total), 1),
            "model": self.model_id,
            "slm_server_online": self.is_llm_server_online(),
            "slm_base_url": self.base_url,
        }

    def list_summaries(
        self,
        limit: int = 30,
        offset: int = 0,
        category: str | None = None,
        keyword: str | None = None,
        only_summarized: bool = True,
    ) -> list[dict[str, Any]]:
        """저장된 뉴스 요약 목록을 조회합니다."""
        where_clauses = ["1=1"]
        params: list[Any] = []
        if only_summarized:
            where_clauses.append("N.SUMMARY IS NOT NULL AND TRIM(N.SUMMARY) != ''")
        if category and category != "전체":
            where_clauses.append("N.CATEGORY = ?")
            params.append(category)
        if keyword and keyword.strip():
            kw = f"%{keyword.strip()}%"
            where_clauses.append("(N.TITLE LIKE ? OR N.CONTENT LIKE ? OR N.SUMMARY LIKE ?)")
            params.extend([kw, kw, kw])

        where_sql = " AND ".join(where_clauses)
        with connect_summary_db(self.db_path) as con:
            cur = con.cursor()
            rows = cur.execute(
                f"""
                SELECT
                    N.PK,
                    N.TITLE,
                    COALESCE(N.CLEAN_TITLE, ''),
                    COALESCE(N.CATEGORY, '시사·종합'),
                    COALESCE(N.CLUSTER_LABEL, ''),
                    COALESCE(N.SUMMARY, ''),
                    COALESCE(N.SUMMARY_MODEL, ''),
                    COALESCE(N.SUMMARY_AT, ''),
                    COALESCE(N.REGDATE, ''),
                    COALESCE(L.URL, ''),
                    SUBSTR(N.CONTENT, 1, 240)
                FROM NEWS N
                LEFT JOIN LINK L ON N.FK = L.PK
                WHERE {where_sql}
                ORDER BY CASE WHEN N.SUMMARY != '' THEN 0 ELSE 1 END, N.PK DESC
                LIMIT ? OFFSET ?
                """,
                (*params, limit, offset),
            ).fetchall()

        out: list[dict[str, Any]] = []
        for pk, title, clean_t, cat, clabel, summary, smodel, sat, regdate, url, snippet in rows:
            out.append(
                {
                    "news_pk": pk,
                    "title": title,
                    "clean_title": clean_t,
                    "category": cat,
                    "cluster_label": clabel,
                    "summary": summary,
                    "summary_model": smodel,
                    "summary_at": sat,
                    "regdate": regdate,
                    "url": url,
                    "content_snippet": snippet,
                }
            )
        return out

    def push_summaries_to_render(
        self,
        render_base_url: str,
        items: list[dict[str, Any]] | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        """
        노트북에서 요약한 NEWS.SUMMARY 데이터를 Render($0 무료 요금제) 웹서비스의
        /api/receive-summaries 엔드포인트로 전송(Push)합니다.
        """
        clean_url = render_base_url.rstrip("/")
        endpoint = f"{clean_url}/api/receive-summaries"

        if items is None:
            items = self.list_summaries(limit=limit, only_summarized=True)

        payload_items = [
            {
                "news_pk": it["news_pk"],
                "title": it.get("title", ""),
                "summary": it.get("summary", ""),
                "model": it.get("model") or it.get("summary_model") or self.model_id,
                "summary_at": it.get("summary_at", ""),
            }
            for it in items
            if it.get("summary")
        ]

        if not payload_items:
            return {"status": "empty", "pushed_count": 0, "target": endpoint}

        headers = {"Content-Type": "application/json"}
        if SUMMARY_SYNC_TOKEN:
            headers["X-Summary-Token"] = SUMMARY_SYNC_TOKEN

        try:
            resp = httpx.post(
                endpoint,
                json={"summaries": payload_items},
                headers=headers,
                timeout=20.0,
            )
            resp.raise_for_status()
            data = resp.json()
            print(f"[Render 동기화 성공] {len(payload_items)}건의 요약을 {endpoint} 로 전송했습니다.")
            return {
                "status": "ok",
                "pushed_count": len(payload_items),
                "target": endpoint,
                "remote_response": data,
            }
        except Exception as e:
            print(f"[Render 동기화 실패] {endpoint} 전송 중 오류: {e}")
            return {
                "status": "error",
                "pushed_count": 0,
                "target": endpoint,
                "error": str(e),
            }


__all__ = [
    "DEFAULT_RENDER_URL",
    "SUMMARY_SYNC_TOKEN",
    "connect_summary_db",
    "ensure_summary_columns",
    "clean_article_content",
    "find_local_gguf_path",
    "find_llama_server_exe",
    "LocalKananaSummarizer",
]
