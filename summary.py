"""
summary.py - 노트북 4GB VRAM GPU 기반 로컬 SLM 뉴스 요약기 & Render($0 요금제) 웹서비스 제공 서버

[핵심 역할]
1. news.db의 NEWS 테이블에서 CONTENT를 읽어와 로컬 SLM
   (kenonix/kanana-nano-2.1b-instruct-Q8_0-GGUF)으로 한국어 핵심 요약 생성
2. VRAM 4GB GPU(예: GTX 1650 Ti 4096MB) 환경에 맞춘 메모리 최적화 구동:
   - Unsloth llama-server.exe 또는 로컬 OpenAI 호환 서버(127.0.0.1:8889) 자동 연동·자동 실행
   - 컨텍스트 길이 제한(-c 2048), KV 캐시 8비트 양자화(-ctk q8_0 -ctv q8_0), 단일 병렬 슬롯(--parallel 1)으로
     2.22GB Q8_0 모델을 4GB VRAM 안에서 OOM 없이 100% GPU 가속 실행
3. 요약 결과를 news.db (NEWS.SUMMARY 및 RAG_INDEX.SUMMARY)에 저장하고 웹서비스에 제공:
   - [모드 A: 배치 요약 + Render 전송] 로컬 DB 요약 저장 후 Render($0) 웹서비스(/api/receive-summaries)로 Push
   - [모드 B: 요약 API & 대시보드 서버] 웹서비스가 호출할 수 있는 REST API(/api/summaries, /api/summarize) 및
     OpenAI 호환 엔드포인트(/v1/chat/completions) 제공

[실행 예시]
- 기본 실행 (미요약 기사 10건 우선 요약 후 요약 제공 웹서버 기동):
    python summary.py
- 배치 요약만 실행 (예: 30건 요약 후 종료):
    python summary.py --mode batch --limit 30
- 요약 후 Render 웹서비스로 자동 Push:
    python summary.py --mode batch --limit 20 --render-url https://내서비스.onrender.com
- 요약 제공 API 서버만 즉시 실행 (기본 포트 8001):
    python summary.py --mode serve --port 8001
"""

from __future__ import annotations

import argparse
import atexit
from datetime import datetime
import json
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import sys
import time
from typing import Any
import urllib.error
import urllib.request

from dotenv import load_dotenv
import httpx

ROOT_DIR = Path(__file__).resolve().parent
ENV_FILE = ROOT_DIR / ".env"
DEFAULT_DB_PATH = ROOT_DIR / "news.db"

if ENV_FILE.exists():
    load_dotenv(ENV_FILE)

# 기본 모델 및 로컬 SLM 서버 설정 (유료 OpenAI API가 아닌 로컬 Unsloth / llama-server 전용)
DEFAULT_MODEL_REPO = os.environ.get("UNSLOTH_MODEL", "kenonix/kanana-nano-2.1b-instruct-Q8_0-GGUF").strip()
DEFAULT_GGUF_FILENAME = "kanana-nano-2.1b-instruct-q8_0.gguf"
DEFAULT_LLAMA_SERVER_PORT = 8891
DEFAULT_LLAMA_BASE_URL = f"http://127.0.0.1:{DEFAULT_LLAMA_SERVER_PORT}/v1"
DEFAULT_API_KEY = os.environ.get(
    "UNSLOTH_API_KEY",
    "sk-unsloth-f65f99d826c89c73362ddeb617e7963e",
).strip()
DEFAULT_SUMMARY_PORT = int(os.environ.get("SUMMARY_PORT", "8001"))
DEFAULT_RENDER_URL = os.environ.get("RENDER_SERVICE_URL", "").strip()
SUMMARY_SYNC_TOKEN = os.environ.get("SUMMARY_SYNC_TOKEN", "").strip()

EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
PHONE_RE = re.compile(r"\b0\d{1,2}[-.\s]?\d{3,4}[-.\s]?\d{4}\b")


def clean_article_content(content: str, max_chars: int = 1600) -> str:
    """
    NEWS.CONTENT에서 기자 이메일, 연락처, 저작권 문구, 과도한 공백을 제거하고
    4GB VRAM 컨텍스트(2048 토큰)에 맞게 핵심 본문 길이를 조절합니다.
    """
    if not content:
        return ""
    text = EMAIL_RE.sub("", content)
    text = PHONE_RE.sub("", text)
    # 불필요한 언론사 저작권/무단전재 문구 제거
    text = re.sub(r"무단\s*전재\s*(및\s*재배포\s*금지|금지).*$", "", text)
    text = re.sub(r"ⓒ\s*[^\n]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0] + "..."
    return text


def ensure_summary_columns(db_path: str | Path = DEFAULT_DB_PATH) -> None:
    """
    news.db의 NEWS 테이블과 RAG_INDEX 테이블에 SUMMARY, SUMMARY_MODEL, SUMMARY_AT 컬럼을 보장합니다.
    """
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"news.db 파일을 찾을 수 없습니다: {db_path}")

    with sqlite3.connect(str(db_path)) as con:
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


def find_local_gguf_path(repo_id: str = DEFAULT_MODEL_REPO, filename: str = DEFAULT_GGUF_FILENAME) -> Path | None:
    """
    로컬 HuggingFace 캐시 폴더에서 kanana-nano-2.1b-instruct-q8_0.gguf 파일 경로를 탐색하고,
    없으면 huggingface_hub를 통해 다운로드를 시도합니다.
    """
    hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
    model_dir_name = "models--" + repo_id.replace("/", "--")
    snapshots_dir = hf_cache / model_dir_name / "snapshots"

    if snapshots_dir.exists():
        for gguf_file in snapshots_dir.rglob("*.gguf"):
            if gguf_file.is_file() and gguf_file.stat().st_size > 100_000_000:
                return gguf_file

    # 현재 작업 폴더 내부에 GGUF 파일이 있는지도 확인
    for local_candidate in ROOT_DIR.glob("*.gguf"):
        if "kanana" in local_candidate.name.lower():
            return local_candidate

    # 캐시에 없으면 huggingface_hub로 다운로드 시도
    try:
        from huggingface_hub import hf_hub_download

        print(f"[모델 다운로드] HuggingFace에서 '{repo_id}/{filename}' 파일을 가져옵니다...")
        downloaded = hf_hub_download(repo_id=repo_id, filename=filename)
        return Path(downloaded)
    except Exception as e:
        print(f"[경고] GGUF 모델 자동 다운로드 실패: {e}")
        return None


def find_llama_server_exe() -> Path | None:
    """
    노트북에 설치된 Unsloth llama.cpp의 llama-server.exe 경로를 찾습니다.
    """
    candidates = [
        Path.home() / ".unsloth" / "llama.cpp" / "build" / "bin" / "Release" / "llama-server.exe",
        Path.home() / ".unsloth" / "llama.cpp" / "build" / "bin" / "llama-server.exe",
        Path.home() / ".unsloth" / "llama.cpp" / "llama-server.exe",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


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
        """
        해당 base_url의 /models 엔드포인트가 실제 VRAM에 모델이 적재된
        llama-server OpenAI 호환 JSON을 반환하는지 검사합니다.
        """
        models_url = f"{base_url.rstrip('/')}/models"
        try:
            req = urllib.request.Request(
                models_url,
                headers={"Authorization": f"Bearer {api_key}"},
            )
            with urllib.request.urlopen(req, timeout=2.5) as resp:
                if resp.status != 200:
                    return False
                payload = json.loads(resp.read().decode("utf-8", errors="ignore"))
                if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
                    return False
                data_list = payload["data"]
                if not data_list:
                    return False
                # llama-server는 실제 모델이 메모리에 로드되어 있을 때 owned_by="llamacpp" 또는 meta 정보를 포함함
                return any(
                    isinstance(m, dict) and (m.get("owned_by") == "llamacpp" or "meta" in m)
                    for m in data_list
                )
        except Exception:
            return False

    def is_llm_server_online(self) -> bool:
        """
        로컬 OpenAI 호환 SLM 추론 서버(8891, 8889 등)가 응답하는지 확인하고,
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
        import socket

        for p in preferred_ports:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if s.connect_ex(("127.0.0.1", p)) != 0:
                    return p
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            return int(s.getsockname()[1])

    def ensure_local_slm_server(self) -> bool:
        """
        로컬 SLM 추론 서버가 켜져 있지 않으면 비어 있는 포트를 찾아
        4GB VRAM 최적화 파라미터로 llama-server.exe를 자동 실행합니다.
        """
        if self.is_llm_server_online():
            return True

        if not self.auto_start_server:
            return False

        llama_exe = find_llama_server_exe()
        gguf_path = find_local_gguf_path()

        if not llama_exe or not gguf_path:
            print(
                f"[안내] llama-server.exe({llama_exe}) 또는 GGUF 파일({gguf_path})을 찾을 수 없습니다.\n"
                "       Unsloth 서버(127.0.0.1:8888 또는 8889)를 먼저 실행하거나 모델 캐시를 확인하세요."
            )
            return False

        port = self._find_free_port()
        target_base_url = f"http://127.0.0.1:{port}/v1"

        # 4GB VRAM GPU 맞춤형 실행 인자:
        # -ngl 99: 레이어 GPU 오프로딩 (2.22GB Q8_0 모델이 4GB VRAM에 적재됨)
        # -c 2048: 뉴스 요약에 충분한 컨텍스트 크기로 제한하여 KV 캐시 VRAM 절약
        # -ctk q8_0 -ctv q8_0: KV 캐시를 8비트로 양자화해 VRAM 사용량 최소화
        # --parallel 1: 동시 슬롯 1개로 고정하여 불필요한 버퍼 할당 방지
        for ngl_try in (self.n_gpu_layers, 20, 0):
            cmd = [
                str(llama_exe),
                "-m",
                str(gguf_path),
                "--alias",
                self.model_id,
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "-ngl",
                str(ngl_try),
                "-c",
                str(self.ctx_size),
                "-b",
                "512",
                "-ub",
                "256",
                "-ctk",
                "q8_0",
                "-ctv",
                "q8_0",
                "--parallel",
                "1",
            ]
            print(
                f"[SLM 서버 기동] 4GB VRAM 최적화 모드로 '{gguf_path.name}' 모델을 로드합니다 "
                f"(port={port}, n_gpu_layers={ngl_try}, ctx={self.ctx_size})..."
            )
            env = os.environ.copy()
            extra_cuda_paths = [
                Path.home() / ".unsloth" / "studio" / "unsloth_studio" / "Lib" / "site-packages" / "torch" / "lib",
                llama_exe.parent,
            ]
            existing_extra = [str(p) for p in extra_cuda_paths if p.exists()]
            if existing_extra:
                env["PATH"] = os.pathsep.join(existing_extra) + os.pathsep + env.get("PATH", "")

            try:
                self._llama_proc = subprocess.Popen(
                    cmd,
                    cwd=str(llama_exe.parent),
                    env=env,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except Exception as e:
                print(f"[오류] llama-server 실행 실패: {e}")
                return False

            # 모델이 VRAM에 올라올 때까지 최대 25초 대기
            start_wait = time.time()
            while time.time() - start_wait < 25.0:
                if self._llama_proc.poll() is not None:
                    break
                if self._probe_openai_models_url(target_base_url, self.api_key):
                    self.base_url = target_base_url
                    print(f"[SLM 준비 완료] {self.model_id} 서버가 {self.base_url} 에서 응답 중입니다.")
                    return True
                time.sleep(0.5)

            print(f"[재시도] n_gpu_layers={ngl_try} 기동 실패 -> 레이어 수를 조정하여 재시도합니다.")
            self.stop_local_server()

        return False

    def stop_local_server(self) -> None:
        """summary.py가 직접 띄운 llama-server 프로세스가 있으면 종료합니다."""
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

                client = OpenAI(base_url=self.base_url, api_key=self.api_key)
                # 서버에서 인식된 모델 확인
                active_model = self.model_id
                try:
                    models = [m.id for m in client.models.list().data]
                    if models and active_model not in models:
                        # kanana 모델이 목록에 있으면 우선 선택
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
                # 불필요한 마크다운 기호나 머리말 정리
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
        with sqlite3.connect(str(self.db_path)) as con:
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

        with sqlite3.connect(str(self.db_path)) as con:
            cur = con.cursor()
            cur.execute(
                """
                UPDATE NEWS
                SET SUMMARY = ?, SUMMARY_MODEL = ?, SUMMARY_AT = ?
                WHERE PK = ?
                """,
                (res["summary"], res["model"], now_str, pk),
            )
            # RAG_INDEX가 존재하면 함께 업데이트
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
        with sqlite3.connect(str(self.db_path)) as con:
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
        with sqlite3.connect(str(self.db_path)) as con:
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
        with sqlite3.connect(str(self.db_path)) as con:
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


# =====================================================================
# 웹서비스 제공용 FastAPI 애플리케이션 (노트북 요약 서버 & 대시보드)
# =====================================================================


def create_summary_app(summarizer: LocalKananaSummarizer | None = None):
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import HTMLResponse
    from pydantic import BaseModel, Field

    engine = summarizer or LocalKananaSummarizer()

    app = FastAPI(
        title="Kanana-Nano 2.1B 로컬 GPU 뉴스 요약 서버 (summary.py)",
        description=(
            "VRAM 4GB 노트북에서 kenonix/kanana-nano-2.1b-instruct-Q8_0-GGUF 모델을 돌려 "
            "news.db의 NEWS.CONTENT를 요약하고 Render($0) 웹서비스에 제공하는 API 서버"
        ),
    )

    # Render에 배포된 웹서비스 브라우저/백엔드 어디서든 호출 가능하도록 CORS 허용
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
        <h2 style="margin:0 0 6px 0;">노트북 4GB GPU 뉴스 요약 서버 (summary.py)</h2>
        <div style="font-size:13px;color:var(--muted);">
          모델: <code>kenonix/kanana-nano-2.1b-instruct-Q8_0-GGUF</code> · DB: <code>news.db (NEWS.CONTENT → NEWS.SUMMARY)</code>
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

                client = OpenAI(base_url=engine.base_url, api_key=engine.api_key)
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


app = create_summary_app()


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

    # 특정 PK 1건 요약 요청 시
    if args.pk is not None:
        res = summarizer.summarize_news_by_pk(args.pk, force=True)
        print(json.dumps(res, ensure_ascii=False, indent=2))
        if args.render_url and res:
            summarizer.push_summaries_to_render(args.render_url, items=[res])
        return

    # 배치 요약만 수행 후 종료하는 모드
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

    # 방법 B (serve / auto): 4GB GPU SLM 서버를 먼저 기동하고 백그라운드 요약 + 요약 제공 웹서버 즉시 실행
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

    server_app = create_summary_app(summarizer)
    uvicorn.run(server_app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
