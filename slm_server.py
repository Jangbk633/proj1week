"""
slm_server.py - 로컬 GGUF 모델(kanana-nano-2.1b-instruct-q8_0.gguf) 및 Unsloth llama-server.exe 탐색·자동 기동 모듈

[핵심 역할]
1. find_local_gguf_path(): HuggingFace 로컬 캐시 및 작업 폴더에서 GGUF 모델 탐색(없으면 자동 다운로드)
2. find_llama_server_exe(): 노트북에 설치된 Unsloth llama.cpp의 llama-server.exe 경로 탐색
3. probe_openai_models_url() / find_free_port() / start_optimized_llama_server():
   4GB VRAM GPU(예: GTX 1650 Ti)에서 OOM 없이 2.22GB Q8_0 모델을 구동하도록
   `-c 2048 -b 512 -ub 256 -ctk q8_0 -ctv q8_0 --parallel 1` 파라미터로 llama-server.exe를 자동 실행
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import subprocess
import time
import urllib.request

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent
ENV_FILE = ROOT_DIR / ".env"
if ENV_FILE.exists():
    load_dotenv(ENV_FILE)

DEFAULT_MODEL_REPO = os.environ.get("UNSLOTH_MODEL", "kenonix/kanana-nano-2.1b-instruct-Q8_0-GGUF").strip()
DEFAULT_GGUF_FILENAME = "kanana-nano-2.1b-instruct-q8_0.gguf"
DEFAULT_LLAMA_SERVER_PORT = int(os.environ.get("LLAMA_SERVER_PORT", "8891"))
DEFAULT_LLAMA_BASE_URL = f"http://127.0.0.1:{DEFAULT_LLAMA_SERVER_PORT}/v1"
DEFAULT_API_KEY = os.environ.get("UNSLOTH_API_KEY", "").strip()


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


def probe_openai_models_url(base_url: str, api_key: str = DEFAULT_API_KEY) -> bool:
    """
    해당 base_url의 /models 엔드포인트가 실제 VRAM에 모델이 적재된
    llama-server OpenAI 호환 JSON을 반환하는지 검사합니다.
    """
    models_url = f"{base_url.rstrip('/')}/models"
    try:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        req = urllib.request.Request(
            models_url,
            headers=headers,
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
            return any(
                isinstance(m, dict) and (m.get("owned_by") == "llamacpp" or "meta" in m)
                for m in data_list
            )
    except Exception:
        return False


def find_free_port(preferred_ports: tuple[int, ...] = (8891, 8892, 11435)) -> int:
    """선호 포트 중 비어 있는 포트를 반환하거나 OS가 할당한 유휴 포트를 반환합니다."""
    for p in preferred_ports:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", p)) != 0:
                return p
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def start_optimized_llama_server(
    model_id: str = DEFAULT_MODEL_REPO,
    api_key: str = DEFAULT_API_KEY,
    n_gpu_layers: int = 99,
    ctx_size: int = 2048,
) -> tuple[subprocess.Popen | None, str | None]:
    """
    4GB VRAM 최적화 파라미터(`-c 2048 -b 512 -ub 256 -ctk q8_0 -ctv q8_0 --parallel 1`)로
    llama-server.exe를 자동 실행합니다. 실패 시 n_gpu_layers를 99 -> 20 -> 0 순으로 낮춰 재시도합니다.
    """
    llama_exe = find_llama_server_exe()
    gguf_path = find_local_gguf_path(repo_id=model_id)

    if not llama_exe or not gguf_path:
        print(
            f"[안내] llama-server.exe({llama_exe}) 또는 GGUF 파일({gguf_path})을 찾을 수 없습니다.\n"
            "       Unsloth 서버(127.0.0.1:8888 또는 8889)를 먼저 실행하거나 모델 캐시를 확인하세요."
        )
        return None, None

    port = find_free_port()
    target_base_url = f"http://127.0.0.1:{port}/v1"

    for ngl_try in (n_gpu_layers, 20, 0):
        cmd = [
            str(llama_exe),
            "-m",
            str(gguf_path),
            "--alias",
            model_id,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "-ngl",
            str(ngl_try),
            "-c",
            str(ctx_size),
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
            f"(port={port}, n_gpu_layers={ngl_try}, ctx={ctx_size})..."
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
            proc = subprocess.Popen(
                cmd,
                cwd=str(llama_exe.parent),
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as e:
            print(f"[오류] llama-server 실행 실패: {e}")
            return None, None

        start_wait = time.time()
        while time.time() - start_wait < 25.0:
            if proc.poll() is not None:
                break
            if probe_openai_models_url(target_base_url, api_key):
                print(f"[SLM 준비 완료] {model_id} 서버가 {target_base_url} 에서 응답 중입니다.")
                return proc, target_base_url
            time.sleep(0.5)

        print(f"[재시도] n_gpu_layers={ngl_try} 기동 실패 -> 레이어 수를 조정하여 재시도합니다.")
        if proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                pass

    return None, None


__all__ = [
    "DEFAULT_MODEL_REPO",
    "DEFAULT_GGUF_FILENAME",
    "DEFAULT_LLAMA_SERVER_PORT",
    "DEFAULT_LLAMA_BASE_URL",
    "DEFAULT_API_KEY",
    "find_local_gguf_path",
    "find_llama_server_exe",
    "probe_openai_models_url",
    "find_free_port",
    "start_optimized_llama_server",
]
