"""
slm_server.py - 파인튜닝 완료된 요약기(kanana_summarizer_full_ft_model/model.safetensors) 로드·추론 및
로컬 OpenAI 호환 SLM 서버 탐색·기동 모듈

[핵심 역할]
1. find_finetuned_safetensors_model() / get_finetuned_summarizer_info():
   `kanana_summarizer_full_ft_model/model.safetensors` 및 `config.json`, `tokenizer.json` 존재 여부와 메타데이터 확인
2. load_finetuned_safetensors_pipeline() / generate_summary_with_safetensors():
   HuggingFace Transformers(`use_safetensors=True`)로 파인튜닝된 Kanana-Nano-2.1B 모델을 로드하여 뉴스 요약 생성
3. find_local_gguf_path() / find_llama_server_exe() / start_optimized_llama_server():
   외부 OpenAI 호환 추론 서버(llama-server.exe 등) 연동 및 폴백 지원
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import subprocess
import time
from typing import Any
import urllib.request

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent
ENV_FILE = ROOT_DIR / ".env"
if ENV_FILE.exists():
    load_dotenv(ENV_FILE)

FINETUNED_SUMMARIZER_DIR = ROOT_DIR / os.environ.get(
    "SUMMARIZER_MODEL_DIR", "kanana_summarizer_full_ft_model"
).strip()
FINETUNED_SAFETENSORS_FILE = FINETUNED_SUMMARIZER_DIR / "model.safetensors"

DEFAULT_MODEL_REPO = os.environ.get(
    "UNSLOTH_MODEL", "kanana_summarizer_full_ft_model"
).strip()
DEFAULT_GGUF_FILENAME = "kanana-nano-2.1b-instruct-q8_0.gguf"
DEFAULT_LLAMA_SERVER_PORT = int(os.environ.get("LLAMA_SERVER_PORT", "8891"))
DEFAULT_LLAMA_BASE_URL = f"http://127.0.0.1:{DEFAULT_LLAMA_SERVER_PORT}/v1"
DEFAULT_API_KEY = os.environ.get("UNSLOTH_API_KEY", "").strip()

_SAFETENSORS_PIPELINE_CACHE: dict[str, Any] | None = None


def find_finetuned_safetensors_model(
    model_dir: str | Path = FINETUNED_SUMMARIZER_DIR,
) -> Path | None:
    """
    파인튜닝 완료된 요약기 폴더(`kanana_summarizer_full_ft_model`) 내 `model.safetensors` 파일 경로를 확인합니다.
    """
    target_dir = Path(model_dir)
    safetensors_path = target_dir / "model.safetensors"
    if safetensors_path.is_file() and safetensors_path.stat().st_size > 1_000_000:
        return safetensors_path
    return None


def get_finetuned_summarizer_info(
    model_dir: str | Path = FINETUNED_SUMMARIZER_DIR,
) -> dict[str, Any]:
    """
    `kanana_summarizer_full_ft_model` 폴더의 `model.safetensors` 및 `config.json` 정보를 반환합니다.
    """
    target_dir = Path(model_dir)
    st_path = find_finetuned_safetensors_model(target_dir)
    cfg_path = target_dir / "config.json"

    cfg_data: dict[str, Any] = {}
    if cfg_path.is_file():
        try:
            cfg_data = json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception:
            cfg_data = {}

    size_bytes = st_path.stat().st_size if st_path else 0
    return {
        "model_name": "kanana_summarizer_full_ft_model",
        "base_model": "kakaocorp/kanana-nano-2.1b-instruct",
        "model_dir": str(target_dir),
        "safetensors_file": str(st_path or (target_dir / "model.safetensors")),
        "exists": st_path is not None,
        "size_bytes": size_bytes,
        "size_gb": round(size_bytes / (1024**3), 2) if size_bytes else 0.0,
        "architectures": cfg_data.get("architectures", ["LlamaForCausalLM"]),
        "dtype": cfg_data.get("dtype", "bfloat16"),
        "vocab_size": cfg_data.get("vocab_size", 128256),
        "max_position_embeddings": cfg_data.get("max_position_embeddings", 8192),
    }


def load_finetuned_safetensors_pipeline(
    model_dir: str | Path = FINETUNED_SUMMARIZER_DIR,
) -> dict[str, Any] | None:
    """
    `kanana_summarizer_full_ft_model/model.safetensors` 가중치와 토크나이저를 HuggingFace Transformers로 로드합니다.
    """
    global _SAFETENSORS_PIPELINE_CACHE
    if _SAFETENSORS_PIPELINE_CACHE is not None:
        return _SAFETENSORS_PIPELINE_CACHE

    st_path = find_finetuned_safetensors_model(model_dir)
    if st_path is None:
        return None

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        target_dir = str(st_path.parent)
        tokenizer = AutoTokenizer.from_pretrained(target_dir, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else (
            torch.float16 if device == "cuda" else torch.float32
        )

        model = AutoModelForCausalLM.from_pretrained(
            target_dir,
            use_safetensors=True,
            torch_dtype=dtype,
            device_map="auto" if device == "cuda" else None,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        if device == "cpu":
            model.to("cpu")
        model.eval()

        _SAFETENSORS_PIPELINE_CACHE = {
            "model": model,
            "tokenizer": tokenizer,
            "device": device,
            "model_dir": target_dir,
            "safetensors_file": str(st_path),
        }
        return _SAFETENSORS_PIPELINE_CACHE
    except Exception as e:
        print(f"[안내] kanana_summarizer_full_ft_model/model.safetensors 직접 로드 생략({e})")
        return None


def generate_summary_with_safetensors(
    title: str,
    content: str,
    max_new_tokens: int = 180,
    model_dir: str | Path = FINETUNED_SUMMARIZER_DIR,
) -> tuple[bool, str, str]:
    """
    파인튜닝된 `kanana_summarizer_full_ft_model/model.safetensors` 모델을 사용하여 뉴스 요약을 생성합니다.
    반환값: (성공 여부, 생성된 요약 텍스트, 사용된 모델 식별자)
    """
    bundle = load_finetuned_safetensors_pipeline(model_dir)
    if bundle is None:
        return False, "", ""

    try:
        import torch

        tokenizer = bundle["tokenizer"]
        model = bundle["model"]
        device = bundle["device"]

        messages = [
            {
                "role": "system",
                "content": (
                    "너는 한국어 뉴스 기사를 정확하고 간결하게 요약하는 AI 요약기야. "
                    "기사에 없는 내용은 절대 지어내지 말고, 핵심 사실과 요지를 한국어 2~3문장으로만 요약해."
                ),
            },
            {
                "role": "user",
                "content": f"[제목] {title}\n[본문]\n{content}\n\n위 기사의 핵심 내용을 2~3문장으로 요약해 줘.",
            },
        ]

        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
        input_len = inputs["input_ids"].shape[-1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        gen_tokens = outputs[0][input_len:]
        summary_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        if summary_text:
            return True, summary_text, "kanana_summarizer_full_ft_model (model.safetensors)"
    except Exception as e:
        print(f"[경고] model.safetensors 추론 실패: {e}")

    return False, "", ""


def find_local_gguf_path(repo_id: str = DEFAULT_MODEL_REPO, filename: str = DEFAULT_GGUF_FILENAME) -> Path | None:
    """
    로컬 작업 폴더 또는 HuggingFace 캐시 폴더에서 GGUF 파일 경로를 탐색합니다.
    """
    for local_candidate in ROOT_DIR.glob("*.gguf"):
        if "kanana" in local_candidate.name.lower():
            return local_candidate

    hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
    model_dir_name = "models--" + repo_id.replace("/", "--")
    snapshots_dir = hf_cache / model_dir_name / "snapshots"

    if snapshots_dir.exists():
        for gguf_file in snapshots_dir.rglob("*.gguf"):
            if gguf_file.is_file() and gguf_file.stat().st_size > 100_000_000:
                return gguf_file

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
    해당 base_url의 /models 엔드포인트가 실제 모델이 적재된 OpenAI 호환 JSON을 반환하는지 검사합니다.
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
                isinstance(m, dict) and (m.get("owned_by") == "llamacpp" or "meta" in m or "id" in m)
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
    로컬 GGUF 파일과 llama-server.exe가 모두 있을 경우 OpenAI 호환 추론 서버를 자동 실행합니다.
    (`kanana_summarizer_full_ft_model/model.safetensors` 사용 시에는 HuggingFace Transformers 직접 추론을 우선 지원합니다.)
    """
    st_path = find_finetuned_safetensors_model()
    llama_exe = find_llama_server_exe()
    gguf_path = find_local_gguf_path(repo_id=model_id)

    if not llama_exe or not gguf_path:
        if st_path is not None:
            print(
                f"[안내] 파인튜닝된 요약기 모델이 확인되었습니다: {st_path} "
                f"({round(st_path.stat().st_size / (1024**3), 2)} GB)"
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

        if proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                pass

    return None, None


__all__ = [
    "FINETUNED_SUMMARIZER_DIR",
    "FINETUNED_SAFETENSORS_FILE",
    "DEFAULT_MODEL_REPO",
    "DEFAULT_GGUF_FILENAME",
    "DEFAULT_LLAMA_SERVER_PORT",
    "DEFAULT_LLAMA_BASE_URL",
    "DEFAULT_API_KEY",
    "find_finetuned_safetensors_model",
    "get_finetuned_summarizer_info",
    "load_finetuned_safetensors_pipeline",
    "generate_summary_with_safetensors",
    "find_local_gguf_path",
    "find_llama_server_exe",
    "probe_openai_models_url",
    "find_free_port",
    "start_optimized_llama_server",
]
