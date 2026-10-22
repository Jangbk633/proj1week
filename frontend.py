"""
frontend.py - 확정된 newslens.html UI 디자인 + news.db RAG 에이전트(rag_agent.py) 연동 서버
실행 방법: python frontend.py
"""

from pathlib import Path
import subprocess
import sys
import time
import urllib.error
import urllib.request
import venv
import webbrowser

ROOT_DIR = Path(__file__).resolve().parent
VENV_DIR = ROOT_DIR / ".venv"
REQ_FILE = ROOT_DIR / "requirements.txt"
HTML_FILE = ROOT_DIR / "newslens.html"
DB_FILE = ROOT_DIR / "news.db"
HOST = "127.0.0.1"
PORT = 8000
URL = f"http://{HOST}:{PORT}"


def create_fastapi_app():
    """FastAPI 앱을 생성하여 확정된 newslens.html UI와 news.db 기반 /api/analyze RAG 엔드포인트를 제공합니다."""
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse
    from pydantic import BaseModel
    from rag_agent import NewsLensAgent

    app = FastAPI(
        title="Nexus (논점콩) - news.db 기반 신문사별 관점 비교·요약 AI 에이전트",
        description="확정된 newslens.html 디자인 + news.db (SQLite) 하이브리드 RAG + LangGraph 에이전트 서버",
    )

    rag_agent = NewsLensAgent(db_path=DB_FILE)

    class AnalyzeRequest(BaseModel):
        query: str
        thread_id: str = "default-session"

    class PressItem(BaseModel):
        name: str
        title: str
        url: str = "#source"

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

    @app.get("/", response_class=HTMLResponse)
    async def serve_newslens_ui() -> str:
        """확정된 newslens.html 디자인 파일을 반환합니다."""
        return HTML_FILE.read_text(encoding="utf-8")

    @app.get("/api/issues")
    async def list_popular_issues():
        """news.db에 적재된 대표 이슈 목록을 반환합니다."""
        return {"issues": rag_agent.rag.get_popular_issues(limit=5)}

    @app.post("/api/analyze", response_model=AnalyzeResponse)
    async def analyze_issue(req: AnalyzeRequest) -> AnalyzeResponse:
        """news.db 기반 LangGraph RAG 에이전트를 실행하여 신문사별 관점 비교 결과를 반환합니다."""
        result = rag_agent.run(query=req.query.strip(), thread_id=req.thread_id)
        return AnalyzeResponse(**result)

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
        [str(venv_python), "-c", "import fastapi, uvicorn, langgraph, numpy"],
        capture_output=True,
        text=True,
    )
    return check.returncode == 0


def install_dependencies(venv_python: Path) -> None:
    """requirements.txt 패키지를 가상환경에 설치합니다."""
    print("[2/4] 패키지 설치 상태를 점검합니다...")
    subprocess.run(
        [str(venv_python), "-m", "pip", "install", "--upgrade", "pip", "--quiet"],
        check=False,
    )

    if REQ_FILE.exists():
        print(f"      -> {REQ_FILE.name} 기준으로 패키지 설치를 시도합니다...")
        res = subprocess.run(
            [str(venv_python), "-m", "pip", "install", "-r", str(REQ_FILE)],
            cwd=str(ROOT_DIR),
        )
        if res.returncode == 0 and is_ui_ready(venv_python):
            print("      -> requirements.txt 전체 설치 완료!")
            return

    subprocess.run(
        [str(venv_python), "-m", "pip", "install", "fastapi", "uvicorn", "httpx", "python-dotenv", "numpy", "langgraph"],
        cwd=str(ROOT_DIR),
        check=True,
    )
    print("      -> 구동 패키지 설치 완료!")


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
        print("[2/4] 필수 패키지가 이미 가상환경에 설치되어 있습니다.")

    print(f"[3/4] news.db RAG + newslens.html 웹 서버를 시작합니다 ({URL})...")
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
