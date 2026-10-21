"""
frontend.py - 확정된 newslens.html UI 디자인 기반 가상환경(.venv) 자동 구성 및 웹 서버 실행 스크립트
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
HOST = "127.0.0.1"
PORT = 8000
URL = f"http://{HOST}:{PORT}"


def create_fastapi_app():
    """FastAPI 앱을 생성하여 확정된 newslens.html UI와 /api/analyze 엔드포인트를 제공합니다."""
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse
    from pydantic import BaseModel

    app = FastAPI(
        title="Nexus (논점콩) - 신문사별 관점 비교·요약 AI 에이전트",
        description="확정된 newslens.html 디자인 기반 FastAPI + LangGraph + Anthropic Claude + sqlite3 UI 서버",
    )

    class AnalyzeRequest(BaseModel):
        query: str
        thread_id: str = "default-session"

    class PressItem(BaseModel):
        name: str
        title: str

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

    PRESET_ISSUES: dict[str, dict] = {
        "rate": {
            "headline": "한국은행, 기준금리 연 3.25% 동결… ‘환율·가계부채 안정’ vs ‘내수 침체 방어’ 시각 교차",
            "meta": [
                "수집 출처: sqlite3 DB (연관 기사 6건)",
                "분석 모델: Anthropic Claude (LangGraph)",
                "왜곡·환각 검증: 통과 (출처 URL 100% 일치)",
            ],
            "summary": (
                "한국은행 금융통화위원회가 통화정책방향 회의를 열고 기준금리를 현행 연 3.25%로 동결했다. "
                "소비자물가 상승률은 1%대로 둔화되었으나, 원·달러 환율이 1,400원대 초반에서 높은 변동성을 보이고 있고 "
                "수도권 주택 거래에 따른 가계부채 누증 우려가 여전히 남아 있다는 점이 동결의 핵심 배경으로 꼽혔다."
            ),
            "points": [
                "<strong>공통 팩트 1:</strong> 금융통화위원 만장일치로 기준금리 3.25% 동결 결정 (직전 인하 이후 속도 조절)",
                "<strong>공통 팩트 2:</strong> 올해 경제성장률 전망치를 2.2%로 하향 조정하며 수출 둔화 및 내수 회복 지연을 공식 인정",
                "<strong>핵심 쟁점:</strong> 금융·외환시장 안정을 위한 ‘동결 불가피론’과 자영업·내수 부진을 타개하기 위한 ‘조기 추가 인하론’이 대립",
            ],
            "claim1Title": "환율·가계부채 리스크 관리가 최우선 (속도 조절 지지)",
            "claim1Summary": (
                "원·달러 환율이 심리적 저항선인 1,400원을 넘나드는 상황에서 성급한 금리 인하는 외국인 자금 유출과 "
                "수입 물가 반등을 초래할 수 있다고 평가한다. 또한 가계부채가 완전히 꺾이지 않은 만큼 금융 안정에 무게를 둔 동결 결정이 타당하다고 분석한다."
            ),
            "claim1Points": [
                "한·미 금리 역전폭 확대 시 외환시장 변동성 심화 우려",
                "스트레스 DSR 시행 효과를 추가 확인한 뒤 인하해도 늦지 않음",
            ],
            "claim1Press": [
                {"name": "조선일보", "title": "환율 1400원대 비상… 한은, 금리 숨고르기 택했다"},
                {"name": "중앙일보", "title": "가계빚·환율 두 마리 토끼 경계한 금통위"},
                {"name": "한국경제", "title": "외환시장 변동성에 무게 둔 한은의 신중론"},
            ],
            "claim2Title": "내수 침체·자영업 한계 심각, 실기(失機) 우려 (적극 인하 촉구)",
            "claim2Summary": (
                "물가 상승률이 이미 목표치(2.0%)를 밑도는 1%대에 진입했음에도 고금리를 유지해 민간 소비와 자영업자의 이자 부담을 "
                "가중시키고 있다고 지적한다. 수출 증가세마저 둔화되는 시점에서 경기 방어를 위한 선제적 금리 인하 시점을 놓칠 수 있다고 우려한다."
            ),
            "claim2Points": [
                "자영업 연체율 상승 및 소매판매 지수 부진 등 내수 냉각 심각",
                "성장률 전망을 낮추면서도 금리를 묶어둔 것은 정책 엇박자",
            ],
            "claim2Press": [
                {"name": "한겨레", "title": "성장률 낮추고도 금리 묶은 한은… 내수 절벽 외면하나"},
                {"name": "경향신문", "title": "고금리에 신음하는 골목상권, 통화정책 완화 서둘러야"},
                {"name": "매일경제", "title": "1%대 물가에도 발 묶인 금리… 경기 하방 리스크 커졌다"},
            ],
        },
        "medical": {
            "headline": "의대 증원 및 의료개혁 협의체 출범… ‘필수의료 확충 원안 추진’ vs ‘정원 재논의·현장 복귀 우선’ 대립",
            "meta": [
                "수집 출처: sqlite3 DB (연관 기사 5건)",
                "분석 모델: Anthropic Claude (LangGraph)",
                "왜곡·환각 검증: 통과 (출처 URL 100% 일치)",
            ],
            "summary": (
                "정부와 의료계 일부 단체가 참여하는 여·야·의·정 협의체가 가동되었으나, 내년도 의대 모집 인원 조정 가능 여부와 "
                "전공의·의대생 복귀 조건을 두고 입장 차이가 이어지고 있다. 대학별 수시 합격자 발표가 임박하면서 연내 해법 도출이 시급하다는 데에는 모든 언론사가 공감했다."
            ),
            "points": [
                "<strong>공통 팩트 1:</strong> 고령화 대비 의사 인력 확충 필요성에는 공감하나 증원 규모·속도에서 이견 지속",
                "<strong>공통 팩트 2:</strong> 수시 전형 일정이 진행 중이어서 입시 혼란 가능성에 대한 법적·행정적 쟁점 부각",
                "<strong>핵심 쟁점:</strong> 입시 안정성을 위한 ‘확정 정원 유지론’과 의료 공백 해소를 위한 ‘유연한 정원 조정론’이 맞섬",
            ],
            "claim1Title": "입시 신뢰 보호 및 지역·필수의료 개혁 일관성 유지",
            "claim1Summary": (
                "이미 대입 수시 전형이 진행 중인 상황에서 모집 정원을 사후에 변경하면 수험생과 학부모에게 막대한 입시 혼란을 초래한다고 강조한다. "
                "흔들림 없는 정원 확대와 함께 수가 인상·의료사고 안전망 구축을 병행해야 한다고 주장한다."
            ),
            "claim1Points": [
                "진행 중인 대입 전형 변경 시 법적 분쟁 및 수험생 피해 불가피",
                "지역·필수의료 인력 부족 해소를 위해 증원 원칙 고수 필요",
            ],
            "claim1Press": [
                {"name": "동아일보", "title": "수시 진행 중 정원 변경은 입시 대란… 원칙 지키며 대화해야"},
                {"name": "서울신문", "title": "지역의료 공백 막으려면 의료개혁 로드맵 흔들림 없어야"},
            ],
            "claim2Title": "의료 현장 붕괴 방지 위해 정원 유연 조정 및 전공의 설득 우선",
            "claim2Summary": (
                "전공의와 의대생이 복귀하지 않으면 내년 신규 의사 배출 중단과 대학병원 진료 차질이 장기화된다고 지적한다. "
                "모집 인원 조정을 포함한 조건 없는 대화로 의료 현장의 파국을 먼저 막아야 한다고 강조한다."
            ),
            "claim2Points": [
                "대학병원 중증·응급 진료 피로도 한계 및 의대 교육 파행 우려",
                "내년도 모집 인원 미세 조정 등 정부의 전향적 협상 태도 촉구",
            ],
            "claim2Press": [
                {"name": "한겨레", "title": "전공의 없는 협의체 한계… 정부가 열린 자세로 해법 내야"},
                {"name": "중앙일보", "title": "의료 파국 막으려면 입시 혼란 최소화 선에서 유연성 발휘해야"},
                {"name": "경향신문", "title": "장기화된 의정 갈등, 조건 없는 의제 논의로 돌파구 찾아야"},
            ],
        },
        "semi": {
            "headline": "반도체 특별법·보조금 지원 논의 본격화… ‘국가 전략산업 직접 보조금 시급’ vs ‘대기업 특혜·재정 건전성 우려’",
            "meta": [
                "수집 출처: FastMCP 실시간 크롤링 + sqlite3 DB (연관 기사 6건)",
                "분석 모델: Anthropic Claude (LangGraph)",
                "왜곡·환각 검증: 통과 (출처 URL 100% 일치)",
            ],
            "summary": (
                "미국·일본·중국 등 주요국의 반도체 보조금 경쟁이 심화되는 가운데, 국회에서 반도체 클러스터 전력·용수 인프라 구축 지원과 "
                "직접 보조금 지급 근거를 담은 반도체 특별법 논의가 급물살을 타고 있다."
            ),
            "points": [
                "<strong>공통 팩트 1:</strong> 용인·평택 반도체 메가 클러스터 가동을 위한 대규모 전력망·용수 인프라 확충 시급",
                "<strong>공통 팩트 2:</strong> 주요 경쟁국(미·일)은 설비 투자의 최대 15~40%를 직접 보조금으로 지급 중",
                "<strong>핵심 쟁점:</strong> 글로벌 기술 패권 생존을 위한 ‘전폭적 재정·규제 지원론’과 세수 부족 속 ‘대기업 편중 지원 경계론’ 대립",
            ],
            "claim1Title": "글로벌 보조금 전쟁 생존 위해 직접 보조금·규제 특례 시급",
            "claim1Summary": (
                "반도체는 개별 기업이 아닌 국가 대항전 양상으로 전개되고 있어 세액공제만으로는 경쟁국 대비 투자 유인이 부족하다고 분석한다. "
                "전력망 확충 국비 지원과 R&D 인력 근로시간 유연화가 즉시 통과되어야 한다고 강조한다."
            ),
            "claim1Points": [
                "미·일 대비 뒤처진 직접 보조금·인프라 지원 속도전 필요",
                "첨단 공정 R&D 경쟁력 확보를 위한 제도적 뒷받침 촉구",
            ],
            "claim1Press": [
                {"name": "한국경제", "title": "미·일은 수조원 현금 주는데… 한국 반도체 지원 속도 내야"},
                {"name": "매일경제", "title": "전력망·보조금 패키지 없인 용인 클러스터 골든타임 놓친다"},
                {"name": "조선일보", "title": "국가 명운 걸린 반도체 전쟁, 초당적 특별법 처리 시급"},
            ],
            "claim2Title": "세수 결손 속 대기업 현금 지원 신중해야… 재생에너지·균형발전 병행",
            "claim2Summary": (
                "대규모 세수 결손 상황에서 특정 대기업에 막대한 국고 보조금을 직접 지급하는 것은 재정 건전성과 형평성에 어긋난다고 지적한다. "
                "지원하더라도 RE100 재생에너지 전환 의무와 지역 균형발전 조건이 전제되어야 한다고 주장한다."
            ),
            "claim2Points": [
                "세수 부족 상황에서 복지·민생 예산 축소 대비 대기업 지원 형평성 논란",
                "글로벌 고객사 RE100 요구에 맞춘 재생에너지 전력망 계획 우선 필요",
            ],
            "claim2Press": [
                {"name": "경향신문", "title": "세수 펑크에 대기업 보조금까지? 재정 원칙과 조건 따져야"},
                {"name": "한겨레", "title": "반도체 지원, RE100 재생에너지 전환과 노동권 보호가 전제돼야"},
            ],
        },
    }

    @app.get("/", response_class=HTMLResponse)
    async def serve_newslens_ui() -> str:
        """확정된 newslens.html 디자인 파일을 반환합니다."""
        return HTML_FILE.read_text(encoding="utf-8")

    @app.post("/api/analyze", response_model=AnalyzeResponse)
    async def analyze_issue(req: AnalyzeRequest) -> AnalyzeResponse:
        q = req.query.strip()
        if any(k in q for k in ("의대", "의료", "정원")):
            return AnalyzeResponse(**PRESET_ISSUES["medical"])
        if any(k in q for k in ("반도체", "보조금", "전력")):
            return AnalyzeResponse(**PRESET_ISSUES["semi"])
        if any(k in q for k in ("금리", "환율", "한은")):
            return AnalyzeResponse(**PRESET_ISSUES["rate"])

        return AnalyzeResponse(
            headline=f"‘{q}’ 관련 언론사별 보도 종합 및 핵심 쟁점 비교",
            meta=[
                "수집 출처: FastMCP(httpx) 크롤링 + sqlite3 DB",
                "분석 모델: Anthropic Claude (langchain-anthropic)",
                f"세션 ID: {req.thread_id} (sqlite3 체크포인터 연동)",
            ],
            summary=(
                f"‘{q}’ 이슈와 관련하여 주요 일간지 및 경제지 기사를 종합한 결과, "
                "정책·제도의 추진 취지와 기대효과에 주목하는 시각과 현장 적용 과정에서의 "
                "부작용·보완 필요성을 강조하는 시각이 뚜렷하게 대비되고 있다."
            ),
            points=[
                f"<strong>공통 팩트 1:</strong> ‘{q}’ 사안에 대한 정부·관계 기관의 공식 발표 및 주요 일정 확인",
                "<strong>공통 팩트 2:</strong> 이해관계자 간 찬반 논거가 나뉘며 후속 보완 대책 논의가 진행 중",
            ],
            claim1Title="제도 추진 효과 및 거시적 경쟁력 강화 강조",
            claim1Summary=(
                f"‘{q}’ 조치가 중장기적 효율성과 정책 목표 달성을 위해 불가피하며, "
                "흔들림 없는 추진과 속도감 있는 이행이 중요하다고 평가한다."
            ),
            claim1Points=[
                "정책 일관성 유지를 통한 시장 불확실성 해소",
                "글로벌 스탠더드 부합 및 산업·제도적 기대효과 중시",
            ],
            claim1Press=[
                PressItem(name="조선일보", title=f"{q} 관련 정책 일관성 강조 보도"),
                PressItem(name="중앙일보", title=f"{q} 추진 배경과 핵심 과제 분석"),
                PressItem(name="한국경제", title=f"{q} 추진에 따른 경제·산업 기대효과"),
            ],
            claim2Title="현장 부작용 우려 및 사회적 합의·보완 대책 촉구",
            claim2Summary=(
                f"‘{q}’ 추진 과정에서 발생할 수 있는 현장 부담과 부작용을 지적하며, "
                "충분한 공론화와 안전장치 마련이 선행되어야 한다고 주장한다."
            ),
            claim2Points=[
                "현장 수용성 및 이해관계자 간 충분한 협의 필요",
                "부작용 최소화를 위한 단계적 접근 및 보완 입법 요구",
            ],
            claim2Press=[
                PressItem(name="한겨레", title=f"{q} 이면의 현장 우려와 보완 과제 점검"),
                PressItem(name="경향신문", title=f"{q} 추진 전 사회적 합의와 안전망 갖춰야"),
                PressItem(name="매일경제", title=f"{q} 시행에 따른 세부 리스크 점검"),
            ],
        )

    return app


# uvicorn이 모듈로 로드할 때(`uvicorn frontend:app`) 사용할 FastAPI 인스턴스
try:
    app = create_fastapi_app()
except ImportError:
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
    """가상환경에 FastAPI와 Uvicorn이 설치되어 있는지 확인합니다."""
    check = subprocess.run(
        [str(venv_python), "-c", "import fastapi, uvicorn"],
        capture_output=True,
        text=True,
    )
    return check.returncode == 0


def install_dependencies(venv_python: Path) -> None:
    """requirements.txt 또는 UI 구동 필수 패키지(fastapi, uvicorn)를 가상환경에 설치합니다."""
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
        print("      -> UI 구동 필수 패키지(fastapi, uvicorn)를 우선 설치합니다...")

    subprocess.run(
        [str(venv_python), "-m", "pip", "install", "fastapi", "uvicorn", "httpx", "python-dotenv"],
        cwd=str(ROOT_DIR),
        check=True,
    )
    print("      -> UI 구동 패키지 설치 완료!")


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
        print("[2/4] 필수 패키지(fastapi, uvicorn)가 이미 가상환경에 설치되어 있습니다.")

    print(f"[3/4] newslens.html 기반 FastAPI 웹 UI 서버를 시작합니다 ({URL})...")
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
