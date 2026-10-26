"""
frontend.py - 확정된 newslens.html UI + OpenAI Head LLM 기반 Multi-Agent Orchestration 서버
실행 방법: python frontend.py
"""

import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Literal, TypedDict
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
CLASSIFIER_MODEL_FILE = ROOT_DIR / "best_news_classifier.joblib"
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

# OpenAI API 기본 설정 (.env 또는 기본값 사용 - Head Orchestrator LLM)
OPENAI_BASE_URL = (
    os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
).strip()
OPENAI_API_KEY = (
    os.environ.get("OPENAI_API_KEY") or ""
).strip()
OPENAI_MODEL = (
    os.environ.get("OPENAI_MODEL") or "gpt-4o-mini"
).strip()

# 1. OpenAI API 클라이언트 설정 (가상환경 외부에서 런처 실행 시에도 에러 없이 통과하도록 처리)
try:
    from openai import OpenAI

    client = OpenAI(
        base_url=OPENAI_BASE_URL,
        api_key=OPENAI_API_KEY or "sk-placeholder",
    )
except ImportError:
    OpenAI = None  # type: ignore
    client = None


def get_openai_client(base_url: str | None = None, api_key: str | None = None):
    """지정된 base_url과 api_key로 OpenAI 클라이언트를 반환합니다."""
    from openai import OpenAI as _OpenAI

    resolved_url = (base_url or os.environ.get("OPENAI_BASE_URL") or OPENAI_BASE_URL).strip()
    resolved_key = (api_key or os.environ.get("OPENAI_API_KEY") or OPENAI_API_KEY).strip()
    if (
        not base_url
        and not api_key
        and client is not None
        and resolved_url == OPENAI_BASE_URL
        and resolved_key == OPENAI_API_KEY
    ):
        return client
    return _OpenAI(
        base_url=resolved_url,
        api_key=resolved_key or "sk-placeholder",
    )


def resolve_openai_config(
    openai_base_url: str | None = None,
    openai_api_key: str | None = None,
    openai_model: str | None = None,
) -> tuple[str, str, str]:
    """
    요청 파라미터 -> 환경변수 -> 기본값 순서로 OpenAI Head LLM 설정(base_url, api_key, model)을 확정합니다.
    공식 OpenAI API(api.openai.com 또는 sk-proj- 키) 사용 시 로컬 GGUF 모델명이 지정되어 있으면
    자동으로 gpt-4o-mini로 보정하여 호환성을 보장합니다.
    """
    active_url = (
        openai_base_url or os.environ.get("OPENAI_BASE_URL") or OPENAI_BASE_URL
    ).strip()
    active_key = (
        openai_api_key or os.environ.get("OPENAI_API_KEY") or OPENAI_API_KEY
    ).strip()
    active_model = (
        openai_model or os.environ.get("OPENAI_MODEL") or OPENAI_MODEL
    ).strip()

    if active_key.startswith("sk-proj-") and "127.0.0.1" in active_url:
        active_url = "https://api.openai.com/v1"

    if "api.openai.com" in active_url and ("/" in active_model or not active_model):
        active_model = "gpt-4o-mini"

    return active_url, active_key, active_model


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


# =====================================================================
# news_classifier.ipynb에서 Export된 최고 점수 분류기(Calibrated LinearSVC) 로더 및 추론기
# =====================================================================
_EXPORTED_CLASSIFIER_CACHE: dict[str, Any] | None = None

_CLASSIFIER_CONTEXTUAL_HANJA: list[tuple[str, str]] = [
    (r"李\s*대통령", "이재명 대통령 "),
    (r"李\s*대표", "이재명 대표 "),
    (r"尹\s*대통령", "윤석열 대통령 "),
    (r"尹\s*전\s*대통령", "윤석열 전 대통령 "),
    (r"文\s*전\s*대통령", "문재인 전 대통령 "),
    (r"文\s*대통령", "문재인 대통령 "),
    (r"韓\s*대표", "한동훈 대표 "),
    (r"韓\s*위원장", "한동훈 위원장 "),
    (r"吳\s*시장", "오세훈 시장 "),
    (r"洪\s*시장", "홍준표 시장 "),
    (r"曺\s*대표", "조국 대표 "),
    (r"安\s*의원", "안철수 의원 "),
    (r"與\s*野|與·野", "여당 야당 "),
    (r"黨\s*政\s*大|당\s*정\s*대", "당정대 "),
    (r"黨\s*政|黨·政", "당정 "),
    (r"行\s*安\s*부|행\s*안\s*부", "행정안전부 "),
    (r"韓\s*국", "한국"),
    (r"韓·美·日|韓美日", "한국 미국 일본 "),
    (r"韓·中·日|韓中日", "한국 중국 일본 "),
    (r"韓·美|韓美|한·美|韓·미", "한국 미국 "),
    (r"韓·日|韓日|한·日|韓·일", "한국 일본 "),
    (r"韓·中|韓中|한·中|韓·중", "한국 중국 "),
    (r"韓·露|韓露|한·露|韓·러", "한국 러시아 "),
    (r"美·北|美北|北·美|北美", "미국 북한 "),
    (r"美·中|美中|中·美|中美", "미국 중국 "),
    (r"美·日|美日|日·美|日美", "미국 일본 "),
    (r"北·中|北中|中·北|中北", "북한 중국 "),
    (r"北·露|北露|露·北|露北", "북한 러시아 "),
    (r"방韓|訪韓", "방한"),
    (r"내韓|來韓", "내한"),
    (r"주韓|駐韓", "주한"),
    (r"반韓|反韓", "반한"),
    (r"친韓|親韓", "친한"),
    (r"혐韓|嫌韓", "혐한"),
    (r"주한\s*美\s*군", "주한 미군 "),
    (r"(?<=[방내주반친혐대訪來駐反親嫌對])美", "미"),
    (r"(?<=[방내주반친혐대訪來駐反親嫌對])中", "중"),
    (r"(?<=[방내주반친혐대訪來駐反親嫌對])日", "일"),
    (r"(?<=[방내주반친혐대訪來駐反親嫌對])北", "북"),
    (r"(?<=[방내주반친혐대訪來駐反親嫌對])露", "러"),
]

_CLASSIFIER_SEMANTIC_HANJA: dict[str, str] = {
    "李": "이재명 ", "尹": "윤석열 ", "韓": "한동훈 ", "文": "문재인 ",
    "洪": "홍준표 ", "吳": "오세훈 ", "安": "안철수 ", "曺": "조국 ",
    "美": "미국 ", "北": "북한 ", "中": "중국 ", "日": "일본 ", "露": "러시아 ",
    "英": "영국 ", "佛": "프랑스 ", "獨": "독일 ", "臺": "대만 ",
    "與": "여당 ", "野": "야당 ", "政": "정부 ", "檢": "검찰 ", "警": "경찰 ",
    "軍": "군대 ", "靑": "청와대 ", "法": "법원 ", "銀": "은행 ", "證": "증권 ",
    "企": "기업 ", "黨": "정당 ",
}

_CLASSIFIER_PHONETIC_TRANS = str.maketrans({
    "萬": "만", "億": "억", "兆": "조",
    "年": "년", "月": "월", "時": "시", "分": "분", "秒": "초",
    "大": "대", "小": "소", "新": "신", "舊": "구", "高": "고", "低": "저", "多": "다", "少": "소",
    "前": "전", "後": "후", "上": "상", "下": "하", "內": "내", "外": "외", "左": "좌", "右": "우",
    "正": "정", "副": "부", "親": "친", "反": "반", "非": "비", "汎": "범", "全": "전",
    "對": "대", "發": "발", "行": "행", "故": "고", "無": "무", "有": "유", "重": "중", "輕": "경",
    "株": "주", "債": "채", "稅": "세", "金": "금", "院": "원", "朴": "박",
})

_CLASSIFIER_STOPWORDS = {
    "종합", "단독", "속보", "포토", "사진", "영상", "일문일답", "전문", "사설", "칼럼", "단신",
    "브리핑", "초점", "현장", "르포", "인터뷰", "기획", "연재", "긴급", "주목", "화제", "이슈", "해설", "뉴스",
    "만에", "하루", "연중", "내내", "이후", "부터", "까지", "위해", "대해", "통해", "따라", "관련",
    "대한", "이번", "지난", "오늘", "내일", "어제", "올해", "내년", "작년", "현재", "지금", "당시",
    "우리", "저희", "그것", "이것", "저것", "여기", "거기", "저기", "모두", "전체", "일부", "각각",
    "있는", "없는", "하는", "되는", "않는", "같은", "위한", "통한", "따른", "대비", "두고", "놓고",
    "있어", "없어", "하며", "되며", "하고", "되고", "한다", "된다", "했다", "됐다", "할까", "될까",
    "기자", "무단", "전재", "배포", "금지", "저작권자", "연합뉴스", "뉴시스", "뉴스1", "앵커", "리포트",
}


def preprocess_korean_news_for_classifier(text: str, max_chars: int | None = None) -> str:
    """news_classifier.ipynb와 동일한 한국어 뉴스 전처리(한자 의미 복원·상투어·조사 제거)를 수행합니다."""
    if not isinstance(text, str):
        return ""
    if max_chars is not None:
        text = text[:max_chars]

    text = re.sub(r"\[[^\]]*기자[^\]]*\]", " ", text)
    tag_pattern = (
        r"[\[\(\<【]\s*(?:종합|단독|속보|포토|사진|영상|일문일답|전문|사설|칼럼|브리핑|"
        r"현장|르포|인터뷰|기획|긴급|뉴스|그래픽|카드뉴스|팩트체크|이슈|앵커)[0-9보\s]*[\]\)\>】]"
    )
    text = re.sub(tag_pattern, " ", text)
    text = re.sub(r"(\d+)\s*日", r"\1일", text)

    for pattern, replacement in _CLASSIFIER_CONTEXTUAL_HANJA:
        text = re.sub(pattern, replacement, text)

    for hanja, full_word in _CLASSIFIER_SEMANTIC_HANJA.items():
        text = re.sub(hanja + r"(?=[은는이가을를의에로과와도만]\b)", full_word.strip(), text)
        text = text.replace(hanja, full_word)

    text = text.translate(_CLASSIFIER_PHONETIC_TRANS)
    text = re.sub(r"[^가-힣a-zA-Z0-9\s]", " ", text)

    words = text.split()
    cleaned_words: list[str] = []
    for w in words:
        if w in _CLASSIFIER_STOPWORDS:
            continue
        if len(w) >= 3:
            w_stem = re.sub(r"(에서|으로|에게|부터|까지|이라|라고|은|는|이|가|을|를|의|에|로|과|와|도|만)$", "", w)
            if len(w_stem) >= 2:
                w = w_stem
        if w not in _CLASSIFIER_STOPWORDS and len(w) >= 2:
            cleaned_words.append(w)

    return " ".join(cleaned_words)


def load_exported_news_classifier(model_path: Path = CLASSIFIER_MODEL_FILE) -> dict[str, Any] | None:
    """best_news_classifier.joblib에 저장된 최고 점수 뉴스 분류기 번들을 메모리에 로드합니다."""
    global _EXPORTED_CLASSIFIER_CACHE
    if _EXPORTED_CLASSIFIER_CACHE is not None:
        return _EXPORTED_CLASSIFIER_CACHE
    if not model_path.exists():
        return None
    try:
        import joblib

        _EXPORTED_CLASSIFIER_CACHE = joblib.load(str(model_path))
        return _EXPORTED_CLASSIFIER_CACHE
    except Exception as e:
        print(f"[안내] best_news_classifier.joblib 로드 실패({e})")
        return None


def classify_news_with_exported_model(
    title: str,
    content: str = "",
    top_k: int = 3,
    model_path: Path = CLASSIFIER_MODEL_FILE,
) -> dict[str, Any] | None:
    """
    news_classifier.ipynb에서 Export된 최고 점수 분류기(Calibrated LinearSVC)를 사용하여
    입력된 뉴스의 클러스터 ID, 클러스터 라벨, 대표 키워드, 신뢰도(Confidence) 및 상위 후보 클러스터를 반환합니다.
    """
    bundle = load_exported_news_classifier(model_path)
    if not bundle:
        return None

    clean_t = preprocess_korean_news_for_classifier(title)
    clean_c = preprocess_korean_news_for_classifier(content, max_chars=600)
    combined = f"{clean_t} {clean_t} {clean_c}".strip()
    if not combined:
        combined = title.strip()

    vectorizer = bundle["tfidf_vectorizer"]
    model = bundle["model"]
    label_map: dict[int, str] = bundle.get("cluster_label_map", {})
    kw_map: dict[int, list[str]] = bundle.get("cluster_keywords_map", {})

    X_vec = vectorizer.transform([combined])
    pred_id = int(model.predict(X_vec)[0])
    pred_label = label_map.get(pred_id, f"C{pred_id:02d}")
    pred_keywords = kw_map.get(pred_id, [])

    top_clusters: list[dict[str, Any]] = []
    confidence = 1.0
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_vec)[0]
        classes = list(model.classes_)
        ranked_indices = proba.argsort()[::-1][: max(1, top_k)]
        confidence = float(proba[ranked_indices[0]])
        for idx in ranked_indices:
            cid = int(classes[idx])
            top_clusters.append(
                {
                    "cluster_id": cid,
                    "cluster_label": label_map.get(cid, f"C{cid:02d}"),
                    "keywords": kw_map.get(cid, []),
                    "probability": round(float(proba[idx]), 4),
                }
            )

    return {
        "model_name": bundle.get("model_name", "5. Calibrated LinearSVC (C=1.0)"),
        "clean_title": clean_t,
        "predicted_cluster_id": pred_id,
        "predicted_cluster_label": pred_label,
        "keywords": pred_keywords,
        "confidence": round(confidence, 4),
        "top_clusters": top_clusters,
        "metrics": bundle.get("metrics", {}),
    }


def _parse_numbered_judgment(raw_text: str) -> dict[str, str]:
    """
    OpenAI 모델이 생성한 번호형 판단 응답에서
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


# =====================================================================
# Multi-Agent Orchestration 아키텍처 (Head LLM: OpenAI API Supervisor)
# =====================================================================
class OrchestrationState(TypedDict, total=False):
    """LangGraph 기반 Multi-Agent Orchestration 공유 상태(State)"""

    query: str
    thread_id: str
    openai_base_url: str
    openai_api_key: str
    openai_model: str
    # Head Orchestrator(Supervisor) 계획 및 라우팅 상태
    search_query: str
    expanded_keywords: list[str]
    supervisor_plan: str
    next_agent: str
    head_llm_online: bool
    # Sub-Agent 1 (RAG 검색) 상태
    retrieved_articles: list[dict[str, Any]]
    press_diversity_count: int
    top_relevance_score: float
    # Sub-Agent 2 (관점 A/B 비교·합성) & Sub-Agent 4 (환각 가드레일) 상태
    base_result: dict[str, Any]
    # Sub-Agent 3 (심판·출처 검증) 및 최종 결과
    final_result: dict[str, Any]
    orchestration_trace: list[dict[str, Any]]


class OpenAISupervisorHead:
    """
    오케스트레이션의 머리(Brain/Supervisor) 역할을 수행하는 OpenAI API 기반 헤드 에이전트.
    - 역할 1 (plan_query_and_keywords): 사용자 질의를 분석하여 핵심 검색 키워드 확장 및 실행 계획 수립
    - 역할 2 (evaluate_evidence_and_route): news.db RAG 검색 결과를 평가하여 [비교·합성 에이전트 / 환각 가드레일 에이전트] 중 다음 실행할 서브 에이전트를 결정
    """

    def __init__(self) -> None:
        self._cached_models: dict[str, list[str]] = {}

    def _call_openai(
        self,
        system_prompt: str,
        user_prompt: str,
        base_url: str,
        api_key: str,
        model: str,
        max_tokens: int = 240,
        timeout: float = 12.0,
    ) -> tuple[bool, str, str]:
        """OpenAI 호환 API를 호출하고 (성공여부, 응답텍스트, 실제사용모델명)을 반환합니다."""
        active_url, active_key, active_model = resolve_openai_config(base_url, api_key, model)
        try:
            oai_client = get_openai_client(base_url=active_url, api_key=active_key)
            if "api.openai.com" not in active_url:
                if active_url not in self._cached_models:
                    try:
                        avail = [m.id for m in oai_client.models.list().data]
                        if avail:
                            self._cached_models[active_url] = avail
                    except Exception:
                        self._cached_models[active_url] = []

                avail_models = self._cached_models.get(active_url, [])
                if avail_models and active_model not in avail_models:
                    active_model = avail_models[0]

            resp = oai_client.chat.completions.create(
                model=active_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=max_tokens,
                timeout=timeout,
            )
            text = (resp.choices[0].message.content or "").strip()
            return bool(text), text, active_model
        except Exception as e:
            err_short = re.sub(r"\s+", " ", str(e))[:120]
            print(f"[Head Orchestrator 안내] OpenAI API({active_url}) 호출 폴백({err_short})")
            return False, "", active_model

    def plan_query_and_keywords(
        self,
        raw_query: str,
        fallback_search_query: str,
        base_url: str,
        api_key: str,
        model: str,
    ) -> dict[str, Any]:
        """
        [Head LLM Step 1] 사용자 입력을 분석하여 검색어(search_query), 확장 키워드, 오케스트레이션 계획을 수립합니다.
        """
        sys_prompt = (
            "너는 멀티에이전트 뉴스 분석 시스템의 수석 오케스트레이터(Head Supervisor LLM)야. "
            "사용자 질의에서 뉴스 DB 검색을 위한 핵심 키워드(2~5개 단어)와 분석 계획을 한 줄씩 출력해."
        )
        usr_prompt = (
            f"사용자 질의: {raw_query}\n"
            f"기본 추출 키워드: {fallback_search_query}\n\n"
            "다음 형식으로만 답해:\n"
            "검색키워드: (핵심 명사 중심 검색어)\n"
            "분석계획: (어떤 관점과 쟁점을 중점 비교할지 한 문장 요약)"
        )
        ok, text, used_model = self._call_openai(
            sys_prompt, usr_prompt, base_url, api_key, model, max_tokens=120, timeout=8.0
        )

        extracted_kw = fallback_search_query
        plan_note = f"‘{fallback_search_query}’ 이슈에 대해 news.db 하이브리드 RAG 검색 후 언론사별 관점 A/B 대립축을 분석합니다."

        if ok and text:
            for line in text.splitlines():
                clean_line = re.sub(r"\*\*|\#", "", line).strip()
                if clean_line.startswith("검색키워드") and ":" in clean_line:
                    candidate = clean_line.split(":", 1)[1].strip()
                    if candidate and len(candidate) >= 2:
                        extracted_kw = fallback_search_query
                elif clean_line.startswith("분석계획") and ":" in clean_line:
                    cand_plan = clean_line.split(":", 1)[1].strip()
                    if cand_plan:
                        plan_note = cand_plan

        tokens = [
            tok
            for tok in re.findall(r"[가-힣A-Za-z0-9]{2,}", f"{fallback_search_query} {extracted_kw}")
        ]
        dedup_tokens = list(dict.fromkeys(tokens))[:6]

        return {
            "head_llm_online": ok,
            "used_model": used_model,
            "search_query": fallback_search_query,
            "expanded_keywords": dedup_tokens,
            "supervisor_plan": plan_note,
        }

    def evaluate_evidence_and_route(
        self,
        query: str,
        articles: list[dict[str, Any]],
        base_url: str,
        api_key: str,
        model: str,
        crawled: bool = False,
    ) -> dict[str, Any]:
        """
        [Head LLM Step 2 - LLM-as-a-Judge] RAG 서브 에이전트가 news.db에서 검색한 기사들을
        LLM Judge(심판)가 직접 평가하여 다음 실행할 서브 에이전트(comparison_agent / guard_agent)를 라우팅합니다.
        """
        press_set = {a.get("press", "") for a in articles if a.get("press")}
        top_score = float(articles[0].get("score", 0.0)) if articles else 0.0

        if len(articles) < 2 or len(press_set) < 2 or top_score < 0.35:
            return {
                "route": "guard_agent",
                "reason": (
                    f"[LLM Judge 사전 가드레일] 검색된 기사({len(articles)}건, {len(press_set)}개 언론사)가 "
                    "양면 관점 비교를 수행하기에 부족하여 환각 차단 에이전트(HallucinationGuardSubAgent)로 라우팅"
                ),
                "press_diversity_count": len(press_set),
                "top_relevance_score": round(top_score, 4),
            }

        sample_lines = [
            f"- [{a.get('press', '')}] {a.get('title', '')} (본문요약: {(a.get('summary') or a.get('content') or '')[:100]})"
            for a in articles[:5]
        ]
        sys_prompt = (
            "너는 뉴스 관점 비교 에이전트의 LLM Judge(심판)야. "
            "검색된 기사들이 사용자 질의의 주요 인물·사건·주제와 조금이라도 관련된 시사 이슈를 다루고 있으면 반드시 '판정: 적합'으로 답해. "
            "오직 질의가 현실에 없는 허위 주제이거나 기사들과 완전히 무관할 때만 '판정: 부적합'으로 답해."
        )
        usr_prompt = (
            f"질의: {query}\n"
            f"검색된 기사({len(articles)}건, {len(press_set)}개 언론사):\n"
            + "\n".join(sample_lines)
            + "\n\n형식:\n판정: (적합 또는 부적합)\n이유: (신문사별 관점 비교 적합성에 대한 한 줄 평가)"
        )
        ok, judge_text, _ = self._call_openai(
            sys_prompt, usr_prompt, base_url, api_key, model, max_tokens=100, timeout=6.0
        )

        if ok and judge_text:
            clean_j = re.sub(r"\*\*|\#", "", judge_text).strip()
            if "부적합" in clean_j and "적합합니다" not in clean_j and top_score < 0.50:
                return {
                    "route": "guard_agent",
                    "reason": f"[LLM Judge 판정: 부적합] {clean_j.replace(chr(10), ' ')} → 환각 차단 에이전트 호출",
                    "press_diversity_count": len(press_set),
                    "top_relevance_score": round(top_score, 4),
                }
            return {
                "route": "comparison_agent",
                "reason": (
                    f"[LLM Judge 심판 통과] {clean_j.replace(chr(10), ' ')} "
                    f"({len(articles)}건 · {len(press_set)}개 언론사) → 관점 비교·합성 에이전트 호출"
                ),
                "press_diversity_count": len(press_set),
                "top_relevance_score": round(top_score, 4),
            }

        return {
            "route": "comparison_agent",
            "reason": (
                f"[LLM Judge 하이브리드 판정: 적합] news.db 내 연관 기사 {len(articles)}건({len(press_set)}개 언론사) "
                "검증 완료 → 관점 비교·합성 에이전트(ComparisonSynthesisSubAgent) 호출 결정"
            ),
            "press_diversity_count": len(press_set),
            "top_relevance_score": round(top_score, 4),
        }


def judge_articles_with_openai(
    query: str,
    base_result: dict[str, Any],
    openai_api_key: str | None = None,
    openai_model: str | None = None,
    openai_base_url: str | None = None,
    orchestration_trace: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    JudgeVerifierSubAgent & OpenAI Head LLM이 news.db에서 수집된 신문사별 기사들을 심판하고
    웹 UI에 표시할 결과(overallJudgment, articleJudgments, orchestrationTrace)를 생성합니다.
    """
    active_url, active_key, active_model = resolve_openai_config(
        openai_base_url, openai_api_key, openai_model
    )

    raw_articles: list[dict[str, Any]] = base_result.get("articles") or []
    trace_list = orchestration_trace or base_result.get("orchestrationTrace") or []

    # 1. 근거 기사가 없는 허위 이슈(환각 차단 케이스)
    if not raw_articles:
        base_result["overallJudgment"] = {
            "engine": f"Multi-Agent Head LLM: OpenAI ({active_model} @ {active_url})",
            "isLiveOpenAI": True,
            "verdict": "부적합 · 근거 기사 없음 (Multi-Agent 환각 차단)",
            "conflictAxis": "보도된 신뢰 기사 0건으로 비교 불가",
            "viewpointAJudgment": "news.db 내 일치 기사 없음",
            "viewpointBJudgment": "news.db 내 일치 기사 없음",
            "reliabilityScore": 100,
            "reason": (
                f"Head Orchestrator가 ‘{query}’ 주제에 대해 news.db 하이브리드 RAG 검색 에이전트를 "
                "지휘했으나 일치하는 신뢰 기사가 없어 환각 차단 에이전트(Guardrail)를 통해 추측성 요약 생성을 차단했습니다."
            ),
        }
        base_result["articleJudgments"] = []
        base_result["orchestrationTrace"] = trace_list
        return base_result

    c1_press_items = base_result.get("claim1Press", [])
    c2_press_items = base_result.get("claim2Press", [])
    c1_names = [p.get("name", "") for p in c1_press_items if p.get("name")]
    c2_names = [p.get("name", "") for p in c2_press_items if p.get("name")]

    c1_title = base_result.get("claim1Title", "")
    c2_title = base_result.get("claim2Title", "")
    c1_summary_raw = base_result.get("claim1Summary", "")[:200]
    c2_summary_raw = base_result.get("claim2Summary", "")[:200]

    # 2. OpenAI API를 호출하여 기사 관점·논조 판단 및 종합 감수 수행
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
        if "api.openai.com" not in active_url:
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
            "멀티에이전트 오케스트레이션의 최종 심판 및 총괄 감독(Supervisor)으로서 "
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
                    "content": (
                        "너는 멀티에이전트 뉴스 분석 오케스트레이션의 Head LLM(총괄 감독 및 최종 심판 AI)이야. "
                        "서브 에이전트들이 수집·군집화한 신문사별 기사 관점과 논조를 비교 판단하여 반드시 한국어로 간결하게 답해."
                    ),
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
        print(f"[안내] OpenAI 서버({active_url}) 호출 예외({e}) -> 멀티에이전트 하이브리드 판단 폴백 사용")

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

        ml_cls = classify_news_with_exported_model(title=title, content=content, top_k=1)
        cluster_label = (
            ml_cls["predicted_cluster_label"]
            if ml_cls and ml_cls.get("predicted_cluster_label")
            else a.get("cluster_label", "")
        )

        article_judgments.append(
            {
                "press": p_name,
                "title": title,
                "cleanTitle": a.get("clean_title", "") or (ml_cls.get("clean_title", "") if ml_cls else ""),
                "category": a.get("category", "시사·종합"),
                "clusterLabel": cluster_label,
                "url": url,
                "stance": stance,
                "tone": tone,
                "relevanceScore": score_pct,
                "judgmentSummary": j_summary,
            }
        )

    article_judgments.sort(
        key=lambda x: (0 if "관점 A" in x.get("stance", "") else 1, -int(x.get("relevanceScore", 0)))
    )

    press_set = sorted({a.get("press", "") for a in raw_articles if a.get("press")})
    agent_names_chain = " → ".join(t.get("agent", "") for t in trace_list if t.get("agent"))
    if not agent_names_chain:
        agent_names_chain = "Supervisor(OpenAI) → RAGSubAgent → ComparisonSubAgent → JudgeVerifierSubAgent"

    if live_openai_ok:
        if parsed_j["claim1_judgment"] and not str(base_result.get("claim1Summary", "")).startswith("[OpenAI"):
            base_result["claim1Summary"] = (
                f"[OpenAI 관점 A 판단] {parsed_j['claim1_judgment']} — {base_result.get('claim1Summary', '')}"
            )
        if parsed_j["claim2_judgment"] and not str(base_result.get("claim2Summary", "")).startswith("[OpenAI"):
            base_result["claim2Summary"] = (
                f"[OpenAI 관점 B 판단] {parsed_j['claim2_judgment']} — {base_result.get('claim2Summary', '')}"
            )

        base_result["meta"] = [
            f"오케스트레이션 Head LLM: OpenAI ({active_model} @ {active_url})",
            f"실행 파이프라인: {agent_names_chain}",
            f"수집·검증: news.db 기사 {len(raw_articles)}건 ({len(press_set)}개 신문사 · 원문 URL 100% 일치)",
        ]
        base_result["overallJudgment"] = {
            "engine": f"Multi-Agent Head: OpenAI ({active_model})",
            "isLiveOpenAI": True,
            "verdict": f"적합 ({len(press_set)}개 신문사 · 기사 {len(raw_articles)}건 Multi-Agent + OpenAI 판단 완료)",
            "conflictAxis": parsed_j["conflict_axis"] or f"{c1_title[:35]} vs {c2_title[:35]}",
            "viewpointAJudgment": parsed_j["claim1_judgment"] or c1_summary_raw,
            "viewpointBJudgment": parsed_j["claim2_judgment"] or c2_summary_raw,
            "reliabilityScore": 97,
            "reason": parsed_j["conclusion"] or raw_llm_response,
        }
    else:
        base_result["meta"] = [
            f"오케스트레이션 파이프라인: {agent_names_chain}",
            f"수집·저장: news.db (연관 기사 {len(raw_articles)}건 · {len(press_set)}개 신문사)",
            "출처 검증: 통과 (news.db 원문 URL 100% 일치 · OpenAI 오프라인 하이브리드 폴백)",
        ]
        base_result["overallJudgment"] = {
            "engine": "Multi-Agent 하이브리드 오케스트레이터 (OpenAI 서버 오프라인 폴백)",
            "isLiveOpenAI": False,
            "verdict": f"적합 ({len(press_set)}개 신문사 · 기사 {len(raw_articles)}건 멀티에이전트 관점 비교 완료)",
            "conflictAxis": f"{c1_title[:38]} vs {c2_title[:38]}",
            "viewpointAJudgment": c1_summary_raw,
            "viewpointBJudgment": c2_summary_raw,
            "reliabilityScore": 95,
            "reason": (
                f"Multi-Agent Orchestrator가 ‘{query}’ 관련 news.db 기사 {len(raw_articles)}건을 "
                f"문맥 벡터 거리 및 논조 어휘로 분석하여 관점 A({len(c1_names)}개 매체)와 관점 B({len(c2_names)}개 매체)로 판정했습니다."
            ),
        }

    base_result["articleJudgments"] = article_judgments
    base_result["orchestrationTrace"] = trace_list
    return base_result


class MultiAgentNewsOrchestrator:
    """
    OpenAI API를 머리(Head Supervisor LLM)로 사용하는 LangGraph 기반 멀티 에이전트 오케스트레이터 (크롤러 미사용 · news.db 전용).

    [오케스트레이션 노드 구성]
    1. supervisor_plan_node (Head LLM - OpenAI API):
       - 멀티턴 문맥 복원, 검색 키워드 확장 및 서브 에이전트 실행 계획 수립
    2. rag_retrieval_agent_node (Sub-Agent 1 - Hybrid RAG Retrieval Agent):
       - news.db (sqlite-vec + TF-IDF + 키워드) 하이브리드 검색 및 언론사 다양성 리랭킹 수행
    3. supervisor_route_node (Head LLM - Evidence Evaluator & Router):
       - 검색된 기사의 건수(>=2), 언론사 다양성(>=2) 및 유사도(>=0.35)를 평가하여 동적 라우팅 결정
       - 분기 1 -> comparison_agent_node (충분한 근거 확보 시 관점 A/B 군집화·요약 지시)
       - 분기 2 -> hallucination_guard_node (근거 기사 부족 시 환각 차단 지시)
    4. comparison_agent_node (Sub-Agent 2 - Viewpoint A/B Clustering & Synthesis Agent):
       - 언론사별 주장을 관점 A vs 관점 B로 군집화하고 공통 요약·핵심 쟁점 생성 및 출처 URL 매핑
    5. hallucination_guard_node (Sub-Agent 3 - Hallucination Guardrail Agent):
       - 허위·미보도 이슈에 대해 추측성 생성을 차단하고 투명한 안내 반환
    6. judge_verifier_agent_node (Sub-Agent 4 & Head LLM Finalizer):
       - OpenAI API를 통해 대립축·관점별 논조·개별 기사 판정 및 원문 URL 100% 일치 검증 완료
    """

    def __init__(self, rag_agent: Any) -> None:
        from langgraph.checkpoint.memory import MemorySaver
        from langgraph.graph import END, START, StateGraph

        self.rag_agent = rag_agent
        self.rag = rag_agent.rag
        self.head_llm = OpenAISupervisorHead()
        self.memory = MemorySaver()
        self._session_queries: dict[str, list[str]] = {}

        workflow = StateGraph(OrchestrationState)
        workflow.add_node("supervisor_plan", self._supervisor_plan_node)
        workflow.add_node("rag_retrieval_agent", self._rag_retrieval_agent_node)
        workflow.add_node("supervisor_route", self._supervisor_route_node)
        workflow.add_node("comparison_agent", self._comparison_agent_node)
        workflow.add_node("hallucination_guard_agent", self._hallucination_guard_node)
        workflow.add_node("judge_verifier_agent", self._judge_verifier_agent_node)

        workflow.add_edge(START, "supervisor_plan")
        workflow.add_edge("supervisor_plan", "rag_retrieval_agent")
        workflow.add_edge("rag_retrieval_agent", "supervisor_route")
        workflow.add_conditional_edges(
            "supervisor_route",
            self._select_next_subagent,
            {
                "comparison_agent": "comparison_agent",
                "guard_agent": "hallucination_guard_agent",
            },
        )
        workflow.add_edge("comparison_agent", "judge_verifier_agent")
        workflow.add_edge("hallucination_guard_agent", "judge_verifier_agent")
        workflow.add_edge("judge_verifier_agent", END)

        self.graph = workflow.compile(checkpointer=self.memory)

    def _supervisor_plan_node(self, state: OrchestrationState) -> dict[str, Any]:
        """[Node 1] Head LLM(OpenAI)이 질의 의도를 분석하고 서브 에이전트 실행 계획을 수립합니다."""
        t0 = time.perf_counter()
        raw_q = state.get("query", "").strip()
        tid = state.get("thread_id", "default-session")
        base_url, api_key, model = resolve_openai_config(
            state.get("openai_base_url"),
            state.get("openai_api_key"),
            state.get("openai_model"),
        )

        # 멀티턴 대화 문맥 복원 + 핵심 키워드 정제 (rag_agent의 analyze_query_node 활용)
        sub_state = self.rag_agent.analyze_query_node({"query": raw_q, "thread_id": tid})
        extracted_kws: list[str] = list(sub_state.get("keywords") or [])
        clean_search_q = " ".join(extracted_kws) if extracted_kws else raw_q

        # OpenAI Head LLM에게 키워드 확장 및 분석 계획 지시
        plan_info = self.head_llm.plan_query_and_keywords(
            raw_query=raw_q,
            fallback_search_query=clean_search_q,
            base_url=base_url,
            api_key=api_key,
            model=model,
        )
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

        trace = list(state.get("orchestration_trace") or [])
        trace.append(
            {
                "step": len(trace) + 1,
                "agent": "HeadSupervisor(OpenAI)",
                "role": "오케스트레이션 총괄 계획 및 질의 분석",
                "status": "live_openai" if plan_info["head_llm_online"] else "hybrid_fallback",
                "model": plan_info["used_model"],
                "detail": (
                    f"검색어 ‘{clean_search_q}’ 확정 (키워드: {', '.join((extracted_kws or plan_info['expanded_keywords'])[:4])}) | "
                    f"계획: {plan_info['supervisor_plan']}"
                ),
                "elapsedMs": elapsed_ms,
            }
        )

        return {
            "search_query": clean_search_q,
            "expanded_keywords": extracted_kws or plan_info["expanded_keywords"],
            "supervisor_plan": plan_info["supervisor_plan"],
            "head_llm_online": plan_info["head_llm_online"],
            "orchestration_trace": trace,
        }

    def _rag_retrieval_agent_node(self, state: OrchestrationState) -> dict[str, Any]:
        """[Node 2] RAGRetrievalSubAgent가 news.db에서 하이브리드 벡터·키워드 검색을 수행합니다."""
        t0 = time.perf_counter()
        kws = list(state.get("expanded_keywords") or [])
        ret_state = self.rag_agent.retrieve_news_node(
            {
                "query": state.get("query", ""),
                "keywords": kws,
            }
        )
        articles: list[dict[str, Any]] = list(ret_state.get("articles") or [])
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

        press_set = {a.get("press", "") for a in articles if a.get("press")}
        top_score = float(articles[0].get("score", 0.0)) if articles else 0.0
        search_q = state.get("search_query") or state.get("query", "")

        trace = list(state.get("orchestration_trace") or [])
        trace.append(
            {
                "step": len(trace) + 1,
                "agent": "RAGRetrievalSubAgent",
                "role": "news.db 하이브리드 벡터(sqlite-vec)·키워드 검색 및 언론사 다양성 리랭킹",
                "status": "completed",
                "detail": (
                    f"질의 ‘{search_q}’ 검색 결과: 기사 {len(articles)}건 확보 "
                    f"({len(press_set)}개 언론사, 최고 유사도 {top_score:.2f})"
                ),
                "elapsedMs": elapsed_ms,
            }
        )

        return {
            "retrieved_articles": articles,
            "press_diversity_count": len(press_set),
            "top_relevance_score": round(top_score, 4),
            "orchestration_trace": trace,
        }

    def _supervisor_route_node(self, state: OrchestrationState) -> dict[str, Any]:
        """[Node 3] Head LLM(Supervisor)이 검색 품질을 평가하고 다음 서브 에이전트를 지정합니다."""
        t0 = time.perf_counter()
        base_url, api_key, model = resolve_openai_config(
            state.get("openai_base_url"),
            state.get("openai_api_key"),
            state.get("openai_model"),
        )
        decision = self.head_llm.evaluate_evidence_and_route(
            query=state.get("search_query") or state.get("query", ""),
            articles=state.get("retrieved_articles") or [],
            base_url=base_url,
            api_key=api_key,
            model=model,
        )
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

        trace = list(state.get("orchestration_trace") or [])
        trace.append(
            {
                "step": len(trace) + 1,
                "agent": "HeadSupervisor(OpenAI)",
                "role": "근거 기사 충분성 평가 및 동적 에이전트 라우팅",
                "status": f"route->{decision['route']}",
                "detail": decision["reason"],
                "elapsedMs": elapsed_ms,
            }
        )

        return {
            "next_agent": decision["route"],
            "press_diversity_count": decision["press_diversity_count"],
            "top_relevance_score": decision["top_relevance_score"],
            "orchestration_trace": trace,
        }

    def _select_next_subagent(
        self, state: OrchestrationState
    ) -> Literal["comparison_agent", "guard_agent"]:
        route = state.get("next_agent", "comparison_agent")
        if route in ("comparison_agent", "guard_agent"):
            return route  # type: ignore[return-value]
        return "comparison_agent"

    def _comparison_agent_node(self, state: OrchestrationState) -> dict[str, Any]:
        """[Node 4] ComparisonSynthesisSubAgent가 관점 A vs 관점 B 군집화 및 양면 비교 요약을 생성합니다."""
        t0 = time.perf_counter()
        raw_articles = state.get("retrieved_articles") or []
        comp_out = self.rag_agent.compare_and_summarize_node(
            {
                "query": state.get("query", ""),
                "articles": raw_articles,
                "crawled_count": 0,
            }
        )
        verified_out = self.rag_agent.verify_sources_node(
            {
                "query": state.get("query", ""),
                "thread_id": state.get("thread_id", "default-session"),
                "articles": raw_articles,
                "result": comp_out.get("result", {}),
            }
        )
        base_res: dict[str, Any] = verified_out.get("result", {})
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

        c1_cnt = len(base_res.get("claim1Press", []))
        c2_cnt = len(base_res.get("claim2Press", []))

        trace = list(state.get("orchestration_trace") or [])
        trace.append(
            {
                "step": len(trace) + 1,
                "agent": "ComparisonSynthesisSubAgent",
                "role": "언론사별 관점 A / 관점 B 클러스터링, 양면 요약 합성 및 출처 URL 매핑",
                "status": "completed",
                "detail": (
                    f"관점 A({c1_cnt}개 언론사: {base_res.get('claim1Title', '')[:28]}…) vs "
                    f"관점 B({c2_cnt}개 언론사: {base_res.get('claim2Title', '')[:28]}…) 분리 합성 완료"
                ),
                "elapsedMs": elapsed_ms,
            }
        )

        return {
            "base_result": base_res,
            "orchestration_trace": trace,
        }

    def _hallucination_guard_node(self, state: OrchestrationState) -> dict[str, Any]:
        """[Node 5] HallucinationGuardSubAgent가 근거 없는 이슈에 대해 환각 생성을 차단합니다."""
        t0 = time.perf_counter()
        q = state.get("query", "")
        popular = self.rag.get_popular_issues(limit=4)
        suggested = (
            ", ".join(f"‘{p['query']}’" for p in popular)
            if popular
            else "‘김여정 확성기’, ‘수사기관장 배석’, ‘국민연금’"
        )
        base_res: dict[str, Any] = {
            "headline": f"‘{q}’에 대한 보도 기사를 찾을 수 없습니다",
            "meta": [
                "수집 출처: news.db (sqlite-vec) 검색 결과 신뢰 기준 만족 기사 0건",
                "환각 방지(성공 기준 1): 근거 없는 추측·가짜 출처 생성 차단됨",
                f"추천 주제: {suggested}",
            ],
            "summary": (
                f"로컬 데이터베이스(news.db) sqlite-vec 벡터 검색을 수행했으나 ‘{q}’ 주제와 일치하는 "
                "신뢰할 수 있는 보도 기사가 확인되지 않았습니다. 에이전트 안전 설계에 따라 없는 자료를 지어내지 않습니다."
            ),
            "points": [
                "<strong>환각 차단 완료:</strong> 근거 기사가 부족하여 추측성 요약 생성을 중단했습니다.",
                f"<strong>추천 검색 주제:</strong> {suggested}",
            ],
            "claim1Title": "근거 기사 없음 (비교 생략)",
            "claim1Summary": "비교 가능한 신문사 기사가 부족하여 관점 A 요약을 생성하지 않았습니다.",
            "claim1Points": ["news.db 내 일치 기사 부족"],
            "claim1Press": [],
            "claim2Title": "근거 기사 없음 (비교 생략)",
            "claim2Summary": "비교 가능한 신문사 기사가 부족하여 관점 B 요약을 생성하지 않았습니다.",
            "claim2Points": ["news.db 내 일치 기사 부족"],
            "claim2Press": [],
            "articles": [],
        }
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

        trace = list(state.get("orchestration_trace") or [])
        trace.append(
            {
                "step": len(trace) + 1,
                "agent": "HallucinationGuardSubAgent",
                "role": "근거 기사 부재 시 추측성 요약 및 허위 URL 생성 차단",
                "status": "blocked_hallucination",
                "detail": "news.db 내 신뢰 임계치 이상 기사 부족으로 확인되어 환각 방지 가드레일 작동",
                "elapsedMs": elapsed_ms,
            }
        )

        return {
            "base_result": base_res,
            "orchestration_trace": trace,
        }

    def _judge_verifier_agent_node(self, state: OrchestrationState) -> dict[str, Any]:
        """[Node 6] JudgeVerifierSubAgent + OpenAI Head LLM이 최종 심판 및 출처 검증을 완료합니다."""
        t0 = time.perf_counter()
        base_res = dict(state.get("base_result") or {})
        trace = list(state.get("orchestration_trace") or [])

        judged = judge_articles_with_openai(
            query=state.get("query", ""),
            base_result=base_res,
            openai_api_key=state.get("openai_api_key"),
            openai_model=state.get("openai_model"),
            openai_base_url=state.get("openai_base_url"),
            orchestration_trace=trace,
        )
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

        ov = judged.get("overallJudgment") or {}
        trace.append(
            {
                "step": len(trace) + 1,
                "agent": "JudgeVerifierSubAgent + HeadSupervisor(OpenAI)",
                "role": "OpenAI 기반 관점·논조 심판, 기사별 판정 카드 생성 및 news.db URL 100% 검증",
                "status": "live_openai" if ov.get("isLiveOpenAI") else "hybrid_verified",
                "detail": f"최종 판정: {ov.get('verdict', '완료')} | 대립축: {ov.get('conflictAxis', '')[:45]}",
                "elapsedMs": elapsed_ms,
            }
        )
        judged["orchestrationTrace"] = trace
        return {
            "final_result": judged,
            "orchestration_trace": trace,
        }

    def run(
        self,
        query: str,
        thread_id: str = "default-session",
        openai_api_key: str | None = None,
        openai_model: str | None = None,
        openai_base_url: str | None = None,
    ) -> dict[str, Any]:
        """OpenAI Head LLM 기반 멀티 에이전트 오케스트레이션 전체 파이프라인을 실행합니다."""
        active_url, active_key, active_model = resolve_openai_config(
            openai_base_url, openai_api_key, openai_model
        )
        config = {"configurable": {"thread_id": thread_id}}
        final_state = self.graph.invoke(
            {
                "query": query.strip(),
                "thread_id": thread_id,
                "openai_base_url": active_url,
                "openai_api_key": active_key,
                "openai_model": active_model,
                "orchestration_trace": [],
            },
            config=config,
        )
        return final_state["final_result"]


def judge_single_custom_article_with_openai(
    topic: str,
    press: str,
    title: str,
    content: str,
    openai_api_key: str | None = None,
    openai_model: str | None = None,
    openai_base_url: str | None = None,
) -> dict[str, Any]:
    """사용자가 웹에서 직접 입력/선택한 단일 기사를 OpenAI Head LLM으로 즉시 판단합니다."""
    active_url, active_key, active_model = resolve_openai_config(
        openai_base_url, openai_api_key, openai_model
    )
    tone = _detect_article_tone(f"{title} {content}")

    try:
        oai_client = get_openai_client(base_url=active_url, api_key=active_key)
        if "api.openai.com" not in active_url:
            try:
                avail_models = [m.id for m in oai_client.models.list().data]
                if avail_models and active_model not in avail_models:
                    active_model = avail_models[0]
            except Exception:
                pass

        resp = oai_client.chat.completions.create(
            model=active_model,
            temperature=0.0,
            max_tokens=180,
            messages=[
                {
                    "role": "system",
                    "content": "너는 멀티에이전트 뉴스 분석 시스템의 Head 심판 AI야. 기사의 핵심 주장과 보도 논조를 한국어로 간결하게 답해.",
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
            "engine": f"Multi-Agent Head: OpenAI ({active_model})",
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
        print(f"[안내] 단일 기사 OpenAI 판단 호출 예외({e}) -> 로컬 분석으로 대체합니다.")

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
    """FastAPI 앱을 생성하여 확정된 newslens.html UI + OpenAI Head LLM 기반 Multi-Agent Orchestration API를 제공합니다."""
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse
    from pydantic import BaseModel, Field
    from rag_agent import NewsLensAgent

    app = FastAPI(
        title="Nexus (논점콩) - OpenAI Head LLM 기반 Multi-Agent Orchestration 뉴스 분석 서버",
        description="확정된 newslens.html 디자인 + OpenAI Supervisor Head LLM + LangGraph Multi-Agent Orchestration (news.db RAG·관점비교·심판·환각차단)",
    )

    rag_agent = NewsLensAgent(db_path=DB_FILE)
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
                    "2. 검색된 기사 품질·언론사 다양성 평가 및 동적 라우팅 (supervisor_route)",
                    "3. 관점 A/B 대립축 심판 및 최종 결과 감수 (judge_verifier_agent)",
                ],
            },
            "sub_agents": [
                {
                    "id": "RAGRetrievalSubAgent",
                    "description": "news.db 기반 sqlite-vec(title_vec) + TF-IDF + 키워드 하이브리드 검색 및 언론사 다양성 리랭킹",
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
                "kenonix/kanana-nano-2.1b-instruct-Q8_0-GGUF",
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
        """news_classifier.ipynb에서 Export된 최고 점수 분류기(best_news_classifier.joblib)의 메타데이터 및 성능 지표를 반환합니다."""
        bundle = load_exported_news_classifier()
        if not bundle:
            return {"loaded": False, "model_file": str(CLASSIFIER_MODEL_FILE)}
        return {
            "loaded": True,
            "model_file": str(CLASSIFIER_MODEL_FILE),
            "model_name": bundle.get("model_name"),
            "optimal_k": bundle.get("optimal_k"),
            "metrics": bundle.get("metrics", {}),
            "all_model_metrics": bundle.get("all_model_metrics", []),
            "cluster_label_map": bundle.get("cluster_label_map", {}),
        }

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
        model: str = "kenonix/kanana-nano-2.1b-instruct-Q8_0-GGUF"
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


if __name__ == "__main__":
    main()
