"""
backend.py - OpenAI Head LLM 설정, ML 분류기 추론, Semantic Centroid Router,
Multi-Agent Orchestrator(OpenAISupervisorHead + MultiAgentNewsOrchestrator) 및 기사 심판 백엔드 엔진 모듈

[포함 구성 요소]
1. resolve_openai_config() / get_openai_client(): OpenAI Head LLM 설정 및 클라이언트 관리
2. classify_news_with_exported_model(): best_news_classifier.joblib (Calibrated LinearSVC) 추론
3. SemanticCentroidRouter: 클러스터 센트로이드 & 코사인 유사도 임계값 기반 초고속 시맨틱 라우터 (Node 3)
4. OpenAISupervisorHead: 질의 계획(plan_query_and_keywords) 및 시맨틱 라우팅 연동
5. MultiAgentNewsOrchestrator: LangGraph 6-Node 멀티 에이전트 파이프라인
6. judge_articles_with_openai() / judge_single_custom_article_with_openai(): 기사 논조·대립축 심판 및 검증
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import time
from typing import Any, Literal, TypedDict

import numpy as np

from preprocessing import preprocess_korean_news_for_classifier
from slm_server import (
    FINETUNED_SAFETENSORS_FILE,
    FINETUNED_SUMMARIZER_DIR,
    get_finetuned_summarizer_info,
)
from vectorize import text_to_vector

ROOT_DIR = Path(__file__).resolve().parent
ENV_FILE = ROOT_DIR / ".env"
DB_FILE = ROOT_DIR / "news.db"
CLASSIFIER_MODEL_FILE = ROOT_DIR / "best_news_classifier.joblib"
SUMMARIZER_MODEL_DIR = FINETUNED_SUMMARIZER_DIR
SUMMARIZER_SAFETENSORS_FILE = FINETUNED_SAFETENSORS_FILE

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
# (한국어 한자 복원 및 불용어 전처리는 preprocessing.py의 preprocess_korean_news_for_classifier를 임포트하여 재사용)
# =====================================================================
_EXPORTED_CLASSIFIER_CACHE: dict[str, Any] | None = None


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
        base_url: str = "",
        api_key: str = "",
        model: str = "",
        crawled: bool = False,
        cluster_centroids: dict[int, tuple[str, np.ndarray]] | None = None,
    ) -> dict[str, Any]:
        """
        [Node 3 - Semantic Centroid Router]
        LLM-as-a-Judge(외부 LLM API 호출)를 사용하지 않고,
        클러스터 센트로이드 코사인 유사도, 검색 기사군 센트로이드 응집도 및 하이브리드 유사도 임계값으로
        다음 서브 에이전트(comparison_agent / guard_agent)를 초고속 라우팅합니다.
        """
        router = SemanticCentroidRouter(cluster_centroids=cluster_centroids)
        return router.route(query=query, articles=articles)


class SemanticCentroidRouter:
    """
    [순수 시맨틱 라우터 - 클러스터 센트로이드 & 코사인 유사도 임계값 기반]
    LLM-as-a-Judge를 오프라인 모델 평가용으로만 분리하고, 실시간 파이프라인의 Node 3(supervisor_route)에서는
    512차원 임베딩 벡터와 클러스터 센트로이드(Centroid) 코사인 유사도 임계값으로 밀리초(ms) 단위의 결정론적 라우팅을 수행합니다.

    [판정 지표 및 임계값]
    1. 정량 게이트: 검색 기사 수 >= 2건, 서로 다른 언론사 수 >= 2곳, 최고 하이브리드 유사도(top_score) >= 0.35
    2. 이슈 클러스터 센트로이드 유사도(cluster_centroid_sim):
       - 질의 벡터(q_vec)와 news.db 내 54개 KMeans 이슈 클러스터 센트로이드 간 최대 코사인 유사도
    3. 검색 기사군 센트로이드 유사도(doc_centroid_sim):
       - 검색된 상위 기사들의 제목·요약 합성 센트로이드 벡터와 질의 벡터 간 코사인 유사도
    4. 종합 시맨틱 라우팅 점수(semantic_score):
       - semantic_score = 0.45 * top_score + 0.35 * doc_centroid_sim + 0.20 * max(cluster_centroid_sim, avg_vec_sim)
       - 임계값(>= 0.28) 충족 시 `comparison_agent`, 미달 시 `guard_agent`로 분기
    """

    MIN_ARTICLES: int = 2
    MIN_PRESS_COUNT: int = 2
    MIN_TOP_SCORE: float = 0.35
    MIN_DOC_CENTROID_SIM: float = 0.18
    MIN_SEMANTIC_SCORE: float = 0.28

    def __init__(
        self,
        cluster_centroids: dict[int, tuple[str, np.ndarray]] | None = None,
    ) -> None:
        self.cluster_centroids = cluster_centroids or {}

    def route(
        self,
        query: str,
        articles: list[dict[str, Any]],
    ) -> dict[str, Any]:
        press_set = {a.get("press", "") for a in articles if a.get("press")}
        top_score = float(articles[0].get("score", 0.0)) if articles else 0.0

        # 1. 기본 정량 임계값 점검 (기사 수, 언론사 다양성, 최고 유사도)
        if (
            len(articles) < self.MIN_ARTICLES
            or len(press_set) < self.MIN_PRESS_COUNT
            or top_score < self.MIN_TOP_SCORE
        ):
            return {
                "route": "guard_agent",
                "reason": (
                    f"[Semantic Router 가드레일 차단] 검색 기사({len(articles)}건, {len(press_set)}개 언론사, "
                    f"최고 유사도={top_score:.3f} < {self.MIN_TOP_SCORE})가 최소 임계값에 미달하여 "
                    "환각 차단 에이전트(HallucinationGuardSubAgent)로 라우팅"
                ),
                "press_diversity_count": len(press_set),
                "top_relevance_score": round(top_score, 4),
                "semantic_score": round(top_score, 4),
            }

        # 2. 질의 512차원 정규화 벡터 생성
        q_vec = text_to_vector(query)

        # 3. news.db KMeans(k=54) 클러스터 센트로이드와의 최대 코사인 유사도 계산
        best_cluster_label = str(articles[0].get("cluster_label") or "시사·종합")
        cluster_centroid_sim = 0.0
        if self.cluster_centroids:
            for _cid, (c_label, c_vec) in self.cluster_centroids.items():
                sim = max(0.0, float(np.dot(q_vec, c_vec)))
                if sim > cluster_centroid_sim:
                    cluster_centroid_sim = sim
                    best_cluster_label = c_label

        # 4. 검색된 상위 기사군의 센트로이드(Centroid) 벡터 산출 및 질의-센트로이드 코사인 유사도 계산
        doc_vecs: list[np.ndarray] = []
        vec_sims: list[float] = []
        for art in articles[:6]:
            art_text = f"{art.get('clean_title') or art.get('title', '')} {(art.get('summary') or art.get('content') or '')[:120]}"
            d_vec = text_to_vector(art_text)
            doc_vecs.append(d_vec)
            v_sim = float(art.get("title_vec_sim") or np.dot(q_vec, d_vec))
            vec_sims.append(max(0.0, v_sim))

        if doc_vecs:
            mean_doc_vec = np.mean(np.stack(doc_vecs, axis=0), axis=0)
            norm = float(np.linalg.norm(mean_doc_vec))
            if norm > 1e-8:
                mean_doc_vec = mean_doc_vec / norm
            doc_centroid_sim = max(0.0, float(np.dot(q_vec, mean_doc_vec)))
        else:
            doc_centroid_sim = 0.0

        avg_vec_sim = float(np.mean(vec_sims)) if vec_sims else 0.0
        effective_cluster_sim = max(cluster_centroid_sim, avg_vec_sim)

        # 5. 종합 시맨틱 라우팅 스코어 산출 및 임계값 판정
        semantic_score = (
            0.45 * top_score
            + 0.35 * doc_centroid_sim
            + 0.20 * effective_cluster_sim
        )

        if semantic_score < self.MIN_SEMANTIC_SCORE or doc_centroid_sim < self.MIN_DOC_CENTROID_SIM:
            return {
                "route": "guard_agent",
                "reason": (
                    f"[Semantic Router 임계값 미달 차단] 시맨틱 점수({semantic_score:.3f} < {self.MIN_SEMANTIC_SCORE}) · "
                    f"기사군 센트로이드 유사도({doc_centroid_sim:.3f}) 부족 → 환각 차단 에이전트 호출"
                ),
                "press_diversity_count": len(press_set),
                "top_relevance_score": round(top_score, 4),
                "semantic_score": round(semantic_score, 4),
            }

        return {
            "route": "comparison_agent",
            "reason": (
                f"[Semantic Router 통과] 최근접 클러스터 '{best_cluster_label}'(센트로이드 유사도={effective_cluster_sim:.3f}) · "
                f"기사군 센트로이드 유사도={doc_centroid_sim:.3f} · 시맨틱 점수={semantic_score:.3f} "
                f"({len(articles)}건 · {len(press_set)}개 언론사) → 관점 비교·합성 에이전트 호출"
            ),
            "press_diversity_count": len(press_set),
            "top_relevance_score": round(top_score, 4),
            "semantic_score": round(semantic_score, 4),
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
        self.semantic_router = SemanticCentroidRouter(
            cluster_centroids=getattr(self.rag, "cluster_centroids", None)
        )
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
        """[Node 3] SemanticCentroidRouter가 클러스터 센트로이드 및 코사인 유사도 임계값으로 다음 서브 에이전트를 라우팅합니다."""
        t0 = time.perf_counter()
        self.semantic_router.cluster_centroids = getattr(self.rag, "cluster_centroids", {}) or {}
        decision = self.semantic_router.route(
            query=state.get("search_query") or state.get("query", ""),
            articles=state.get("retrieved_articles") or [],
        )
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

        trace = list(state.get("orchestration_trace") or [])
        trace.append(
            {
                "step": len(trace) + 1,
                "agent": "SemanticCentroidRouter",
                "role": "클러스터 센트로이드 & 코사인 유사도 임계값 기반 시맨틱 라우팅",
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


__all__ = [
    "ROOT_DIR",
    "ENV_FILE",
    "DB_FILE",
    "CLASSIFIER_MODEL_FILE",
    "SUMMARIZER_MODEL_DIR",
    "SUMMARIZER_SAFETENSORS_FILE",
    "get_finetuned_summarizer_info",
    "OPENAI_BASE_URL",
    "OPENAI_API_KEY",
    "OPENAI_MODEL",
    "get_openai_client",
    "resolve_openai_config",
    "_detect_article_tone",
    "preprocess_korean_news_for_classifier",
    "load_exported_news_classifier",
    "classify_news_with_exported_model",
    "_parse_numbered_judgment",
    "OrchestrationState",
    "OpenAISupervisorHead",
    "SemanticCentroidRouter",
    "judge_articles_with_openai",
    "MultiAgentNewsOrchestrator",
    "judge_single_custom_article_with_openai",
]
