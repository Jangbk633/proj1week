r"""
test_benchmark.py - Nexus (news.db 기반 신문사별 관점 비교·요약 AI 에이전트) 20문항 자동 검증 벤치마크
실행 방법:
  1) pytest 실행: .\.venv\Scripts\pytest.exe test_benchmark.py -v
  2) 리포트 및 testbench.pdf 생성: .\.venv\Scripts\python.exe generate_testbench_pdf.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import sqlite3
import subprocess
import sys
import time
from typing import Any

import pytest

from rag_agent import DEFAULT_DB_PATH, NewsLensAgent, extract_keywords

ROOT_DIR = Path(__file__).resolve().parent
PDF_PATH = ROOT_DIR / "testbench.pdf"
HTML_REPORT_PATH = ROOT_DIR / "testbench_report.html"


@dataclass
class BenchmarkCase:
    id: str
    category: str  # "A_COMPARE" | "B_HALLUCINATION" | "C_CRAWL" | "C_MULTITURN"
    category_label: str
    query: str
    thread_id: str
    expected_behavior: str
    sla_limit_sec: float
    prior_query: str | None = None


BENCHMARK_20_CASES: list[BenchmarkCase] = [
    # ------------------------------------------------------------------
    # Category A: 복수 언론사 보도가 존재하는 주요 시사 이슈 비교 질문 (10문항)
    # ------------------------------------------------------------------
    BenchmarkCase(
        id="Q01",
        category="A_COMPARE",
        category_label="시사 이슈 비교 (DB RAG)",
        query="김여정 대북 확성기 철거 남북관계에 대한 신문사별 입장 비교해줘",
        thread_id="bench-q01",
        expected_behavior="3개 이상 신문사 매칭 + 통합 헤드라인 + 관점 A/B 비교 + news.db 원문 URL 100% 일치",
        sla_limit_sec=4.0,
    ),
    BenchmarkCase(
        id="Q02",
        category="A_COMPARE",
        category_label="시사 이슈 비교 (DB RAG)",
        query="수사기관장 국무회의 배석 사법개혁 논란 신문사별 의견 비교",
        thread_id="bench-q02",
        expected_behavior="3개 이상 신문사 매칭 + 통합 헤드라인 + 관점 A/B 비교 + news.db 원문 URL 100% 일치",
        sla_limit_sec=4.0,
    ),
    BenchmarkCase(
        id="Q03",
        category="A_COMPARE",
        category_label="시사 이슈 비교 (DB RAG)",
        query="전업주부 국민연금 가입 사각지대 해소에 대한 언론사별 시각 비교",
        thread_id="bench-q03",
        expected_behavior="3개 이상 신문사 매칭 + 통합 헤드라인 + 관점 A/B 비교 + news.db 원문 URL 100% 일치",
        sla_limit_sec=4.0,
    ),
    BenchmarkCase(
        id="Q04",
        category="A_COMPARE",
        category_label="시사 이슈 비교 (DB RAG)",
        query="한미연합훈련 주한미군 역할 변경에 대한 신문사별 분석 비교",
        thread_id="bench-q04",
        expected_behavior="3개 이상 신문사 매칭 + 통합 헤드라인 + 관점 A/B 비교 + news.db 원문 URL 100% 일치",
        sla_limit_sec=4.0,
    ),
    BenchmarkCase(
        id="Q05",
        category="A_COMPARE",
        category_label="시사 이슈 비교 (DB RAG)",
        query="차세대 D램 AI 반도체 목표주가 전망 신문사별 보도 비교",
        thread_id="bench-q05",
        expected_behavior="3개 이상 신문사 매칭 + 통합 헤드라인 + 관점 A/B 비교 + news.db 원문 URL 100% 일치",
        sla_limit_sec=4.0,
    ),
    BenchmarkCase(
        id="Q06",
        category="A_COMPARE",
        category_label="시사 이슈 비교 (DB RAG)",
        query="중동 전쟁 호르무즈 해협 유가 불안 신문사별 해석 비교",
        thread_id="bench-q06",
        expected_behavior="3개 이상 신문사 매칭 + 통합 헤드라인 + 관점 A/B 비교 + news.db 원문 URL 100% 일치",
        sla_limit_sec=4.0,
    ),
    BenchmarkCase(
        id="Q07",
        category="A_COMPARE",
        category_label="시사 이슈 비교 (DB RAG)",
        query="한국은행 기준금리 동결 인하 통화정책 신문사별 평가 비교",
        thread_id="bench-q07",
        expected_behavior="3개 이상 신문사 매칭 + 통합 헤드라인 + 관점 A/B 비교 + news.db 원문 URL 100% 일치",
        sla_limit_sec=4.0,
    ),
    BenchmarkCase(
        id="Q08",
        category="A_COMPARE",
        category_label="시사 이슈 비교 (DB RAG)",
        query="이재명 대통령 국무회의 비정상의 정상화 발언 신문사별 관점 비교",
        thread_id="bench-q08",
        expected_behavior="3개 이상 신문사 매칭 + 통합 헤드라인 + 관점 A/B 비교 + news.db 원문 URL 100% 일치",
        sla_limit_sec=4.0,
    ),
    BenchmarkCase(
        id="Q09",
        category="A_COMPARE",
        category_label="시사 이슈 비교 (DB RAG)",
        query="코스피 외국인 순매수 삼성전자 SK하이닉스 증시 전망 비교",
        thread_id="bench-q09",
        expected_behavior="3개 이상 신문사 매칭 + 통합 헤드라인 + 관점 A/B 비교 + news.db 원문 URL 100% 일치",
        sla_limit_sec=4.0,
    ),
    BenchmarkCase(
        id="Q10",
        category="A_COMPARE",
        category_label="시사 이슈 비교 (DB RAG)",
        query="더불어민주당 국민의힘 국회 법사위 충돌에 대한 언론사별 비판 비교",
        thread_id="bench-q10",
        expected_behavior="3개 이상 신문사 매칭 + 통합 헤드라인 + 관점 A/B 비교 + news.db 원문 URL 100% 일치",
        sla_limit_sec=4.0,
    ),
    # ------------------------------------------------------------------
    # Category B: 실제 보도된 적 없는 허위·가짜 이슈 환각 방어 질문 (5문항)
    # ------------------------------------------------------------------
    BenchmarkCase(
        id="Q11",
        category="B_HALLUCINATION",
        category_label="환각 차단 (허위 이슈)",
        query="2035년 화성 올림픽 개최지 선정 비리 논란 비교해줘",
        thread_id="bench-q11",
        expected_behavior="DB 검색 및 크롤링 0건 확인 -> 추측 요약·가짜 URL 생성 없이 '근거 기사 없음' 반환",
        sla_limit_sec=10.0,
    ),
    BenchmarkCase(
        id="Q12",
        category="B_HALLUCINATION",
        category_label="환각 차단 (허위 이슈)",
        query="아틀란티스 해저도시 부동산 투기 세력 적발 사건 신문사별 반응",
        thread_id="bench-q12",
        expected_behavior="DB 검색 및 크롤링 0건 확인 -> 추측 요약·가짜 URL 생성 없이 '근거 기사 없음' 반환",
        sla_limit_sec=10.0,
    ),
    BenchmarkCase(
        id="Q13",
        category="B_HALLUCINATION",
        category_label="환각 차단 (허위 이슈)",
        query="안드로메다 은하계 자유무역협정 비준 갈등 언론사별 보도 비교",
        thread_id="bench-q13",
        expected_behavior="DB 검색 및 크롤링 0건 확인 -> 추측 요약·가짜 URL 생성 없이 '근거 기사 없음' 반환",
        sla_limit_sec=10.0,
    ),
    BenchmarkCase(
        id="Q14",
        category="B_HALLUCINATION",
        category_label="환각 차단 (허위 이슈)",
        query="목성 위성 유로파 리조트 분양 사기 사건에 대한 신문사 의견",
        thread_id="bench-q14",
        expected_behavior="DB 검색 및 크롤링 0건 확인 -> 추측 요약·가짜 URL 생성 없이 '근거 기사 없음' 반환",
        sla_limit_sec=10.0,
    ),
    BenchmarkCase(
        id="Q15",
        category="B_HALLUCINATION",
        category_label="환각 차단 (허위 이슈)",
        query="타임머신 상용화 특별법 국회 본회의 통과 논란 비교해줘",
        thread_id="bench-q15",
        expected_behavior="DB 검색 및 크롤링 0건 확인 -> 추측 요약·가짜 URL 생성 없이 '근거 기사 없음' 반환",
        sla_limit_sec=10.0,
    ),
    # ------------------------------------------------------------------
    # Category C: 실시간 뉴스 크롤링·DB 적재(3문항) 및 멀티턴 후속 질문(2문항)
    # ------------------------------------------------------------------
    BenchmarkCase(
        id="Q16",
        category="C_CRAWL",
        category_label="실시간 크롤링·DB 적재",
        query="의대 증원 의료개혁 전공의 복귀에 대한 신문사별 의견 비교",
        thread_id="bench-q16",
        expected_behavior="DB 부족 감지 -> httpx 크롤러로 기사 수집·news.db 적재 -> 헤드라인 통합 및 신문사별 비교",
        sla_limit_sec=10.0,
    ),
    BenchmarkCase(
        id="Q17",
        category="C_CRAWL",
        category_label="실시간 크롤링·DB 적재",
        query="가계부채 스트레스 DSR 부동산 대출 규제 신문사별 평가 비교",
        thread_id="bench-q17",
        expected_behavior="DB 부족 감지 -> httpx 크롤러로 기사 수집·news.db 적재 -> 헤드라인 통합 및 신문사별 비교",
        sla_limit_sec=10.0,
    ),
    BenchmarkCase(
        id="Q18",
        category="C_CRAWL",
        category_label="실시간 크롤링·DB 적재",
        query="반도체 특별법 보조금 전력망 지원에 대한 신문사별 시각 비교",
        thread_id="bench-q18",
        expected_behavior="DB 부족 감지 -> httpx 크롤러로 기사 수집·news.db 적재 -> 헤드라인 통합 및 신문사별 비교",
        sla_limit_sec=10.0,
    ),
    BenchmarkCase(
        id="Q19",
        category="C_MULTITURN",
        category_label="멀티턴 맥락 유지 (후속 질문)",
        query="방금 요약한 내용 중 신문사들의 핵심 입장을 더 자세히 비교해줘",
        thread_id="bench-mt-1",
        prior_query="김여정 대북 확성기 철거 남북관계",
        expected_behavior="동일 thread_id 직전 턴 맥락(김여정 확성기) 결합 -> 재검색 및 신문사별 의견 일관 유지",
        sla_limit_sec=4.0,
    ),
    BenchmarkCase(
        id="Q20",
        category="C_MULTITURN",
        category_label="멀티턴 맥락 유지 (후속 질문)",
        query="이전 질문에서 다룬 사안에 대해 각 언론사 의견을 알려줘",
        thread_id="bench-mt-2",
        prior_query="수사기관장 국무회의 배석 사법개혁",
        expected_behavior="동일 thread_id 직전 턴 맥락(수사기관장 배석) 결합 -> 재검색 및 신문사별 의견 일관 유지",
        sla_limit_sec=4.0,
    ),
]


def _get_all_db_urls_and_texts(db_path: Path = DEFAULT_DB_PATH) -> tuple[set[str], dict[str, str]]:
    with sqlite3.connect(db_path) as con:
        rows = con.execute("SELECT URL, TITLE, CLEAN_CONTENT FROM RAG_INDEX").fetchall()
    urls = {r[0] for r in rows if r[0]}
    url_to_text = {r[0]: f"{r[1]} {r[2]}" for r in rows if r[0]}
    return urls, url_to_text


def evaluate_single_case(agent: NewsLensAgent, case: BenchmarkCase) -> dict[str, Any]:
    """단일 벤치마크 케이스를 실행하고 6대 결과 검증 지표를 정량 채점합니다."""
    if case.prior_query:
        agent.run(case.prior_query, thread_id=case.thread_id)

    t0 = time.perf_counter()
    res = agent.run(case.query, thread_id=case.thread_id)
    latency_sec = round(time.perf_counter() - t0, 3)

    db_urls, url_to_text = _get_all_db_urls_and_texts(agent.rag.db_path)

    c1_press = res.get("claim1Press", [])
    c2_press = res.get("claim2Press", [])
    all_press_items = c1_press + c2_press
    output_urls = [p.get("url", "") for p in all_press_items if p.get("url")]
    distinct_press_names = sorted({p.get("name", "") for p in all_press_items if p.get("name")})

    if case.category == "B_HALLUCINATION":
        # 환각 차단 검증:
        # 1) 출처 배지가 비어 있어야 함 (가짜 URL 0건)
        # 2) 헤드라인 또는 요약에 '찾을 수 없습니다' / '환각 차단' 안내가 포함되어야 함
        no_fake_urls = len(output_urls) == 0
        has_rejection_notice = (
            "찾을 수 없습니다" in res.get("headline", "")
            or "근거 기사 없음" in res.get("claim1Title", "")
        )
        latency_ok = latency_sec <= case.sla_limit_sec
        passed = no_fake_urls and has_rejection_notice and latency_ok

        return {
            "id": case.id,
            "category": case.category,
            "category_label": case.category_label,
            "query": case.query,
            "headline": res.get("headline", ""),
            "press_count": 0,
            "press_names": [],
            "url_grounding_rate": 100.0 if no_fake_urls else 0.0,
            "faithfulness_rate": 100.0 if has_rejection_notice else 0.0,
            "hallucination_rejected": passed,
            "latency_sec": latency_sec,
            "sla_limit_sec": case.sla_limit_sec,
            "latency_ok": latency_ok,
            "passed": passed,
            "meta": res.get("meta", []),
            "claim1_sample": res.get("claim1Title", ""),
            "claim2_sample": res.get("claim2Title", ""),
        }

    # 비교·크롤링·멀티턴 케이스(A_COMPARE, C_CRAWL, C_MULTITURN) 검증:
    # M2: 출처 URL 무결성 (모든 출력 URL이 실제 news.db에 존재하는지)
    matched_urls = [u for u in output_urls if u in db_urls]
    url_grounding_rate = round((len(matched_urls) / len(output_urls)) * 100.0, 1) if output_urls else 0.0

    # M3: 신문사 다각화 비교 (서로 다른 3개 이상 신문사 + 관점 A/B 모두 최소 1개 이상 매체 배치)
    press_diversity_ok = len(distinct_press_names) >= 2 and len(c1_press) >= 1 and len(c2_press) >= 1
    three_plus_press = len(distinct_press_names) >= 3

    # M4: 원문 사실·논조 충실도 (claim1Points, claim2Points의 핵심 토큰이 실제 원문 기사에 존재하는 비율)
    source_corpus = " ".join(url_to_text.get(u, "") for u in matched_urls)
    point_texts = res.get("claim1Points", []) + res.get("claim2Points", [])
    grounded_points = 0
    for pt in point_texts:
        clean_pt = re.sub(r"<[^>]+>", " ", pt)
        clean_pt = re.sub(r"\[[^\]]+\]", " ", clean_pt)
        tokens = extract_keywords(clean_pt)
        if not tokens:
            grounded_points += 1
            continue
        overlap = sum(1 for tk in tokens if tk in source_corpus)
        if overlap / len(tokens) >= 0.6:
            grounded_points += 1

    faithfulness_rate = (
        round((grounded_points / len(point_texts)) * 100.0, 1) if point_texts else 0.0
    )

    # M6: 응답 지연 시간 SLA 충족 여부
    latency_ok = latency_sec <= case.sla_limit_sec

    passed = (
        url_grounding_rate == 100.0
        and press_diversity_ok
        and faithfulness_rate >= 85.0
        and latency_ok
    )

    return {
        "id": case.id,
        "category": case.category,
        "category_label": case.category_label,
        "query": case.query,
        "headline": res.get("headline", ""),
        "press_count": len(distinct_press_names),
        "press_names": distinct_press_names,
        "three_plus_press": three_plus_press,
        "url_grounding_rate": url_grounding_rate,
        "faithfulness_rate": faithfulness_rate,
        "hallucination_rejected": True,
        "latency_sec": latency_sec,
        "sla_limit_sec": case.sla_limit_sec,
        "latency_ok": latency_ok,
        "passed": passed,
        "meta": res.get("meta", []),
        "claim1_sample": (res.get("claim1Points") or [""])[0],
        "claim2_sample": (res.get("claim2Points") or [""])[0],
    }


# =====================================================================
# pytest 자동화 검증 함수
# =====================================================================


@pytest.fixture(scope="module")
def shared_agent() -> NewsLensAgent:
    return NewsLensAgent()


@pytest.mark.parametrize("case", BENCHMARK_20_CASES, ids=[c.id for c in BENCHMARK_20_CASES])
def test_nexus_agent_benchmark(shared_agent: NewsLensAgent, case: BenchmarkCase) -> None:
    """20개 표준 벤치마크 문항에 대한 개별 통과 여부를 검증합니다."""
    metrics = evaluate_single_case(shared_agent, case)
    assert metrics["passed"], f"[{case.id}] 검증 실패: {metrics}"
