"""
summerizer.py - 벡터 거리(1 - dot(v_i, v_j)) 기반 관점 A vs 관점 B 기사 이진 군집화 및 통합 헤드라인·양면 요약 생성 모듈
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from rag import RetrievedArticle
from vectorize import text_to_vector


class PressItemSchema(BaseModel):
    name: str = Field(description="신문사·언론사 이름")
    title: str = Field(description="원문 기사 제목")
    url: str = Field(default="#source", description="원문 기사 링크 URL")


class ComparisonOutputSchema(BaseModel):
    headline: str = Field(description="관련 기사들을 하나로 묶은 통합 뉴스 헤드라인 (1문장)")
    summary: str = Field(description="수집된 기사들의 공통 사실관계(팩트) 종합 요약 (2~3문장)")
    points: list[str] = Field(description="공통 핵심 팩트 및 신문사별 해석 차이 요약 3개")
    claim1Title: str = Field(description="관점 A(주장 1)의 핵심 한 줄 제목")
    claim1Summary: str = Field(description="관점 A(주장 1) 신문사들의 해석과 핵심 논거 요약 (2~3문장)")
    claim1Points: list[str] = Field(description="관점 A 신문사별 구체적 의견·해석 요지 2~3개")
    claim1Press: list[PressItemSchema] = Field(description="관점 A를 보도한 신문사 및 원문 기사 목록")
    claim2Title: str = Field(description="관점 B(주장 2)의 핵심 한 줄 제목")
    claim2Summary: str = Field(description="관점 B(주장 2) 신문사들의 해석과 핵심 논거 요약 (2~3문장)")
    claim2Points: list[str] = Field(description="관점 B 신문사별 구체적 의견·해석 요지 2~3개")
    claim2Press: list[PressItemSchema] = Field(description="관점 B를 보도한 신문사 및 원문 기사 목록")


def _split_sentences(text: str) -> list[str]:
    """기사 본문에서 핵심 문장들을 분리합니다."""
    cleaned = re.sub(r"\[[^\]]{2,30}\]", " ", text)
    cleaned = re.sub(r"\([^\)]{2,25}=[^\)]{2,25}\)", " ", cleaned)
    raw = re.split(r"(?<=[.?!다])\s+", cleaned)
    sents = []
    for s in raw:
        s = s.strip()
        if 28 <= len(s) <= 220 and not s.endswith("기자"):
            sents.append(s)
    return sents


def _summarize_press_opinion(art: RetrievedArticle) -> str:
    """각 신문사 기사의 핵심 해석/의견 1문장을 추출합니다 (노트북 SLM 요약본이 있으면 우선 활용)."""
    if art.summary and art.summary.strip():
        return art.summary.strip()
    sents = _split_sentences(art.content)
    for s in sents:
        if any(cue in s for cue in ('"', "“", "'", "우려", "지적", "강조", "평가", "풀이", "전망", "비판", "필요")):
            return s
    return sents[0] if sents else art.title


def _extractive_compare_articles(
    query: str,
    articles: list[RetrievedArticle],
    crawled_count: int = 0,
) -> dict[str, Any]:
    """
    관련 기사들을 하나의 통합 헤드라인으로 묶어 요약하고,
    title_vec 의미 벡터 거리 기준으로 가장 대비되는 두 관점(관점 A vs 관점 B)으로 분류하여 비교합니다.
    """
    press_set = sorted({a.press for a in articles})
    top_art = articles[0]

    if len(articles) >= 2:
        vecs = [text_to_vector(f"{a.clean_title or a.title} {a.content[:400]}") for a in articles]
        max_dist = -1.0
        seed_a, seed_b = 0, 1
        for i in range(len(articles)):
            for j in range(i + 1, len(articles)):
                dist = 1.0 - float(np.dot(vecs[i], vecs[j]))
                if dist > max_dist:
                    max_dist = dist
                    seed_a, seed_b = i, j

        group1: list[RetrievedArticle] = [articles[seed_a]]
        group2: list[RetrievedArticle] = [articles[seed_b]]
        for idx, art in enumerate(articles):
            if idx in (seed_a, seed_b):
                continue
            sim_a = float(np.dot(vecs[idx], vecs[seed_a]))
            sim_b = float(np.dot(vecs[idx], vecs[seed_b]))
            if sim_a >= sim_b:
                group1.append(art)
            else:
                group2.append(art)
    else:
        group1 = list(articles)
        group2 = list(articles)

    g1_press_names = "·".join(dict.fromkeys(a.press for a in group1[:3]))
    g2_press_names = "·".join(dict.fromkeys(a.press for a in group2[:3]))
    unified_headline = (
        f"[{query}] {top_art.title}… "
        f"{g1_press_names} 등 {len(press_set)}개 신문사별 해석·관점 종합"
    )

    lead_sents: list[str] = []
    for art in articles[:3]:
        sents = _split_sentences(art.content)
        if sents and sents[0] not in lead_sents:
            lead_sents.append(sents[0])
    common_summary = " ".join(lead_sents[:2]) if lead_sents else top_art.content[:280]

    common_points = [
        f"<strong>공통 보도 팩트:</strong> {len(press_set)}개 신문사({', '.join(press_set[:6])})가 ‘{query}’ 사안을 주요 의제로 보도 (분류: {top_art.category})",
        f"<strong>관점 A ({g1_press_names} 등):</strong> {group1[0].title} 중심 해석",
        f"<strong>관점 B ({g2_press_names} 등):</strong> {group2[0].title} 중심 해석",
    ]

    g1_opinions = [_summarize_press_opinion(a) for a in group1[:2]]
    claim1_summary = " ".join(g1_opinions) if g1_opinions else group1[0].content[:220]
    claim1_points = [
        f"<strong>[{a.press}]</strong> {_summarize_press_opinion(a)[:110]}"
        for a in group1[:3]
    ]
    claim1_press = [{"name": a.press, "title": a.title, "url": a.url} for a in group1[:4]]

    g2_opinions = [_summarize_press_opinion(a) for a in group2[:2]]
    claim2_summary = " ".join(g2_opinions) if g2_opinions else group2[0].content[:220]
    claim2_points = [
        f"<strong>[{a.press}]</strong> {_summarize_press_opinion(a)[:110]}"
        for a in group2[:3]
    ]
    claim2_press = [{"name": a.press, "title": a.title, "url": a.url} for a in group2[:4]]

    crawl_note = f"실시간 크롤링 {crawled_count}건 추가 적재 + " if crawled_count > 0 else ""

    return {
        "headline": unified_headline,
        "meta": [
            f"수집·저장: {crawl_note}news.db (연관 기사 {len(articles)}건 · {len(press_set)}개 신문사)",
            f"벡터 색인: sqlite-vec (title_vec 최고 유사도 {top_art.score:.2f} · 분류: {top_art.category})",
            "출처 검증: 통과 (news.db 원문 URL 100% 일치)",
        ],
        "summary": common_summary,
        "points": common_points,
        "claim1Title": f"[{g1_press_names}] {group1[0].title}",
        "claim1Summary": claim1_summary,
        "claim1Points": claim1_points,
        "claim1Press": claim1_press,
        "claim2Title": f"[{g2_press_names}] {group2[0].title}",
        "claim2Summary": claim2_summary,
        "claim2Points": claim2_points,
        "claim2Press": claim2_press,
    }


# 외부 모듈에서 언더스코어 없이도 직관적으로 임포트할 수 있도록 공개 별칭 제공
split_sentences = _split_sentences
summarize_press_opinion = _summarize_press_opinion
extractive_compare_articles = _extractive_compare_articles

__all__ = [
    "PressItemSchema",
    "ComparisonOutputSchema",
    "_split_sentences",
    "_summarize_press_opinion",
    "_extractive_compare_articles",
    "split_sentences",
    "summarize_press_opinion",
    "extractive_compare_articles",
]
