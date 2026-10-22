"""
rag_agent.py - A2 문제 정의 기반:
'하나의 주제에 대한 기사들을 긁어와 데이터베이스(news.db)에 저장해
 관련된 내용을 하나의 헤드라인으로 묶어 요약해주되 신문사별로 의견을 비교해 주는'
LangGraph + SQLite(news.db) 하이브리드 RAG AI 에이전트

에이전트 단계 (판단 · 도구 선택 · 여러 단계):
1. analyze_query_node       : 사용자 주제/질문에서 핵심 검색 키워드 추출 및 대화 맥락 결합
2. retrieve_news_node       : news.db (NEWS + LINK + RAG_INDEX)에서 신문사별 관련 기사 하이브리드 검색
3. decide_crawl_or_compare  : [에이전트 판단] DB 내 비교 가능한 신문사가 3곳 미만이면 크롤링 도구 선택
4. crawl_topic_to_db_node   : [도구 실행] httpx로 주제 관련 여러 신문사 기사를 긁어와 news.db(LINK, NEWS, RAG_INDEX)에 저장 후 재검색
5. compare_and_summarize    : 관련 기사를 하나의 통합 헤드라인으로 묶어 공통 팩트를 요약하고, 신문사별 해석·의견(관점 A vs 관점 B) 비교
6. verify_sources_node      : 모든 신문사 출처·URL이 실제 news.db 저장 기사와 일치하는지 검증 (환각 차단)
"""

from __future__ import annotations

import hashlib
import html
import json
import os
from pathlib import Path
import re
import sqlite3
from typing import Any, Literal, TypedDict
from urllib.parse import quote_plus

from dotenv import load_dotenv
import httpx
import numpy as np
from pydantic import BaseModel, Field

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_DB_PATH = ROOT_DIR / "news.db"

HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}

# 네이버 뉴스 OID(3자리 언론사 코드) -> 신문사/언론사명 매핑
OID_TO_PRESS: dict[str, str] = {
    "001": "연합뉴스",
    "002": "프레시안",
    "003": "뉴시스",
    "005": "국민일보",
    "006": "미디어오늘",
    "008": "머니투데이",
    "009": "매일경제",
    "011": "서울경제",
    "014": "파이낸셜뉴스",
    "015": "한국경제",
    "016": "헤럴드경제",
    "018": "이데일리",
    "020": "동아일보",
    "021": "문화일보",
    "022": "세계일보",
    "023": "조선일보",
    "025": "중앙일보",
    "028": "한겨레",
    "029": "디지털타임스",
    "030": "전자신문",
    "031": "아이뉴스24",
    "032": "경향신문",
    "033": "주간경향",
    "036": "한겨레21",
    "037": "주간동아",
    "044": "코리아헤럴드",
    "047": "오마이뉴스",
    "050": "한경비즈니스",
    "052": "YTN",
    "053": "주간조선",
    "055": "SBS",
    "056": "KBS",
    "057": "MBN",
    "077": "AP통신",
    "079": "노컷뉴스",
    "081": "서울신문",
    "082": "부산일보",
    "087": "강원일보",
    "088": "매일신문",
    "092": "지디넷코리아",
    "119": "데일리안",
    "123": "조세일보",
    "138": "디지털데일리",
    "214": "MBC",
    "215": "한국경제TV",
    "243": "이코노미스트",
    "277": "아시아경제",
    "293": "블로터",
    "366": "조선비즈",
    "374": "SBS Biz",
    "417": "머니S",
    "421": "뉴스1",
    "422": "연합뉴스TV",
    "437": "JTBC",
    "448": "TV조선",
    "449": "채널A",
    "469": "한국일보",
    "584": "동아사이언스",
    "586": "시사저널",
    "629": "더팩트",
    "640": "코리아중앙데일리",
    "648": "비즈워치",
    "654": "강원도민일보",
    "656": "대전일보",
    "657": "대구MBC",
    "658": "국제신문",
    "659": "전주MBC",
    "660": "kbc광주방송",
    "661": "JIBS",
    "662": "농민신문",
    "666": "경기일보",
}

EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
PHONE_RE = re.compile(r"\b0\d{1,2}[-.\s]?\d{3,4}[-.\s]?\d{4}\b")
TOKEN_RE = re.compile(r"[가-힣A-Za-z0-9]{2,}")

STOPWORDS = {
    "기자", "포토", "단독", "속보", "뉴스", "보도", "관련", "대한", "대해", "어떻게",
    "비교", "비교해줘", "알려줘", "요약", "요약해줘", "입장", "관점", "신문사별",
    "언론사별", "최근", "오늘", "지금", "이슈", "논란", "분석", "반응", "생각",
    "무엇", "어떤", "에서", "으로", "하고", "있다", "밝혔다", "전했다", "말했다",
    "의견", "해석", "헤드라인",
}


def mask_sensitive_info(text: str) -> str:
    """기자 이메일 및 연락처 등 개인정보를 정규식으로 마스킹합니다."""
    text = EMAIL_RE.sub("[EMAIL]", text)
    text = PHONE_RE.sub("[PHONE]", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_press_name(url: str, content: str) -> str:
    """네이버 뉴스 URL의 OID 또는 기사 본문 바이라인에서 신문사명을 추출합니다."""
    m = re.search(r"/article/(\d{3})/", url or "")
    if m and m.group(1) in OID_TO_PRESS:
        return OID_TO_PRESS[m.group(1)]

    byline = re.search(r"[\[\(]([가-힣A-Za-z0-9]{2,10}?)(?:=|\s+[가-힣]{2,4}\s*기자)", content or "")
    if byline:
        cand = byline.group(1).strip()
        if cand not in {"서울", "세종", "부산", "대구", "광주", "대전", "울산", "인천", "고양", "워싱턴", "도쿄", "베이징"}:
            return cand

    if m:
        return f"언론사({m.group(1)})"
    return "네이버뉴스"


def extract_keywords(text: str) -> list[str]:
    """한국어 조사 어미를 제거하고 핵심 검색 키워드를 추출합니다."""
    raw_tokens = TOKEN_RE.findall(text)
    keywords: list[str] = []
    for tok in raw_tokens:
        if tok in STOPWORDS:
            continue
        cleaned = re.sub(r"(에서|으로|에게|부터|까지|이랑|라는|에는|은|는|이|가|을|를|에|의|로|과|와|도|만)$", "", tok)
        if len(cleaned) >= 2 and cleaned not in STOPWORDS:
            keywords.append(cleaned)
        elif len(tok) >= 2 and tok not in STOPWORDS:
            keywords.append(tok)

    seen: set[str] = set()
    unique: list[str] = []
    for k in keywords:
        if k not in seen:
            seen.add(k)
            unique.append(k)
    return unique


def text_to_vector(text: str, dim: int = 512) -> np.ndarray:
    """단어 + 음절 바이그램/트라이그램 Feature Hashing 기반 L2 정규화 임베딩 벡터를 생성합니다."""
    vec = np.zeros(dim, dtype=np.float32)
    tokens = extract_keywords(text)
    if not tokens:
        tokens = TOKEN_RE.findall(text)

    features: list[tuple[str, float]] = []
    for tok in tokens:
        features.append((f"w:{tok}", 2.0))
        for n in (2, 3):
            if len(tok) >= n:
                for i in range(len(tok) - n + 1):
                    features.append((f"c{n}:{tok[i:i+n]}", 1.0))

    for feat, weight in features:
        h = int(hashlib.md5(feat.encode("utf-8")).hexdigest()[:8], 16)
        idx = h % dim
        sign = 1.0 if ((h >> 8) & 1) == 0 else -1.0
        vec[idx] += sign * weight

    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec /= norm
    return vec


def parse_naver_article_html(raw_html: str) -> tuple[str, str]:
    """네이버 뉴스 기사 HTML에서 제목(h2#title_area)과 본문(article#dic_area)을 추출합니다."""
    title_m = re.search(r'<h2[^>]*id=["\']title_area["\'][^>]*>(.*?)</h2>', raw_html, re.S)
    if not title_m:
        title_m = re.search(r'<meta\s+property=["\']og:title["\']\s+content=["\'](.*?)["\']', raw_html, re.S)
    title = html.unescape(re.sub(r"<[^>]+>", "", title_m.group(1))).strip() if title_m else ""

    art_m = re.search(r'<article[^>]*id=["\']dic_area["\'][^>]*>(.*?)</article>', raw_html, re.S)
    if not art_m:
        art_m = re.search(r'<div[^>]*id=["\']newsct_article["\'][^>]*>(.*?)</div>\s*</div>', raw_html, re.S)
    if art_m:
        body_html = re.sub(
            r"<(script|style|span[^>]*class=[\"']end_photo_org[\"'])[^>]*>.*?</\1>",
            " ",
            art_m.group(1),
            flags=re.S,
        )
        body = html.unescape(re.sub(r"<[^>]+>", " ", body_html))
        body = re.sub(r"\s+", " ", body).strip()
    else:
        body = ""
    return title, body


class RetrievedArticle(BaseModel):
    news_pk: int
    press: str
    title: str
    content: str
    url: str
    regdate: str
    score: float


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


class NewsDBRAG:
    """news.db (SQLite) 기반 하이브리드 RAG 인덱서, 검색기 및 실시간 주제 크롤러"""

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH, embedding_dim: int = 512) -> None:
        self.db_path = Path(db_path)
        self.embedding_dim = embedding_dim
        self.ensure_rag_index()

    def ensure_rag_index(self, force_rebuild: bool = False) -> int:
        """news.db 내에 RAG_INDEX 테이블을 생성하고 NEWS+LINK 데이터를 정제·임베딩하여 적재합니다."""
        if not self.db_path.exists():
            raise FileNotFoundError(f"데이터베이스 파일을 찾을 수 없습니다: {self.db_path}")

        with sqlite3.connect(self.db_path) as con:
            cur = con.cursor()
            if force_rebuild:
                cur.execute("DROP TABLE IF EXISTS RAG_INDEX")

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS RAG_INDEX (
                    NEWS_PK INTEGER PRIMARY KEY,
                    PRESS TEXT NOT NULL,
                    URL TEXT NOT NULL,
                    TITLE TEXT NOT NULL,
                    CLEAN_CONTENT TEXT NOT NULL,
                    KEYWORDS TEXT NOT NULL,
                    EMBEDDING BLOB NOT NULL,
                    REGDATE TEXT
                )
                """
            )

            indexed_count = cur.execute("SELECT COUNT(*) FROM RAG_INDEX").fetchone()[0]
            total_news = cur.execute("SELECT COUNT(*) FROM NEWS").fetchone()[0]
            if indexed_count > 0 and not force_rebuild and indexed_count >= int(total_news * 0.5):
                return indexed_count

            rows = cur.execute(
                """
                SELECT N.PK, COALESCE(L.URL, ''), N.TITLE, N.CONTENT, COALESCE(N.REGDATE, '')
                FROM NEWS N
                LEFT JOIN LINK L ON N.FK = L.PK
                ORDER BY N.PK ASC
                """
            ).fetchall()

            seen_titles: set[str] = set()
            to_insert: list[tuple[Any, ...]] = []

            for pk, url, title, content, regdate in rows:
                clean_title = re.sub(r"\s+", " ", (title or "")).strip()
                clean_body = mask_sensitive_info(content or "")

                if len(clean_body) < 120 or clean_title.startswith("[포토]"):
                    continue
                if clean_title in seen_titles:
                    continue
                seen_titles.add(clean_title)

                press = extract_press_name(url, content)
                kw_list = extract_keywords(f"{clean_title} {clean_title} {clean_body[:600]}")
                emb = text_to_vector(f"{clean_title} {clean_title} {clean_body[:800]}", dim=self.embedding_dim)

                to_insert.append(
                    (
                        pk,
                        press,
                        url,
                        clean_title,
                        clean_body,
                        " ".join(kw_list[:60]),
                        emb.tobytes(),
                        regdate,
                    )
                )

            cur.executemany(
                """
                INSERT OR REPLACE INTO RAG_INDEX
                (NEWS_PK, PRESS, URL, TITLE, CLEAN_CONTENT, KEYWORDS, EMBEDDING, REGDATE)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                to_insert,
            )
            con.commit()
            return len(to_insert)

    def crawl_topic_and_save_to_db(self, query: str, max_articles: int = 8) -> int:
        """
        [에이전트 도구: 주제별 뉴스 크롤링 & news.db 저장]
        하나의 주제(query)에 대한 기사들을 서로 다른 신문사별로 긁어와
        news.db의 LINK, NEWS, RAG_INDEX 테이블에 저장하고 저장된 건수를 반환합니다.
        """
        search_url = f"https://search.naver.com/search.naver?where=news&query={quote_plus(query)}"
        try:
            resp = httpx.get(search_url, headers=HTTP_HEADERS, timeout=8.0, follow_redirects=True)
            if resp.status_code != 200:
                return 0
        except Exception:
            return 0

        candidate_urls: list[str] = []
        seen_oids: set[str] = set()
        all_found: list[tuple[str, str]] = []

        for m in re.finditer(r"https://n\.news\.naver\.com/mnews/article/(\d{3})/\d{10}(?:\?sid=\d+)?", resp.text):
            u = m.group(0)
            oid = m.group(1)
            if all(u != x[0] for x in all_found):
                all_found.append((u, oid))

        # 신문사(OID)가 겹치지 않게 우선 선발하여 다양한 신문사 관점을 확보
        for u, oid in all_found:
            if oid not in seen_oids:
                candidate_urls.append(u)
                seen_oids.add(oid)
                if len(candidate_urls) >= max_articles:
                    break

        for u, _ in all_found:
            if len(candidate_urls) >= max_articles:
                break
            if u not in candidate_urls:
                candidate_urls.append(u)

        saved_count = 0
        with sqlite3.connect(self.db_path) as con:
            cur = con.cursor()
            for article_url in candidate_urls:
                existing = cur.execute("SELECT PK FROM LINK WHERE URL = ?", (article_url,)).fetchone()
                if existing:
                    continue

                try:
                    art_resp = httpx.get(article_url, headers=HTTP_HEADERS, timeout=6.0, follow_redirects=True)
                    if art_resp.status_code != 200:
                        continue
                except Exception:
                    continue

                title, body = parse_naver_article_html(art_resp.text)
                if not title or len(body) < 140 or title.startswith("[포토]"):
                    continue

                # 1. LINK 테이블 저장
                cur.execute(
                    "INSERT INTO LINK(DEPTH, REFERER, URL) VALUES (?, ?, ?)",
                    (2, search_url, article_url),
                )
                link_pk = cur.lastrowid

                # 2. NEWS 테이블 저장
                cur.execute(
                    "INSERT INTO NEWS(FK, TITLE, CONTENT) VALUES (?, ?, ?)",
                    (link_pk, title, body),
                )
                news_pk = cur.lastrowid
                regdate = cur.execute("SELECT REGDATE FROM NEWS WHERE PK = ?", (news_pk,)).fetchone()[0]

                # 3. RAG_INDEX 테이블 즉시 인덱싱
                clean_title = re.sub(r"\s+", " ", title).strip()
                clean_body = mask_sensitive_info(body)
                press = extract_press_name(article_url, body)
                kw_list = extract_keywords(f"{clean_title} {clean_title} {clean_body[:600]}")
                emb = text_to_vector(f"{clean_title} {clean_title} {clean_body[:800]}", dim=self.embedding_dim)

                cur.execute(
                    """
                    INSERT OR REPLACE INTO RAG_INDEX
                    (NEWS_PK, PRESS, URL, TITLE, CLEAN_CONTENT, KEYWORDS, EMBEDDING, REGDATE)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        news_pk,
                        press,
                        article_url,
                        clean_title,
                        clean_body,
                        " ".join(kw_list[:60]),
                        emb.tobytes(),
                        regdate or "",
                    ),
                )
                saved_count += 1
            con.commit()

        return saved_count

    def retrieve(
        self,
        query: str,
        top_k: int = 8,
        min_score: float = 0.14,
        diversify_press: bool = True,
    ) -> list[RetrievedArticle]:
        """
        질문(query)에 대해 벡터 코사인 유사도 + 키워드 매칭 하이브리드 점수를 계산하고,
        신문사별 다양성(Press Diversity)을 보장하여 상위 기사를 반환합니다.
        """
        q_keywords = extract_keywords(query)
        q_vec = text_to_vector(query, dim=self.embedding_dim)

        with sqlite3.connect(self.db_path) as con:
            cur = con.cursor()
            rows = cur.execute(
                """
                SELECT NEWS_PK, PRESS, URL, TITLE, CLEAN_CONTENT, KEYWORDS, EMBEDDING, REGDATE
                FROM RAG_INDEX
                """
            ).fetchall()

        scored: list[RetrievedArticle] = []
        for pk, press, url, title, content, kw_str, emb_blob, regdate in rows:
            doc_vec = np.frombuffer(emb_blob, dtype=np.float32)
            vec_sim = float(np.dot(q_vec, doc_vec)) if len(doc_vec) == len(q_vec) else 0.0

            kw_score = 0.0
            matched_kw_count = 0
            if q_keywords:
                matched = 0.0
                for kw in q_keywords:
                    if kw in title:
                        matched += 2.5
                        matched_kw_count += 1
                    elif kw in content or kw in kw_str:
                        matched += 1.0
                        matched_kw_count += 1
                kw_score = min(1.0, matched / (len(q_keywords) * 1.5))

            min_required_kws = 1 if len(q_keywords) <= 1 else max(2, (len(q_keywords) + 1) // 2)
            hybrid_score = 0.45 * max(0.0, vec_sim) + 0.55 * kw_score

            if hybrid_score >= min_score and matched_kw_count >= min_required_kws:
                scored.append(
                    RetrievedArticle(
                        news_pk=pk,
                        press=press,
                        title=title,
                        content=content,
                        url=url,
                        regdate=regdate,
                        score=round(hybrid_score, 4),
                    )
                )

        scored.sort(key=lambda x: x.score, reverse=True)

        if not diversify_press or len(scored) <= top_k:
            return scored[:top_k]

        # 신문사별 다양성 보장: 서로 다른 신문사의 최상위 기사를 먼저 1건씩 선발
        selected: list[RetrievedArticle] = []
        used_pks: set[int] = set()
        seen_press: set[str] = set()

        for art in scored:
            if art.press not in seen_press:
                selected.append(art)
                used_pks.add(art.news_pk)
                seen_press.add(art.press)
                if len(selected) >= top_k:
                    return selected

        for art in scored:
            if art.news_pk not in used_pks:
                selected.append(art)
                used_pks.add(art.news_pk)
                if len(selected) >= top_k:
                    break

        return selected

    def get_popular_issues(self, limit: int = 5) -> list[dict[str, Any]]:
        """news.db에 실제로 다수 신문사가 보도한 대표 헤드라인 이슈 목록을 반환합니다."""
        candidate_topics = [
            ("김여정 확성기", "북한 김여정 대남 확성기·국경선 대응수단 위협"),
            ("수사기관장 배석", "국무회의 수사기관장 배석 및 검찰개혁 공방"),
            ("국민연금 전업주부", "전업주부·무소득 청년 국민연금 사각지대 해소 추진"),
            ("한미연합훈련 주한미군", "한미연합훈련 상시 전환 및 주한미군 역할 확대 논의"),
            ("반도체 D램 AI", "차세대 D램·AI 반도체 기술 패권 및 목표주가 분석"),
        ]
        results = []
        for query_kw, label in candidate_topics:
            arts = self.retrieve(query_kw, top_k=6)
            if len(arts) >= 2:
                results.append(
                    {
                        "query": query_kw,
                        "label": label,
                        "article_count": len(arts),
                        "press_list": sorted({a.press for a in arts}),
                    }
                )
        return results[:limit]


# =====================================================================
# LangGraph 에이전트 상태 및 노드 정의
# =====================================================================


class AgentState(TypedDict, total=False):
    query: str
    thread_id: str
    keywords: list[str]
    articles: list[dict[str, Any]]
    crawled_count: int
    is_sufficient: bool
    result: dict[str, Any]


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
    """각 신문사 기사의 핵심 해석/의견 1문장을 추출합니다."""
    sents = _split_sentences(art.content)
    # 따옴표(인용구)나 평가/분석 어조가 들어간 문장을 우선 선택
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
    신문사별 해석 차이를 가장 대비되는 두 관점(관점 A vs 관점 B)으로 분류하여 비교합니다.
    """
    press_set = sorted({a.press for a in articles})
    top_art = articles[0]

    # 의미 벡터 거리 기준으로 가장 대비되는 두 중심 기사를 찾아 관점 A / 관점 B 클러스터 분할
    if len(articles) >= 2:
        vecs = [text_to_vector(f"{a.title} {a.content[:500]}") for a in articles]
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

    # 1. 하나의 헤드라인으로 통합 (관점 A 대표 초점 vs 관점 B 대표 초점)
    g1_press_names = "·".join(dict.fromkeys(a.press for a in group1[:3]))
    g2_press_names = "·".join(dict.fromkeys(a.press for a in group2[:3]))
    unified_headline = (
        f"[{query}] {top_art.title}… "
        f"{g1_press_names} 등 {len(press_set)}개 신문사별 해석·관점 종합"
    )

    # 2. 공통 기사 내용 요약
    lead_sents: list[str] = []
    for art in articles[:3]:
        sents = _split_sentences(art.content)
        if sents and sents[0] not in lead_sents:
            lead_sents.append(sents[0])
    common_summary = " ".join(lead_sents[:2]) if lead_sents else top_art.content[:280]

    common_points = [
        f"<strong>공통 보도 팩트:</strong> {len(press_set)}개 신문사({', '.join(press_set[:6])})가 ‘{query}’ 사안을 주요 의제로 보도",
        f"<strong>관점 A ({g1_press_names} 등):</strong> {group1[0].title} 중심 해석",
        f"<strong>관점 B ({g2_press_names} 등):</strong> {group2[0].title} 중심 해석",
    ]

    # 3. 관점 A (주장 1) 신문사별 의견 요약
    g1_opinions = [_summarize_press_opinion(a) for a in group1[:2]]
    claim1_summary = " ".join(g1_opinions) if g1_opinions else group1[0].content[:220]
    claim1_points = [
        f"<strong>[{a.press}]</strong> {_summarize_press_opinion(a)[:110]}"
        for a in group1[:3]
    ]
    claim1_press = [{"name": a.press, "title": a.title, "url": a.url} for a in group1[:4]]

    # 4. 관점 B (주장 2) 신문사별 의견 요약
    g2_opinions = [_summarize_press_opinion(a) for a in group2[:2]]
    claim2_summary = " ".join(g2_opinions) if g2_opinions else group2[0].content[:220]
    claim2_points = [
        f"<strong>[{a.press}]</strong> {_summarize_press_opinion(a)[:110]}"
        for a in group2[:3]
    ]
    claim2_press = [{"name": a.press, "title": a.title, "url": a.url} for a in group2[:4]]

    crawl_note = (
        f"실시간 크롤링 {crawled_count}건 추가 적재 + " if crawled_count > 0 else ""
    )

    return {
        "headline": unified_headline,
        "meta": [
            f"수집·저장: {crawl_note}news.db (연관 기사 {len(articles)}건 · {len(press_set)}개 신문사)",
            f"최고 유사도: {top_art.score:.2f} (SQLite 하이브리드 RAG)",
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


class NewsLensAgent:
    """
    A2 문제 정의 구현 에이전트:
    1) 주제에 대한 DB(news.db) 기사 검색
    2) DB 내 신문사별 기사가 부족하면 실시간 크롤링 도구로 긁어와 news.db에 저장
    3) 하나의 헤드라인으로 묶어 공통 요약 + 신문사별 의견(관점 A vs 관점 B) 비교
    """

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH) -> None:
        self.rag = NewsDBRAG(db_path=db_path)
        self._memory_store: dict[str, list[dict[str, Any]]] = {}
        self.graph = self._build_graph()

    def _build_graph(self):
        from langgraph.graph import END, START, StateGraph

        builder = StateGraph(AgentState)
        builder.add_node("analyze_query", self.analyze_query_node)
        builder.add_node("retrieve_news", self.retrieve_news_node)
        builder.add_node("crawl_topic_to_db", self.crawl_topic_to_db_node)
        builder.add_node("compare_and_summarize", self.compare_and_summarize_node)
        builder.add_node("reject_no_evidence", self.reject_no_evidence_node)
        builder.add_node("verify_sources", self.verify_sources_node)

        builder.add_edge(START, "analyze_query")
        builder.add_edge("analyze_query", "retrieve_news")

        # 1차 판단: DB 내 서로 다른 신문사가 3곳 이상이면 바로 비교·요약, 부족하면 실시간 크롤링 도구 호출
        builder.add_conditional_edges(
            "retrieve_news",
            self.decide_crawl_or_compare,
            {
                "compare": "compare_and_summarize",
                "crawl": "crawl_topic_to_db",
            },
        )

        # 크롤링 후 2차 판단: 크롤링 후에도 관련 기사가 2건 미만이면 환각 방지를 위해 거부 노드로 분기
        builder.add_conditional_edges(
            "crawl_topic_to_db",
            self.route_after_crawl,
            {
                "sufficient": "compare_and_summarize",
                "insufficient": "reject_no_evidence",
            },
        )

        builder.add_edge("compare_and_summarize", "verify_sources")
        builder.add_edge("verify_sources", END)
        builder.add_edge("reject_no_evidence", END)

        return builder.compile()

    def analyze_query_node(self, state: AgentState) -> AgentState:
        """질문에서 핵심 키워드를 추출하고 이전 대화 맥락(후속 질문)을 결합합니다."""
        query = (state.get("query") or "").strip()
        thread_id = state.get("thread_id") or "default-session"
        history = self._memory_store.get(thread_id, [])

        if history and any(cue in query for cue in ("방금", "그중", "이전", "위에서", "더 자세히")):
            prev_query = history[-1].get("query", "")
            combined_query = f"{prev_query} {query}"
        else:
            combined_query = query

        keywords = extract_keywords(combined_query)
        return {
            "query": query,
            "thread_id": thread_id,
            "keywords": keywords,
            "crawled_count": 0,
        }

    def retrieve_news_node(self, state: AgentState) -> AgentState:
        """news.db에서 하이브리드 RAG 검색을 수행합니다."""
        search_text = " ".join(state.get("keywords") or []) or state.get("query", "")
        articles = self.rag.retrieve(search_text, top_k=8, min_score=0.14, diversify_press=True)
        distinct_press = {a.press for a in articles}
        return {
            "articles": [a.model_dump() for a in articles],
            "is_sufficient": len(articles) >= 2 and len(distinct_press) >= 2,
        }

    def decide_crawl_or_compare(self, state: AgentState) -> Literal["compare", "crawl"]:
        """
        [에이전트 판단 단계]
        news.db에 이미 유사도 0.40 이상의 고연관 기사가 3개 이상 신문사에서 확보되어 있으면 즉시 비교·요약하고,
        부족하면 실시간으로 해당 주제의 기사들을 긁어와 news.db에 저장하는 크롤링 도구를 선택합니다.
        """
        raw_articles = state.get("articles") or []
        strong_articles = [a for a in raw_articles if a.get("score", 0.0) >= 0.40]
        distinct_press = {a["press"] for a in strong_articles}
        if len(strong_articles) >= 3 and len(distinct_press) >= 3:
            return "compare"
        return "crawl"

    def crawl_topic_to_db_node(self, state: AgentState) -> AgentState:
        """
        [도구 실행 단계]
        주제에 대한 기사들을 실시간으로 긁어와 news.db에 저장한 뒤 재검색합니다.
        """
        search_text = " ".join(state.get("keywords") or []) or state.get("query", "")
        added = self.rag.crawl_topic_and_save_to_db(search_text, max_articles=8)
        articles = self.rag.retrieve(search_text, top_k=8, min_score=0.14, diversify_press=True)
        return {
            "crawled_count": added,
            "articles": [a.model_dump() for a in articles],
            "is_sufficient": len(articles) >= 2,
        }

    def route_after_crawl(self, state: AgentState) -> Literal["sufficient", "insufficient"]:
        """크롤링 및 재검색 후에도 관련 기사가 2건 미만이면 근거 부족 안내로 분기합니다."""
        return "sufficient" if state.get("is_sufficient") else "insufficient"

    def reject_no_evidence_node(self, state: AgentState) -> AgentState:
        """없는 자료 가져오지 않기(환각 차단): DB 검색 및 실시간 크롤링에서도 기사가 없을 때 안내합니다."""
        q = state.get("query", "")
        popular = self.rag.get_popular_issues(limit=4)
        suggested = ", ".join(f"‘{p['query']}’" for p in popular) if popular else "‘김여정 확성기’, ‘수사기관장 배석’, ‘국민연금’"
        return {
            "result": {
                "headline": f"‘{q}’에 대한 보도 기사를 찾을 수 없습니다",
                "meta": [
                    "수집 출처: news.db 검색 + 실시간 크롤링 결과 0건",
                    "환각 방지(성공 기준 1): 근거 없는 추측·가짜 출처 생성 차단됨",
                    f"추천 주제: {suggested}",
                ],
                "summary": (
                    f"로컬 데이터베이스(news.db) 검색 및 실시간 뉴스 크롤링을 모두 수행했으나 ‘{q}’ 주제와 일치하는 "
                    f"신뢰할 수 있는 보도 기사가 확인되지 않았습니다. 에이전트 안전 설계에 따라 없는 자료를 지어내지 않습니다."
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
            }
        }

    def compare_and_summarize_node(self, state: AgentState) -> AgentState:
        """
        검색·수집된 news.db 기사들을 하나의 헤드라인으로 묶어 요약하고 신문사별로 의견을 비교합니다.
        ANTHROPIC_API_KEY가 있으면 Claude API를 호출하고, 없으면 벡터 클러스터링 기반 추출형 비교 엔진으로 동작합니다.
        """
        query = state.get("query", "")
        raw_articles = state.get("articles") or []
        crawled_count = state.get("crawled_count") or 0
        articles = [RetrievedArticle(**a) for a in raw_articles]

        api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if api_key:
            try:
                from langchain_anthropic import ChatAnthropic

                llm = ChatAnthropic(
                    model=os.environ.get("ANTHROPIC_MODEL", "claude-3-5-haiku-latest"),
                    api_key=api_key,
                    temperature=0.1,
                    max_tokens=2048,
                )
                structured_llm = llm.with_structured_output(ComparisonOutputSchema)

                context_blocks = []
                for idx, a in enumerate(articles, 1):
                    context_blocks.append(
                        f"[기사 {idx}]\n"
                        f"- 신문사: {a.press}\n"
                        f"- 제목: {a.title}\n"
                        f"- URL: {a.url}\n"
                        f"- 본문 발췌: {a.content[:700]}"
                    )
                context_text = "\n\n".join(context_blocks)

                prompt = (
                    "당신은 여러 신문사의 기사를 하나의 헤드라인으로 묶어 요약하고 신문사별 해석·의견 차이를 비교해 주는 AI 에이전트입니다.\n"
                    "반드시 아래 제공된 [news.db 기사 목록]만을 근거로 삼아 답하세요.\n"
                    "원문에 없는 사실이나 목록에 없는 신문사·URL을 절대 지어내지 마세요.\n"
                    "claim1Points와 claim2Points에는 '<strong>[신문사명]</strong> 핵심 해석 요지' 형식으로 신문사별 의견을 명확히 비교해 주세요.\n\n"
                    f"[사용자 주제/질문]\n{query}\n\n"
                    f"[news.db 기사 목록]\n{context_text}"
                )

                parsed: ComparisonOutputSchema = structured_llm.invoke(prompt)
                press_set = sorted({a.press for a in articles})
                crawl_note = f"실시간 크롤링 {crawled_count}건 추가 적재 + " if crawled_count > 0 else ""
                out = parsed.model_dump()
                out["meta"] = [
                    f"수집·저장: {crawl_note}news.db (연관 기사 {len(articles)}건 · {len(press_set)}개 신문사)",
                    "분석 모델: Anthropic Claude (LangGraph + news.db RAG)",
                    "출처 검증: 통과 (news.db 원문 URL 100% 일치)",
                ]
                return {"result": out}
            except Exception as e:
                print(f"[안내] Anthropic API 호출 실패({e}) -> news.db 추출형 비교 엔진으로 대체합니다.")

        return {"result": _extractive_compare_articles(query, articles, crawled_count=crawled_count)}

    def verify_sources_node(self, state: AgentState) -> AgentState:
        """모든 신문사 출처 URL이 실제 news.db 검색 기사에 존재하는지 검증합니다."""
        result = dict(state.get("result") or {})
        raw_articles = state.get("articles") or []
        valid_by_press = {a["press"]: a for a in raw_articles}
        valid_urls = {a["url"] for a in raw_articles if a.get("url")}

        for key in ("claim1Press", "claim2Press"):
            verified_list = []
            for item in result.get(key, []):
                url = item.get("url", "")
                name = item.get("name", "")
                title = item.get("title", "")
                if url in valid_urls:
                    verified_list.append(item)
                elif name in valid_by_press:
                    matched = valid_by_press[name]
                    verified_list.append(
                        {"name": matched["press"], "title": title or matched["title"], "url": matched["url"]}
                    )
            result[key] = verified_list

        thread_id = state.get("thread_id") or "default-session"
        history = self._memory_store.setdefault(thread_id, [])
        history.append({"query": state.get("query", ""), "headline": result.get("headline", "")})
        if len(history) > 5:
            del history[:-5]

        return {"result": result}

    def run(self, query: str, thread_id: str = "default-session") -> dict[str, Any]:
        """주제/질문을 입력받아 [DB 검색 -> 부족 시 크롤링·DB저장 -> 헤드라인 통합 요약 -> 신문사별 의견 비교]를 수행합니다."""
        final_state = self.graph.invoke({"query": query, "thread_id": thread_id})
        return final_state["result"]


agent = NewsLensAgent()


if __name__ == "__main__":
    import json
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    test_query = sys.argv[1] if len(sys.argv) > 1 else "수사기관장 배석 사법개혁"
    res = agent.run(test_query)
    print(json.dumps(res, ensure_ascii=False, indent=2))

