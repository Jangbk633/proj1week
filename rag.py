"""
rag.py - news.db (SQLite + sqlite-vec 확장) 기반 고속 벡터 색인, 6대 카테고리 및 KMeans 이슈 군집화,
하이브리드 RAG 검색, 실시간 네이버 뉴스 크롤링 모듈
"""

from __future__ import annotations

from pathlib import Path
import re
import sqlite3
import time
from typing import Any
from urllib.parse import quote_plus

import httpx
import numpy as np
from pydantic import BaseModel
import sqlite_vec

from preprocessing import (
    HTTP_HEADERS,
    extract_keywords,
    extract_press_name,
    mask_sensitive_info,
    parse_naver_article_html,
    preprocess_title,
)
from vectorize import TITLE_VEC_DIM, serialize_vec, text_to_vector

ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_DB_PATH = ROOT_DIR / "news.db"

# title_vec 기반 6대 뉴스 분류 카테고리 정의 (대표 시드 어휘 및 설명)
NEWS_CATEGORY_SPECS: dict[str, dict[str, str]] = {
    "정치·국정": {
        "icon": "🏛️",
        "desc": "대통령실·국회·정당·국무회의·선거 및 국정 현안",
        "seed": (
            "대통령 이재명 윤석열 국회 여당 야당 더불어민주당 국민의힘 조국혁신당 대표 최고위원 "
            "원내대표 국무회의 총리 김민석 장동혁 한동훈 조국 법사위 본회의 특검 탄핵 정국 지지율 정당"
        ),
    },
    "경제·금융·산업": {
        "icon": "📈",
        "desc": "금리·환율·증시·부동산·연금·물가·기업 실적 및 재정",
        "seed": (
            "한국은행 기준금리 금리 동결 인하 환율 달러 물가 코스피 코스닥 증시 외국인 순매수 주식 "
            "국민연금 전업주부 가계부채 스트레스 DSR 부동산 대출 아파트 은행 금융 수출 무역 추경 예산 관세 유가"
        ),
    },
    "사회·사법·의료": {
        "icon": "⚖️",
        "desc": "검찰·경찰·법원·사법개혁·의대증원·노동·교육 및 사건사고",
        "seed": (
            "검찰 경찰 법원 수사기관장 배석 사법개혁 검찰개혁 조희대 대법원 재판 구속 수사 기소 "
            "의대 증원 의료개혁 전공의 병원 의사 환자 교육 대학 노조 파업 사건 사고 범죄 화재 안전"
        ),
    },
    "외교·안보·국제": {
        "icon": "🌐",
        "desc": "남북관계·주한미군·한미동맹·중동·미중갈등 및 국제 정세",
        "seed": (
            "북한 김여정 확성기 대남 대북 남북관계 국경선 주한미군 한미연합훈련 미8군 사령관 국방 안보 군대 "
            "미국 트럼프 중국 일본 러시아 중동 이란 이스라엘 호르무즈 해협 전쟁 유엔 외교 정상회담"
        ),
    },
    "IT·과학·반도체": {
        "icon": "🔬",
        "desc": "AI·반도체·D램·빅테크·원전·우주항공 및 첨단 과학기술",
        "seed": (
            "AI 인공지능 반도체 D램 HBM 메모리 파운드리 삼성전자 SK하이닉스 엔비디아 목표주가 빅테크 "
            "클라우드 로봇 자율주행 원전 원자력 재생에너지 배터리 전기차 우주 누리호 과학 기술 디지털 통신"
        ),
    },
    "생활·문화·스포츠": {
        "icon": "🎭",
        "desc": "날씨·건강·관광·문화예술·연예·방송 및 스포츠",
        "seed": (
            "날씨 기온 미세먼지 비 폭염 한파 건강 운동 음식 여행 관광 축제 공연 영화 드라마 연예 "
            "가수 배우 방송 스포츠 축구 야구 프로야구 올림픽 선수 감독 우승 경기"
        ),
    },
}


def connect_vec_db(db_path: str | Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """SQLite 연결을 생성하고 가상환경에 설치된 sqlite-vec 확장을 로드합니다."""
    con = sqlite3.connect(str(db_path))
    con.enable_load_extension(True)
    sqlite_vec.load(con)
    con.enable_load_extension(False)
    return con


class RetrievedArticle(BaseModel):
    news_pk: int
    press: str
    title: str
    clean_title: str = ""
    category: str = "시사·종합"
    cluster_label: str = ""
    content: str
    summary: str = ""
    url: str
    regdate: str
    score: float
    title_vec_sim: float = 0.0


class NewsDBRAG:
    """
    news.db (SQLite + sqlite-vec 확장) 기반:
    - NEWS.CLEAN_TITLE -> NEWS.title_vec 벡터화 및 VEC_NEWS_TITLE(vec0) 고속 색인
    - title_vec 기반 6대 뉴스 카테고리 분류 & KMeans(k=54) 이슈 클러스터 분류
    - 하이브리드 RAG 검색 및 실시간 주제 크롤러
    """

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH, embedding_dim: int = TITLE_VEC_DIM) -> None:
        self.db_path = Path(db_path)
        self.embedding_dim = embedding_dim
        self.category_vectors: dict[str, np.ndarray] = {
            cat: text_to_vector(spec["seed"], dim=self.embedding_dim)
            for cat, spec in NEWS_CATEGORY_SPECS.items()
        }
        self.cluster_centroids: dict[int, tuple[str, np.ndarray]] = {}
        self.last_index_elapsed_ms: float = 0.0
        self.sqlite_vec_version: str = "v0.1.9"
        self.ensure_rag_index()

    def _classify_category_from_vec(self, clean_title: str, vec: np.ndarray) -> tuple[str, dict[str, float]]:
        """title_vec 코사인 유사도 및 도메인 핵심어 매칭으로 6대 뉴스 카테고리를 정밀 분류합니다."""
        scores: dict[str, float] = {}
        title_text = clean_title or ""
        for cat, cat_vec in self.category_vectors.items():
            sim = max(0.0, float(np.dot(vec, cat_vec)))
            seed_tokens = NEWS_CATEGORY_SPECS[cat]["seed"].split()
            overlap_hits = sum(1 for tok in seed_tokens if len(tok) >= 2 and tok in title_text)
            scores[cat] = round(sim * 0.55 + min(0.65, overlap_hits * 0.18), 4)

        best_cat = max(scores.items(), key=lambda x: x[1])[0]
        if scores[best_cat] < 0.04:
            best_cat = "사회·사법·의료"
        return best_cat, scores

    def ensure_rag_index(self, force_rebuild: bool = False) -> int:
        """
        1) NEWS 테이블에 CLEAN_TITLE, title_vec, CATEGORY, CLUSTER_ID, CLUSTER_LABEL 칼럼을 생성·업데이트하고
        2) sqlite-vec의 vec0 가상 테이블(VEC_NEWS_TITLE) 및 RAG_INDEX를 고속 구축합니다.
        """
        if not self.db_path.exists():
            raise FileNotFoundError(f"데이터베이스 파일을 찾을 수 없습니다: {self.db_path}")

        t0 = time.perf_counter()
        with connect_vec_db(self.db_path) as con:
            cur = con.cursor()
            self.sqlite_vec_version = cur.execute("SELECT vec_version()").fetchone()[0]

            # 1. NEWS 테이블에 CLEAN_TITLE, title_vec, CATEGORY, CLUSTER_ID, CLUSTER_LABEL, SUMMARY 칼럼 보장
            news_cols = {c[1] for c in cur.execute("PRAGMA table_info(NEWS)").fetchall()}
            if "CLEAN_TITLE" not in news_cols:
                cur.execute("ALTER TABLE NEWS ADD COLUMN CLEAN_TITLE TEXT")
            if "title_vec" not in news_cols:
                cur.execute("ALTER TABLE NEWS ADD COLUMN title_vec BLOB")
            if "CATEGORY" not in news_cols:
                cur.execute("ALTER TABLE NEWS ADD COLUMN CATEGORY TEXT")
            if "CLUSTER_ID" not in news_cols:
                cur.execute("ALTER TABLE NEWS ADD COLUMN CLUSTER_ID INTEGER DEFAULT 0")
            if "CLUSTER_LABEL" not in news_cols:
                cur.execute("ALTER TABLE NEWS ADD COLUMN CLUSTER_LABEL TEXT DEFAULT ''")
            if "SUMMARY" not in news_cols:
                cur.execute("ALTER TABLE NEWS ADD COLUMN SUMMARY TEXT DEFAULT ''")
            if "SUMMARY_MODEL" not in news_cols:
                cur.execute("ALTER TABLE NEWS ADD COLUMN SUMMARY_MODEL TEXT DEFAULT ''")
            if "SUMMARY_AT" not in news_cols:
                cur.execute("ALTER TABLE NEWS ADD COLUMN SUMMARY_AT TEXT DEFAULT ''")

            if force_rebuild:
                cur.execute("DROP TABLE IF EXISTS RAG_INDEX")
                cur.execute("DROP TABLE IF EXISTS VEC_NEWS_TITLE")

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS RAG_INDEX (
                    NEWS_PK INTEGER PRIMARY KEY,
                    PRESS TEXT NOT NULL,
                    URL TEXT NOT NULL,
                    TITLE TEXT NOT NULL,
                    CLEAN_TITLE TEXT NOT NULL DEFAULT '',
                    CATEGORY TEXT NOT NULL DEFAULT '시사·종합',
                    CLUSTER_ID INTEGER NOT NULL DEFAULT 0,
                    CLUSTER_LABEL TEXT NOT NULL DEFAULT '',
                    CLEAN_CONTENT TEXT NOT NULL,
                    SUMMARY TEXT NOT NULL DEFAULT '',
                    KEYWORDS TEXT NOT NULL,
                    EMBEDDING BLOB NOT NULL,
                    title_vec BLOB,
                    REGDATE TEXT
                )
                """
            )
            rag_cols = {c[1] for c in cur.execute("PRAGMA table_info(RAG_INDEX)").fetchall()}
            for col_name, col_def in [
                ("CLEAN_TITLE", "TEXT NOT NULL DEFAULT ''"),
                ("CATEGORY", "TEXT NOT NULL DEFAULT '시사·종합'"),
                ("CLUSTER_ID", "INTEGER NOT NULL DEFAULT 0"),
                ("CLUSTER_LABEL", "TEXT NOT NULL DEFAULT ''"),
                ("SUMMARY", "TEXT NOT NULL DEFAULT ''"),
                ("title_vec", "BLOB"),
            ]:
                if col_name not in rag_cols:
                    cur.execute(f"ALTER TABLE RAG_INDEX ADD COLUMN {col_name} {col_def}")

            # sqlite-vec 가상 테이블(vec0) 생성: clean_title을 벡터화한 title_vec 코사인 거리 인덱스
            cur.execute(
                f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS VEC_NEWS_TITLE USING vec0(
                    news_pk INTEGER PRIMARY KEY,
                    title_vec float[{self.embedding_dim}] distance_metric=cosine
                )
                """
            )

            # 이미 모든 NEWS 행에 title_vec과 CLUSTER_LABEL이 채워져 있고 VEC_NEWS_TITLE이 준비된 경우 빠른 패스
            missing_vec_cnt = cur.execute(
                "SELECT COUNT(*) FROM NEWS WHERE title_vec IS NULL OR CATEGORY IS NULL OR CLUSTER_LABEL IS NULL OR CLUSTER_LABEL = ''"
            ).fetchone()[0]
            vec_table_cnt = cur.execute("SELECT COUNT(*) FROM VEC_NEWS_TITLE").fetchone()[0]
            rag_missing_vec = cur.execute(
                "SELECT COUNT(*) FROM RAG_INDEX WHERE title_vec IS NULL OR CLEAN_TITLE = ''"
            ).fetchone()[0]

            if not force_rebuild and missing_vec_cnt == 0 and vec_table_cnt > 0 and rag_missing_vec == 0:
                self._load_cluster_centroids_from_db(cur)
                self.last_index_elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
                return vec_table_cnt

            # 2. 전체 NEWS 로드 후 CLEAN_TITLE 기반으로 title_vec 고속 벡터화
            rows = cur.execute(
                """
                SELECT N.PK, COALESCE(L.URL, ''), N.TITLE, COALESCE(N.CLEAN_TITLE, ''), N.CONTENT, COALESCE(N.REGDATE, '')
                FROM NEWS N
                LEFT JOIN LINK L ON N.FK = L.PK
                ORDER BY N.PK ASC
                """
            ).fetchall()

            pks: list[int] = []
            clean_titles: list[str] = []
            vecs_list: list[np.ndarray] = []
            vec_blobs: list[bytes] = []
            categories: list[str] = []

            for pk, _url, raw_title, db_clean_title, _content, _regdate in rows:
                c_title = db_clean_title.strip() if db_clean_title and db_clean_title.strip() else preprocess_title(raw_title or "")
                if not c_title:
                    c_title = re.sub(r"\s+", " ", (raw_title or "")).strip()
                t_vec = text_to_vector(c_title, dim=self.embedding_dim)
                t_blob = serialize_vec(t_vec)
                cat, _ = self._classify_category_from_vec(c_title, t_vec)

                pks.append(pk)
                clean_titles.append(c_title)
                vecs_list.append(t_vec)
                vec_blobs.append(t_blob)
                categories.append(cat)

            # 3. preprocessing.ipynb 기준 C (k=54) 방식으로 CLEAN_TITLE / title_vec 기반 이슈 클러스터 분류
            cluster_ids, cluster_labels = self._compute_issue_clusters(clean_titles, vecs_list, n_clusters=54)

            # 4. NEWS 테이블에 CLEAN_TITLE, title_vec, CATEGORY, CLUSTER_ID, CLUSTER_LABEL 일괄 업데이트
            news_updates = [
                (clean_titles[i], vec_blobs[i], categories[i], int(cluster_ids[i]), cluster_labels[i], pks[i])
                for i in range(len(pks))
            ]
            cur.executemany(
                """
                UPDATE NEWS
                SET CLEAN_TITLE = ?, title_vec = ?, CATEGORY = ?, CLUSTER_ID = ?, CLUSTER_LABEL = ?
                WHERE PK = ?
                """,
                news_updates,
            )

            # 5. sqlite-vec 가상 테이블(VEC_NEWS_TITLE) 일괄 색인
            cur.execute("DELETE FROM VEC_NEWS_TITLE")
            cur.executemany(
                "INSERT INTO VEC_NEWS_TITLE(news_pk, title_vec) VALUES (?, ?)",
                [(pks[i], vec_blobs[i]) for i in range(len(pks))],
            )

            # 6. RAG_INDEX 테이블 동기화
            seen_titles: set[str] = set()
            rag_to_insert: list[tuple[Any, ...]] = []
            for idx, (pk, url, raw_title, _db_clean, content, regdate) in enumerate(rows):
                norm_title = re.sub(r"\s+", " ", (raw_title or "")).strip()
                clean_body = mask_sensitive_info(content or "")
                if len(clean_body) < 120 or norm_title.startswith("[포토]"):
                    continue
                if norm_title in seen_titles:
                    continue
                seen_titles.add(norm_title)

                press = extract_press_name(url, content)
                c_title = clean_titles[idx]
                kw_list = extract_keywords(f"{c_title} {c_title} {clean_body[:400]}")

                rag_to_insert.append(
                    (
                        pk,
                        press,
                        url,
                        norm_title,
                        c_title,
                        categories[idx],
                        int(cluster_ids[idx]),
                        cluster_labels[idx],
                        clean_body,
                        " ".join(kw_list[:60]),
                        vec_blobs[idx],
                        vec_blobs[idx],
                        regdate,
                    )
                )

            cur.executemany(
                """
                INSERT OR REPLACE INTO RAG_INDEX
                (NEWS_PK, PRESS, URL, TITLE, CLEAN_TITLE, CATEGORY, CLUSTER_ID, CLUSTER_LABEL,
                 CLEAN_CONTENT, KEYWORDS, EMBEDDING, title_vec, REGDATE)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rag_to_insert,
            )
            con.commit()
            self._load_cluster_centroids_from_db(cur)

        self.last_index_elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        return len(pks)

    def _compute_issue_clusters(
        self,
        clean_titles: list[str],
        vecs_list: list[np.ndarray],
        n_clusters: int = 20,
    ) -> tuple[list[int], list[str]]:
        """
        CLEAN_TITLE 및 title_vec을 기반으로 뉴스 기사들을 k개 이슈 클러스터로 자동 분류하고
        각 클러스터의 대표 키워드 3개로 라벨(예: 'C03: 대통령, 이재명, 국무회의')을 부여합니다.
        """
        if not clean_titles:
            return [], []

        k = min(n_clusters, max(2, len(clean_titles) // 5))
        try:
            from sklearn.cluster import KMeans
            from sklearn.feature_extraction.text import TfidfVectorizer

            vectorizer = TfidfVectorizer(
                token_pattern=r"(?u)\b[가-힣a-zA-Z][가-힣a-zA-Z0-9]+\b",
                max_features=3000,
            )
            X_tfidf = vectorizer.fit_transform(clean_titles)
            feature_names = vectorizer.get_feature_names_out()

            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            c_ids = km.fit_predict(X_tfidf).tolist()

            label_map: dict[int, str] = {}
            for i in range(k):
                center = km.cluster_centers_[i]
                top_idx = center.argsort()[::-1][:3]
                max_weight = float(center[top_idx[0]]) if len(top_idx) > 0 else 0.0
                kws = [str(feature_names[j]) for j in top_idx if center[j] > 0.03]
                if kws and max_weight >= 0.08:
                    label_map[i] = f"C{i:02d}: {', '.join(kws)}"
                else:
                    label_map[i] = f"C{i:02d}: 종합·일반 시사"

            c_labels = [label_map[cid] for cid in c_ids]
            return c_ids, c_labels
        except Exception:
            mat = np.vstack(vecs_list)
            step = max(1, len(vecs_list) // k)
            centers = mat[::step][:k]
            sims = mat @ centers.T
            c_ids = np.argmax(sims, axis=1).tolist()
            c_labels = [f"C{cid}: 주요 이슈" for cid in c_ids]
            return c_ids, c_labels

    def _load_cluster_centroids_from_db(self, cur: sqlite3.Cursor) -> None:
        """DB에 저장된 CLUSTER_ID별 title_vec 평균 센트로이드를 메모리에 로드합니다."""
        rows = cur.execute(
            "SELECT CLUSTER_ID, CLUSTER_LABEL, title_vec FROM NEWS WHERE title_vec IS NOT NULL"
        ).fetchall()
        grouped: dict[int, tuple[str, list[np.ndarray]]] = {}
        for cid, clabel, blob in rows:
            if blob is None:
                continue
            v = np.frombuffer(blob, dtype=np.float32)
            if len(v) != self.embedding_dim:
                continue
            if cid not in grouped:
                grouped[cid] = (clabel or f"C{cid}", [])
            grouped[cid][1].append(v)

        self.cluster_centroids.clear()
        for cid, (clabel, vlist) in grouped.items():
            mean_v = np.mean(np.vstack(vlist), axis=0).astype(np.float32)
            norm = float(np.linalg.norm(mean_v))
            if norm > 0:
                mean_v /= norm
            self.cluster_centroids[int(cid)] = (clabel, mean_v)

    def crawl_topic_and_save_to_db(self, query: str, max_articles: int = 8) -> int:
        """
        [에이전트 도구: 주제별 뉴스 크롤링 & news.db + sqlite-vec 자동 색인]
        하나의 주제(query)에 대한 기사들을 서로 다른 신문사별로 긁어와
        LINK, NEWS(CLEAN_TITLE, title_vec, CATEGORY, CLUSTER_LABEL), VEC_NEWS_TITLE, RAG_INDEX에 저장합니다.
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

        q_keywords = extract_keywords(query)
        min_required_kws = 1 if len(q_keywords) <= 1 else max(2, (len(q_keywords) + 1) // 2)

        saved_count = 0
        with connect_vec_db(self.db_path) as con:
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

                matched_kws = sum(1 for kw in q_keywords if kw in title or kw in body)
                has_subject = any(kw in title or kw in body for kw in q_keywords[:2]) if q_keywords else True
                if q_keywords and (matched_kws < min_required_kws or not has_subject):
                    continue

                norm_title = re.sub(r"\s+", " ", title).strip()
                c_title = preprocess_title(norm_title) or norm_title
                t_vec = text_to_vector(c_title, dim=self.embedding_dim)
                t_blob = serialize_vec(t_vec)
                cat, _ = self._classify_category_from_vec(c_title, t_vec)
                best_cid, best_clabel = self._match_nearest_cluster(t_vec)

                # 1. LINK 테이블 저장
                cur.execute(
                    "INSERT INTO LINK(DEPTH, REFERER, URL) VALUES (?, ?, ?)",
                    (2, search_url, article_url),
                )
                link_pk = cur.lastrowid

                # 2. NEWS 테이블 저장
                cur.execute(
                    """
                    INSERT INTO NEWS(FK, TITLE, CONTENT, CLEAN_TITLE, title_vec, CATEGORY, CLUSTER_ID, CLUSTER_LABEL)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (link_pk, norm_title, body, c_title, t_blob, cat, best_cid, best_clabel),
                )
                news_pk = cur.lastrowid
                regdate = cur.execute("SELECT REGDATE FROM NEWS WHERE PK = ?", (news_pk,)).fetchone()[0]

                # 3. sqlite-vec VEC_NEWS_TITLE 가상 테이블 즉시 인덱싱
                cur.execute(
                    "INSERT OR REPLACE INTO VEC_NEWS_TITLE(news_pk, title_vec) VALUES (?, ?)",
                    (news_pk, t_blob),
                )

                # 4. RAG_INDEX 테이블 즉시 인덱싱
                clean_body = mask_sensitive_info(body)
                press = extract_press_name(article_url, body)
                kw_list = extract_keywords(f"{c_title} {c_title} {clean_body[:400]}")

                cur.execute(
                    """
                    INSERT OR REPLACE INTO RAG_INDEX
                    (NEWS_PK, PRESS, URL, TITLE, CLEAN_TITLE, CATEGORY, CLUSTER_ID, CLUSTER_LABEL,
                     CLEAN_CONTENT, KEYWORDS, EMBEDDING, title_vec, REGDATE)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        news_pk,
                        press,
                        article_url,
                        norm_title,
                        c_title,
                        cat,
                        best_cid,
                        best_clabel,
                        clean_body,
                        " ".join(kw_list[:60]),
                        t_blob,
                        t_blob,
                        regdate or "",
                    ),
                )
                saved_count += 1
            con.commit()

        return saved_count

    def _match_nearest_cluster(self, vec: np.ndarray) -> tuple[int, str]:
        """주어진 title_vec과 가장 가까운 이슈 클러스터 ID 및 라벨을 반환합니다."""
        if not self.cluster_centroids:
            return 0, "C00: 주요 시사"
        best_cid = 0
        best_label = "C00: 주요 시사"
        best_sim = -1.0
        for cid, (clabel, cvec) in self.cluster_centroids.items():
            if "종합·일반" in clabel:
                continue
            sim = float(np.dot(vec, cvec))
            if sim > best_sim:
                best_sim = sim
                best_cid = cid
                best_label = clabel
        return best_cid, best_label

    def retrieve(
        self,
        query: str,
        top_k: int = 8,
        min_score: float = 0.14,
        diversify_press: bool = True,
    ) -> list[RetrievedArticle]:
        """
        sqlite-vec 확장(vec_distance_cosine(title_vec, ?))을 활용하여
        DB 엔진 내부에서 고속으로 title_vec 코사인 유사도를 계산하고,
        키워드 매칭 및 신문사별 다양성(Press Diversity)을 결합해 상위 기사를 반환합니다.
        """
        q_keywords = extract_keywords(query)
        q_vec = text_to_vector(query, dim=self.embedding_dim)
        q_blob = serialize_vec(q_vec)

        with connect_vec_db(self.db_path) as con:
            cur = con.cursor()
            rows = cur.execute(
                """
                SELECT
                    R.NEWS_PK,
                    R.PRESS,
                    R.URL,
                    R.TITLE,
                    R.CLEAN_TITLE,
                    R.CATEGORY,
                    R.CLUSTER_LABEL,
                    R.CLEAN_CONTENT,
                    COALESCE(N.SUMMARY, R.SUMMARY, ''),
                    R.KEYWORDS,
                    R.REGDATE,
                    (1.0 - vec_distance_cosine(R.title_vec, ?)) AS vec_sim
                FROM RAG_INDEX R
                LEFT JOIN NEWS N ON N.PK = R.NEWS_PK
                WHERE R.title_vec IS NOT NULL
                ORDER BY vec_sim DESC
                """,
                (q_blob,),
            ).fetchall()

        scored: list[RetrievedArticle] = []
        for pk, press, url, title, c_title, category, cluster_label, content, summary, kw_str, regdate, vec_sim_raw in rows:
            vec_sim = float(vec_sim_raw or 0.0)

            kw_score = 0.0
            matched_kw_count = 0
            if q_keywords:
                matched = 0.0
                for kw in q_keywords:
                    if kw in c_title or kw in title:
                        matched += 2.5
                        matched_kw_count += 1
                    elif kw in content or kw in kw_str:
                        matched += 1.0
                        matched_kw_count += 1
                kw_score = min(1.0, matched / (len(q_keywords) * 1.5))

            min_required_kws = 1 if len(q_keywords) <= 1 else max(2, (len(q_keywords) + 1) // 2)
            has_subject_kw = (
                any(kw in c_title or kw in title or kw in content or kw in kw_str for kw in q_keywords[:2])
                if q_keywords
                else True
            )
            hybrid_score = 0.45 * max(0.0, vec_sim) + 0.55 * kw_score

            if hybrid_score >= min_score and matched_kw_count >= min_required_kws and has_subject_kw:
                scored.append(
                    RetrievedArticle(
                        news_pk=pk,
                        press=press,
                        title=title,
                        clean_title=c_title,
                        category=category,
                        cluster_label=cluster_label,
                        content=content,
                        summary=summary or "",
                        url=url,
                        regdate=regdate,
                        score=round(hybrid_score, 4),
                        title_vec_sim=round(max(0.0, vec_sim), 4),
                    )
                )

        scored.sort(key=lambda x: x.score, reverse=True)

        if not diversify_press or len(scored) <= top_k:
            return scored[:top_k]

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

    def classify_news_by_title_vec(
        self,
        title: str,
        content: str = "",
        top_k: int = 5,
    ) -> dict[str, Any]:
        """
        입력된 뉴스 제목(또는 제목+본문)을 CLEAN_TITLE로 정제하고 title_vec으로 벡터화한 뒤,
        sqlite-vec 확장(VEC_NEWS_TITLE KNN 검색 & 센트로이드 코사인 거리)을 이용해
        1) 6대 뉴스 카테고리 분류
        2) KMeans 이슈 클러스터 분류
        3) news.db 내 가장 유사한 뉴스 Top-K 검색을 초고속으로 수행합니다.
        """
        t0 = time.perf_counter()
        clean_t = preprocess_title(title) or re.sub(r"\s+", " ", title).strip()
        t_vec = text_to_vector(f"{clean_t} {preprocess_title(content[:200])}".strip(), dim=self.embedding_dim)
        t_blob = serialize_vec(t_vec)

        best_cat, cat_scores = self._classify_category_from_vec(clean_t, t_vec)
        best_cid, best_clabel = self._match_nearest_cluster(t_vec)

        with connect_vec_db(self.db_path) as con:
            cur = con.cursor()
            knn_rows = cur.execute(
                """
                SELECT
                    V.news_pk,
                    V.distance,
                    N.TITLE,
                    N.CLEAN_TITLE,
                    N.CATEGORY,
                    N.CLUSTER_LABEL,
                    COALESCE(L.URL, ''),
                    N.CONTENT
                FROM VEC_NEWS_TITLE V
                JOIN NEWS N ON N.PK = V.news_pk
                LEFT JOIN LINK L ON N.FK = L.PK
                WHERE V.title_vec MATCH ? AND k = ?
                """,
                (t_blob, max(top_k, 8)),
            ).fetchall()

        similar_articles: list[dict[str, Any]] = []
        neighbor_cats: list[str] = []
        for pk, dist, n_title, n_clean, n_cat, n_cluster, n_url, n_body in knn_rows:
            sim_pct = round(max(0.0, 1.0 - float(dist or 0.0)) * 100, 1)
            press_name = extract_press_name(n_url, n_body or "")
            if sim_pct >= 25.0 and n_cat:
                neighbor_cats.append(n_cat)
            if len(similar_articles) < top_k:
                similar_articles.append(
                    {
                        "news_pk": pk,
                        "press": press_name,
                        "title": n_title,
                        "clean_title": n_clean,
                        "category": n_cat or best_cat,
                        "cluster_label": n_cluster or best_clabel,
                        "similarity": sim_pct,
                        "url": n_url or "#source",
                    }
                )

        if neighbor_cats and similar_articles and similar_articles[0]["similarity"] >= 55.0:
            from collections import Counter

            top_knn_cat = Counter(neighbor_cats[:3]).most_common(1)[0][0]
            if cat_scores.get(best_cat, 0.0) < 0.12:
                best_cat = top_knn_cat
            if similar_articles[0].get("cluster_label") and "종합·일반" not in similar_articles[0]["cluster_label"]:
                best_clabel = similar_articles[0]["cluster_label"]

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        cat_ranking = [
            {
                "category": c,
                "icon": NEWS_CATEGORY_SPECS[c]["icon"],
                "score": round(min(99.0, max(10.0, s * 160 + (20.0 if c == best_cat else 0.0))), 1),
            }
            for c, s in sorted(cat_scores.items(), key=lambda x: x[1], reverse=True)
        ]

        return {
            "engine": f"sqlite-vec ({self.sqlite_vec_version}) · title_vec[{self.embedding_dim}]",
            "input_title": title,
            "clean_title": clean_t,
            "category": best_cat,
            "category_icon": NEWS_CATEGORY_SPECS.get(best_cat, {}).get("icon", "📰"),
            "category_desc": NEWS_CATEGORY_SPECS.get(best_cat, {}).get("desc", ""),
            "cluster_id": best_cid,
            "cluster_label": best_clabel,
            "search_time_ms": elapsed_ms,
            "category_ranking": cat_ranking,
            "similar_articles": similar_articles,
        }

    def get_classification_overview(
        self,
        category: str | None = None,
        cluster_id: int | None = None,
        limit: int = 18,
    ) -> dict[str, Any]:
        """
        news.db의 title_vec 기반 전체 뉴스 분류 현황(카테고리별 분포, 상위 이슈 클러스터, 분류된 기사 목록)을 반환합니다.
        """
        t0 = time.perf_counter()
        with connect_vec_db(self.db_path) as con:
            cur = con.cursor()
            total_news = cur.execute("SELECT COUNT(*) FROM NEWS WHERE title_vec IS NOT NULL").fetchone()[0]
            vec_index_cnt = cur.execute("SELECT COUNT(*) FROM VEC_NEWS_TITLE").fetchone()[0]

            # 1. 6대 카테고리별 기사 건수 집계
            cat_rows = cur.execute(
                """
                SELECT CATEGORY, COUNT(*) AS cnt
                FROM NEWS
                WHERE title_vec IS NOT NULL
                GROUP BY CATEGORY
                ORDER BY cnt DESC
                """
            ).fetchall()

            categories_summary = []
            for cat_name, cnt in cat_rows:
                spec = NEWS_CATEGORY_SPECS.get(cat_name, {"icon": "📰", "desc": ""})
                categories_summary.append(
                    {
                        "category": cat_name,
                        "icon": spec["icon"],
                        "desc": spec["desc"],
                        "count": cnt,
                        "ratio": round(cnt * 100.0 / max(1, total_news), 1),
                    }
                )

            # 2. 상위 이슈 클러스터 집계
            cluster_rows = cur.execute(
                """
                SELECT
                    N.CLUSTER_ID,
                    N.CLUSTER_LABEL,
                    N.CATEGORY,
                    COUNT(*) AS cnt,
                    MAX(N.TITLE) AS sample_title,
                    MAX(N.CLEAN_TITLE) AS sample_clean_title
                FROM NEWS N
                WHERE N.title_vec IS NOT NULL
                  AND N.TITLE NOT LIKE '[포토]%'
                  AND N.CLUSTER_LABEL NOT LIKE '%종합·일반%'
                GROUP BY N.CLUSTER_ID, N.CLUSTER_LABEL
                ORDER BY cnt DESC
                LIMIT 12
                """
            ).fetchall()

            clusters_summary = []
            for cid, clabel, ccat, cnt, stitle, sclean in cluster_rows:
                kw_part = clabel.split(":", 1)[1].strip() if ":" in clabel else clabel
                search_q = kw_part.replace(",", "").strip() or sclean
                clusters_summary.append(
                    {
                        "cluster_id": cid,
                        "cluster_label": clabel,
                        "category": ccat,
                        "count": cnt,
                        "sample_title": stitle,
                        "search_query": search_q,
                    }
                )

            # 3. 선택된 카테고리 또는 클러스터의 기사 목록 조회
            where_clauses = ["N.title_vec IS NOT NULL", "N.TITLE NOT LIKE '[포토]%'"]
            params: list[Any] = []
            if category and category != "전체":
                where_clauses.append("N.CATEGORY = ?")
                params.append(category)
            if cluster_id is not None:
                where_clauses.append("N.CLUSTER_ID = ?")
                params.append(cluster_id)

            where_sql = " AND ".join(where_clauses)
            art_rows = cur.execute(
                f"""
                SELECT
                    N.PK,
                    N.TITLE,
                    N.CLEAN_TITLE,
                    N.CATEGORY,
                    N.CLUSTER_ID,
                    N.CLUSTER_LABEL,
                    COALESCE(L.URL, ''),
                    N.CONTENT,
                    COALESCE(N.REGDATE, '')
                FROM NEWS N
                LEFT JOIN LINK L ON N.FK = L.PK
                WHERE {where_sql}
                ORDER BY N.PK DESC
                LIMIT ?
                """,
                (*params, limit),
            ).fetchall()

            articles_list = []
            for pk, title, c_title, cat_name, cid, clabel, url, body, regdate in art_rows:
                articles_list.append(
                    {
                        "news_pk": pk,
                        "press": extract_press_name(url, body or ""),
                        "title": title,
                        "clean_title": c_title,
                        "category": cat_name,
                        "category_icon": NEWS_CATEGORY_SPECS.get(cat_name, {}).get("icon", "📰"),
                        "cluster_id": cid,
                        "cluster_label": clabel,
                        "url": url or "#source",
                        "regdate": regdate,
                    }
                )

        query_ms = round((time.perf_counter() - t0) * 1000, 2)
        return {
            "sqlite_vec_version": self.sqlite_vec_version,
            "vector_column": "NEWS.title_vec (float32[512])",
            "virtual_table": "VEC_NEWS_TITLE USING vec0(distance_metric=cosine)",
            "total_vectorized_news": total_news,
            "vec_index_count": vec_index_cnt,
            "index_build_ms": self.last_index_elapsed_ms,
            "query_elapsed_ms": query_ms,
            "categories": categories_summary,
            "clusters": clusters_summary,
            "articles": articles_list,
        }

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


__all__ = [
    "ROOT_DIR",
    "DEFAULT_DB_PATH",
    "NEWS_CATEGORY_SPECS",
    "connect_vec_db",
    "RetrievedArticle",
    "NewsDBRAG",
]
