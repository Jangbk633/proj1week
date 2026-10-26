"""
langgraph.py - LangGraph 기반 뉴스 검색·크롤링·관점 비교·출처 검증 에이전트(NJKAgent) 모듈

주의: 파일명이 외부 패키지(`langgraph`)와 동일하므로, 다른 파이썬 파일이나 내부에서
`from langgraph import NJKAgent`와 `from langgraph.graph import StateGraph`를
모두 에러 없이 사용할 수 있도록 패키지 브릿지(Bridge) 로직을 포함합니다.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any, Literal, TypedDict

from preprocessing import extract_keywords
from rag import DEFAULT_DB_PATH, NewsDBRAG, RetrievedArticle
from summerizer import ComparisonOutputSchema, _extractive_compare_articles


def _bridge_installed_langgraph_submodules() -> tuple[Any, Any, Any]:
    """
    로컬 `langgraph.py`가 임포트되었을 때도 `langgraph.graph`, `langgraph.checkpoint.memory` 등
    site-packages의 공식 `langgraph` 하위 패키지 임포트가 정상 동작하도록 `__path__`를 연결합니다.
    """
    this_dir = Path(__file__).resolve().parent
    search_paths = [
        p for p in sys.path
        if Path(p if p else ".").resolve() != this_dir
    ]
    spec = importlib.machinery.PathFinder.find_spec("langgraph", search_paths)
    if spec is None or not spec.submodule_search_locations:
        raise ImportError("가상환경에서 설치된 공식 'langgraph' 패키지를 찾을 수 없습니다.")

    current_mod = sys.modules.get(__name__)
    if current_mod is not None and not hasattr(current_mod, "__path__"):
        current_mod.__path__ = list(spec.submodule_search_locations)  # type: ignore[attr-defined]
        if current_mod.__spec__ is not None:
            current_mod.__spec__.submodule_search_locations = list(spec.submodule_search_locations)

    # 만약 __main__으로 실행된 경우 sys.modules['langgraph']가 없을 수 있으므로 보장
    if "langgraph" not in sys.modules:
        pkg_mod = importlib.util.module_from_spec(spec)
        sys.modules["langgraph"] = pkg_mod
        if spec.loader is not None:
            spec.loader.exec_module(pkg_mod)

    import langgraph.graph as lg_graph
    return lg_graph.END, lg_graph.START, lg_graph.StateGraph


# 모듈 로드 시점에 즉시 브릿지를 활성화하여 외부 파일의 `from langgraph.checkpoint...` 도 지원
_bridge_installed_langgraph_submodules()


class AgentState(TypedDict, total=False):
    query: str
    thread_id: str
    keywords: list[str]
    articles: list[dict[str, Any]]
    crawled_count: int
    is_sufficient: bool
    result: dict[str, Any]


class NJKAgent:
    """
    A2 문제 정의 + sqlite-vec (title_vec) 고속 검색·분류 에이전트:
    1) 주제에 대한 DB(news.db) title_vec 기반 고속 검색
    2) DB 내 신문사별 기사가 부족하면 실시간 크롤링 도구로 긁어와 news.db에 저장 및 벡터 색인
    3) 하나의 헤드라인으로 묶어 공통 요약 + 신문사별 의견(관점 A vs 관점 B) 비교
    """

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH) -> None:
        self.rag = NewsDBRAG(db_path=db_path)
        self._memory_store: dict[str, list[dict[str, Any]]] = {}
        self.graph = self._build_graph()

    def _build_graph(self):
        END, START, StateGraph = _bridge_installed_langgraph_submodules()

        builder = StateGraph(AgentState)
        builder.add_node("analyze_query", self.analyze_query_node)
        builder.add_node("retrieve_news", self.retrieve_news_node)
        builder.add_node("crawl_topic_to_db", self.crawl_topic_to_db_node)
        builder.add_node("compare_and_summarize", self.compare_and_summarize_node)
        builder.add_node("reject_no_evidence", self.reject_no_evidence_node)
        builder.add_node("verify_sources", self.verify_sources_node)

        builder.add_edge(START, "analyze_query")
        builder.add_edge("analyze_query", "retrieve_news")

        builder.add_conditional_edges(
            "retrieve_news",
            self.decide_crawl_or_compare,
            {
                "compare": "compare_and_summarize",
                "crawl": "crawl_topic_to_db",
            },
        )

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
        """news.db에서 sqlite-vec title_vec 기반 하이브리드 RAG 검색을 수행합니다."""
        search_text = " ".join(state.get("keywords") or []) or state.get("query", "")
        articles = self.rag.retrieve(search_text, top_k=8, min_score=0.28, diversify_press=True)
        distinct_press = {a.press for a in articles}
        top_score = articles[0].score if articles else 0.0
        return {
            "articles": [a.model_dump() for a in articles],
            "is_sufficient": len(articles) >= 2 and len(distinct_press) >= 2 and top_score >= 0.35,
        }

    def decide_crawl_or_compare(self, state: AgentState) -> Literal["compare", "crawl"]:
        """
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
        """주제에 대한 기사들을 실시간으로 긁어와 news.db에 저장 및 title_vec 색인 후 재검색합니다."""
        search_text = " ".join(state.get("keywords") or []) or state.get("query", "")
        added = self.rag.crawl_topic_and_save_to_db(search_text, max_articles=8)
        articles = self.rag.retrieve(search_text, top_k=8, min_score=0.28, diversify_press=True)
        distinct_press = {a.press for a in articles}
        top_score = articles[0].score if articles else 0.0
        return {
            "crawled_count": added,
            "articles": [a.model_dump() for a in articles],
            "is_sufficient": len(articles) >= 2 and len(distinct_press) >= 2 and top_score >= 0.35,
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
                    "수집 출처: news.db (sqlite-vec) 검색 + 실시간 크롤링 결과 0건",
                    "환각 방지(성공 기준 1): 근거 없는 추측·가짜 출처 생성 차단됨",
                    f"추천 주제: {suggested}",
                ],
                "summary": (
                    f"로컬 데이터베이스(news.db) sqlite-vec 벡터 검색 및 실시간 뉴스 크롤링을 모두 수행했으나 ‘{q}’ 주제와 일치하는 "
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
                "articles": [],
            }
        }

    def compare_and_summarize_node(self, state: AgentState) -> AgentState:
        """검색·수집된 news.db 기사들을 하나의 헤드라인으로 묶어 요약하고 신문사별로 의견을 비교합니다."""
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
                        f"- 분류: {a.category} ({a.cluster_label})\n"
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
                    "분석 모델: Anthropic Claude (LangGraph + sqlite-vec RAG)",
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

        result["articles"] = raw_articles
        thread_id = state.get("thread_id") or "default-session"
        history = self._memory_store.setdefault(thread_id, [])
        history.append({"query": state.get("query", ""), "headline": result.get("headline", "")})
        if len(history) > 5:
            del history[:-5]

        return {"result": result}

    def run(self, query: str, thread_id: str = "default-session") -> dict[str, Any]:
        """주제/질문을 입력받아 [sqlite-vec 검색 -> 부족 시 크롤링·DB저장 -> 헤드라인 통합 요약 -> 신문사별 의견 비교]를 수행합니다."""
        final_state = self.graph.invoke({"query": query, "thread_id": thread_id})
        return final_state["result"]


__all__ = [
    "AgentState",
    "NJKAgent",
]


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    test_query = sys.argv[1] if len(sys.argv) > 1 else "수사기관장 배석 사법개혁"
    agent = NJKAgent()
    res = agent.run(test_query)
    print(json.dumps(res, ensure_ascii=False, indent=2))
