# -*- coding: utf-8 -*-
"""
[0조] [팀명-주제] AI Agent 미니 프로젝트 결과 발표.pptx
가독성 개선 스크립트 (1번 슬라이드는 수정하지 않음, 2~14번 슬라이드 레이아웃·여백·줄바꿈 최적화)
"""
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

PPTX_PATH = "[2조]-[논점콩-뉴스_기사_비교_에이전트].pptx"

COLOR_DARK = RGBColor(0x1F, 0x29, 0x37)
COLOR_MUTED = RGBColor(0x4B, 0x55, 0x63)
COLOR_BLUE = RGBColor(0x1D, 0x4E, 0xD8)


def set_box(shape, left_in=None, top_in=None, width_in=None, height_in=None):
    if left_in is not None:
        shape.left = Inches(left_in)
    if top_in is not None:
        shape.top = Inches(top_in)
    if width_in is not None:
        shape.width = Inches(width_in)
    if height_in is not None:
        shape.height = Inches(height_in)


def set_shape_text(
    shape,
    text,
    size_pt=12,
    bold=False,
    color=COLOR_DARK,
    space_after_pt=4,
    line_spacing=1.25,
    align=PP_ALIGN.LEFT,
):
    tf = shape.text_frame
    tf.word_wrap = True
    font_name = "IBM Plex Sans KR"
    if tf.paragraphs and tf.paragraphs[0].runs and tf.paragraphs[0].runs[0].font.name:
        font_name = tf.paragraphs[0].runs[0].font.name

    lines = text.split("\n")
    tf.clear()
    for idx, line in enumerate(lines):
        p = tf.paragraphs[0] if idx == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(space_after_pt)
        p.line_spacing = line_spacing
        run = p.add_run()
        run.text = line
        run.font.name = font_name
        run.font.size = Pt(size_pt)
        run.font.bold = bold
        run.font.color.rgb = color


def set_labeled_bullet(
    shape,
    label,
    body,
    size_pt=12,
    label_color=COLOR_BLUE,
    body_color=COLOR_DARK,
    line_spacing=1.3,
):
    tf = shape.text_frame
    tf.word_wrap = True
    font_name = "IBM Plex Sans KR"
    if tf.paragraphs and tf.paragraphs[0].runs and tf.paragraphs[0].runs[0].font.name:
        font_name = tf.paragraphs[0].runs[0].font.name

    tf.clear()
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    p.line_spacing = line_spacing
    p.space_after = Pt(4)

    r1 = p.add_run()
    r1.text = label
    r1.font.name = font_name
    r1.font.size = Pt(size_pt)
    r1.font.bold = True
    r1.font.color.rgb = label_color

    r2 = p.add_run()
    r2.text = body
    r2.font.name = font_name
    r2.font.size = Pt(size_pt)
    r2.font.bold = False
    r2.font.color.rgb = body_color


def set_cell_text(cell, text, size_pt=11, bold=False, color=COLOR_DARK):
    tf = cell.text_frame
    tf.word_wrap = True
    font_name = "IBM Plex Sans KR"
    if tf.paragraphs and tf.paragraphs[0].runs and tf.paragraphs[0].runs[0].font.name:
        font_name = tf.paragraphs[0].runs[0].font.name

    lines = text.split("\n")
    tf.clear()
    for idx, line in enumerate(lines):
        p = tf.paragraphs[0] if idx == 0 else tf.add_paragraph()
        p.space_after = Pt(2)
        p.line_spacing = 1.15
        run = p.add_run()
        run.text = line
        run.font.name = font_name
        run.font.size = Pt(size_pt)
        run.font.bold = bold
        run.font.color.rgb = color


def main():
    prs = Presentation(PPTX_PATH)

    # ※ Slide 1 (prs.slides[0])은 절대 수정하지 않음

    # =========================================================================
    # Slide 2: 01 · 프로젝트와 팀
    # =========================================================================
    s2 = prs.slides[1]
    set_box(s2.shapes[4], left_in=1.11, top_in=2.18, width_in=11.10, height_in=1.95)
    set_shape_text(
        s2.shapes[4],
        "• 동일한 사회·경제 이슈라도 언론사마다 강조하는 프레임이 달라 독자가 확증편향에 빠지기 쉽습니다.\n"
        "• 네이버 뉴스 실시간 크롤링 + SQLite 하이브리드 RAG(news.db) + 로컬 LLM(Kanana-Nano 2.1B)을 결합했습니다.\n"
        "• 단일 통합 헤드라인과 공통 요약, 양측 관점(관점 A vs 관점 B) 비교, 기사별 신뢰도 판정을 한눈에 제공합니다.",
        size_pt=13,
        space_after_pt=8,
        line_spacing=1.35,
    )
    t2 = s2.shapes[7].table
    team_rows = [
        ("장범규", "팀장 · AI 아키텍트", "LangGraph · 하이브리드 RAG", "멀티에이전트 워크플로우 설계 및 LLM-as-a-Judge 구축"),
        ("백예은", "데이터 엔지니어", "실시간 뉴스 크롤러 · SQLite", "네이버 뉴스 수집 파이프라인 및 news.db(sqlite-vec) 구축"),
        ("김민지", "프롬프트 · 검증 엔지니어", "관점 A/B 분리 · LLM Judge", "관점 비교 프롬프트, 개인정보 마스킹, LLM Judge 심판 설계"),
        ("손경택", "프론트엔드 · API 개발", "FastAPI · 반응형 웹 UI", "웹 백엔드(frontend.py) 및 관점별 클러스터 카드 UI"),
    ]
    for r_idx, row_data in enumerate(team_rows, start=1):
        for c_idx, val in enumerate(row_data):
            set_cell_text(
                t2.cell(r_idx, c_idx),
                val,
                size_pt=11.5,
                bold=(c_idx == 0),
                color=(COLOR_BLUE if c_idx == 0 else COLOR_DARK),
            )

    # =========================================================================
    # Slide 3: 02 · 배경과 문제 (카드 높이 확장 및 줄 간격 확보)
    # =========================================================================
    s3 = prs.slides[2]
    set_box(s3.shapes[2], left_in=0.89, top_in=1.65, width_in=5.67, height_in=4.55)
    set_box(s3.shapes[7], left_in=6.76, top_in=1.65, width_in=5.68, height_in=4.55)

    set_box(s3.shapes[4], left_in=1.11, top_in=2.30, width_in=5.25, height_in=0.95)
    set_labeled_bullet(
        s3.shapes[4],
        "• 확증편향 심화: ",
        "알고리즘 추천으로 특정 성향 매체만 소비하는 필터버블 발생\n  (77개 언론사 · 1,013건 기사 데이터셋 기반 분석)",
        size_pt=12.5,
    )

    set_box(s3.shapes[5], left_in=1.11, top_in=3.45, width_in=5.25, height_in=0.95)
    set_labeled_bullet(
        s3.shapes[5],
        "• 기존 AI 요약의 한계: ",
        "단일 요약만 제시해 언론사별 논조 차이를 뭉개거나,\n  실시간 원문 근거 없이 환각(Hallucination)을 유발",
        size_pt=12.5,
    )

    set_box(s3.shapes[6], left_in=1.11, top_in=4.60, width_in=5.25, height_in=0.95)
    set_labeled_bullet(
        s3.shapes[6],
        "• 해결 필요성: ",
        "실제 보도 원문(URL)을 바탕으로 공통 팩트와 상반된 시각\n  (관점 A vs 관점 B)을 동시에 대조할 수 있는 도구 필요",
        size_pt=12.5,
    )

    set_box(s3.shapes[9], left_in=6.99, top_in=2.30, width_in=5.25, height_in=0.95)
    set_labeled_bullet(
        s3.shapes[9],
        "누가  |  ",
        "시사·정책·경제 현안을 균형 있게 파악하려는\n뉴스 독자, 리서처 및 취업·논술 준비생",
        size_pt=12.5,
    )

    set_box(s3.shapes[10], left_in=6.99, top_in=3.45, width_in=5.25, height_in=0.95)
    set_labeled_bullet(
        s3.shapes[10],
        "어디가 불편한가  |  ",
        "언론사마다 같은 사안을 다르게 보도해 여러 기사를\n일일이 검색·대조해야 하며 팩트와 논조 구분이 어려움",
        size_pt=12.5,
    )

    set_box(s3.shapes[11], left_in=6.99, top_in=4.60, width_in=5.25, height_in=0.95)
    set_labeled_bullet(
        s3.shapes[11],
        "해결 방향  |  ",
        "에이전트가 다매체 기사를 실시간 수집·RAG 검색한 뒤\n공통 요약 · 관점 A/B 비교 · AI 신뢰도 판정을 자동 제공",
        size_pt=12.5,
    )

    # =========================================================================
    # Slide 4: 03 · 목표와 추진
    # =========================================================================
    s4 = prs.slides[3]
    set_box(s4.shapes[2], height_in=1.60)
    set_box(s4.shapes[5], height_in=1.60)
    set_box(s4.shapes[9], height_in=1.60)

    set_box(s4.shapes[4], left_in=1.11, top_in=2.15, width_in=4.25, height_in=0.95)
    set_shape_text(
        s4.shapes[4],
        "사용자 질의에 대해 최소 2개 이상의 상반된 언론사 관점(관점 A·B)과 원문 출처를 100% 무결하게 비교·판단하는 에이전트 완성",
        size_pt=12,
        bold=True,
        line_spacing=1.3,
    )

    set_box(s4.shapes[7], top_in=2.10, height_in=0.48)
    set_shape_text(s4.shapes[7], "LLM Judge", size_pt=26, bold=True, color=COLOR_BLUE)
    set_box(s4.shapes[8], top_in=2.62, width_in=2.88, height_in=0.52)
    set_shape_text(
        s4.shapes[8],
        "OpenAI Head LLM 심판 기반\n대립축·논조·출처 무결성 실시간 판정",
        size_pt=11,
        color=COLOR_DARK,
        space_after_pt=2,
    )

    set_box(s4.shapes[11], top_in=2.10, height_in=0.48)
    set_shape_text(s4.shapes[11], "3.7개 사", size_pt=30, bold=True, color=COLOR_BLUE)
    set_box(s4.shapes[12], top_in=2.62, width_in=2.88, height_in=0.52)
    set_shape_text(
        s4.shapes[12],
        "이슈당 평균 비교 신문사 수\n(목표 2.0개 대비 185% 달성)",
        size_pt=11,
        color=COLOR_DARK,
        space_after_pt=2,
    )

    # Day 1 ~ Day 5 카드 높이 확장
    for card_idx in (14, 17, 20, 23, 26):
        set_box(s4.shapes[card_idx], top_in=3.82, height_in=1.85)
    set_shape_text(s4.shapes[24], "Day 4", size_pt=12, bold=True, color=COLOR_BLUE)
    set_shape_text(s4.shapes[27], "Day 5", size_pt=12, bold=True, color=COLOR_BLUE)

    day_contents = [
        (16, "• 주제 선정·요구사항 정의\n• news.db 스키마 설계\n• 초기 기사 수집 및 전처리"),
        (19, "• sqlite-vec(title_vec) 색인\n• 신문사 다양성 리랭킹\n• 뉴스 분류기(LinearSVC) 구축"),
        (22, "• LangGraph 멀티에이전트 설계\n• FastAPI 백엔드 구축\n• 반응형 웹 UI 연동"),
        (25, "• Unsloth 로컬 SLM 요약 연동\n• 기사별 관점·신뢰도 판정\n• 관점 A/B 그룹 정렬 UI"),
        (28, "• LLM-as-a-Judge 심판 연동\n• 대립축·관점별 논조 자동 평가\n• 환각 차단 가드레일 완성"),
    ]
    for shape_idx, text in day_contents:
        set_box(s4.shapes[shape_idx], top_in=4.32, width_in=1.90, height_in=1.20)
        set_shape_text(s4.shapes[shape_idx], text, size_pt=10.5, space_after_pt=4, line_spacing=1.25)

    # =========================================================================
    # Slide 5: 04 · 서비스 시나리오
    # =========================================================================
    s5 = prs.slides[4]
    set_box(s5.shapes[2], top_in=1.65, height_in=3.35)
    set_box(s5.shapes[4], left_in=1.11, top_in=2.18, width_in=11.10, height_in=2.60)
    set_shape_text(
        s5.shapes[4],
        "• 상황: 사용자가 논쟁적인 최신 이슈(중동 호르무즈 해협 봉쇄, 추경 편성, 원전 정책 등)에 대해 언론사별 시각 차이를 비교하고 싶을 때\n\n"
        "[ 실제 대화 예시 ]\n"
        "• 사용자 입력: “이란 호르무즈 해협 봉쇄 위기와 한국 선박 통항 논란에 대해 신문사별 관점을 비교해줘.”\n"
        "• 에이전트 응답 요약:\n"
        "   - 통합 헤드라인: ‘호르무즈 해협 봉쇄 위기 고조… 안보·제재 공조 vs 외교·경제 실리 해법 엇갈려’\n"
        "   - 관점 A (안보·리스크 강조): 해협 봉쇄 장기화에 따른 원유 수급 차질 경고 및 국제 해양안보 공조·대응 태세 강조\n"
        "   - 관점 B (외교·완화 강조): 외교 채널을 통한 선박 안전 통항 확보 및 국내 유가·물류비 충격 완화 대책 우선 주문",
        size_pt=11.5,
        space_after_pt=4,
        line_spacing=1.28,
    )

    for card_idx in (5, 8, 11, 14):
        set_box(s5.shapes[card_idx], top_in=5.18, height_in=1.35)
    for title_idx in (6, 9, 12, 15):
        set_box(s5.shapes[title_idx], top_in=5.32)

    scenario_steps = [
        (7, "• 이슈 키워드 또는 질문 입력\n• 필요 시 ‘실시간 뉴스 크롤링’\n  스위치 활성화"),
        (10, "• 의도 분류(compare)\n• news.db 하이브리드 RAG 검색\n• 기사 부족 시 크롤러 자동 호출"),
        (13, "• 다매체 기사 6건 수집·선발\n• 관점 A / 관점 B 자동 군집화\n• OpenAI API 기사별 신뢰도 판정"),
        (16, "• 통합 헤드라인 + 공통 요약\n• 관점 A/B 인접 정렬 판단 카드\n• 신뢰도 점수 및 원문 URL 제공"),
    ]
    for shape_idx, text in scenario_steps:
        set_box(s5.shapes[shape_idx], top_in=5.62, width_in=2.48, height_in=0.85)
        set_shape_text(
            s5.shapes[shape_idx],
            text,
            size_pt=10.5,
            bold=(shape_idx == 16),
            color=(COLOR_BLUE if shape_idx == 16 else COLOR_DARK),
            space_after_pt=2,
            line_spacing=1.2,
        )

    # =========================================================================
    # Slide 6: 05 · 시스템 - AI Agent 시스템 구성도 (박스 내 겹침 완전 제거)
    # =========================================================================
    s6 = prs.slides[5]
    # 좌측 사용자/UI 박스
    set_box(s6.shapes[3], left_in=1.07, top_in=2.22, width_in=2.53, height_in=0.28)
    set_box(s6.shapes[4], left_in=1.07, top_in=2.60, width_in=2.53, height_in=1.20)
    set_shape_text(
        s6.shapes[4],
        "• 반응형 웹 (newslens.html)\n• FastAPI (/api/analyze, /chat)\n• CLI 대화형 인터페이스",
        size_pt=11,
        space_after_pt=5,
    )

    # 중앙 AI Agent 앱 박스 (pos 4.37, 1.65, size 4.33 x 2.76in)
    set_box(s6.shapes[8], left_in=4.55, top_in=1.82, width_in=4.00, height_in=0.24)
    set_box(s6.shapes[9], left_in=4.55, top_in=2.10, width_in=4.00, height_in=0.28)
    set_box(s6.shapes[10], left_in=4.55, top_in=2.45, width_in=4.00, height_in=0.70)
    set_shape_text(
        s6.shapes[10],
        "• 프레임워크: LangGraph StateGraph (5 Nodes)\n• LLM: Unsloth OpenAI API (Kanana-Nano 2.1B)",
        size_pt=11,
        bold=True,
        color=COLOR_BLUE,
        space_after_pt=3,
    )
    set_box(s6.shapes[11], left_in=4.55, top_in=3.25, width_in=4.00, height_in=0.95)
    set_shape_text(
        s6.shapes[11],
        "• 분기 규칙: 의도 분류(compare / crawl) 및\n  검색 기사 수(<2건 시 실시간 크롤링) 동적 라우팅\n• 유사도 < 0.28 시 환각 차단 응답 즉시 반환",
        size_pt=10.5,
        color=COLOR_DARK,
        space_after_pt=3,
    )

    # 우측 3개 박스 (지식 RAG, 도구 MCP, 메모리)
    set_box(s6.shapes[15], top_in=1.72)
    set_box(s6.shapes[16], left_in=9.52, top_in=2.00, width_in=2.82, height_in=0.45)
    set_shape_text(
        s6.shapes[16],
        "SQLite (news.db · 1,013건 · 77개 매체)\n384차원 임베딩 + 언론사 다양성 리랭킹",
        size_pt=10,
        space_after_pt=2,
        line_spacing=1.15,
    )

    set_box(s6.shapes[18], top_in=2.69)
    set_box(s6.shapes[19], left_in=9.52, top_in=2.97, width_in=2.82, height_in=0.45)
    set_shape_text(
        s6.shapes[19],
        "네이버 뉴스 실시간 크롤러 (httpx)\nDB 자동 적재 및 개인정보(PII) 마스킹",
        size_pt=10,
        space_after_pt=2,
        line_spacing=1.15,
    )

    set_box(s6.shapes[21], top_in=3.65)
    set_box(s6.shapes[22], left_in=9.52, top_in=3.93, width_in=2.82, height_in=0.45)
    set_shape_text(
        s6.shapes[22],
        "thread_id 세션별 인메모리 저장소\n최근 5턴 슬라이딩 윈도우 맥락 유지",
        size_pt=10,
        space_after_pt=2,
        line_spacing=1.15,
    )

    # 하단 실행 환경 박스
    set_box(s6.shapes[25], left_in=1.11, top_in=5.15, width_in=11.10, height_in=1.05)
    set_shape_text(
        s6.shapes[25],
        "• 웹 & 에이전트 서버: 로컬 FastAPI + Uvicorn (http://0.0.0.0:8000, 동일 네트워크 외부 접속 지원)\n"
        "• LLM 추론 서버: 로컬 Unsloth OpenAI 호환 서버 (http://127.0.0.1:8889/v1, Kanana-Nano-2.1B-Instruct-Q8_0-GGUF)\n"
        "• 데이터베이스: 로컬 SQLite 단일 파일 (news.db — LINK, NEWS, RAG_INDEX 테이블 통합 운용)",
        size_pt=11.5,
        space_after_pt=4,
    )

    # =========================================================================
    # Slide 7: 05 · 시스템 - AI Agent 처리 흐름도
    # =========================================================================
    s7 = prs.slides[6]
    set_box(s7.shapes[3], top_in=2.10)
    set_box(s7.shapes[4], left_in=1.02, top_in=2.40, width_in=1.45, height_in=0.48)
    set_shape_text(s7.shapes[4], "이슈 질문 입력\n+ 크롤링 옵션", size_pt=10, space_after_pt=1, line_spacing=1.15)

    set_box(s7.shapes[7], top_in=2.10)
    set_box(s7.shapes[8], left_in=3.28, top_in=2.40, width_in=1.78, height_in=0.48)
    set_shape_text(s7.shapes[8], "비교 / 크롤링 분류\n+ 이전 턴 맥락 결합", size_pt=10, space_after_pt=1, line_spacing=1.15)

    set_shape_text(
        s7.shapes[10],
        "③-A 검색 (RAG)\nnews.db 다양성 검색",
        size_pt=9.5,
        bold=True,
        color=COLOR_BLUE,
        space_after_pt=1,
        line_spacing=1.1,
    )
    set_shape_text(
        s7.shapes[11],
        "③-B 도구 호출 (크롤러)\n네이버 뉴스 실시간 수집",
        size_pt=9.5,
        bold=True,
        color=COLOR_DARK,
        space_after_pt=1,
        line_spacing=1.1,
    )
    set_shape_text(
        s7.shapes[12],
        "③-C 환각 차단 안내\n임계치(0.28) 미달 차단",
        size_pt=9.5,
        bold=True,
        color=COLOR_DARK,
        space_after_pt=1,
        line_spacing=1.1,
    )

    set_box(s7.shapes[15], top_in=2.10)
    set_box(s7.shapes[16], left_in=8.28, top_in=2.40, width_in=1.78, height_in=0.48)
    set_shape_text(s7.shapes[16], "관점 A/B 비교 요약\n+ AI 기사별 신뢰도", size_pt=10, space_after_pt=1, line_spacing=1.15)

    set_box(s7.shapes[19], top_in=2.10)
    set_box(s7.shapes[20], left_in=10.88, top_in=2.40, width_in=1.45, height_in=0.48)
    set_shape_text(s7.shapes[20], "관점별 그룹 카드\n+ 원문 링크 렌더링", size_pt=10, space_after_pt=1, line_spacing=1.15)

    # 하단 3개 카드 높이 확장 (3.56 ~ 6.10in)
    for card_idx in (21, 24, 27):
        set_box(s7.shapes[card_idx], top_in=3.56, height_in=2.55)
    set_shape_text(s7.shapes[25], "검증·반복", size_pt=12, bold=True, color=COLOR_BLUE)

    set_box(s7.shapes[23], left_in=1.11, top_in=4.10, width_in=3.32, height_in=1.85)
    set_shape_text(
        s7.shapes[23],
        "• 일반 비교 질의:\n  ③-A (news.db RAG 검색) 우선 수행\n"
        "• 실시간 크롤링 ON 또는 검색 < 2건:\n  ③-B (네이버 뉴스 수집 후 재검색)\n"
        "• 유사도 점수 < 0.28:\n  ③-C (환각 차단 안내 즉시 반환)",
        size_pt=11,
        space_after_pt=4,
        line_spacing=1.25,
    )

    set_box(s7.shapes[26], left_in=5.03, top_in=4.10, width_in=3.32, height_in=1.85)
    set_shape_text(
        s7.shapes[26],
        "• DB 검색 기사가 2건 미만이면\n  실시간 크롤러 1회 자동 호출 후 재검색\n"
        "• 1차 라운드에서 신문사 중복을 배제해\n  최소 2개 이상 매체 다양성 강제 보장\n"
        "• 기사별 판단 완료 후 관점 A → B 순 정렬",
        size_pt=11,
        space_after_pt=4,
        line_spacing=1.25,
    )

    set_box(s7.shapes[29], left_in=8.94, top_in=4.10, width_in=3.32, height_in=1.85)
    set_shape_text(
        s7.shapes[29],
        "• 허위·무관 이슈(화성 헌법 등) 질의 시\n  억지 요약 없이 ‘보도 없음’ 안내 (100% 차단)\n"
        "• 소형 LLM JSON 파싱 실패 시 정규식 복구\n  및 프레임 키워드 폴백으로 무중단 응답\n"
        "• 기자 이메일·전화번호 정규식 자동 마스킹",
        size_pt=11,
        space_after_pt=4,
        line_spacing=1.25,
    )

    # =========================================================================
    # Slide 8: 06 · 구성요소 - 상세 설명
    # =========================================================================
    s8 = prs.slides[7]
    t8 = s8.shapes[4].table
    comp_rows = [
        (
            "오케스트레이터·LLM",
            "질의 의도 분류, 동적 라우팅,\n관점 A/B 요약 및 신뢰도 판정",
            "LangGraph StateGraph (5노드)\nUnsloth Kanana-Nano-2.1B (OpenAI API)",
            "명시적 상태 그래프로 분기 투명성을 확보하고\n로컬 GGUF 추론 서버로 API 비용 제로 달성",
        ),
        (
            "지식 (RAG)",
            "다매체 뉴스 인덱싱 및\n언론사 다양성 보장 하이브리드 검색",
            "SQLite (news.db · 1,013건)\n384차원 N-gram 임베딩 + 키워드 매칭",
            "별도 무거운 벡터 DB 없이 단일 SQLite 파일과\n인메모리 내적으로 평균 0.04초 초고속 검색",
        ),
        (
            "도구 (MCP/크롤러)",
            "최신 이슈 실시간 네이버 뉴스 수집,\nDB 적재 및 개인정보 보호",
            "httpx 비동기 크롤러 · BeautifulSoup\n이메일·전화번호 정규식 마스킹",
            "정적 DB의 시의성 한계를 극복하기 위해\n검색 부족 시 에이전트가 스스로 뉴스 수집",
        ),
        (
            "메모리",
            "멀티턴 후속 질문(‘경제적 영향은?’)\n처리를 위한 세션 맥락 유지",
            "thread_id 기반 인메모리 저장소\n최근 5턴 슬라이딩 윈도우",
            "직전 질문의 핵심 키워드를 후속 질문에 결합해\n생략·대명사 질의에서도 검색 정확도 유지",
        ),
        (
            "인터페이스",
            "이슈 검색, 실시간 크롤링 토글,\n관점별 그룹 카드 시각화",
            "FastAPI + Uvicorn (frontend.py)\n반응형 웹 대시보드 (newslens.html)",
            "관점 A와 관점 B 기사를 서로 인접하게 묶어 배치해\n독자가 언론사별 논조 차이를 직관적으로 대조",
        ),
    ]
    for r_idx, row_data in enumerate(comp_rows, start=1):
        for c_idx, val in enumerate(row_data):
            set_cell_text(
                t8.cell(r_idx, c_idx),
                val,
                size_pt=10.5,
                bold=(c_idx == 0),
                color=(COLOR_BLUE if c_idx == 0 else COLOR_DARK),
            )

    # =========================================================================
    # Slide 9: 06 · 구성요소 - 핵심 구성요소 깊이 보기 (우측 코드 박스 영역 확장)
    # =========================================================================
    s9 = prs.slides[8]
    set_box(s9.shapes[4], left_in=1.11, top_in=2.08, width_in=4.85, height_in=0.42)
    set_shape_text(
        s9.shapes[4],
        "다매체 다양성 하이브리드 RAG & 관점별 클러스터링 엔진",
        size_pt=12.5,
        bold=True,
        color=COLOR_BLUE,
    )

    set_box(s9.shapes[7], left_in=1.11, top_in=3.10, width_in=4.85, height_in=0.48)
    set_shape_text(
        s9.shapes[7],
        "단순 Top-K 검색의 특정 언론사 독점 문제를 해결하기 위해\n서로 다른 신문사를 우선 선발하고 관점 A/B별로 묶어 정렬",
        size_pt=10.5,
        space_after_pt=2,
        line_spacing=1.2,
    )

    set_box(s9.shapes[10], left_in=1.11, top_in=4.12, width_in=4.85, height_in=0.48)
    set_shape_text(
        s9.shapes[10],
        "• 하이브리드 점수 = 0.45×코사인(384d) + 0.55×키워드 매칭\n• OpenAI API로 기사별 관점(A/B)·논조·신뢰도(0~100) 판정",
        size_pt=10.5,
        space_after_pt=2,
        line_spacing=1.2,
    )

    set_box(s9.shapes[13], left_in=1.11, top_in=5.14, width_in=4.85, height_in=0.48)
    set_shape_text(
        s9.shapes[13],
        "• 효과: 이슈당 평균 3.7개 언론사 비교, 환각 차단율 100% 달성\n• 개선: 소형 LLM JSON 오류를 정규식·키워드 폴백으로 100% 보완",
        size_pt=10.5,
        space_after_pt=2,
        line_spacing=1.2,
    )

    # 우측 코드 카드 내부 레이아웃 최적화 (pos 6.28, 1.65, size 6.16 x 3.95in)
    set_box(s9.shapes[15], left_in=6.52, top_in=1.88, width_in=5.68, height_in=3.10)
    set_shape_text(
        s9.shapes[15],
        "# 1) 신문사 다양성 우선 리랭킹 (rag_agent.py)\n"
        "for item in scored_articles:\n"
        "    if item['press'] not in seen_press:\n"
        "        selected.append(item)\n"
        "        seen_press.add(item['press'])\n\n"
        "# 2) OpenAI API 기사 판단 및 관점별 인접 정렬 (frontend.py)\n"
        "resp = client.chat.completions.create(\n"
        "    model='kenonix/kanana-nano-2.1b-instruct-Q8_0-GGUF',\n"
        "    messages=[{'role': 'system', 'content': prompt}, ...]\n"
        ")\n"
        "judgments.sort(key=lambda x: (\n"
        "    0 if 'A' in x['perspective'] else 1, -x['trust_score']\n"
        "))",
        size_pt=10.5,
        bold=False,
        color=COLOR_DARK,
        space_after_pt=2,
        line_spacing=1.18,
    )
    set_box(s9.shapes[16], left_in=6.52, top_in=5.12, width_in=5.68, height_in=0.32)
    set_shape_text(
        s9.shapes[16],
        "▲ rag_agent.py 다매체 다양성 리랭킹 및 frontend.py 관점 A/B 인접 정렬 핵심 로직",
        size_pt=10,
        bold=True,
        color=COLOR_BLUE,
    )

    # =========================================================================
    # Slide 10: 07 · 데모 - 데모 화면
    # =========================================================================
    s10 = prs.slides[9]
    # 상단 데모 카드 내부 (pos 0.90, 1.65, size 11.54 x 2.90in)
    set_box(s10.shapes[3], left_in=1.20, top_in=1.88, width_in=10.95, height_in=2.10)
    set_shape_text(
        s10.shapes[3],
        "🌐 라이브 데모 웹 대시보드 (http://127.0.0.1:8000)\n\n"
        "① 상단 요약 영역: 단일 통합 헤드라인 + 다매체 공통 사실 요약 + 비교 신문사 태그\n"
        "② 중단 관점 대조: [ 관점 A (비판·리스크·안보 프레임) ]  vs  [ 관점 B (기회·정책추진·완화 프레임) ] 나란히 비교\n"
        "③ 하단 AI 기사 판단: OpenAI API(Kanana-Nano 2.1B) 기반 기사별 관점(A/B)·논조·신뢰도 점수(0~100점) 카드\n"
        "   → 같은 관점(관점 A 그룹 / 관점 B 그룹)끼리 서로 붙어 정렬되어 언론사별 시각 차이를 직관적으로 확인",
        size_pt=12,
        bold=False,
        color=COLOR_DARK,
        space_after_pt=5,
        line_spacing=1.28,
    )
    # 첫 줄만 파란색 볼드 강조
    s10.shapes[3].text_frame.paragraphs[0].runs[0].font.bold = True
    s10.shapes[3].text_frame.paragraphs[0].runs[0].font.color.rgb = COLOR_BLUE
    s10.shapes[3].text_frame.paragraphs[0].runs[0].font.size = Pt(13.5)

    set_box(s10.shapes[4], left_in=1.20, top_in=4.12, width_in=10.95, height_in=0.30)
    set_shape_text(
        s10.shapes[4],
        "▲ FastAPI 백엔드(frontend.py) + 반응형 웹 UI(newslens.html) 실시간 연동 화면 구성",
        size_pt=10.5,
        bold=True,
        color=COLOR_BLUE,
    )

    # 하단 데모 순서 카드 (pos 0.89, 4.79, size 11.56 x 1.60in)
    set_box(s10.shapes[7], left_in=1.11, top_in=5.28, width_in=11.10, height_in=0.98)
    set_shape_text(
        s10.shapes[7],
        "1) 입력: 웹 검색창에 ‘호르무즈 해협 봉쇄’ 또는 ‘추경 편성’ 입력 (필요 시 ‘실시간 뉴스 크롤링’ 스위치 ON)\n"
        "2) 에이전트 동작: news.db 하이브리드 RAG 검색 후 로컬 OpenAI API(127.0.0.1:8889)로 기사별 관점(A/B)·논조·신뢰도 실시간 판정\n"
        "3) 결과 확인: 통합 헤드라인 · 양측 관점 요약 · 관점별로 서로 붙어 그룹 정렬된 AI 기사 판단 카드 및 원문 URL 확인",
        size_pt=11.5,
        space_after_pt=4,
        line_spacing=1.25,
    )

    # =========================================================================
    # Slide 11: 08 · 평가 - LLM-as-a-Judge 기반 평가 결과
    # =========================================================================
    s11 = prs.slides[10]
    t11 = s11.shapes[4].table
    eval_rows = [
        (
            "대립축 및 관점 분리 타당성",
            "LLM-as-a-Judge (Head Supervisor 심판 평가)",
            "양측 핵심 쟁점·논조 명확 분리",
            "적합 판정 (관점 A vs B 대립축 도출)",
        ),
        (
            "기사별 논조·근거 충실도 검증",
            "JudgeVerifierSubAgent 개별 기사 심판 및 URL 대조",
            "원문 왜곡 없음 · URL 100% 일치",
            "신뢰도 95~97점 · 원문 100% 일치",
        ),
        (
            "무관·허위 이슈 환각 차단",
            "LLM Judge 사전 라우팅 + HallucinationGuardSubAgent",
            "근거 부재 시 억지 요약 즉시 차단",
            "환각 차단 작동 (추측 생성 0건)",
        ),
    ]
    for r_idx, row_data in enumerate(eval_rows, start=1):
        for c_idx, val in enumerate(row_data):
            set_cell_text(
                t11.cell(r_idx, c_idx),
                val,
                size_pt=11,
                bold=(c_idx in (0, 3)),
                color=(COLOR_BLUE if c_idx == 3 else COLOR_DARK),
            )

    # 하단 카드 2개 높이 여유 확보
    set_box(s11.shapes[5], top_in=3.45, height_in=1.75)
    set_box(s11.shapes[9], top_in=3.45, height_in=1.75)

    set_box(s11.shapes[6], top_in=3.62)
    set_box(s11.shapes[7], top_in=3.92, height_in=0.52)
    set_shape_text(s11.shapes[7], "LLM-as-a-Judge", size_pt=26, bold=True, color=COLOR_BLUE)
    set_box(s11.shapes[8], top_in=4.52, width_in=5.30, height_in=0.45)
    set_shape_text(
        s11.shapes[8],
        "OpenAI Head LLM 심판 에이전트가 매 응답마다 대립축·관점별 논조·기사 신뢰도를 실시간 감수",
        size_pt=11,
    )

    set_box(s11.shapes[10], top_in=3.62)
    set_box(s11.shapes[11], top_in=3.95, height_in=0.38)
    set_shape_text(s11.shapes[11], "실시간 심판 · 출처 검증 통과", size_pt=16, bold=True, color=COLOR_BLUE)
    set_box(s11.shapes[12], top_in=4.42, width_in=5.30, height_in=0.55)
    set_shape_text(
        s11.shapes[12],
        "1) 대립축 도출 · 2) 관점 A/B 논조 판정 · 3) 종합 결론 · 4) 개별 기사 신뢰도 카드를 웹 UI에 실시간 제공",
        size_pt=11,
        line_spacing=1.25,
    )

    # =========================================================================
    # Slide 12: 08 · 평가 - 잘된 점과 한계 (카드 높이 4.3in으로 확장 및 간격 분리)
    # =========================================================================
    s12 = prs.slides[11]
    for card_idx in (2, 6, 10):
        set_box(s12.shapes[card_idx], top_in=1.65, height_in=4.45)

    # 잘된 점
    set_box(s12.shapes[4], left_in=1.11, top_in=2.28, width_in=3.32, height_in=1.60)
    set_labeled_bullet(
        s12.shapes[4],
        "• 다매체 관점 비교 & 클러스터링\n",
        "신문사 다양성 리랭킹과 관점 A/B 그룹 정렬 UI를 통해 이슈당 평균 3.7개 언론사의 논조 차이를 출처 무결성 100%로 직관적 비교",
        size_pt=11.5,
    )
    set_box(s12.shapes[5], left_in=1.11, top_in=4.05, width_in=3.32, height_in=1.60)
    set_labeled_bullet(
        s12.shapes[5],
        "• 완벽한 환각 차단 & 개인정보 보호\n",
        "유사도 임계치(0.28) 가드레일로 허위 이슈 질의를 100%(5/5) 차단하고, 기사 내 이메일·전화번호를 정규식으로 100% 자동 마스킹",
        size_pt=11.5,
    )

    # 한계·실패 사례
    set_box(s12.shapes[8], left_in=5.03, top_in=2.28, width_in=3.32, height_in=1.60)
    set_labeled_bullet(
        s12.shapes[8],
        "• 실패 사례 (초기 버전)\n",
        "‘화성 식민지 헌법 개정 논란’ 등 DB에 없는 허위 질의 시 일반 시사 어휘 중복으로 무관 기사가 매칭되거나, 환경변수 누락 시 기동 오류 발생",
        size_pt=11.5,
    )
    set_box(s12.shapes[9], left_in=5.03, top_in=4.05, width_in=3.32, height_in=1.60)
    set_labeled_bullet(
        s12.shapes[9],
        "• 원인 분석 및 해결\n",
        "단순 코사인 유사도에 핵심 키워드 매칭 가중치(0.55)를 결합하고 임계치(0.28) 필터링 및 환경변수 기본값 안전 처리를 적용해 해결",
        size_pt=11.5,
    )

    # 배운 점
    set_box(s12.shapes[12], left_in=8.94, top_in=2.28, width_in=3.32, height_in=1.60)
    set_labeled_bullet(
        s12.shapes[12],
        "• 기술적 배운 점\n",
        "단순 Top-K 벡터 검색보다 도메인 특성(언론사 다양성 강제, 하이브리드 가중치, LLM JSON 폴백 파서)이 실제 에이전트 품질을 좌우함",
        size_pt=11.5,
    )
    set_box(s12.shapes[13], left_in=8.94, top_in=4.05, width_in=3.32, height_in=1.60)
    set_labeled_bullet(
        s12.shapes[13],
        "• 협업에서 배운 점\n",
        "DB 전처리(sqlite-vec) · 멀티에이전트(LangGraph) · 로컬 SLM 요약(summary.py) · LLM Judge 웹 UI(frontend.py)를 모듈화하여 빠른 통합 달성",
        size_pt=11.5,
    )

    # =========================================================================
    # Slide 13: 09 · 향후 과제 (표 1줄 최적화 + 확장 아이디어 여백 확보)
    # =========================================================================
    s13 = prs.slides[12]
    t13 = s13.shapes[4].table
    future_rows = [
        (
            "한국어 Dense Embedding 고도화 (BGE-M3)",
            "동의어·유사 표현 시맨틱 검색 정밀도 및 장문 문맥 파악 향상",
            "상",
            "고도화 1주차",
        ),
        (
            "기사별 LLM 판단 병렬 비동기화 (AsyncOpenAI)",
            "다수 기사(6건) 동시 판정으로 응답 지연을 4초 → 1.5초로 단축",
            "상",
            "고도화 2주차",
        ),
        (
            "이슈 타임라인 시각화 및 팩트체크 API 연동",
            "시간축에 따른 논조 변화 추적 및 공신력 있는 팩트체크 결합",
            "중",
            "고도화 3~4주차",
        ),
    ]
    for r_idx, row_data in enumerate(future_rows, start=1):
        for c_idx, val in enumerate(row_data):
            set_cell_text(
                t13.cell(r_idx, c_idx),
                val,
                size_pt=11,
                bold=(c_idx in (0, 2)),
                color=(COLOR_BLUE if c_idx == 2 else COLOR_DARK),
            )

    set_box(s13.shapes[5], top_in=3.45, height_in=2.85)
    set_box(s13.shapes[6], top_in=3.65)
    set_box(s13.shapes[7], left_in=1.11, top_in=4.02, width_in=11.10, height_in=2.05)
    set_shape_text(
        s13.shapes[7],
        "• 글로벌 다매체 확장: 국내 언론사뿐 아니라 해외 주요 외신(Reuters, AP, BBC 등) 및 유튜브 뉴스 자막까지 수집 범위를 넓혀 국내외 시각 대조\n"
        "• 맞춤형 모닝 브리핑 에이전트: 사용자가 등록한 관심 키워드를 매일 아침 자동 크롤링·관점 비교 요약하여 전달하는 스케줄링 에이전트\n"
        "• 편향성 리포트 대시보드: 특정 이슈에 대해 언론사별 논조 스펙트럼과 보도량 추이를 시계열 차트로 시각화",
        size_pt=12,
        space_after_pt=8,
        line_spacing=1.35,
    )

    # =========================================================================
    # Slide 14: 감사합니다
    # =========================================================================
    s14 = prs.slides[13]
    tf14 = s14.shapes[4].text_frame
    tf14.clear()
    p14 = tf14.paragraphs[0]
    for txt_part, is_bold, col in [
        ("팀명 ", False, COLOR_DARK),
        ("2조 (논점콩 · 장범규·백예은·김민지·손경택)", True, COLOR_BLUE),
        ("  ·  연락처 ", False, COLOR_DARK),
        ("https://github.com/Jangbk633/proj1week", False, COLOR_DARK),
    ]:
        r = p14.add_run()
        r.text = txt_part
        r.font.name = "IBM Plex Sans KR"
        r.font.size = Pt(13)
        r.font.bold = is_bold
        r.font.color.rgb = col

    prs.save(PPTX_PATH)
    print("Successfully updated slides 2-14 (Slide 1 untouched):", PPTX_PATH)


if __name__ == "__main__":
    main()
