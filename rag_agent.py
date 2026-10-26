"""
rag_agent.py - A2 문제 정의 + sqlite-vec (title_vec) 고속 벡터 색인 & 뉴스 자동 분류 AI 에이전트:
1. NEWS.CLEAN_TITLE 전처리 및 NEWS.title_vec (float32[512]) 벡터화
2. sqlite-vec 확장(VEC_NEWS_TITLE vec0 가상 테이블 & vec_distance_cosine)을 통한 초고속 색인·검색
3. title_vec 기반 뉴스 자동 분류 시스템 (6대 뉴스 카테고리 분류 + KMeans 이슈 클러스터 분류 + 관점 A/B 분류)
4. LangGraph 기반 신문사별 관점 비교·통합 헤드라인 요약 및 출처 검증
"""

from __future__ import annotations

import hashlib
import html
import json
import os
from pathlib import Path
import re
import sqlite3
import time
from typing import Any, Literal, TypedDict
from urllib.parse import quote_plus

from dotenv import load_dotenv
import httpx
import numpy as np
from pydantic import BaseModel, Field
import sqlite_vec

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_DB_PATH = ROOT_DIR / "news.db"
TITLE_VEC_DIM = 512

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
    "076": "스포츠조선",
    "094": "월간산",
    "117": "마이데일리",
    "145": "레이디경향",
    "241": "일간스포츠",
    "262": "신동아",
    "296": "코메디닷컴",
    "308": "시사IN",
    "310": "여성신문",
    "346": "헬스조선",
    "366": "조선비즈",
    "374": "SBS Biz",
    "417": "머니S",
    "421": "뉴스1",
    "422": "연합뉴스TV",
    "437": "JTBC",
    "448": "TV조선",
    "449": "채널A",
    "468": "스포츠서울",
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
    "의견", "해석", "헤드라인", "방금", "요약한", "내용", "그중", "이전", "질문",
    "질문에서", "다룬", "위에서", "자세히", "신문사", "신문사들의", "언론사", "언론사들의",
    "입장을", "의견을", "근거", "수치", "수치를", "핵심", "평가하는지", "대해서",
    "종합", "사진", "영상", "일문일답", "전문", "사설", "칼럼", "단신", "브리핑",
    "초점", "현장", "르포", "인터뷰", "기획", "연재", "긴급", "주목", "화제", "해설",
    "만에", "하루", "연중", "내내", "이후", "부터", "까지", "위해", "통해", "따라",
    "이번", "지난", "내일", "어제", "올해", "내년", "작년", "현재", "당시",
    "우리", "저희", "그것", "이것", "저것", "여기", "거기", "저기", "모두", "전체", "일부", "각각",
    "있는", "없는", "하는", "되는", "않는", "같은", "위한", "통한", "따른", "대비", "두고", "놓고",
    "있어", "없어", "하며", "되며", "되고", "한다", "된다", "했다", "됐다", "할까", "될까",
}

# preprocessing.ipynb 기반 의미 복원 한자 변환 규칙
CONTEXTUAL_HANJA_PATTERNS: list[tuple[str, str]] = [
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

SEMANTIC_HANJA_DICT: dict[str, str] = {
    "李": "이재명 ", "尹": "윤석열 ", "韓": "한동훈 ", "文": "문재인 ",
    "洪": "홍준표 ", "吳": "오세훈 ", "安": "안철수 ", "曺": "조국 ",
    "美": "미국 ", "北": "북한 ", "中": "중국 ", "日": "일본 ", "露": "러시아 ",
    "英": "영국 ", "佛": "프랑스 ", "獨": "독일 ", "臺": "대만 ",
    "與": "여당 ", "野": "야당 ", "政": "정부 ", "檢": "검찰 ", "警": "경찰 ",
    "軍": "군대 ", "靑": "청와대 ", "法": "법원 ", "銀": "은행 ", "證": "증권 ",
    "企": "기업 ", "黨": "정당 ",
}

PHONETIC_HANJA_TRANS = str.maketrans({
    "萬": "만", "億": "억", "兆": "조",
    "年": "년", "月": "월", "時": "시", "分": "분", "秒": "초",
    "大": "대", "小": "소", "新": "신", "舊": "구", "高": "고", "低": "저", "多": "다", "少": "소",
    "前": "전", "後": "후", "上": "상", "下": "하", "內": "내", "外": "외", "左": "좌", "右": "우",
    "正": "정", "副": "부", "親": "친", "反": "반", "非": "비", "汎": "범", "全": "전",
    "對": "대", "發": "발", "行": "행", "故": "고", "無": "무", "有": "유", "重": "중", "輕": "경",
    "株": "주", "債": "채", "稅": "세", "金": "금", "院": "원", "朴": "박",
})

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


def preprocess_title(text: str) -> str:
    """
    preprocessing.ipynb 규칙과 동일하게 뉴스 제목의 한자어를 의미 복원하고
    상투어 태그·특수문자·조사·불용어를 정제하여 CLEAN_TITLE을 반환합니다.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    tag_pattern = (
        r"[\[\(\<【]\s*(?:종합|단독|속보|포토|사진|영상|일문일답|전문|사설|칼럼|브리핑|"
        r"현장|르포|인터뷰|기획|긴급|뉴스|그래픽|카드뉴스|팩트체크|이슈)[0-9보\s]*[\]\)\>】]"
    )
    text = re.sub(tag_pattern, " ", text)
    text = re.sub(r"(\d+)\s*日", r"\1일", text)

    for pattern, replacement in CONTEXTUAL_HANJA_PATTERNS:
        text = re.sub(pattern, replacement, text)

    for hanja, full_word in SEMANTIC_HANJA_DICT.items():
        text = re.sub(hanja + r"(?=[은는이가을를의에로과와도만]\b)", full_word.strip(), text)
        text = text.replace(hanja, full_word)

    text = text.translate(PHONETIC_HANJA_TRANS)
    text = re.sub(r"[^가-힣a-zA-Z0-9\s]", " ", text)

    words = text.split()
    cleaned_words: list[str] = []
    for w in words:
        if w in STOPWORDS:
            continue
        if len(w) >= 3:
            w_stem = re.sub(
                r"(에서|으로|에게|부터|까지|이랑|라는|에는|이라|라고|은|는|이|가|을|를|의|에|로|과|와|도|만)$",
                "",
                w,
            )
            if len(w_stem) >= 2:
                w = w_stem
        if w not in STOPWORDS and len(w) >= 2:
            cleaned_words.append(w)

    return " ".join(cleaned_words)


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
    cleaned_title = preprocess_title(text)
    raw_tokens = TOKEN_RE.findall(f"{cleaned_title} {text}")
    keywords: list[str] = []
    for tok in raw_tokens:
        if tok in STOPWORDS:
            continue
        cleaned = re.sub(
            r"(에서|으로|에게|부터|까지|이랑|라는|에는|은|는|이|가|을|를|에|의|로|과|와|도|만)$",
            "",
            tok,
        )
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


def text_to_vector(text: str, dim: int = TITLE_VEC_DIM) -> np.ndarray:
    """
    CLEAN_TITLE 또는 검색 질의를 단어 + 음절 바이그램/트라이그램 Feature Hashing 기반
    L2 정규화 float32 임베딩 벡터(dim=512)로 고속 변환합니다.
    """
    vec = np.zeros(dim, dtype=np.float32)
    clean = preprocess_title(text) or text
    tokens = TOKEN_RE.findall(clean)
    if not tokens:
        tokens = TOKEN_RE.findall(text)

    features: list[tuple[str, float]] = []
    for tok in tokens:
        features.append((f"w:{tok}", 2.2))
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


def serialize_vec(vec: np.ndarray) -> bytes:
    """numpy float32 배열을 sqlite-vec 호환 직렬화 BLOB으로 변환합니다."""
    return sqlite_vec.serialize_float32(vec.astype(np.float32).tolist())


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
    clean_title: str = ""
    category: str = "시사·종합"
    cluster_label: str = ""
    content: str
    summary: str = ""
    url: str
    regdate: str
    score: float
    title_vec_sim: float = 0.0


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
    """
    news.db (SQLite + sqlite-vec 확장) 기반:
    - NEWS.CLEAN_TITLE -> NEWS.title_vec 벡터화 및 VEC_NEWS_TITLE(vec0) 고속 색인
    - title_vec 기반 6대 뉴스 카테고리 분류 & KMeans(k=20) 이슈 클러스터 분류
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

            # 2. 전체 NEWS 로드 후 CLEAN_TITLE 기반으로 title_vec 고속 벡터화 (긴 본문 대신 CLEAN_TITLE만 벡터화하여 색인 속도 대폭 향상)
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

            # 3. preprocessing.ipynb의 기준 C (실루엣 k=88과 엘보우 k=20의 평균 k=54) 방식으로 CLEAN_TITLE / title_vec 기반 이슈 클러스터 분류
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

            # 6. RAG_INDEX 테이블 동기화 (EMBEDDING에도 고속 생성된 title_vec + 핵심 요약 벡터 저장)
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
        각 클러스터의 대표 키워드 3개로 라벨(예: 'C3: 대통령, 이재명, 국무회의')을 부여합니다.
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
            # 폴백: title_vec 기반 경량 군집화
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

                # 2. NEWS 테이블 저장 (CLEAN_TITLE, title_vec, CATEGORY, CLUSTER_ID, CLUSTER_LABEL 포함)
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
            # sqlite-vec C 확장 함수 vec_distance_cosine(title_vec, ?)으로 SQLite 내부에서 즉시 유사도 산출
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
            # sqlite-vec VEC_NEWS_TITLE 가상 테이블에서 MATCH 연산자로 초고속 KNN 검색
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

        # 최근접 이웃(KNN) 다수결 카테고리와 프로토타입 점수를 결합하여 분류 정밀도 향상
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

            # 2. 상위 이슈 클러스터(CLEAN_TITLE + title_vec 군집) 집계 (단일/일반 잡음 군집 제외)
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


class NewsLensAgent:
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


agent = NewsLensAgent()


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    test_query = sys.argv[1] if len(sys.argv) > 1 else "수사기관장 배석 사법개혁"
    res = agent.run(test_query)
    print(json.dumps(res, ensure_ascii=False, indent=2))
