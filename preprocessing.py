"""
preprocessing.py - 뉴스 제목·본문 전처리, 한자 의미 복원, 개인정보 마스킹 및 키워드 추출 모듈
"""

from __future__ import annotations

import html
import re

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


# news_classifier.ipynb와 동일한 추가 불용어(기자 바이라인·무단전재·앵커 등)
CLASSIFIER_EXTRA_STOPWORDS = {
    "무단", "전재", "배포", "금지", "저작권자", "연합뉴스", "뉴시스", "뉴스1", "앵커", "리포트",
}


def preprocess_korean_news_for_classifier(text: str, max_chars: int | None = None) -> str:
    """
    news_classifier.ipynb와 동일한 한국어 뉴스 전처리(한자 의미 복원·상투어·조사 제거)를 수행합니다.
    공통 전처리 규칙(CONTEXTUAL_HANJA_PATTERNS, SEMANTIC_HANJA_DICT, PHONETIC_HANJA_TRANS, STOPWORDS)을 재사용합니다.
    """
    if not isinstance(text, str):
        return ""
    if max_chars is not None:
        text = text[:max_chars]

    text = re.sub(r"\[[^\]]*기자[^\]]*\]", " ", text)
    cleaned = preprocess_title(text)
    if not cleaned:
        return ""
    return " ".join(w for w in cleaned.split() if w not in CLASSIFIER_EXTRA_STOPWORDS)


def clean_article_content(content: str, max_chars: int = 1600) -> str:
    """
    NEWS.CONTENT에서 기자 이메일, 연락처, 저작권 문구, 과도한 공백을 제거하고
    4GB VRAM 컨텍스트(2048 토큰)에 맞게 핵심 본문 길이를 조절합니다.
    """
    if not content:
        return ""
    text = EMAIL_RE.sub("", content)
    text = PHONE_RE.sub("", text)
    # 불필요한 언론사 저작권/무단전재 문구 제거
    text = re.sub(r"무단\s*전재\s*(및\s*재배포\s*금지|금지).*$", "", text)
    text = re.sub(r"ⓒ\s*[^\n]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0] + "..."
    return text


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


__all__ = [
    "HTTP_HEADERS",
    "OID_TO_PRESS",
    "EMAIL_RE",
    "PHONE_RE",
    "TOKEN_RE",
    "STOPWORDS",
    "CONTEXTUAL_HANJA_PATTERNS",
    "SEMANTIC_HANJA_DICT",
    "PHONETIC_HANJA_TRANS",
    "preprocess_title",
    "CLASSIFIER_EXTRA_STOPWORDS",
    "preprocess_korean_news_for_classifier",
    "clean_article_content",
    "mask_sensitive_info",
    "extract_press_name",
    "extract_keywords",
    "parse_naver_article_html",
]
