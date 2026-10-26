# 논점콩 - 뉴스 기사 비교 에이전트

### 1. powershell 켜기

### 2. git clone 하기
```bash
git clone https://github.com/Jangbk633/proj1week.git
```
### 3. frontend.py 실행하기
```bash
python frontend.py
```
### 4. 우아하게 종료하기
ctrl+c로 종료 후 powershell에서 가상환경 종료하기
```bash
deactivate
```
---
preprocessing.py  : 뉴스 제목 한자 복원·정제(preprocess_title), 본문 노이즈 제거(clean_article_content), 키워드·언론사명 추출 및 HTML 파싱
vectorize.py      : CLEAN_TITLE 512차원 Feature Hashing 임베딩(text_to_vector) 및 sqlite-vec 직렬화(serialize_vec)
rag.py            : NewsDBRAG 기반 sqlite-vec 하이브리드 RAG 검색, 6대 카테고리·KMeans 군집화 및 실시간 뉴스 크롤링(ensure_rag_index, retrieve 등)
summerizer.py     : 벡터 거리 기반 관점 A/B 이진 군집화 및 통합 헤드라인·양면 비교 요약 생성(_extractive_compare_articles)
langgraph.py      : LangGraph 기반 뉴스 검색·관점 비교·출처 검증 단독/연동 에이전트(NJKAgent, AgentState) 및 패키지 브릿지
backend.py        : OpenAI Head LLM 감독기(OpenAISupervisorHead), LangGraph 6-Node 멀티에이전트 오케스트레이터(MultiAgentNewsOrchestrator), ML 분류기 추론 및 기사 논조 심판 엔진
frontend.py       : userinterface.html 웹 UI 서빙, 12개 분석·분류·동기화 REST API(create_fastapi_app) 및 가상환경·브라우저 자동 런처(main)
app.py            : 배포 및 ASGI 서버(uvicorn app:app) 구동을 위해 frontend.py의 FastAPI 앱(app) 객체를 재노출하는 엔트리포인트 모듈
slm_server.py     : 로컬 GGUF 모델(Kanana-Nano 2.1B) 탐색·다운로드 및 4GB VRAM 최적화 llama-server.exe 자동 기동(start_optimized_llama_server)
summary_engine.py : LocalKananaSummarizer 기반 로컬 GPU 뉴스 요약 생성, SQLite WAL 모드 DB(NEWS/RAG_INDEX) 저장 및 Render 웹서비스 Push 동기화
summary.py        : 로컬 SLM 뉴스 요약 대시보드 UI, 단건/배치 요약 REST API, OpenAI 호환 프록시 서버(create_summary_app) 및 CLI 런처