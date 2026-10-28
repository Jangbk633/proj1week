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
  
train_classifier_and_summarizer.html에서 모델 튜닝 과정 확인 가능
    
---
preprocessing.py  : 뉴스 제목 및 본문 전처리, 키워드·언론사명 추출 및 HTML 파싱  
vectorize.py      : 뉴스 제목 text_to_vector 임베딩 및 sqlite-vec 직렬화  
rag.py            : NewsDBRAG 기반 sqlite-vec 하이브리드 RAG 검색, 6대 카테고리·KMeans 군집화  
summerizer.py     : 벡터 거리 기반 관점 군집화 및 비교 요약 생성  
langgraph.py      : LangGraph 기반 뉴스 검색·관점 비교·출처 검증 및 패키지 브릿지  
backend.py        : Head LLM, LangGraph 6-Node 멀티에이전트 오케스트레이터, ML 분류기 추론 및 기사 논조 심판 엔진  
frontend.py       : userinterface.html 웹 UI 서빙  
app.py            : frontend.py의 FastAPI 앱 객체를 재노출하는 엔트리포인트  
slm_server.py     : 로컬 SLM 모델 탐색·다운로드 및 llama-server.exe 자동 기동  
summary_engine.py : LocalKananaSummarizer 기반 로컬 GPU 뉴스 요약 생성  
summary.py        : 로컬 SLM 뉴스 요약 대시보드 UI, 단건/배치 요약 REST API, OpenAI 호환 프록시 서버 및 CLI 런처  
---
.env파일에 로컬 SLM에 대한 내용이 없으면 요약기로 openai모델을 사용합니다
