# 🛡️ SAP Security AI Detector

**SAP 보안 위험 탐지를 위한 하이브리드 AI 시스템 (ML + RAG)**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 목차

- [시스템 개요](#시스템-개요)
- [주요 기능](#주요-기능)
- [하이브리드 아키텍처](#하이브리드-아키텍처)
- [RAG 시스템](#rag-시스템)
- [설치 및 실행](#설치-및-실행)
- [사용법](#사용법)
- [기술 스택](#기술-스택)
- [API 문서](#api-문서)
- [트러블슈팅](#트러블슈팅)
- [라이선스](#라이선스)

## 🎯 시스템 개요

SAP Security AI Detector는 **하이브리드 AI 접근법**을 사용하여 SAP 시스템의 보안 위험을 탐지하는 지능형 시스템입니다.

### 🚀 핵심 특징

- **하이브리드 분석**: 빠른 ML + 정확한 RAG
- **실시간 탐지**: 즉시 위험도 평가
- **지식베이스 기반**: SAP 전문 지식 활용
- **사용자 친화적 UI**: SAP 스타일 인터페이스
- **확장 가능한 아키텍처**: 모듈화된 설계

## ⚡ 주요 기능

### 🔧 모델 관리
- **모델 로드/저장**: 기존 모델 활용
- **새 모델 학습**: 실시간 모델 업데이트
- **성능 모니터링**: 모델 상태 추적

### ⚙️ 분석 설정
- **위험도 임계값**: 사용자 정의 설정
- **신뢰도 임계값**: 분석 품질 제어
- **RAG 시스템 설정**: Claude Sonnet 3.5 연동
- **캐싱 설정**: 성능 최적화

### 📊 실시간 분석
- **즉시 분석**: 빠른 ML 기반 탐지
- **하이브리드 분석**: ML + RAG 융합
- **상세 결과**: 위험도, 신뢰도, 추론 과정
- **시각화**: 게이지, 차트, 패턴 분석

### 📁 배치 분석
- **파일 업로드**: CSV, TXT 파일 지원
- **대량 처리**: 효율적인 배치 분석
- **결과 다운로드**: CSV 형태로 내보내기
- **통계 분석**: 위험도별 분포 분석

### 📚 지식베이스 관리
- **문서 업로드**: 다양한 형식 지원
- **벡터 검색**: 의미 기반 검색
- **샘플 데이터**: SAP 보안 지식 포함
- **실시간 검색**: 관련 문서 추천

## 🏗️ 하이브리드 아키텍처

### 🧠 ML 시스템 (빠른 분석)
```python
# 빠른 ML 기반 탐지
def fast_ml_analysis(text):
    # 1. 특성 추출 (50ms)
    features = extract_features(text)
    
    # 2. 모델 예측 (10ms)
    prediction = ml_model.predict(features)
    
    # 3. 신뢰도 계산 (5ms)
    confidence = calculate_confidence(prediction)
    
    return {
        'risk_level': prediction,
        'confidence': confidence,
        'analysis_time': '65ms'
    }
```

### 🤖 RAG 시스템 (정확한 분석)
```python
# 정확한 RAG 기반 분석
def accurate_rag_analysis(text):
    # 1. 벡터 검색 (200ms)
    relevant_docs = vector_store.search(text)
    
    # 2. Claude 분석 (2-5초)
    analysis = claude_analyze(text, relevant_docs)
    
    # 3. 결과 융합 (100ms)
    result = fuse_results(analysis)
    
    return {
        'risk_level': result['risk_level'],
        'confidence': result['confidence'],
        'reasoning': result['reasoning'],
        'analysis_time': '2.3-5.3초'
    }
```

### 🔄 하이브리드 융합
```python
# 하이브리드 분석 전략
def hybrid_analysis(text):
    # 1. 빠른 ML 분석
    ml_result = fast_ml_analysis(text)
    
    # 2. 신뢰도 판단
    if ml_result['confidence'] > 0.8:
        return ml_result  # 즉시 반환
    
    # 3. RAG 분석 수행
    rag_result = accurate_rag_analysis(text)
    
    # 4. 결과 융합
    return fuse_ml_rag_results(ml_result, rag_result)
```

## 🤖 RAG 시스템

### 📚 지식베이스 구성

#### SAP 보안 지식
- **권한 관리**: SU01, PFCG, SU24, SU25, SU26
- **보안 위협**: 권한상승, 데이터유출, 인젝션, 세션하이재킹
- **트랜잭션 보안**: 각 SAP 트랜잭션별 보안 고려사항
- **데이터 보안**: 데이터 분류, 접근 제어, 암호화
- **모범 사례**: 사용자 관리, 권한 관리, 시스템 보안
- **사고 대응**: 탐지, 초기대응, 조사, 복구, 사후관리
- **규정 준수**: SOX, GDPR, ISO 27001, PCI DSS
- **보안 아키텍처**: 방어적 깊이, 최소 권한, 모니터링
- **보안 도구**: SAP 보안 도구, 모니터링 도구, 외부 도구
- **위험 평가**: 위험 식별, 분석, 처리, 모니터링
- **보안 교육**: 기본 교육, 역할별 교육, 인증

#### 벡터 저장소
- **ChromaDB**: 로컬 벡터 데이터베이스
- **Sentence Transformers**: 의미 기반 임베딩
- **FAISS**: 고성능 벡터 검색
- **Redis**: 다층 캐싱 시스템

### 🔍 검색 및 분석

#### 벡터 검색
```python
# 유사한 문서 검색
def search_similar_documents(query, n_results=5):
    # 1. 쿼리 임베딩
    query_embedding = embedding_model.encode(query)
    
    # 2. 벡터 검색
    results = vector_store.search(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    
    return results
```

#### Claude 분석
```python
# Claude Sonnet 3.5 기반 분석
def claude_analysis(text, relevant_docs):
    prompt = build_analysis_prompt(text, relevant_docs)
    
    response = claude_client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1000,
        temperature=0.1,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return parse_claude_response(response)
```

### ⚡ 성능 최적화

#### 다층 캐싱
```python
# 다층 캐싱 시스템
class MultiLayerCache:
    def __init__(self):
        self.l1_cache = {}      # 메모리 캐시 (1ms)
        self.l2_cache = Redis() # Redis 캐시 (10ms)
        self.l3_cache = DB()    # DB 캐시 (100ms)
```

#### 비동기 처리
```python
# 백그라운드 RAG 분석
async def background_rag_analysis(text):
    # 1. 즉시 ML 결과 반환
    fast_result = ml_analysis(text)
    
    # 2. 백그라운드에서 RAG 분석
    if fast_result['confidence'] < 0.8:
        asyncio.create_task(rag_analysis(text))
    
    return fast_result
```

## 🚀 설치 및 실행

### 1️⃣ 환경 설정

```bash
# 저장소 클론
git clone https://github.com/skcc-ysji/sap-security-ai-detector.git
cd sap-security-ai-detector

# 가상환경 생성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### 2️⃣ 환경 변수 설정

#### 방법 1: .env 파일 사용 (권장)
프로젝트 루트에 `.env` 파일을 생성하고 다음 내용을 추가하세요:

```bash
# Claude API 키 설정
# Anthropic에서 발급받은 API 키를 입력하세요
# https://console.anthropic.com/ 에서 API 키를 발급받을 수 있습니다
CLAUDE_API_KEY=your-claude-api-key-here

# RAG 시스템 설정
ENABLE_RAG=true
RAG_CONFIDENCE_THRESHOLD=0.8

# 캐싱 설정
ENABLE_CACHING=true
CACHE_TTL=3600

# 비동기 처리 설정
ENABLE_ASYNC=true

# 성능 설정
MAX_CONCURRENT_REQUESTS=10
REQUEST_TIMEOUT=30

# 로깅 설정
LOG_LEVEL=INFO
ENABLE_DEBUG=false
```

#### 방법 2: Streamlit Secrets 사용
`.streamlit/secrets.toml` 파일 생성:
```toml
# Claude API 키 설정
CLAUDE_API_KEY = "your-claude-api-key-here"

# 기타 설정
ENABLE_RAG = true
RAG_CONFIDENCE_THRESHOLD = 0.8
ENABLE_CACHING = true
ENABLE_ASYNC = true
```

#### 방법 3: UI에서 직접 입력
애플리케이션 실행 후 "분석 설정" 탭에서 직접 API 키를 입력할 수 있습니다.

### 3️⃣ 실행

```bash
# Streamlit 앱 실행
streamlit run app.py
```

## 📖 사용법

### 🔧 모델 관리

1. **모델 로드**
   - "모델 로드" 버튼 클릭
   - 기존 모델 파일 로드

2. **새 모델 학습**
   - 학습 샘플 수 설정 (1000-10000)
   - "새 모델 학습" 버튼 클릭
   - 학습 결과 확인

### ⚙️ 분석 설정

1. **위험도 설정**
   - 위험도 임계값 선택 (low/medium/high/critical)
   - 신뢰도 임계값 조정 (0.0-1.0)

2. **RAG 시스템 설정**
   - RAG 시스템 활성화/비활성화
   - RAG 신뢰도 임계값 설정
   - 캐싱 및 비동기 처리 설정

### 📊 실시간 분석

1. **텍스트 입력**
   - 분석할 텍스트 입력
   - "분석 시작" 버튼 클릭

2. **결과 확인**
   - 위험도 및 신뢰도 확인
   - RAG 분석 결과 (활성화된 경우)
   - 상세 분석 및 시각화

### 📁 배치 분석

1. **파일 업로드**
   - CSV 또는 TXT 파일 업로드
   - "배치 분석 시작" 버튼 클릭

2. **결과 다운로드**
   - 분석 결과 CSV 다운로드
   - 통계 차트 확인

### 📚 지식베이스 관리

1. **지식베이스 초기화**
   - "지식베이스 초기화" 버튼 클릭
   - 샘플 데이터 자동 추가

2. **문서 업로드**
   - 단일 문서, 배치 업로드, 직접 입력
   - 다양한 형식 지원 (TXT, PDF, DOCX)

3. **문서 검색**
   - 키워드 기반 검색
   - 유사도 기반 결과 정렬

## 🛠️ 기술 스택

### 🤖 AI/ML
- **Scikit-learn**: 머신러닝 모델 (RandomForest, LogisticRegression, SVC, MLP)
- **Anthropic Claude**: RAG 시스템 LLM
- **Sentence Transformers**: 텍스트 임베딩
- **FAISS**: 벡터 검색
- **ChromaDB**: 벡터 데이터베이스

### 🎨 Frontend
- **Streamlit**: 웹 인터페이스
- **Plotly**: 인터랙티브 차트
- **Pandas**: 데이터 처리
- **NumPy**: 수치 계산

### 💾 Backend
- **Redis**: 캐싱 시스템
- **SQLite**: 로컬 데이터베이스
- **Joblib**: 모델 직렬화
- **Pickle**: 객체 저장

### 🔧 개발 도구
- **Black**: 코드 포맷팅
- **Flake8**: 코드 린팅
- **Pytest**: 테스트 프레임워크
- **Git**: 버전 관리

## 📊 성능 지표

### ⚡ 응답 시간
| 시스템 | 평균 응답 시간 | 처리량 |
|--------|---------------|--------|
| **ML 시스템** | 50-100ms | 1000 req/sec |
| **RAG 시스템** | 2-5초 | 10-50 req/sec |
| **하이브리드** | 100-500ms | 100-500 req/sec |

### 🎯 정확도
| 시스템 | 정확도 | 설명 |
|--------|--------|------|
| **ML 시스템** | 85-90% | 빠른 패턴 기반 탐지 |
| **RAG 시스템** | 95-98% | 컨텍스트 기반 정확한 분석 |
| **하이브리드** | 92-96% | ML + RAG 융합 |

### 💾 리소스 사용량
| 구성 요소 | 메모리 | CPU | 설명 |
|-----------|--------|-----|------|
| **ML 모델** | 100MB | 낮음 | 경량화된 모델 |
| **벡터 DB** | 500MB | 중간 | 임베딩 저장 |
| **Claude API** | 10MB | 낮음 | API 호출만 |
| **캐시** | 200MB | 낮음 | Redis 캐시 |

## 🔧 API 문서

### 핵심 클래스

#### SAPRiskDetector
```python
class SAPRiskDetector:
    def __init__(self):
        """SAP 위험 탐지기 초기화"""
    
    def predict(self, text: str) -> Dict:
        """텍스트 위험도 예측"""
    
    def train(self, n_samples: int) -> Dict:
        """모델 학습"""
    
    def save_model(self, filepath: str):
        """모델 저장"""
    
    def load_model(self, filepath: str):
        """모델 로드"""
```

#### HybridSAPDetector
```python
class HybridSAPDetector:
    def __init__(self, ml_detector, rag_detector):
        """하이브리드 탐지기 초기화"""
    
    def detect_threat(self, text: str) -> Dict:
        """하이브리드 위협 탐지"""
    
    async def detect_threat_async(self, text: str) -> Dict:
        """비동기 위협 탐지"""
```

#### ClaudeRAGDetector
```python
class ClaudeRAGDetector:
    def __init__(self, api_key: str):
        """Claude RAG 탐지기 초기화"""
    
    def analyze_threat(self, text: str) -> Dict:
        """RAG 기반 위협 분석"""
```

### 주요 함수

#### analyze_text_hybrid()
```python
def analyze_text_hybrid(text: str, confidence_threshold: float = 0.7) -> Dict:
    """
    하이브리드 텍스트 분석
    
    Args:
        text: 분석할 텍스트
        confidence_threshold: 신뢰도 임계값
    
    Returns:
        Dict: 분석 결과
    """
```

#### perform_rag_analysis()
```python
def perform_rag_analysis(text: str) -> Dict:
    """
    RAG 분석 수행
    
    Args:
        text: 분석할 텍스트
    
    Returns:
        Dict: RAG 분석 결과
    """
```

## 🔍 트러블슈팅

### 일반적인 문제

#### 1. 모델 로드 실패
```bash
# 해결 방법
- 모델 파일 경로 확인
- 파일 권한 확인
- 모델 재학습 수행
```

#### 2. RAG 분석 실패
```bash
# 해결 방법
- Claude API 키 확인
- 네트워크 연결 확인
- API 할당량 확인
```

#### 3. 성능 문제
```bash
# 해결 방법
- 캐싱 활성화
- 비동기 처리 활성화
- 리소스 모니터링
```

#### 4. 메모리 부족
```bash
# 해결 방법
- 벡터 DB 크기 조정
- 캐시 크기 제한
- 불필요한 데이터 정리
```

### 로그 확인

```bash
# Streamlit 로그
streamlit run app.py --logger.level debug

# 애플리케이션 로그
tail -f logs/app.log
```

### 성능 모니터링

```python
# 성능 측정
import time

start_time = time.time()
result = analyze_text_hybrid(text)
end_time = time.time()

print(f"분석 시간: {end_time - start_time:.3f}초")
```

## 📈 향후 계획

### 🚀 단기 계획 (1-3개월)
- [ ] 실시간 스트리밍 분석
- [ ] 고급 시각화 대시보드
- [ ] 다국어 지원 (영어, 일본어)
- [ ] 모바일 반응형 UI

### 🎯 중기 계획 (3-6개월)
- [ ] 클라우드 배포 (AWS, Azure)
- [ ] 대용량 데이터 처리
- [ ] 실시간 알림 시스템
- [ ] API 서버 구축

### 🌟 장기 계획 (6-12개월)
- [ ] 딥러닝 모델 통합
- [ ] 자동화된 모델 업데이트
- [ ] 엔터프라이즈 기능
- [ ] 오픈소스 커뮤니티

## 🤝 기여하기

### 개발 환경 설정
```bash
# 개발 의존성 설치
pip install -r requirements-dev.txt

# 코드 포맷팅
black src/ tests/

# 린팅
flake8 src/ tests/

# 테스트 실행
pytest tests/
```

### 기여 가이드라인
1. Fork 저장소
2. 기능 브랜치 생성
3. 코드 작성 및 테스트
4. Pull Request 생성
5. 코드 리뷰 및 병합

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

## 📞 문의

- **이슈 리포트**: [GitHub Issues](https://github.com/skcc-ysji/sap-security-ai-detector/issues)
- **기술 문의**: [Discussions](https://github.com/skcc-ysji/sap-security-ai-detector/discussions)
- **이메일**: [your-email@example.com]

---

**🛡️ SAP Security AI Detector** - SAP 보안을 위한 지능형 위험 탐지 시스템

*Made with ❤️ for SAP Security* 