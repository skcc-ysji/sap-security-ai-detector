# 🛡️ SAP 위험도 탐지 시스템 v0.1

SAP 시스템에서 보안 위험을 실시간으로 탐지하는 AI 기반 보안 관리 시스템입니다.

## 📋 목차

- [시스템 개요](#-시스템-개요)
- [주요 기능](#-주요-기능)
- [시스템 아키텍처](#-시스템-아키텍처)
- [설치 및 실행](#-설치-및-실행)
- [사용법](#-사용법)
- [기술적 상세](#-기술적-상세)
- [모델 학습 과정](#-모델-학습-과정)
- [위험도 계산 로직](#-위험도-계산-로직)
- [API 문서](#-api-문서)
- [트러블슈팅](#-트러블슈팅)
- [라이선스](#-라이선스)

## 🎯 시스템 개요

### 목적
SAP 시스템에서 발생할 수 있는 악성 프롬프트와 보안 위험을 실시간으로 탐지하여 시스템 보안을 강화합니다.

### 핵심 특징
- **AI 기반 탐지**: 머신러닝 모델을 통한 정확한 위험도 판단
- **실시간 분석**: 즉시 텍스트 분석 및 위험도 평가
- **배치 처리**: 대량의 텍스트를 한 번에 분석
- **시각화**: 직관적인 차트와 그래프로 결과 표시
- **SAP 특화**: SAP 시스템에 특화된 패턴 감지

## 🚀 주요 기능

### 1. 모델 관리
- **모델 로드**: 기존 학습된 모델 불러오기
- **새 모델 학습**: 새로운 데이터로 모델 재학습
- **모델 저장**: 학습된 모델을 파일로 저장
- **모델 상태 확인**: 현재 모델의 상태 및 성능 확인

### 2. 실시간 분석
- **단일 텍스트 분석**: 개별 텍스트의 위험도 평가
- **실시간 결과**: 즉시 위험도 및 신뢰도 표시
- **상세 분석**: 패턴별 상세 분석 결과 제공
- **시각화**: 게이지 차트, 확률 분포 등 시각적 표현

### 3. 배치 분석
- **파일 업로드**: CSV, TXT 파일을 통한 대량 분석
- **텍스트 직접 입력**: 여러 텍스트를 한 번에 분석
- **진행 상황 표시**: 분석 진행률 실시간 표시
- **결과 다운로드**: CSV 형태로 분석 결과 저장

### 4. 분석 설정
- **위험도 임계값**: 경고를 표시할 위험도 수준 설정
- **신뢰도 임계값**: 신뢰도 경고 기준 설정
- **분석 모드**: 단일/배치 분석 모드 선택
- **고급 설정**: 상세 분석 및 경고 기능 활성화/비활성화

## 🏗️ 시스템 아키텍처

```
📁 프로젝트 구조
├── 📄 main.py                    # 메인 실행 파일
├── 📁 models/                    # 학습된 모델 저장소
│   └── enhanced_sap_risk_model_v2.pkl
├── 📁 src/                       # 소스 코드
│   ├── __init__.py
│   ├── constants.py              # 상수 정의
│   ├── data_generator.py         # 데이터 생성기
│   ├── feature_extractors.py     # 특성 추출기
│   ├── model_trainer.py          # 모델 학습기
│   ├── preprocessor.py           # 전처리기
│   └── sap_risk_detector.py      # SAP 위험도 탐지기
├── 📁 venv/                      # 가상환경
└── 📄 README.md                  # 이 파일
```

### 핵심 모듈 설명

#### 1. `sap_risk_detector.py`
- **역할**: 메인 탐지 엔진
- **기능**: 
  - 텍스트 전처리
  - 특성 추출
  - 모델 예측
  - 결과 후처리

#### 2. `feature_extractors.py`
- **역할**: 텍스트에서 특성 추출
- **추출 특성**:
  - SAP 트랜잭션 패턴
  - 인젝션 패턴
  - 역할 사칭 패턴
  - 민감 정보 접근 패턴
  - 언어 혼합 감지
  - 기술적 복잡도

#### 3. `model_trainer.py`
- **역할**: 머신러닝 모델 학습
- **알고리즘**:
  - RandomForestClassifier
  - LogisticRegression
  - SVC (Support Vector Classifier)
  - MLPClassifier (Neural Network)
  - VotingClassifier (앙상블)

## 💻 설치 및 실행

### 1. 환경 요구사항
- Python 3.8 이상
- 8GB RAM 이상 권장
- Windows 10/11 또는 Linux

### 2. 설치 과정

```bash
# 1. 저장소 클론
git clone <repository-url>
cd AI_YT_FINAL_PJT

# 2. 가상환경 생성 및 활성화
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# 3. 의존성 설치
pip install -r requirements.txt

# 4. 모델 파일 확인
ls models/
```

### 3. 실행

```bash
# Streamlit 앱 실행
streamlit run app.py
```

### 4. 접속
브라우저에서 `http://localhost:8501` 접속

## 📖 사용법

### 1. 초기 설정

#### 모델 로드
1. **모델 관리** 탭 클릭
2. **모델 로드** 버튼 클릭
3. 로드 완료 메시지 확인

#### 새 모델 학습 (선택사항)
1. **새 모델 학습** 섹션에서 샘플 수 설정
2. **새 모델 학습** 버튼 클릭
3. 학습 완료 후 성능 지표 확인

### 2. 분석 설정

#### 기본 설정
1. **분석 설정** 탭 클릭
2. **위험도 임계값** 설정 (기본값: high)
3. **신뢰도 임계값** 설정 (기본값: 0.7)
4. **분석 모드** 선택
5. **설정 저장** 버튼 클릭

#### 고급 설정
- **상세 분석 활성화**: 패턴별 상세 분석 표시
- **신뢰도 경고 활성화**: 낮은 신뢰도 시 경고 표시

### 3. 실시간 분석

#### 단일 텍스트 분석
1. **실시간 분석** 탭 클릭
2. 분석할 텍스트 입력
3. **분석 시작** 버튼 클릭
4. 결과 확인:
   - 위험도 레벨
   - 신뢰도 점수
   - 확률 분포
   - 상세 분석 (활성화 시)

### 4. 배치 분석

#### 파일 업로드 방식
1. **배치 분석** 탭 클릭
2. **파일 업로드** 선택
3. CSV 또는 TXT 파일 업로드
4. **배치 분석 시작** 버튼 클릭
5. 결과 확인 및 다운로드

#### 텍스트 직접 입력 방식
1. **텍스트 직접 입력** 선택
2. 여러 텍스트를 줄바꿈으로 구분하여 입력
3. **배치 분석 시작** 버튼 클릭
4. 결과 확인 및 다운로드

## 🔬 기술적 상세

### 1. 특성 추출 과정

#### 텍스트 전처리
```python
# 1. 텍스트 정규화
text = text.lower().strip()

# 2. 특수문자 처리
text = re.sub(r'[^\w\s]', '', text)

# 3. 토큰화
tokens = text.split()
```

#### SAP 특화 특성
```python
# SAP 트랜잭션 패턴
sap_transactions = ['SU01', 'PFCG', 'VA01', 'MM01', 'FI01']

# 인젝션 패턴
injection_patterns = [
    'ignore previous', 'ignore security',
    'actually', 'pretend', 'role play'
]

# 역할 사칭 패턴
role_patterns = [
    'you are admin', 'you are ceo',
    'act as administrator', 'pretend to be'
]
```

### 2. 모델 아키텍처

#### 앙상블 모델 구성
```python
# 개별 모델들
rf_model = RandomForestClassifier(n_estimators=100)
lr_model = LogisticRegression(max_iter=1000)
svc_model = SVC(probability=True)
mlp_model = MLPClassifier(hidden_layer_sizes=(100, 50))

# 앙상블 모델
ensemble = VotingClassifier(
    estimators=[
        ('rf', rf_model),
        ('lr', lr_model),
        ('svc', svc_model),
        ('mlp', mlp_model)
    ],
    voting='soft'
)
```

### 3. 특성 벡터화

#### TF-IDF 벡터화
```python
# 단어 레벨 TF-IDF
word_vectorizer = TfidfVectorizer(
    max_features=1000,
    ngram_range=(1, 2)
)

# 문자 레벨 TF-IDF
char_vectorizer = TfidfVectorizer(
    max_features=500,
    ngram_range=(3, 5),
    analyzer='char'
)
```

## 🎓 모델 학습 과정

### 1. 데이터 생성

#### 합성 데이터 생성
```python
def generate_synthetic_data(n_samples=3000):
    # 안전한 텍스트 (70%)
    safe_texts = generate_safe_sap_texts(int(n_samples * 0.7))
    
    # 위험한 텍스트 (30%)
    dangerous_texts = generate_dangerous_sap_texts(int(n_samples * 0.3))
    
    return safe_texts + dangerous_texts
```

#### 텍스트 카테고리
- **안전한 텍스트**: 일반적인 SAP 사용법 문의
- **위험한 텍스트**: 권한 상승, 민감 정보 접근 시도

### 2. 특성 엔지니어링

#### 기본 특성
- 텍스트 길이
- 단어 수
- 특수문자 비율
- 대문자 비율

#### SAP 특화 특성
- SAP 트랜잭션 감지 수
- 인젝션 패턴 감지 수
- 역할 사칭 패턴 감지 수
- 민감 정보 접근 패턴 감지 수

#### 고급 특성
- 언어 혼합 감지
- 기술적 복잡도 점수
- 컨텍스트 특화 패턴
- 위험 강도 점수

### 3. 모델 학습

#### 데이터 분할
```python
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, random_state=42
)
```

#### 클래스 불균형 처리
```python
# SMOTE를 통한 오버샘플링
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
```

#### 하이퍼파라미터 튜닝
```python
# Grid Search를 통한 최적 파라미터 탐색
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}
```

### 4. 모델 평가

#### 평가 지표
- **정확도 (Accuracy)**: 전체 예측 중 올바른 예측 비율
- **F1 Score**: 정밀도와 재현율의 조화평균
- **혼동 행렬**: 실제 vs 예측 클래스 비교

#### 교차 검증
```python
# 5-fold 교차 검증
cv_scores = cross_val_score(model, X, y, cv=5)
```

## ⚖️ 위험도 계산 로직

### 1. 위험도 레벨 분류

#### 4단계 위험도 체계
```python
RISK_LEVELS = {
    'low': 1,      # 낮은 위험
    'medium': 2,   # 중간 위험
    'high': 3,     # 높은 위험
    'critical': 4  # 치명적 위험
}
```

### 2. 위험도 판단 기준

#### 패턴 기반 점수
```python
def calculate_risk_score(features):
    score = 0
    
    # SAP 트랜잭션 패턴 (가중치: 0.3)
    score += features['sap_transaction_count'] * 0.3
    
    # 인젝션 패턴 (가중치: 0.4)
    score += features['injection_pattern_count'] * 0.4
    
    # 역할 사칭 패턴 (가중치: 0.2)
    score += features['role_impersonation_count'] * 0.2
    
    # 민감 정보 접근 (가중치: 0.1)
    score += features['sensitive_data_access_count'] * 0.1
    
    return score
```

#### 위험도 매핑
```python
def map_risk_level(score):
    if score < 0.3:
        return 'low'
    elif score < 0.6:
        return 'medium'
    elif score < 0.8:
        return 'high'
    else:
        return 'critical'
```

### 3. 신뢰도 계산

#### 모델 신뢰도
```python
def calculate_confidence(predictions):
    # 예측 확률의 최대값을 신뢰도로 사용
    confidence = np.max(predictions)
    return confidence
```

#### 신뢰도 해석
- **0.9 이상**: 매우 높은 신뢰도
- **0.7-0.9**: 높은 신뢰도
- **0.5-0.7**: 중간 신뢰도
- **0.5 미만**: 낮은 신뢰도

## 📚 API 문서

### 1. SAPRiskDetector 클래스

#### 초기화
```python
detector = SAPRiskDetector()
```

#### 메서드

##### `load_model(model_path)`
- **목적**: 저장된 모델 로드
- **매개변수**: `model_path` (str) - 모델 파일 경로
- **반환값**: bool - 로드 성공 여부

##### `predict(text)`
- **목적**: 텍스트 위험도 예측
- **매개변수**: `text` (str) - 분석할 텍스트
- **반환값**: dict - 예측 결과

##### `train_model(n_samples=3000)`
- **목적**: 새 모델 학습
- **매개변수**: `n_samples` (int) - 학습 샘플 수
- **반환값**: dict - 학습 결과

### 2. 예측 결과 구조

```python
{
    'predicted_risk': 'high',
    'confidence': 0.85,
    'probabilities': {
        'low': 0.05,
        'medium': 0.10,
        'high': 0.85,
        'critical': 0.00
    },
    'detailed_analysis': {
        'sap_transaction_count': 2,
        'injection_pattern_count': 1,
        'role_impersonation_count': 0,
        'sensitive_data_access_count': 1,
        'text_complexity': 0.7,
        'language_mix': False
    }
}
```

## 🔧 트러블슈팅

### 1. 일반적인 문제

#### 모델 로드 실패
**증상**: "모델 로드 실패" 메시지
**해결책**:
1. `models/` 폴더에 모델 파일 존재 확인
2. 모델 파일 권한 확인
3. 새 모델 학습 후 재시도

#### 특성 벡터화 오류
**증상**: "특성 벡터화 실패" 메시지
**해결책**:
1. 모델 재학습
2. 특성 추출기 업데이트
3. 캐시 삭제 후 재시도

#### 메모리 부족
**증상**: 분석 중 메모리 오류
**해결책**:
1. 배치 크기 줄이기
2. 불필요한 프로세스 종료
3. 시스템 메모리 확장

### 2. 성능 최적화

#### 분석 속도 향상
- 모델 캐싱 활성화
- 배치 크기 조정
- 불필요한 특성 제거

#### 정확도 향상
- 더 많은 학습 데이터 사용
- 특성 엔지니어링 개선
- 하이퍼파라미터 튜닝

### 3. 로그 확인

#### Streamlit 로그
```bash
# 로그 레벨 설정
export STREAMLIT_LOG_LEVEL=debug
```

#### Python 로그
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 🤝 기여하기

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📞 지원

- **이슈 리포트**: GitHub Issues 사용
- **문의사항**: 프로젝트 관리자에게 연락
- **문서**: 이 README 파일 참조

---

**개발자**: AI 팀  
**버전**: v0.1  
**최종 업데이트**: 2024년 12월 