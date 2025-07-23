"""SAP 위험도 탐지 모델 학습 모듈"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
import warnings

# sklearn deprecation 경고 억제
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# 최신 sklearn import 방식 (1.6.1+)
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
    from sklearn.ensemble import RandomForestClassifier, VotingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
    from sklearn.preprocessing import StandardScaler
except ImportError as e:
    print(f"sklearn import 오류: {e}")
    raise

try:
    from imblearn.over_sampling import SMOTE
except ImportError as e:
    print(f"imbalanced-learn import 오류: {e}")
    raise

from scipy.sparse import hstack, vstack, csr_matrix


class ModelTrainer:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.vectorizers = {}
        self.scaler = StandardScaler()
        self.models = {}
        self.best_model = None
        
    def create_vectorizers(self) -> None:
        """텍스트 벡터화 모델 생성"""
        # 기본 TF-IDF
        self.vectorizers['basic'] = TfidfVectorizer(
            max_features=3000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8,
            token_pattern=r'(?u)\b\w+\b|[가-힣]+',
            analyzer='word'
        )
        
        # 문자 레벨 TF-IDF
        self.vectorizers['char'] = TfidfVectorizer(
            max_features=1000,
            ngram_range=(2, 4),
            analyzer='char',
            min_df=3
        )
        
        # SAP 특화 벡터화
        from .feature_extractors import SAPFeatureExtractor
        self.vectorizers['sap'] = TfidfVectorizer(
            max_features=500,
            ngram_range=(1, 1),
            vocabulary=SAPFeatureExtractor.create_sap_vocabulary(),
            binary=True
        )
    
    def prepare_features(self, X_text: pd.Series, X_numeric: pd.DataFrame, 
                        is_training: bool = True) -> csr_matrix:
        """특성 벡터화 및 결합"""
        try:
            if is_training:
                # 학습 시 벡터화
                X_basic = self.vectorizers['basic'].fit_transform(X_text)
                X_char = self.vectorizers['char'].fit_transform(X_text)
                X_sap = self.vectorizers['sap'].fit_transform(X_text)
                X_numeric_scaled = self.scaler.fit_transform(X_numeric)
            else:
                # 예측 시 벡터화
                X_basic = self.vectorizers['basic'].transform(X_text)
                X_char = self.vectorizers['char'].transform(X_text)
                X_sap = self.vectorizers['sap'].transform(X_text)
                X_numeric_scaled = self.scaler.transform(X_numeric)
            
            # 벡터 결합
            combined_features = hstack([X_basic, X_char, X_sap, X_numeric_scaled])
            return combined_features
            
        except KeyError as ke:
            raise ValueError(f"벡터화 오류: {ke}. 벡터화 모델이 초기화되지 않았습니다.")
        except Exception as e:
            raise ValueError(f"특성 벡터화 실패: {str(e)}")
    
    def train_models(self, X: csr_matrix, y: np.ndarray) -> Dict[str, float]:
        """개별 모델 학습"""
        models = {
            'rf': RandomForestClassifier(
                n_estimators=500,        # 300 → 500 (더 많은 트리)
                max_depth=30,            # 25 → 30 (더 깊은 트리)
                min_samples_split=2,     # 3 → 2 (더 세밀한 분할)
                min_samples_leaf=1,
                class_weight='balanced',
                random_state=self.random_state,
                bootstrap=True,          # 부트스트랩 활성화
                oob_score=True          # Out-of-bag 점수 계산
            ),
            'lr': LogisticRegression(
                max_iter=2000,           # 1000 → 2000 (더 많은 반복)
                class_weight='balanced',
                random_state=self.random_state,
                solver='liblinear',
                C=1.0,                  # 정규화 강도
                penalty='l2'            # L2 정규화
            ),
            'svm': SVC(
                kernel='rbf',
                probability=True,
                class_weight='balanced',
                random_state=self.random_state,
                C=10.0,                 # 정규화 강도 증가
                gamma='scale',          # 커널 계수
                decision_function_shape='ovr'  # One-vs-Rest
            ),
            'mlp': MLPClassifier(
                hidden_layer_sizes=(200, 100, 50),  # (100, 50) → (200, 100, 50)
                max_iter=1000,          # 500 → 1000
                random_state=self.random_state,
                early_stopping=True,
                learning_rate='adaptive',  # 적응적 학습률
                alpha=0.001,            # L2 정규화
                activation='relu'        # 활성화 함수
            )
        }
        
        model_scores = {}
        for name, model in models.items():
            model.fit(X, y)
            self.models[name] = model
            model_scores[name] = model.score(X, y)
        
        return model_scores
    
    def create_ensemble(self, model_scores: Dict[str, float]) -> None:
        """앙상블 모델 생성"""
        # 더 엄격한 임계값 (0.7 → 0.75)
        voting_models = [(name, model) for name, model in self.models.items() 
                        if model_scores[name] > 0.75]
        
        if len(voting_models) >= 2:
            self.models['ensemble'] = VotingClassifier(
                estimators=voting_models,
                voting='soft',
                weights=[model_scores[name] for name, _ in voting_models]  # 가중치 추가
            )
    
    def evaluate_with_cross_validation(self, X: csr_matrix, y: np.ndarray) -> Dict[str, float]:
        """교차 검증을 통한 모델 평가"""
        cv_scores = {}
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        for name, model in self.models.items():
            # 5-fold 교차 검증
            scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
            cv_scores[name] = {
                'mean_accuracy': scores.mean(),
                'std_accuracy': scores.std(),
                'min_accuracy': scores.min(),
                'max_accuracy': scores.max()
            }
        
        return cv_scores
    
    def train(self, df: pd.DataFrame, test_size: float = 0.2) -> Dict[str, Any]:
        """전체 모델 학습 프로세스"""
        try:
            # 라벨 인코딩
            from .constants import RISK_LEVEL_MAPPING
            df['risk_numeric'] = df['risk_level'].map(RISK_LEVEL_MAPPING)
            
            # 특성 분리
            text_features = df['processed_text']
            numeric_features = df.select_dtypes(include=[np.number]).drop(columns=['risk_numeric'])
            
            # 데이터 분할
            X_text_train, X_text_test, X_num_train, X_num_test, y_train, y_test = train_test_split(
                text_features, numeric_features, df['risk_numeric'],
                test_size=test_size, random_state=self.random_state, stratify=df['risk_numeric']
            )
            
            # 벡터화 모델 생성 (중요!)
            self.create_vectorizers()
            
            # 벡터화 모델 확인
            if not self.vectorizers:
                raise ValueError("벡터화 모델 생성에 실패했습니다.")
            
            # 특성 준비
            X_train_combined = self.prepare_features(X_text_train, X_num_train, is_training=True)
            X_test_combined = self.prepare_features(X_text_test, X_num_test, is_training=False)
            
            # SMOTE 적용
            smote = SMOTE(random_state=self.random_state, k_neighbors=3)
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train_combined, y_train)
            
            # 모델 학습
            model_scores = self.train_models(X_train_balanced, y_train_balanced)
            
            # 앙상블 생성
            self.create_ensemble(model_scores)
            if 'ensemble' in self.models:
                self.models['ensemble'].fit(X_train_balanced, y_train_balanced)
                model_scores['ensemble'] = self.models['ensemble'].score(X_test_combined, y_test)
            
            # 최고 성능 모델 선택
            best_model_name = max(model_scores.items(), key=lambda x: x[1])[0]
            self.best_model = self.models[best_model_name]
            
            # 최종 평가
            y_pred = self.best_model.predict(X_test_combined)
            
            return {
                'best_model': best_model_name,
                'model_scores': model_scores,
                'accuracy': accuracy_score(y_test, y_pred),
                'f1_score': f1_score(y_test, y_pred, average='weighted'),
                'classification_report': classification_report(y_test, y_pred),
                'confusion_matrix': confusion_matrix(y_test, y_pred)
            }
            
        except Exception as e:
            raise ValueError(f"모델 학습 실패: {str(e)}")
