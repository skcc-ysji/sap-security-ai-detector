"""SAP 위험도 탐지 시스템의 메인 클래스"""

import pandas as pd
import numpy as np
import numpy as np
from typing import Dict, Any
import pickle
from datetime import datetime

from .preprocessor import TextPreprocessor
from .feature_extractors import SAPFeatureExtractor
from .data_generator import DataGenerator
from .model_trainer import ModelTrainer
from .constants import RISK_LEVEL_MAPPING

class SAPRiskDetector:
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.feature_extractor = SAPFeatureExtractor()
        self.data_generator = DataGenerator()
        self.model_trainer = ModelTrainer()
        
        # 학습된 모델과 벡터화 도구들
        self.models = {}
        self.vectorizers = {}
        self.scaler = None
        self.best_model = None
        
        # 성능 모니터링
        self.performance_history = []
        self.confidence_threshold = 0.6
        self.prediction_count = 0
        self.high_confidence_count = 0
    
    def _extract_features_for_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """데이터프레임에 대한 특성 추출"""
        # 텍스트 전처리
        df['processed_text'] = df['text'].apply(self.preprocessor.preprocess_text)
        
        # 고정된 특성 목록 정의 (학습 시와 예측 시 일관성 보장)
        expected_features = self._get_expected_features()
        
        # 모든 특성 컬럼을 0으로 초기화
        for feature_name, default_value in expected_features.items():
            df[feature_name] = default_value
        
        # 각 샘플에 대해 특성 추출
        for idx, row in df.iterrows():
            basic_features = self.preprocessor.extract_basic_features(row['text'], row['processed_text'])
            sap_features = self.feature_extractor.extract_all_features(row['text'], row['processed_text'])
            
            # 특성 결합
            all_features = {**basic_features, **sap_features}
            
            # 예상된 특성만 설정
            for feature_name in expected_features.keys():
                if feature_name in all_features:
                    value = all_features[feature_name]
                    # boolean 값을 int로 변환
                    if isinstance(value, bool):
                        value = int(value)
                    df.at[idx, feature_name] = value
        
        return df
    
    def train(self, n_samples: int = 3000) -> Dict[str, Any]:
        """모델 학습"""
        print("=== SAP 위험도 탐지 시스템 학습 시작 ===")
        
        # 1. 학습 데이터 생성
        print("\n1. 학습 데이터 생성 중...")
        df = self.data_generator.generate_training_data(n_samples)
        print(f"생성된 데이터 분포:")
        print(df['risk_level'].value_counts())
        
        # 2. 특성 추출
        print("\n2. 특성 추출 중...")
        df = self._extract_features_for_df(df)
        
        # 4. 모델 학습
        print("\n4. 모델 학습 중...")
        results = self.model_trainer.train(df)
        
        # 5. 학습된 모델과 도구들 저장
        self.models = self.model_trainer.models
        self.vectorizers = self.model_trainer.vectorizers
        self.scaler = self.model_trainer.scaler
        self.best_model = self.model_trainer.best_model
        
        print("\n=== 학습 완료 ===")
        print(f"최고 성능 모델: {results['best_model']}")
        print(f"정확도: {results['accuracy']:.4f}")
        print(f"F1 Score: {results['f1_score']:.4f}")
        
        return results
    
    def predict(self, text: str, confidence_threshold: float = 0.6) -> Dict[str, Any]:
        """위험도 예측 (항상 명확한 결과 반환)"""
        if not self.best_model:
            raise ValueError("모델이 학습되지 않았습니다.")
        
        try:
            # 1. 임시 DataFrame 생성 및 특성 추출
            temp_df = pd.DataFrame({'text': [text]})
            temp_df = self._extract_features_for_df(temp_df)
            
            # 2. 특성 벡터화
            X_text = pd.Series([temp_df['processed_text'].iloc[0]])
            X_numeric = temp_df.select_dtypes(include=[np.number])
            
            # 벡터화 모델 확인
            if not hasattr(self.model_trainer, 'vectorizers') or not self.model_trainer.vectorizers:
                if not self.vectorizers:
                    raise ValueError("벡터화 모델이 초기화되지 않았습니다. 모델을 다시 로드하거나 학습해주세요.")
                else:
                    self.model_trainer.vectorizers = self.vectorizers
                    self.model_trainer.scaler = self.scaler
            
            X_combined = self.model_trainer.prepare_features(
                X_text, X_numeric, is_training=False
            )
            
            # 3. 예측
            prediction = self.best_model.predict(X_combined)[0]
            probabilities = self.best_model.predict_proba(X_combined)[0]
            
            # 4. 신뢰도 계산
            confidence = max(probabilities)
            
            # 5. 항상 명확한 위험도 반환 (uncertain 제거)
            risk_level_mapping = {v: k for k, v in RISK_LEVEL_MAPPING.items()}
            risk_level = risk_level_mapping[prediction]
            
            prob_dict = {
                risk_level_mapping[i]: prob for i, prob in enumerate(probabilities)
            }
            
            # 6. 특성 분석 추출
            analysis = {
                'sap_transaction_count': temp_df['sap_transaction_count'].iloc[0],
                'sap_table_count': temp_df['sap_table_count'].iloc[0],
                'injection_pattern_count': temp_df['injection_pattern_count'].iloc[0],
                'role_impersonation_count': temp_df['role_impersonation_count'].iloc[0],
                'sensitive_data_access_count': temp_df['sensitive_data_access_count'].iloc[0],
                'context_specific_pattern_count': temp_df['context_specific_pattern_count'].iloc[0],
                'risk_intensity_score': temp_df['risk_intensity_score'].iloc[0],
                'technical_complexity_count': temp_df['technical_complexity_count'].iloc[0],
                'text_complexity': temp_df['word_count'].iloc[0],
                'language_mix': bool(temp_df['mixed_language'].iloc[0]),
                'prediction_confidence': confidence,
                'uncertainty_level': 'high' if confidence < confidence_threshold else 'low'
            }

            # 성능 통계 업데이트
            self.update_performance_stats(confidence, risk_level)

            return {
                'predicted_risk': risk_level,
                'confidence': confidence,
                'probabilities': prob_dict,
                'detailed_analysis': analysis,
                'processed_text': temp_df['processed_text'].iloc[0],
                'requires_manual_review': confidence < confidence_threshold
            }
            
        except ValueError as ve:
            raise ve
        except Exception as e:
            raise ValueError(f"예측 과정에서 오류 발생: {str(e)}")
    
    def save_model(self, filepath: str) -> None:
        """모델 저장"""
        model_data = {
            'models': self.models,
            'vectorizers': self.vectorizers,
            'scaler': self.scaler,
            'best_model': self.best_model,
            'feature_names': list(self._get_expected_features().keys()),  # 특성 목록 저장
            'version': '2.0',
            'created_at': datetime.now().isoformat()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"모델이 {filepath}에 저장되었습니다.")
    
    def _get_expected_features(self):
        """예상되는 특성 목록 반환"""
        return {
            # 기본 특성
            'text_length': 0,
            'word_count': 0,
            'sentence_count': 0,
            'avg_word_length': 0.0,
            'special_char_ratio': 0.0,
            'uppercase_ratio': 0.0,
            'is_korean': 0,
            'is_english': 0,
            'mixed_language': 0,
            
            # SAP 특성
            'sap_transaction_count': 0,
            'sap_table_count': 0,
            'injection_pattern_count': 0,
            'role_impersonation_count': 0,
            'sensitive_data_access_count': 0,
            'context_specific_pattern_count': 0,
            'risk_intensity_score': 0.0,
            'technical_complexity_count': 0,
            
            # 키워드 특성
            'critical_korean_count': 0,
            'critical_english_count': 0,
            'critical_sap_specific_count': 0,
            'high_korean_count': 0,
            'high_english_count': 0,
            'high_sap_specific_count': 0,
            'medium_korean_count': 0,
            'medium_english_count': 0,
            'medium_sap_specific_count': 0,
            'low_korean_count': 0,
            'low_english_count': 0,
            'low_sap_specific_count': 0
        }
    
    def load_model(self, filepath: str) -> None:
        """모델 로드"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            # 모델 데이터 복원
            self.models = model_data['models']
            self.vectorizers = model_data['vectorizers']
            self.scaler = model_data['scaler']
            self.best_model = model_data['best_model']
            
            # model_trainer 객체에 벡터화 모델 복원
            self.model_trainer.vectorizers = self.vectorizers
            self.model_trainer.scaler = self.scaler
            self.model_trainer.models = self.models
            self.model_trainer.best_model = self.best_model
            
            print(f"모델이 {filepath}에서 로드되었습니다.")
            print(f"모델 버전: {model_data.get('version', 'unknown')}")
            
            # 벡터화 모델 상태 확인
            if not self.vectorizers:
                raise ValueError("벡터화 모델이 로드되지 않았습니다.")
            
            print("벡터화 모델 로드 완료")
            
        except Exception as e:
            raise ValueError(f"모델 로드 실패: {str(e)}")

    def update_performance_stats(self, confidence: float, risk_level: str):
        """성능 통계 업데이트"""
        self.prediction_count += 1
        
        if confidence >= self.confidence_threshold:
            self.high_confidence_count += 1
        
        # 성능 히스토리 업데이트
        self.performance_history.append({
            'timestamp': datetime.now(),
            'confidence': confidence,
            'risk_level': risk_level,
            'high_confidence': confidence >= self.confidence_threshold
        })
        
        # 최근 100개만 유지
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """성능 통계 반환"""
        if self.prediction_count == 0:
            return {
                'total_predictions': 0,
                'high_confidence_rate': 0.0,
                'average_confidence': 0.0,
                'confidence_distribution': {}
            }
        
        high_confidence_rate = self.high_confidence_count / self.prediction_count
        avg_confidence = sum([p['confidence'] for p in self.performance_history]) / len(self.performance_history)
        
        # 신뢰도 분포 계산
        confidence_ranges = {
            '0.0-0.2': 0, '0.2-0.4': 0, '0.4-0.6': 0,
            '0.6-0.8': 0, '0.8-1.0': 0
        }
        
        for p in self.performance_history:
            conf = p['confidence']
            if conf < 0.2:
                confidence_ranges['0.0-0.2'] += 1
            elif conf < 0.4:
                confidence_ranges['0.2-0.4'] += 1
            elif conf < 0.6:
                confidence_ranges['0.4-0.6'] += 1
            elif conf < 0.8:
                confidence_ranges['0.6-0.8'] += 1
            else:
                confidence_ranges['0.8-1.0'] += 1
        
        return {
            'total_predictions': self.prediction_count,
            'high_confidence_rate': high_confidence_rate,
            'average_confidence': avg_confidence,
            'confidence_distribution': confidence_ranges
        }
