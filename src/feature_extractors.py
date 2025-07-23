"""SAP 특화 특성 추출기"""

import re
from typing import Dict, List, Any
from .constants import (
    RISK_KEYWORDS, INJECTION_PATTERNS,
    SAP_TRANSACTIONS, SAP_TABLES, SAP_TERMS
)

class SAPFeatureExtractor:
    @staticmethod
    def count_sap_transactions(text: str) -> int:
        """SAP 트랜잭션 코드 개수 계산"""
        transactions = re.findall(r'\b[A-Z]{2,4}\d{1,3}[A-Z]?\b', text.upper())
        return len(set(transactions))
    
    @staticmethod
    def count_sap_tables(text: str) -> int:
        """SAP 테이블명 개수 계산"""
        tables = re.findall(r'\b[A-Z]{3,6}\d{0,3}\b', text.upper())
        return sum([1 for table in tables if table in SAP_TABLES])
    
    @staticmethod
    def count_injection_patterns(text: str) -> int:
        """프롬프트 인젝션 패턴 개수 계산"""
        count = 0
        for language, patterns in INJECTION_PATTERNS.items():
            for pattern in patterns:
                count += len(re.findall(pattern, text, re.IGNORECASE))
        return count
    
    @staticmethod
    def count_role_impersonation_patterns(text: str) -> int:
        """역할 사칭 패턴 개수 계산"""
        role_patterns = [
            r'너는\s*(ceo|사장|대표|이사|관리자|시스템관리자|admin|administrator)',
            r'나는\s*(ceo|사장|대표|이사|관리자|시스템관리자|admin|administrator)',
            r'당신은\s*(ceo|사장|대표|이사|관리자|시스템관리자|admin|administrator)',
            r'you\s+are\s+(ceo|president|director|manager|administrator|admin)',
            r'i\s+am\s+(ceo|president|director|manager|administrator|admin)',
            r'you\s+are\s+now\s+(ceo|president|director|manager|administrator|admin)'
        ]
        
        count = 0
        for pattern in role_patterns:
            count += len(re.findall(pattern, text, re.IGNORECASE))
        return count
    
    @staticmethod
    def count_sensitive_data_access_patterns(text: str) -> int:
        """민감 정보 접근 패턴 개수 계산"""
        sensitive_patterns = [
            r'(연봉|급여|봉급|월급|개인정보|민감정보)\s*(공개|노출|추출|다운로드)',
            r'(전체|모든|전 인원|전 직원|모든 사용자)\s*(연봉|급여|개인정보)',
            r'(비밀번호|패스워드|인증정보|로그인정보)\s*(공개|노출|추출)',
            r'(salary|payroll|personal\s+data|sensitive\s+data)\s*(expose|reveal|extract|download)',
            r'(all|everyone|all\s+users|all\s+employees)\s+(salary|payroll|personal\s+data)',
            r'(password|credential|authentication)\s+(expose|reveal|extract)'
        ]
        
        count = 0
        for pattern in sensitive_patterns:
            count += len(re.findall(pattern, text, re.IGNORECASE))
        return count
    
    @staticmethod
    def count_context_specific_patterns(text: str) -> int:
        """컨텍스트 특화 패턴 개수 계산"""
        context_patterns = [
            r'(production|prod)\s*(system|환경)',
            r'(test|개발)\s*(system|환경)',
            r'(emergency|긴급)\s*(access|접근)',
            r'(bypass|우회)\s*(security|보안)',
            r'(override|재정의)\s*(authorization|권한)',
            r'(debug|디버그)\s*(mode|모드)',
            r'(backdoor|백도어)',
            r'(exploit|익스플로잇)',
            r'(vulnerability|취약점)',
            r'(breach|침해)'
        ]
        
        count = 0
        for pattern in context_patterns:
            count += len(re.findall(pattern, text, re.IGNORECASE))
        return count
    
    @staticmethod
    def calculate_risk_intensity_score(text: str) -> float:
        """위험도 강도 점수 계산"""
        # 위험 키워드별 가중치
        risk_weights = {
            'critical': 3.0,
            'high': 2.0,
            'medium': 1.0,
            'low': 0.5
        }
        
        total_score = 0.0
        total_keywords = 0
        
        for risk_level, categories in RISK_KEYWORDS.items():
            weight = risk_weights[risk_level]
            for category, keywords in categories.items():
                for keyword in keywords:
                    if keyword.lower() in text.lower():
                        total_score += weight
                        total_keywords += 1
        
        return total_score / max(total_keywords, 1)  # 평균 위험도
    
    @staticmethod
    def count_technical_complexity(text: str) -> int:
        """기술적 복잡도 계산"""
        complexity_indicators = [
            r'\b[A-Z]{2,4}\d{1,3}[A-Z]?\b',  # SAP 트랜잭션
            r'\b[A-Z]{3,6}\d{0,3}\b',         # SAP 테이블
            r'(authorization|권한)\s+(object|객체)',
            r'(profile|프로파일)\s+(parameter|파라미터)',
            r'(RFC|BAPI|ABAP)',
            r'(client|클라이언트)\s+\d{3}',
            r'(mandt|MANDT)',
            r'(transport|트랜스포트)',
            r'(customizing|커스터마이징)',
            r'(user\s+exit|사용자\s+출구)'
        ]
        
        count = 0
        for pattern in complexity_indicators:
            count += len(re.findall(pattern, text, re.IGNORECASE))
        return count
    
    @staticmethod
    def extract_keyword_features(processed_text: str) -> Dict[str, int]:
        """키워드 기반 특성 추출"""
        features = {}
        
        for risk_level, categories in RISK_KEYWORDS.items():
            for category, keywords in categories.items():
                feature_name = f'{risk_level}_{category}_count'
                features[feature_name] = sum([
                    1 for keyword in keywords 
                    if keyword.lower() in processed_text.lower()
                ])
        
        return features
    
    @staticmethod
    def create_sap_vocabulary() -> Dict[str, int]:
        """SAP 특화 어휘 사전 생성"""
        sap_vocab = []
        sap_vocab.extend(SAP_TRANSACTIONS)
        sap_vocab.extend(SAP_TABLES)
        sap_vocab.extend(SAP_TERMS)
        return {word: idx for idx, word in enumerate(sap_vocab)}
    
    @classmethod
    def extract_all_features(cls, text: str, processed_text: str) -> Dict[str, Any]:
        """모든 SAP 관련 특성 추출"""
        features = {}
        
        # 키워드 특성
        features.update(cls.extract_keyword_features(processed_text))
        
        # SAP 특화 특성
        features.update({
            'sap_transaction_count': cls.count_sap_transactions(processed_text),
            'sap_table_count': cls.count_sap_tables(processed_text),
            'injection_pattern_count': cls.count_injection_patterns(processed_text),
            'role_impersonation_count': cls.count_role_impersonation_patterns(processed_text),
            'sensitive_data_access_count': cls.count_sensitive_data_access_patterns(processed_text),
            'context_specific_pattern_count': cls.count_context_specific_patterns(processed_text),
            'risk_intensity_score': cls.calculate_risk_intensity_score(processed_text),
            'technical_complexity_count': cls.count_technical_complexity(processed_text)
        })
        
        return features
