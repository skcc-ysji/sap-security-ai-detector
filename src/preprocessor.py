"""텍스트 전처리 및 기본 특성 추출을 위한 모듈"""

import re
import numpy as np
from typing import Dict, List, Any

class TextPreprocessor:
    @staticmethod
    def preprocess_text(text: str) -> str:
        """텍스트 전처리
        
        Args:
            text (str): 원본 텍스트
            
        Returns:
            str: 전처리된 텍스트
        """
        # 기본 정리
        text = str(text).strip()
        
        # SAP 트랜잭션 코드 보존
        sap_tokens = re.findall(r'\b[A-Z]{2,4}\d{1,3}[A-Z]?\b', text)
        for i, token in enumerate(sap_tokens):
            text = text.replace(token, f"__SAP_TOKEN_{i}__")
        
        # 영문 소문자 변환 (한글 제외)
        text = ''.join([c.lower() if c.isascii() and c.isalpha() else c for c in text])
        
        # 특수문자 처리
        text = re.sub(r'[^\w가-힣\s]', ' ', text)
        
        # 공백 정리
        text = re.sub(r'\s+', ' ', text).strip()
        
        # SAP 토큰 복원
        for i, token in enumerate(sap_tokens):
            text = text.replace(f"__sap_token_{i}__", token.lower())
        
        return text

    @staticmethod
    def extract_basic_features(text: str, processed_text: str) -> Dict[str, Any]:
        """기본 텍스트 특성 추출
        
        Args:
            text (str): 원본 텍스트
            processed_text (str): 전처리된 텍스트
            
        Returns:
            Dict[str, Any]: 추출된 특성들
        """
        words = processed_text.split()
        
        features = {
            'text_length': len(processed_text),
            'word_count': len(words),
            'sentence_count': len(re.split(r'[.!?]', text)),
            'avg_word_length': np.mean([len(word) for word in words]) if words else 0,
            'special_char_ratio': len(re.findall(r'[^a-zA-Z0-9가-힣\s]', text)) / len(text) if text else 0,
            'uppercase_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0,
            'is_korean': int(bool(re.search(r'[가-힣]', text))),
            'is_english': int(bool(re.search(r'[a-zA-Z]', text)))
        }
        
        features['mixed_language'] = int(features['is_korean'] and features['is_english'])
        
        return features
