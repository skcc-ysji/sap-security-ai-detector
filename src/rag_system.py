"""
RAG (Retrieval-Augmented Generation) System for SAP Security
Hybrid approach combining fast ML with accurate RAG
"""

import asyncio
import threading
import time
import hashlib
import json
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor
import redis
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import anthropic
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
import faiss
import numpy as np
import streamlit as st

class MultiLayerCache:
    """다층 캐싱 시스템"""
    
    def __init__(self):
        self.l1_cache = {}  # 메모리 캐시 (가장 빠름)
        self.l2_cache = None  # Redis 캐시 (선택적)
        self.cache_ttl = 3600  # 1시간
        
    def get(self, key: str) -> Optional[Dict]:
        """캐시에서 값 조회"""
        # L1 캐시 확인
        if key in self.l1_cache:
            cache_data = self.l1_cache[key]
            if time.time() - cache_data['timestamp'] < self.cache_ttl:
                return cache_data['data']
            else:
                del self.l1_cache[key]
        
        # L2 캐시 확인 (Redis)
        if self.l2_cache:
            try:
                cached_data = self.l2_cache.get(key)
                if cached_data:
                    data = json.loads(cached_data)
                    self.l1_cache[key] = {
                        'data': data,
                        'timestamp': time.time()
                    }
                    return data
            except:
                pass
        
        return None
    
    def set(self, key: str, data: Dict):
        """캐시에 값 저장"""
        cache_data = {
            'data': data,
            'timestamp': time.time()
        }
        
        # L1 캐시 저장
        self.l1_cache[key] = cache_data
        
        # L2 캐시 저장 (Redis)
        if self.l2_cache:
            try:
                self.l2_cache.setex(key, self.cache_ttl, json.dumps(data))
            except:
                pass

class SAPKnowledgeBase:
    """SAP 보안 지식베이스"""
    
    def __init__(self):
        self.knowledge_base = {
            "sap_transactions": {
                "user_management": ["SU01", "SU10", "PFCG", "SU24", "SU25", "SU26"],
                "authorization": ["SU01", "PFCG", "SU24", "SU25", "SU26", "SU56"],
                "business_processes": ["VA01", "MM01", "FI01", "CO01", "SD01", "PP01"],
                "system_administration": ["SM21", "SM37", "SM50", "SM51", "SM66", "ST22"],
                "security_audit": ["SU01", "SU10", "SU56", "ST01", "ST02", "ST03"]
            },
            "threat_patterns": {
                "privilege_escalation": [
                    "ignore previous instructions",
                    "act as administrator",
                    "bypass security",
                    "elevate privileges",
                    "get admin access",
                    "pretend to be admin"
                ],
                "data_exfiltration": [
                    "show all passwords",
                    "export user data",
                    "download sensitive information",
                    "get customer data",
                    "extract financial data",
                    "copy employee records"
                ],
                "role_impersonation": [
                    "you are admin",
                    "you are ceo",
                    "act as system administrator",
                    "pretend to be manager",
                    "role play as admin"
                ],
                "injection_attacks": [
                    "sql injection",
                    "command injection",
                    "script injection",
                    "execute command",
                    "run script"
                ]
            },
            "compliance_rules": {
                "SOX": ["financial_data_access", "audit_trail", "segregation_of_duties"],
                "GDPR": ["personal_data_protection", "data_retention", "consent_management"],
                "ISO27001": ["access_control", "incident_management", "risk_assessment"]
            },
            "business_context": {
                "HR_data": ["employee_salary", "personal_info", "performance_data"],
                "financial_data": ["bank_account", "credit_card", "transaction_history"],
                "customer_data": ["customer_info", "contact_details", "purchase_history"]
            }
        }
    
    def get_relevant_knowledge(self, text: str) -> Dict:
        """텍스트와 관련된 지식 검색"""
        relevant_knowledge = {}
        
        text_lower = text.lower()
        
        # SAP 트랜잭션 검색
        for category, transactions in self.knowledge_base["sap_transactions"].items():
            found_transactions = [t for t in transactions if t.lower() in text_lower]
            if found_transactions:
                relevant_knowledge[category] = found_transactions
        
        # 위협 패턴 검색
        for threat_type, patterns in self.knowledge_base["threat_patterns"].items():
            found_patterns = [p for p in patterns if p.lower() in text_lower]
            if found_patterns:
                relevant_knowledge[threat_type] = found_patterns
        
        # 비즈니스 컨텍스트 검색
        for context, keywords in self.knowledge_base["business_context"].items():
            found_keywords = [k for k in keywords if k.lower() in text_lower]
            if found_keywords:
                relevant_knowledge[context] = found_keywords
        
        return relevant_knowledge

class VectorStore:
    """벡터 저장소"""
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection("sap_security")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def add_documents(self, documents: List[str], metadata: List[Dict] = None):
        """문서 추가"""
        embeddings = self.embedding_model.encode(documents)
        
        ids = [f"doc_{i}" for i in range(len(documents))]
        if metadata is None:
            metadata = [{"source": "sap_security"} for _ in documents]
        
        self.collection.add(
            embeddings=embeddings.tolist(),
            documents=documents,
            metadatas=metadata,
            ids=ids
        )
    
    def search(self, query: str, n_results: int = 5) -> List[Dict]:
        """유사한 문서 검색"""
        query_embedding = self.embedding_model.encode([query])
        
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=n_results
        )
        
        return [
            {
                'document': doc,
                'metadata': meta,
                'distance': dist
            }
            for doc, meta, dist in zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            )
        ]

class ClaudeRAGDetector:
    """Claude Sonnet 3.5 기반 RAG 탐지기"""
    
    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.vector_store = VectorStore()
        self.knowledge_base = SAPKnowledgeBase()
        self.cache = MultiLayerCache()
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    def analyze_threat(self, text: str) -> Dict:
        """위협 분석"""
        # 캐시 확인
        cache_key = hashlib.md5(text.encode()).hexdigest()
        cached_result = self.cache.get(cache_key)
        if cached_result:
            return cached_result
        
        # 관련 지식 검색
        relevant_knowledge = self.knowledge_base.get_relevant_knowledge(text)
        vector_results = self.vector_store.search(text)
        
        # Claude 분석
        analysis_result = self._analyze_with_claude(text, relevant_knowledge, vector_results)
        
        # 캐시 저장
        self.cache.set(cache_key, analysis_result)
        
        return analysis_result
    
    def _analyze_with_claude(self, text: str, knowledge: Dict, vector_results: List[Dict]) -> Dict:
        """Claude를 사용한 분석"""
        prompt = self._build_analysis_prompt(text, knowledge, vector_results)
        
        try:
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1000,
                temperature=0.1,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            
            # 응답 파싱
            return self._parse_claude_response(response.content[0].text)
            
        except Exception as e:
            return {
                "risk_level": "unknown",
                "confidence": 0.0,
                "reasoning": f"Analysis failed: {str(e)}",
                "threat_type": "unknown",
                "recommended_actions": []
            }
    
    def _build_analysis_prompt(self, text: str, knowledge: Dict, vector_results: List[Dict]) -> str:
        """분석 프롬프트 구성"""
        knowledge_str = json.dumps(knowledge, indent=2)
        vector_str = json.dumps([r['document'] for r in vector_results], indent=2)
        
        return f"""
        Analyze the following SAP-related text for security threats:

        TEXT TO ANALYZE:
        {text}

        RELEVANT KNOWLEDGE:
        {knowledge_str}

        SIMILAR DOCUMENTS:
        {vector_str}

        Please provide a JSON response with the following structure:
        {{
            "risk_level": "low|medium|high|critical",
            "confidence": 0.0-1.0,
            "reasoning": "detailed explanation",
            "threat_type": "privilege_escalation|data_exfiltration|role_impersonation|injection_attack|other",
            "recommended_actions": ["action1", "action2", "action3"]
        }}

        Focus on:
        1. SAP-specific security threats
        2. Business context and compliance implications
        3. Specific risk factors and their severity
        4. Practical recommendations for mitigation
        """
    
    def _parse_claude_response(self, response_text: str) -> Dict:
        """Claude 응답 파싱"""
        try:
            # JSON 추출
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            json_str = response_text[start_idx:end_idx]
            
            result = json.loads(json_str)
            
            # 기본값 설정
            result.setdefault("risk_level", "unknown")
            result.setdefault("confidence", 0.0)
            result.setdefault("reasoning", "Analysis completed")
            result.setdefault("threat_type", "unknown")
            result.setdefault("recommended_actions", [])
            
            return result
            
        except Exception as e:
            return {
                "risk_level": "unknown",
                "confidence": 0.0,
                "reasoning": f"Failed to parse response: {str(e)}",
                "threat_type": "unknown",
                "recommended_actions": []
            }

class HybridSAPDetector:
    """하이브리드 SAP 탐지기 (빠른 ML + 정확한 RAG)"""
    
    def __init__(self, ml_detector, rag_detector: ClaudeRAGDetector):
        self.ml_detector = ml_detector
        self.rag_detector = rag_detector
        self.cache = MultiLayerCache()
        self.confidence_threshold = 0.8
        
    def detect_threat(self, text: str) -> Dict:
        """위협 탐지 (하이브리드 접근)"""
        # 캐시 확인
        cache_key = hashlib.md5(text.encode()).hexdigest()
        cached_result = self.cache.get(cache_key)
        if cached_result:
            return cached_result
        
        # 1. 빠른 ML 검사
        ml_result = self.ml_detector.predict(text)
        
        # 2. 신뢰도가 높으면 즉시 반환
        if ml_result.get('confidence', 0) > self.confidence_threshold:
            self.cache.set(cache_key, ml_result)
            return ml_result
        
        # 3. 신뢰도가 낮으면 RAG 사용
        rag_result = self.rag_detector.analyze_threat(text)
        
        # 4. 결과 융합
        fused_result = self._fuse_results(ml_result, rag_result)
        self.cache.set(cache_key, fused_result)
        
        return fused_result
    
    def _fuse_results(self, ml_result: Dict, rag_result: Dict) -> Dict:
        """ML과 RAG 결과 융합"""
        # RAG 결과가 더 신뢰할 수 있으면 RAG 결과 사용
        if rag_result.get('confidence', 0) > ml_result.get('confidence', 0):
            return {
                **rag_result,
                'analysis_method': 'rag',
                'ml_confidence': ml_result.get('confidence', 0),
                'rag_confidence': rag_result.get('confidence', 0)
            }
        else:
            return {
                **ml_result,
                'analysis_method': 'ml',
                'ml_confidence': ml_result.get('confidence', 0),
                'rag_confidence': rag_result.get('confidence', 0)
            }
    
    async def detect_threat_async(self, text: str) -> Dict:
        """비동기 위협 탐지"""
        # 1. 즉시 빠른 ML 결과 반환
        fast_result = self.ml_detector.predict(text)
        
        # 2. 백그라운드에서 RAG 분석
        if fast_result.get('confidence', 0) < self.confidence_threshold:
            asyncio.create_task(self._background_rag_analysis(text))
        
        return fast_result
    
    async def _background_rag_analysis(self, text: str):
        """백그라운드 RAG 분석"""
        try:
            rag_result = await asyncio.get_event_loop().run_in_executor(
                self.rag_detector.executor,
                self.rag_detector.analyze_threat,
                text
            )
            
            # 결과를 캐시에 저장
            cache_key = hashlib.md5(text.encode()).hexdigest()
            self.cache.set(cache_key, rag_result)
            
        except Exception as e:
            print(f"Background RAG analysis failed: {e}")

class KnowledgeBaseManager:
    """지식베이스 관리자"""
    
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.knowledge_base = SAPKnowledgeBase()
        
    def add_document(self, document: str, metadata: Dict = None):
        """문서 추가"""
        self.vector_store.add_documents([document], [metadata or {}])
        
    def add_documents_batch(self, documents: List[str], metadata_list: List[Dict] = None):
        """배치로 문서 추가"""
        if metadata_list is None:
            metadata_list = [{"source": "user_upload"} for _ in documents]
        
        self.vector_store.add_documents(documents, metadata_list)
        
    def search_documents(self, query: str, n_results: int = 5) -> List[Dict]:
        """문서 검색"""
        return self.vector_store.search(query, n_results)
        
    def get_knowledge_stats(self) -> Dict:
        """지식베이스 통계"""
        try:
            collection = self.vector_store.collection
            count = collection.count()
            return {
                "total_documents": count,
                "knowledge_base_size": len(self.knowledge_base.knowledge_base),
                "last_updated": time.time()
            }
        except:
            return {
                "total_documents": 0,
                "knowledge_base_size": len(self.knowledge_base.knowledge_base),
                "last_updated": time.time()
            } 