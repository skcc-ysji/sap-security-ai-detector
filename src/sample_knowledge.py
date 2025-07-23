"""
SAP 보안 관련 샘플 지식베이스 데이터
"""

SAP_SECURITY_KNOWLEDGE = [
    # SAP 권한 관리
    {
        "title": "SAP 권한 관리 가이드",
        "content": """
        SAP 시스템에서 권한 관리는 보안의 핵심입니다.
        
        주요 권한 관리 트랜잭션:
        - SU01: 사용자 관리
        - PFCG: 역할 관리
        - SU24: 권한 객체 관리
        - SU25: 권한 프로파일 관리
        - SU26: 권한 분석
        
        권한 관리 모범 사례:
        1. 최소 권한 원칙 적용
        2. 정기적인 권한 검토
        3. 역할 기반 접근 제어
        4. 권한 승인 프로세스 수립
        """,
        "category": "authorization_management",
        "tags": ["SU01", "PFCG", "SU24", "SU25", "SU26", "권한관리", "보안"]
    },
    
    # SAP 보안 위협
    {
        "title": "SAP 보안 위협 유형",
        "content": """
        SAP 시스템에서 발생할 수 있는 주요 보안 위협:
        
        1. 권한 상승 공격
        - 관리자 권한 사칭
        - 시스템 권한 탈취
        - 역할 기반 접근 우회
        
        2. 데이터 유출
        - 민감 정보 무단 접근
        - 고객 데이터 유출
        - 재무 정보 탈취
        
        3. 인젝션 공격
        - SQL 인젝션
        - 명령어 인젝션
        - 스크립트 인젝션
        
        4. 세션 하이재킹
        - 사용자 세션 탈취
        - 인증 우회
        - 세션 고정 공격
        """,
        "category": "security_threats",
        "tags": ["권한상승", "데이터유출", "인젝션", "세션하이재킹", "보안위협"]
    },
    
    # SAP 트랜잭션 보안
    {
        "title": "SAP 트랜잭션 보안 가이드",
        "content": """
        SAP 트랜잭션별 보안 고려사항:
        
        사용자 관리 트랜잭션:
        - SU01: 사용자 생성/수정 시 권한 검증 필수
        - SU10: 대량 사용자 관리 시 승인 프로세스
        - SU56: 사용자 세션 모니터링
        
        권한 관리 트랜잭션:
        - PFCG: 역할 생성 시 최소 권한 원칙 적용
        - SU24: 권한 객체 설정 시 보안 검토
        - SU25: 권한 프로파일 변경 시 승인 필요
        
        시스템 관리 트랜잭션:
        - SM21: 시스템 로그 모니터링
        - SM37: 백그라운드 작업 보안
        - SM50: 작업 프로세스 관리
        """,
        "category": "transaction_security",
        "tags": ["SU01", "SU10", "SU56", "PFCG", "SU24", "SU25", "SM21", "SM37", "SM50"]
    },
    
    # SAP 데이터 보안
    {
        "title": "SAP 데이터 보안 정책",
        "content": """
        SAP 데이터 보안을 위한 정책:
        
        1. 데이터 분류
        - 공개 데이터: 일반 사용자 접근 가능
        - 내부 데이터: 승인된 사용자만 접근
        - 기밀 데이터: 제한된 사용자만 접근
        - 극비 데이터: 최고 관리자만 접근
        
        2. 접근 제어
        - 역할 기반 접근 제어 (RBAC)
        - 데이터 레벨 보안 (DLS)
        - 필드 레벨 보안 (FLS)
        
        3. 데이터 암호화
        - 저장 데이터 암호화
        - 전송 데이터 암호화
        - 백업 데이터 암호화
        
        4. 감사 및 모니터링
        - 데이터 접근 로그
        - 권한 변경 이력
        - 비정상 접근 탐지
        """,
        "category": "data_security",
        "tags": ["데이터분류", "접근제어", "암호화", "감사", "모니터링"]
    },
    
    # SAP 보안 모범 사례
    {
        "title": "SAP 보안 모범 사례",
        "content": """
        SAP 시스템 보안을 위한 모범 사례:
        
        1. 사용자 관리
        - 정기적인 사용자 계정 검토
        - 퇴사자 계정 즉시 비활성화
        - 강력한 패스워드 정책 적용
        - 다중 인증 (MFA) 구현
        
        2. 권한 관리
        - 최소 권한 원칙 적용
        - 역할 기반 접근 제어
        - 권한 승인 프로세스 수립
        - 정기적인 권한 검토
        
        3. 시스템 보안
        - 정기적인 보안 패치 적용
        - 네트워크 보안 강화
        - 백업 및 복구 계획 수립
        - 보안 사고 대응 계획
        
        4. 모니터링 및 감사
        - 실시간 보안 모니터링
        - 로그 분석 및 보고
        - 비정상 활동 탐지
        - 정기적인 보안 감사
        """,
        "category": "security_best_practices",
        "tags": ["사용자관리", "권한관리", "시스템보안", "모니터링", "감사"]
    },
    
    # SAP 보안 사고 대응
    {
        "title": "SAP 보안 사고 대응 가이드",
        "content": """
        SAP 보안 사고 발생 시 대응 절차:
        
        1. 사고 탐지
        - 비정상 활동 모니터링
        - 보안 로그 분석
        - 사용자 신고 접수
        - 자동화된 탐지 시스템
        
        2. 초기 대응
        - 사고 상황 평가
        - 영향 범위 파악
        - 긴급 조치 실행
        - 관련자 통보
        
        3. 사고 조사
        - 증거 수집 및 보존
        - 원인 분석
        - 영향도 평가
        - 재발 방지책 수립
        
        4. 복구 및 복원
        - 시스템 복구
        - 데이터 복원
        - 서비스 정상화
        - 사후 모니터링
        
        5. 사후 관리
        - 사고 보고서 작성
        - 교훈 도출
        - 보안 강화 조치
        - 정책 개선
        """,
        "category": "incident_response",
        "tags": ["사고탐지", "초기대응", "사고조사", "복구복원", "사후관리"]
    },
    
    # SAP 규정 준수
    {
        "title": "SAP 규정 준수 가이드",
        "content": """
        SAP 시스템의 주요 규정 준수 요구사항:
        
        1. SOX (Sarbanes-Oxley Act)
        - 재무 데이터 무결성
        - 내부 통제 평가
        - 감사 추적 유지
        - 정기적인 보안 평가
        
        2. GDPR (General Data Protection Regulation)
        - 개인정보 보호
        - 데이터 처리 동의
        - 데이터 삭제 권리
        - 개인정보 유출 통보
        
        3. ISO 27001
        - 정보보안 관리체계
        - 위험 평가 및 처리
        - 보안 정책 수립
        - 지속적 개선
        
        4. PCI DSS
        - 신용카드 데이터 보호
        - 네트워크 보안
        - 접근 제어
        - 정기적인 보안 테스트
        """,
        "category": "compliance",
        "tags": ["SOX", "GDPR", "ISO27001", "PCIDSS", "규정준수"]
    },
    
    # SAP 보안 아키텍처
    {
        "title": "SAP 보안 아키텍처 설계",
        "content": """
        SAP 시스템 보안 아키텍처 설계 원칙:
        
        1. 방어적 깊이 (Defense in Depth)
        - 다층 보안 구조
        - 네트워크 분리
        - 접근 제어 계층화
        - 모니터링 체계화
        
        2. 최소 권한 원칙
        - 필요한 권한만 부여
        - 역할 기반 접근
        - 정기적인 권한 검토
        - 권한 승인 프로세스
        
        3. 보안 모니터링
        - 실시간 로그 분석
        - 비정상 활동 탐지
        - 보안 사고 대응
        - 성능 모니터링
        
        4. 비즈니스 연속성
        - 백업 및 복구
        - 재해 복구 계획
        - 고가용성 구성
        - 서비스 수준 협약
        """,
        "category": "security_architecture",
        "tags": ["방어적깊이", "최소권한", "모니터링", "비즈니스연속성"]
    },
    
    # SAP 보안 도구
    {
        "title": "SAP 보안 도구 및 솔루션",
        "content": """
        SAP 보안을 위한 주요 도구 및 솔루션:
        
        1. SAP 보안 도구
        - SAP Security Optimization Service
        - SAP Access Control
        - SAP Process Control
        - SAP Risk Management
        
        2. 모니터링 도구
        - SAP Solution Manager
        - SAP Focused Run
        - SAP Cloud ALM
        - SAP Analytics Cloud
        
        3. 보안 솔루션
        - SAP Identity Management
        - SAP Single Sign-On
        - SAP Cloud Identity
        - SAP Identity Provisioning
        
        4. 외부 보안 도구
        - SIEM (Security Information and Event Management)
        - IDS/IPS (Intrusion Detection/Prevention System)
        - DLP (Data Loss Prevention)
        - EDR (Endpoint Detection and Response)
        """,
        "category": "security_tools",
        "tags": ["SAP보안도구", "모니터링", "보안솔루션", "외부도구"]
    },
    
    # SAP 보안 위험 평가
    {
        "title": "SAP 보안 위험 평가 방법론",
        "content": """
        SAP 시스템 보안 위험 평가 방법론:
        
        1. 위험 식별
        - 자산 분류 및 가치 평가
        - 위협 시나리오 분석
        - 취약점 평가
        - 영향도 분석
        
        2. 위험 분석
        - 위험 확률 평가
        - 위험 영향도 평가
        - 위험 우선순위 결정
        - 위험 매트릭스 작성
        
        3. 위험 처리
        - 위험 회피
        - 위험 전이
        - 위험 완화
        - 위험 수용
        
        4. 위험 모니터링
        - 정기적인 위험 재평가
        - 위험 지표 모니터링
        - 위험 보고서 작성
        - 위험 관리 개선
        """,
        "category": "risk_assessment",
        "tags": ["위험식별", "위험분석", "위험처리", "위험모니터링"]
    },
    
    # SAP 보안 교육
    {
        "title": "SAP 보안 교육 프로그램",
        "content": """
        SAP 보안 교육 프로그램 구성:
        
        1. 기본 보안 교육
        - SAP 보안 개요
        - 사용자 보안 책임
        - 패스워드 관리
        - 사회공학 방지
        
        2. 역할별 보안 교육
        - 개발자 보안 교육
        - 관리자 보안 교육
        - 사용자 보안 교육
        - 감사자 보안 교육
        
        3. 보안 인식 제고
        - 정기적인 보안 교육
        - 보안 사고 사례 학습
        - 보안 모범 사례 공유
        - 보안 문화 조성
        
        4. 보안 인증
        - SAP 보안 인증 과정
        - 보안 전문가 양성
        - 보안 역량 개발
        - 지속적인 학습
        """,
        "category": "security_training",
        "tags": ["기본보안교육", "역할별교육", "보안인식", "보안인증"]
    }
]

def get_sample_knowledge():
    """샘플 지식베이스 데이터 반환"""
    return SAP_SECURITY_KNOWLEDGE

def get_knowledge_by_category(category):
    """카테고리별 지식베이스 데이터 반환"""
    return [item for item in SAP_SECURITY_KNOWLEDGE if item['category'] == category]

def get_knowledge_by_tag(tag):
    """태그별 지식베이스 데이터 반환"""
    return [item for item in SAP_SECURITY_KNOWLEDGE if tag in item['tags']]

def get_all_categories():
    """모든 카테고리 반환"""
    return list(set([item['category'] for item in SAP_SECURITY_KNOWLEDGE]))

def get_all_tags():
    """모든 태그 반환"""
    all_tags = []
    for item in SAP_SECURITY_KNOWLEDGE:
        all_tags.extend(item['tags'])
    return list(set(all_tags)) 