"""SAP Risk Detection 시스템의 상수 및 설정값"""

# 리스크 레벨 매핑
RISK_LEVEL_MAPPING = {
    'low': 0,
    'medium': 1,
    'high': 2,
    'critical': 3
}

# SAP 키워드 사전
RISK_KEYWORDS = {
    'critical': {
        'korean': [
            '해킹', '크랙', '우회', '침입', '무단접근', '권한상승', '시스템침투',
            '백도어', '악성코드', '보안허점', '취약점악용', 'SQL인젝션',
            '관리자비밀번호', '루트권한', '시스템관리자', '무력화',
            # 추가된 민감한 키워드들
            'ceo', '사장', '대표', '이사', '관리자', '시스템관리자',
            '연봉', '급여', '봉급', '월급', '개인정보', '민감정보',
            '전체', '모든', '전 인원', '전 직원', '모든 사용자',
            '공개', '노출', '추출', '다운로드', '복사', '백업',
            '비밀번호', '패스워드', '인증정보', '로그인정보',
            '사칭', '가짜', '위조', '변조', '조작'
        ],
        'english': [
            'hack', 'crack', 'bypass', 'exploit', 'unauthorized', 'privilege escalation',
            'backdoor', 'malware', 'vulnerability', 'sql injection', 'admin password',
            'root access', 'compromise', 'breach',
            # 추가된 민감한 키워드들
            'ceo', 'president', 'director', 'manager', 'administrator', 'system admin',
            'salary', 'payroll', 'compensation', 'personal data', 'sensitive data',
            'all', 'everyone', 'all users', 'all employees', 'entire staff',
            'expose', 'reveal', 'extract', 'download', 'copy', 'backup',
            'password', 'credential', 'authentication', 'login info',
            'impersonate', 'fake', 'forge', 'manipulate', 'tamper'
        ],
        'sap_specific': [
            'SU01 hack', 'PFCG bypass', 'client 000', 'MANDT 000', 'debug modify',
            'system breakpoint', 'RFC exploit', 'SE80 unauthorized',
            # 추가된 SAP 특화 위험 패턴
            'PA0008', 'PA0001', 'PA0002', 'PA0003', 'PA0004', 'PA0005',
            'HR master data', 'personnel data', 'employee data',
            'USR02', 'USR40', 'USR21', 'USR01', 'user master',
            'SAP_ALL', 'SAP_NEW', 'SAP_BC_BASIS_ADMIN', 'SAP_BC_USER_ADMIN'
        ]
    },
    'high': {
        'korean': [
            '권한', '승인', '역할할당', '사용자관리', '접근제어', '시스템설정',
            '데이터베이스접근', '민감정보', '보안설정', '권한관리', '관리자기능'
        ],
        'english': [
            'authorization', 'permission', 'role assignment', 'user management',
            'access control', 'system configuration', 'database access',
            'sensitive data', 'admin function'
        ],
        'sap_specific': [
            'SU01', 'PFCG', 'SU03', 'SM59', 'RZ10', 'RZ11', 'SCC4',
            'profile parameter', 'authorization object', 'composite role'
        ]
    },
    'medium': {
        'korean': [
            '설정', '구성', '파라미터', '옵션', '기능설정', '시스템옵션'
        ],
        'english': [
            'configuration', 'setup', 'parameter', 'option', 'setting'
        ],
        'sap_specific': [
            'SPRO', 'SM30', 'SE11', 'customizing', 'IMG', 'variant'
        ]
    },
    'low': {
        'korean': [
            '방법', '사용법', '튜토리얼', '가이드', '도움말', '기본기능',
            '일반질문', '조회', '검색', '리포트', '생성', '입력'
        ],
        'english': [
            'how to', 'tutorial', 'guide', 'help', 'basic', 'create',
            'display', 'report', 'generate', 'enter'
        ],
        'sap_specific': [
            'VA01', 'VA03', 'ME21N', 'ME23N', 'FB01', 'FB03',
            'MM01', 'MM03', 'create order', 'display document'
        ]
    }
}

# SAP 컨텍스트 가중치
SAP_CONTEXT_WEIGHTS = {
    'transaction_codes': 2.0,
    'table_names': 1.5,
    'user_exits': 1.8,
    'rfc_functions': 1.6,
    'authorization_objects': 1.7
}

# 프롬프트 인젝션 패턴
INJECTION_PATTERNS = {
    'korean': [
        r'이전\s*(지시사항|명령|instruction)\s*(무시|ignore)',
        r'새로운\s*(역할|role)\s*(부여|assign)',
        r'실제로는\s*',
        r'사실은\s*',
        r'최상위\s*(권한|authority)',
        r'모든\s*(권한|permission)\s*(부여|grant)',
        # 추가된 역할 사칭 패턴
        r'너는\s*(ceo|사장|대표|이사|관리자|시스템관리자|admin|administrator)',
        r'나는\s*(ceo|사장|대표|이사|관리자|시스템관리자|admin|administrator)',
        r'당신은\s*(ceo|사장|대표|이사|관리자|시스템관리자|admin|administrator)',
        r'(ceo|사장|대표|이사|관리자|시스템관리자|admin|administrator)\s*야',
        r'(ceo|사장|대표|이사|관리자|시스템관리자|admin|administrator)\s*입니다',
        # 민감 정보 접근 패턴
        r'(연봉|급여|봉급|월급|개인정보|민감정보)\s*(공개|노출|추출|다운로드)',
        r'(전체|모든|전 인원|전 직원|모든 사용자)\s*(연봉|급여|개인정보)',
        r'(비밀번호|패스워드|인증정보|로그인정보)\s*(공개|노출|추출)',
        # 권한 남용 패턴
        r'(모든|전체|전)\s*(권한|permission|authority)\s*(부여|grant|give)',
        r'(최고|최상위|관리자)\s*(권한|permission|authority)',
        r'(무시|ignore|bypass)\s*(보안|security|authorization)'
    ],
    'english': [
        r'ignore\s+(previous|above|prior)\s+(instruction|command)',
        r'(new|updated)\s+(role|instruction)',
        r'(actually|in\s+fact)',
        r'you\s+are\s+(now|actually)',
        r'highest\s+(authority|privilege)',
        r'grant\s+(all|full)\s+(access|permission)',
        # 추가된 역할 사칭 패턴
        r'you\s+are\s+(ceo|president|director|manager|administrator|admin)',
        r'i\s+am\s+(ceo|president|director|manager|administrator|admin)',
        r'you\s+are\s+now\s+(ceo|president|director|manager|administrator|admin)',
        r'(ceo|president|director|manager|administrator|admin)\s+(here|now)',
        # 민감 정보 접근 패턴
        r'(salary|payroll|personal\s+data|sensitive\s+data)\s+(expose|reveal|extract|download)',
        r'(all|everyone|all\s+users|all\s+employees)\s+(salary|payroll|personal\s+data)',
        r'(password|credential|authentication)\s+(expose|reveal|extract)',
        # 권한 남용 패턴
        r'(all|full|complete)\s+(permission|authority|access)\s+(grant|give)',
        r'(highest|top|admin)\s+(permission|authority|access)',
        r'(ignore|bypass)\s+(security|authorization|access\s+control)'
    ]
}

# SAP 트랜잭션 코드
SAP_TRANSACTIONS = [
    'su01', 'su03', 'pfcg', 'sm59', 'rz10', 'rz11', 'scc4',
    'se80', 'se11', 'sm30', 'spro', 'va01', 'va03', 'me21n',
    'me23n', 'fb01', 'fb03', 'mm01', 'mm03'
]

# SAP 테이블
SAP_TABLES = [
    'usr02', 'usr40', 'bseg', 'bkpf', 'kna1', 'lfa1',
    'mara', 'marc', 'makt', 'pa0001', 'pa0008'
]

# SAP 용어
SAP_TERMS = [
    'authorization', 'profile', 'role', 'transaction', 'client',
    'mandt', 'rfc', 'bapi', 'abap', 'sapgui', 'fiori'
]
