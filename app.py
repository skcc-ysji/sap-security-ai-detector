"""SAP 위험도 탐지 시스템 Streamlit UI"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import time
import io
from datetime import datetime
import base64
import logging
import warnings
import os
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

def check_environment_setup():
    """환경 설정 상태 확인"""
    status = {
        "claude_api_key": bool(os.getenv("CLAUDE_API_KEY") or st.secrets.get("CLAUDE_API_KEY", "")),
        "enable_rag": os.getenv("ENABLE_RAG", "true").lower() == "true",
        "enable_caching": os.getenv("ENABLE_CACHING", "true").lower() == "true",
        "enable_async": os.getenv("ENABLE_ASYNC", "true").lower() == "true"
    }
    return status

# pandas 경고 메시지 억제
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
pd.options.mode.chained_assignment = None

from src.sap_risk_detector import SAPRiskDetector
from src.constants import RISK_LEVEL_MAPPING, RISK_KEYWORDS

# 페이지 설정
st.set_page_config(
    page_title="SAP 위험도 탐지 시스템 v0.1",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# SAP 스타일 색상 테마
SAP_COLORS = {
    'primary': '#0070C0',      # SAP 파란색
    'secondary': '#F0F0F0',    # 밝은 회색
    'success': '#107C10',       # SAP 녹색
    'warning': '#FF8C00',       # SAP 주황색
    'danger': '#D13438',        # SAP 빨간색
    'info': '#0078D4',          # 정보 파란색
    'light': '#FFFFFF',         # 흰색
    'dark': '#323130',          # 어두운 회색
    'border': '#E1DFDD'         # 테두리 회색
}

# 위험도별 색상 (SAP 스타일)
RISK_COLORS = {
    'low': '#107C10',      # SAP 녹색
    'medium': '#FF8C00',   # SAP 주황색
    'high': '#D13438',     # SAP 빨간색
    'critical': '#8B0000'  # 진한 빨간색
}

# CSS 스타일 적용
st.markdown("""
<style>
    /* SAP 스타일 CSS */
    .main {
        background-color: #FFFFFF;
    }
    
    .stApp {
        background-color: #F8F9FA;
    }
    
    .stSidebar {
        background-color: #FFFFFF;
        border-right: 1px solid #E1DFDD;
    }
    
    /* 모든 텍스트를 검정색으로 강제 설정 */
    * {
        color: #000000 !important;
    }
    
    /* 특정 요소들만 예외 처리 */
    .stButton > button {
        color: white !important;
    }
    
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        color: white !important;
    }
    
    /* 성공/경고/에러/정보 메시지 색상 */
    .stSuccess {
        color: #107C10 !important;
    }
    
    .stWarning {
        color: #FF8C00 !important;
    }
    
    .stError {
        color: #D13438 !important;
    }
    
    .stInfo {
        color: #0078D4 !important;
    }
    
    /* 헤더 텍스트는 흰색 유지 */
    .sap-header h1, .sap-header p {
        color: white !important;
    }
    
    /* 사이드바 헤더 텍스트는 흰색 유지 */
    .stSidebar div[data-testid="stMarkdownContainer"] div:first-child h2,
    .stSidebar div[data-testid="stMarkdownContainer"] div:first-child p {
        color: white !important;
    }
    
    /* 버튼 스타일 */
    .stButton > button {
        background-color: #0070C0;
        color: white !important;
        border: none;
        border-radius: 4px;
        padding: 8px 16px;
        font-weight: 500;
    }
    
    .stButton > button:hover {
        background-color: #005A9E;
    }
    
    /* 입력 필드 스타일 */
    .stSelectbox > div > div > div {
        background-color: white;
        border: 1px solid #E1DFDD;
        border-radius: 4px;
        color: #000000 !important;
    }
    
    .stTextInput > div > div > input {
        background-color: white;
        border: 1px solid #E1DFDD;
        border-radius: 4px;
        color: #000000 !important;
    }
    
    .stTextArea > div > div > textarea {
        background-color: white;
        border: 1px solid #E1DFDD;
        border-radius: 4px;
        color: #000000 !important;
    }
    
    .stNumberInput > div > div > input {
        background-color: white;
        border: 1px solid #E1DFDD;
        border-radius: 4px;
        color: #000000 !important;
    }
    
    /* 라디오 버튼 스타일 */
    .stRadio > div > div > div > div {
        color: #000000 !important;
    }
    
    /* 슬라이더 스타일 */
    .stSlider > div > div > div > div {
        color: #000000 !important;
    }
    
    /* 카드 스타일 */
    .metric-card {
        background-color: white;
        border: 1px solid #E1DFDD;
        border-radius: 8px;
        padding: 16px;
        margin: 8px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* 헤더 스타일 */
    .sap-header {
        background: linear-gradient(135deg, #0070C0 0%, #005A9E 100%);
        color: white;
        padding: 20px;
        border-radius: 8px;
        margin-bottom: 20px;
    }
    
    /* 탭 스타일 */
    .stTabs > div > div > div > div {
        background-color: white;
        border: 1px solid #E1DFDD;
        border-radius: 4px;
    }
    
    .stTabs > div > div > div > div[data-baseweb="tab"] {
        background-color: #F0F0F0;
        color: #000000 !important;
    }
    
    .stTabs > div > div > div > div[aria-selected="true"] {
        background-color: #0070C0;
        color: white !important;
    }
    
    /* 메트릭 카드 스타일 */
    .stMetric > div > div > div {
        color: #000000 !important;
    }
    
    /* 라벨 텍스트 색상 */
    .stSelectbox label, .stNumberInput label, .stRadio label, .stSlider label {
        color: #000000 !important;
    }
    
    /* 플레이스홀더 텍스트 색상 */
    .stTextArea textarea::placeholder {
        color: #666666 !important;
    }
    
    .stTextInput input::placeholder {
        color: #666666 !important;
    }
    
    /* 파일 업로더 스타일 */
    .stFileUploader > div > div > div {
        color: #000000 !important;
    }
    
    /* 데이터프레임 스타일 */
    .stDataFrame {
        color: #000000 !important;
    }
    
    /* 진행바 스타일 */
    .stProgress > div > div > div {
        color: #000000 !important;
    }
    
    /* 스피너 스타일 */
    .stSpinner > div > div {
        color: #000000 !important;
    }
</style>
""", unsafe_allow_html=True)

# 세션 상태 초기화
if 'detector' not in st.session_state:
    st.session_state.detector = None
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []

def load_model():
    """모델 로드"""
    try:
        import os
        
        model_path = 'models/enhanced_sap_risk_model_v2.pkl'
        
        # 파일 존재 확인
        if not os.path.exists(model_path):
            st.error(f"모델 파일이 존재하지 않습니다: {model_path}")
            st.info("먼저 '새 모델 학습'을 실행하여 모델을 생성해주세요.")
            return False
        
        # 파일 크기 확인
        file_size = os.path.getsize(model_path)
        if file_size == 0:
            st.error("모델 파일이 비어있습니다.")
            return False
        
        st.info(f"모델 파일 로드 중... ({file_size / (1024*1024):.1f} MB)")
        
        detector = SAPRiskDetector()
        detector.load_model(model_path)
        
        # 벡터화 모델 상태 확인
        if not detector.vectorizers:
            st.error("벡터화 모델 로드에 실패했습니다.")
            return False
        
        st.session_state.detector = detector
        st.session_state.model_loaded = True
        st.session_state.model_trained = False
        
        st.success("모델 로드 완료!")
        st.info(f"벡터화 모델: {len(detector.vectorizers)}개")
        return True
        
    except FileNotFoundError:
        st.error("모델 파일을 찾을 수 없습니다.")
        st.info("먼저 '새 모델 학습'을 실행하여 모델을 생성해주세요.")
        return False
    except Exception as e:
        st.error(f"모델 로드 실패: {str(e)}")
        st.info("모델 파일이 손상되었을 수 있습니다. 새로 학습해주세요.")
        return False

def train_model(n_samples):
    """모델 학습"""
    try:
        with st.spinner("모델 학습 중..."):
            # 진행률 표시
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # 1단계: 데이터 생성
            status_text.text("1/4 - 학습 데이터 생성 중...")
            progress_bar.progress(25)
            
            detector = SAPRiskDetector()
            
            # 2단계: 특성 추출
            status_text.text("2/4 - 특성 추출 중...")
            progress_bar.progress(50)
            
            # 3단계: 모델 학습
            status_text.text("3/4 - 모델 학습 중...")
            progress_bar.progress(75)
            
            results = detector.train(n_samples=n_samples)
            
            # 4단계: 완료
            status_text.text("4/4 - 학습 완료!")
            progress_bar.progress(100)
            
            # 상태 업데이트
            st.session_state.detector = detector
            st.session_state.model_loaded = True
            st.session_state.model_trained = True
            
            # 진행률 표시 제거
            progress_bar.empty()
            status_text.empty()
            
            return results
    except Exception as e:
        st.error(f"모델 학습 실패: {str(e)}")
        return None

def analyze_text_hybrid(text, confidence_threshold=0.7):
    """하이브리드 분석 (빠른 ML + 정확한 RAG)"""
    if not st.session_state.model_loaded:
        return None
    
    try:
        # 1. 빠른 ML 분석
        ml_result = st.session_state.detector.predict(text)
        
        # 2. RAG 시스템 활성화 확인
        enable_rag = st.session_state.get('enable_rag', True)
        rag_confidence_threshold = st.session_state.get('rag_confidence_threshold', 0.8)
        
        # 3. 신뢰도가 낮으면 RAG 분석 수행
        if enable_rag and ml_result.get('confidence', 0) < rag_confidence_threshold:
            try:
                # RAG 분석 수행
                rag_result = perform_rag_analysis(text)
                
                # 결과 융합
                fused_result = fuse_ml_rag_results(ml_result, rag_result)
                
                # 분석 히스토리에 추가
                add_to_analysis_history(text, fused_result)
                
                return fused_result
                
            except Exception as e:
                st.warning(f"RAG 분석 실패: {str(e)}. ML 결과만 사용합니다.")
                # RAG 실패 시 ML 결과만 사용
                add_to_analysis_history(text, ml_result)
                return ml_result
        
        # 4. 신뢰도가 높으면 ML 결과만 사용
        add_to_analysis_history(text, ml_result)
        return ml_result
        
    except Exception as e:
        st.error(f"분석 실패: {str(e)}")
        return None

def perform_rag_analysis(text):
    """RAG 분석 수행"""
    try:
        # Claude API 키 확인 (환경 변수 우선, 그 다음 Streamlit secrets)
        claude_api_key = os.getenv("CLAUDE_API_KEY") or st.secrets.get("CLAUDE_API_KEY", "")
        if not claude_api_key:
            return {
                "risk_level": "unknown",
                "confidence": 0.0,
                "reasoning": "Claude API 키가 설정되지 않았습니다. .env 파일 또는 Streamlit secrets에 API 키를 설정해주세요.",
                "threat_type": "unknown",
                "recommended_actions": []
            }
        
        # RAG 시스템 초기화
        from src.rag_system import ClaudeRAGDetector
        rag_detector = ClaudeRAGDetector(claude_api_key)
        
        # RAG 분석 수행
        rag_result = rag_detector.analyze_threat(text)
        
        return rag_result
        
    except Exception as e:
        return {
            "risk_level": "unknown",
            "confidence": 0.0,
            "reasoning": f"RAG 분석 실패: {str(e)}",
            "threat_type": "unknown",
            "recommended_actions": []
        }

def fuse_ml_rag_results(ml_result, rag_result):
    """ML과 RAG 결과 융합"""
    # RAG 결과가 더 신뢰할 수 있으면 RAG 결과 사용
    if rag_result.get('confidence', 0) > ml_result.get('confidence', 0):
        return {
            **rag_result,
            'analysis_method': 'rag',
            'ml_confidence': ml_result.get('confidence', 0),
            'rag_confidence': rag_result.get('confidence', 0),
            'rag_analysis': rag_result,
            'probabilities': ml_result.get('probabilities', {}),
            'detailed_analysis': ml_result.get('detailed_analysis', {})
        }
    else:
        return {
            **ml_result,
            'analysis_method': 'ml',
            'ml_confidence': ml_result.get('confidence', 0),
            'rag_confidence': rag_result.get('confidence', 0),
            'rag_analysis': rag_result
        }

def add_to_analysis_history(text, result):
    """분석 히스토리에 추가"""
    if 'analysis_history' not in st.session_state:
        st.session_state.analysis_history = []
    
    history_record = {
        'text': text,
        'result': result,
        'timestamp': datetime.now()
    }
    
    st.session_state.analysis_history.append(history_record)
    
    # 히스토리 크기 제한 (최근 100개만 유지)
    if len(st.session_state.analysis_history) > 100:
        st.session_state.analysis_history = st.session_state.analysis_history[-100:]

def analyze_text(text, confidence_threshold=0.6):
    """기존 텍스트 분석 (하위 호환성)"""
    return analyze_text_hybrid(text, confidence_threshold)

def create_risk_gauge(confidence, risk_level):
    """위험도 게이지 생성 (SAP 스타일)"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=confidence * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"위험도: {risk_level.upper()}", 'font': {'color': '#323130'}},
        delta={'reference': 50},
        gauge={
            'axis': {'range': [None, 100], 'tickcolor': '#E1DFDD'},
            'bar': {'color': RISK_COLORS[risk_level]},
            'bgcolor': '#F0F0F0',
            'borderwidth': 2,
            'bordercolor': '#E1DFDD',
            'steps': [
                {'range': [0, 25], 'color': "#E1DFDD"},
                {'range': [25, 50], 'color': "#C8C6C4"},
                {'range': [50, 75], 'color': "#A19F9D"},
                {'range': [75, 100], 'color': "#605E5C"}
            ],
            'threshold': {
                'line': {'color': "#D13438", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#323130'}
    )
    return fig

def create_probability_chart(probabilities):
    """확률 분포 차트 생성 (SAP 스타일)"""
    df = pd.DataFrame(list(probabilities.items()), columns=['Risk Level', 'Probability'])
    df['Color'] = df['Risk Level'].map(RISK_COLORS)
    
    fig = px.bar(
        df, 
        x='Risk Level', 
        y='Probability',
        color='Risk Level',
        color_discrete_map=RISK_COLORS,
        title="위험도별 확률 분포"
    )
    
    fig.update_layout(
        height=300,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#323130'},
        title={'font': {'color': '#323130'}},
        xaxis={'gridcolor': '#E1DFDD'},
        yaxis={'gridcolor': '#E1DFDD'}
    )
    
    return fig

def create_pattern_chart(analysis):
    """패턴 감지 차트 생성 (SAP 스타일)"""
    patterns = {
        'SAP 트랜잭션': analysis['sap_transaction_count'],
        '인젝션 패턴': analysis['injection_pattern_count'],
        '역할 사칭': analysis['role_impersonation_count'],
        '민감 정보 접근': analysis['sensitive_data_access_count']
    }
    
    df = pd.DataFrame(list(patterns.items()), columns=['Pattern', 'Count'])
    
    fig = px.bar(
        df,
        x='Pattern',
        y='Count',
        title="감지된 패턴 수",
        color='Count',
        color_continuous_scale='Blues'
    )
    
    fig.update_layout(
        height=300,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#323130'},
        title={'font': {'color': '#323130'}},
        xaxis={'gridcolor': '#E1DFDD'},
        yaxis={'gridcolor': '#E1DFDD'}
    )
    
    return fig

def download_csv(data, filename):
    """CSV 다운로드"""
    csv = data.to_csv(index=False, encoding='utf-8-sig')
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">CSV 다운로드</a>'
    return href

# 메인 영역
st.markdown("""
<div class="sap-header">
    <h1 style="margin: 0; color: white;">🛡️ SAP 위험도 탐지 시스템 v2.0</h1>
    <p style="margin: 10px 0 0 0; color: white; opacity: 0.9;">SAP 시스템에서 보안 위험을 실시간으로 탐지하는 AI 시스템입니다.</p>
</div>
""", unsafe_allow_html=True)

# 탭 구성
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔧 모델 관리", "⚙️ 분석 설정", "📊 실시간 분석", "📁 배치 분석", "📚 지식베이스 관리"])

with tab1:
    st.markdown("### 🔄 모델 로드")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("모델 로드", type="primary", use_container_width=True):
            if load_model():
                st.success("✅ 모델 로드 완료!")
    
    with col2:
        # 모델 저장
        if st.session_state.model_loaded:
            if st.button("💾 모델 저장", type="secondary", use_container_width=True):
                try:
                    st.session_state.detector.save_model('models/enhanced_sap_risk_model_v2.pkl')
                    st.success("✅ 모델 저장 완료!")
                except Exception as e:
                    st.error(f"❌ 모델 저장 실패: {str(e)}")
    
    st.markdown("---")
    st.markdown("### 🎯 새 모델 학습")
    
    # 학습 샘플 수 입력
    n_samples = st.number_input("학습 샘플 수", min_value=1000, max_value=10000, value=3000, step=500)
    if st.button("새 모델 학습", type="primary", use_container_width=True):
        results = train_model(n_samples)
        if results:
            st.success("✅ 모델 학습 완료!")
            st.info(f"정확도: {results['accuracy']:.4f}")
            st.info(f"F1 Score: {results['f1_score']:.4f}")
            st.info(f"최고 성능 모델: {results['best_model']}")
        else:
            st.error("❌ 모델 학습에 실패했습니다.")
    
    # 모델 상태
    st.markdown("---")
    st.markdown("### 📈 모델 상태")
    
    if st.session_state.model_loaded:
        if st.session_state.model_trained:
            st.success("✅ 모델 학습됨")
            st.info("새로 학습된 모델이 사용 중입니다.")
        else:
            st.info("✅ 모델 로드됨")
            st.info("기존 모델이 로드되었습니다.")
        
        # 모델 정보 표시
        if st.session_state.detector and st.session_state.detector.best_model:
            st.success("✅ 모델 준비 완료")
            st.info("텍스트 분석이 가능합니다.")
        else:
            st.warning("⚠️ 모델이 준비되지 않음")
            st.info("모델을 다시 로드하거나 학습해주세요.")
    else:
        st.warning("❌ 모델 없음")
        st.info("위에서 모델을 로드하거나 학습해주세요.")
    
    # 분석 히스토리
    if st.session_state.analysis_history:
        st.markdown("---")
        st.markdown("### 📝 최근 분석")
        st.info(f"총 {len(st.session_state.analysis_history)}회 분석 완료")

with tab2:
    st.markdown("### ⚙️ 분석 설정")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown("#### 🎯 위험도 설정")
        risk_threshold = st.selectbox(
            "위험도 임계값",
            ['low', 'medium', 'high', 'critical'],
            index=2,
            help="이 임계값을 초과하는 위험도는 경고를 표시합니다."
        )
        
        confidence_threshold = st.slider(
            "신뢰도 임계값",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="이 값보다 낮은 신뢰도는 경고를 표시합니다."
        )
    
    with col_b:
        st.markdown("#### 📊 분석 모드")
        analysis_mode = st.radio(
            "분석 모드",
            ["단일 분석", "배치 분석"],
            help="분석할 텍스트의 수에 따라 선택하세요."
        )
        
        # 추가 설정
        st.markdown("#### 🔍 고급 설정")
        enable_detailed_analysis = st.checkbox(
            "상세 분석 활성화",
            value=True,
            help="패턴 감지 및 상세 분석 결과를 표시합니다."
        )
        
        enable_confidence_warning = st.checkbox(
            "신뢰도 경고 활성화",
            value=True,
            help="낮은 신뢰도에 대한 경고를 표시합니다."
        )
    
    # RAG 설정
    st.markdown("---")
    st.markdown("#### 🤖 RAG 시스템 설정")
    
    # API 키 설정
    st.markdown("**🔑 Claude API 키 설정**")
    api_key_source = st.radio(
        "API 키 소스 선택",
        ["환경 변수 (.env)", "Streamlit Secrets", "직접 입력"],
        help="API 키를 어디서 가져올지 선택하세요."
    )
    
    if api_key_source == "직접 입력":
        claude_api_key = st.text_input(
            "Claude API 키",
            type="password",
            help="Anthropic에서 발급받은 API 키를 입력하세요."
        )
        if claude_api_key:
            os.environ["CLAUDE_API_KEY"] = claude_api_key
            st.success("✅ API 키가 설정되었습니다!")
    else:
        current_api_key = os.getenv("CLAUDE_API_KEY") or st.secrets.get("CLAUDE_API_KEY", "")
        if current_api_key:
            st.success(f"✅ API 키가 설정되어 있습니다. ({api_key_source})")
            st.info(f"키 길이: {len(current_api_key)} 문자")
        else:
            st.warning(f"⚠️ API 키가 설정되지 않았습니다. ({api_key_source})")
            
            # 도움말 표시
            with st.expander("🔧 API 키 설정 방법"):
                st.markdown("""
                ### Claude API 키 설정 방법
                
                1. **Anthropic 콘솔에서 API 키 발급**
                   - https://console.anthropic.com/ 접속
                   - 계정 생성 후 API 키 발급
                
                2. **환경 변수 파일 (.env) 사용 (권장)**
                   ```bash
                   # 프로젝트 루트에 .env 파일 생성
                   CLAUDE_API_KEY=your-api-key-here
                   ```
                
                3. **Streamlit Secrets 사용**
                   ```toml
                   # .streamlit/secrets.toml 파일에 추가
                   CLAUDE_API_KEY = "your-api-key-here"
                   ```
                
                4. **직접 입력**
                   - 위에서 "직접 입력" 선택 후 API 키 입력
                """)
    
    col_c, col_d = st.columns(2)
    
    with col_c:
        enable_rag = st.checkbox(
            "RAG 시스템 활성화",
            value=True,
            help="Claude Sonnet 3.5 기반 RAG 시스템을 사용합니다."
        )
        
        rag_confidence_threshold = st.slider(
            "RAG 신뢰도 임계값",
            min_value=0.0,
            max_value=1.0,
            value=0.8,
            step=0.1,
            help="이 값보다 낮은 신뢰도일 때 RAG 분석을 수행합니다."
        )
    
    with col_d:
        enable_caching = st.checkbox(
            "캐싱 활성화",
            value=True,
            help="분석 결과를 캐시하여 속도를 향상시킵니다."
        )
        
        enable_async = st.checkbox(
            "비동기 처리 활성화",
            value=True,
            help="백그라운드에서 정확도 개선을 수행합니다."
        )
    
    # 설정 저장
    st.markdown("---")
    st.markdown("#### 💾 설정 저장")
    
    col_e, col_f = st.columns(2)
    
    with col_e:
        if st.button("설정 저장", type="primary", use_container_width=True):
            # 설정을 세션 상태에 저장
            st.session_state.risk_threshold = risk_threshold
            st.session_state.confidence_threshold = confidence_threshold
            st.session_state.analysis_mode = analysis_mode
            st.session_state.enable_detailed_analysis = enable_detailed_analysis
            st.session_state.enable_confidence_warning = enable_confidence_warning
            st.session_state.enable_rag = enable_rag
            st.session_state.rag_confidence_threshold = rag_confidence_threshold
            st.session_state.enable_caching = enable_caching
            st.session_state.enable_async = enable_async
            st.success("✅ 설정이 저장되었습니다!")
    
    with col_f:
        if st.button("기본값으로 복원", type="secondary", use_container_width=True):
            st.session_state.risk_threshold = 'high'
            st.session_state.confidence_threshold = 0.7
            st.session_state.analysis_mode = "단일 분석"
            st.session_state.enable_detailed_analysis = True
            st.session_state.enable_confidence_warning = True
            st.session_state.enable_rag = True
            st.session_state.rag_confidence_threshold = 0.8
            st.session_state.enable_caching = True
            st.session_state.enable_async = True
            st.success("✅ 기본 설정으로 복원되었습니다!")
    
    # 현재 설정 표시
    st.markdown("---")
    st.markdown("#### 📋 현재 설정")
    
    # 환경 설정 상태 확인
    env_status = check_environment_setup()
    
    col_g, col_h = st.columns(2)
    
    with col_g:
        st.info(f"**위험도 임계값:** {risk_threshold}")
        st.info(f"**신뢰도 임계값:** {confidence_threshold}")
        st.info(f"**RAG 활성화:** {'예' if enable_rag else '아니오'}")
        
        # API 키 상태 표시
        if env_status["claude_api_key"]:
            st.success("✅ Claude API 키: 설정됨")
        else:
            st.warning("⚠️ Claude API 키: 설정되지 않음")
    
    with col_h:
        st.info(f"**분석 모드:** {analysis_mode}")
        st.info(f"**상세 분석:** {'활성화' if enable_detailed_analysis else '비활성화'}")
        st.info(f"**캐싱:** {'활성화' if enable_caching else '비활성화'}")
        
        # 환경 설정 상태 표시
        if env_status["enable_caching"]:
            st.success("✅ 캐싱: 활성화됨")
        else:
            st.warning("⚠️ 캐싱: 비활성화됨")
        
        if env_status["enable_async"]:
            st.success("✅ 비동기 처리: 활성화됨")
        else:
            st.warning("⚠️ 비동기 처리: 비활성화됨")

with tab3:
    st.markdown("### 📊 실시간 분석")
    
    if not st.session_state.model_loaded:
        st.warning("먼저 모델 관리 탭에서 모델을 로드하거나 학습해주세요.")
    else:
        # 텍스트 입력
        text_input = st.text_area(
            "분석할 텍스트를 입력하세요",
            height=150,
            placeholder="SAP 시스템에서 분석할 텍스트를 입력하세요..."
        )
        
        # 분석 버튼
        if st.button("🔍 분석 시작", type="primary", use_container_width=True):
            if text_input.strip():
                with st.spinner("분석 중..."):
                    # 하이브리드 분석 수행
                    result = analyze_text_hybrid(text_input, confidence_threshold)
                    
                    if result:
                        # 결과 표시
                        st.markdown("### 📊 분석 결과")
                        
                        # 위험도 표시
                        risk_level = result['predicted_risk']
                        confidence = result['confidence']
                        analysis_method = result.get('analysis_method', 'ml')
                        
                        col_a, col_b = st.columns(2)
                        
                        with col_a:
                            st.markdown(f"**위험도:** {risk_level.upper()}")
                            st.markdown(f"**신뢰도:** {confidence:.3f}")
                            st.markdown(f"**분석 방법:** {analysis_method.upper()}")
                            
                            # 신뢰도가 낮은 경우 경고 표시
                            if confidence < confidence_threshold:
                                st.warning(f"⚠️ 신뢰도가 낮습니다 ({confidence:.1%}). 결과를 주의 깊게 검토하세요.")
                            
                            # 위험도별 색상 표시 (SAP 스타일)
                            color = RISK_COLORS[risk_level]
                            st.markdown(f"""
                            <div style="background-color: {color}; padding: 15px; border-radius: 8px; color: white; text-align: center; font-weight: bold; font-size: 18px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
                                {risk_level.upper()}
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col_b:
                            # 게이지 차트
                            gauge_fig = create_risk_gauge(confidence, risk_level)
                            st.plotly_chart(gauge_fig, use_container_width=True)
                        
                        # 확률 분포
                        st.markdown("### 📈 확률 분포")
                        prob_fig = create_probability_chart(result['probabilities'])
                        st.plotly_chart(prob_fig, use_container_width=True)
                        
                        # RAG 분석 결과 (활성화된 경우)
                        if enable_rag and 'rag_analysis' in result:
                            st.markdown("### 🤖 RAG 분석 결과")
                            rag_result = result['rag_analysis']
                            
                            col_c, col_d = st.columns(2)
                            
                            with col_c:
                                st.markdown("**위협 유형:**")
                                st.info(rag_result.get('threat_type', 'unknown'))
                                
                                st.markdown("**추론 과정:**")
                                st.text_area("", value=rag_result.get('reasoning', ''), height=100, disabled=True)
                            
                            with col_d:
                                st.markdown("**권장 조치:**")
                                for action in rag_result.get('recommended_actions', []):
                                    st.info(f"• {action}")
                        
                        # 상세 분석 (SAP 스타일 카드)
                        if enable_detailed_analysis:
                            st.markdown("### 🔍 상세 분석")
                            analysis = result['detailed_analysis']
                            
                            col_e, col_f = st.columns(2)
                            
                            with col_e:
                                st.markdown("""
                                <div class="metric-card">
                                    <h4 style="margin: 0 0 10px 0; color: #323130;">SAP 관련 지표</h4>
                                </div>
                                """, unsafe_allow_html=True)
                                st.metric("SAP 트랜잭션 감지", analysis['sap_transaction_count'])
                                st.metric("인젝션 패턴 감지", analysis['injection_pattern_count'])
                                st.metric("텍스트 복잡도", analysis['text_complexity'])
                                st.metric("컨텍스트 특화 패턴", analysis.get('context_specific_pattern_count', 0))
                            
                            with col_f:
                                st.markdown("""
                                <div class="metric-card">
                                    <h4 style="margin: 0 0 10px 0; color: #323130;">보안 패턴 지표</h4>
                                </div>
                                """, unsafe_allow_html=True)
                                st.metric("역할 사칭 패턴 감지", analysis['role_impersonation_count'])
                                st.metric("민감 정보 접근 패턴 감지", analysis['sensitive_data_access_count'])
                                st.metric("언어 혼합", "예" if analysis['language_mix'] else "아니오")
                                st.metric("기술적 복잡도", analysis.get('technical_complexity_count', 0))
                            
                            # 패턴 차트
                            pattern_fig = create_pattern_chart(analysis)
                            st.plotly_chart(pattern_fig, use_container_width=True)
                        
                        # 임계값 경고 (SAP 스타일)
                        if RISK_LEVEL_MAPPING[risk_level] >= RISK_LEVEL_MAPPING[risk_threshold]:
                            st.markdown("""
                            <div style="background-color: #D13438; color: white; padding: 15px; border-radius: 8px; margin: 10px 0;">
                                <h4 style="margin: 0;">⚠️ 위험도 임계값 초과</h4>
                                <p style="margin: 5px 0 0 0;">설정된 위험도 임계값을 초과했습니다!</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        if confidence < confidence_threshold and enable_confidence_warning:
                            st.markdown("""
                            <div style="background-color: #FF8C00; color: white; padding: 15px; border-radius: 8px; margin: 10px 0;">
                                <h4 style="margin: 0;">ℹ️ 신뢰도 주의</h4>
                                <p style="margin: 5px 0 0 0;">신뢰도가 낮아 결과를 주의 깊게 검토하시기 바랍니다.</p>
                            </div>
                            """, unsafe_allow_html=True)
            else:
                st.error("분석할 텍스트를 입력해주세요.")

with tab4:
    st.markdown("### 📁 배치 분석")
    
    if not st.session_state.model_loaded:
        st.warning("먼저 모델 관리 탭에서 모델을 로드하거나 학습해주세요.")
    else:
        # 파일 업로드 또는 텍스트 입력
        upload_option = st.radio(
            "입력 방식 선택",
            ["파일 업로드", "텍스트 직접 입력"]
        )
        
        if upload_option == "파일 업로드":
            uploaded_file = st.file_uploader(
                "CSV 또는 TXT 파일을 업로드하세요",
                type=['csv', 'txt']
            )
            
            if uploaded_file is not None:
                try:
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                    else:
                        content = uploaded_file.read().decode('utf-8')
                        texts = content.split('\n')
                        df = pd.DataFrame({'text': [t.strip() for t in texts if t.strip()]})
                    
                    st.success(f"파일 로드 완료: {len(df)} 개의 텍스트")
                    
                    if st.button("배치 분석 시작", type="primary", use_container_width=True):
                        results_list = []
                        
                        with st.spinner("배치 분석 중..."):
                            progress_bar = st.progress(0)
                            
                            for i, row in df.iterrows():
                                text = row['text']
                                result = analyze_text_hybrid(text, confidence_threshold)
                                
                                if result:
                                    results_list.append({
                                        'text': text,
                                        'risk_level': result['predicted_risk'],
                                        'confidence': result['confidence'],
                                        'analysis_method': result.get('analysis_method', 'ml'),
                                        'sap_transaction_count': result['detailed_analysis']['sap_transaction_count'],
                                        'injection_pattern_count': result['detailed_analysis']['injection_pattern_count'],
                                        'role_impersonation_count': result['detailed_analysis']['role_impersonation_count'],
                                        'sensitive_data_access_count': result['detailed_analysis']['sensitive_data_access_count']
                                    })
                                
                                progress_bar.progress((i + 1) / len(df))
                        
                        # 결과 표시
                        if results_list:
                            results_df = pd.DataFrame(results_list)
                            
                            # 위험도별 통계
                            st.markdown("### 📊 위험도별 통계")
                            risk_stats = results_df['risk_level'].value_counts()
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                fig = px.pie(
                                    values=risk_stats.values,
                                    names=risk_stats.index,
                                    title="위험도 분포",
                                    color_discrete_map=RISK_COLORS
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            
                            with col2:
                                st.dataframe(risk_stats.reset_index().rename(columns={'index': '위험도', 'risk_level': '개수'}))
                            
                            # 결과 테이블
                            st.markdown("### 📋 분석 결과")
                            
                            # 정렬 옵션
                            sort_by = st.selectbox(
                                "정렬 기준",
                                ['위험도', '신뢰도', '분석 방법', 'SAP 트랜잭션', '인젝션 패턴']
                            )
                            
                            if sort_by == '위험도':
                                results_df = results_df.sort_values('risk_level', key=lambda x: x.map(RISK_LEVEL_MAPPING), ascending=False)
                            elif sort_by == '신뢰도':
                                results_df = results_df.sort_values('confidence', ascending=False)
                            elif sort_by == '분석 방법':
                                results_df = results_df.sort_values('analysis_method')
                            elif sort_by == 'SAP 트랜잭션':
                                results_df = results_df.sort_values('sap_transaction_count', ascending=False)
                            elif sort_by == '인젝션 패턴':
                                results_df = results_df.sort_values('injection_pattern_count', ascending=False)
                            
                            st.dataframe(results_df, use_container_width=True)
                            
                            # 다운로드
                            st.markdown("### 💾 결과 다운로드")
                            csv_link = download_csv(results_df, "sap_risk_analysis_results.csv")
                            st.markdown(csv_link, unsafe_allow_html=True)
        
                except Exception as e:
                    st.error(f"파일 처리 오류: {str(e)}")
        
        else:
            batch_text = st.text_area(
                "분석할 텍스트들을 입력하세요 (한 줄에 하나씩)",
                height=200,
                placeholder="텍스트 1\n텍스트 2\n텍스트 3"
            )
            
            if st.button("배치 분석 시작", type="primary", use_container_width=True):
                if batch_text.strip():
                    texts = [t.strip() for t in batch_text.split('\n') if t.strip()]
                    
                    results_list = []
                    
                    with st.spinner("배치 분석 중..."):
                        progress_bar = st.progress(0)
                        
                        for i, text in enumerate(texts):
                            result = analyze_text_hybrid(text, confidence_threshold)
                            
                            if result:
                                results_list.append({
                                    'text': text,
                                    'risk_level': result['predicted_risk'],
                                    'confidence': result['confidence'],
                                    'analysis_method': result.get('analysis_method', 'ml'),
                                    'sap_transaction_count': result['detailed_analysis']['sap_transaction_count'],
                                    'injection_pattern_count': result['detailed_analysis']['injection_pattern_count'],
                                    'role_impersonation_count': result['detailed_analysis']['role_impersonation_count'],
                                    'sensitive_data_access_count': result['detailed_analysis']['sensitive_data_access_count']
                                })
                            
                            progress_bar.progress((i + 1) / len(texts))
                    
                    # 결과 표시 (위와 동일)
                    if results_list:
                        results_df = pd.DataFrame(results_list)
                        
                        st.markdown("### 📊 위험도별 통계")
                        risk_stats = results_df['risk_level'].value_counts()
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            fig = px.pie(
                                values=risk_stats.values,
                                names=risk_stats.index,
                                title="위험도 분포",
                                color_discrete_map=RISK_COLORS
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            st.dataframe(risk_stats.reset_index().rename(columns={'index': '위험도', 'risk_level': '개수'}))
                        
                        st.markdown("### 📋 분석 결과")
                        st.dataframe(results_df, use_container_width=True)
                        
                        # 다운로드
                        st.markdown("### 💾 결과 다운로드")
                        csv_link = download_csv(results_df, "sap_risk_analysis_results.csv")
                        st.markdown(csv_link, unsafe_allow_html=True)
                else:
                    st.error("분석할 텍스트를 입력해주세요.")

with tab5:
    st.markdown("### 📚 지식베이스 관리")
    
    # 지식베이스 초기화
    if 'knowledge_base_manager' not in st.session_state:
        st.session_state.knowledge_base_manager = None
    
    # 지식베이스 상태 확인
    if st.session_state.knowledge_base_manager is None:
        st.info("지식베이스를 초기화하려면 아래 버튼을 클릭하세요.")
        if st.button("지식베이스 초기화", type="primary"):
            try:
                from src.rag_system import KnowledgeBaseManager, VectorStore
                from src.sample_knowledge import get_sample_knowledge
                
                vector_store = VectorStore()
                st.session_state.knowledge_base_manager = KnowledgeBaseManager(vector_store)
                
                # 샘플 데이터 추가
                sample_knowledge = get_sample_knowledge()
                for item in sample_knowledge:
                    metadata = {
                        "title": item['title'],
                        "category": item['category'],
                        "tags": item['tags'],
                        "source": "sample_data"
                    }
                    st.session_state.knowledge_base_manager.add_document(item['content'], metadata)
                
                st.success("✅ 지식베이스가 초기화되었습니다! (샘플 데이터 포함)")
            except Exception as e:
                st.error(f"❌ 지식베이스 초기화 실패: {str(e)}")
    
    else:
        kb_manager = st.session_state.knowledge_base_manager
        
        # 지식베이스 통계
        stats = kb_manager.get_knowledge_stats()
        
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            st.metric("총 문서 수", stats['total_documents'])
        
        with col_b:
            st.metric("지식베이스 크기", stats['knowledge_base_size'])
        
        with col_c:
            st.metric("마지막 업데이트", time.strftime('%Y-%m-%d %H:%M', time.localtime(stats['last_updated'])))
        
        st.markdown("---")
        
        # 문서 업로드 섹션
        st.markdown("#### 📄 문서 업로드")
        
        upload_option = st.radio(
            "업로드 방식",
            ["단일 문서", "배치 업로드", "텍스트 직접 입력"]
        )
        
        if upload_option == "단일 문서":
            uploaded_file = st.file_uploader(
                "문서 파일을 업로드하세요 (TXT, PDF, DOCX)",
                type=['txt', 'pdf', 'docx']
            )
            
            if uploaded_file is not None:
                if st.button("문서 추가", type="primary"):
                    try:
                        # 파일 내용 읽기
                        if uploaded_file.name.endswith('.txt'):
                            content = uploaded_file.read().decode('utf-8')
                        else:
                            st.warning("현재 TXT 파일만 지원됩니다.")
                            content = ""
                        
                        if content:
                            metadata = {
                                "filename": uploaded_file.name,
                                "upload_time": time.time(),
                                "source": "file_upload"
                            }
                            
                            kb_manager.add_document(content, metadata)
                            st.success(f"✅ 문서 '{uploaded_file.name}'이 추가되었습니다!")
                            
                    except Exception as e:
                        st.error(f"❌ 문서 추가 실패: {str(e)}")
        
        elif upload_option == "배치 업로드":
            uploaded_files = st.file_uploader(
                "여러 문서 파일을 업로드하세요",
                type=['txt'],
                accept_multiple_files=True
            )
            
            if uploaded_files and st.button("배치 추가", type="primary"):
                try:
                    documents = []
                    metadata_list = []
                    
                    for file in uploaded_files:
                        content = file.read().decode('utf-8')
                        metadata = {
                            "filename": file.name,
                            "upload_time": time.time(),
                            "source": "batch_upload"
                        }
                        
                        documents.append(content)
                        metadata_list.append(metadata)
                    
                    kb_manager.add_documents_batch(documents, metadata_list)
                    st.success(f"✅ {len(documents)}개의 문서가 추가되었습니다!")
                    
                except Exception as e:
                    st.error(f"❌ 배치 추가 실패: {str(e)}")
        
        else:  # 텍스트 직접 입력
            text_input = st.text_area(
                "문서 내용을 직접 입력하세요",
                height=200,
                placeholder="SAP 보안 관련 문서 내용을 입력하세요..."
            )
            
            if text_input.strip() and st.button("문서 추가", type="primary"):
                try:
                    metadata = {
                        "source": "direct_input",
                        "upload_time": time.time()
                    }
                    
                    kb_manager.add_document(text_input, metadata)
                    st.success("✅ 문서가 추가되었습니다!")
                    
                except Exception as e:
                    st.error(f"❌ 문서 추가 실패: {str(e)}")
        
        st.markdown("---")
        
        # 문서 검색 섹션
        st.markdown("#### 🔍 문서 검색")
        
        search_query = st.text_input(
            "검색할 키워드를 입력하세요",
            placeholder="SAP 보안, 권한 관리, 위협 탐지..."
        )
        
        if search_query and st.button("검색", type="primary"):
            try:
                search_results = kb_manager.search_documents(search_query, n_results=5)
                
                if search_results:
                    st.markdown("**검색 결과:**")
                    for i, result in enumerate(search_results):
                        with st.expander(f"문서 {i+1} (유사도: {1-result['distance']:.3f})"):
                            st.text(result['document'][:500] + "...")
                            st.caption(f"메타데이터: {result['metadata']}")
                else:
                    st.info("검색 결과가 없습니다.")
                    
            except Exception as e:
                st.error(f"❌ 검색 실패: {str(e)}")
        
        st.markdown("---")
        
        # 지식베이스 관리
        st.markdown("#### ⚙️ 지식베이스 관리")
        
        col_d, col_e = st.columns(2)
        
        with col_d:
            if st.button("지식베이스 통계 새로고침", type="secondary"):
                stats = kb_manager.get_knowledge_stats()
                st.success("✅ 통계가 업데이트되었습니다!")
        
        with col_e:
            if st.button("지식베이스 초기화", type="secondary"):
                st.session_state.knowledge_base_manager = None
                st.success("✅ 지식베이스가 초기화되었습니다!")
                st.rerun() 