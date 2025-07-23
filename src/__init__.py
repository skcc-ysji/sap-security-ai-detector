"""SAP 위험도 탐지 시스템 패키지"""

from .sap_risk_detector import SAPRiskDetector
from .constants import (
    RISK_LEVEL_MAPPING,
    RISK_KEYWORDS,
    SAP_CONTEXT_WEIGHTS,
    INJECTION_PATTERNS
)

__version__ = '2.0.0'
__author__ = '지영석'
