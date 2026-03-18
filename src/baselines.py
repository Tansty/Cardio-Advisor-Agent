"""
基线方法实现
论文 Section 4.3

基线方法:
1. Single Guideline Rule - 单指南规则方法
2. RAG-only - 检索增强生成方法
3. w/o Conflict - 无冲突解决的框架变体

对比表 (Table 1):
| Method               | Guideline Usage        | Conflict Handling | Optimization |
| Single Guideline Rule| Single                 | None              | None         |
| RAG-only             | Multi (retrieval)      | Implicit          | None         |
| w/o Conflict         | Multi (structured)     | None              | Partial      |
| Cardio-Advisor Agent | Multi (structured)     | Explicit          | Full         |
"""

import logging
from typing import Dict, List

import numpy as np

from src.config import AgentConfig
from src.data_loader import PatientRecord
from src.patient_state import PatientStateModeler
from src.guideline_tools import (
    ESCGuidelineTool,
    GuidelineToolSystem,
    TreatmentStrategy,
)
from src.evaluator import UtilityEvaluator
from src.planner import DecisionResult

logger = logging.getLogger(__name__)


class SingleGuidelineRule:
    """
    单指南规则方法 (Single Guideline Rule)
    仅依赖单一临床指南的预定义规则进行决策
    """

    def __init__(self, config: AgentConfig):
        self.config = config
        self.state_modeler = PatientStateModeler(config)
        self.guideline_tool = ESCGuidelineTool()  # 只使用一个指南

    def predict(self, patient: PatientRecord) -> int:
        """预测标签"""
        risk_profile = self.state_modeler.get_risk_profile(patient)
        weighted_state = risk_profile["weighted_state"]

        # 使用单一指南生成策略
        strategies = self.guideline_tool.generate_strategies(
            patient, weighted_state, risk_profile
        )

        # 简单规则: 如果有药物治疗建议, 预测为阳性
        has_medication = any(s.category == "medication" for s in strategies)
        risk_level = risk_profile.get("risk_level", "medium")

        if risk_level in ["high", "very_high"]:
            return 1
        elif has_medication and risk_level == "medium":
            return 1
        else:
            return 0

    def predict_batch(self, patients: List[PatientRecord]) -> List[int]:
        return [self.predict(p) for p in patients]


class RAGOnlyMethod:
    """
    RAG-only 方法
    使用多源指南检索, 但无显式冲突处理和决策优化
    通过隐式方式(简单加权平均)整合多指南知识
    """

    def __init__(self, config: AgentConfig, llm_client=None):
        self.config = config
        self.state_modeler = PatientStateModeler(config)
        self.guideline_system = GuidelineToolSystem(llm_client)
        self.llm_client = llm_client

    def predict(self, patient: PatientRecord) -> int:
        """预测标签"""
        risk_profile = self.state_modeler.get_risk_profile(patient)
        weighted_state = risk_profile["weighted_state"]

        # 使用多个指南生成策略 (检索阶段)
        strategies = self.guideline_system.generate_all_strategies(
            patient, weighted_state, risk_profile
        )

        if not strategies:
            return 0

        # 隐式整合: 简单加权平均, 无显式冲突处理
        avg_risk_reduction = np.mean([s.risk_reduction for s in strategies])
        avg_consistency = np.mean([s.guideline_consistency for s in strategies])
        medication_ratio = sum(1 for s in strategies if s.category == "medication") / len(strategies)

        risk_score = risk_profile.get("risk_score", 0.5)

        # 简单阈值决策 (无优化)
        combined_score = (
            0.3 * risk_score
            + 0.3 * avg_risk_reduction
            + 0.2 * avg_consistency
            + 0.2 * medication_ratio
        )

        return 1 if combined_score >= 0.4 else 0

    def predict_batch(self, patients: List[PatientRecord]) -> List[int]:
        return [self.predict(p) for p in patients]

