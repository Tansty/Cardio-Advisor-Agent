"""
Cardio-Advisor Agent 主体模块
论文 Section 3.1 System Overview

集成五个核心模块:
1. Patient State Modeling (患者状态建模)
2. Multi-Source Guideline Knowledge Tools (多源指南知识工具)
3. Candidate Strategy Generation (候选策略生成)
4. Objective and Constraint Evaluation (目标与约束评估)
5. Multi-Step Planning with Conflict Resolution (多步规划与冲突解决)
"""

import logging
from typing import Dict, List, Optional

from src.config import AgentConfig
from src.data_loader import PatientRecord
from src.patient_state import PatientStateModeler
from src.guideline_tools import GuidelineToolSystem
from src.evaluator import UtilityEvaluator
from src.conflict_resolver import ConflictDetector, ConflictResolver
from src.planner import MultiStepPlanner, DecisionResult

logger = logging.getLogger(__name__)


class CardioAdvisorAgent:
    """
    Cardio-Advisor Agent
    结构化决策优化框架, 通过多指南LLM集成提供个性化心血管支持
    """

    def __init__(self, config: AgentConfig, llm_client=None):
        """
        初始化Agent
        Args:
            config: Agent配置
            llm_client: LLM客户端 (可选, None则使用规则方法)
        """
        self.config = config
        self.llm_client = llm_client

        # 初始化核心模块
        self.state_modeler = PatientStateModeler(config, llm_client)
        self.guideline_system = GuidelineToolSystem(llm_client)
        self.evaluator = UtilityEvaluator(config)
        self.conflict_detector = ConflictDetector(config)
        self.conflict_resolver = ConflictResolver(config, llm_client)

        # 多步规划器
        self.planner = MultiStepPlanner(
            config=config,
            state_modeler=self.state_modeler,
            guideline_system=self.guideline_system,
            evaluator=self.evaluator,
            conflict_detector=self.conflict_detector,
            conflict_resolver=self.conflict_resolver,
        )

        logger.info("Cardio-Advisor Agent 初始化完成")

    def predict(self, patient: PatientRecord) -> DecisionResult:
        """
        对单个患者进行决策预测
        Args:
            patient: 患者记录
        Returns:
            DecisionResult: 完整的决策结果
        """
        return self.planner.plan(patient)

    def predict_batch(self, patients: List[PatientRecord]) -> List[DecisionResult]:
        """
        批量预测
        """
        results = []
        for patient in patients:
            try:
                result = self.predict(patient)
                results.append(result)
            except Exception as e:
                logger.error(f"患者 {patient.patient_id} 决策失败: {e}")
                # 创建默认结果
                results.append(self._create_default_result(patient))
        return results

    def predict_label(self, patient: PatientRecord) -> int:
        """简化预测接口: 只返回标签"""
        result = self.predict(patient)
        return result.predicted_label

    def predict_labels_batch(self, patients: List[PatientRecord]) -> List[int]:
        """批量标签预测"""
        results = self.predict_batch(patients)
        return [r.predicted_label for r in results]

    def get_decision_report(self, patient: PatientRecord) -> str:
        """获取完整的决策报告"""
        result = self.predict(patient)
        return self.planner.format_reasoning_trajectory(result)

    def _create_default_result(self, patient: PatientRecord) -> DecisionResult:
        """创建默认决策结果(处理异常情况)"""
        return DecisionResult(
            patient_id=patient.patient_id,
            optimal_strategy=None,
            all_evaluations=[],
            risk_profile={"risk_score": 0.5, "risk_level": "medium"},
            conflicts=[],
            reasoning_trajectory=[],
            decision_confidence=0.0,
            predicted_label=0,
        )


class CardioAdvisorAgentNoConflict(CardioAdvisorAgent):
    """
    无冲突解决的变体 (w/o Conflict)
    用于消融实验, 移除冲突检测和解决模块
    """

    def predict(self, patient: PatientRecord) -> DecisionResult:
        """跳过冲突检测和解决"""
        reasoning_trajectory = []

        # Stage 1: 风险因素识别
        risk_profile = self.state_modeler.get_risk_profile(patient)
        weighted_state = risk_profile["weighted_state"]

        # Stage 2: 候选策略生成
        strategies = self.guideline_system.generate_all_strategies(
            patient, weighted_state, risk_profile
        )

        # Stage 3: 跳过冲突检测 (直接评估)

        # Stage 4: 评估与排序 (无冲突惩罚)
        evaluations = self.evaluator.evaluate_and_rank(
            strategies, patient, weighted_state, risk_profile
        )

        # Stage 5: 输出决策
        optimal = self.evaluator.select_optimal(evaluations)
        optimal_strategy = optimal["strategy"] if optimal else None

        predicted_label = self.planner._predict_label(risk_profile, evaluations)
        confidence = self.planner._compute_decision_confidence(evaluations)

        return DecisionResult(
            patient_id=patient.patient_id,
            optimal_strategy=optimal_strategy,
            all_evaluations=evaluations,
            risk_profile=risk_profile,
            conflicts=[],
            reasoning_trajectory=reasoning_trajectory,
            decision_confidence=confidence,
            predicted_label=predicted_label,
        )


class CardioAdvisorAgentNoPersonalization(CardioAdvisorAgent):
    """
    无个性化的变体 (w/o Personalization)
    用于消融实验, 使用均匀权重代替LLM权重
    """

    def __init__(self, config: AgentConfig, llm_client=None):
        super().__init__(config, llm_client)
        # 覆盖状态建模器的权重方法
        self._original_compute_weights = self.state_modeler.compute_weights
        self.state_modeler.compute_weights = self._uniform_weights

    def _uniform_weights(self, patient: PatientRecord):
        """返回均匀权重 (无个性化)"""
        n = len(patient.get_feature_names())
        return np.ones(n) * 0.5


class CardioAdvisorAgentNoPlanning(CardioAdvisorAgent):
    """
    无多步规划的变体 (w/o Planning)
    用于消融实验, 使用单步决策代替多步规划
    """

    def predict(self, patient: PatientRecord) -> DecisionResult:
        """单步决策: 直接从第一个指南获取推荐"""
        risk_profile = self.state_modeler.get_risk_profile(patient)
        weighted_state = risk_profile["weighted_state"]

        # 只使用第一个指南工具
        strategies = self.guideline_system.tools[0].generate_strategies(
            patient, weighted_state, risk_profile
        )

        evaluations = self.evaluator.evaluate_and_rank(
            strategies, patient, weighted_state, risk_profile
        )

        optimal = self.evaluator.select_optimal(evaluations)
        optimal_strategy = optimal["strategy"] if optimal else None
        predicted_label = self.planner._predict_label(risk_profile, evaluations)
        confidence = self.planner._compute_decision_confidence(evaluations)

        return DecisionResult(
            patient_id=patient.patient_id,
            optimal_strategy=optimal_strategy,
            all_evaluations=evaluations,
            risk_profile=risk_profile,
            conflicts=[],
            reasoning_trajectory=[],
            decision_confidence=confidence,
            predicted_label=predicted_label,
        )


class CardioAdvisorAgentNoWeight(CardioAdvisorAgent):
    """
    无个性化权重的变体 (w/o Personalization Weight)
    用于消融实验, 保留个性化但使用固定权重
    """

    def __init__(self, config: AgentConfig, llm_client=None):
        super().__init__(config, llm_client)
        self.state_modeler.compute_weights = self._fixed_weights

    def _fixed_weights(self, patient: PatientRecord):
        """固定权重: 按照通用心血管风险因素重要性"""
        feature_names = patient.get_feature_names()
        fixed = {
            "age": 0.8, "gender": 0.5, "height": 0.2, "weight": 0.4, "bmi": 0.7,
            "systolic_bp": 0.85, "diastolic_bp": 0.85, "cholesterol": 0.75,
            "glucose": 0.6, "smoking": 0.8, "alcohol": 0.4, "physical_activity": 0.5,
            "diabetes": 0.8, "hypertension": 0.85, "family_history": 0.6,
            "heart_rate": 0.5, "ldl_c": 0.8, "hdl_c": 0.6,
            "total_cholesterol": 0.75, "triglycerides": 0.6,
        }
        return np.array([fixed.get(name, 0.5) for name in feature_names])


class CardioAdvisorAgentNoMultiSource(CardioAdvisorAgent):
    """
    无多源指南的变体 (w/o Multi-Source Guidelines)
    用于消融实验, 只使用单个指南
    """

    def __init__(self, config: AgentConfig, llm_client=None):
        super().__init__(config, llm_client)
        # 只保留第一个指南工具
        self.guideline_system.tools = self.guideline_system.tools[:1]


import numpy as np

