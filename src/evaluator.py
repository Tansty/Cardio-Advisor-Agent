"""
目标函数与约束评估模块 (Decision Objective and Constraint Modeling)
论文 Section 3.3

核心功能:
- 候选策略集 A = {a_1, a_2, ..., a_m}                         (公式4)
- 最优决策 a* = argmax U(a|S̃_t)                               (公式5)
- 效用函数 U(a|S̃_t) = αR(a|S̃_t) + βG(a) - γP(a)            (公式6)
- 安全约束 C_safety(a) = 0                                     (公式7)
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.config import AgentConfig
from src.data_loader import PatientRecord
from src.guideline_tools import TreatmentStrategy

logger = logging.getLogger(__name__)


class SafetyConstraintChecker:
    """安全约束检查器 (公式7: C_safety(a) = 0)"""

    def __init__(self, config: AgentConfig):
        self.config = config

    def check_safety(
        self, strategy: TreatmentStrategy, patient: PatientRecord
    ) -> Tuple[bool, List[str]]:
        """
        检查策略是否违反安全约束
        返回: (is_safe, violation_reasons)
        """
        violations = []
        features = patient.features

        # 1. 检查年龄约束
        age = features.get("age", 0)
        if age < self.config.safety_age_min or age > self.config.safety_age_max:
            violations.append(f"年龄({age})超出安全范围[{self.config.safety_age_min}, {self.config.safety_age_max}]")

        # 2. 检查禁忌症
        for contraindication in strategy.contraindications:
            if self._check_contraindication(contraindication, patient):
                violations.append(f"存在禁忌症: {contraindication}")

        # 3. 检查血压安全范围（降压药物策略）
        if strategy.category == "medication" and "bp" in strategy.strategy_id.lower():
            sbp = features.get("systolic_bp", 120)
            dbp = features.get("diastolic_bp", 80)
            # 如果血压已经很低，不应再降压
            if sbp < self.config.safety_sbp_min:
                violations.append(f"收缩压({sbp})已过低，不宜使用降压药物")
            if dbp < self.config.safety_dbp_min:
                violations.append(f"舒张压({dbp})已过低，不宜使用降压药物")

        # 4. 年龄特定的药物限制
        if "antiplatelet" in strategy.strategy_id and age < 40:
            violations.append("年龄<40岁，不推荐阿司匹林一级预防")

        # 5. 极端血压值检查
        sbp = features.get("systolic_bp", 120)
        dbp = features.get("diastolic_bp", 80)
        if sbp > self.config.safety_sbp_max or dbp > self.config.safety_dbp_max:
            if strategy.category == "lifestyle" and "exercise" in strategy.strategy_id:
                violations.append(f"血压过高({sbp}/{dbp})，暂不建议剧烈运动")

        is_safe = len(violations) == 0
        return is_safe, violations

    def _check_contraindication(self, contraindication: str, patient: PatientRecord) -> bool:
        """检查患者是否存在特定禁忌症"""
        features = patient.features
        # 妊娠相关
        if "妊娠" in contraindication:
            # 简化判断: 年龄在育龄且女性
            if features.get("gender", 1) == 2 and 15 <= features.get("age", 0) <= 50:
                return False  # 需要更多信息判断，默认不禁忌
        return False  # 默认无禁忌（实际中需更精细判断）


class UtilityEvaluator:
    """
    效用函数评估器
    U(a|S̃_t) = αR(a|S̃_t) + βG(a) - γP(a)  (公式6)
    """

    def __init__(self, config: AgentConfig):
        self.config = config
        self.safety_checker = SafetyConstraintChecker(config)

    def evaluate_strategy(
        self,
        strategy: TreatmentStrategy,
        patient: PatientRecord,
        weighted_state: np.ndarray,
        risk_profile: Dict,
        conflict_penalty: float = 0.0,
    ) -> Dict:
        """
        评估单个策略的效用值
        返回评估结果字典
        """
        alpha = self.config.utility.alpha
        beta = self.config.utility.beta
        gamma = self.config.utility.gamma

        # R(a|S̃_t): 风险降低收益
        risk_reduction = self._compute_risk_reduction(strategy, patient, weighted_state, risk_profile)

        # G(a): 指南一致性评分
        guideline_consistency = strategy.guideline_consistency

        # P(a): 冲突/违规惩罚
        penalty = conflict_penalty + self._compute_penalty(strategy, patient)

        # U(a|S̃_t) = αR + βG - γP  (公式6)
        utility = alpha * risk_reduction + beta * guideline_consistency - gamma * penalty

        # 安全约束检查 C_safety(a) = 0  (公式7)
        is_safe, violations = self.safety_checker.check_safety(strategy, patient)

        return {
            "strategy": strategy,
            "utility": utility,
            "risk_reduction": risk_reduction,
            "guideline_consistency": guideline_consistency,
            "penalty": penalty,
            "is_safe": is_safe,
            "safety_violations": violations,
            "confidence": strategy.confidence,
        }

    def evaluate_and_rank(
        self,
        strategies: List[TreatmentStrategy],
        patient: PatientRecord,
        weighted_state: np.ndarray,
        risk_profile: Dict,
        conflict_penalties: Optional[Dict[str, float]] = None,
    ) -> List[Dict]:
        """
        评估所有候选策略并排序
        a* = argmax U(a|S̃_t)  (公式5)
        """
        if conflict_penalties is None:
            conflict_penalties = {}

        evaluations = []
        for strategy in strategies:
            conflict_penalty = conflict_penalties.get(strategy.strategy_id, 0.0)
            eval_result = self.evaluate_strategy(
                strategy, patient, weighted_state, risk_profile, conflict_penalty
            )
            evaluations.append(eval_result)

        # 过滤不安全的策略 (C_safety(a) = 0)
        safe_evaluations = [e for e in evaluations if e["is_safe"]]
        unsafe_evaluations = [e for e in evaluations if not e["is_safe"]]

        if unsafe_evaluations:
            logger.info(
                f"安全约束过滤了 {len(unsafe_evaluations)} 个不安全策略"
            )

        # 按效用值排序
        safe_evaluations.sort(key=lambda x: x["utility"], reverse=True)

        return safe_evaluations

    def _compute_risk_reduction(
        self,
        strategy: TreatmentStrategy,
        patient: PatientRecord,
        weighted_state: np.ndarray,
        risk_profile: Dict,
    ) -> float:
        """
        计算策略的风险降低收益 R(a|S̃_t)
        基于策略固有降低值和患者特定的风险匹配度
        """
        base_reduction = strategy.risk_reduction
        risk_score = risk_profile.get("risk_score", 0.5)

        # 风险匹配调整: 高风险患者从药物治疗获益更大
        if strategy.category == "medication":
            risk_multiplier = 0.8 + 0.4 * risk_score  # [0.8, 1.2]
        elif strategy.category == "lifestyle":
            risk_multiplier = 1.0 + 0.2 * (1 - risk_score)  # 低风险患者从生活方式获益更多
        else:
            risk_multiplier = 1.0

        # 特征匹配度: 策略与患者特征的匹配程度
        feature_match = self._compute_feature_match(strategy, patient)

        adjusted_reduction = base_reduction * risk_multiplier * feature_match
        return min(adjusted_reduction, 1.0)

    def _compute_feature_match(
        self, strategy: TreatmentStrategy, patient: PatientRecord
    ) -> float:
        """计算策略与患者特征的匹配度"""
        features = patient.features
        conditions = strategy.conditions
        match_score = 1.0

        if "sbp_threshold" in conditions:
            sbp = features.get("systolic_bp", 0)
            if sbp >= conditions["sbp_threshold"]:
                match_score *= 1.1
            else:
                match_score *= 0.7

        if "ldl_c_threshold" in conditions:
            ldl = features.get("ldl_c", 0)
            if ldl >= conditions["ldl_c_threshold"]:
                match_score *= 1.1
            else:
                match_score *= 0.7

        if "diabetes" in conditions and conditions["diabetes"]:
            if features.get("diabetes", 0) == 1:
                match_score *= 1.1
            else:
                match_score *= 0.8

        if "bmi_threshold" in conditions:
            bmi = features.get("bmi", 0)
            if bmi >= conditions["bmi_threshold"]:
                match_score *= 1.1
            else:
                match_score *= 0.7

        return min(match_score, 1.5)

    def _compute_penalty(
        self, strategy: TreatmentStrategy, patient: PatientRecord
    ) -> float:
        """
        计算策略的惩罚项 P(a)
        基于禁忌症风险和策略适用性
        """
        penalty = 0.0

        # 禁忌症数量惩罚
        penalty += len(strategy.contraindications) * 0.05

        # 低置信度惩罚
        if strategy.confidence < 0.7:
            penalty += (0.7 - strategy.confidence) * 0.3

        # 低指南一致性惩罚
        if strategy.guideline_consistency < 0.8:
            penalty += (0.8 - strategy.guideline_consistency) * 0.2

        return penalty

    def select_optimal(self, ranked_evaluations: List[Dict]) -> Optional[Dict]:
        """选择最优策略 a* = argmax U(a|S̃_t)"""
        if not ranked_evaluations:
            return None
        return ranked_evaluations[0]

