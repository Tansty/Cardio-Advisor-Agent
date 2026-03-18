"""
多步规划与推理机制 (Multi-Step Planning and Reasoning Mechanism)
论文 Section 3.5

核心功能:
- 多步规划: S_t → A_t → S_{t+1}                              (公式10)
- 五个阶段:
  1. 识别主要风险因素
  2. 调用多源指南工具生成候选策略
  3. 冲突检测
  4. 效用函数评估与排序
  5. 输出最优决策和推理轨迹
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from src.config import AgentConfig
from src.data_loader import PatientRecord
from src.patient_state import PatientStateModeler
from src.guideline_tools import GuidelineToolSystem, TreatmentStrategy
from src.evaluator import UtilityEvaluator
from src.conflict_resolver import ConflictDetector, ConflictResolver, ConflictRecord

logger = logging.getLogger(__name__)


@dataclass
class ReasoningStep:
    """推理步骤记录"""
    step_id: int
    step_name: str
    description: str
    details: Dict = field(default_factory=dict)


@dataclass
class DecisionResult:
    """决策结果"""
    patient_id: str
    optimal_strategy: Optional[TreatmentStrategy]
    all_evaluations: List[Dict]
    risk_profile: Dict
    conflicts: List[ConflictRecord]
    reasoning_trajectory: List[ReasoningStep]
    decision_confidence: float
    predicted_label: int  # 0: 低风险/不需要积极干预, 1: 高风险/需要积极干预


class MultiStepPlanner:
    """
    多步规划器
    实现 S_t → A_t → S_{t+1} 的多步规划推理 (公式10)
    """

    def __init__(
        self,
        config: AgentConfig,
        state_modeler: PatientStateModeler,
        guideline_system: GuidelineToolSystem,
        evaluator: UtilityEvaluator,
        conflict_detector: ConflictDetector,
        conflict_resolver: ConflictResolver,
    ):
        self.config = config
        self.state_modeler = state_modeler
        self.guideline_system = guideline_system
        self.evaluator = evaluator
        self.conflict_detector = conflict_detector
        self.conflict_resolver = conflict_resolver

    def plan(self, patient: PatientRecord) -> DecisionResult:
        """
        执行多步规划
        五个阶段的决策过程
        """
        reasoning_trajectory = []

        # ========== Stage 1: 识别主要风险因素 ==========
        risk_profile = self.state_modeler.get_risk_profile(patient)
        weighted_state = risk_profile["weighted_state"]

        step1 = ReasoningStep(
            step_id=1,
            step_name="风险因素识别",
            description=(
                f"患者 {patient.patient_id} 风险评分: {risk_profile['risk_score']:.3f}, "
                f"风险等级: {risk_profile['risk_level']}"
            ),
            details={
                "risk_score": risk_profile["risk_score"],
                "risk_level": risk_profile["risk_level"],
                "dominant_factors": risk_profile["dominant_risk_factors"],
            },
        )
        reasoning_trajectory.append(step1)

        # ========== Stage 2: 调用多源指南工具生成候选策略 ==========
        candidate_strategies = self.guideline_system.generate_all_strategies(
            patient, weighted_state, risk_profile
        )

        step2 = ReasoningStep(
            step_id=2,
            step_name="候选策略生成",
            description=f"从 {len(self.guideline_system.tools)} 个指南工具生成了 {len(candidate_strategies)} 个候选策略",
            details={
                "num_strategies": len(candidate_strategies),
                "guidelines_invoked": self.guideline_system.get_guideline_names(),
                "strategy_categories": self._count_categories(candidate_strategies),
            },
        )
        reasoning_trajectory.append(step2)

        # ========== Stage 3: 冲突检测 ==========
        conflicts = self.conflict_detector.detect_conflicts(candidate_strategies)

        step3 = ReasoningStep(
            step_id=3,
            step_name="冲突检测",
            description=f"检测到 {len(conflicts)} 个跨指南冲突",
            details={
                "num_conflicts": len(conflicts),
                "conflict_types": [c.conflict_type for c in conflicts],
            },
        )
        reasoning_trajectory.append(step3)

        # 冲突解决
        conflict_penalties = {}
        if conflicts:
            conflict_penalties, resolved_conflicts = self.conflict_resolver.resolve_conflicts(
                conflicts, patient, risk_profile
            )
            resolution_rate = self.conflict_resolver.get_conflict_resolution_rate(conflicts)

            step3_resolve = ReasoningStep(
                step_id=3,
                step_name="冲突解决",
                description=f"冲突解决率: {resolution_rate:.2%}",
                details={
                    "resolution_rate": resolution_rate,
                    "penalties_assigned": len(conflict_penalties),
                },
            )
            reasoning_trajectory.append(step3_resolve)

        # ========== Stage 4: 效用评估与排序 ==========
        evaluations = self.evaluator.evaluate_and_rank(
            candidate_strategies, patient, weighted_state, risk_profile, conflict_penalties
        )

        step4 = ReasoningStep(
            step_id=4,
            step_name="效用评估与排序",
            description=(
                f"评估了 {len(candidate_strategies)} 个策略, "
                f"{len(evaluations)} 个通过安全约束"
            ),
            details={
                "total_evaluated": len(candidate_strategies),
                "safe_strategies": len(evaluations),
                "top3_utilities": [
                    {
                        "strategy": e["strategy"].strategy_id,
                        "utility": round(e["utility"], 4),
                    }
                    for e in evaluations[:3]
                ] if evaluations else [],
            },
        )
        reasoning_trajectory.append(step4)

        # ========== Stage 5: 输出最优决策 ==========
        optimal = self.evaluator.select_optimal(evaluations)

        # 计算决策置信度
        decision_confidence = self._compute_decision_confidence(evaluations)

        # 预测标签: 基于风险评估和策略推荐
        predicted_label = self._predict_label(risk_profile, evaluations)

        optimal_strategy = optimal["strategy"] if optimal else None

        step5 = ReasoningStep(
            step_id=5,
            step_name="最优决策输出",
            description=(
                f"最优策略: {optimal_strategy.strategy_id if optimal_strategy else 'None'}, "
                f"决策置信度: {decision_confidence:.3f}, "
                f"预测: {'需要积极干预' if predicted_label == 1 else '低风险'}"
            ),
            details={
                "optimal_strategy": optimal_strategy.strategy_id if optimal_strategy else None,
                "optimal_utility": optimal["utility"] if optimal else 0,
                "decision_confidence": decision_confidence,
                "predicted_label": predicted_label,
            },
        )
        reasoning_trajectory.append(step5)

        return DecisionResult(
            patient_id=patient.patient_id,
            optimal_strategy=optimal_strategy,
            all_evaluations=evaluations,
            risk_profile=risk_profile,
            conflicts=conflicts,
            reasoning_trajectory=reasoning_trajectory,
            decision_confidence=decision_confidence,
            predicted_label=predicted_label,
        )

    def _count_categories(self, strategies: List[TreatmentStrategy]) -> Dict[str, int]:
        """统计策略类别分布"""
        counts = {}
        for s in strategies:
            counts[s.category] = counts.get(s.category, 0) + 1
        return counts

    def _compute_decision_confidence(self, evaluations: List[Dict]) -> float:
        """
        计算决策置信度
        基于top策略的效用值分布
        """
        if not evaluations:
            return 0.0

        if len(evaluations) == 1:
            return evaluations[0]["confidence"]

        # 基于top-1和top-2效用差距
        top1_utility = evaluations[0]["utility"]
        top2_utility = evaluations[1]["utility"]
        gap = top1_utility - top2_utility

        # 基于策略自身置信度
        strategy_confidence = evaluations[0]["confidence"]

        # 综合置信度
        confidence = 0.5 * strategy_confidence + 0.3 * min(gap * 2, 1.0) + 0.2 * min(top1_utility, 1.0)
        return min(max(confidence, 0.0), 1.0)

    def _predict_label(self, risk_profile: Dict, evaluations: List[Dict]) -> int:
        """
        基于风险评估和策略分析预测标签
        0: 低风险/不需要积极干预
        1: 高风险/需要积极干预(存在心血管疾病风险)
        """
        risk_score = risk_profile.get("risk_score", 0.5)
        risk_level = risk_profile.get("risk_level", "medium")

        # 基于风险评分
        if risk_level in ["high", "very_high"]:
            base_prediction = 1
        elif risk_level == "low":
            base_prediction = 0
        else:
            base_prediction = 1 if risk_score >= 0.4 else 0

        # 基于策略推荐的调整
        if evaluations:
            # 如果最优策略是高风险降低药物治疗, 倾向于预测阳性
            top_strategy = evaluations[0]["strategy"]
            if (top_strategy.category == "medication" and
                top_strategy.risk_reduction >= 0.3):
                return 1

            # 如果只有生活方式建议, 倾向于预测阴性
            all_lifestyle = all(
                e["strategy"].category == "lifestyle" for e in evaluations[:3]
            )
            if all_lifestyle and risk_score < 0.35:
                return 0

        return base_prediction

    def format_reasoning_trajectory(self, result: DecisionResult) -> str:
        """格式化推理轨迹为可读文本"""
        lines = [
            f"{'='*60}",
            f"患者 {result.patient_id} 决策推理轨迹",
            f"{'='*60}",
        ]

        for step in result.reasoning_trajectory:
            lines.append(f"\n[Step {step.step_id}] {step.step_name}")
            lines.append(f"  {step.description}")

        lines.append(f"\n{'='*60}")
        lines.append(f"最终决策: {'需要积极干预' if result.predicted_label == 1 else '低风险'}")
        lines.append(f"决策置信度: {result.decision_confidence:.3f}")

        if result.optimal_strategy:
            lines.append(f"最优策略: {result.optimal_strategy.recommendation}")
            lines.append(f"来源指南: {result.optimal_strategy.source_guideline}")

        return "\n".join(lines)

