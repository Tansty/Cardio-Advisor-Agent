"""
冲突检测与解决模块 (Conflict Detection and Resolution)
论文 Section 3.4 & 3.5

核心功能:
- 检测来自不同指南的候选策略之间的矛盾
- 基于效用函数重新评估和整合冲突策略
- 为冲突策略分配惩罚分数
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.config import AgentConfig
from src.data_loader import PatientRecord
from src.guideline_tools import TreatmentStrategy

logger = logging.getLogger(__name__)


@dataclass
class ConflictRecord:
    """冲突记录"""
    conflict_id: str
    strategies: List[TreatmentStrategy]  # 冲突的策略列表
    conflict_type: str                    # 冲突类型
    description: str                      # 冲突描述
    resolution: Optional[str] = None      # 解决结果
    resolved: bool = False
    winner_id: Optional[str] = None       # 获胜策略ID


class ConflictDetector:
    """冲突检测器"""

    def __init__(self, config: AgentConfig):
        self.config = config
        self.conflict_threshold = config.conflict_threshold

    def detect_conflicts(
        self, strategies: List[TreatmentStrategy]
    ) -> List[ConflictRecord]:
        """
        检测候选策略之间的冲突
        冲突类型:
        1. 目标值冲突 (不同指南推荐不同的目标值)
        2. 药物类别冲突 (推荐互斥的药物)
        3. 治疗强度冲突 (同类治疗的不同强度)
        """
        conflicts = []
        conflict_idx = 0

        # 按治疗类别分组
        category_groups = self._group_by_category(strategies)

        for category, group in category_groups.items():
            if len(group) <= 1:
                continue

            # 按来源指南分组
            guideline_groups = {}
            for s in group:
                guideline_groups.setdefault(s.source_guideline, []).append(s)

            # 跨指南冲突检测
            guidelines = list(guideline_groups.keys())
            for i in range(len(guidelines)):
                for j in range(i + 1, len(guidelines)):
                    g1_strategies = guideline_groups[guidelines[i]]
                    g2_strategies = guideline_groups[guidelines[j]]

                    for s1 in g1_strategies:
                        for s2 in g2_strategies:
                            conflict = self._check_pair_conflict(s1, s2)
                            if conflict:
                                conflict_idx += 1
                                conflicts.append(ConflictRecord(
                                    conflict_id=f"conflict_{conflict_idx}",
                                    strategies=[s1, s2],
                                    conflict_type=conflict["type"],
                                    description=conflict["description"],
                                ))

        logger.info(f"检测到 {len(conflicts)} 个指南冲突")
        return conflicts

    def _group_by_category(
        self, strategies: List[TreatmentStrategy]
    ) -> Dict[str, List[TreatmentStrategy]]:
        """按治疗类别分组"""
        groups = {}
        for s in strategies:
            groups.setdefault(s.category, []).append(s)
        return groups

    def _check_pair_conflict(
        self, s1: TreatmentStrategy, s2: TreatmentStrategy
    ) -> Optional[Dict]:
        """检查两个策略是否冲突"""
        # 1. 同类药物的不同目标值
        if s1.category == "medication" and s2.category == "medication":
            # 血压目标冲突
            if "bp" in s1.strategy_id and "bp" in s2.strategy_id:
                if abs(s1.risk_reduction - s2.risk_reduction) > self.conflict_threshold:
                    return {
                        "type": "target_value_conflict",
                        "description": (
                            f"血压管理目标冲突: {s1.source_guideline} 推荐 '{s1.recommendation[:50]}...' "
                            f"vs {s2.source_guideline} 推荐 '{s2.recommendation[:50]}...'"
                        ),
                    }

            # 血脂目标冲突
            if "statin" in s1.strategy_id and "statin" in s2.strategy_id:
                ldl_target_1 = s1.conditions.get("ldl_c_threshold", 0)
                ldl_target_2 = s2.conditions.get("ldl_c_threshold", 0)
                if ldl_target_1 != ldl_target_2:
                    return {
                        "type": "target_value_conflict",
                        "description": (
                            f"LDL-C目标值冲突: {s1.source_guideline} (阈值={ldl_target_1}) "
                            f"vs {s2.source_guideline} (阈值={ldl_target_2})"
                        ),
                    }

        # 2. 治疗强度冲突
        if (s1.category == s2.category and
            abs(s1.risk_reduction - s2.risk_reduction) > 0.15):
            category_overlap = self._check_category_overlap(s1, s2)
            if category_overlap:
                return {
                    "type": "intensity_conflict",
                    "description": (
                        f"治疗强度冲突: {s1.source_guideline} (风险降低={s1.risk_reduction:.2f}) "
                        f"vs {s2.source_guideline} (风险降低={s2.risk_reduction:.2f})"
                    ),
                }

        return None

    def _check_category_overlap(self, s1: TreatmentStrategy, s2: TreatmentStrategy) -> bool:
        """检查两个策略是否针对相同的临床问题"""
        # 简单关键词匹配
        keywords_1 = set(s1.strategy_id.lower().split("_"))
        keywords_2 = set(s2.strategy_id.lower().split("_"))
        common = keywords_1 & keywords_2 - {"esc", "aha", "chs", "who"}
        return len(common) >= 1


class ConflictResolver:
    """
    冲突解决器
    通过效用函数重新评估冲突策略, 并选择最优解决方案
    """

    def __init__(self, config: AgentConfig, llm_client=None):
        self.config = config
        self.llm_client = llm_client

    def resolve_conflicts(
        self,
        conflicts: List[ConflictRecord],
        patient: PatientRecord,
        risk_profile: Dict,
    ) -> Tuple[Dict[str, float], List[ConflictRecord]]:
        """
        解决所有检测到的冲突
        返回: (conflict_penalties, resolved_conflicts)
        - conflict_penalties: 策略ID -> 额外惩罚分数
        - resolved_conflicts: 已解决的冲突记录
        """
        conflict_penalties = {}
        resolved_conflicts = []

        for conflict in conflicts:
            if self.llm_client is not None:
                result = self._llm_resolve(conflict, patient, risk_profile)
            else:
                result = self._rule_resolve(conflict, patient, risk_profile)

            conflict.resolved = True
            conflict.resolution = result["resolution"]
            conflict.winner_id = result.get("winner_id")

            # 为落败策略分配惩罚
            for strategy in conflict.strategies:
                if strategy.strategy_id != result.get("winner_id"):
                    current_penalty = conflict_penalties.get(strategy.strategy_id, 0.0)
                    conflict_penalties[strategy.strategy_id] = current_penalty + result["penalty"]

            resolved_conflicts.append(conflict)

        logger.info(
            f"解决了 {len(resolved_conflicts)}/{len(conflicts)} 个冲突, "
            f"分配了 {len(conflict_penalties)} 个惩罚"
        )
        return conflict_penalties, resolved_conflicts

    def _llm_resolve(
        self, conflict: ConflictRecord, patient: PatientRecord, risk_profile: Dict
    ) -> Dict:
        """使用LLM解决冲突"""
        strategies_desc = "\n".join([
            f"- 来源: {s.source_guideline}, 建议: {s.recommendation}, "
            f"风险降低: {s.risk_reduction:.2f}, 置信度: {s.confidence:.2f}"
            for s in conflict.strategies
        ])

        risk_desc = (
            f"风险评分: {risk_profile.get('risk_score', 0):.2f}, "
            f"风险等级: {risk_profile.get('risk_level', 'unknown')}"
        )

        prompt = f"""作为心血管疾病临床决策专家，请解决以下指南冲突:

冲突类型: {conflict.conflict_type}
冲突描述: {conflict.description}

冲突策略:
{strategies_desc}

患者风险概况: {risk_desc}

请分析并选择最优策略，以JSON格式返回:
{{"winner_strategy_index": 0, "reason": "选择原因", "penalty_for_losers": 0.1}}
"""
        try:
            response = self.llm_client.chat.completions.create(
                model=self.config.llm.model_name,
                messages=[
                    {"role": "system", "content": "你是心血管临床决策冲突解决专家。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.config.llm.temperature,
                max_tokens=512,
            )
            result = json.loads(response.choices[0].message.content)
            winner_idx = result.get("winner_strategy_index", 0)
            winner_id = conflict.strategies[winner_idx].strategy_id

            return {
                "winner_id": winner_id,
                "resolution": result.get("reason", "LLM选择"),
                "penalty": result.get("penalty_for_losers", 0.15),
            }
        except Exception as e:
            logger.warning(f"LLM冲突解决失败, 回退到规则方法: {e}")
            return self._rule_resolve(conflict, patient, risk_profile)

    def _rule_resolve(
        self, conflict: ConflictRecord, patient: PatientRecord, risk_profile: Dict
    ) -> Dict:
        """
        基于规则的冲突解决方案 (LLM回退方案)
        优先级: 指南一致性 > 风险降低 > 置信度
        """
        strategies = conflict.strategies
        risk_level = risk_profile.get("risk_level", "medium")

        # 综合评分
        scores = []
        for s in strategies:
            score = (
                s.guideline_consistency * 0.4
                + s.risk_reduction * 0.35
                + s.confidence * 0.25
            )
            # 高风险患者优先选择更激进的药物治疗
            if risk_level in ["high", "very_high"] and s.category == "medication":
                score *= 1.1
            scores.append(score)

        winner_idx = int(np.argmax(scores))
        winner = strategies[winner_idx]

        # 冲突惩罚基于分数差距
        score_diff = max(scores) - min(scores)
        penalty = 0.1 + score_diff * 0.3  # 差距越大,惩罚越大

        resolution = (
            f"基于综合评分选择 {winner.source_guideline} 的方案 "
            f"(评分: {scores[winner_idx]:.3f}), 原因: "
            f"指南一致性={winner.guideline_consistency:.2f}, "
            f"风险降低={winner.risk_reduction:.2f}, "
            f"置信度={winner.confidence:.2f}"
        )

        return {
            "winner_id": winner.strategy_id,
            "resolution": resolution,
            "penalty": penalty,
        }

    def get_conflict_resolution_rate(self, conflicts: List[ConflictRecord]) -> float:
        """计算冲突解决率"""
        if not conflicts:
            return 1.0
        resolved = sum(1 for c in conflicts if c.resolved)
        return resolved / len(conflicts)

