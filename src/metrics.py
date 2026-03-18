"""
评估指标模块
论文 Section 4.2

四个评估指标:
1. Accuracy (准确率):    s(d_i, y_i)           (公式11)
2. GC (指南一致性):      g(d_i, G_i)           (公式12)
3. PI (个性化指数):      p(d_i, d_j)           (公式13)
4. SVR (安全违规率):     c(d_i)                (公式14)
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.data_loader import PatientRecord
from src.planner import DecisionResult

logger = logging.getLogger(__name__)


def compute_accuracy(predictions: List[int], labels: List[int]) -> float:
    """
    准确率 (公式11)
    Accuracy = (1/N) * Σ s(d_i, y_i)
    s(d_i, y_i) = 1 if d_i == y_i, else 0
    """
    if not predictions or not labels:
        return 0.0
    correct = sum(1 for p, l in zip(predictions, labels) if p == l)
    return correct / len(predictions)


def compute_guideline_consistency(
    results: List[DecisionResult], patients: List[PatientRecord]
) -> float:
    """
    指南一致性 GC (公式12)
    GC = (1/N) * Σ g(d_i, G_i)
    g(d_i, G_i) = 1 if decision is consistent with at least one guideline

    评估决策是否与至少一个指南推荐一致
    """
    if not results:
        return 0.0

    consistent_count = 0
    for result in results:
        if result.optimal_strategy is not None:
            # 如果最优策略的指南一致性 > 0.7, 认为与指南一致
            if result.optimal_strategy.guideline_consistency >= 0.7:
                consistent_count += 1
        elif result.predicted_label == 0:
            # 低风险预测: 如果无需药物干预, 也视为与指南一致
            consistent_count += 1

    return consistent_count / len(results)


def compute_personalization_index(
    results: List[DecisionResult], patients: List[PatientRecord]
) -> float:
    """
    个性化指数 PI (公式13)
    PI = (1/|P|) * Σ p(d_i, d_j)
    P = 相似临床结果但不同风险特征的患者对

    评估模型是否为不同患者生成不同的个性化决策
    """
    if len(results) < 2:
        return 0.0

    # 找出相似标签但不同风险特征的患者对
    pairs = []
    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            if patients[i].label == patients[j].label:
                # 检查风险特征差异
                features_i = patients[i].features
                features_j = patients[j].features
                feature_diff = _compute_feature_difference(features_i, features_j)
                if feature_diff > 0.3:  # 有显著差异的患者对
                    pairs.append((i, j))

            if len(pairs) >= 500:  # 限制对数避免计算量过大
                break
        if len(pairs) >= 500:
            break

    if not pairs:
        return 0.0

    # 计算个性化得分: 不同患者是否获得了不同的策略
    personalized_count = 0
    for i, j in pairs:
        decision_diff = _compare_decisions(results[i], results[j])
        if decision_diff:
            personalized_count += 1

    return personalized_count / len(pairs)


def compute_safety_violation_rate(results: List[DecisionResult]) -> float:
    """
    安全违规率 SVR (公式14)
    SVR = (1/N) * Σ c(d_i)
    c(d_i) = 1 when decision violates clinical constraints

    衡量不安全决策的频率
    """
    if not results:
        return 0.0

    violation_count = 0
    for result in results:
        if _check_decision_safety_violation(result):
            violation_count += 1

    return violation_count / len(results)


def compute_conflict_resolution_rate(results: List[DecisionResult]) -> float:
    """计算冲突解决率"""
    total_conflicts = 0
    resolved_conflicts = 0

    for result in results:
        for conflict in result.conflicts:
            total_conflicts += 1
            if conflict.resolved:
                resolved_conflicts += 1

    if total_conflicts == 0:
        return 1.0
    return resolved_conflicts / total_conflicts


def compute_all_metrics(
    results: List[DecisionResult],
    patients: List[PatientRecord],
    predictions: Optional[List[int]] = None,
) -> Dict[str, float]:
    """计算所有评估指标"""
    labels = [p.label for p in patients]

    if predictions is None:
        predictions = [r.predicted_label for r in results]

    metrics = {
        "Accuracy": compute_accuracy(predictions, labels),
        "GC": compute_guideline_consistency(results, patients),
        "PI": compute_personalization_index(results, patients),
        "SVR": compute_safety_violation_rate(results),
    }

    # 额外指标
    metrics["Conflict_Resolution_Rate"] = compute_conflict_resolution_rate(results)
    metrics["Avg_Confidence"] = (
        np.mean([r.decision_confidence for r in results]) if results else 0.0
    )

    return metrics


# ============ 辅助函数 ============

def _compute_feature_difference(features_i: Dict, features_j: Dict) -> float:
    """计算两个患者之间的特征差异度"""
    diff_count = 0
    total = 0

    for key in features_i:
        if key in features_j:
            v1 = features_i[key]
            v2 = features_j[key]
            total += 1
            if abs(v1 - v2) > 0.5 * max(abs(v1), abs(v2), 1e-6):
                diff_count += 1

    return diff_count / max(total, 1)


def _compare_decisions(result_i: DecisionResult, result_j: DecisionResult) -> bool:
    """比较两个决策是否不同 (体现个性化)"""
    # 策略不同
    if (result_i.optimal_strategy is not None and result_j.optimal_strategy is not None):
        if result_i.optimal_strategy.strategy_id != result_j.optimal_strategy.strategy_id:
            return True
        if abs(result_i.decision_confidence - result_j.decision_confidence) > 0.1:
            return True

    # 风险等级不同
    if result_i.risk_profile.get("risk_level") != result_j.risk_profile.get("risk_level"):
        return True

    return False


def _check_decision_safety_violation(result: DecisionResult) -> bool:
    """检查决策是否存在安全违规"""
    # 检查是否所有策略都被安全过滤了
    for evaluation in result.all_evaluations:
        if not evaluation.get("is_safe", True):
            return True

    # 检查高风险患者是否被错误预测为低风险
    risk_level = result.risk_profile.get("risk_level", "medium")
    if risk_level == "very_high" and result.predicted_label == 0:
        return True

    return False


def format_metrics_table(metrics_dict: Dict[str, Dict[str, float]]) -> str:
    """格式化指标为表格"""
    header = f"{'Method':<30} {'Accuracy':>10} {'GC':>10} {'PI':>10} {'SVR':>10}"
    lines = [header, "-" * len(header)]

    for method_name, metrics in metrics_dict.items():
        line = (
            f"{method_name:<30} "
            f"{metrics.get('Accuracy', 0):.4f}     "
            f"{metrics.get('GC', 0):.4f}     "
            f"{metrics.get('PI', 0):.4f}     "
            f"{metrics.get('SVR', 0):.4f}"
        )
        lines.append(line)

    return "\n".join(lines)

