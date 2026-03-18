"""
患者状态建模模块 (Patient State Modeling)
论文 Section 3.2

核心功能:
- 将患者信息形式化为结构化状态表示: S_t = {x_1, x_2, ..., x_n}  (公式1)
- 使用LLM驱动的权重机制: W_t = LLM(S_t)                        (公式2)
- 计算加权状态表示: S̃_t = S_t ⊙ W_t                            (公式3)
"""

import json
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.config import AgentConfig
from src.data_loader import PatientRecord, get_feature_description

logger = logging.getLogger(__name__)


class PatientStateModeler:
    """患者状态建模器"""

    def __init__(self, config: AgentConfig, llm_client=None):
        self.config = config
        self.llm_client = llm_client
        self.feature_descriptions = get_feature_description()

    def build_state(self, patient: PatientRecord) -> np.ndarray:
        """
        构建患者状态向量 S_t (公式1)
        S_t = {x_1, x_2, ..., x_n}
        """
        return patient.to_state_vector()

    def compute_weights(self, patient: PatientRecord) -> np.ndarray:
        """
        使用LLM计算特征权重向量 W_t (公式2)
        W_t = LLM(S_t)

        LLM根据患者的临床上下文动态评估每个特征的重要性
        """
        if self.llm_client is not None:
            return self._llm_weight(patient)
        else:
            return self._rule_based_weight(patient)

    def compute_weighted_state(self, patient: PatientRecord) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算加权状态表示 (公式3)
        S̃_t = S_t ⊙ W_t

        返回: (weighted_state, weight_vector)
        """
        state = self.build_state(patient)
        weights = self.compute_weights(patient)
        weighted_state = state * weights  # 逐元素加权
        return weighted_state, weights

    def get_risk_profile(self, patient: PatientRecord) -> Dict:
        """
        获取患者的综合风险概况
        """
        weighted_state, weights = self.compute_weighted_state(patient)
        feature_names = patient.get_feature_names()

        # 识别主要风险因素 (权重最高的特征)
        risk_factors = []
        sorted_indices = np.argsort(-weights)
        for idx in sorted_indices[:5]:  # Top-5 风险因素
            if weights[idx] > 0.5:
                risk_factors.append({
                    "feature": feature_names[idx],
                    "value": patient.features[feature_names[idx]],
                    "weight": float(weights[idx]),
                    "description": self.feature_descriptions.get(feature_names[idx], ""),
                })

        # 计算整体风险评分
        risk_score = self._calculate_risk_score(patient, weighted_state)

        # 风险分层
        risk_level = self._stratify_risk(risk_score)

        return {
            "patient_id": patient.patient_id,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "dominant_risk_factors": risk_factors,
            "weighted_state": weighted_state,
            "weight_vector": weights,
        }

    def _llm_weight(self, patient: PatientRecord) -> np.ndarray:
        """通过 LLM 生成特征权重"""
        feature_names = patient.get_feature_names()
        features_str = "\n".join([
            f"- {name} ({self.feature_descriptions.get(name, '')}): {patient.features[name]}"
            for name in feature_names
        ])

        prompt = f"""作为心血管疾病临床决策专家，请分析以下患者的临床特征，
评估每个特征对心血管风险的重要性权重（0.0-1.0之间）。

患者临床特征:
{features_str}

请以JSON格式返回每个特征的权重，格式如下:
{{"weights": {{{", ".join([f'"{name}": 0.0' for name in feature_names])}}}}}

要求:
1. 权重范围为0.0到1.0
2. 对心血管风险贡献大的特征给予高权重
3. 考虑特征之间的相互作用
4. 特别关注异常值指标
"""
        try:
            response = self.llm_client.chat.completions.create(
                model=self.config.llm.model_name,
                messages=[
                    {"role": "system", "content": "你是一名心血管领域临床决策支持专家。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.config.llm.temperature,
                max_tokens=self.config.llm.max_tokens,
            )
            result = response.choices[0].message.content
            # 解析JSON
            weight_dict = json.loads(result)
            if "weights" in weight_dict:
                weight_dict = weight_dict["weights"]
            weights = np.array([weight_dict.get(name, 0.5) for name in feature_names])
            return np.clip(weights, 0.0, 1.0)
        except Exception as e:
            logger.warning(f"LLM权重计算失败，回退到规则方法: {e}")
            return self._rule_based_weight(patient)

    def _rule_based_weight(self, patient: PatientRecord) -> np.ndarray:
        """
        基于规则的特征权重计算（作为LLM的回退方案）
        根据心血管临床指南设定特征权重
        """
        features = patient.features
        feature_names = patient.get_feature_names()
        n = len(feature_names)
        weights = np.ones(n) * 0.5  # 默认权重

        weight_map = {}

        # === 年龄权重 ===
        age = features.get("age", 0)
        if age > 65:
            weight_map["age"] = 0.9
        elif age > 55:
            weight_map["age"] = 0.8
        elif age > 45:
            weight_map["age"] = 0.7
        else:
            weight_map["age"] = 0.4

        # === 血压权重 ===
        sbp = features.get("systolic_bp", 0)
        dbp = features.get("diastolic_bp", 0)
        if sbp >= 160 or dbp >= 100:
            weight_map["systolic_bp"] = 0.95
            weight_map["diastolic_bp"] = 0.95
        elif sbp >= 140 or dbp >= 90:
            weight_map["systolic_bp"] = 0.85
            weight_map["diastolic_bp"] = 0.85
        else:
            weight_map["systolic_bp"] = 0.6
            weight_map["diastolic_bp"] = 0.6

        # === BMI 权重 ===
        bmi = features.get("bmi", 0)
        if bmi > 30:
            weight_map["bmi"] = 0.85
        elif bmi > 25:
            weight_map["bmi"] = 0.7
        else:
            weight_map["bmi"] = 0.4

        # === 吸烟权重 ===
        if features.get("smoking", 0) == 1:
            weight_map["smoking"] = 0.9
        else:
            weight_map["smoking"] = 0.2

        # === 糖尿病权重 ===
        if features.get("diabetes", 0) == 1:
            weight_map["diabetes"] = 0.9
        else:
            weight_map["diabetes"] = 0.3

        # === 高血压权重 ===
        if features.get("hypertension", 0) == 1:
            weight_map["hypertension"] = 0.9
        else:
            weight_map["hypertension"] = 0.3

        # === 家族史权重 ===
        if features.get("family_history", 0) == 1:
            weight_map["family_history"] = 0.8
        else:
            weight_map["family_history"] = 0.2

        # === 胆固醇和血脂 ===
        cholesterol = features.get("cholesterol", 0)
        if cholesterol >= 3:
            weight_map["cholesterol"] = 0.9
        elif cholesterol >= 2:
            weight_map["cholesterol"] = 0.7
        else:
            weight_map["cholesterol"] = 0.4

        ldl = features.get("ldl_c", 0)
        if ldl > 4.1:
            weight_map["ldl_c"] = 0.95
        elif ldl > 3.4:
            weight_map["ldl_c"] = 0.85
        elif ldl > 2.6:
            weight_map["ldl_c"] = 0.7
        else:
            weight_map["ldl_c"] = 0.4

        hdl = features.get("hdl_c", 0)
        if hdl < 1.0:
            weight_map["hdl_c"] = 0.85
        elif hdl < 1.3:
            weight_map["hdl_c"] = 0.6
        else:
            weight_map["hdl_c"] = 0.3

        # === 心率权重 ===
        hr = features.get("heart_rate", 0)
        if hr > 100 or (0 < hr < 60):
            weight_map["heart_rate"] = 0.8
        else:
            weight_map["heart_rate"] = 0.4

        # === 血糖权重 ===
        glucose = features.get("glucose", 0)
        if glucose >= 3:
            weight_map["glucose"] = 0.85
        elif glucose >= 2:
            weight_map["glucose"] = 0.65
        else:
            weight_map["glucose"] = 0.3

        # 默认其他特征
        weight_map.setdefault("gender", 0.5)
        weight_map.setdefault("height", 0.2)
        weight_map.setdefault("weight", 0.4)
        weight_map.setdefault("alcohol", 0.5 if features.get("alcohol", 0) == 1 else 0.2)
        weight_map.setdefault("physical_activity", 0.4)
        weight_map.setdefault("total_cholesterol", weight_map.get("cholesterol", 0.4))
        weight_map.setdefault("triglycerides", 0.5)

        for i, name in enumerate(feature_names):
            if name in weight_map:
                weights[i] = weight_map[name]

        return weights

    def _calculate_risk_score(self, patient: PatientRecord, weighted_state: np.ndarray) -> float:
        """
        计算综合心血管风险评分 (0-1)
        基于加权状态向量的归一化评分
        """
        features = patient.features
        risk = 0.0
        total_weight = 0.0

        # 年龄风险
        age = features.get("age", 0)
        if age > 0:
            age_risk = min(age / 100.0, 1.0)
            risk += age_risk * 0.15
            total_weight += 0.15

        # 血压风险
        sbp = features.get("systolic_bp", 0)
        if sbp > 0:
            sbp_risk = min(max((sbp - 90) / 100.0, 0), 1.0)
            risk += sbp_risk * 0.2
            total_weight += 0.2

        # BMI风险
        bmi = features.get("bmi", 0)
        if bmi > 0:
            bmi_risk = min(max((bmi - 18.5) / 20.0, 0), 1.0)
            risk += bmi_risk * 0.1
            total_weight += 0.1

        # 二元风险因素
        binary_factors = {
            "smoking": 0.12,
            "diabetes": 0.15,
            "hypertension": 0.15,
            "family_history": 0.08,
        }
        for factor, weight in binary_factors.items():
            val = features.get(factor, 0)
            risk += float(val) * weight
            total_weight += weight

        # 胆固醇风险
        cholesterol = features.get("cholesterol", 0)
        if cholesterol > 0:
            chol_risk = min((cholesterol - 1) / 2.0, 1.0)
            risk += chol_risk * 0.1
            total_weight += 0.1

        # LDL-C 风险
        ldl = features.get("ldl_c", 0)
        if ldl > 0:
            ldl_risk = min(ldl / 5.0, 1.0)
            risk += ldl_risk * 0.1
            total_weight += 0.1

        if total_weight > 0:
            risk = risk / total_weight
        return min(max(risk, 0.0), 1.0)

    def _stratify_risk(self, risk_score: float) -> str:
        """风险分层"""
        thresholds = self.config.risk_thresholds
        if risk_score < thresholds["low"]:
            return "low"
        elif risk_score < thresholds["medium"]:
            return "medium"
        elif risk_score < thresholds["high"]:
            return "high"
        else:
            return "very_high"

    def format_state_for_llm(self, patient: PatientRecord) -> str:
        """将患者状态格式化为LLM输入文本"""
        lines = ["## 患者临床状态"]
        for name, value in patient.features.items():
            desc = self.feature_descriptions.get(name, name)
            lines.append(f"- {desc}: {value}")
        return "\n".join(lines)

