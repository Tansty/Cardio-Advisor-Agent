"""
多源指南知识工具系统 (Multi-Source Guideline Knowledge Tool System)
论文 Section 3.4

核心功能:
- 每个临床指南封装为独立的知识工具 T_k
- 基于患者状态 S̃_t 生成候选策略集: A_k = T_k(S̃_t)     (公式8)
- 并行调用多个指南工具, 合并输出: A = ∪ A_k             (公式9)
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from src.data_loader import PatientRecord

logger = logging.getLogger(__name__)


@dataclass
class TreatmentStrategy:
    """治疗策略"""
    strategy_id: str
    source_guideline: str           # 来源指南
    recommendation: str             # 治疗建议描述
    category: str                   # 策略类别: lifestyle, medication, intervention
    risk_reduction: float           # 预期风险降低 R(a|S̃_t) ∈ [0,1]
    guideline_consistency: float    # 指南一致性 G(a) ∈ [0,1]
    contraindications: List[str] = field(default_factory=list)   # 禁忌症
    conditions: Dict = field(default_factory=dict)               # 适用条件
    confidence: float = 0.5         # 置信度

    def __repr__(self):
        return (f"Strategy({self.strategy_id}, source={self.source_guideline}, "
                f"category={self.category}, R={self.risk_reduction:.2f})")


class BaseGuidelineTool(ABC):
    """
    指南工具基类
    每个指南工具 T_k 生成候选策略集 A_k = T_k(S̃_t)
    """

    def __init__(self, guideline_name: str):
        self.guideline_name = guideline_name

    @abstractmethod
    def generate_strategies(
        self, patient: PatientRecord, weighted_state: np.ndarray, risk_profile: Dict
    ) -> List[TreatmentStrategy]:
        """根据患者加权状态生成候选策略"""
        pass

    def check_applicability(self, patient: PatientRecord) -> bool:
        """检查指南对当前患者的适用性"""
        return True


class ESCGuidelineTool(BaseGuidelineTool):
    """
    ESC (欧洲心脏病学会) 指南工具
    主要针对: 血脂管理, 动脉粥样硬化性心血管疾病预防
    """

    def __init__(self):
        super().__init__("ESC_Dyslipidemia_2019")

    def generate_strategies(
        self, patient: PatientRecord, weighted_state: np.ndarray, risk_profile: Dict
    ) -> List[TreatmentStrategy]:
        strategies = []
        features = patient.features
        risk_level = risk_profile.get("risk_level", "medium")

        # === 血脂管理策略 ===
        ldl = features.get("ldl_c", 0)
        total_chol = features.get("total_cholesterol", 0)
        cholesterol = features.get("cholesterol", 0)

        # 高LDL-C患者: 他汀类药物治疗
        if ldl > 3.0 or cholesterol >= 2:
            intensity = "高强度" if risk_level in ["high", "very_high"] else "中等强度"
            target_ldl = 1.4 if risk_level == "very_high" else (1.8 if risk_level == "high" else 2.6)
            strategies.append(TreatmentStrategy(
                strategy_id=f"esc_statin_{risk_level}",
                source_guideline=self.guideline_name,
                recommendation=f"推荐{intensity}他汀类药物治疗，目标LDL-C < {target_ldl} mmol/L",
                category="medication",
                risk_reduction=0.35 if risk_level in ["high", "very_high"] else 0.25,
                guideline_consistency=0.92,
                contraindications=["肝功能不全", "妊娠", "哺乳期"],
                conditions={"ldl_c_threshold": 3.0, "risk_level": risk_level},
                confidence=0.88,
            ))

        # 极高危患者: 联合降脂
        if risk_level == "very_high" and ldl > 1.8:
            strategies.append(TreatmentStrategy(
                strategy_id="esc_combo_lipid",
                source_guideline=self.guideline_name,
                recommendation="他汀联合依折麦布治疗，必要时加用PCSK9抑制剂",
                category="medication",
                risk_reduction=0.45,
                guideline_consistency=0.85,
                contraindications=["严重肾功能不全"],
                conditions={"ldl_c_threshold": 1.8, "risk_level": "very_high"},
                confidence=0.82,
            ))

        # === 生活方式干预 ===
        bmi = features.get("bmi", 0)
        smoking = features.get("smoking", 0)

        if bmi > 25:
            strategies.append(TreatmentStrategy(
                strategy_id="esc_weight_mgmt",
                source_guideline=self.guideline_name,
                recommendation="控制体重，目标BMI < 25 kg/m²，推荐地中海饮食",
                category="lifestyle",
                risk_reduction=0.15,
                guideline_consistency=0.95,
                conditions={"bmi_threshold": 25},
                confidence=0.90,
            ))

        if smoking == 1:
            strategies.append(TreatmentStrategy(
                strategy_id="esc_smoking_cessation",
                source_guideline=self.guideline_name,
                recommendation="强烈建议戒烟，可考虑戒烟药物辅助",
                category="lifestyle",
                risk_reduction=0.30,
                guideline_consistency=0.98,
                confidence=0.95,
            ))

        return strategies


class AHAGuidelineTool(BaseGuidelineTool):
    """
    AHA/ACC (美国心脏协会/美国心脏病学院) 指南工具
    主要针对: 高血压管理, 心血管风险评估
    """

    def __init__(self):
        super().__init__("AHA_ACC_Hypertension_2017")

    def generate_strategies(
        self, patient: PatientRecord, weighted_state: np.ndarray, risk_profile: Dict
    ) -> List[TreatmentStrategy]:
        strategies = []
        features = patient.features
        risk_level = risk_profile.get("risk_level", "medium")

        sbp = features.get("systolic_bp", 0)
        dbp = features.get("diastolic_bp", 0)
        diabetes = features.get("diabetes", 0)
        age = features.get("age", 0)

        # === 高血压管理 ===
        if sbp >= 130 or dbp >= 80:
            # 一级高血压 (130-139/80-89)
            if sbp < 140 and dbp < 90:
                if risk_level in ["high", "very_high"]:
                    strategies.append(TreatmentStrategy(
                        strategy_id="aha_bp_med_stage1_high",
                        source_guideline=self.guideline_name,
                        recommendation="一级高血压合并高风险：启动降压药物(ACEI/ARB)，目标BP < 130/80 mmHg",
                        category="medication",
                        risk_reduction=0.25,
                        guideline_consistency=0.88,
                        contraindications=["双侧肾动脉狭窄", "妊娠(ACEI/ARB禁忌)"],
                        conditions={"sbp_range": [130, 140], "high_risk": True},
                        confidence=0.85,
                    ))
                else:
                    strategies.append(TreatmentStrategy(
                        strategy_id="aha_bp_lifestyle_stage1",
                        source_guideline=self.guideline_name,
                        recommendation="一级高血压低风险：先进行3-6个月生活方式干预",
                        category="lifestyle",
                        risk_reduction=0.12,
                        guideline_consistency=0.90,
                        conditions={"sbp_range": [130, 140], "high_risk": False},
                        confidence=0.88,
                    ))

            # 二级高血压 (≥140/90)
            if sbp >= 140 or dbp >= 90:
                med_type = "ACEI/ARB + CCB 或利尿剂联合"
                if diabetes == 1:
                    med_type = "ACEI/ARB 为首选，联合CCB"
                strategies.append(TreatmentStrategy(
                    strategy_id="aha_bp_med_stage2",
                    source_guideline=self.guideline_name,
                    recommendation=f"二级高血压：启动联合降压治疗({med_type})，目标BP < 130/80 mmHg",
                    category="medication",
                    risk_reduction=0.35,
                    guideline_consistency=0.92,
                    contraindications=["低血压", "严重肾功能不全"],
                    conditions={"sbp_threshold": 140},
                    confidence=0.90,
                ))

        # === 抗血小板治疗 ===
        if risk_level in ["high", "very_high"] and age >= 40:
            strategies.append(TreatmentStrategy(
                strategy_id="aha_antiplatelet",
                source_guideline=self.guideline_name,
                recommendation="高危患者推荐低剂量阿司匹林(75-100mg/d)二级预防",
                category="medication",
                risk_reduction=0.20,
                guideline_consistency=0.85,
                contraindications=["消化道活动性出血", "阿司匹林过敏", "年龄<40岁"],
                conditions={"age_min": 40, "risk_level": ["high", "very_high"]},
                confidence=0.82,
            ))

        # === 运动建议 ===
        physical = features.get("physical_activity", 0)
        if physical == 0:
            strategies.append(TreatmentStrategy(
                strategy_id="aha_exercise",
                source_guideline=self.guideline_name,
                recommendation="推荐每周至少150分钟中等强度有氧运动",
                category="lifestyle",
                risk_reduction=0.15,
                guideline_consistency=0.95,
                contraindications=["急性心肌梗死", "不稳定性心绞痛"],
                conditions={},
                confidence=0.92,
            ))

        return strategies


class CHSGuidelineTool(BaseGuidelineTool):
    """
    中国心血管病预防指南工具 (CHS)
    主要针对: 中国人群心血管风险综合评估和管理
    """

    def __init__(self):
        super().__init__("CHS_CVD_Prevention_2020")

    def generate_strategies(
        self, patient: PatientRecord, weighted_state: np.ndarray, risk_profile: Dict
    ) -> List[TreatmentStrategy]:
        strategies = []
        features = patient.features
        risk_level = risk_profile.get("risk_level", "medium")

        age = features.get("age", 0)
        sbp = features.get("systolic_bp", 0)
        diabetes = features.get("diabetes", 0)
        smoking = features.get("smoking", 0)
        ldl = features.get("ldl_c", 0)
        cholesterol = features.get("cholesterol", 0)

        # === 中国指南血压管理 (目标值与AHA略有不同) ===
        if sbp >= 140:
            target = "< 140/90 mmHg"
            if diabetes == 1:
                target = "< 130/80 mmHg"
            if age >= 65:
                target = "< 150/90 mmHg"

            strategies.append(TreatmentStrategy(
                strategy_id="chs_bp_management",
                source_guideline=self.guideline_name,
                recommendation=f"高血压药物治疗，降压目标 {target}。优先选择CCB或ACEI/ARB",
                category="medication",
                risk_reduction=0.30,
                guideline_consistency=0.90,
                contraindications=["低血压", "严重肝肾功能不全"],
                conditions={"sbp_threshold": 140},
                confidence=0.88,
            ))

        # === 血脂管理 (中国指南LDL-C目标与ESC有差异) ===
        if ldl > 2.6 or cholesterol >= 2:
            if risk_level in ["high", "very_high"]:
                target_ldl = 1.8
                strategies.append(TreatmentStrategy(
                    strategy_id="chs_statin_high_risk",
                    source_guideline=self.guideline_name,
                    recommendation=f"高危患者降脂治疗：他汀类药物，目标LDL-C < {target_ldl} mmol/L",
                    category="medication",
                    risk_reduction=0.30,
                    guideline_consistency=0.88,
                    contraindications=["活动性肝病", "不明原因转氨酶持续升高"],
                    conditions={"ldl_c_threshold": 2.6, "risk_level": risk_level},
                    confidence=0.85,
                ))
            else:
                strategies.append(TreatmentStrategy(
                    strategy_id="chs_statin_moderate",
                    source_guideline=self.guideline_name,
                    recommendation="中低危患者：生活方式干预3-6个月，必要时中等强度他汀",
                    category="medication",
                    risk_reduction=0.20,
                    guideline_consistency=0.85,
                    conditions={"ldl_c_threshold": 2.6},
                    confidence=0.82,
                ))

        # === 糖尿病合并心血管风险 ===
        if diabetes == 1:
            strategies.append(TreatmentStrategy(
                strategy_id="chs_diabetes_cv",
                source_guideline=self.guideline_name,
                recommendation="糖尿病患者心血管保护：严格血糖控制(HbA1c<7%)，联合他汀和降压治疗",
                category="medication",
                risk_reduction=0.25,
                guideline_consistency=0.90,
                contraindications=["严重低血糖风险"],
                conditions={"diabetes": True},
                confidence=0.87,
            ))

        # === 中医结合 ===
        if risk_level in ["medium", "high"] and age >= 50:
            strategies.append(TreatmentStrategy(
                strategy_id="chs_tcm_support",
                source_guideline=self.guideline_name,
                recommendation="可辅助中医药调理(如血脂康、丹参等)，改善血脂和血管功能",
                category="lifestyle",
                risk_reduction=0.08,
                guideline_consistency=0.60,
                conditions={"age_min": 50},
                confidence=0.65,
            ))

        # === 综合生活方式 ===
        strategies.append(TreatmentStrategy(
            strategy_id="chs_lifestyle_comprehensive",
            source_guideline=self.guideline_name,
            recommendation="综合生活方式管理：限盐(<6g/d)、戒烟限酒、规律运动、控制体重",
            category="lifestyle",
            risk_reduction=0.18,
            guideline_consistency=0.95,
            confidence=0.92,
        ))

        return strategies


class WHOGuidelineTool(BaseGuidelineTool):
    """
    WHO (世界卫生组织) HEARTS 技术包指南工具
    主要针对: 初级卫生保健环境中的心血管风险管理
    """

    def __init__(self):
        super().__init__("WHO_HEARTS_2020")

    def generate_strategies(
        self, patient: PatientRecord, weighted_state: np.ndarray, risk_profile: Dict
    ) -> List[TreatmentStrategy]:
        strategies = []
        features = patient.features
        risk_level = risk_profile.get("risk_level", "medium")
        risk_score = risk_profile.get("risk_score", 0.5)

        # === WHO HEARTS 简化风险评估和管理协议 ===
        if risk_score >= 0.3:  # 中高危
            strategies.append(TreatmentStrategy(
                strategy_id="who_hearts_protocol",
                source_guideline=self.guideline_name,
                recommendation="按WHO HEARTS方案：健康咨询 + 循证药物治疗 + 团队随访管理",
                category="medication",
                risk_reduction=0.22,
                guideline_consistency=0.80,
                conditions={"risk_score_min": 0.3},
                confidence=0.78,
            ))

        # === 基本药物方案 ===
        sbp = features.get("systolic_bp", 0)
        if sbp >= 140:
            strategies.append(TreatmentStrategy(
                strategy_id="who_basic_bp",
                source_guideline=self.guideline_name,
                recommendation="WHO基本降压方案：单药(氨氯地平/赖诺普利)起始，必要时联合",
                category="medication",
                risk_reduction=0.28,
                guideline_consistency=0.82,
                contraindications=["严重低血压"],
                conditions={"sbp_threshold": 140},
                confidence=0.80,
            ))

        # === 全民健康素养 ===
        strategies.append(TreatmentStrategy(
            strategy_id="who_health_literacy",
            source_guideline=self.guideline_name,
            recommendation="加强患者健康教育：风险认知、服药依从性、定期随访",
            category="lifestyle",
            risk_reduction=0.10,
            guideline_consistency=0.88,
            confidence=0.85,
        ))

        return strategies


class GuidelineToolSystem:
    """
    多源指南工具系统
    并行调用多个指南工具, 合并输出候选策略集
    A = ∪(k=1..K) A_k  (公式9)
    """

    def __init__(self, llm_client=None):
        self.tools: List[BaseGuidelineTool] = [
            ESCGuidelineTool(),
            AHAGuidelineTool(),
            CHSGuidelineTool(),
            WHOGuidelineTool(),
        ]
        self.llm_client = llm_client

    def generate_all_strategies(
        self, patient: PatientRecord, weighted_state: np.ndarray, risk_profile: Dict
    ) -> List[TreatmentStrategy]:
        """
        并行调用多个指南工具,生成统一候选策略集
        A = ∪(k=1..K) T_k(S̃_t)  (公式9)
        """
        all_strategies = []

        for tool in self.tools:
            if tool.check_applicability(patient):
                try:
                    strategies = tool.generate_strategies(patient, weighted_state, risk_profile)
                    all_strategies.extend(strategies)
                    logger.debug(
                        f"指南 {tool.guideline_name} 生成了 {len(strategies)} 个候选策略"
                    )
                except Exception as e:
                    logger.error(f"指南 {tool.guideline_name} 策略生成失败: {e}")

        logger.info(f"多源指南工具系统共生成 {len(all_strategies)} 个候选策略")
        return all_strategies

    def get_guideline_names(self) -> List[str]:
        return [tool.guideline_name for tool in self.tools]

