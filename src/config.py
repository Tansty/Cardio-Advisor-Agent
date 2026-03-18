"""
Cardio-Advisor Agent 配置文件
定义所有超参数、路径和系统配置
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LLMConfig:
    """LLM 配置"""
    model_name: str = "gpt-4"
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    temperature: float = 0.3
    max_tokens: int = 2048
    top_p: float = 0.9

    def __post_init__(self):
        if self.api_key is None:
            self.api_key = os.getenv("OPENAI_API_KEY", "")
        if self.api_base is None:
            self.api_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")


@dataclass
class DataConfig:
    """数据路径配置"""
    base_dir: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    clinical_data_path: str = ""
    framingham_data_path: str = ""
    kaggle_data_path: str = ""
    test_ratio: float = 0.2
    random_seed: int = 42

    def __post_init__(self):
        self.clinical_data_path = os.path.join(self.base_dir, "personal", "cleaned_merged.csv")
        self.framingham_data_path = os.path.join(self.base_dir, "Framingham Dataset.csv")
        self.kaggle_data_path = os.path.join(self.base_dir, "cardio_train.csv")


@dataclass
class UtilityConfig:
    """效用函数权重配置 (公式6: U(a|S̃t) = αR(a|S̃t) + βG(a) - γP(a))"""
    alpha: float = 0.4     # 风险降低收益权重
    beta: float = 0.35     # 指南一致性权重
    gamma: float = 0.25    # 冲突/违规惩罚权重


@dataclass
class AgentConfig:
    """Agent 总体配置"""
    llm: LLMConfig = field(default_factory=LLMConfig)
    data: DataConfig = field(default_factory=DataConfig)
    utility: UtilityConfig = field(default_factory=UtilityConfig)

    # 风险分层阈值
    risk_thresholds: dict = field(default_factory=lambda: {
        "low": 0.2,
        "medium": 0.4,
        "high": 0.6,
        "very_high": 1.0,
    })

    # 安全约束参数
    safety_sbp_max: float = 180.0
    safety_sbp_min: float = 90.0
    safety_dbp_max: float = 120.0
    safety_dbp_min: float = 60.0
    safety_ldl_max: float = 4.9  # mmol/L
    safety_age_min: int = 18
    safety_age_max: int = 100

    # 评估参数
    num_candidate_strategies: int = 5
    conflict_threshold: float = 0.3

