"""
数据加载模块
支持三个异构数据集:
1. 真实临床数据集 (2,260例患者, 27个特征)
2. Framingham心脏研究数据集 (~4,200名受试者)
3. Kaggle心血管疾病数据集 (~70,000个样本)
"""

import os
import logging
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from src.config import DataConfig

logger = logging.getLogger(__name__)


class PatientRecord:
    """统一的患者记录结构"""

    def __init__(self, patient_id: str, features: Dict, label: int, source: str):
        self.patient_id = patient_id
        self.features = features          # 标准化特征字典
        self.label = label                # 0: 无心血管疾病, 1: 有心血管疾病
        self.source = source              # 数据来源标识

    def to_state_vector(self) -> np.ndarray:
        """转换为状态向量 S_t = {x_1, x_2, ..., x_n} (论文公式1)"""
        return np.array(list(self.features.values()), dtype=np.float64)

    def get_feature_names(self) -> List[str]:
        return list(self.features.keys())

    def __repr__(self):
        return f"PatientRecord(id={self.patient_id}, label={self.label}, source={self.source})"


class DataLoader:
    """统一数据加载器"""

    # 标准化特征名 -> 各数据集中的对应列名映射
    UNIFIED_FEATURES = [
        "age", "gender", "height", "weight", "bmi",
        "systolic_bp", "diastolic_bp",
        "cholesterol", "glucose",
        "smoking", "alcohol", "physical_activity",
        "diabetes", "hypertension", "family_history",
        "heart_rate",
        "ldl_c", "hdl_c", "total_cholesterol", "triglycerides",
    ]

    def __init__(self, config: DataConfig):
        self.config = config
        self.scaler = StandardScaler()

    def load_all_datasets(self) -> Dict[str, Tuple[List[PatientRecord], List[PatientRecord]]]:
        """加载所有数据集并返回训练/测试划分"""
        datasets = {}

        # 1. Kaggle 心血管疾病数据集
        if os.path.exists(self.config.kaggle_data_path):
            logger.info("正在加载 Kaggle 心血管疾病数据集...")
            records = self._load_kaggle_data()
            train, test = self._split_data(records)
            datasets["kaggle"] = (train, test)
            logger.info(f"Kaggle 数据集: 训练集 {len(train)}, 测试集 {len(test)}")

        # 2. Framingham 心脏研究数据集
        if os.path.exists(self.config.framingham_data_path):
            logger.info("正在加载 Framingham 心脏研究数据集...")
            records = self._load_framingham_data()
            train, test = self._split_data(records)
            datasets["framingham"] = (train, test)
            logger.info(f"Framingham 数据集: 训练集 {len(train)}, 测试集 {len(test)}")

        # 3. 真实临床数据集
        if os.path.exists(self.config.clinical_data_path):
            logger.info("正在加载真实临床数据集...")
            records = self._load_clinical_data()
            train, test = self._split_data(records)
            datasets["clinical"] = (train, test)
            logger.info(f"临床数据集: 训练集 {len(train)}, 测试集 {len(test)}")

        return datasets

    def _load_kaggle_data(self) -> List[PatientRecord]:
        """加载 Kaggle 心血管疾病数据集 (~70,000 样本, 12个特征)"""
        df = pd.read_csv(self.config.kaggle_data_path, sep=";")
        records = []

        for _, row in df.iterrows():
            age_years = row["age"] / 365.25
            height_m = row["height"] / 100.0
            weight_kg = row["weight"]
            bmi = weight_kg / (height_m ** 2) if height_m > 0 else 0

            features = {
                "age": age_years,
                "gender": float(row["gender"]),
                "height": row["height"],
                "weight": weight_kg,
                "bmi": bmi,
                "systolic_bp": float(row["ap_hi"]),
                "diastolic_bp": float(row["ap_lo"]),
                "cholesterol": float(row["cholesterol"]),
                "glucose": float(row["gluc"]),
                "smoking": float(row["smoke"]),
                "alcohol": float(row["alco"]),
                "physical_activity": float(row["active"]),
                "diabetes": 0.0,
                "hypertension": 1.0 if row["ap_hi"] >= 140 or row["ap_lo"] >= 90 else 0.0,
                "family_history": 0.0,
                "heart_rate": 0.0,
                "ldl_c": 0.0,
                "hdl_c": 0.0,
                "total_cholesterol": float(row["cholesterol"]),
                "triglycerides": 0.0,
            }

            records.append(PatientRecord(
                patient_id=f"kaggle_{row['id']}",
                features=features,
                label=int(row["cardio"]),
                source="kaggle"
            ))

        return records

    def _load_framingham_data(self) -> List[PatientRecord]:
        """加载 Framingham 心脏研究数据集 (~4,200 受试者, 24个特征)"""
        df = pd.read_csv(self.config.framingham_data_path)
        # 去重: 同一 RANDID 取第一次记录
        df = df.drop_duplicates(subset=["RANDID"], keep="first")
        df = df.dropna(subset=["CVD"])
        records = []

        for _, row in df.iterrows():
            bmi = row.get("BMI", 0.0)
            if pd.isna(bmi):
                bmi = 0.0

            features = {
                "age": float(row["AGE"]) if not pd.isna(row["AGE"]) else 0.0,
                "gender": float(row["SEX"]),
                "height": 0.0,
                "weight": 0.0,
                "bmi": float(bmi),
                "systolic_bp": float(row["SYSBP"]) if not pd.isna(row["SYSBP"]) else 0.0,
                "diastolic_bp": float(row["DIABP"]) if not pd.isna(row["DIABP"]) else 0.0,
                "cholesterol": float(row["TOTCHOL"]) if not pd.isna(row["TOTCHOL"]) else 0.0,
                "glucose": float(row["GLUCOSE"]) if not pd.isna(row["GLUCOSE"]) else 0.0,
                "smoking": float(row["CURSMOKE"]) if not pd.isna(row["CURSMOKE"]) else 0.0,
                "alcohol": 0.0,
                "physical_activity": 0.0,
                "diabetes": float(row["DIABETES"]) if not pd.isna(row["DIABETES"]) else 0.0,
                "hypertension": float(row["PREVHYP"]) if not pd.isna(row["PREVHYP"]) else 0.0,
                "family_history": 0.0,
                "heart_rate": float(row["HEARTRTE"]) if not pd.isna(row["HEARTRTE"]) else 0.0,
                "ldl_c": float(row["LDLC"]) if not pd.isna(row.get("LDLC", np.nan)) else 0.0,
                "hdl_c": float(row["HDLC"]) if not pd.isna(row.get("HDLC", np.nan)) else 0.0,
                "total_cholesterol": float(row["TOTCHOL"]) if not pd.isna(row["TOTCHOL"]) else 0.0,
                "triglycerides": 0.0,
            }

            records.append(PatientRecord(
                patient_id=f"framingham_{int(row['RANDID'])}",
                features=features,
                label=int(row["CVD"]),
                source="framingham"
            ))

        return records

    def _load_clinical_data(self) -> List[PatientRecord]:
        """加载真实临床数据集 (2,260例患者, 27个特征)"""
        df = pd.read_csv(self.config.clinical_data_path, sep=",")
        records = []

        for _, row in df.iterrows():
            # 解析标签: is_coronary
            label_raw = str(row.get("is_coronary", "")).strip()
            if label_raw in ["是", "1", "True", "true"]:
                label = 1
            elif label_raw in ["否", "0", "False", "false"]:
                label = 0
            else:
                continue  # 跳过标签不清的样本

            gender = 1.0 if "男" in str(row.get("gender", "")) else 2.0

            def safe_float(val, default=0.0):
                try:
                    v = str(val).strip()
                    if v in ["", "nan", "None", "卧床"]:
                        return default
                    return float(v)
                except (ValueError, TypeError):
                    return default

            features = {
                "age": safe_float(row.get("age", 0)),
                "gender": gender,
                "height": safe_float(row.get("height", 0)),
                "weight": safe_float(row.get("weight", 0)),
                "bmi": safe_float(row.get("body_mass_index", 0)),
                "systolic_bp": safe_float(row.get("systolic_pressure", 0)),
                "diastolic_bp": safe_float(row.get("diastolic_pressure", 0)),
                "cholesterol": 0.0,
                "glucose": 0.0,
                "smoking": 1.0 if str(row.get("is_smoking", "")).strip() == "是" else 0.0,
                "alcohol": 1.0 if str(row.get("is_drinking", "")).strip() == "是" else 0.0,
                "physical_activity": 0.0,
                "diabetes": 1.0 if str(row.get("is_diabetes", "")).strip() == "是" else 0.0,
                "hypertension": 1.0 if str(row.get("is_hypertension", "")).strip() == "是" else 0.0,
                "family_history": 1.0 if str(row.get("is_family_history", "")).strip() == "是" else 0.0,
                "heart_rate": safe_float(row.get("heart_rate", 0)),
                "ldl_c": safe_float(row.get("lab_biochemical_tests_LDL-C_test_result", 0)),
                "hdl_c": safe_float(row.get("lab_biochemical_tests_HDL-C_test_result", 0)),
                "total_cholesterol": safe_float(row.get("lab_biochemical_tests_TC_test_result", 0)),
                "triglycerides": safe_float(row.get("lab_biochemical_tests_TG_test_result", 0)),
            }

            records.append(PatientRecord(
                patient_id=f"clinical_{row.get('patient_sn', '')}".strip(),
                features=features,
                label=label,
                source="clinical"
            ))

        return records

    def _split_data(
        self, records: List[PatientRecord]
    ) -> Tuple[List[PatientRecord], List[PatientRecord]]:
        """按配置比例划分训练集和测试集"""
        labels = [r.label for r in records]
        train_records, test_records = train_test_split(
            records,
            test_size=self.config.test_ratio,
            random_state=self.config.random_seed,
            stratify=labels if len(set(labels)) > 1 else None,
        )
        return train_records, test_records


def get_feature_description() -> Dict[str, str]:
    """获取特征的中文描述"""
    return {
        "age": "年龄(岁)",
        "gender": "性别(1=男,2=女)",
        "height": "身高(cm)",
        "weight": "体重(kg)",
        "bmi": "体质指数(BMI)",
        "systolic_bp": "收缩压(mmHg)",
        "diastolic_bp": "舒张压(mmHg)",
        "cholesterol": "胆固醇水平(1:正常,2:偏高,3:远高于正常)",
        "glucose": "血糖水平(1:正常,2:偏高,3:远高于正常)",
        "smoking": "吸烟状态(0/1)",
        "alcohol": "饮酒状态(0/1)",
        "physical_activity": "体力活动(0/1)",
        "diabetes": "糖尿病(0/1)",
        "hypertension": "高血压(0/1)",
        "family_history": "家族史(0/1)",
        "heart_rate": "心率(次/分)",
        "ldl_c": "低密度脂蛋白胆固醇LDL-C(mmol/L)",
        "hdl_c": "高密度脂蛋白胆固醇HDL-C(mmol/L)",
        "total_cholesterol": "总胆固醇TC(mmol/L)",
        "triglycerides": "甘油三酯TG(mmol/L)",
    }

