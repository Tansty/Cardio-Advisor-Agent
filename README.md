# Cardio-Advisor Agent

> A Structured Decision Optimization Framework for Personalized Cardiovascular Support via Multi-Guideline LLM Integration

基于多指南LLM集成的个性化心血管决策支持结构化优化框架。

## 项目简介

Cardio-Advisor Agent 是一个面向心血管疾病管理的临床决策支持系统，通过整合多源临床指南知识、个性化患者状态建模和显式安全约束，为患者提供可解释、可控的个性化治疗建议。

### 核心模块

| 模块 | 说明 |
|------|------|
| **Patient State Modeling** | 将患者信息形式化为结构化状态，并通过LLM动态计算特征权重 |
| **Multi-Source Guideline Tools** | 封装ESC、AHA/ACC、CHS、WHO四大临床指南为独立知识工具 |
| **Objective & Constraint Evaluation** | 基于效用函数 U(a) = αR + βG - γP 评估候选策略，并施加安全约束 |
| **Conflict Resolution** | 检测跨指南冲突，通过评分机制或LLM进行解决 |
| **Multi-Step Planning** | 五阶段推理：风险识别 → 策略生成 → 冲突检测 → 效用排序 → 最优决策输出 |

## 项目结构

```
├── src/
│   ├── config.py               # 全局配置
│   ├── data_loader.py          # 三个数据集的统一加载接口
│   ├── patient_state.py        # 患者状态建模
│   ├── guideline_tools.py      # 多源指南知识工具系统
│   ├── evaluator.py            # 效用函数与安全约束评估
│   ├── conflict_resolver.py    # 冲突检测与解决
│   ├── planner.py              # 多步规划推理
│   ├── agent.py                # Agent主体及消融实验变体
│   ├── baselines.py            # 基线方法 (Single Rule, RAG-only)
│   └── metrics.py              # 评估指标 (Accuracy, GC, PI, SVR)
├── data/
│   ├── cardio_train.csv        # Kaggle心血管疾病数据集 (~70,000样本)
│   ├── Framingham Dataset.csv  # Framingham心脏研究数据集 (~4,200受试者)
│   └── personal/               # 真实临床数据集 (2,260例患者)
├── run_experiment.py           # 主实验脚本
└── requirements.txt
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行实验

```bash
python run_experiment.py
```

运行后将在 `figures/` 目录生成可视化图表，在 `results/` 目录保存实验结果JSON。

### 3. (可选) 启用LLM

设置环境变量以启用LLM驱动的特征权重计算和冲突解决：

```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_API_BASE="https://api.openai.com/v1"  # 或其他兼容接口
python run_experiment.py
```

未配置LLM时，系统自动回退到基于规则的方法。

## 数据集

| 数据集 | 样本数 | 特征数 | 来源 |
|--------|--------|--------|------|
| 真实临床数据 | 2,260 | 27 | 三级医院 |
| Framingham Heart Study | ~4,200 | 24 | [Kaggle](https://www.kaggle.com/datasets/shreyjain601/framingham-heart-study) |
| Cardiovascular Disease | ~70,000 | 12 | [Kaggle](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset) |

## 评估指标

- **Accuracy**：预测准确率
- **GC (Guideline Consistency)**：决策与临床指南的一致性
- **PI (Personalization Index)**：个性化决策的差异度
- **SVR (Safety Violation Rate)**：安全违规率（越低越好）


