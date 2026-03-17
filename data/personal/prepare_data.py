import pandas as pd


def merge():
    # 读取三个CSV文件
    data1 = pd.read_csv('./first.csv')
    data2 = pd.read_csv('./second.csv', skiprows=2)
    # 以patient_sn为键进行合并
    merged_data = pd.merge(data1, data2, on='patient_sn', how='outer')
    # 筛选出需要保留的字段
    columns_to_keep = [
        'patient_sn', 'gender', 'birth_date', 'age', 'is_coronary', 'is_diabetes',
        'is_hypertension', 'is_smoking', 'years_of_smoking', 'is_drinking',
        'is_family_history', 'heart_rate', 'respiratory_rate', 'diastolic_pressure',
        'systolic_pressure', 'height', 'weight', 'body_mass_index', 'exam_conclusion',
        'exam_conclusion.1', 'exam_conclusion.2',
        'lab_biochemical_tests_TC_qualitative_result',
        'lab_biochemical_tests_TC_test_result',
        'lab_biochemical_tests_TG_qualitative_result',
        'lab_biochemical_tests_TG_test_result',
        'lab_biochemical_tests_HDL-C_qualitative_result',
        'lab_biochemical_tests_HDL-C_test_result',
        'lab_biochemical_tests_LDL-C_qualitative_result',
        'lab_biochemical_tests_LDL-C_test_result',
        'lab_biochemical_tests_LPa_qualitative_result',
        'lab_biochemical_tests_LPa_test_result',
        'lab_biochemical_tests_apoAI_qualitative_result',
        'lab_biochemical_tests_apoAI_test_result',
        'lab_biochemical_tests_apoB_qualitative_result',
        'lab_biochemical_tests_apoB_test_result'
    ]
    merged_data = merged_data[columns_to_keep]

    # 保存合并后的数据到新的CSV文件
    merged_data.to_csv('merged.csv', index=False)



def clend_data():
    # 读取merged.csv文件
    data = pd.read_csv('merged.csv')

    # 剔除is_coronary字段值不符合要求的行
    data = data[data['is_coronary'].apply(clean_coronary)]
    # 保存处理后的数据
    data.to_csv('cleaned_merged.csv', index=False)
 # 定义一个清理函数
def clean_coronary(value):
    value = str(value).strip()
    return value not in ['', '\t'] and value is not None

def have_blood():
    columns_to_keep = ["lab_biochemical_tests_TC_qualitative_result","lab_biochemical_tests_TG_qualitative_result",
                       "lab_biochemical_tests_HDL-C_qualitative_result","lab_biochemical_tests_LDL-C_qualitative_result",
                       "lab_biochemical_tests_LPa_qualitative_result","lab_biochemical_tests_apoAI_qualitative_result",
                       "lab_biochemical_tests_apoB_qualitative_result"]
    data = pd.read_csv('cleaned_merged.csv')
    data = data.drop(columns_to_keep,axis=1)

    # 保存合并后的数据到新的CSV文件
    data.to_csv('cleaned_merged.csv', index=False)

data = pd.read_csv('cleaned_merged.csv')
# 剔除is_coronary字段值不符合要求的行
data = data[data['lab_biochemical_tests_TC_test_result'].apply(clean_coronary)]
# 保存处理后的数据
data.to_csv('cleaned_merged_without_blood.csv', index=False)