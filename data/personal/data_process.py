import re
import pandas as pd

data1 = pd.read_csv('./first.csv')
data2 = pd.read_csv('./second.csv', skiprows=2)
data3 = pd.read_csv('./third.csv', skiprows=2)

def process_data1():
    dict_1_csv = []
    for one in data1.columns:
        if "lab_blood_routine_examination" not in one and "lab_urine_routine_test" not in one:
            dict_1_csv.append(one)
            continue
        if "lab_urine_routine_test" in one:
            if "BIL" in one:
                continue
            prefix2 = "lab_urine_routine_test_"
            pattern2 = re.compile(rf"{prefix2}(.*?)(?=_)")
            match2 = pattern2.search(one)
            if match2:
                if len(match2.group(1)) > 1:
                    dict_1_csv.append(prefix2 + match2.group(1))
        if "lab_blood_routine_examination" in one:
            if "原值" in one or "归一" in one:
                continue
            prefix1 = "lab_blood_routine_examination_"
            pattern1 = re.compile(rf"{prefix1}(.*?)(?=_)")
            match1 = pattern1.search(one)
            if match1:
                if len(match1.group(1))>1:
                    dict_1_csv.append(prefix1 +match1.group(1))

    dict_1_csv = list(set(dict_1_csv))
    patients = []
    for _, row in data1.iterrows():
        one_patient = {}
        for title in dict_1_csv:
            if "lab_blood_routine_examination" in title or "lab_urine_routine_test" in title:
                one_patient[title] = {
                    "test_result" : row[title+"_test_result"].strip(),
                    "test_time" : row[title+"_test_time"].strip(),
                    "pure_item_name" : row[title+"_pure_item_name"].strip(),
                    "item_name" : row[title+"_item_name"].strip(),
                    "qualitative_result" : row[title+"_qualitative_result"].strip(),
                }
                continue
            one_patient[title] = row[title].strip()
        # print(one_patient)
        patients.append(one_patient)
    return patients

def process_data2():
    dict_2_csv = []
    for one in data2.columns:
        if "lab_urine_routine_test" in one:
            if "原值" in one or "归一" in one:
                continue
            prefix2 = "lab_urine_routine_test_"
            pattern2 = re.compile(rf"{prefix2}(.*?)(?=_)")
            match2 = pattern2.search(one)
            if match2:
                if len(match2.group(1)) > 1:
                    dict_2_csv.append(prefix2 + match2.group(1))

            continue
        if "lab_biochemical_tests" in one:
            prefix1 = "lab_biochemical_tests_"
            pattern1 = re.compile(rf"{prefix1}(.*?)(?=_)")
            match1 = pattern1.search(one)
            if match1:
                if len(match1.group(1)) > 1:
                    dict_2_csv.append(prefix1 + match1.group(1))

            continue
        if "lab_stool_routine_test" in one:
            if "原值" in one or "归一" in one:
                continue
            prefix3 = "lab_stool_routine_test_"
            pattern3 = re.compile(rf"{prefix3}(.*?)(?=_)")
            match1 = pattern3.search(one)
            if match1:
                if len(match1.group(1)) > 1:
                    dict_2_csv.append(prefix3 + match1.group(1))
            continue
        dict_2_csv.append(one)
    dict_2_csv = list(set(dict_2_csv))
    patients = []
    for _, row in data2.iterrows():
        one_patient = {}
        for title in dict_2_csv:
            if "lab_stool_routine_test" in title or "lab_urine_routine_test" in title or "lab_biochemical_tests" in title:
                test_time = row.get(title+"_test_time","")
                pure_item_name = row.get(title+"_pure_item_name","")
                if len(test_time)>0:
                    test_time = test_time.strip()
                if len(pure_item_name)>0:
                    pure_item_name = pure_item_name.strip()
                one_patient[title] = {
                    "test_result": row[title + "_test_result"].strip(),
                    "test_time": test_time,
                    "pure_item_name": pure_item_name,
                    "item_name": row[title + "_item_name"].strip(),
                    "qualitative_result": row[title + "_qualitative_result"].strip(),
                }
                continue
            one_patient[title] = row[title].strip()
        # print(one_patient)
        patients.append(one_patient)
    return patients

def process_data3():
    dict_3_csv = []
    for one in data3.columns:
        if "lab_biochemical_tests" in one:
            prefix2 = "lab_biochemical_tests_"
            pattern2 = re.compile(rf"{prefix2}(.*?)(?=_)")
            match2 = pattern2.search(one)
            if match2:
                if len(match2.group(1)) > 1:
                    dict_3_csv.append(prefix2 + match2.group(1))

            continue
        if "lab_myocardial_injury_markers" in one:
            prefix1 = "lab_myocardial_injury_markers_"
            pattern1 = re.compile(rf"{prefix1}(.*?)(?=_)")
            match1 = pattern1.search(one)
            if match1:
                if len(match1.group(1)) > 1:
                    dict_3_csv.append(prefix1 + match1.group(1))
            continue
        if "lab_cancer_marker" in one:
            if "原值" in one or "归一" in one:
                continue
            prefix1 = "lab_cancer_marker_"
            pattern1 = re.compile(rf"{prefix1}(.*?)(?=_)")
            match1 = pattern1.search(one)
            if match1:
                if len(match1.group(1)) > 1:
                    dict_3_csv.append(prefix1 + match1.group(1))
            continue
        if "lab_blood_gas_analysis" in one:
            prefix1 = "lab_blood_gas_analysis_"
            pattern1 = re.compile(rf"{prefix1}(.*?)(?=_)")
            match1 = pattern1.search(one)
            if match1:
                if len(match1.group(1)) > 1:
                    dict_3_csv.append(prefix1 + match1.group(1))
            continue
        if "lab_cruor_testing" in one:
            prefix1 = "lab_cruor_testing_"
            pattern1 = re.compile(rf"{prefix1}(.*?)(?=_)")
            match1 = pattern1.search(one)
            if match1:
                if len(match1.group(1)) > 1:
                    dict_3_csv.append(prefix1 + match1.group(1))
            continue
        if "lab_glucose_metabolism_test" in one:
            prefix1 = "lab_glucose_metabolism_test_"
            pattern1 = re.compile(rf"{prefix1}(.*?)(?=_)")
            match1 = pattern1.search(one)
            if match1:
                if len(match1.group(1)) > 1:
                    dict_3_csv.append(prefix1 + match1.group(1))
            continue
        dict_3_csv.append(one)
    dict_3_csv = list(set(dict_3_csv))
    patients = []
    for _, row in data3.iterrows():
        one_patient = {}
        for title in dict_3_csv:
            if ("lab_biochemical_tests" in title or "lab_myocardial_injury_markers" in title
                    or "lab_cancer_marker" in title  or "lab_blood_gas_analysis" in title
            or "lab_cruor_testing" in title  or "lab_glucose_metabolism_test" in title):
                test_time = row.get(title + "_test_time", "")
                pure_item_name = row.get(title + "_pure_item_name", "")
                test_result = row.get(title + "_test_result", "")
                item_name = row.get(title + "_item_name", "")
                qualitative_result = row.get(title + "_qualitative_result", "")
                if len(test_time) > 0:
                    test_time = test_time.strip()
                if len(pure_item_name) > 0:
                    pure_item_name = pure_item_name.strip()
                if len(test_result) > 0:
                    test_result = test_result.strip()
                if len(item_name) > 0:
                    item_name = item_name.strip()
                if len(qualitative_result) > 0:
                    qualitative_result = qualitative_result.strip()
                one_patient[title] = {
                    "test_result": test_result,
                    "test_time": test_time,
                    "pure_item_name": pure_item_name,
                    "item_name": item_name,
                    "qualitative_result": qualitative_result,
                }
                continue
            one_patient[title] = row[title].strip()
        # print(one_patient)
        patients.append(one_patient)
    return patients


def merge_patients(patients):
    merged_patients = {}
    for patient_data in patients3:
        patient_sn = patient_data['patient_sn']
        if patient_sn not in merged_patients:
            merged_patients[patient_sn] = patient_data
        else:
            for key, value in patient_data.items():
                if key != 'patient_sn':
                    if isinstance(value, dict):
                        for sub_key, sub_value in value.items():
                            merged_patients[patient_sn][key][sub_key] = sub_value
                    else:
                        merged_patients[patient_sn][key] = value

    merged_patients_list = list(merged_patients.values())
    return merged_patients_list
# 清除空值
def clean_empty(data_dict):
    # 遍历字典的键值对，如果值是字典类型，进一步处理其内部的空值元素
    for key, value in data_dict.copy().items():
        if isinstance(value, dict):
            new_value = {k: v for k, v in value.items() if v}
            data_dict[key] = new_value

    # 移除外层字典中值为空字典的键值对
    data_dict = {k: v for k, v in data_dict.items() if v}
    return data_dict


if __name__ == '__main__':
    patients1 = process_data1()
    patients2 = process_data2()
    patients3 = process_data3()
    print(len(patients1))
    patients2 = merge_patients(patients2)
    patients3 = merge_patients(patients3)

