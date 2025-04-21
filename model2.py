#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# 加载模型
model = joblib.load('xgb.pkl')

# 特征定义
feature_ranges = {
    "Height": {"type": "numerical"},
    "HBP": {"type": "categorical", "options": ["Yes", "No"]},
    "Postoperative Platelet Count (x10⁹/L)": {"type": "numerical"},
    "Urgent Postoperative APTT (s)": {"type": "numerical"},
    "Day 1 Postoperative APTT (s)": {"type": "numerical"},
    "Day 1 Postoperative Antithrombin III Activity (%)": {"type": "numerical"},
    "Postoperative CRRT (Continuous Renal Replacement Therapy)": {"type": "categorical", "options": ["Yes", "No"]},
    "Postoperative Anticoagulation": {"type": "categorical", "options": ["Yes", "No"]},
    "Transplant Side": {"type": "categorical", "options": ["Left", "Right", "Both"]},
    "Primary Graft Dysfunction (PGD, Level)": {"type": "categorical", "options": ["3", "2", "1", "0"]},
}

category_to_numeric_mapping = {
    "Transplant Side": {"Left": 1, "Right": 2, "Both": 0},
    "HBP": {"Yes": 1, "No": 0},
    "Postoperative CRRT (Continuous Renal Replacement Therapy)": {"Yes": 1, "No": 0},
    "Postoperative Anticoagulation": {"Yes": 1, "No": 0},
    "Primary Graft Dysfunction (PGD, Level)": {"3": 3, "2": 2, "1": 1, "0": 0}
}

# 定义特征的缩写
feature_abbr = {
    "Postoperative Platelet Count (x10⁹/L)": "post_plt",
    "Urgent Postoperative APTT (s)": "post_APTT_u",
    "Day 1 Postoperative APTT (s)": "post_APTT_1",
    "Day 1 Postoperative Antithrombin III Activity (%)": "post_antithrombin_III_1",
    "Postoperative CRRT (Continuous Renal Replacement Therapy)": "post_CRRT",
    "Postoperative Anticoagulation": "post_anticoagulation",
    "Transplant Side": "trans_side",
    "Primary Graft Dysfunction (PGD, Level)": "PGD",
    "Height": "height",  # 其他特征也可以添加缩写
    "HBP": "hbp"
}

# UI
st.title("Prediction Model for Thrombosis After Lung Transplantation")
st.header("Enter the following feature values:")

feature_values = []
feature_keys = list(feature_ranges.keys())

# 输入
for feature in feature_keys:
    prop = feature_ranges[feature]
    if prop["type"] == "numerical":
        value = st.number_input(label=f"{feature}", value=0.0)
        feature_values.append(value)
    elif prop["type"] == "categorical":
        value = st.selectbox(label=f"{feature} (Select a value)", options=prop["options"], index=0)
        numeric_value = category_to_numeric_mapping[feature][value]
        feature_values.append(numeric_value)

# 数值特征标准化
numerical_features = [f for f, p in feature_ranges.items() if p["type"] == "numerical"]
numerical_values = [feature_values[feature_keys.index(f)] for f in numerical_features]

# 创建一个标准化器
scaler = StandardScaler()

if numerical_values:
    # 使用新的标准化器进行拟合并转换数值特征
    numerical_values_scaled = scaler.fit_transform([numerical_values])  # 使用fit_transform进行标准化
    for idx, f in enumerate(numerical_features):
        feature_values[feature_keys.index(f)] = numerical_values_scaled[0][idx]

features = np.array([feature_values])


# In[3]:


if st.button("Predict"):
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]
    probability = predicted_proba[predicted_class] * 100

    # 显示预测结果
    text = f"Based on feature values, predicted possibility of hemorrhage after lung transplantation is {probability:.2f}%"
    fig, ax = plt.subplots(figsize=(8, 1))
    ax.text(0.5, 0.5, text, fontsize=16, ha='center', va='center', fontname='Times New Roman', transform=ax.transAxes)
    ax.axis('off')
    plt.savefig("prediction_text.png", bbox_inches='tight', dpi=300)
    st.image("prediction_text.png")

    # SHAP 解释
    # 提取底层模型（支持 pipeline 或直接模型）
    def get_tree_model(model):
        from sklearn.pipeline import Pipeline
        if isinstance(model, Pipeline):
            return model.named_steps['clf']
        return model

    tree_model = get_tree_model(model)
    explainer = shap.TreeExplainer(tree_model)
    shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_keys))

    shap.initjs()

    # 获取缩写特征名列表
    feature_names_abbr = [feature_abbr.get(f, f) for f in feature_keys]  # 用缩写替换特征名

    # 处理 SHAP 输出
    if isinstance(shap_values, list):
        # 如果有多个类别，选择目标类别（例如类别1）
        shap_values_class = shap_values[1]  # 类别 1 的 SHAP 值
    else:
        shap_values_class = shap_values  # 二分类问题中直接使用

    shap_fig = shap.plots.force(
        explainer.expected_value[1],  # 类别 1 的基准值
        shap_values_class[0],  # 类别 1 的 SHAP 值
        pd.DataFrame([feature_values], columns=feature_names_abbr),  # 使用缩写名称
        matplotlib=True,
        show=False  # 不自动显示图形
    )

    st.pyplot(shap_fig)
