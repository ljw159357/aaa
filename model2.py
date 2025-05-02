#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# 加载模型
model = joblib.load('xgb.pkl')

# 特征定义
feature_ranges = {
    "Height": {"type": "numerical"},
    "HBP": {"type": "categorical", "options": ["1", "0"]},  # 1 和 0
    "Postoperative Platelet Count (x10⁹/L)": {"type": "numerical"},
    "Urgent Postoperative APTT (s)": {"type": "numerical"},
    "Day 1 Postoperative APTT (s)": {"type": "numerical"},
    "Day 1 Postoperative Antithrombin III Activity (%)": {"type": "numerical"},
    "Postoperative CRRT (Continuous Renal Replacement Therapy)": {"type": "categorical", "options": ["1", "0"]},  # 1 和 0
    "Postoperative Anticoagulation": {"type": "categorical", "options": ["1", "0"]},  # 1 和 0
    "Transplant Side": {"type": "categorical", "options": ["1", "2", "0"]},  # 1, 2, 0
    "Primary Graft Dysfunction (PGD, Level)": {"type": "categorical", "options": ["3", "2", "1", "0"]},  # 3, 2, 1, 0
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
        feature_values.append(int(value))  # 将分类输入直接转化为整数（例如 "1" -> 1）

features = np.array([feature_values])

# 预测按钮
if st.button("Predict"):
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]
    probability = predicted_proba[predicted_class] * 100

    # 显示预测结果
    text = f"Based on feature values, predicted possibility of hemorrhage after lung transplantation is {probability:.2f}%"
    
    # 强制转换为字符串
    text = str(text)
    
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

    # 更新：修正 SHAP 值的索引
    if isinstance(shap_values, list):
        # 多分类问题，shap_values 为 list，分别对应每个类别的 SHAP 值
        shap_values_for_class = shap_values[1]  # 选择类别 1 的 SHAP 值
    else:
        # 二分类问题，shap_values 是一个 ndarray
        shap_values_for_class = shap_values[:, :, 1]  # 获取类别 1 的 SHAP 值

    shap_fig = shap.plots.force(
        explainer.expected_value[1],  # 类别 1 的基准值
        shap_values_for_class[0, :],  # 类别 1 的 SHAP 值
        pd.DataFrame([feature_values], columns=feature_keys),
        matplotlib=True,
        show=False  # 不自动显示图形
    )

    st.pyplot(shap_fig)
