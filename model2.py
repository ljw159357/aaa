#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# 特征的缩写字典
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

# 加载模型
model = joblib.load('xgb.pkl')
scaler = StandardScaler()

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

# UI
st.header("Prediction Model for Thrombosis After Lung Transplantation")
st.write("Enter the following feature values:")

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

if numerical_values:
    numerical_values_scaled = scaler.fit_transform([numerical_values])
    for idx, f in enumerate(numerical_features):
        feature_values[feature_keys.index(f)] = numerical_values_scaled[0][idx]

features = np.array([feature_values])

# 预测
if st.button("Predict"):
    prediction = model.predict(features)[0]
    Predict_proba = model.predict_proba(features)[:, 1][0]
    # 输出概率
    st.write(f"Based on feature values, predicted possibility of hemorrhage after lung transplantation is :  {'%.2f' % float(Predict_proba * 100) + '%'}")
