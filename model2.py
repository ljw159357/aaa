import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# 加载模型
model = joblib.load('xgb.pkl')

# 特征定义（保持原始特征名称）
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

# UI界面
st.title("Prediction Model for Thrombosis After Lung Transplantation")
st.header("Enter the following feature values:")

feature_values = []
raw_values = []  # 保存原始值用于SHAP解释
feature_keys = list(feature_ranges.keys())

# 输入处理
for feature in feature_keys:
    prop = feature_ranges[feature]
    if prop["type"] == "numerical":
        value = st.number_input(label=f"{feature}", value=0.0)
        feature_values.append(float(value))
        raw_values.append(value)  # 保存原始值
    elif prop["type"] == "categorical":
        value = st.selectbox(label=f"{feature}", options=prop["options"], index=0)
        numeric_value = category_to_numeric_mapping[feature][value]
        feature_values.append(numeric_value)
        raw_values.append(value)  # 保存原始分类值

# 模型预测和解释
if st.button("Predict"):
    # 转换特征格式
    features_df = pd.DataFrame([feature_values], columns=feature_keys)
    
    # 获取预测结果
    predicted_class = model.predict(features_df)[0]
    predicted_proba = model.predict_proba(features_df)[0]
    probability = predicted_proba[1] * 100  # 直接获取类别1的概率

    # 显示预测结果
    result_text = f"Predicted probability of thrombosis after lung transplantation: {probability:.2f}%"
    fig, ax = plt.subplots(figsize=(8, 1))
    ax.text(0.5, 0.5, result_text, fontsize=16, ha='center', va='center', 
            fontname='Times New Roman', transform=ax.transAxes)
    ax.axis('off')
    st.pyplot(fig)

    # SHAP解释
    def get_model_from_pipeline(model):
        from sklearn.pipeline import Pipeline
        return model.named_steps['clf'] if isinstance(model, Pipeline) else model

    explainer = shap.TreeExplainer(get_model_from_pipeline(model))
    shap_values = explainer.shap_values(features_df)
    
    # 创建特征名称映射（显示原始值）
    formatted_features = []
    for f, v in zip(feature_keys, raw_values):
        if feature_ranges[f]["type"] == "categorical":
            formatted_features.append(f"{f} = {v}")
        else:
            formatted_features.append(f"{f} = {v:.2f}")

    # 绘制SHAP力图
    plt.figure()
    shap.force_plot(
        base_value=explainer.expected_value[1],
        shap_values=shap_values[1][0, :],  # 类别1的SHAP值
        features=features_df.iloc[0],
        feature_names=formatted_features,  # 使用带原始值的特征名称
        matplotlib=True,
        show=False
    )
    st.pyplot(plt.gcf())
