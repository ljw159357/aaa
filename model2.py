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
    "HBP": {"type": "categorical", "options": ["1", "0"]},
    "Postoperative Platelet Count (x10⁹/L)": {"type": "numerical"},
    "Urgent Postoperative APTT (s)": {"type": "numerical"},
    "Day 1 Postoperative APTT (s)": {"type": "numerical"},
    "Day 1 Postoperative Antithrombin III Activity (%)": {"type": "numerical"},
    "Postoperative CRRT (Continuous Renal Replacement Therapy)": {"type": "categorical", "options": ["1", "0"]},
    "Postoperative Anticoagulation": {"type": "categorical", "options": ["1", "0"]},
    "Transplant Side": {"type": "categorical", "options": ["1", "2", "0"]},
    "Primary Graft Dysfunction (PGD, Level)": {"type": "categorical", "options": ["3", "2", "1", "0"]},
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
        feature_values.append(int(value))  # 将分类输入直接转化为整数

features = np.array([feature_values])

# 预测按钮
if st.button("Predict"):
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]
    probability = predicted_proba[predicted_class] * 100

    # 显示预测结果
    text = f"Based on feature values, predicted possibility of hemorrhage after lung transplantation is {probability:.2f}%"
    fig, ax = plt.subplots(figsize=(8, 1))
    ax.text(0.5, 0.5, text, fontsize=16, ha='center', va='center', transform=ax.transAxes)  # 移除 fontname，避免字体问题
    ax.axis('off')
    plt.savefig("prediction_text.png", bbox_inches='tight', dpi=300)
    st.image("prediction_text.png")

    # SHAP 解释
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_keys))

    shap.initjs()

    # 根据 SHAP 值类型调整访问方式
    if isinstance(shap_values, list):
        # 多分类问题
        shap_values_for_class = shap_values[1]  # 选择类别 1 的 SHAP 值
        expected_value = explainer.expected_value[1]
    else:
        # 二分类问题
        shap_values_for_class = shap_values  # 直接使用 shap_values
        expected_value = explainer.expected_value

    shap_fig = shap.force_plot(
        expected_value,  # 基准值
        shap_values_for_class[0],  # 第一个样本的 SHAP 值
        pd.DataFrame([feature_values], columns=feature_keys),
        matplotlib=True,
        show=False
    )

    st.pyplot(shap_fig)
