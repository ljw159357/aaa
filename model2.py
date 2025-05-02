import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置全局字体解决警告
rcParams['font.family'] = 'sans-serif'

# 加载模型（确保scikit-learn版本匹配）
try:
    model = joblib.load('xgb.pkl')
except Exception as e:
    st.error(f"模型加载失败: {str(e)}")
    st.stop()

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

# 创建输入界面
st.title("Prediction Model for Thrombosis After Lung Transplantation")
st.header("Enter the following feature values:")

# 生成输入控件并收集特征值
feature_values = []
for feature in feature_ranges:
    config = feature_ranges[feature]
    if config["type"] == "numerical":
        value = st.number_input(f"{feature}", value=0.0)
        feature_values.append(float(value))
    else:
        value = st.selectbox(f"{feature}", options=config["options"])
        feature_values.append(int(value))

# 转换为DataFrame保持特征名称
features_df = pd.DataFrame([feature_values], columns=feature_ranges.keys())

if st.button("Predict"):
    try:
        # 执行预测
        proba = model.predict_proba(features_df)[0]
        prediction = model.predict(features_df)[0]
        probability = proba[prediction] * 100

        # 显示预测结果（移除字体指定）
        result_text = f"Predicted probability of hemorrhage: {probability:.2f}%"
        fig, ax = plt.subplots(figsize=(8, 1))
        ax.text(0.5, 0.5, result_text, 
                fontsize=16, 
                ha='center', 
                va='center')
        ax.axis('off')
        st.pyplot(fig)

        # SHAP解释处理
        def extract_model(pipeline):
            """处理可能存在的Pipeline结构"""
            from sklearn.pipeline import Pipeline
            return pipeline[-1] if isinstance(pipeline, Pipeline) else pipeline

        explainer = shap.TreeExplainer(extract_model(model))
        shap_values = explainer.shap_values(features_df)

        # 动态处理SHAP值结构
        if isinstance(shap_values, list):
            # 多分类情况
            class_idx = 1  # 假设关注正类
            expected_value = explainer.expected_value[class_idx]
            shap_val = shap_values[class_idx][0]
        else:
            # 二分类情况
            expected_value = explainer.expected_value[1]
            shap_val = shap_values[0, :]

        # 生成SHAP可视化
        shap_fig = shap.plots.force(
            base_value=expected_value,
            shap_values=shap_val,
            features=features_df,
            matplotlib=True,
            show=False
        )
        st.pyplot(shap_fig)

    except Exception as e:
        st.error(f"预测过程中发生错误: {str(e)}")
        st.exception(e)  # 显示完整错误堆栈
