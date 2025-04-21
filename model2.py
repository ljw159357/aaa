import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# 加载模型
model = joblib.load('xgb.pkl')
scaler = StandardScaler()

# 特征定义
feature_ranges = {
    "Height (cm)": {"type": "numerical"},
    "HBP": {"type": "categorical", "options": ["Yes", "No"]},
    "post_plt": {"type": "numerical"},
    "post_APTT_u (s)": {"type": "numerical"},
    "post_APTT_1 (s)": {"type": "numerical"},
    "post_AIII(%)": {"type": "numerical"},
    "post_CRRT": {"type": "categorical", "options": ["Yes", "No"]},
    "post_anti": {"type": "categorical", "options": ["Yes", "No"]},
    "Transplant Side": {"type": "categorical", "options": ["Left", "Right", "Both"]},
    "PGD": {"type": "categorical", "options": ["3", "2", "1", "0"]},
}

category_to_numeric_mapping = {
    "Transplant Side": {"Left": 1, "Right": 2, "Both": 0},
    "HBP": {"Yes": 1, "No": 0},
    "post_CRRT": {"Yes": 1, "No": 0},
    "post_anti": {"Yes": 1, "No": 0},
    "PGD": {"3": 3, "2": 2, "1": 1, "0": 0}
}

# UI
st.title("Prediction Model for Hemorrhage After Lung Transplantation")
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

if numerical_values:
    numerical_values_scaled = scaler.fit_transform([numerical_values])
    for idx, f in enumerate(numerical_features):
        feature_values[feature_keys.index(f)] = numerical_values_scaled[0][idx]

features = np.array([feature_values])

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
    
    # 获取 SHAP 值
    shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_keys))
    shap_values_for_display = shap_values[1]  # 类别 1 的 SHAP 值
    base_value = explainer.expected_value[1]  # 类别 1 的基准值
    shap.initjs()
    shap_fig = shap.plots.force(
        base_value,  # 基准值
        shap_values_for_display,  # 类别 1 的 SHAP 值
        pd.DataFrame([feature_values], columns=feature_keys),
        matplotlib=True,
        show=False  # 不自动显示图形
    )
    st.pyplot(shap_fig
