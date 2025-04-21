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

# 训练时已拟合的标准化器，使用 transform 进行标准化
if numerical_values:
    numerical_values_scaled = scaler.transform([numerical_values])  # 使用 transform 进行标准化
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
    
    # SHAP 值计算
    shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_keys))

    shap.initjs()

    # 修改 SHAP 图的处理
    if isinstance(shap_values, list):  # 如果有多个类别
        shap_fig = shap.plots.force(
            explainer.expected_value[1],  # 类别 1 的基准值
            shap_values[1],  # 获取类别 1 的 SHAP 值
            pd.DataFrame([feature_values], columns=feature_keys),
            matplotlib=True,
            show=False
        )
    else:  # 单一类别
        shap_fig = shap.plots.force(
            explainer.expected_value,  # 获取基准值
            shap_values,  # 获取 SHAP 值
            pd.DataFrame([feature_values], columns=feature_keys),
            matplotlib=True,
            show=False
        )
        
    st.pyplot(shap_fig)
