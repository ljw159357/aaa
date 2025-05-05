import io
import streamlit as st
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(page_title="Hemorrhage Risk Prediction", layout="centered")

# --------------------------------------------------
# Feature definitions & mappings
# --------------------------------------------------
feature_defs = {
    "Height": ("numerical", 0.0),
    "HBP": ("categorical", ["Yes", "No"]),
    "Postoperative Platelet Count (x10⁹/L)": ("numerical", 0.0),
    "Urgent Postoperative APTT (s)": ("numerical", 0.0),
    "Day 1 Postoperative APTT (s)": ("numerical", 0.0),
    "Day 1 Postoperative Antithrombin III Activity (%)": ("numerical", 0.0),
    "Postoperative CRRT (Continuous Renal Replacement Therapy)": ("categorical", ["Yes", "No"]),
    "Postoperative Anticoagulation": ("categorical", ["Yes", "No"]),
    "Transplant Side": ("categorical", ["Left", "Right", "Both"]),
    "Primary Graft Dysfunction (PGD, Level)": ("categorical", ["3", "2", "1", "0"]),
}

categorical_mapping = {
    "HBP": {"Yes": 1, "No": 0},
    "Postoperative CRRT (Continuous Renal Replacement Therapy)": {"Yes": 1, "No": 0},
    "Postoperative Anticoagulation": {"Yes": 1, "No": 0},
    "Transplant Side": {"Left": 1, "Right": 2, "Both": 0},
    "Primary Graft Dysfunction (PGD, Level)": {"3": 3, "2": 2, "1": 1, "0": 0},
}

numerical_cols = [k for k, v in feature_defs.items() if v[0] == "numerical"]
categorical_cols = [k for k, v in feature_defs.items() if v[0] == "categorical"]

# --------------------------------------------------
# Utility – figure serialization
# --------------------------------------------------
def _fig_to_png_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=300)
    buf.seek(0)
    return buf.read()

# --------------------------------------------------
# Load model & scaler
# --------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_assets():
    """
    ① 加载训练好的模型 (xgb.pkl)
    ② 如果模型不是 Pipeline，则尝试加载同一次训练里保存的外部 scaler；
       这样就能复用训练时对数值特征做的“标准化 / 归一化”。
    """
    model_ = joblib.load("xgb.pkl")             # ← 改成你的模型文件名
    scaler_ = None
    if not isinstance(model_, Pipeline):        # 只有非 Pipeline 才需要外部 scaler
        try:
            scaler_ = joblib.load("scaler.pkl") # ← MinMaxScaler / StandardScaler 等
        except FileNotFoundError:
            pass
    return model_, scaler_

model, external_scaler = load_assets()
uses_pipeline = isinstance(model, Pipeline)

# --------------------------------------------------
# UI – gather user inputs
# --------------------------------------------------
st.title("Prediction Model for Hemorrhage After Lung Transplantation")

user_inputs = {}
for feat, (ftype, default) in feature_defs.items():
    if ftype == "numerical":
        user_inputs[feat] = st.number_input(feat, value=float(default))
    else:
        user_inputs[feat] = st.selectbox(feat, feature_defs[feat][1], index=0)

user_df_raw = pd.DataFrame([user_inputs])

# --------------------------------------------------
# Pre‑processing – mirror training pipeline
# --------------------------------------------------
user_df_proc = user_df_raw.copy()
user_df_proc[categorical_cols] = user_df_proc[categorical_cols].replace(categorical_mapping)

# 如果存在外部 scaler 且模型本身不是 Pipeline，则手动做数值特征标准化
if (external_scaler is not None) and (not uses_pipeline):
    user_df_proc[numerical_cols] = external_scaler.transform(user_df_proc[numerical_cols])

# --------------------------------------------------
# Inference & SHAP explanation
# --------------------------------------------------
if st.button("Predict"):
    # ---------------- Prediction ----------------
    proba = model.predict_proba(user_df_proc)[:, 1][0]
    st.success(f"Predicted risk of postoperative hemorrhage: {proba * 100:.2f}%")

    # ---------------- Build SHAP explainer ----------------
    @st.cache_resource(show_spinner=False)
    def build_explainer(_m):
        try:
            return shap.Explainer(_m)
        except Exception:
            if isinstance(_m, Pipeline):
                return shap.TreeExplainer(_m.steps[-1][1])
            raise

    explainer = build_explainer(model)

    # ---------------- Compute SHAP values ----------------
    shap_exp = explainer(user_df_proc)

    # ---------------- Select explanation for positive class (if binary) ----------------
    instance_exp = shap_exp[0]
    if instance_exp.values.ndim == 2:  # shape (n_features, n_outputs)
        instance_exp = instance_exp[:, 1]

    # =====================================================
    # WATERFALL PLOT
    # =====================================================
    st.subheader("Model Explanation – SHAP Waterfall Plot")
    shap.plots.waterfall(instance_exp, max_display=15, show=False)
    fig_water = plt.gcf()
    st.pyplot(fig_water)

    with st.expander("Download SHAP waterfall plot"):
        st.download_button(
            label="Download PNG",
            data=_fig_to_png_bytes(fig_water),
            file_name="shap_waterfall_plot.png",
            mime="image/png",
        )

    # =====================================================
    # FORCE PLOT
    # =====================================================
    st.subheader("Model Explanation – SHAP Force Plot")

    base_val = float(instance_exp.base_values if hasattr(instance_exp.base_values, "__len__") else instance_exp.base_values)
    shap_vec = instance_exp.values            # 1‑D SHAP contributions
    feature_vals = instance_exp.data          # 原始特征值
    feature_names = instance_exp.feature_names

    shap.plots.force(
        base_val,
        shap_vec,
        features=feature_vals,
        feature_names=feature_names,
        matplotlib=True,
        show=False,
    )
    fig_force = plt.gcf()
    st.pyplot(fig_force)

    with st.expander("Download SHAP force plot"):
        st.download_button(
            label="Download PNG",
            data=_fig_to_png_bytes(fig_force),
            file_name="shap_force_plot.png",
            mime="image/png",
        )
