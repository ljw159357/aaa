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
# UI Feature definitions (display names)
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

# --------------------------------------------------
# 1) Display‑name  →  Internal‑name 映射
#    **务必与训练阶段保持完全一致！**
# --------------------------------------------------
rename_cols = {
    "Height": "height",
    "HBP": "HBP",
    "Postoperative Platelet Count (x10⁹/L)": "post_PLT",
    "Urgent Postoperative APTT (s)": "post_APTT_e",
    "Day 1 Postoperative APTT (s)": "post_APTT_1",
    "Day 1 Postoperative Antithrombin III Activity (%)": "post_antithrombin_III_1",
    "Postoperative CRRT (Continuous Renal Replacement Therapy)": "post_CRRT",
    "Postoperative Anticoagulation": "anticoagulant_therapy",
    "Transplant Side": "transplant_side",
    "Primary Graft Dysfunction (PGD, Level)": "PGD",
}

# --------------------------------------------------
# 2) 类别取值映射（键用内部列名）
# --------------------------------------------------
categorical_mapping_internal = {
    "HBP": {"Yes": 1, "No": 0},
    "post_CRRT": {"Yes": 1, "No": 0},
    "anticoagulant_therapy": {"Yes": 1, "No": 0},
    "transplant_side": {"Left": 1, "Right": 2, "Both": 0},
    "PGD": {"3": 3, "2": 2, "1": 1, "0": 0},
}

# 原始分类/数值列（display 名）
numerical_cols_display = [k for k, v in feature_defs.items() if v[0] == "numerical"]
categorical_cols_display = [k for k, v in feature_defs.items() if v[0] == "categorical"]

# 对应的内部列名列表
numerical_cols_internal = [rename_cols[c] for c in numerical_cols_display]
categorical_cols_internal = [rename_cols[c] for c in categorical_cols_display]

# --------------------------------------------------
# Utility – figure serialization
# --------------------------------------------------
def _fig_to_png_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=300)
    buf.seek(0)
    return buf.read()

# --------------------------------------------------
# Load model & (optional) external scaler
# --------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_assets():
    model_ = joblib.load("xgb.pkl")      # ← 你的模型
    scaler_ = None
    if not isinstance(model_, Pipeline):
        try:
            scaler_ = joblib.load("scaler.pkl")  # ← 训练用的 Standard/MinMaxScaler
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

user_df_raw = pd.DataFrame([user_inputs])  # 列名 = display 名

# --------------------------------------------------
# Pre‑processing – mirror training pipeline
# --------------------------------------------------
# 1) 改列名
user_df_proc = user_df_raw.rename(columns=rename_cols)

# 2) 类别映射
user_df_proc[categorical_cols_internal] = (
    user_df_proc[categorical_cols_internal]
        .replace(categorical_mapping_internal)
)

# 3) 数值标准化（仅在模型非 Pipeline 且存在外部 scaler 时）
if (external_scaler is not None) and (not uses_pipeline):
    user_df_proc[numerical_cols_internal] = external_scaler.transform(
        user_df_proc[numerical_cols_internal]
    )

# 4) （可选）保证列顺序严格等于训练列顺序
if hasattr(model, "feature_names_in_"):
    user_df_proc = user_df_proc[model.feature_names_in_]

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
    shap_values = explainer(user_df_proc)
    instance_exp = shap_values[0]
    if instance_exp.values.ndim == 2:          # 多输出时取第 2 列（正类）
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
    shap_vec = instance_exp.values
    feature_vals = instance_exp.data
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
