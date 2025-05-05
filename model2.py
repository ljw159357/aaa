import io, tempfile, os, base64
import streamlit as st
import streamlit.components.v1 as components
import joblib, pandas as pd
import shap, matplotlib.pyplot as plt
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
# Display‑name → Internal‑name 映射（务必与训练一致）
# --------------------------------------------------
rename_cols = {
    "Height": "height",
    "HBP": "HBP",
    "Postoperative Platelet Count (x10⁹/L)": "post_PLT",
    "Urgent Postoperative APTT (s)": "post_APTT_e",
    "Day 1 Postoperative APTT (s)": "post_APTT_1",
    "Day 1 Postoperative Antithrombin III Activity (%)": "post_antithrombase_III_1",
    "Postoperative CRRT (Continuous Renal Replacement Therapy)": "post_CRRT",
    "Postoperative Anticoagulation": "anticoagulant_therapy",
    "Transplant Side": "transplant_side",
    "Primary Graft Dysfunction (PGD, Level)": "PGD",
}

# --------------------------------------------------
# 类别值映射（键用内部列名）
# --------------------------------------------------
categorical_mapping_internal = {
    "HBP": {"Yes": 1, "No": 0},
    "post_CRRT": {"Yes": 1, "No": 0},
    "anticoagulant_therapy": {"Yes": 1, "No": 0},
    "transplant_side": {"Left": 1, "Right": 2, "Both": 0},
    "PGD": {"3": 3, "2": 2, "1": 1, "0": 0},
}

# --------------------------------------------------
# Lists of columns
# --------------------------------------------------
numerical_cols_display    = [k for k, v in feature_defs.items() if v[0] == "numerical"]
categorical_cols_display  = [k for k, v in feature_defs.items() if v[0] == "categorical"]
numerical_cols_internal   = [rename_cols[c] for c in numerical_cols_display]
categorical_cols_internal = [rename_cols[c] for c in categorical_cols_display]

# --------------------------------------------------
# Utility – figure serialization
# --------------------------------------------------
def _fig_to_png_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    buf.seek(0)
    return buf.read()

# --------------------------------------------------
# Load model & (optional) external scaler
# --------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_assets():
    model_  = joblib.load("xgb.pkl")
    scaler_ = None
    if not isinstance(model_, Pipeline):
        try:
            scaler_ = joblib.load("scaler.pkl")
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

user_df_raw = pd.DataFrame([user_inputs])          # display‑name columns

# --------------------------------------------------
# Pre‑processing – mirror training pipeline
# --------------------------------------------------
user_df_proc = user_df_raw.rename(columns=rename_cols)
user_df_proc[categorical_cols_internal] = (
    user_df_proc[categorical_cols_internal]
        .replace(categorical_mapping_internal)
)

if (external_scaler is not None) and (not uses_pipeline):
    user_df_proc[numerical_cols_internal] = external_scaler.transform(
        user_df_proc[numerical_cols_internal]
    )

if hasattr(model, "feature_names_in_"):
    user_df_proc = user_df_proc[model.feature_names_in_]

# --------------------------------------------------
# Inference & SHAP explanation
# --------------------------------------------------
if st.button("Predict"):
    proba = model.predict_proba(user_df_proc)[:, 1][0]
    st.success(f"Predicted risk of postoperative hemorrhage: {proba*100:.2f}%")

    # ---------------- Build SHAP explainer ----------------
    @st.cache_resource(show_spinner=False)
    def build_explainer(_m):
        try:
            return shap.Explainer(_m)        # log‑odds explainer
        except Exception:
            if isinstance(_m, Pipeline):
                return shap.TreeExplainer(_m.steps[-1][1])
            raise

    explainer = build_explainer(model)

    # ---------------- Compute SHAP values ----------------
    shap_values = explainer(user_df_proc)
    instance_exp = shap_values[0]
    if instance_exp.values.ndim == 2:          # 多输出 -> 取正类
        instance_exp = instance_exp[:, 1]

    # =====================================================
    # WATERFALL PLOT（位图，保持 log‑odds）
    # =====================================================
    st.subheader("SHAP Waterfall Plot")
    shap.plots.waterfall(instance_exp, max_display=15, show=False)
    fig_water = plt.gcf()
    st.pyplot(fig_water)
    with st.expander("Download waterfall plot"):
        st.download_button("Download PNG", _fig_to_png_bytes(fig_water),
                           "shap_waterfall_plot.png", "image/png")

    # =====================================================
    # FORCE PLOT（交互式 HTML；显示概率）
    # =====================================================
    st.subheader("SHAP Force Plot (Probability)")

    force_plot = shap.plots.force(
        instance_exp.base_values,
        instance_exp.values,
        features=instance_exp.data,
        feature_names=instance_exp.feature_names,
        link="logit",          
        matplotlib=False,      
    )

    # --- 嵌入 Streamlit ---
    html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"
    components.html(html, height=300, scrolling=True)

    # --- （可选）提供 PNG 下载 ---
    with st.expander("Download force plot PNG"):
        # 把交互式图临时渲染为 PNG（利用 shap.screenshot if present）
        try:
            import shap
            tmp_html = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
            shap.save_html(tmp_html.name, force_plot)
            png_buf = shap.utils.save_html_to_png(tmp_html.name)
            os.unlink(tmp_html.name)
            st.download_button("Download PNG", png_buf, "shap_force_plot.png", "image/png")
        except Exception:
            st.info("PNG export unavailable；请使用浏览器右键另存为图片。")
