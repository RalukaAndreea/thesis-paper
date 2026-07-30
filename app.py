#!/usr/bin/env python3
"""TP53 Variant Classification — Streamlit Web App."""

import os
import sys
import json as _json
from datetime import datetime

import streamlit as st
import pandas as pd
import joblib

# Ensure project imports work
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

from database import init_db, create_user, verify_user, save_upload, get_user_uploads, delete_upload
from pipeline import (
    parse_vcf,
    engineer_features,
    classify_variants,
    OUTPUT_COLUMNS,
)
from PLOTS.visualizations import plot_class_distribution
from PLOTS.visualizations_xgboost import plot_class_distribution_xgb

# ─── App Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="TP53 Variant Classifier",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODELS_DIR = os.path.join(PROJECT_DIR, "models")
UPLOADS_DIR = os.path.join(PROJECT_DIR, "uploads")

# Initialize database
init_db()


# ─── Load Pre-trained Models (cached) ────────────────────────
@st.cache_resource
def load_models():
    """Load pre-trained pathogenicity model and optimal threshold once and cache them."""
    model_path = os.path.join(MODELS_DIR, "pathogenicity_model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            "Missing models/pathogenicity_model.pkl. Run `venv/bin/python pretrain_models.py` first."
        )
    path_model = joblib.load(model_path)

    # Load CV-derived optimal threshold (falls back to 0.5 if not found)
    threshold_path = os.path.join(MODELS_DIR, "optimal_threshold.pkl")
    if os.path.exists(threshold_path):
        optimal_threshold = joblib.load(threshold_path)
    else:
        optimal_threshold = 0.5

    return path_model, optimal_threshold


@st.cache_resource
def load_xgb_models():
    """Load pre-trained XGBoost model and optimal threshold once and cache them."""
    model_path = os.path.join(MODELS_DIR, "xgb_pathogenicity_model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            "Missing models/xgb_pathogenicity_model.pkl. "
            "Run `venv/bin/python pretrain_models_xgboost.py` first."
        )
    xgb_model = joblib.load(model_path)

    threshold_path = os.path.join(MODELS_DIR, "xgb_optimal_threshold.pkl")
    if os.path.exists(threshold_path):
        optimal_threshold = joblib.load(threshold_path)
    else:
        optimal_threshold = 0.5

    return xgb_model, optimal_threshold


def _write_run_metadata(run_dir: str, filename: str, n_variants: int, model_name: str):
    """Write a metadata.json into the run folder."""
    meta = {
        "filename": filename,
        "timestamp": datetime.now().isoformat(),
        "n_variants": n_variants,
        "model": model_name,
    }
    with open(os.path.join(run_dir, "metadata.json"), "w") as f:
        _json.dump(meta, f, indent=2)


def _read_run_model(run_dir: str) -> str:
    """Read which model was used for a run. Defaults to 'Random Forest' for old runs."""
    meta_path = os.path.join(run_dir, "metadata.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path) as f:
                meta = _json.load(f)
            return meta.get("model", "Random Forest")
        except Exception:
            pass
    return "Random Forest"


# ─── Custom CSS ───────────────────────────────────────────────
def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, p, span, label, input, textarea, select, h1, h2, h3, h4, h5, h6, div {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #7f5af0 0%, #2cb67d 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    .sub-header {
        color: #94a3b8;
        font-size: 1rem;
        margin-bottom: 2rem;
    }

    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #7f5af0;
    }

    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        margin-top: 0.3rem;
    }

    .upload-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
        cursor: pointer;
        transition: border-color 0.2s;
    }

    .upload-card:hover {
        border-color: #7f5af0;
    }

    .status-badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }

    .badge-success { background: #064e3b; color: #34d399; }
    .badge-info    { background: #1e3a5f; color: #60a5fa; }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }

    .stButton > button,
    .stFormSubmitButton > button {
        background: linear-gradient(135deg, #7f5af0 0%, #6d28d9 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        transition: all 0.2s;
    }

    .stButton > button:hover,
    .stFormSubmitButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 15px rgba(127, 90, 240, 0.4);
    }


    /* Don't style file uploader and download buttons */
    .stFileUploader button,
    .stDownloadButton > button {
        background: none !important;
        color: inherit !important;
        border: 1px solid rgba(49, 51, 63, 0.2) !important;
        transform: none !important;
        box-shadow: none !important;
        font-weight: 400 !important;
    }

    /* Hide browser password manager key icons */
    input::-webkit-credentials-auto-fill-button,
    input::-webkit-contacts-auto-fill-button {
        display: none !important;
        visibility: hidden !important;
    }

    /* Hide Streamlit form submit hint text */
    .stForm [data-testid="InputInstructions"] {
        display: none !important;
    }
    </style>
    """, unsafe_allow_html=True)


# ─── NAVIGATION HELPER ────────────────────────────────────────
_PAGE_TO_LABEL = {
    "upload": "Upload VCF",
    "current_results": "Current Results",
    "past_results": "Past Results",
}

def _set_page(page_key: str):
    """Set the active page.

    The sidebar radio is synced at the start of the next rerun, before the
    widget is instantiated.
    """
    st.session_state["page"] = page_key


# ─── AUTH PAGES ───────────────────────────────────────────────
def page_login():
    """Login / Register page."""
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<p class="main-header">🧬 TP53 Variant Classifier</p>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Machine Learning Pipeline for TP53 Variant Classification</p>', unsafe_allow_html=True)

        tab_login, tab_register = st.tabs(["Login", "Register"])

        with tab_login:
            with st.form("login_form"):
                username = st.text_input("Username", placeholder="Enter your username")
                password = st.text_input("Password", type="password", placeholder="Enter your password")
                submitted = st.form_submit_button("Login", use_container_width=True)

                if submitted:
                    if not username or not password:
                        st.error("Please fill in all fields.")
                    else:
                        user = verify_user(username, password)
                        if user:
                            st.session_state["user"] = user
                            _set_page("upload")
                            st.rerun()
                        else:
                            st.error("Invalid username or password.")

        with tab_register:
            with st.form("register_form"):
                new_username = st.text_input("Choose a Username", placeholder="Username")
                new_password = st.text_input("Choose a Password", type="password", placeholder="Min 4 characters")
                confirm_password = st.text_input("Confirm Password", type="password", placeholder="Repeat password")
                reg_submitted = st.form_submit_button("Create Account", use_container_width=True)

                if reg_submitted:
                    if not new_username or not new_password:
                        st.error("Please fill in all fields.")
                    elif len(new_password) < 4:
                        st.error("Password must be at least 4 characters.")
                    elif new_password != confirm_password:
                        st.error("Passwords do not match.")
                    else:
                        if create_user(new_username, new_password):
                            st.success(f"Account created! You can now log in as **{new_username}**.")
                        else:
                            st.error("Username already exists. Choose a different one.")


# ─── UPLOAD PAGE ──────────────────────────────────────────────
def page_upload():
    """VCF upload page."""
    st.markdown('<p class="main-header">📤 Upload VCF File</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload a VCF file to classify TP53 variants</p>', unsafe_allow_html=True)

    # Model selector
    model_choice = st.radio(
        "Classification Model",
        ["Random Forest", "XGBoost (Optuna)"],
        index=0,
        horizontal=True,
        help="Select which pre-trained model to use for variant classification.",
    )

    uploaded_file = st.file_uploader(
        "Choose a VCF file",
        type=["vcf"],
        help="Upload a VCF file containing TP53 variants with annotations"
    )

    if uploaded_file is not None:
        st.info(f"📁 **{uploaded_file.name}** — {uploaded_file.size / 1024:.1f} KB")

        if st.button("Run Pipeline", use_container_width=True):
            with st.spinner(f"Running TP53 classification pipeline ({model_choice})..."):
                try:
                    # Load model and threshold based on selection
                    if model_choice == "XGBoost (Optuna)":
                        path_model, optimal_threshold = load_xgb_models()
                    else:
                        path_model, optimal_threshold = load_models()

                    # Save uploaded VCF to temp file
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    user = st.session_state["user"]
                    user_dir = os.path.join(UPLOADS_DIR, user["username"], timestamp)
                    os.makedirs(user_dir, exist_ok=True)

                    vcf_path = os.path.join(user_dir, uploaded_file.name)
                    with open(vcf_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    # Run pipeline
                    df = parse_vcf(vcf_path)
                    df = engineer_features(df)
                    results_df = classify_variants(df, path_model, optimal_threshold=optimal_threshold)

                    # Save results
                    results_csv = os.path.join(user_dir, "results.csv")
                    results_df[OUTPUT_COLUMNS].to_csv(results_csv, index=False)

                    # Generate class distribution plot (model-specific filename)
                    if model_choice == "XGBoost (Optuna)":
                        plot_class_distribution_xgb(results_df, user_dir)
                    else:
                        plot_class_distribution(results_df, user_dir)

                    # Write per-run metadata (records which model was used)
                    _write_run_metadata(user_dir, uploaded_file.name, len(results_df), model_choice)

                    # Save to database
                    upload_id = save_upload(
                        user_id=user["id"],
                        filename=uploaded_file.name,
                        results_dir=user_dir,
                        num_variants=len(results_df),
                    )

                    # Store in session for current results view
                    st.session_state["current_results"] = results_df
                    st.session_state["current_results_dir"] = user_dir
                    st.session_state["current_filename"] = uploaded_file.name
                    st.session_state["current_model"] = model_choice
                    _set_page("current_results")
                    st.rerun()

                except Exception as e:
                    st.error(f"Pipeline error: {e}")
                    import traceback
                    st.code(traceback.format_exc())


# ─── CURRENT RESULTS PAGE ────────────────────────────────────
def page_current_results():
    """Show results from the most recent upload."""
    if "current_results" not in st.session_state:
        st.warning("No current results. Please upload a VCF file first.")
        return

    results_df = st.session_state["current_results"]
    results_dir = st.session_state["current_results_dir"]
    filename = st.session_state.get("current_filename", "upload")
    model_name = st.session_state.get("current_model", _read_run_model(results_dir))

    st.markdown(f'<p class="main-header">📊 Results — {filename}</p>', unsafe_allow_html=True)

    # Model badge
    _render_model_badge(model_name)

    # Metrics row
    n_total = len(results_df)
    n_nonfunc = (results_df["Pathogenicity_Prediction"] == "Non-functional").sum()
    n_func = (results_df["Pathogenicity_Prediction"] == "Functional").sum()

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Variants", n_total)
    c2.metric("Functional", n_func)
    c3.metric("Non-functional", n_nonfunc)

    st.divider()

    # Results table
    st.subheader("📋 Classification Results")
    display_cols = [c for c in OUTPUT_COLUMNS if c in results_df.columns]
    st.dataframe(
        results_df[display_cols],
        use_container_width=True,
        height=400,
    )

    # Download button
    csv_data = results_df[display_cols].to_csv(index=False)
    st.download_button(
        " Download Results CSV",
        csv_data,
        file_name=f"tp53_results_{filename}.csv",
        mime="text/csv",
    )

    st.divider()

    # Visualizations — pick correct plot based on model
    st.subheader("📈 Class Distribution")
    dist_plot = _get_plot_path(results_dir, model_name, "class_distribution.png")
    if os.path.exists(dist_plot):
        col1, col2, col3 = st.columns([1, 1, 1])

        with col2:
            st.image(dist_plot, use_container_width=True)
    else:
        st.info("Class distribution plot not available.")


# ─── PAST RESULTS PAGE ───────────────────────────────────────
def page_past_results():
    """List past uploads and view their results."""
    st.markdown('<p class="main-header">📂 Past Results</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">View results from your previous uploads</p>', unsafe_allow_html=True)

    user = st.session_state["user"]
    uploads = get_user_uploads(user["id"])

    if not uploads:
        st.info("You haven't uploaded any VCF files yet. Go to **Upload VCF** to get started.")
        return

    # Check if we're viewing a specific upload
    if "view_upload_id" in st.session_state and st.session_state["view_upload_id"] is not None:
        upload_id = st.session_state["view_upload_id"]
        upload = next((u for u in uploads if u["id"] == upload_id), None)
        if upload:
            _show_past_result_detail(upload)
            if st.button("← Back to list"):
                st.session_state["view_upload_id"] = None
                st.rerun()
            return

    # Filter and sort controls
    col_header, col_filter, col_sort = st.columns([2, 1, 1])
    with col_filter:
        model_filter = st.selectbox(
            "Model",
            ["All Models", "Random Forest", "XGBoost (Optuna)"],
            label_visibility="collapsed",
        )
    with col_sort:
        sort_by = st.selectbox(
            "Sort by",
            ["Date (newest)", "Date (oldest)", "Name (A–Z)", "Name (Z–A)"],
            label_visibility="collapsed",
        )

    # Attach model name to each upload for filtering
    for u in uploads:
        u["_model"] = _read_run_model(u.get("results_dir", ""))

    # Filter by model
    if model_filter != "All Models":
        uploads = [u for u in uploads if u["_model"] == model_filter]

    if sort_by == "Date (newest)":
        uploads.sort(key=lambda u: u["upload_date"], reverse=True)
    elif sort_by == "Date (oldest)":
        uploads.sort(key=lambda u: u["upload_date"])
    elif sort_by == "Name (A–Z)":
        uploads.sort(key=lambda u: u["filename"].lower())
    elif sort_by == "Name (Z–A)":
        uploads.sort(key=lambda u: u["filename"].lower(), reverse=True)

    with col_header:
        st.markdown(f"**{len(uploads)}** upload(s) found")

    st.divider()

    # Delete confirmation state
    confirm_id = st.session_state.get("_confirm_delete_id", None)

    for upload in uploads:
        date_str = upload["upload_date"][:16].replace("T", " at ")
        run_model = upload.get("_model", _read_run_model(upload.get("results_dir", "")))
        c1, c2, c3, c4 = st.columns([3, 1, 1, 1])
        with c1:
            st.markdown(f"**{upload['filename']}**")
            st.caption(f" {date_str}")
        with c2:
            st.markdown(f"**{upload['num_variants']}** variants")
            _render_model_badge(run_model)
        with c3:
            if st.button("View →", key=f"view_{upload['id']}"):
                st.session_state["view_upload_id"] = upload["id"]
                st.rerun()
        with c4:
            if confirm_id == upload["id"]:
                # Confirmation buttons
                yes, no = st.columns(2)
                with yes:
                    if st.button("✓", key=f"yes_{upload['id']}", help="Confirm delete"):
                        results_dir = delete_upload(upload["id"])
                        if results_dir and os.path.isdir(results_dir):
                            import shutil
                            shutil.rmtree(results_dir, ignore_errors=True)
                        st.session_state["_confirm_delete_id"] = None
                        st.rerun()
                with no:
                    if st.button("✗", key=f"no_{upload['id']}", help="Cancel"):
                        st.session_state["_confirm_delete_id"] = None
                        st.rerun()
            else:
                if st.button("🗑️", key=f"del_{upload['id']}", help="Delete this upload"):
                    st.session_state["_confirm_delete_id"] = upload["id"]
                    st.rerun()
        st.divider()


def _render_model_badge(model_name: str):
    """Render a small colored pill showing which model was used."""
    if "XGBoost" in model_name:
        color, bg = "#e0e0e0", "#7f5af0"
    else:
        color, bg = "#e0e0e0", "#2cb67d"
    st.markdown(
        f'<span style="background:{bg};color:{color};padding:2px 10px;'
        f'border-radius:12px;font-size:0.78rem;font-weight:600;">{model_name}</span>',
        unsafe_allow_html=True,
    )


def _get_plot_path(results_dir: str, model_name: str, base_filename: str) -> str:
    """Return the correct plot path based on the model used."""
    if "XGBoost" in model_name:
        return os.path.join(results_dir, f"xgb_{base_filename}")
    return os.path.join(results_dir, base_filename)


def _show_past_result_detail(upload: dict):
    """Display detailed results for a past upload."""
    st.markdown(f'<p class="main-header">📊 {upload["filename"]}</p>', unsafe_allow_html=True)

    date_str = upload["upload_date"][:16].replace("T", " at ")
    results_dir = upload["results_dir"]
    run_model = _read_run_model(results_dir)

    st.caption(f"Uploaded on {date_str} — {upload['num_variants']} variants")
    _render_model_badge(run_model)

    results_csv = os.path.join(results_dir, "results.csv")

    if not os.path.exists(results_csv):
        st.error("Results file not found. The upload data may have been moved or deleted.")
        return

    results_df = pd.read_csv(results_csv)

    # Metrics
    n_nonfunc = (results_df["Pathogenicity_Prediction"] == "Non-functional").sum()
    n_func = (results_df["Pathogenicity_Prediction"] == "Functional").sum()

    c1, c2 = st.columns(2)
    c1.metric("Functional", n_func)
    c2.metric("Non-functional", n_nonfunc)

    st.divider()

    # Table
    st.subheader("📋 Classification Results")
    st.dataframe(results_df, use_container_width=True, height=400)

    # Download
    csv_data = results_df.to_csv(index=False)
    st.download_button(
        "⬇️ Download Results CSV",
        csv_data,
        file_name=f"tp53_results_{upload['filename']}.csv",
        mime="text/csv",
        key=f"dl_{upload['id']}",
    )

    st.divider()

    # Visualizations — pick correct plot based on model
    st.subheader("📈 Class Distribution")
    dist_plot = _get_plot_path(results_dir, run_model, "class_distribution.png")
    if os.path.exists(dist_plot):
        col1, col2, col3 = st.columns([1, 1, 1])

        with col2:
            st.image(dist_plot, use_container_width=True)
    else:
        st.info("Plot not available for this upload.")


# ─── EXPLAINABILITY / CASE STUDIES PAGE ──────────────────────
def page_explainability():
    """Interactive SHAP explainability & case studies from test data."""
    st.markdown('<p class="main-header"> Explainability & Case Studies</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">SHAP analysis on the 20% hold-out test set — explore how the models make decisions</p>', unsafe_allow_html=True)


    case_dir = os.path.join(PROJECT_DIR, "case_studies")
    if not os.path.isdir(case_dir):
        st.warning(
            "Case studies have not been generated yet. "
            "Run `python explainability.py` first."
        )
        return

    # Model selector (replaces old single-option stage selector)
    MODEL_DIRS = {
        "Random Forest":      "stage1_pathogenicity",
        "XGBoost (Optuna)":   "xgb_pathogenicity",
        "Extended RF (7 Refined Features)": "extended_pathogenicity",
    }
    MODEL_SCRIPTS = {
        "Random Forest":      "explainability.py",
        "XGBoost (Optuna)":   "explainability_xgboost.py",
        "Extended RF (7 Refined Features)": "explainability_extended.py",
    }

    model_label = st.selectbox("Select Model", list(MODEL_DIRS.keys()))
    stage_key = MODEL_DIRS[model_label]
    stage_path = os.path.join(case_dir, stage_key)

    summary_file = os.path.join(stage_path, "summary.json")
    if not os.path.exists(summary_file):
        script = MODEL_SCRIPTS[model_label]
        st.info(
            f"{model_label} explainability hasn't been generated yet. "
            f"Run `python {script}` first."
        )
        return

    import json as _json
    with open(summary_file) as f:
        summary = _json.load(f)

    # ── Test set info ──
    st.divider()
    dist = summary.get("class_distribution", {})
    class_names = summary.get("class_names", ["Class 0", "Class 1"])
    c1, c2, c3 = st.columns(3)
    c1.metric("Test Set Size", summary.get("test_set_size", "?"))
    c2.metric(class_names[0], dist.get(class_names[0], "?"))
    c3.metric(class_names[1], dist.get(class_names[1], "?"))

    # ── Global SHAP ──
    st.divider()
    st.subheader("📊 Global SHAP — Entire Test Set")
    st.caption(
        "These plots show how each feature contributes to the model's "
        "predictions across **all** test samples."
    )

    tab_bee, tab_bar, tab_dep = st.tabs(
        ["Beeswarm (Summary)", "Mean |SHAP| (Bar)", "Dependence Plots"]
    )

    summary_img = os.path.join(stage_path, "global_shap_summary.png")
    bar_img = os.path.join(stage_path, "global_shap_bar.png")

    with tab_bee:
        if os.path.exists(summary_img):
            st.image(summary_img, use_container_width=True)
            st.caption(
                "Each dot is one test sample. Horizontal position = SHAP value "
                "(impact on prediction). Color = feature value (red=high, blue=low)."
            )
        else:
            st.info("Beeswarm plot not found.")

    with tab_bar:
        if os.path.exists(bar_img):
            st.image(bar_img, use_container_width=True)
            st.caption(
                "Mean absolute SHAP value per feature — measures average impact "
                "on model output magnitude."
            )
        else:
            st.info("Bar plot not found.")

    with tab_dep:
        dep_data = summary.get("dependence_plots", [])
        if dep_data:
            st.caption(
                "Each plot shows one feature's value (x-axis) vs. its SHAP impact "
                "(y-axis). The color represents the feature with the strongest "
                "interaction, auto-detected by SHAP."
            )
            for dep in dep_data:
                dep_path = dep.get("path", "")
                feat = dep.get("feature", "?")
                if os.path.exists(dep_path):
                    st.image(dep_path, use_container_width=True)
                else:
                    st.info(f"Dependence plot for {feat} not found.")
        else:
            st.info("No dependence plots available. Re-run `python explainability.py`.")


    # ── Training Set SHAP ──
    train_summary_file = os.path.join(stage_path, "training", "summary.json")
    if os.path.exists(train_summary_file):
        with open(train_summary_file) as f:
            train_summary = _json.load(f)

        st.divider()
        n_train = train_summary.get("n_samples", "?")
        st.markdown(
            '<p class="sub-header">SHAP analysis on the 80% training set — explore what the models learned during training</p>',
            unsafe_allow_html=True)
        st.subheader(f"📊 Global SHAP — Training Set (n={n_train})")
        st.caption(
            "Same SHAP analysis as above, but computed on the **training data** "
            "the model was fitted on. Comparing training vs. test SHAP helps "
            "assess whether the model generalises or overfits to specific patterns."
        )

        train_dir = os.path.join(stage_path, "training")
        tr_tab_bee, tr_tab_bar, tr_tab_dep = st.tabs(
            ["Beeswarm (Summary)", "Mean |SHAP| (Bar)", "Dependence Plots"]
        )

        tr_summary_img = os.path.join(train_dir, "global_shap_summary.png")
        tr_bar_img = os.path.join(train_dir, "global_shap_bar.png")

        with tr_tab_bee:
            if os.path.exists(tr_summary_img):
                st.image(tr_summary_img, use_container_width=True)
                st.caption(
                    "Each dot is one training sample. Horizontal position = SHAP value "
                    "(impact on prediction). Color = feature value (red=high, blue=low)."
                )
            else:
                st.info("Training beeswarm plot not found.")

        with tr_tab_bar:
            if os.path.exists(tr_bar_img):
                st.image(tr_bar_img, use_container_width=True)
                st.caption(
                    "Mean absolute SHAP value per feature on the training set."
                )
            else:
                st.info("Training bar plot not found.")

        with tr_tab_dep:
            tr_dep_data = train_summary.get("dependence_plots", [])
            if tr_dep_data:
                st.caption(
                    "Feature value (x-axis) vs. SHAP impact (y-axis) on training data. "
                    "Color = strongest interaction feature."
                )
                for dep in tr_dep_data:
                    dep_path = dep.get("path", "")
                    feat = dep.get("feature", "?")
                    if os.path.exists(dep_path):
                        st.image(dep_path, use_container_width=True)
                    else:
                        st.info(f"Training dependence plot for {feat} not found.")
            else:
                st.info("No training dependence plots available.")

    # ── Case Studies ──
    st.divider()
    st.subheader(" Individual Case Studies")
    st.caption(
        "Five representative variants extracted from the hold-out test set: "
        "a confident correct prediction for each class, two borderline edge cases "
        "(one correct, one incorrect), and a confident misclassification. "
        "Each includes the IARC MUT_ID, Individual_ID, and a SHAP waterfall "
        "explaining the model's reasoning."
    )

    CASE_LABELS = {
        "true_positive":      (" True Positive",               "success"),
        "true_negative":      (" True Negative",               "success"),
        "edge_case_correct":  (" Edge Case — Correct",        "warning"),
        "edge_case_incorrect":(" Edge Case — Incorrect",      "warning"),
        "error":              (" Misclassification",           "error"),
    }

    cases_data = summary.get("cases", {})

    for case_key, (display_name, badge_type) in CASE_LABELS.items():
        if case_key not in cases_data:
            continue

        case = cases_data[case_key]
        case_subdir = os.path.join(stage_path, f"case_{case_key}")

        with st.expander(f"{display_name} — {case.get('label', '')}", expanded=False):
            # Description
            st.markdown(f"**{case.get('description', '')}**")

            # IARC database identifiers
            mut_id   = case.get("mut_id", "N/A")
            ind_id   = case.get("individual_id", "N/A")
            prot_desc = case.get("prot_description", "N/A")
            st.markdown(
                f" **MUT\_ID:** `{mut_id}` &nbsp;|&nbsp; "
                f" **Individual\_ID:** `{ind_id}` &nbsp;|&nbsp; "
                f" **Mutation:** `{prot_desc}`",
                unsafe_allow_html=True,
            )

            st.divider()

            # Prediction details
            pred_col, true_col, conf_col = st.columns(3)
            pred_col.metric("Predicted", case.get("predicted_label", "?"))
            true_col.metric("True Label", case.get("true_label", "?"))
            conf_val = case.get("confidence", 0)
            conf_col.metric("Confidence", f"{conf_val:.1%}")

            # Probabilities
            p0 = case.get("prob_class_0", 0)
            p1 = case.get("prob_class_1", 0)
            st.markdown(
                f"Probability: **{class_names[0]}** = {p0:.3f} · "
                f"**{class_names[1]}** = {p1:.3f}"
            )

            # Feature values table
            features = case.get("features", {})
            if features:
                st.markdown("**Feature Values:**")
                feat_df = pd.DataFrame(
                    list(features.items()), columns=["Feature", "Value"]
                )
                st.dataframe(feat_df, use_container_width=True, hide_index=True)

            # SHAP waterfall
            wf_path = os.path.join(case_subdir, "shap_waterfall.png")
            if os.path.exists(wf_path):
                st.markdown("**SHAP Waterfall — Why did the model decide this?**")
                st.image(wf_path, use_container_width=True)
                st.caption(
                    "Red bars push the prediction higher (toward "
                    f"{class_names[1]}), blue bars push it lower (toward "
                    f"{class_names[0]}). The bottom shows the base value "
                    "(average prediction) and the top shows the final output."
                )
            else:
                st.info("Waterfall plot not found for this case.")


# ─── AUTOMATED FEATURE SELECTION PAGE ────────────────────────
def page_feature_selection():
    """Automated Feature Selection results & methodology."""
    st.markdown('<p class="main-header">🔬 Automated Feature Selection</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">'
        'RFECV + Permutation Importance analysis on the combined IARC TP53 dataset '
        '(Germline + Somatic, 1367 deduplicated missense variants)'
        '</p>',
        unsafe_allow_html=True,
    )

    FEAT_DIR = os.path.join(PROJECT_DIR, "Feature_importance")

    # ── Section 1: RFECV Results ─────────────────────────────
    st.divider()
    st.subheader("📉 Step 1 — Recursive Feature Elimination (RFECV)")
    st.markdown(
        "RFECV starts with all **13 candidate features**, trains a Random Forest "
        "with 5-fold cross-validation, identifies the weakest feature, removes it, "
        "and repeats. The plot below shows the F1 score at each step."
    )

    rfecv_img = os.path.join(FEAT_DIR, "extended_rfecv.png")
    if os.path.exists(rfecv_img):
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            st.image(rfecv_img, use_container_width=True)
        st.caption(
            "The red star marks the optimal subset size. "
            "Performance peaks at 12 features — adding the 13th (Is\\_CpG) "
            "decreases the cross-validation score."
        )
    else:
        st.info("RFECV plot not found. Run `python automate_feature_selection_extended.py` first.")

    st.markdown("#### RFECV Conclusion")
    st.info("RFECV determined the optimal number of features: **12** (out of 13)")

    col_kept, col_dropped = st.columns(2)
    with col_kept:
        st.markdown("##### ✅ Features Kept (12)")
        kept_features = [
            "Grantham_Score", "REVEL", "BAYESDEL", "AGVGDClass", "SIFTClass",
            "Polyphen2", "Is_Hotspot", "SpliceAI_DS_AG", "SpliceAI_DS_AL",
            "SpliceAI_DS_DG", "SpliceAI_DS_DL", "Exon_Number",
        ]
        for f in kept_features:
            st.markdown(f"- `{f}`")
    with col_dropped:
        st.markdown("##### ❌ Features Dropped (1)")
        st.markdown("- `Is_CpG`")
        st.markdown(
            "*RFECV determined that Is\\_CpG is redundant — "
            "its information is already captured by other predictors "
            "(especially Is\\_Hotspot and the meta-predictors).*"
        )

    # ── Section 2: Permutation Importance ─────────────────────
    st.divider()
    st.subheader("📊 Step 2 — Permutation Importance (Held-Out Test Set)")
    st.markdown(
        "After RFECV selected 12 features, we trained the model on 80% of the data "
        "and measured **Permutation Importance on the 20% held-out test set** "
        "(data the model never saw). For each feature, we shuffled its column "
        "30 times and measured the drop in F1 score."
    )

    perm_img = os.path.join(FEAT_DIR, "extended_permutation_importance.png")
    if os.path.exists(perm_img):
        st.image(perm_img, use_container_width=True)
        st.caption(
            "Each box shows the distribution of F1 drops across 30 shuffles. "
            "Features on the right are the most important. "
            "Features near zero or negative are noise."
        )
    else:
        st.info("Permutation importance plot not found.")

    st.markdown("#### Feature Rankings")
    perm_data = [
        ("BAYESDEL",       0.0233, 0.0132),
        ("REVEL",          0.0200, 0.0133),
        ("Is_Hotspot",     0.0158, 0.0085),
        ("Exon_Number",    0.0141, 0.0079),
        ("Grantham_Score", 0.0110, 0.0076),
        ("SpliceAI_DS_AG", 0.0065, 0.0047),
        ("AGVGDClass",     0.0061, 0.0103),
        ("SpliceAI_DS_DG", 0.0032, 0.0023),
        ("SIFTClass",      0.0012, 0.0041),
        ("SpliceAI_DS_DL",-0.0000, 0.0023),
        ("SpliceAI_DS_AL",-0.0053, 0.0028),
        ("Polyphen2",     -0.0077, 0.0070),
    ]
    perm_df = pd.DataFrame(perm_data, columns=["Feature", "Mean F1 Drop", "Std"])
    perm_df["Mean − Std"] = perm_df["Mean F1 Drop"] - perm_df["Std"]
    perm_df["Verdict"] = perm_df["Mean − Std"].apply(
        lambda x: "✅ Keep" if x > 0 else "❌ Drop (noise)"
    )
    st.dataframe(
        perm_df.style.format({
            "Mean F1 Drop": "{:.4f}",
            "Std": "{:.4f}",
            "Mean − Std": "{:.4f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    # ── Section 3: Held-Out Test Performance (12 features) ───
    st.divider()
    st.subheader("📈 Held-Out Test Performance (12-Feature Model)")
    st.markdown("Performance of the 12-feature model on the 20% held-out test set (274 variants):")

    m1, m2, m3 = st.columns(3)
    m1.metric("Accuracy", "0.8285")
    m2.metric("F1 Score", "0.8288")
    m3.metric("AUC", "0.8989")

    st.markdown("**Classification Report:**")
    report_data = {
        "Class": ["Functional", "Non-functional", "—", "Accuracy", "Macro Avg", "Weighted Avg"],
        "Precision": ["0.79", "0.86", "", "", "0.82", "0.83"],
        "Recall": ["0.81", "0.84", "", "0.83", "0.83", "0.83"],
        "F1-Score": ["0.80", "0.85", "", "0.83", "0.82", "0.83"],
        "Support": ["114", "160", "", "274", "274", "274"],
    }
    st.dataframe(pd.DataFrame(report_data), use_container_width=True, hide_index=True)

    # ── Section 4: Statistical Filtering ─────────────────────
    st.divider()
    st.subheader("🧮 Step 3 — Statistical Noise Filtering")
    st.markdown(
        "While RFECV kept 12 features, Permutation Importance revealed that "
        "some of them introduce **noise** on unseen data. To identify which "
        "features are statistically reliable, we applied a simple filtering rule:"
    )

    st.latex(r"\text{Keep feature if: } \quad \mu_{\text{F1 drop}} - \sigma_{\text{F1 drop}} > 0")

    st.markdown(
        "**Interpretation:** A feature is kept only if its mean contribution to "
        "the model (measured by F1 drop when shuffled) remains **positive even "
        "after subtracting one standard deviation**. This ensures the feature's "
        "importance is not just a random fluctuation."
    )

    st.markdown("#### Calculation for Each Feature")
    calc_data = [
        ("BAYESDEL",       "0.0233 − 0.0132 = **0.0101**", "✅ > 0 → Keep"),
        ("REVEL",          "0.0200 − 0.0133 = **0.0067**", "✅ > 0 → Keep"),
        ("Is_Hotspot",     "0.0158 − 0.0085 = **0.0073**", "✅ > 0 → Keep"),
        ("Exon_Number",    "0.0141 − 0.0079 = **0.0062**", "✅ > 0 → Keep"),
        ("Grantham_Score", "0.0110 − 0.0076 = **0.0034**", "✅ > 0 → Keep"),
        ("SpliceAI_DS_AG", "0.0065 − 0.0047 = **0.0018**", "✅ > 0 → Keep"),
        ("SpliceAI_DS_DG", "0.0032 − 0.0023 = **0.0009**", "✅ > 0 → Keep (borderline)"),
        ("AGVGDClass",     "0.0061 − 0.0103 = **−0.0042**", "❌ < 0 → Drop"),
        ("SIFTClass",      "0.0012 − 0.0041 = **−0.0029**", "❌ < 0 → Drop"),
        ("SpliceAI_DS_DL", "−0.0000 − 0.0023 = **−0.0023**", "❌ < 0 → Drop"),
        ("SpliceAI_DS_AL", "−0.0053 − 0.0028 = **−0.0081**", "❌ < 0 → Drop"),
        ("Polyphen2",      "−0.0077 − 0.0070 = **−0.0147**", "❌ < 0 → Drop"),
    ]
    for feature, calc, verdict in calc_data:
        st.markdown(f"- `{feature}`: {calc} → {verdict}")

    # ── Section 5: Final Refined Feature Set ─────────────────
    st.divider()
    st.subheader("🏆 Final Result — 7 Refined Features")
    st.success(
        "After applying RFECV (Step 1) and Statistical Noise Filtering (Step 3), "
        "the automated pipeline reduced the original 13 candidate features down to "
        "**7 statistically validated features**."
    )

    final_features = [
        ("BAYESDEL",       "Meta-predictor combining multiple pathogenicity scores"),
        ("REVEL",          "Ensemble meta-predictor (13 individual tools combined)"),
        ("Is_Hotspot",     "TP53-specific: mutation at a known cancer hotspot position"),
        ("Exon_Number",    "Spatial context: which exon the mutation falls in (DNA-binding domain = exons 5–8)"),
        ("Grantham_Score", "Biochemical distance between the original and mutant amino acid"),
        ("SpliceAI_DS_AG", "Deep learning prediction: acceptor gain splice disruption"),
        ("SpliceAI_DS_DG", "Deep learning prediction: donor gain splice disruption"),
    ]
    final_df = pd.DataFrame(final_features, columns=["Feature", "Biological Role"])
    st.dataframe(final_df, use_container_width=True, hide_index=True)

    st.markdown(
        "These 7 features cover **three orthogonal dimensions** of pathogenicity:\n"
        "1. **Biochemical impact** (Grantham, REVEL, BayesDel)\n"
        "2. **Genomic context** (Exon\\_Number, Is\\_Hotspot)\n"
        "3. **RNA splicing disruption** (SpliceAI\\_DS\\_AG, SpliceAI\\_DS\\_DG)\n\n"
        "This ensures the model captures diverse mechanisms of TP53 dysfunction "
        "without redundancy or noise."
    )

    # ── Section 6: Feature Correlation Heatmaps ──────────────
    st.divider()
    st.subheader("🔗 Feature Correlation Analysis")
    st.markdown(
        "To further validate our selection, we can examine the Pearson correlation matrices. "
        "Notice how the refined set minimizes highly redundant clusters while retaining "
        "independent predictors."
    )

    corr_base = os.path.join(PROJECT_DIR, "feature_correlation_heatmap.png")
    corr_ext = os.path.join(PROJECT_DIR, "extended_feature_correlation_heatmap.png")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Baseline Features (8)**")
        if os.path.exists(corr_base):
            st.image(corr_base, use_container_width=True)
        else:
            st.info("Baseline correlation heatmap not found.")
            
    with c2:
        st.markdown("**Extended Analysis Model Features (7)**")
        if os.path.exists(corr_ext):
            st.image(corr_ext, use_container_width=True)
        else:
            st.info("Extended correlation heatmap not found.")

# ─── MODEL COMPARISON PAGE ───────────────────────────────────
def page_model_comparison():
    """Side-by-side comparison of the three models."""
    st.markdown('<p class="main-header">⚖️ Model Comparison</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">'
        'Compare the Baseline Random Forest, XGBoost, and Extended Random Forest models.'
        '</p>',
        unsafe_allow_html=True,
    )

    # File paths
    rf_fi = os.path.join(PROJECT_DIR, "stage1_feature_importance.png")
    xgb_fi = os.path.join(PROJECT_DIR, "xgb_feature_importance.png")
    ext_fi = os.path.join(PROJECT_DIR, "extended_feature_importance.png")

    rf_cm = os.path.join(PROJECT_DIR, "stage1_confusion_matrix.png")
    xgb_cm = os.path.join(PROJECT_DIR, "xgb_confusion_matrix.png")
    ext_cm = os.path.join(PROJECT_DIR, "extended_confusion_matrix.png")

    roc_curve = os.path.join(PROJECT_DIR, "PLOTS", "combined_roc_curve.png")

    # ── Section 1: Feature Importances ──
    st.divider()
    st.subheader("📊 Feature Importances (Gini/Gain)")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Baseline RF (8 features)**")
        if os.path.exists(rf_fi):
            st.image(rf_fi, use_container_width=True)
        else:
            st.info("Not found")
            
    with c2:
        st.markdown("**XGBoost (8 features)**")
        if os.path.exists(xgb_fi):
            st.image(xgb_fi, use_container_width=True)
        else:
            st.info("Not found")
            
    with c3:
        st.markdown("**Extended RF (12 features)**")
        if os.path.exists(ext_fi):
            st.image(ext_fi, use_container_width=True)
        else:
            st.info("Not found")

    # ── Section 2: Confusion Matrices ──
    st.divider()
    st.subheader("🧩 Confusion Matrices (Test Set)")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Baseline RF**")
        if os.path.exists(rf_cm):
            st.image(rf_cm, use_container_width=True)
        else:
            st.info("Not found")
            
    with c2:
        st.markdown("**XGBoost**")
        if os.path.exists(xgb_cm):
            st.image(xgb_cm, use_container_width=True)
        else:
            st.info("Not found")
            
    with c3:
        st.markdown("**Extended RF**")
        if os.path.exists(ext_cm):
            st.image(ext_cm, use_container_width=True)
        else:
            st.info("Not found")

    # ── Section 3: ROC Curve ──
    st.divider()
    st.subheader("📈 Combined ROC Curve & Thresholds")
    
    if os.path.exists(roc_curve):
        c1, c2, c3 = st.columns([1, 4, 1])
        with c2:
            st.image(roc_curve, use_container_width=True)
    else:
        st.info("Combined ROC curve not found.")


# ─── SIDEBAR & ROUTING ───────────────────────────────────────
def main():
    inject_css()

    # Initialize session state
    if "user" not in st.session_state:
        st.session_state["user"] = None
    if "page" not in st.session_state:
        st.session_state["page"] = "login"

    # Not logged in → show login
    if st.session_state["user"] is None:
        page_login()
        return

    # Sidebar navigation
    user = st.session_state["user"]

    PAGE_OPTIONS = [
        "Upload VCF", "Current Results", "Past Results", 
        "Model Comparison", "Explainability", "Feature Selection"
    ]
    PAGE_MAP = {
        "Upload VCF": "upload",
        "Current Results": "current_results",
        "Past Results": "past_results",
        "Model Comparison": "model_comparison",
        "Explainability": "explainability",
        "Feature Selection": "feature_selection",
    }
    REVERSE_MAP = {v: k for k, v in PAGE_MAP.items()}

    def _on_nav_change():
        st.session_state["page"] = PAGE_MAP[st.session_state["_nav_radio"]]

    # Determine current index from session state
    current_page = st.session_state.get("page", "upload")
    current_label = REVERSE_MAP.get(current_page, "Upload VCF")
    current_index = PAGE_OPTIONS.index(current_label)

    # Keep the widget state aligned with the routed page before instantiation.
    if st.session_state.get("_nav_radio") != current_label:
        st.session_state["_nav_radio"] = current_label

    with st.sidebar:
        st.markdown(f"### 🧬 TP53 Classifier")
        st.markdown(f"Logged in as **{user['username']}**")
        st.divider()

        st.radio(
            "Navigation",
            PAGE_OPTIONS,
            index=current_index,
            key="_nav_radio",
            on_change=_on_nav_change,
            label_visibility="collapsed",
        )

        st.divider()
        if st.button(" Logout", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    # Route to page
    current_page = st.session_state.get("page", "upload")
    if current_page == "upload":
        page_upload()
    elif current_page == "current_results":
        page_current_results()
    elif current_page == "past_results":
        page_past_results()
    elif current_page == "model_comparison":
        page_model_comparison()
    elif current_page == "explainability":
        page_explainability()
    elif current_page == "feature_selection":
        page_feature_selection()


if __name__ == "__main__":
    main()
