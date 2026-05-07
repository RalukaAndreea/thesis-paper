#!/usr/bin/env python3
"""TP53 Variant Classification — Streamlit Web App."""

import os
import sys
import shutil
import tempfile
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
    PATHOGENICITY_FEATURES,
    ORIGIN_FEATURES,
    OUTPUT_COLUMNS,
)
from visualizations import plot_class_distribution

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
    """Load pre-trained models once and cache them."""
    path_model = joblib.load(os.path.join(MODELS_DIR, "pathogenicity_model.pkl"))
    origin_model = joblib.load(os.path.join(MODELS_DIR, "origin_model.pkl"))
    nf_origin_model = joblib.load(os.path.join(MODELS_DIR, "nf_origin_model.pkl"))
    return path_model, origin_model, nf_origin_model


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

    uploaded_file = st.file_uploader(
        "Choose a VCF file",
        type=["vcf"],
        help="Upload a VCF file containing TP53 variants with annotations"
    )

    if uploaded_file is not None:
        st.info(f"📁 **{uploaded_file.name}** — {uploaded_file.size / 1024:.1f} KB")

        if st.button("Run Pipeline", use_container_width=True):
            with st.spinner("Running TP53 classification pipeline..."):
                try:
                    # Load models
                    path_model, origin_model, nf_origin_model = load_models()

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
                    results_df = classify_variants(df, path_model, origin_model, nf_origin_model)

                    # Save results
                    results_csv = os.path.join(user_dir, "results.csv")
                    results_df[OUTPUT_COLUMNS].to_csv(results_csv, index=False)

                    # Generate class distribution plot
                    plot_class_distribution(results_df, user_dir)

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

    st.markdown(f'<p class="main-header">📊 Results — {filename}</p>', unsafe_allow_html=True)

    # Metrics row
    n_total = len(results_df)
    n_nonfunc = (results_df["Pathogenicity_Prediction"] == "Non-functional").sum()
    n_func = (results_df["Pathogenicity_Prediction"] == "Functional").sum()
    n_germ = (results_df["Origin_Prediction"] == "Germline").sum()
    n_soma = (results_df["Origin_Prediction"] == "Somatic").sum()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Variants", n_total)
    c2.metric("Functional", n_func)
    c3.metric("Non-functional", n_nonfunc)
    c4.metric("Germline", n_germ)
    c5.metric("Somatic", n_soma)

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
        "⬇️ Download Results CSV",
        csv_data,
        file_name=f"tp53_results_{filename}.csv",
        mime="text/csv",
    )

    st.divider()

    # Visualizations
    st.subheader("📈 Class Distribution")
    dist_plot = os.path.join(results_dir, "class_distribution.png")
    if os.path.exists(dist_plot):
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

    # Sort and count
    col_header, col_sort = st.columns([3, 1])
    with col_header:
        st.markdown(f"**{len(uploads)}** upload(s) found")
    with col_sort:
        sort_by = st.selectbox(
            "Sort by",
            ["Date (newest)", "Date (oldest)", "Name (A–Z)", "Name (Z–A)"],
            label_visibility="collapsed",
        )

    if sort_by == "Date (newest)":
        uploads.sort(key=lambda u: u["upload_date"], reverse=True)
    elif sort_by == "Date (oldest)":
        uploads.sort(key=lambda u: u["upload_date"])
    elif sort_by == "Name (A–Z)":
        uploads.sort(key=lambda u: u["filename"].lower())
    elif sort_by == "Name (Z–A)":
        uploads.sort(key=lambda u: u["filename"].lower(), reverse=True)

    st.divider()

    # Delete confirmation state
    confirm_id = st.session_state.get("_confirm_delete_id", None)

    for upload in uploads:
        date_str = upload["upload_date"][:16].replace("T", " at ")
        c1, c2, c3, c4 = st.columns([3, 1, 1, 1])
        with c1:
            st.markdown(f"**{upload['filename']}**")
            st.caption(f"📅 {date_str}")
        with c2:
            st.markdown(f"**{upload['num_variants']}** variants")
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


def _show_past_result_detail(upload: dict):
    """Display detailed results for a past upload."""
    st.markdown(f'<p class="main-header">📊 {upload["filename"]}</p>', unsafe_allow_html=True)

    date_str = upload["upload_date"][:16].replace("T", " at ")
    st.caption(f"Uploaded on {date_str} — {upload['num_variants']} variants")

    results_dir = upload["results_dir"]
    results_csv = os.path.join(results_dir, "results.csv")

    if not os.path.exists(results_csv):
        st.error("Results file not found. The upload data may have been moved or deleted.")
        return

    results_df = pd.read_csv(results_csv)

    # Metrics
    n_nonfunc = (results_df["Pathogenicity_Prediction"] == "Non-functional").sum()
    n_func = (results_df["Pathogenicity_Prediction"] == "Functional").sum()
    n_germ = (results_df["Origin_Prediction"] == "Germline").sum()
    n_soma = (results_df["Origin_Prediction"] == "Somatic").sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Functional", n_func)
    c2.metric("Non-functional", n_nonfunc)
    c3.metric("Germline", n_germ)
    c4.metric("Somatic", n_soma)

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

    # Visualizations
    st.subheader("📈 Class Distribution")
    dist_plot = os.path.join(results_dir, "class_distribution.png")
    if os.path.exists(dist_plot):
        st.image(dist_plot, use_container_width=True)
    else:
        st.info("Plot not available for this upload.")


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

    PAGE_OPTIONS = ["Upload VCF", "Current Results", "Past Results"]
    PAGE_MAP = {
        "Upload VCF": "upload",
        "Current Results": "current_results",
        "Past Results": "past_results",
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
        if st.button("🚪 Logout", use_container_width=True):
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


if __name__ == "__main__":
    main()
