# 🧬 TP53 Variant Classification Pipeline

> A machine learning pipeline for classifying TP53 gene variants by **pathogenicity** (functional vs. non-functional), built on the IARC TP53 Database and delivered through an interactive Streamlit web application.

---

## Table of Contents

- [Overview](#overview)
- [Backend — Machine Learning Pipeline](#backend--machine-learning-pipeline)
  - [Training Data: IARC TP53 Database](#training-data-iarc-tp53-database)
  - [Classification Architecture](#classification-architecture)
  - [Pathogenicity Classification](#pathogenicity-classification)
  - [Feature Engineering](#feature-engineering)
  - [Rule-Based Overrides](#rule-based-overrides)
  - [Model Training & Optimization](#model-training--optimization)
  - [Explainability (SHAP)](#explainability-shap)
- [Frontend — Streamlit Web Application](#frontend--streamlit-web-application)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Visualizations](#visualizations)
- [Technologies Used](#technologies-used)
- [License](#license)

---

## Overview

TP53 is the most frequently mutated gene in human cancer. Mutations in TP53 can result in either **functional** (benign) or **non-functional** (pathogenic) protein. Distinguishing between these categories is critical for clinical decision-making, genetic counseling, and cancer research.

This project implements an end-to-end machine learning pipeline that:

1. **Ingests** patient variant data in standard VCF format
2. **Engineers** biologically meaningful features (Grantham distance, REVEL, BayesDel, AGVGD, SIFT, PolyPhen-2, and more)
3. **Classifies** each variant using robust machine learning models (Random Forest and XGBoost) trained on the IARC TP53 Database (Release 21)
4. **Explains** model predictions using SHAP (SHapley Additive exPlanations) analysis
5. **Delivers** results through a modern Streamlit web interface with user authentication, result history, and publication-ready visualizations

---

## Backend — Machine Learning Pipeline

### Training Data: IARC TP53 Database

The backbone of this project is the **IARC TP53 Database (Release 21)** — the most comprehensive and curated dataset of TP53 mutations worldwide, maintained by the International Agency for Research on Cancer (IARC/WHO).

Two datasets are used:

| Dataset | File | Description | Scale |
|---|---|---|---|
| **Germline Variants** | `GermlineDownload_r21.csv` | Inherited TP53 mutations collected from Li-Fraumeni families and clinical genetic testing | ~2.3 MB |
| **Tumor/Somatic Variants** | `TumorVariantDownload_r21-2.csv` | Somatic TP53 mutations observed in human tumors across all cancer types | ~21.6 MB |

Each record in the IARC database contains rich annotation fields:

- **Mutation identity**: Codon number, wild-type and mutant amino acids/nucleotides, genomic coordinates (hg38)
- **Functional impact**: TransactivationClass (functional, non-functional, partially functional, supertrans)
- **In-silico predictions**: REVEL score, BayesDel score, AGVGD classification, SIFT prediction, PolyPhen-2 prediction
- **Hotspot & context**: Hotspot status, CpG site location
- **Epidemiological data**: TCGA/ICGC/GENIE occurrence counts, cancer topography

The IARC data is **not synthetically generated** — it represents real, clinically observed TP53 mutations curated from published literature, clinical repositories, and large-scale sequencing consortia (TCGA, ICGC, GENIE).

---

### Classification Architecture

The pipeline uses two parallel machine learning models to solve the pathogenicity classification problem:

```text
                    ┌─────────────────────────────────┐
                    │     Input: Patient VCF File      │
                    └─────────────┬───────────────────┘
                                  │
                          ┌───────▼────────┐
                          │ VCF Parsing &   │
                          │ Feature Eng.    │
                          └───────┬────────┘
                                  │
                 ┌────────────────┴────────────────┐
                 │                                 │
        ┌────────▼─────────┐              ┌────────▼─────────┐
        │  Random Forest   │              │ XGBoost (Optuna) │
        │  (GridSearchCV)  │              │                  │
        └────────┬─────────┘              └────────┬─────────┘
                 │                                 │
         ┌───────┴───────┐                 ┌───────┴───────┐
         ▼               ▼                 ▼               ▼
   Functional     Non-functional     Functional     Non-functional
    (benign)       (pathogenic)       (benign)       (pathogenic)
```

---

### Pathogenicity Classification

**Objective**: Determine whether a TP53 variant disrupts protein function.

- **Labels**: `Functional` (class 0) — includes functional & supertrans; `Non-functional` (class 1) — includes non-functional & partially functional
- **Training data**: All unique missense variants from both germline and somatic IARC datasets, deduplicated by `(WT_AA, Mutant_AA, Codon_number)`
- **Ground truth**: The `TransactivationClass` field from the IARC database, which is based on experimentally measured transactivation activity of TP53 mutants in yeast functional assays

**Features used** (8 features):

| Feature | Description | Source |
|---|---|---|
| `Grantham_Score` | Physicochemical distance between wild-type and mutant amino acids | Computed (Grantham 1974 matrix) |
| `REVEL` | Ensemble pathogenicity meta-predictor score (0–1) | IARC database |
| `BAYESDEL` | Deleteriousness score integrating multiple annotations | IARC database |
| `AGVGDClass` | Align-GVGD classification (C0–C65, ordinal encoded) | IARC database |
| `SIFTClass` | SIFT prediction (Tolerated=0, Damaging=1) | IARC database |
| `Polyphen2` | PolyPhen-2 prediction (Benign=0, Possibly=1, Damaging=2) | IARC database |
| `Is_Hotspot` | Whether the mutation is at a known TP53 hotspot codon | IARC database |
| `Is_CpG` | Whether the mutation occurs at a CpG dinucleotide site | IARC database |

**Models**: 
1. **Random Forest Classifier**: tuned with `GridSearchCV` over hyperparameters (`n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`).
2. **XGBoost Classifier**: tuned via **Optuna** (50 trials) optimizing for weighted F1-score and using a robust `scale_pos_weight` to handle class imbalances.

**Evaluation**: 5-fold Stratified Cross-Validation reporting Accuracy, Weighted F1, AUC-ROC, Sensitivity, and Specificity. Both models select their optimal decision threshold by maximizing Youden's J statistic ($TPR - FPR$) on the out-of-fold validation probabilities, ensuring unbiased held-out performance.

---

### Feature Engineering

The `engineer_features()` function in `pipeline.py` transforms raw VCF data into ML-ready features:

1. **Grantham Distance**: Computed from the full 20×20 Grantham substitution matrix (`grantham.py`), measuring the physicochemical difference between the reference and alternate amino acids. Classified into: conservative (≤50), moderately conservative (51–100), moderately radical (101–150), radical (>150).

2. **In-silico Predictor Encoding**: AGVGD (ordinal C0=0 to C65=6), SIFT (binary), PolyPhen-2 (ordinal B=0, P=1, D=2).

3. **Missense-specific handling**: REVEL, BayesDel, AGVGD, SIFT, and PolyPhen-2 are only meaningful for missense variants. Non-missense variants receive neutral defaults (0) to prevent the model from learning spurious patterns.

4. **Database membership flags**: `In_ClinVar` and `In_COSMIC` are derived from the `DB_SOURCE` annotation in the VCF.

---

### Rule-Based Overrides

The ML model is trained on **missense variants only** (amino acid substitutions), but real patient VCFs contain many other variant types. The pipeline applies biologically grounded rule-based overrides:

| Variant Type | Override | Rationale |
|---|---|---|
| **Nonsense** (stop-gain) | → Non-functional (100% confidence) | Truncates the protein, always loss-of-function |
| **Frameshift** | → Non-functional (100%) | Disrupts the reading frame, completely alters downstream protein |
| **Splice** | → Non-functional (100%) | Destroys mRNA splicing, produces aberrant protein |
| **Silent** (synonymous) | → Functional (100%) | No amino acid change, protein is unaffected |
| **Intronic** | → Functional (100%) | Outside coding region, usually no impact on protein |
| **Large deletion** | → Non-functional (100%) | Removes significant portions of the gene |

---

### Model Training & Optimization

Models are trained via `pretrain_models.py` (for RF) and `pretrain_models_xgboost.py` (for XGBoost), and saved as serialized `.pkl` files for fast loading:

```text
models/
├── pathogenicity_model.pkl          # Random Forest model
├── optimal_threshold.pkl            # RF Youden's J optimal threshold
├── xgb_pathogenicity_model.pkl      # XGBoost model
└── xgb_optimal_threshold.pkl        # XGBoost Youden's J optimal threshold
```

The full training + evaluation pipeline is executed by `run_pipeline.py` (RF) and `run_pipeline_xgboost.py` (XGB), which:
1. Train the models with cross-validation
2. Classify a mock patient VCF
3. Extract feature importances
4. Generate thesis-ready visualizations (ROC curves, confusion matrices, feature importance charts, model comparison)

---

### Explainability (SHAP)

The pipeline integrates **SHAP (SHapley Additive exPlanations)** via `TreeExplainer` to interpret the model predictions on the probability scale.

Two scopes of explainability are provided:
1. **Global/Training Explainability**: Visualizes the overall importance and directionality of features across the entire training dataset (n=1093) using beeswarm, bar, and dependence plots.
2. **Local/Case Studies**: Extracts 5 representative variants from the held-out test set for each model (True Positive, True Negative, Edge Case Correct, Edge Case Incorrect, Misclassification) and generates individual force plots to explain *why* the model made a specific prediction.

---

## Frontend — Streamlit Web Application

The web interface (`app.py`) provides:

- **User Authentication**: Registration and login with bcrypt-hashed passwords (SQLite backend via `database.py`)
- **VCF Upload**: Drag-and-drop VCF file upload with real-time pipeline execution
- **Results Dashboard**: Interactive data table with classification predictions, confidence scores, and downloadable CSV export
- **Explainability Tab**: View global SHAP training plots and dynamic local SHAP force plots for selected case study variants across both models.
- **Visualizations**: Class distribution donut charts, feature correlation heatmaps, and feature importance.
- **Result History**: Browse, sort, and manage past uploads with delete functionality

The UI uses a modern dark theme with gradient accents, custom CSS, and the Inter typeface.

---

## Project Structure

```
Tp53 variants analysis/
│
├── app.py                          # Streamlit web application (frontend)
├── pipeline.py                     # Core RF pipeline (parsing, feature engineering, training)
├── pipeline_xgboost.py             # XGBoost optimization and training pipeline
├── pipeline_revel.py               # RF pipeline for REVEL ablation testing
├── pretrain_models.py              # Pre-train and serialize RF models for fast loading
├── pretrain_models_xgboost.py      # Pre-train and serialize XGB models for fast loading
├── run_pipeline.py                 # RF pipeline execution script
├── run_pipeline_xgboost.py         # XGBoost pipeline execution script
├── run_pipeline_revel.py           # Ablation testing execution script
├── explainability.py               # SHAP explainability generation for RF
├── explainability_xgboost.py       # SHAP explainability generation for XGBoost
├── database.py                     # SQLite user/upload management with bcrypt authentication
├── grantham.py                     # Full 20×20 Grantham distance matrix and classification
│
├── IARC_TP53_DB/
│   ├── GermlineDownload_r21.csv        # IARC TP53 Germline Database (R21)
│   └── TumorVariantDownload_r21-2.csv  # IARC TP53 Tumor Variant Database (R21)
│
├── IARC TP53 study/
│   ├── Germline/
│   │   └── germline.py                 # Exploratory data analysis — germline dataset
│   └── Somatic/
│       └── somatic.py                  # Exploratory data analysis — somatic/tumor dataset
│
├── VCF/
│   ├── gen_vcf_files/
│   │   ├── generate_vcf.py             # Generate synthetic VCFs from real IARC data
│   │   └── generate_novel_vcf.py       # Generate VCFs with novel unseen mutations
│   ├── Generated vcf files/            # Generated mock patient VCFs for testing
│   └── Case_Studies_VCF/               # Edge case VCFs for testing
│
├── PLOTS/                              # Plotting scripts and generated visualizations
│   ├── visualizations.py               # Thesis-quality plot generation for RF
│   ├── visualizations_xgboost.py       # Thesis-quality plot generation for XGBoost
│   ├── visualizations_revel.py         # Thesis-quality plot generation for ablation
│   └── *.png                           # Generated visualizations
│
├── models/                             # Pre-trained model artifacts (.pkl)
│   ├── pathogenicity_model.pkl
│   ├── optimal_threshold.pkl
│   ├── xgb_pathogenicity_model.pkl
│   └── xgb_optimal_threshold.pkl
│
├── uploads/                            # User upload storage (per-user, timestamped)
├── case_studies/                       # Generated cases for explainability
├── Feature_importance/                 # Feature importance tables
├── tp53_app.db                         # SQLite database (users + upload records)
│
├── pipeline_results.csv                # Example pipeline output
├── cv_model_comparison.json            # Cross-validation metrics (RF)
├── cv_model_comparison_xgboost.json    # Cross-validation metrics (XGBoost)
└── requirements.txt                    # Python dependencies
```

---

## Installation & Setup

### Prerequisites

- Python 3.10+
- pip

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/tp53-variant-classification.git
cd tp53-variant-classification

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate   # macOS/Linux
# venv\Scripts\activate    # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Pre-train the models (required before first run)
python pretrain_models.py
python pretrain_models_xgboost.py

# 5. Launch the web application
streamlit run app.py
```

### Required Python Packages

```
streamlit
pandas
numpy
scikit-learn
matplotlib
joblib
bcrypt
scipy
```

---

## Usage

### Web Application

1. Run `streamlit run app.py`
2. Create an account or log in
3. Upload a VCF file containing TP53 variants
4. View classification results (pathogenicity and confidence scores)
5. Download results as CSV
6. Access past results from the sidebar

### Command-Line Pipeline

```bash
# Run the full pipeline (train models + classify + generate visualizations)
python run_pipeline.py

# Pre-train models only (for the web app)
python pretrain_models.py
python pretrain_models_xgboost.py

# Generate test VCF files from the IARC data
python VCF/gen_vcf_files/generate_vcf.py

# Generate novel mutation VCFs (for generalization testing)
python VCF/gen_vcf_files/generate_novel_vcf.py

```

---

## Visualizations

The pipeline generates publication-ready visualizations:

- **ROC Curves** — with AUC scores and optimal threshold markers
- **Confusion Matrices** — normalized heatmaps with count and percentage annotations
- **Feature Importance Charts** — horizontal bar charts ranked by Gini/Gain importance
- **SHAP Summary Plots** — beeswarm and bar charts for global explainability
- **SHAP Force Plots** — individual variant explanation plots for case studies
- **Feature Correlation Heatmaps** — visualizing feature independence

All plots use a consistent dark-themed visual style designed for thesis inclusion.

---

## Technologies Used

| Category | Technologies |
|---|---|
| **Machine Learning** | scikit-learn (Random Forest, CV), XGBoost, Optuna, SHAP |
| **Data Processing** | pandas, NumPy |
| **Visualization** | matplotlib |
| **Web Framework** | Streamlit |
| **Database** | SQLite3 |
| **Authentication** | bcrypt |
| **Model Serialization** | joblib |

---

## License

This project was developed as part of a thesis. The IARC TP53 Database is publicly available from the [IARC TP53 Database website](https://tp53.isb-cgc.org/).