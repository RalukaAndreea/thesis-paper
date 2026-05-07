# 🧬 TP53 Variant Classification Pipeline

> A two-stage machine learning pipeline for classifying TP53 gene variants by **pathogenicity** (functional vs. non-functional) and **origin** (germline vs. somatic), built on the IARC TP53 Database and delivered through an interactive Streamlit web application.

---

## Table of Contents

- [Overview](#overview)
- [Backend — Machine Learning Pipeline](#backend--machine-learning-pipeline)
  - [Training Data: IARC TP53 Database](#training-data-iarc-tp53-database)
  - [Two-Stage Classification Architecture](#two-stage-classification-architecture)
  - [Stage 1 — Pathogenicity Classification](#stage-1--pathogenicity-classification)
  - [Stage 2a — Origin Classification (All Variants)](#stage-2a--origin-classification-all-variants)
  - [Stage 2b — Origin Classification (Non-Functional Only)](#stage-2b--origin-classification-non-functional-only)
  - [Feature Engineering](#feature-engineering)
  - [Rule-Based Overrides](#rule-based-overrides)
  - [Model Training & Optimization](#model-training--optimization)
  - [Statistical Validation](#statistical-validation)
- [Frontend — Streamlit Web Application](#frontend--streamlit-web-application)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Visualizations](#visualizations)
- [Technologies Used](#technologies-used)
- [License](#license)

---

## Overview

TP53 is the most frequently mutated gene in human cancer. Mutations in TP53 can be **germline** (inherited, predisposing to Li-Fraumeni syndrome) or **somatic** (acquired during tumor development), and they can result in either **functional** (benign) or **non-functional** (pathogenic) protein. Distinguishing between these categories is critical for clinical decision-making, genetic counseling, and cancer research.

This project implements an end-to-end machine learning pipeline that:

1. **Ingests** patient variant data in standard VCF format
2. **Engineers** biologically meaningful features (Grantham distance, REVEL, BayesDel, AGVGD, SIFT, PolyPhen-2, and more)
3. **Classifies** each variant across two stages using Random Forest models trained on the IARC TP53 Database (Release 21)
4. **Delivers** results through a modern Streamlit web interface with user authentication, result history, and publication-ready visualizations

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

### Two-Stage Classification Architecture

The pipeline uses a **two-stage cascading architecture** to solve two biologically distinct classification problems:

```
                    ┌─────────────────────────────────┐
                    │     Input: Patient VCF File      │
                    └─────────────┬───────────────────┘
                                  │
                          ┌───────▼────────┐
                          │ VCF Parsing &   │
                          │ Feature Eng.    │
                          └───────┬────────┘
                                  │
                    ┌─────────────▼─────────────────┐
                    │   STAGE 1: Pathogenicity       │
                    │   Functional vs Non-functional  │
                    │   (Random Forest + GridSearchCV) │
                    └──┬──────────────────────────┬──┘
                       │                          │
               ┌───────▼───────┐          ┌───────▼───────┐
               │  Functional   │          │Non-functional │
               │  (benign)     │          │ (pathogenic)  │
               └───────┬───────┘          └───────┬───────┘
                       │                          │
              ┌────────▼────────┐       ┌─────────▼─────────┐
              │ STAGE 2a: Origin│       │ STAGE 2b: Origin   │
              │ Germline vs     │       │ Germline vs Somatic│
              │ Somatic (All)   │       │ (Non-Functional)   │
              └─────────────────┘       └────────────────────┘
```

---

### Stage 1 — Pathogenicity Classification

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

**Model**: Random Forest Classifier with `GridSearchCV` hyperparameter tuning over:
- `n_estimators`: [200, 400, 600]
- `max_depth`: [8, 12, 16, None]
- `min_samples_split`: [2, 5, 10]
- `min_samples_leaf`: [1, 2, 4]

**Evaluation**: 5-fold Stratified Cross-Validation reporting Accuracy, Weighted F1, and AUC-ROC.

---

### Stage 2a — Origin Classification (All Variants)

**Objective**: Predict whether a variant is germline (inherited) or somatic (tumor-acquired) for all classified variants.

**Features used** (8 features):

| Feature | Description |
|---|---|
| `VAF` | Variant Allele Frequency — germline variants cluster ~0.50, somatic variants are lower and more variable |
| `DP` | Total read depth at the variant position |
| `REVEL` | Pathogenicity score |
| `BAYESDEL` | Deleteriousness score |
| `Is_Hotspot` | Hotspot flag |
| `TCGA_COUNT` | Recurrence count in TCGA/ICGC/GENIE |
| `Is_Missense` | Whether the variant is a missense mutation |
| `Is_CpG` | CpG site flag (somatic mutagenesis signature) |

**Training data**: Real unique germline and somatic variants from the IARC database with balanced downsampling (somatic capped at 2× germline count). VAF and DP are simulated with biologically realistic distributions to reflect the known differences between germline heterozygous calls (~50% VAF) and subclonal somatic mutations (~5–40% VAF).

**Model**: Random Forest Classifier (n_estimators=300, max_depth=12).

---

### Stage 2b — Origin Classification (Non-Functional Only)

**Objective**: A specialized origin classifier trained exclusively on non-functional (pathogenic) variants. Non-functional variants have different clinical significance when germline vs. somatic, making this distinction particularly important.

This model is applied only to variants classified as "Non-functional" by Stage 1, providing a more refined prediction for clinically actionable mutations.

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

Models are trained via `pretrain_models.py` and saved as serialized `.pkl` files for fast loading:

```
models/
├── pathogenicity_model.pkl    # Stage 1 (GridSearchCV-optimized RF)
├── origin_model.pkl           # Stage 2a (all variants)
└── nf_origin_model.pkl        # Stage 2b (non-functional only)
```

The full training + evaluation pipeline is executed by `run_pipeline.py`, which:
1. Generates synthetic VCF files from real IARC data
2. Trains all three models with cross-validation
3. Classifies a mock patient VCF
4. Extracts feature importances
5. Generates thesis-ready visualizations (ROC curves, confusion matrices, feature importance charts, model comparison)

---

### Statistical Validation

`statistical_significance.py` validates that the models are learning real patterns, not noise:

- **One-sample t-test**: Model CV accuracy vs. majority-class baseline
- **Binomial test**: Hold-out accuracy vs. random chance
- **Permutation test** (100 permutations): Compares true model score against a null distribution of shuffled labels

---

## Frontend — Streamlit Web Application

The web interface (`app.py`) provides:

- **User Authentication**: Registration and login with bcrypt-hashed passwords (SQLite backend via `database.py`)
- **VCF Upload**: Drag-and-drop VCF file upload with real-time pipeline execution
- **Results Dashboard**: Interactive data table with classification predictions, confidence scores, and downloadable CSV export
- **Visualizations**: Class distribution donut charts for all three classification stages
- **Result History**: Browse, sort, and manage past uploads with delete functionality

The UI uses a modern dark theme with gradient accents, custom CSS, and the Inter typeface.

---

## Project Structure

```
Tp53 variants analysis/
│
├── app.py                          # Streamlit web application (frontend)
├── pipeline.py                     # Core ML pipeline (parsing, feature engineering, training, classification)
├── pretrain_models.py              # Pre-train and serialize models for fast loading
├── run_pipeline.py                 # Full pipeline execution script (train + classify + visualize)
├── visualizations.py               # Thesis-quality plot generation (ROC, confusion matrix, feature importance)
├── database.py                     # SQLite user/upload management with bcrypt authentication
├── grantham.py                     # Full 20×20 Grantham distance matrix and classification
├── generate_vcf.py                 # Generate synthetic VCFs from real IARC data (for testing)
├── generate_novel_vcf.py           # Generate VCFs with novel, unseen mutations (generalization testing)
├── statistical_significance.py     # Statistical tests (t-test, binomial, permutation)
├── germline.py                     # Exploratory data analysis — germline dataset
├── somatic.py                      # Exploratory data analysis — somatic/tumor dataset
│
├── GermlineDownload_r21.csv        # IARC TP53 Germline Database (R21)
├── TumorVariantDownload_r21-2.csv  # IARC TP53 Tumor Variant Database (R21)
│
├── models/                         # Pre-trained model artifacts (.pkl)
│   ├── pathogenicity_model.pkl
│   ├── origin_model.pkl
│   └── nf_origin_model.pkl
│
├── uploads/                        # User upload storage (per-user, timestamped)
├── tp53_app.db                     # SQLite database (users + upload records)
│
├── mock_germline.vcf               # Test VCF — germline variants (sampled from IARC)
├── mock_somatic.vcf                # Test VCF — somatic variants (sampled from IARC)
├── mock_patient.vcf                # Test VCF — mixed patient (germline + somatic)
├── novel_germline.vcf              # Test VCF — novel unseen germline mutations
├── novel_somatic.vcf               # Test VCF — novel unseen somatic mutations
├── novel_patient.vcf               # Test VCF — novel mixed patient
│
├── pipeline_results.csv            # Example pipeline output
├── cv_model_comparison.json        # Cross-validation metrics
├── feature_importances_*.csv       # Feature importance tables
│
├── *.png                           # Generated visualizations
└── requirements.txt                # Python dependencies
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
4. View classification results (pathogenicity, origin, confidence scores)
5. Download results as CSV
6. Access past results from the sidebar

### Command-Line Pipeline

```bash
# Run the full pipeline (train models + classify + generate visualizations)
python run_pipeline.py

# Pre-train models only (for the web app)
python pretrain_models.py

# Generate test VCF files from the IARC data
python generate_vcf.py

# Generate novel mutation VCFs (for generalization testing)
python generate_novel_vcf.py

# Run statistical significance tests
python statistical_significance.py
```

---

## Visualizations

The pipeline generates publication-ready visualizations:

- **ROC Curves** — with AUC scores and optimal threshold markers (Stages 1, 2a, 2b)
- **Confusion Matrices** — normalized heatmaps with count and percentage annotations
- **Feature Importance Charts** — horizontal bar charts ranked by Gini importance
- **Model Comparison** — grouped bar chart comparing Accuracy, Precision, Recall, and F1 across all stages
- **Class Distribution** — donut charts showing prediction breakdowns for each stage

All plots use a consistent dark-themed visual style designed for thesis inclusion.

---

## Technologies Used

| Category | Technologies |
|---|---|
| **Machine Learning** | scikit-learn (Random Forest, GridSearchCV, StratifiedKFold, cross_validate) |
| **Data Processing** | pandas, NumPy |
| **Visualization** | matplotlib |
| **Web Framework** | Streamlit |
| **Database** | SQLite3 |
| **Authentication** | bcrypt |
| **Statistical Testing** | SciPy (t-test, binomial test, permutation test) |
| **Model Serialization** | joblib |

---

## License

This project was developed as part of a thesis. The IARC TP53 Database is publicly available from the [IARC TP53 Database website](https://tp53.isb-cgc.org/).