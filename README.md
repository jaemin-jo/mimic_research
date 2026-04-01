# MIMIC-III 임상 데이터 기반 중환자 예후 예측 AI 모델

**Development of Ensemble Machine Learning Models for ICU Patient Outcome Prediction Using MIMIC-III Clinical Data**

> 연세대학교 AI반도체학부 · 데이터사이언스학부 산학연 프로젝트

---

## Overview

MIMIC-III Clinical Database v1.4를 활용하여 중환자실(ICU) 환자의 예후를 예측하는 3가지 AI 모델을 개발한 프로젝트입니다.

| Task | Description | Best Model | Performance |
|------|-------------|------------|-------------|
| **Task 1** | In-Hospital Mortality Prediction | Weighted Ensemble (XGBoost + LightGBM) | **AUROC 0.9880**, AUPRC 0.9301 |
| **Task 2** | ICD-9 Diagnostic Group Classification | LightGBM | Accuracy 0.5994, AUC 0.8894 |
| **Task 3** | Length of Stay Prediction | XGBoost Regressor | MAE 3.384 days, R² 0.606 |

---

## Ultimate Model Results (Task 1: Mortality Prediction)

### Performance Summary

```
+----------------------------------------------------------------------+
| Model                AUROC     AUPRC     F1       Accuracy           |
+----------------------------------------------------------------------+
| Weighted Ensemble    0.9880    0.9301    0.858    0.97     <<<       |
| XGBoost (single)     0.9879    0.9299    0.858    0.97               |
| LightGBM (single)    0.9879    0.9294    0.855    0.97               |
| Stacking Ensemble    0.9869    0.9253    --       --                 |
+----------------------------------------------------------------------+
| 5-Fold CV AUROC:     0.9847 (± 0.0012)                              |
| 5-Fold CV AUPRC:     0.9164                                         |
+----------------------------------------------------------------------+
```

### Classification Report (Optimal Threshold = 0.59)

```
              precision    recall  f1-score   support
Survived       0.98      0.99      0.98      8995
Died           0.91      0.81      0.86      1141
Accuracy                           0.97     10136
```

### Performance Evolution

| Version | Features | AUROC | AUPRC | Key Addition |
|---------|----------|-------|-------|--------------|
| v1 Baseline | 37 | 0.8425 | 0.4348 | Demographics + admission info |
| v2 Clinical | 255 | 0.9353 | 0.7277 | + Lab values + Vital signs (24h) |
| **v3 Ultimate** | **662** | **0.9880** | **0.9301** | + SOFA + Vasopressors + NLP + Ensemble |

### Top 10 Feature Importance

| Rank | Feature | Importance | Clinical Meaning |
|------|---------|------------|------------------|
| 1 | GCS Verbal (max) | 0.1377 | Consciousness level - verbal response |
| 2 | GCS Motor (last) | 0.0351 | Consciousness level - motor response |
| 3 | DNR keyword (NLP) | 0.0201 | "Do Not Resuscitate" in clinical notes |
| 4 | GCS Verbal (trend) | 0.0195 | Consciousness trend over 48h |
| 5 | GCS Eye (mean) | 0.0189 | Consciousness level - eye opening |
| 6 | NLP Topic #4 | 0.0187 | Latent topic from clinical notes |
| 7 | Norepinephrine Rx | 0.0093 | Vasopressor prescription |
| 8 | Morphine Rx | 0.0092 | Opioid analgesic prescription |
| 9 | Lactate (last) | 0.0081 | Tissue perfusion marker |
| 10 | BUN (min) | 0.0078 | Kidney function indicator |

---

## Feature Engineering (662 Features)

Features extracted from **first 48 hours** of ICU admission:

| Category | Source Table | # Features | Description |
|----------|-------------|------------|-------------|
| Demographics | ADMISSIONS, PATIENTS | 18 | Age, gender, admission type, insurance, ethnicity, time features |
| Diagnosis Groups | DIAGNOSES_ICD | 19 | ICD-9 code group one-hot encoding |
| Lab Results | LABEVENTS | 363 | 33 lab items × (mean, min, max, std, last, trend, 4 time-windows) |
| Vital Signs | CHARTEVENTS | 168 | 13 vitals × (stats + trends + time-windows) + urine output |
| Medications | INPUTEVENTS, PRESCRIPTIONS | 29 | Vasopressor use, fluid totals, 12 high-risk drug counts |
| Clinical Notes NLP | NOTEEVENTS | 65 | TF-IDF SVD 50-dim + 15 clinical keywords |
| **Total** | | **662** | |

### Derived Clinical Scores
- **SOFA Score**: 6 organ components (respiratory, coagulation, liver, cardiovascular, CNS, renal)
- **Shock Index**: HR / SBP
- **PF Ratio**: PaO2 / FiO2
- **Anion Gap**: Na - Cl - HCO3
- **BUN/Creatinine Ratio**

---

## Project Structure

```
mimic_research/
├── mimic-iii-clinical-database-1.4/   # MIMIC-III data (not included, requires PhysioNet access)
│   ├── ADMISSIONS.csv.gz
│   ├── CHARTEVENTS.csv.gz             # 4.3GB - vital signs
│   ├── LABEVENTS.csv.gz               # 336MB - lab results
│   ├── NOTEEVENTS.csv.gz              # 1.1GB - clinical notes
│   ├── INPUTEVENTS_MV.csv.gz          # 151MB - medication inputs
│   └── ... (26 tables total)
├── eda.py                             # Exploratory Data Analysis
├── run_all_tasks.py                   # Baseline models (3 tasks, 37 features)
├── run_enhanced.py                    # Enhanced models (+clinical data, 255 features)
├── run_ultimate.py                    # Ultimate pipeline (662 features, AUROC 0.988)
├── generate_figures.py                # Paper figure generation
├── paper.html                         # Academic paper (2-column layout)
└── paper_figures/                     # Generated figures for paper
    ├── fig1_auroc_comparison.png
    ├── fig2_feature_importance.png
    ├── fig3_roc_curves.png
    └── fig4_three_tasks.png
```

---

## How to Run

### Prerequisites

```bash
pip install pandas numpy scikit-learn xgboost lightgbm matplotlib seaborn
```

### Data Setup

1. Obtain MIMIC-III access from [PhysioNet](https://physionet.org/content/mimiciii/1.4/)
2. Place `.csv.gz` files in `mimic-iii-clinical-database-1.4/` directory

### Execution

```bash
# EDA - Data exploration
python eda.py

# Baseline models (3 tasks, ~3 min)
python run_all_tasks.py

# Enhanced models with clinical data (~10 min)
python run_enhanced.py

# Ultimate pipeline - AUROC 0.988 (~18 min, no GPU required)
python run_ultimate.py

# Generate paper figures
python generate_figures.py
```

---

## Runtime Performance

Full pipeline completes in **18 minutes on CPU** (no GPU required):

| Step | Time | % |
|------|------|---|
| LABEVENTS processing (48h) | 21s | 2.0% |
| CHARTEVENTS processing (4.3GB, 48h) | 298s | 27.8% |
| INPUTEVENTS processing | 25s | 2.3% |
| PRESCRIPTIONS processing | 11s | 1.0% |
| NOTEEVENTS NLP (TF-IDF + SVD) | 134s | 12.5% |
| Feature engineering | 2s | 0.2% |
| **Model training (XGBoost + LightGBM)** | **94s** | **8.8%** |
| Stacking ensemble + 5-fold CV | 486s | 45.4% |
| **Total** | **1,071s** | **100%** |

---

## Tech Stack

- **Python 3.12**
- **XGBoost 3.2** / **LightGBM 4.6** - Gradient boosting ensemble
- **scikit-learn 1.8** - Preprocessing, evaluation, stacking
- **pandas 3.0** - Data processing (direct .csv.gz reading)
- **matplotlib 3.10** / **seaborn 0.13** - Visualization

---

## References

1. Johnson, A.E.W. et al., "MIMIC-III, a freely accessible critical care database," *Scientific Data*, 3, 160035, 2016.
2. Chen, T. and Guestrin, C., "XGBoost: A scalable tree boosting system," *Proc. KDD*, 785-794, 2016.
3. Ke, G. et al., "LightGBM: A highly efficient gradient boosting decision tree," *Proc. NeurIPS*, 3146-3154, 2017.
4. Harutyunyan, H. et al., "Multitask learning and benchmarking with clinical time series data," *Scientific Data*, 6(96), 2019.

---

## License

This project uses the [MIMIC-III Clinical Database](https://physionet.org/content/mimiciii/1.4/) under the PhysioNet Credentialed Health Data License v1.4.0.
