"""
MIMIC-III Enhanced Mortality Prediction - AUROC 0.9+ Target
LABEVENTS + CHARTEVENTS (first 24h of ICU stay) features added
"""

import os
import warnings
import time
import gc
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    classification_report, accuracy_score
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings('ignore')
np.random.seed(42)

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mimic-iii-clinical-database-1.4')

# ============================================================
# STEP 1: 기본 데이터 로딩
# ============================================================
print("=" * 70)
print(" STEP 1: 기본 데이터 로딩")
print("=" * 70)

t_start = time.time()
admissions = pd.read_csv(os.path.join(BASE, 'ADMISSIONS.csv.gz'))
patients = pd.read_csv(os.path.join(BASE, 'PATIENTS.csv.gz'))
icustays = pd.read_csv(os.path.join(BASE, 'ICUSTAYS.csv.gz'))
diagnoses = pd.read_csv(os.path.join(BASE, 'DIAGNOSES_ICD.csv.gz'))
services = pd.read_csv(os.path.join(BASE, 'SERVICES.csv.gz'))

for df_temp, cols in [
    (admissions, ['ADMITTIME', 'DISCHTIME', 'DEATHTIME', 'EDREGTIME', 'EDOUTTIME']),
    (patients, ['DOB', 'DOD', 'DOD_HOSP']),
    (icustays, ['INTIME', 'OUTTIME']),
]:
    for c in cols:
        if c in df_temp.columns:
            df_temp[c] = pd.to_datetime(df_temp[c], errors='coerce')

# first ICU stay per admission
first_icu = icustays.sort_values('INTIME').groupby('HADM_ID').first().reset_index()
first_icu_map = first_icu.set_index('HADM_ID')[['ICUSTAY_ID', 'INTIME', 'OUTTIME', 'FIRST_CAREUNIT', 'LOS']].to_dict('index')

print(f"  기본 데이터 로딩 완료: {time.time()-t_start:.1f}s")

# ============================================================
# STEP 2: LABEVENTS - 핵심 검사항목 추출 (first 24h)
# ============================================================
print("\n" + "=" * 70)
print(" STEP 2: LABEVENTS 처리 (336MB compressed)")
print("=" * 70)

LAB_ITEMS = {
    50862: 'albumin', 50882: 'bicarbonate', 50885: 'bilirubin',
    50912: 'creatinine', 50902: 'chloride', 50931: 'glucose',
    50971: 'potassium', 50983: 'sodium', 51006: 'bun',
    51222: 'hemoglobin', 51265: 'platelet', 51300: 'wbc',
    51301: 'wbc2', 50813: 'lactate', 51237: 'inr',
    50820: 'ph', 50821: 'pao2', 50818: 'paco2',
    50863: 'alp', 50861: 'alt', 50878: 'ast',
    51277: 'rdw', 51279: 'rbc', 50893: 'calcium',
    50960: 'magnesium', 50970: 'phosphate',
    50809: 'glucose2', 50811: 'hemoglobin2',
    51144: 'bands', 50889: 'crp',
}
LAB_ITEM_IDS = set(LAB_ITEMS.keys())

# ICU 입실 시간 맵 (HADM_ID -> INTIME)
hadm_intime = {}
for _, row in first_icu.iterrows():
    hadm_intime[row['HADM_ID']] = row['INTIME']

t0 = time.time()
lab_chunks = []
chunk_count = 0

# intime을 Series로 (vectorized lookup)
intime_series = pd.Series(hadm_intime)

for chunk in pd.read_csv(
    os.path.join(BASE, 'LABEVENTS.csv.gz'),
    chunksize=500000,
    usecols=['SUBJECT_ID', 'HADM_ID', 'ITEMID', 'CHARTTIME', 'VALUENUM']
):
    chunk_count += 1
    chunk = chunk[chunk['ITEMID'].isin(LAB_ITEM_IDS)]
    chunk = chunk.dropna(subset=['VALUENUM', 'HADM_ID', 'CHARTTIME'])
    if len(chunk) == 0:
        continue
    chunk['HADM_ID'] = chunk['HADM_ID'].astype(int)
    chunk['CHARTTIME'] = pd.to_datetime(chunk['CHARTTIME'], errors='coerce')

    # vectorized first 24h filter
    intimes = chunk['HADM_ID'].map(intime_series)
    valid = intimes.notna() & chunk['CHARTTIME'].notna()
    chunk = chunk[valid].copy()
    intimes = intimes[valid]
    hours_diff = (chunk['CHARTTIME'] - intimes).dt.total_seconds() / 3600
    chunk = chunk[(hours_diff >= 0) & (hours_diff <= 24)]

    if len(chunk) > 0:
        lab_chunks.append(chunk[['HADM_ID', 'ITEMID', 'VALUENUM']].copy())

    if chunk_count % 10 == 0:
        recs = sum(len(c) for c in lab_chunks)
        print(f"    chunk {chunk_count}, {recs:,} records ({time.time()-t0:.0f}s)")

lab_df = pd.concat(lab_chunks, ignore_index=True) if lab_chunks else pd.DataFrame()
print(f"  LABEVENTS 완료: {len(lab_df):,} records (first 24h), {time.time()-t0:.1f}s")

# Lab name 매핑
lab_df['LAB_NAME'] = lab_df['ITEMID'].map(LAB_ITEMS)
# wbc2 -> wbc, glucose2 -> glucose, hemoglobin2 -> hemoglobin 통합
lab_df['LAB_NAME'] = lab_df['LAB_NAME'].replace({'wbc2': 'wbc', 'glucose2': 'glucose', 'hemoglobin2': 'hemoglobin'})

# 이상치 제거 (극단값)
lab_df = lab_df[(lab_df['VALUENUM'] > -1000) & (lab_df['VALUENUM'] < 100000)]

# 집계: 입원(HADM_ID)별, 검사항목별 통계
lab_agg = lab_df.groupby(['HADM_ID', 'LAB_NAME'])['VALUENUM'].agg(['mean', 'min', 'max', 'std', 'last']).reset_index()
lab_agg.columns = ['HADM_ID', 'LAB_NAME', 'mean', 'min', 'max', 'std', 'last']

# Pivot to wide format
lab_features = {}
for stat in ['mean', 'min', 'max', 'std', 'last']:
    pivot = lab_agg.pivot_table(index='HADM_ID', columns='LAB_NAME', values=stat, aggfunc='first')
    pivot.columns = [f'lab_{col}_{stat}' for col in pivot.columns]
    lab_features[stat] = pivot

lab_wide = pd.concat(lab_features.values(), axis=1).reset_index()
print(f"  Lab features: {lab_wide.shape[1]-1} features for {lab_wide.shape[0]:,} admissions")

del lab_df, lab_chunks, lab_agg
gc.collect()

# ============================================================
# STEP 3: CHARTEVENTS - 핵심 활력징후 추출 (first 24h)
# ============================================================
print("\n" + "=" * 70)
print(" STEP 3: CHARTEVENTS 처리 (4.3GB compressed - 시간 소요)")
print("=" * 70)

VITAL_ITEMS = {
    # Heart Rate
    211: 'hr', 220045: 'hr',
    # Systolic BP
    51: 'sbp', 442: 'sbp', 455: 'sbp', 6701: 'sbp', 220179: 'sbp', 220050: 'sbp',
    # Diastolic BP
    8368: 'dbp', 8440: 'dbp', 8441: 'dbp', 8555: 'dbp', 220180: 'dbp', 220051: 'dbp',
    # Mean BP
    456: 'mbp', 52: 'mbp', 6702: 'mbp', 443: 'mbp', 220052: 'mbp', 220181: 'mbp',
    # Respiratory Rate
    615: 'resp', 618: 'resp', 220210: 'resp', 224690: 'resp',
    # SpO2
    646: 'spo2', 220277: 'spo2',
    # Temperature (F)
    678: 'temp', 223761: 'temp',
    # Temperature (C) -> convert to F
    676: 'temp_c', 223762: 'temp_c',
    # GCS Total
    198: 'gcs',
    # GCS components
    723: 'gcs_verbal', 223900: 'gcs_verbal',
    454: 'gcs_motor', 223901: 'gcs_motor',
    184: 'gcs_eye', 220739: 'gcs_eye',
    # Weight (kg)
    762: 'weight', 763: 'weight', 224639: 'weight', 226512: 'weight',
    # FiO2
    190: 'fio2', 3420: 'fio2', 223835: 'fio2',
}
VITAL_ITEM_IDS = set(VITAL_ITEMS.keys())

t0 = time.time()
vital_records = []
chunk_count = 0
total_rows = 0

for chunk in pd.read_csv(
    os.path.join(BASE, 'CHARTEVENTS.csv.gz'),
    chunksize=1000000,
    usecols=['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'ITEMID', 'CHARTTIME', 'VALUENUM'],
    low_memory=False
):
    chunk_count += 1
    total_rows += len(chunk)

    chunk = chunk[chunk['ITEMID'].isin(VITAL_ITEM_IDS)]
    chunk = chunk.dropna(subset=['VALUENUM', 'HADM_ID', 'CHARTTIME'])
    chunk['HADM_ID'] = chunk['HADM_ID'].astype(int)
    chunk['CHARTTIME'] = pd.to_datetime(chunk['CHARTTIME'], errors='coerce')

    # vectorized first 24h filter
    intimes = chunk['HADM_ID'].map(hadm_intime)
    valid_mask = intimes.notna() & chunk['CHARTTIME'].notna()
    chunk = chunk[valid_mask].copy()
    intimes = intimes[valid_mask]
    hours_diff = (chunk['CHARTTIME'] - intimes).dt.total_seconds() / 3600
    chunk = chunk[(hours_diff >= 0) & (hours_diff <= 24)]

    if len(chunk) > 0:
        vital_records.append(chunk[['HADM_ID', 'ITEMID', 'VALUENUM']].copy())

    elapsed = time.time() - t0
    if chunk_count % 50 == 0:
        recs = sum(len(r) for r in vital_records)
        print(f"    chunk {chunk_count}, {total_rows/1e6:.0f}M rows scanned, {recs:,} vital records ({elapsed:.0f}s)")

vital_df = pd.concat(vital_records, ignore_index=True) if vital_records else pd.DataFrame()
print(f"  CHARTEVENTS 완료: {len(vital_df):,} records (first 24h), {time.time()-t0:.1f}s")

# Vital name 매핑
vital_df['VITAL_NAME'] = vital_df['ITEMID'].map(VITAL_ITEMS)

# Temperature C -> F 변환
mask_c = vital_df['VITAL_NAME'] == 'temp_c'
vital_df.loc[mask_c, 'VALUENUM'] = vital_df.loc[mask_c, 'VALUENUM'] * 9/5 + 32
vital_df.loc[mask_c, 'VITAL_NAME'] = 'temp'

# FiO2: fraction -> percentage 통일
mask_fio2_frac = (vital_df['VITAL_NAME'] == 'fio2') & (vital_df['VALUENUM'] <= 1.0)
vital_df.loc[mask_fio2_frac, 'VALUENUM'] = vital_df.loc[mask_fio2_frac, 'VALUENUM'] * 100

# 생리학적 범위 필터 (이상치 제거)
vital_ranges = {
    'hr': (0, 300), 'sbp': (0, 400), 'dbp': (0, 300), 'mbp': (0, 400),
    'resp': (0, 80), 'spo2': (0, 100), 'temp': (90, 115), 'gcs': (3, 15),
    'gcs_verbal': (1, 5), 'gcs_motor': (1, 6), 'gcs_eye': (1, 4),
    'weight': (20, 500), 'fio2': (21, 100),
}
for vital, (vmin, vmax) in vital_ranges.items():
    mask = vital_df['VITAL_NAME'] == vital
    vital_df.loc[mask & ((vital_df['VALUENUM'] < vmin) | (vital_df['VALUENUM'] > vmax)), 'VALUENUM'] = np.nan

vital_df = vital_df.dropna(subset=['VALUENUM'])

# 집계
vital_agg = vital_df.groupby(['HADM_ID', 'VITAL_NAME'])['VALUENUM'].agg(
    ['mean', 'min', 'max', 'std', 'last', 'count']
).reset_index()
vital_agg.columns = ['HADM_ID', 'VITAL_NAME', 'mean', 'min', 'max', 'std', 'last', 'count']

# Pivot
vital_features = {}
for stat in ['mean', 'min', 'max', 'std', 'last']:
    pivot = vital_agg.pivot_table(index='HADM_ID', columns='VITAL_NAME', values=stat, aggfunc='first')
    pivot.columns = [f'vital_{col}_{stat}' for col in pivot.columns]
    vital_features[stat] = pivot

# 측정 횟수도 피처로 추가
count_pivot = vital_agg.pivot_table(index='HADM_ID', columns='VITAL_NAME', values='count', aggfunc='first')
count_pivot.columns = [f'vital_{col}_count' for col in count_pivot.columns]
vital_features['count'] = count_pivot

vital_wide = pd.concat(vital_features.values(), axis=1).reset_index()
print(f"  Vital features: {vital_wide.shape[1]-1} features for {vital_wide.shape[0]:,} admissions")

del vital_df, vital_records, vital_agg
gc.collect()

# ============================================================
# STEP 4: 피처 엔지니어링 + 합치기
# ============================================================
print("\n" + "=" * 70)
print(" STEP 4: 통합 피처 엔지니어링")
print("=" * 70)


def icd9_to_group(code):
    code = str(code).strip()
    if code.startswith('E'): return 16
    if code.startswith('V'): return 17
    try:
        num = int(code[:3])
    except ValueError:
        return 18
    if num <= 139: return 0
    if num <= 239: return 1
    if num <= 279: return 2
    if num <= 289: return 3
    if num <= 319: return 4
    if num <= 389: return 5
    if num <= 459: return 6
    if num <= 519: return 7
    if num <= 579: return 8
    if num <= 629: return 9
    if num <= 679: return 10
    if num <= 709: return 11
    if num <= 739: return 12
    if num <= 759: return 13
    if num <= 779: return 14
    if num <= 799: return 15
    return 18


diagnoses['ICD9_GROUP'] = diagnoses['ICD9_CODE'].apply(icd9_to_group)
diag_count = diagnoses.groupby('HADM_ID').size().reset_index(name='NUM_DIAGNOSES')
primary_diag = diagnoses[diagnoses['SEQ_NUM'] == 1][['HADM_ID', 'ICD9_GROUP']].rename(
    columns={'ICD9_GROUP': 'PRIMARY_ICD9_GROUP'}
)
diag_groups_pivot = diagnoses.groupby(['HADM_ID', 'ICD9_GROUP']).size().unstack(fill_value=0)
diag_groups_pivot.columns = [f'DIAG_GRP_{c}' for c in diag_groups_pivot.columns]
diag_groups_pivot = diag_groups_pivot.reset_index()

icu_count = icustays.groupby('HADM_ID').size().reset_index(name='NUM_ICU_STAYS')
svc_count = services.groupby('HADM_ID').size().reset_index(name='NUM_SERVICES')

# Master dataset
df = admissions.merge(patients[['SUBJECT_ID', 'GENDER', 'DOB', 'DOD', 'EXPIRE_FLAG']], on='SUBJECT_ID', how='left')
df = df.merge(first_icu[['HADM_ID', 'ICUSTAY_ID', 'FIRST_CAREUNIT', 'LOS', 'INTIME']].rename(
    columns={'LOS': 'ICU_LOS', 'FIRST_CAREUNIT': 'ICU_TYPE'}), on='HADM_ID', how='left')
df = df.merge(icu_count, on='HADM_ID', how='left')
df = df.merge(diag_count, on='HADM_ID', how='left')
df = df.merge(primary_diag, on='HADM_ID', how='left')
df = df.merge(diag_groups_pivot, on='HADM_ID', how='left')
df = df.merge(svc_count, on='HADM_ID', how='left')
df = df.merge(lab_wide, on='HADM_ID', how='left')
df = df.merge(vital_wide, on='HADM_ID', how='left')

# Age
df['AGE'] = (df['ADMITTIME'] - df['DOB']).dt.days / 365.25
df.loc[df['AGE'] > 200, 'AGE'] = 91.4

# 성인만
df = df[df['AGE'] >= 18].copy()
df['LOS_DAYS'] = (df['DISCHTIME'] - df['ADMITTIME']).dt.total_seconds() / 86400
df = df[df['LOS_DAYS'] > 0].copy()

# ED 체류
df['ED_DURATION'] = (df['EDOUTTIME'] - df['EDREGTIME']).dt.total_seconds() / 3600
df['HAS_ED'] = df['EDREGTIME'].notna().astype(int)

# 시간 피처
df['ADMIT_HOUR'] = df['ADMITTIME'].dt.hour
df['ADMIT_DOW'] = df['ADMITTIME'].dt.dayofweek
df['ADMIT_MONTH'] = df['ADMITTIME'].dt.month
df['IS_WEEKEND'] = (df['ADMIT_DOW'] >= 5).astype(int)
df['IS_NIGHT'] = ((df['ADMIT_HOUR'] >= 22) | (df['ADMIT_HOUR'] <= 6)).astype(int)

# 파생 임상 피처
if 'vital_sbp_mean' in df.columns and 'vital_dbp_mean' in df.columns:
    df['pulse_pressure'] = df['vital_sbp_mean'] - df['vital_dbp_mean']
if 'vital_sbp_mean' in df.columns and 'vital_hr_mean' in df.columns:
    df['shock_index'] = df['vital_hr_mean'] / df['vital_sbp_mean'].replace(0, np.nan)
if 'lab_pao2_mean' in df.columns and 'vital_fio2_mean' in df.columns:
    df['pf_ratio'] = df['lab_pao2_mean'] / (df['vital_fio2_mean'] / 100).replace(0, np.nan)
if 'lab_bun_mean' in df.columns and 'lab_creatinine_mean' in df.columns:
    df['bun_cr_ratio'] = df['lab_bun_mean'] / df['lab_creatinine_mean'].replace(0, np.nan)
if 'lab_sodium_mean' in df.columns and 'lab_chloride_mean' in df.columns and 'lab_bicarbonate_mean' in df.columns:
    df['anion_gap'] = df['lab_sodium_mean'] - df['lab_chloride_mean'] - df['lab_bicarbonate_mean']

# 범주형 인코딩
cat_cols = ['ADMISSION_TYPE', 'INSURANCE', 'ETHNICITY', 'MARITAL_STATUS', 'ICU_TYPE', 'GENDER']
for col in cat_cols:
    df[col] = df[col].fillna('UNKNOWN')
    le = LabelEncoder()
    df[col + '_ENC'] = le.fit_transform(df[col].astype(str))

# 피처 목록 구성
base_features = [
    'AGE', 'GENDER_ENC', 'ADMISSION_TYPE_ENC', 'INSURANCE_ENC',
    'ETHNICITY_ENC', 'MARITAL_STATUS_ENC', 'ICU_TYPE_ENC',
    'ICU_LOS', 'NUM_ICU_STAYS', 'NUM_DIAGNOSES', 'NUM_SERVICES',
    'HAS_ED', 'ED_DURATION', 'ADMIT_HOUR', 'ADMIT_DOW', 'ADMIT_MONTH',
    'IS_WEEKEND', 'IS_NIGHT',
]
diag_grp_cols = [c for c in df.columns if c.startswith('DIAG_GRP_')]
lab_cols = [c for c in df.columns if c.startswith('lab_')]
vital_cols = [c for c in df.columns if c.startswith('vital_')]
derived_cols = [c for c in ['pulse_pressure', 'shock_index', 'pf_ratio', 'bun_cr_ratio', 'anion_gap'] if c in df.columns]

ALL_FEATURES = base_features + diag_grp_cols + lab_cols + vital_cols + derived_cols

# 결측치 처리 (수치형은 중앙값으로)
for col in ALL_FEATURES:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median() if df[col].notna().sum() > 0 else 0)

# 무한대 처리
df[ALL_FEATURES] = df[ALL_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)

feature_cols = [c for c in ALL_FEATURES if c in df.columns]
print(f"  총 피처 수: {len(feature_cols)}")
print(f"    - 기본 피처: {len(base_features)}")
print(f"    - 진단 그룹: {len(diag_grp_cols)}")
print(f"    - Lab 피처: {len(lab_cols)}")
print(f"    - Vital 피처: {len(vital_cols)}")
print(f"    - 파생 피처: {len(derived_cols)}")
print(f"  데이터: {len(df):,} admissions")


# ============================================================
# STEP 5: 모델 학습 - TASK 1 (Mortality Prediction)
# ============================================================
print("\n" + "=" * 70)
print(" STEP 5: MORTALITY PREDICTION - Enhanced")
print("=" * 70)

X = df[feature_cols].values
y = df['HOSPITAL_EXPIRE_FLAG'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pos_ratio = y_train.sum() / len(y_train)
scale_pos_weight = (1 - pos_ratio) / pos_ratio

print(f"  Train: {len(X_train):,} (pos={y_train.sum():,}, {y_train.mean()*100:.1f}%)")
print(f"  Test:  {len(X_test):,} (pos={y_test.sum():,}, {y_test.mean()*100:.1f}%)")

models = {
    'XGBoost': XGBClassifier(
        n_estimators=1000, max_depth=7, learning_rate=0.03,
        scale_pos_weight=scale_pos_weight, subsample=0.8,
        colsample_bytree=0.6, reg_alpha=0.1, reg_lambda=1.0,
        min_child_weight=5, gamma=0.1,
        random_state=42, eval_metric='aucpr', use_label_encoder=False,
        early_stopping_rounds=50
    ),
    'LightGBM': LGBMClassifier(
        n_estimators=1000, max_depth=8, learning_rate=0.03,
        scale_pos_weight=scale_pos_weight, subsample=0.8,
        colsample_bytree=0.6, reg_alpha=0.1, reg_lambda=1.0,
        min_child_weight=5, num_leaves=127,
        random_state=42, verbose=-1
    ),
}

results = []
trained_models = {}

for name, model in models.items():
    print(f"\n  [{name}] Training...")
    t0 = time.time()

    if name == 'XGBoost':
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
    else:
        model.fit(X_train, y_train)

    elapsed = time.time() - t0
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    auroc = roc_auc_score(y_test, y_proba)
    auprc = average_precision_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)

    results.append({
        'Model': name, 'AUROC': auroc, 'AUPRC': auprc, 'F1': f1, 'Accuracy': acc
    })
    trained_models[name] = model

    print(f"    AUROC:    {auroc:.4f}")
    print(f"    AUPRC:    {auprc:.4f}")
    print(f"    F1:       {f1:.4f}")
    print(f"    Accuracy: {acc:.4f}")
    print(f"    Time:     {elapsed:.1f}s")

results_df = pd.DataFrame(results)
best_name = results_df.sort_values('AUROC', ascending=False).iloc[0]['Model']
best_model = trained_models[best_name]
best_auroc = results_df.sort_values('AUROC', ascending=False).iloc[0]['AUROC']

print(f"\n  >>> BEST MODEL: {best_name} (AUROC = {best_auroc:.4f})")

# Classification report
y_pred_best = best_model.predict(X_test)
y_proba_best = best_model.predict_proba(X_test)[:, 1]
print(f"\n  [Classification Report]")
print(classification_report(y_test, y_pred_best, target_names=['Survived', 'Died']))

# Threshold 최적화
from sklearn.metrics import precision_recall_curve
precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba_best)
f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-8)
best_thresh_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_thresh_idx]
print(f"  [Optimal Threshold for F1]")
print(f"    Threshold: {best_threshold:.4f}")
y_pred_opt = (y_proba_best >= best_threshold).astype(int)
print(f"    F1 (optimized): {f1_score(y_test, y_pred_opt):.4f}")
print(classification_report(y_test, y_pred_opt, target_names=['Survived', 'Died']))

# Feature importance
print(f"\n  [Top 25 Feature Importance - {best_name}]")
imp = pd.Series(best_model.feature_importances_, index=feature_cols)
imp = imp.sort_values(ascending=False).head(25)
for feat, val in imp.items():
    bar = '#' * int(val / imp.max() * 30)
    print(f"    {feat:<35s} {val:.4f} {bar}")

# Cross-validation
print(f"\n  [5-Fold Cross-Validation]")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []
for fold, (tr_idx, val_idx) in enumerate(cv.split(X, y)):
    X_tr, X_val = X[tr_idx], X[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]

    cv_model = LGBMClassifier(
        n_estimators=500, max_depth=8, learning_rate=0.03,
        scale_pos_weight=scale_pos_weight, subsample=0.8,
        colsample_bytree=0.6, random_state=42, verbose=-1, num_leaves=127
    )
    cv_model.fit(X_tr, y_tr)
    y_val_proba = cv_model.predict_proba(X_val)[:, 1]
    fold_auroc = roc_auc_score(y_val, y_val_proba)
    fold_auprc = average_precision_score(y_val, y_val_proba)
    cv_scores.append({'fold': fold+1, 'AUROC': fold_auroc, 'AUPRC': fold_auprc})
    print(f"    Fold {fold+1}: AUROC={fold_auroc:.4f}, AUPRC={fold_auprc:.4f}")

cv_df = pd.DataFrame(cv_scores)
print(f"    Mean AUROC: {cv_df['AUROC'].mean():.4f} (+/- {cv_df['AUROC'].std():.4f})")
print(f"    Mean AUPRC: {cv_df['AUPRC'].mean():.4f} (+/- {cv_df['AUPRC'].std():.4f})")


# ============================================================
# STEP 6: TASK 2 & 3 with enhanced features
# ============================================================
print("\n" + "=" * 70)
print(" STEP 6: TASK 2 - ICD-9 Group Prediction (Enhanced)")
print("=" * 70)

# Task 2: 진단 관련 피처 제외
feature_cols_t2 = [c for c in feature_cols if not c.startswith('DIAG_GRP_')]
df_t2 = df.dropna(subset=['PRIMARY_ICD9_GROUP']).copy()
df_t2['PRIMARY_ICD9_GROUP'] = df_t2['PRIMARY_ICD9_GROUP'].astype(int)

grp_counts = df_t2['PRIMARY_ICD9_GROUP'].value_counts()
valid_groups = grp_counts[grp_counts >= 30].index.tolist()
df_t2 = df_t2[df_t2['PRIMARY_ICD9_GROUP'].isin(valid_groups)]

le_t2 = LabelEncoder()
y2 = le_t2.fit_transform(df_t2['PRIMARY_ICD9_GROUP'].values)
X2 = df_t2[feature_cols_t2].values

X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42, stratify=y2)

t0 = time.time()
model_t2 = LGBMClassifier(
    n_estimators=500, max_depth=10, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.6, random_state=42,
    verbose=-1, num_leaves=63, class_weight='balanced'
)
model_t2.fit(X2_train, y2_train)
y2_pred = model_t2.predict(X2_test)
y2_proba = model_t2.predict_proba(X2_test)

acc2 = accuracy_score(y2_test, y2_pred)
f1w2 = f1_score(y2_test, y2_pred, average='weighted', zero_division=0)
try:
    auc2 = roc_auc_score(y2_test, y2_proba, multi_class='ovr', average='weighted')
except:
    auc2 = 0.0

print(f"  LightGBM ({time.time()-t0:.1f}s)")
print(f"    Accuracy:    {acc2:.4f}")
print(f"    F1 (weight): {f1w2:.4f}")
print(f"    AUC (OVR):   {auc2:.4f}")


print("\n" + "=" * 70)
print(" STEP 6: TASK 3 - Length of Stay Prediction (Enhanced)")
print("=" * 70)

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df['LOG_LOS'] = np.log1p(df['LOS_DAYS'])
X3 = df[feature_cols].values
y3 = df['LOG_LOS'].values
y3_raw = df['LOS_DAYS'].values

X3_train, X3_test, y3_train, y3_test, _, y3_test_raw = train_test_split(
    X3, y3, y3_raw, test_size=0.2, random_state=42
)

t0 = time.time()
model_t3 = XGBRegressor(
    n_estimators=1000, max_depth=7, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.6, random_state=42,
    early_stopping_rounds=50, objective='reg:squarederror'
)
model_t3.fit(X3_train, y3_train, eval_set=[(X3_test, y3_test)], verbose=False)
y3_pred = np.expm1(model_t3.predict(X3_test))
y3_pred = np.maximum(y3_pred, 0)

mae3 = mean_absolute_error(y3_test_raw, y3_pred)
rmse3 = np.sqrt(mean_squared_error(y3_test_raw, y3_pred))
r2_3 = r2_score(y3_test_raw, y3_pred)

print(f"  XGBoost Regressor ({time.time()-t0:.1f}s)")
print(f"    MAE:  {mae3:.3f} days")
print(f"    RMSE: {rmse3:.3f} days")
print(f"    R2:   {r2_3:.4f}")


# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "=" * 70)
print(" FINAL RESULTS SUMMARY")
print("=" * 70)
print(f"""
+----------------------------------------------------------------------+
| TASK 1: MORTALITY PREDICTION (Enhanced with Clinical Features)       |
+----------------------------------------------------------------------+""")
for _, row in results_df.sort_values('AUROC', ascending=False).iterrows():
    star = " <<<" if row['Model'] == best_name else ""
    print(f"|  {row['Model']:<25s} AUROC={row['AUROC']:.4f}  AUPRC={row['AUPRC']:.4f}  F1={row['F1']:.4f}{star}")
print(f"|  5-Fold CV Mean AUROC: {cv_df['AUROC'].mean():.4f} (+/- {cv_df['AUROC'].std():.4f})")
print(f"""|
+----------------------------------------------------------------------+
| TASK 2: ICD-9 GROUP PREDICTION (Enhanced)                            |
+----------------------------------------------------------------------+
|  LightGBM                Acc={acc2:.4f}  F1w={f1w2:.4f}  AUC={auc2:.4f}
|
+----------------------------------------------------------------------+
| TASK 3: LENGTH OF STAY PREDICTION (Enhanced)                         |
+----------------------------------------------------------------------+
|  XGBoost                 MAE={mae3:.3f}d  RMSE={rmse3:.3f}d  R2={r2_3:.4f}
+----------------------------------------------------------------------+

Total runtime: {time.time()-t_start:.0f}s
""")
print("Done!")
