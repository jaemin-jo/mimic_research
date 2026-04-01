"""
MIMIC-III 산학연 프로젝트 - 3가지 예측 태스크 종합 파이프라인
Task 1: Mortality Prediction (사망 예측)
Task 2: ICD-9 Code Group Prediction (진단 그룹 예측)
Task 3: Length of Stay Prediction (입원기간 예측)
"""

import os
import warnings
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, classification_report,
    accuracy_score, mean_absolute_error, mean_squared_error, r2_score,
    confusion_matrix
)
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

warnings.filterwarnings('ignore')
np.random.seed(42)

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mimic-iii-clinical-database-1.4')

# ============================================================
# 1. 데이터 로딩
# ============================================================
print("=" * 70)
print(" MIMIC-III 데이터 로딩")
print("=" * 70)

t0 = time.time()
admissions = pd.read_csv(os.path.join(BASE, 'ADMISSIONS.csv.gz'))
patients = pd.read_csv(os.path.join(BASE, 'PATIENTS.csv.gz'))
icustays = pd.read_csv(os.path.join(BASE, 'ICUSTAYS.csv.gz'))
diagnoses = pd.read_csv(os.path.join(BASE, 'DIAGNOSES_ICD.csv.gz'))
d_icd = pd.read_csv(os.path.join(BASE, 'D_ICD_DIAGNOSES.csv.gz'))
services = pd.read_csv(os.path.join(BASE, 'SERVICES.csv.gz'))
print(f"  로딩 완료: {time.time()-t0:.1f}s")

for df, cols in [
    (admissions, ['ADMITTIME', 'DISCHTIME', 'DEATHTIME', 'EDREGTIME', 'EDOUTTIME']),
    (patients, ['DOB', 'DOD', 'DOD_HOSP', 'DOD_SSN']),
    (icustays, ['INTIME', 'OUTTIME']),
]:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors='coerce')


# ============================================================
# 2. 공통 피처 엔지니어링
# ============================================================
print("\n" + "=" * 70)
print(" 공통 피처 엔지니어링")
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


ICD9_GROUP_NAMES = {
    0: '001-139: Infectious', 1: '140-239: Neoplasms', 2: '240-279: Endocrine',
    3: '280-289: Blood', 4: '290-319: Mental', 5: '320-389: Nervous',
    6: '390-459: Circulatory', 7: '460-519: Respiratory', 8: '520-579: Digestive',
    9: '580-629: Genitourinary', 10: '630-679: Pregnancy', 11: '680-709: Skin',
    12: '710-739: Musculoskeletal', 13: '740-759: Congenital', 14: '760-779: Perinatal',
    15: '780-799: Symptoms', 16: 'E: External Causes', 17: 'V: Supplementary',
    18: 'Unknown'
}

# 진단 코드별 그룹 매핑
diagnoses['ICD9_GROUP'] = diagnoses['ICD9_CODE'].apply(icd9_to_group)

# 입원당 진단 수 / 주 진단 그룹
diag_count = diagnoses.groupby('HADM_ID').size().reset_index(name='NUM_DIAGNOSES')
primary_diag = diagnoses[diagnoses['SEQ_NUM'] == 1][['HADM_ID', 'ICD9_CODE', 'ICD9_GROUP']].rename(
    columns={'ICD9_CODE': 'PRIMARY_ICD9', 'ICD9_GROUP': 'PRIMARY_ICD9_GROUP'}
)

# 입원당 ICD9 그룹 원핫 (상위 그룹)
diag_groups_pivot = diagnoses.groupby(['HADM_ID', 'ICD9_GROUP']).size().unstack(fill_value=0)
diag_groups_pivot.columns = [f'DIAG_GRP_{c}' for c in diag_groups_pivot.columns]
diag_groups_pivot = diag_groups_pivot.reset_index()

# ICU 정보 (첫 ICU 체류)
first_icu = icustays.sort_values('INTIME').groupby('HADM_ID').first().reset_index()
first_icu = first_icu[['HADM_ID', 'ICUSTAY_ID', 'FIRST_CAREUNIT', 'LOS']].rename(
    columns={'LOS': 'ICU_LOS', 'FIRST_CAREUNIT': 'ICU_TYPE'}
)
icu_count = icustays.groupby('HADM_ID').size().reset_index(name='NUM_ICU_STAYS')

# 서비스 수
svc_count = services.groupby('HADM_ID').size().reset_index(name='NUM_SERVICES')

# 마스터 데이터셋 구축
df = admissions.merge(patients[['SUBJECT_ID', 'GENDER', 'DOB', 'DOD', 'EXPIRE_FLAG']], on='SUBJECT_ID', how='left')
df = df.merge(first_icu, on='HADM_ID', how='left')
df = df.merge(icu_count, on='HADM_ID', how='left')
df = df.merge(diag_count, on='HADM_ID', how='left')
df = df.merge(primary_diag, on='HADM_ID', how='left')
df = df.merge(diag_groups_pivot, on='HADM_ID', how='left')
df = df.merge(svc_count, on='HADM_ID', how='left')

# 나이 계산 (MIMIC 규칙: 89세 이상은 300으로 시프트됨 -> 91.4로 보정)
df['AGE'] = (df['ADMITTIME'] - df['DOB']).dt.days / 365.25
df.loc[df['AGE'] > 200, 'AGE'] = 91.4

# 신생아 제외 (나이 < 1세) - 성인 환자만 분석
df = df[df['AGE'] >= 18].copy()

# 입원기간 (타겟 for Task 3)
df['LOS_DAYS'] = (df['DISCHTIME'] - df['ADMITTIME']).dt.total_seconds() / 86400
df = df[df['LOS_DAYS'] > 0].copy()

# ED 체류시간
df['ED_DURATION'] = (df['EDOUTTIME'] - df['EDREGTIME']).dt.total_seconds() / 3600
df['HAS_ED'] = df['EDREGTIME'].notna().astype(int)

# 시간 피처
df['ADMIT_HOUR'] = df['ADMITTIME'].dt.hour
df['ADMIT_DOW'] = df['ADMITTIME'].dt.dayofweek
df['ADMIT_MONTH'] = df['ADMITTIME'].dt.month
df['IS_WEEKEND'] = (df['ADMIT_DOW'] >= 5).astype(int)
df['IS_NIGHT'] = ((df['ADMIT_HOUR'] >= 22) | (df['ADMIT_HOUR'] <= 6)).astype(int)

# 범주형 인코딩
cat_cols = ['ADMISSION_TYPE', 'INSURANCE', 'ETHNICITY', 'MARITAL_STATUS', 'ICU_TYPE', 'GENDER']
label_encoders = {}
for col in cat_cols:
    df[col] = df[col].fillna('UNKNOWN')
    le = LabelEncoder()
    df[col + '_ENC'] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# 수치형 결측치 처리
num_fill_cols = ['ICU_LOS', 'NUM_ICU_STAYS', 'NUM_DIAGNOSES', 'NUM_SERVICES', 'ED_DURATION']
for c in num_fill_cols:
    df[c] = df[c].fillna(0)

# 진단 그룹 원핫 결측치
diag_grp_cols = [c for c in df.columns if c.startswith('DIAG_GRP_')]
df[diag_grp_cols] = df[diag_grp_cols].fillna(0)

# 최종 피처 목록
FEATURE_COLS = [
    'AGE', 'GENDER_ENC', 'ADMISSION_TYPE_ENC', 'INSURANCE_ENC',
    'ETHNICITY_ENC', 'MARITAL_STATUS_ENC', 'ICU_TYPE_ENC',
    'ICU_LOS', 'NUM_ICU_STAYS', 'NUM_DIAGNOSES', 'NUM_SERVICES',
    'HAS_ED', 'ED_DURATION', 'ADMIT_HOUR', 'ADMIT_DOW', 'ADMIT_MONTH',
    'IS_WEEKEND', 'IS_NIGHT',
] + diag_grp_cols

# Task2 피처: 진단 그룹 자체가 타겟이므로 진단 정보 제외
FEATURE_COLS_TASK2 = [
    'AGE', 'GENDER_ENC', 'ADMISSION_TYPE_ENC', 'INSURANCE_ENC',
    'ETHNICITY_ENC', 'MARITAL_STATUS_ENC', 'ICU_TYPE_ENC',
    'ICU_LOS', 'NUM_ICU_STAYS', 'NUM_SERVICES',
    'HAS_ED', 'ED_DURATION', 'ADMIT_HOUR', 'ADMIT_DOW', 'ADMIT_MONTH',
    'IS_WEEKEND', 'IS_NIGHT',
]

print(f"  전처리 완료: 총 {len(df):,}건 (성인 환자, LOS > 0)")
print(f"  피처 수: {len(FEATURE_COLS)}개")


# ============================================================
# 유틸리티 함수
# ============================================================
def print_divider(title):
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def train_and_evaluate_classifier(X_train, X_test, y_train, y_test, models, task_name, multiclass=False):
    results = []
    for name, model in models.items():
        t0 = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - t0
        y_pred = model.predict(X_test)

        if multiclass:
            y_proba = model.predict_proba(X_test)
            acc = accuracy_score(y_test, y_pred)
            f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
            f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            try:
                auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
            except Exception:
                auc = 0.0
            results.append({
                'Model': name, 'Accuracy': acc, 'F1_macro': f1_macro,
                'F1_weighted': f1_weighted, 'AUC_weighted': auc, 'Time(s)': elapsed
            })
            print(f"\n  [{name}] ({elapsed:.1f}s)")
            print(f"    Accuracy:    {acc:.4f}")
            print(f"    F1 (macro):  {f1_macro:.4f}")
            print(f"    F1 (weight): {f1_weighted:.4f}")
            print(f"    AUC (OVR):   {auc:.4f}")
        else:
            y_proba = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_proba)
            auprc = average_precision_score(y_test, y_proba)
            f1 = f1_score(y_test, y_pred)
            acc = accuracy_score(y_test, y_pred)
            results.append({
                'Model': name, 'AUROC': auc, 'AUPRC': auprc,
                'F1': f1, 'Accuracy': acc, 'Time(s)': elapsed
            })
            print(f"\n  [{name}] ({elapsed:.1f}s)")
            print(f"    AUROC:    {auc:.4f}")
            print(f"    AUPRC:    {auprc:.4f}")
            print(f"    F1:       {f1:.4f}")
            print(f"    Accuracy: {acc:.4f}")

    return pd.DataFrame(results), models


def train_and_evaluate_regressor(X_train, X_test, y_train, y_test, models, task_name):
    results = []
    for name, model in models.items():
        t0 = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - t0
        y_pred = model.predict(X_test)
        y_pred = np.maximum(y_pred, 0)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        results.append({
            'Model': name, 'MAE': mae, 'RMSE': rmse, 'R2': r2, 'Time(s)': elapsed
        })
        print(f"\n  [{name}] ({elapsed:.1f}s)")
        print(f"    MAE:  {mae:.3f} days")
        print(f"    RMSE: {rmse:.3f} days")
        print(f"    R2:   {r2:.4f}")

    return pd.DataFrame(results), models


def print_feature_importance(model, feature_names, top_n=15):
    if hasattr(model, 'feature_importances_'):
        imp = pd.Series(model.feature_importances_, index=feature_names)
        imp = imp.sort_values(ascending=False).head(top_n)
        print(f"\n  [Top {top_n} Feature Importance]")
        for feat, val in imp.items():
            bar = '#' * int(val / imp.max() * 30)
            print(f"    {feat:<30s} {val:.4f} {bar}")


# ============================================================
# TASK 1: MORTALITY PREDICTION (In-Hospital)
# ============================================================
print_divider("TASK 1: IN-HOSPITAL MORTALITY PREDICTION")

X1 = df[FEATURE_COLS].values
y1 = df['HOSPITAL_EXPIRE_FLAG'].values

X1_train, X1_test, y1_train, y1_test = train_test_split(
    X1, y1, test_size=0.2, random_state=42, stratify=y1
)

pos_ratio = y1_train.sum() / len(y1_train)
scale_pos_weight = (1 - pos_ratio) / pos_ratio

print(f"  Train: {len(X1_train):,} (pos={y1_train.sum():,}, {y1_train.mean()*100:.1f}%)")
print(f"  Test:  {len(X1_test):,} (pos={y1_test.sum():,}, {y1_test.mean()*100:.1f}%)")

models_t1 = {
    'Logistic Regression': LogisticRegression(
        max_iter=1000, class_weight='balanced', C=0.1, random_state=42
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=300, max_depth=12, min_samples_leaf=20,
        class_weight='balanced', random_state=42, n_jobs=-1
    ),
    'XGBoost': XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        scale_pos_weight=scale_pos_weight, subsample=0.8,
        colsample_bytree=0.8, random_state=42,
        eval_metric='aucpr', use_label_encoder=False
    ),
    'LightGBM': LGBMClassifier(
        n_estimators=500, max_depth=8, learning_rate=0.05,
        scale_pos_weight=scale_pos_weight, subsample=0.8,
        colsample_bytree=0.8, random_state=42, verbose=-1,
        num_leaves=63
    ),
}

scaler1 = StandardScaler()
X1_train_s = scaler1.fit_transform(X1_train)
X1_test_s = scaler1.transform(X1_test)

models_t1_run = {
    'Logistic Regression': models_t1['Logistic Regression'],
}
results1_lr, _ = train_and_evaluate_classifier(
    X1_train_s, X1_test_s, y1_train, y1_test, models_t1_run, 'Task1'
)

models_t1_tree = {k: v for k, v in models_t1.items() if k != 'Logistic Regression'}
results1_tree, trained_t1 = train_and_evaluate_classifier(
    X1_train, X1_test, y1_train, y1_test, models_t1_tree, 'Task1'
)

results1 = pd.concat([results1_lr, results1_tree], ignore_index=True)

best_t1_name = results1.sort_values('AUROC', ascending=False).iloc[0]['Model']
best_t1_model = trained_t1.get(best_t1_name, models_t1['Logistic Regression'])
print(f"\n  >>> Best Model: {best_t1_name} (AUROC={results1.sort_values('AUROC', ascending=False).iloc[0]['AUROC']:.4f})")

print_feature_importance(best_t1_model, FEATURE_COLS)

print(f"\n  [Best Model - Classification Report]")
if best_t1_name == 'Logistic Regression':
    y1_pred_best = best_t1_model.predict(X1_test_s)
else:
    y1_pred_best = best_t1_model.predict(X1_test)
print(classification_report(y1_test, y1_pred_best, target_names=['Survived', 'Died']))


# ============================================================
# TASK 2: ICD-9 CODE GROUP PREDICTION
# ============================================================
print_divider("TASK 2: ICD-9 CODE GROUP PREDICTION")

df_t2 = df.dropna(subset=['PRIMARY_ICD9_GROUP']).copy()
df_t2['PRIMARY_ICD9_GROUP'] = df_t2['PRIMARY_ICD9_GROUP'].astype(int)

# 소수 클래스 제거 (30건 미만)
grp_counts = df_t2['PRIMARY_ICD9_GROUP'].value_counts()
valid_groups = grp_counts[grp_counts >= 30].index.tolist()
df_t2 = df_t2[df_t2['PRIMARY_ICD9_GROUP'].isin(valid_groups)].copy()

le_t2 = LabelEncoder()
y2_encoded = le_t2.fit_transform(df_t2['PRIMARY_ICD9_GROUP'].values)
n_classes = len(le_t2.classes_)

X2 = df_t2[FEATURE_COLS_TASK2].values
y2 = y2_encoded

X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2, y2, test_size=0.2, random_state=42, stratify=y2
)

print(f"  클래스 수: {n_classes}")
print(f"  Train: {len(X2_train):,}, Test: {len(X2_test):,}")
print(f"  클래스 분포:")
for cls_enc in range(n_classes):
    orig = le_t2.inverse_transform([cls_enc])[0]
    cnt = (y2 == cls_enc).sum()
    name = ICD9_GROUP_NAMES.get(orig, f'Group {orig}')
    print(f"    {name}: {cnt:,} ({cnt/len(y2)*100:.1f}%)")

models_t2 = {
    'Random Forest': RandomForestClassifier(
        n_estimators=300, max_depth=15, min_samples_leaf=10,
        class_weight='balanced', random_state=42, n_jobs=-1
    ),
    'XGBoost': XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        use_label_encoder=False, eval_metric='mlogloss',
        objective='multi:softprob', num_class=n_classes
    ),
    'LightGBM': LGBMClassifier(
        n_estimators=500, max_depth=10, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        verbose=-1, num_leaves=63, class_weight='balanced',
        objective='multiclass', num_class=n_classes
    ),
}

results2, trained_t2 = train_and_evaluate_classifier(
    X2_train, X2_test, y2_train, y2_test, models_t2, 'Task2', multiclass=True
)

best_t2_name = results2.sort_values('F1_weighted', ascending=False).iloc[0]['Model']
print(f"\n  >>> Best Model: {best_t2_name} (F1_weighted={results2.sort_values('F1_weighted', ascending=False).iloc[0]['F1_weighted']:.4f})")

best_t2_model = trained_t2[best_t2_name]
print_feature_importance(best_t2_model, FEATURE_COLS_TASK2)

y2_pred_best = best_t2_model.predict(X2_test)
group_names_ordered = [ICD9_GROUP_NAMES.get(le_t2.inverse_transform([i])[0], f'G{i}') for i in range(n_classes)]
print(f"\n  [Best Model - Classification Report (Top classes)]")
print(classification_report(y2_test, y2_pred_best, target_names=group_names_ordered, zero_division=0))


# ============================================================
# TASK 3: LENGTH OF STAY PREDICTION
# ============================================================
print_divider("TASK 3: LENGTH OF STAY PREDICTION")

# 로그 변환 (right-skewed 분포 보정)
df['LOG_LOS'] = np.log1p(df['LOS_DAYS'])

X3 = df[FEATURE_COLS].values
y3 = df['LOG_LOS'].values
y3_raw = df['LOS_DAYS'].values

X3_train, X3_test, y3_train, y3_test, _, y3_test_raw = train_test_split(
    X3, y3, y3_raw, test_size=0.2, random_state=42
)

print(f"  Train: {len(X3_train):,}, Test: {len(X3_test):,}")
print(f"  LOS 통계 - Mean: {y3_test_raw.mean():.1f}d, Median: {np.median(y3_test_raw):.1f}d")

models_t3 = {
    'Ridge Regression': Ridge(alpha=1.0),
    'Random Forest': RandomForestRegressor(
        n_estimators=300, max_depth=15, min_samples_leaf=10,
        random_state=42, n_jobs=-1
    ),
    'XGBoost': XGBRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        objective='reg:squarederror'
    ),
    'LightGBM': LGBMRegressor(
        n_estimators=500, max_depth=10, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        verbose=-1, num_leaves=63
    ),
}

# 스케일링 (Ridge용)
scaler3 = StandardScaler()
X3_train_s = scaler3.fit_transform(X3_train)
X3_test_s = scaler3.transform(X3_test)


class LogTransformWrapper:
    """로그 스케일 예측 -> 원래 스케일로 변환하여 평가"""
    def __init__(self, model):
        self.model = model
    def fit(self, X, y):
        self.model.fit(X, y)
        return self
    def predict(self, X):
        return np.expm1(self.model.predict(X))


print("\n  -- Ridge Regression (scaled) --")
ridge_model = models_t3['Ridge Regression']
ridge_model.fit(X3_train_s, y3_train)
y3_pred_ridge = np.expm1(ridge_model.predict(X3_test_s))
y3_pred_ridge = np.maximum(y3_pred_ridge, 0)
mae_r = mean_absolute_error(y3_test_raw, y3_pred_ridge)
rmse_r = np.sqrt(mean_squared_error(y3_test_raw, y3_pred_ridge))
r2_r = r2_score(y3_test_raw, y3_pred_ridge)
print(f"    MAE:  {mae_r:.3f} days")
print(f"    RMSE: {rmse_r:.3f} days")
print(f"    R2:   {r2_r:.4f}")

tree_results = []
for name in ['Random Forest', 'XGBoost', 'LightGBM']:
    model = models_t3[name]
    t0 = time.time()
    model.fit(X3_train, y3_train)
    elapsed = time.time() - t0
    y3_pred = np.expm1(model.predict(X3_test))
    y3_pred = np.maximum(y3_pred, 0)
    mae = mean_absolute_error(y3_test_raw, y3_pred)
    rmse = np.sqrt(mean_squared_error(y3_test_raw, y3_pred))
    r2 = r2_score(y3_test_raw, y3_pred)
    tree_results.append({'Model': name, 'MAE': mae, 'RMSE': rmse, 'R2': r2, 'Time(s)': elapsed})
    print(f"\n  [{name}] ({elapsed:.1f}s)")
    print(f"    MAE:  {mae:.3f} days")
    print(f"    RMSE: {rmse:.3f} days")
    print(f"    R2:   {r2:.4f}")

results3 = pd.DataFrame([
    {'Model': 'Ridge Regression', 'MAE': mae_r, 'RMSE': rmse_r, 'R2': r2_r, 'Time(s)': 0}
] + tree_results)

best_t3_name = results3.sort_values('MAE').iloc[0]['Model']
best_t3_mae = results3.sort_values('MAE').iloc[0]['MAE']
print(f"\n  >>> Best Model: {best_t3_name} (MAE={best_t3_mae:.3f} days)")

best_t3_model = models_t3[best_t3_name]
print_feature_importance(best_t3_model, FEATURE_COLS)

# LOS 구간별 예측 성능
print(f"\n  [LOS 구간별 예측 오차 (Best Model)]")
y3_pred_best = np.expm1(best_t3_model.predict(X3_test))
y3_pred_best = np.maximum(y3_pred_best, 0)
bins_eval = [(0, 3), (3, 7), (7, 14), (14, 30), (30, 9999)]
for lo, hi in bins_eval:
    mask = (y3_test_raw >= lo) & (y3_test_raw < hi)
    if mask.sum() > 0:
        mae_bin = mean_absolute_error(y3_test_raw[mask], y3_pred_best[mask])
        label = f"{lo}-{hi}d" if hi < 9999 else f"{lo}d+"
        print(f"    {label}: MAE={mae_bin:.2f}d (n={mask.sum():,})")


# ============================================================
# LOS Classification (구간 분류 접근)
# ============================================================
print_divider("TASK 3b: LENGTH OF STAY - CLASSIFICATION (구간 분류)")

los_bins = [0, 3, 7, 14, 30, 9999]
los_labels = ['Short(<3d)', 'Medium(3-7d)', 'Long(7-14d)', 'VeryLong(14-30d)', 'Extended(30d+)']
df['LOS_CLASS'] = pd.cut(df['LOS_DAYS'], bins=los_bins, labels=range(len(los_labels)), right=False).astype(int)

X3c = df[FEATURE_COLS].values
y3c = df['LOS_CLASS'].values

X3c_train, X3c_test, y3c_train, y3c_test = train_test_split(
    X3c, y3c, test_size=0.2, random_state=42, stratify=y3c
)

models_t3c = {
    'XGBoost': XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        use_label_encoder=False, eval_metric='mlogloss'
    ),
    'LightGBM': LGBMClassifier(
        n_estimators=500, max_depth=10, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        verbose=-1, num_leaves=63, class_weight='balanced'
    ),
}

print(f"  클래스: {los_labels}")
print(f"  Train: {len(X3c_train):,}, Test: {len(X3c_test):,}")

results3c, trained_t3c = train_and_evaluate_classifier(
    X3c_train, X3c_test, y3c_train, y3c_test, models_t3c, 'Task3c', multiclass=True
)

best_t3c = results3c.sort_values('F1_weighted', ascending=False).iloc[0]
print(f"\n  >>> Best: {best_t3c['Model']} (F1_weighted={best_t3c['F1_weighted']:.4f})")

y3c_pred = trained_t3c[best_t3c['Model']].predict(X3c_test)
print(classification_report(y3c_test, y3c_pred, target_names=los_labels, zero_division=0))


# ============================================================
# 최종 요약
# ============================================================
print_divider("최종 결과 요약")

print("+" + "-" * 68 + "+")
print("| TASK 1: IN-HOSPITAL MORTALITY PREDICTION                           |")
print("+" + "-" * 68 + "+")
for _, row in results1.sort_values('AUROC', ascending=False).iterrows():
    star = " <<<" if row['Model'] == best_t1_name else ""
    print(f"|  {row['Model']:<25s} AUROC={row['AUROC']:.4f}  AUPRC={row['AUPRC']:.4f}  F1={row['F1']:.4f}{star}")

print("+" + "-" * 68 + "+")
print("| TASK 2: ICD-9 CODE GROUP PREDICTION                               |")
print("+" + "-" * 68 + "+")
for _, row in results2.sort_values('F1_weighted', ascending=False).iterrows():
    star = " <<<" if row['Model'] == best_t2_name else ""
    print(f"|  {row['Model']:<25s} Acc={row['Accuracy']:.4f}  F1w={row['F1_weighted']:.4f}  AUC={row['AUC_weighted']:.4f}{star}")

print("+" + "-" * 68 + "+")
print("| TASK 3: LENGTH OF STAY PREDICTION                                 |")
print("+" + "-" * 68 + "+")
for _, row in results3.sort_values('MAE').iterrows():
    star = " <<<" if row['Model'] == best_t3_name else ""
    print(f"|  {row['Model']:<25s} MAE={row['MAE']:.3f}d  RMSE={row['RMSE']:.3f}d  R2={row['R2']:.4f}{star}")

print("+" + "-" * 68 + "+")
print("| TASK 3b: LOS CLASSIFICATION                                       |")
print("+" + "-" * 68 + "+")
for _, row in results3c.sort_values('F1_weighted', ascending=False).iterrows():
    print(f"|  {row['Model']:<25s} Acc={row['Accuracy']:.4f}  F1w={row['F1_weighted']:.4f}")
print("+" + "-" * 68 + "+")
print("\nDone!")
