import pandas as pd
import numpy as np
import os

BASE = os.path.join(os.path.dirname(__file__), 'mimic-iii-clinical-database-1.4')

print("=" * 60)
print("MIMIC-III 데이터 로딩 시작")
print("=" * 60)

admissions = pd.read_csv(os.path.join(BASE, 'ADMISSIONS.csv.gz'))
patients = pd.read_csv(os.path.join(BASE, 'PATIENTS.csv.gz'))
icustays = pd.read_csv(os.path.join(BASE, 'ICUSTAYS.csv.gz'))
diagnoses = pd.read_csv(os.path.join(BASE, 'DIAGNOSES_ICD.csv.gz'))
d_icd = pd.read_csv(os.path.join(BASE, 'D_ICD_DIAGNOSES.csv.gz'))

print(f"ADMISSIONS: {admissions.shape}")
print(f"PATIENTS: {patients.shape}")
print(f"ICUSTAYS: {icustays.shape}")
print(f"DIAGNOSES_ICD: {diagnoses.shape}")
print(f"D_ICD_DIAGNOSES: {d_icd.shape}")

admissions['ADMITTIME'] = pd.to_datetime(admissions['ADMITTIME'])
admissions['DISCHTIME'] = pd.to_datetime(admissions['DISCHTIME'])
admissions['DEATHTIME'] = pd.to_datetime(admissions['DEATHTIME'])
patients['DOB'] = pd.to_datetime(patients['DOB'])
patients['DOD'] = pd.to_datetime(patients['DOD'])
patients['DOD_HOSP'] = pd.to_datetime(patients['DOD_HOSP'])
icustays['INTIME'] = pd.to_datetime(icustays['INTIME'])
icustays['OUTTIME'] = pd.to_datetime(icustays['OUTTIME'])

# ============================================================
# Task 1: Mortality Prediction
# ============================================================
print("\n" + "=" * 60)
print("TASK 1: MORTALITY PREDICTION")
print("=" * 60)

print(f"\n[HOSPITAL_EXPIRE_FLAG 분포]")
mort = admissions['HOSPITAL_EXPIRE_FLAG'].value_counts()
print(f"  생존 (0): {mort[0]:,} ({mort[0]/len(admissions)*100:.1f}%)")
print(f"  사망 (1): {mort[1]:,} ({mort[1]/len(admissions)*100:.1f}%)")
print(f"  불균형 비율: 1:{mort[0]//mort[1]}")

merged = admissions.merge(patients[['SUBJECT_ID', 'GENDER', 'DOB', 'DOD']], on='SUBJECT_ID')
merged['AGE'] = (merged['ADMITTIME'] - merged['DOB']).dt.days / 365.25
merged.loc[merged['AGE'] > 200, 'AGE'] = 91.4  # MIMIC convention: 300+ -> ~91.4

print(f"\n[사망 vs 생존 - 나이 비교]")
for flag, label in [(0, '생존'), (1, '사망')]:
    sub = merged[merged['HOSPITAL_EXPIRE_FLAG'] == flag]['AGE']
    print(f"  {label}: mean={sub.mean():.1f}, median={sub.median():.1f}, std={sub.std():.1f}")

print(f"\n[사망 vs 생존 - 성별 비교]")
ct = pd.crosstab(merged['GENDER'], merged['HOSPITAL_EXPIRE_FLAG'], normalize='columns')
print(ct.round(3))

print(f"\n[사망 vs 생존 - 입원유형별]")
ct2 = pd.crosstab(admissions['ADMISSION_TYPE'], admissions['HOSPITAL_EXPIRE_FLAG'])
ct2['mortality_rate'] = ct2[1] / (ct2[0] + ct2[1]) * 100
print(ct2.sort_values('mortality_rate', ascending=False).to_string())

print(f"\n[Short-term mortality (48h 이내 사망)]")
admissions['TIME_TO_DEATH'] = (admissions['DEATHTIME'] - admissions['ADMITTIME']).dt.total_seconds() / 3600
dead = admissions[admissions['HOSPITAL_EXPIRE_FLAG'] == 1]
print(f"  48h 이내 사망: {(dead['TIME_TO_DEATH'] <= 48).sum():,} ({(dead['TIME_TO_DEATH'] <= 48).sum()/len(dead)*100:.1f}%)")
print(f"  72h 이내 사망: {(dead['TIME_TO_DEATH'] <= 72).sum():,} ({(dead['TIME_TO_DEATH'] <= 72).sum()/len(dead)*100:.1f}%)")
print(f"  7일 이내 사망: {(dead['TIME_TO_DEATH'] <= 168).sum():,} ({(dead['TIME_TO_DEATH'] <= 168).sum()/len(dead)*100:.1f}%)")

print(f"\n[Long-term mortality (퇴원 후)]")
survived_discharged = merged[(merged['HOSPITAL_EXPIRE_FLAG'] == 0) & (merged['DOD'].notna())]
survived_discharged = survived_discharged.copy()
survived_discharged['DAYS_TO_DEATH'] = (survived_discharged['DOD'] - survived_discharged['DISCHTIME']).dt.days
print(f"  퇴원 후 사망 기록 있는 환자 입원 건수: {len(survived_discharged):,}")
print(f"  30일 이내 사망: {(survived_discharged['DAYS_TO_DEATH'] <= 30).sum():,}")
print(f"  1년 이내 사망: {(survived_discharged['DAYS_TO_DEATH'] <= 365).sum():,}")

# ============================================================
# Task 2: ICD-9 Code Group Prediction
# ============================================================
print("\n" + "=" * 60)
print("TASK 2: ICD-9 CODE GROUP PREDICTION")
print("=" * 60)

def icd9_to_group(code):
    code = str(code).strip()
    if code.startswith('E'):
        return 'E: External Causes'
    if code.startswith('V'):
        return 'V: Supplementary'
    try:
        num = int(code[:3])
    except ValueError:
        return 'Unknown'
    if num <= 139: return '001-139: Infectious'
    if num <= 239: return '140-239: Neoplasms'
    if num <= 279: return '240-279: Endocrine'
    if num <= 289: return '280-289: Blood'
    if num <= 319: return '290-319: Mental'
    if num <= 389: return '320-389: Nervous'
    if num <= 459: return '390-459: Circulatory'
    if num <= 519: return '460-519: Respiratory'
    if num <= 579: return '520-579: Digestive'
    if num <= 629: return '580-629: Genitourinary'
    if num <= 679: return '630-679: Pregnancy'
    if num <= 709: return '680-709: Skin'
    if num <= 739: return '710-739: Musculoskeletal'
    if num <= 759: return '740-759: Congenital'
    if num <= 779: return '760-779: Perinatal'
    if num <= 799: return '780-799: Symptoms'
    if num <= 999: return '800-999: Injury'
    return 'Unknown'

diagnoses['ICD9_GROUP'] = diagnoses['ICD9_CODE'].apply(icd9_to_group)

primary = diagnoses[diagnoses['SEQ_NUM'] == 1].copy()
print(f"\n[전체 진단 레코드]: {len(diagnoses):,}")
print(f"[주 진단(SEQ_NUM=1)]: {len(primary):,}")
print(f"[고유 ICD-9 코드 수]: {diagnoses['ICD9_CODE'].nunique():,}")
print(f"[ICD-9 그룹 수]: {diagnoses['ICD9_GROUP'].nunique()}")
print(f"[입원당 평균 진단 수]: {diagnoses.groupby('HADM_ID').size().mean():.1f}")

print(f"\n[주 진단 기준 - ICD-9 그룹 분포]")
grp = primary['ICD9_GROUP'].value_counts()
for g, c in grp.items():
    print(f"  {g}: {c:,} ({c/len(primary)*100:.1f}%)")

print(f"\n[그룹별 사망률]")
primary_with_mort = primary.merge(admissions[['HADM_ID', 'HOSPITAL_EXPIRE_FLAG']], on='HADM_ID')
grp_mort = primary_with_mort.groupby('ICD9_GROUP')['HOSPITAL_EXPIRE_FLAG'].agg(['mean', 'count'])
grp_mort.columns = ['mortality_rate', 'count']
grp_mort['mortality_rate'] = (grp_mort['mortality_rate'] * 100).round(1)
print(grp_mort.sort_values('mortality_rate', ascending=False).to_string())

# ============================================================
# Task 3: Length of Stay Prediction
# ============================================================
print("\n" + "=" * 60)
print("TASK 3: LENGTH OF STAY PREDICTION")
print("=" * 60)

admissions['LOS_DAYS'] = (admissions['DISCHTIME'] - admissions['ADMITTIME']).dt.total_seconds() / 86400

print(f"\n[병원 입원기간 (LOS) 통계]")
los = admissions['LOS_DAYS'].describe()
print(los.to_string())

print(f"\n[ICU 체류기간 통계]")
print(icustays['LOS'].describe().to_string())

print(f"\n[LOS 구간별 분포]")
bins = [0, 1, 3, 7, 14, 30, 9999]
labels = ['<1d', '1-3d', '3-7d', '7-14d', '14-30d', '30d+']
admissions['LOS_BIN'] = pd.cut(admissions['LOS_DAYS'], bins=bins, labels=labels, right=False)
los_dist = admissions['LOS_BIN'].value_counts().sort_index()
for b, c in los_dist.items():
    print(f"  {b}: {c:,} ({c/len(admissions)*100:.1f}%)")

print(f"\n[입원유형별 LOS]")
for at in admissions['ADMISSION_TYPE'].unique():
    sub = admissions[admissions['ADMISSION_TYPE'] == at]['LOS_DAYS']
    print(f"  {at}: mean={sub.mean():.1f}, median={sub.median():.1f}")

print(f"\n[ICU 유형별 LOS]")
for cu in icustays['FIRST_CAREUNIT'].unique():
    sub = icustays[icustays['FIRST_CAREUNIT'] == cu]['LOS']
    print(f"  {cu}: mean={sub.mean():.1f}, median={sub.median():.1f}, n={len(sub):,}")

print(f"\n[사망 여부별 LOS]")
for flag, label in [(0, '생존'), (1, '사망')]:
    sub = admissions[admissions['HOSPITAL_EXPIRE_FLAG'] == flag]['LOS_DAYS']
    print(f"  {label}: mean={sub.mean():.1f}, median={sub.median():.1f}")

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("요약: 3개 태스크 데이터 준비 상태")
print("=" * 60)
print(f"""
Task 1 - Mortality Prediction:
  - 타겟: HOSPITAL_EXPIRE_FLAG (binary)
  - 양성 비율: {mort[1]/len(admissions)*100:.1f}% (불균형)
  - 필요 테이블: ADMISSIONS, PATIENTS, ICUSTAYS, CHARTEVENTS, LABEVENTS
  - 서브태스크: In-hospital / Short-term / Long-term

Task 2 - ICD-9 Code Group Prediction:
  - 타겟: ICD-9 그룹 ({diagnoses['ICD9_GROUP'].nunique()}개 클래스, multi-class)
  - 주 진단 기준 최빈 그룹: {grp.index[0]} ({grp.iloc[0]/len(primary)*100:.1f}%)
  - 필요 테이블: DIAGNOSES_ICD, D_ICD_DIAGNOSES, ADMISSIONS, NOTEEVENTS

Task 3 - Length of Stay Prediction:
  - 타겟: LOS_DAYS (regression) 또는 LOS_BIN (classification)
  - 평균: {admissions['LOS_DAYS'].mean():.1f}일, 중앙값: {admissions['LOS_DAYS'].median():.1f}일
  - 필요 테이블: ADMISSIONS, ICUSTAYS, DIAGNOSES_ICD, CHARTEVENTS
""")
print("데이터 분석 완료!")
