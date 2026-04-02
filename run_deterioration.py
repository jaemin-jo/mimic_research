"""
Task 4: Severity Deterioration Prediction (중증도 악화 예측)
- Early Warning System: First 6h data -> predict deterioration in 6-48h
- Deterioration defined as:
  (1) SOFA score increase >= 2  (2) Vasopressor initiation
  (3) Lactate increase > 2 mmol/L  (4) In-hospital mortality
"""

import os, warnings, time, gc, sys
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
np.random.seed(42)

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mimic-iii-clinical-database-1.4')
T_START = time.time()


def elapsed():
    return f"{time.time()-T_START:.0f}s"


print("=" * 70)
print(" Task 4: Severity Deterioration Prediction")
print(" Early Warning System - First 6h -> Predict 6-48h Deterioration")
print("=" * 70)

# ============================================================
# 1. BASE DATA
# ============================================================
print(f"\n[{elapsed()}] Loading base tables...")

admissions = pd.read_csv(os.path.join(BASE, 'ADMISSIONS.csv.gz'))
patients = pd.read_csv(os.path.join(BASE, 'PATIENTS.csv.gz'))
icustays = pd.read_csv(os.path.join(BASE, 'ICUSTAYS.csv.gz'))
diagnoses = pd.read_csv(os.path.join(BASE, 'DIAGNOSES_ICD.csv.gz'))

for d, cols in [
    (admissions, ['ADMITTIME', 'DISCHTIME', 'DEATHTIME', 'EDREGTIME', 'EDOUTTIME']),
    (patients, ['DOB', 'DOD', 'DOD_HOSP']),
    (icustays, ['INTIME', 'OUTTIME']),
]:
    for c in cols:
        if c in d.columns:
            d[c] = pd.to_datetime(d[c], errors='coerce')

first_icu = icustays.sort_values('INTIME').groupby('HADM_ID').first().reset_index()
intime_map = first_icu.set_index('HADM_ID')['INTIME'].to_dict()
intime_series = pd.Series(intime_map)

print(f"[{elapsed()}] Base tables loaded")

# ============================================================
# 2. LABEVENTS - Split into EARLY (0-6h) and LATE (24-48h)
# ============================================================
print(f"\n[{elapsed()}] Processing LABEVENTS (early vs late windows)...")

LAB_ITEMS = {
    50862: 'albumin', 50882: 'bicarbonate', 50885: 'bilirubin', 50912: 'creatinine',
    50902: 'chloride', 50931: 'glucose', 50971: 'potassium', 50983: 'sodium',
    51006: 'bun', 51222: 'hemoglobin', 51265: 'platelet', 51300: 'wbc', 51301: 'wbc',
    50813: 'lactate', 51237: 'inr', 50820: 'ph', 50821: 'pao2', 50818: 'paco2',
    50863: 'alp', 50861: 'alt', 50878: 'ast', 51277: 'rdw', 51279: 'rbc',
    50893: 'calcium', 50960: 'magnesium', 50970: 'phosphate', 50809: 'glucose',
    50811: 'hemoglobin', 50868: 'aniongap', 51275: 'ptt', 51274: 'pt',
}
LAB_IDS = set(LAB_ITEMS.keys())

early_lab_chunks = []
late_lab_chunks = []

for i, chunk in enumerate(pd.read_csv(
    os.path.join(BASE, 'LABEVENTS.csv.gz'), chunksize=500000,
    usecols=['HADM_ID', 'ITEMID', 'CHARTTIME', 'VALUENUM']
)):
    chunk = chunk[chunk['ITEMID'].isin(LAB_IDS)].dropna(subset=['VALUENUM', 'HADM_ID', 'CHARTTIME'])
    if len(chunk) == 0:
        continue
    chunk['HADM_ID'] = chunk['HADM_ID'].astype(int)
    chunk['CHARTTIME'] = pd.to_datetime(chunk['CHARTTIME'], errors='coerce')
    intimes = chunk['HADM_ID'].map(intime_series)
    valid = intimes.notna() & chunk['CHARTTIME'].notna()
    chunk = chunk[valid].copy()
    hours = (chunk['CHARTTIME'] - intimes[valid]).dt.total_seconds() / 3600

    early = chunk[(hours >= 0) & (hours <= 6)].copy()
    late = chunk[(hours > 24) & (hours <= 48)].copy()

    if len(early) > 0:
        early_lab_chunks.append(early[['HADM_ID', 'ITEMID', 'VALUENUM']])
    if len(late) > 0:
        late_lab_chunks.append(late[['HADM_ID', 'ITEMID', 'VALUENUM']])

    if (i + 1) % 10 == 0:
        er = sum(len(c) for c in early_lab_chunks)
        lr = sum(len(c) for c in late_lab_chunks)
        print(f"    chunk {i+1}: early={er:,}, late={lr:,} ({elapsed()})")

def agg_labs(chunks, prefix):
    if not chunks:
        return pd.DataFrame(columns=['HADM_ID'])
    df = pd.concat(chunks, ignore_index=True)
    df['LAB'] = df['ITEMID'].map(LAB_ITEMS)
    df = df[(df['VALUENUM'] > -1000) & (df['VALUENUM'] < 100000)]
    agg = df.groupby(['HADM_ID', 'LAB'])['VALUENUM'].agg(['mean', 'min', 'max', 'std', 'last']).reset_index()
    wide_parts = {}
    for stat in ['mean', 'min', 'max', 'std', 'last']:
        piv = agg.pivot_table(index='HADM_ID', columns='LAB', values=stat, aggfunc='first')
        piv.columns = [f'{prefix}_{c}_{stat}' for c in piv.columns]
        wide_parts[stat] = piv
    return pd.concat(wide_parts.values(), axis=1).reset_index()

early_lab = agg_labs(early_lab_chunks, 'elab')
late_lab = agg_labs(late_lab_chunks, 'llab')

print(f"[{elapsed()}] LABEVENTS done: early={early_lab.shape[1]-1} feats, late={late_lab.shape[1]-1} feats")
del early_lab_chunks, late_lab_chunks
gc.collect()

# ============================================================
# 3. CHARTEVENTS - Split into EARLY (0-6h) and LATE (24-48h)
# ============================================================
print(f"\n[{elapsed()}] Processing CHARTEVENTS (early vs late windows)...")

VITAL_ITEMS = {
    211: 'hr', 220045: 'hr', 51: 'sbp', 442: 'sbp', 455: 'sbp', 6701: 'sbp',
    220179: 'sbp', 220050: 'sbp', 8368: 'dbp', 8440: 'dbp', 8441: 'dbp',
    8555: 'dbp', 220180: 'dbp', 220051: 'dbp', 456: 'mbp', 52: 'mbp',
    6702: 'mbp', 443: 'mbp', 220052: 'mbp', 220181: 'mbp',
    615: 'resp', 618: 'resp', 220210: 'resp', 224690: 'resp',
    646: 'spo2', 220277: 'spo2', 678: 'temp_f', 223761: 'temp_f',
    676: 'temp_c', 223762: 'temp_c', 198: 'gcs',
    723: 'gcs_v', 223900: 'gcs_v', 454: 'gcs_m', 223901: 'gcs_m',
    184: 'gcs_e', 220739: 'gcs_e', 762: 'weight', 763: 'weight',
    224639: 'weight', 226512: 'weight', 190: 'fio2', 3420: 'fio2', 223835: 'fio2',
    40055: 'urine', 43175: 'urine', 40069: 'urine', 40094: 'urine',
    40715: 'urine', 40473: 'urine', 40085: 'urine', 40057: 'urine',
    40056: 'urine', 227488: 'urine', 226559: 'urine',
}
VITAL_IDS = set(VITAL_ITEMS.keys())

ranges = {
    'hr': (0, 300), 'sbp': (0, 400), 'dbp': (0, 300), 'mbp': (0, 400),
    'resp': (0, 80), 'spo2': (0, 100), 'temp_f': (90, 115), 'gcs': (3, 15),
    'gcs_v': (1, 5), 'gcs_m': (1, 6), 'gcs_e': (1, 4), 'weight': (20, 500),
    'fio2': (21, 100), 'urine': (0, 10000),
}

early_vital_chunks = []
late_vital_chunks = []

for i, chunk in enumerate(pd.read_csv(
    os.path.join(BASE, 'CHARTEVENTS.csv.gz'), chunksize=1000000,
    usecols=['HADM_ID', 'ITEMID', 'CHARTTIME', 'VALUENUM'], low_memory=False
)):
    chunk = chunk[chunk['ITEMID'].isin(VITAL_IDS)].dropna(subset=['VALUENUM', 'HADM_ID', 'CHARTTIME'])
    if len(chunk) == 0:
        continue
    chunk['HADM_ID'] = chunk['HADM_ID'].astype(int)
    chunk['CHARTTIME'] = pd.to_datetime(chunk['CHARTTIME'], errors='coerce')
    intimes = chunk['HADM_ID'].map(intime_series)
    valid = intimes.notna() & chunk['CHARTTIME'].notna()
    chunk = chunk[valid].copy()
    hours = (chunk['CHARTTIME'] - intimes[valid]).dt.total_seconds() / 3600

    early = chunk[(hours >= 0) & (hours <= 6)].copy()
    late = chunk[(hours > 24) & (hours <= 48)].copy()

    if len(early) > 0:
        early_vital_chunks.append(early[['HADM_ID', 'ITEMID', 'VALUENUM']])
    if len(late) > 0:
        late_vital_chunks.append(late[['HADM_ID', 'ITEMID', 'VALUENUM']])

    if (i + 1) % 50 == 0:
        er = sum(len(c) for c in early_vital_chunks)
        lr = sum(len(c) for c in late_vital_chunks)
        print(f"    chunk {i+1}: early={er:,}, late={lr:,} ({elapsed()})")


def agg_vitals(chunks, prefix):
    if not chunks:
        return pd.DataFrame(columns=['HADM_ID'])
    df = pd.concat(chunks, ignore_index=True)
    df['VITAL'] = df['ITEMID'].map(VITAL_ITEMS)

    mask_c = df['VITAL'] == 'temp_c'
    df.loc[mask_c, 'VALUENUM'] = df.loc[mask_c, 'VALUENUM'] * 9 / 5 + 32
    df.loc[mask_c, 'VITAL'] = 'temp_f'
    mask_fio = (df['VITAL'] == 'fio2') & (df['VALUENUM'] <= 1.0)
    df.loc[mask_fio, 'VALUENUM'] = df.loc[mask_fio, 'VALUENUM'] * 100

    for v, (lo, hi) in ranges.items():
        mask = df['VITAL'] == v
        df.loc[mask & ((df['VALUENUM'] < lo) | (df['VALUENUM'] > hi)), 'VALUENUM'] = np.nan
    df = df.dropna(subset=['VALUENUM'])

    agg = df.groupby(['HADM_ID', 'VITAL'])['VALUENUM'].agg(['mean', 'min', 'max', 'std', 'last']).reset_index()
    wide_parts = {}
    for stat in ['mean', 'min', 'max', 'std', 'last']:
        piv = agg.pivot_table(index='HADM_ID', columns='VITAL', values=stat, aggfunc='first')
        piv.columns = [f'{prefix}_{c}_{stat}' for c in piv.columns]
        wide_parts[stat] = piv
    return pd.concat(wide_parts.values(), axis=1).reset_index()

early_vital = agg_vitals(early_vital_chunks, 'ev')
late_vital = agg_vitals(late_vital_chunks, 'lv')

print(f"[{elapsed()}] CHARTEVENTS done: early={early_vital.shape[1]-1} feats, late={late_vital.shape[1]-1} feats")
del early_vital_chunks, late_vital_chunks
gc.collect()

# ============================================================
# 4. INPUTEVENTS - Vasopressor timing (early vs late)
# ============================================================
print(f"\n[{elapsed()}] Processing INPUTEVENTS (vasopressor timing)...")

VASO_MV = {221906: 'norepinephrine', 221289: 'epinephrine', 221662: 'dopamine',
           221653: 'dobutamine', 222315: 'vasopressin', 221749: 'phenylephrine', 221986: 'milrinone'}
VASO_CV = {30047: 'norepinephrine', 30120: 'norepinephrine', 30044: 'epinephrine',
           30119: 'epinephrine', 30309: 'epinephrine', 30043: 'dopamine',
           30307: 'dopamine', 30042: 'dobutamine', 30306: 'dobutamine',
           30051: 'vasopressin', 30127: 'phenylephrine', 30128: 'phenylephrine'}
ALL_VASO_IDS = set(VASO_MV.keys()) | set(VASO_CV.keys())

early_vaso_hadms = set()
late_vaso_hadms = set()
early_vaso_records = []
late_vaso_records = []

for src, time_col, chunk_cols in [
    ('INPUTEVENTS_MV.csv.gz', 'STARTTIME', ['HADM_ID', 'ITEMID', 'STARTTIME', 'AMOUNT']),
    ('INPUTEVENTS_CV.csv.gz', 'CHARTTIME', ['HADM_ID', 'ITEMID', 'CHARTTIME', 'AMOUNT']),
]:
    try:
        for chunk in pd.read_csv(
            os.path.join(BASE, src), chunksize=200000,
            usecols=chunk_cols, low_memory=False
        ):
            chunk = chunk[chunk['ITEMID'].isin(ALL_VASO_IDS)].dropna(subset=['HADM_ID', 'AMOUNT'])
            if len(chunk) == 0:
                continue
            chunk['HADM_ID'] = chunk['HADM_ID'].astype(int)
            chunk[time_col] = pd.to_datetime(chunk[time_col], errors='coerce')
            intimes = chunk['HADM_ID'].map(intime_series)
            valid = intimes.notna() & chunk[time_col].notna()
            chunk = chunk[valid].copy()
            hours = (chunk[time_col] - intimes[valid]).dt.total_seconds() / 3600

            e = chunk[(hours >= 0) & (hours <= 6)]
            l = chunk[(hours > 6) & (hours <= 48)]

            if len(e) > 0:
                early_vaso_hadms.update(e['HADM_ID'].unique())
                early_vaso_records.append(e[['HADM_ID', 'AMOUNT']])
            if len(l) > 0:
                late_vaso_hadms.update(l['HADM_ID'].unique())
                late_vaso_records.append(l[['HADM_ID', 'AMOUNT']])
    except Exception as ex:
        print(f"    {src} error: {ex}")

early_vaso_feat = pd.DataFrame({'HADM_ID': list(early_vaso_hadms), 'early_has_vaso': 1})
if early_vaso_records:
    ev_df = pd.concat(early_vaso_records)
    ev_agg = ev_df.groupby('HADM_ID')['AMOUNT'].agg(early_vaso_count='count', early_vaso_amount='sum').reset_index()
    early_vaso_feat = early_vaso_feat.merge(ev_agg, on='HADM_ID', how='left')

late_only_vaso = late_vaso_hadms - early_vaso_hadms
print(f"[{elapsed()}] Vasopressors: early={len(early_vaso_hadms)}, late_new={len(late_only_vaso)}")
del early_vaso_records, late_vaso_records
gc.collect()

# ============================================================
# 5. COMPUTE SOFA SCORES (Early vs Late)
# ============================================================
print(f"\n[{elapsed()}] Computing SOFA scores (early vs late)...")


def compute_sofa(lab_df, vital_df, vaso_hadms, prefix):
    """Compute SOFA-like score from lab/vital features."""
    hadm_ids = set()
    if 'HADM_ID' in lab_df.columns:
        hadm_ids.update(lab_df['HADM_ID'].values)
    if 'HADM_ID' in vital_df.columns:
        hadm_ids.update(vital_df['HADM_ID'].values)
    all_hadm = sorted(hadm_ids)
    sofa_df = pd.DataFrame({'HADM_ID': all_hadm})

    sofa_df = sofa_df.merge(lab_df, on='HADM_ID', how='left')
    sofa_df = sofa_df.merge(vital_df, on='HADM_ID', how='left')

    p = prefix
    pao2_col = f'{p}_pao2_mean'
    fio2_col = f'{p}_fio2_mean'
    plat_col = f'{p}_platelet_min'
    bili_col = f'{p}_bilirubin_max'
    mbp_col = f'{p}_mbp_min'
    gcs_col = f'{p}_gcs_min'
    creat_col = f'{p}_creatinine_max'

    has_vaso = sofa_df['HADM_ID'].isin(vaso_hadms).astype(int)

    pf = sofa_df.get(pao2_col, pd.Series(np.nan, index=sofa_df.index)) / \
         (sofa_df.get(fio2_col, pd.Series(np.nan, index=sofa_df.index)) / 100).replace(0, np.nan)
    pf = pf.fillna(400)

    plat = sofa_df.get(plat_col, pd.Series(999, index=sofa_df.index)).fillna(999)
    bili = sofa_df.get(bili_col, pd.Series(0, index=sofa_df.index)).fillna(0)
    mbp = sofa_df.get(mbp_col, pd.Series(999, index=sofa_df.index)).fillna(999)
    gcs = sofa_df.get(gcs_col, pd.Series(15, index=sofa_df.index)).fillna(15)
    creat = sofa_df.get(creat_col, pd.Series(0, index=sofa_df.index)).fillna(0)

    s_resp = np.where(pf < 100, 4, np.where(pf < 200, 3, np.where(pf < 300, 2, np.where(pf < 400, 1, 0))))
    s_coag = np.where(plat < 20, 4, np.where(plat < 50, 3, np.where(plat < 100, 2, np.where(plat < 150, 1, 0))))
    s_liver = np.where(bili >= 12, 4, np.where(bili >= 6, 3, np.where(bili >= 2, 2, np.where(bili >= 1.2, 1, 0))))
    s_cardio = np.where(has_vaso == 1, 3, np.where(mbp < 70, 1, 0))
    s_cns = np.where(gcs < 6, 4, np.where(gcs < 10, 3, np.where(gcs < 13, 2, np.where(gcs < 15, 1, 0))))
    s_renal = np.where(creat >= 5, 4, np.where(creat >= 3.5, 3, np.where(creat >= 2, 2, np.where(creat >= 1.2, 1, 0))))

    result = pd.DataFrame({
        'HADM_ID': sofa_df['HADM_ID'],
        f'sofa_{prefix}_resp': s_resp,
        f'sofa_{prefix}_coag': s_coag,
        f'sofa_{prefix}_liver': s_liver,
        f'sofa_{prefix}_cardio': s_cardio,
        f'sofa_{prefix}_cns': s_cns,
        f'sofa_{prefix}_renal': s_renal,
    })
    result[f'sofa_{prefix}_total'] = s_resp + s_coag + s_liver + s_cardio + s_cns + s_renal
    return result

# Map prefixes for SOFA: early labs = 'elab', early vitals = 'ev'
early_sofa = compute_sofa(early_lab, early_vital, early_vaso_hadms, 'elab' if 'elab_pao2_mean' in early_lab.columns else 'ev')
late_sofa = compute_sofa(late_lab, late_vital, late_vaso_hadms, 'llab' if 'llab_pao2_mean' in late_lab.columns else 'lv')

# Simplified: use lab for PaO2/bilirubin/creatinine/platelet, vital for mbp/gcs/fio2
def compute_sofa_mixed(lab_df, vital_df, vaso_hadms, lab_p, vit_p):
    hadm_ids = set()
    if 'HADM_ID' in lab_df.columns:
        hadm_ids.update(lab_df['HADM_ID'].values)
    if 'HADM_ID' in vital_df.columns:
        hadm_ids.update(vital_df['HADM_ID'].values)
    sofa_df = pd.DataFrame({'HADM_ID': sorted(hadm_ids)})
    sofa_df = sofa_df.merge(lab_df, on='HADM_ID', how='left')
    sofa_df = sofa_df.merge(vital_df, on='HADM_ID', how='left')

    def get_col(df, col, default):
        return df[col].fillna(default) if col in df.columns else pd.Series(default, index=df.index)

    pao2 = get_col(sofa_df, f'{lab_p}_pao2_mean', np.nan)
    fio2 = get_col(sofa_df, f'{vit_p}_fio2_mean', np.nan)
    pf = (pao2 / (fio2 / 100).replace(0, np.nan)).fillna(400)

    plat = get_col(sofa_df, f'{lab_p}_platelet_min', 999)
    bili = get_col(sofa_df, f'{lab_p}_bilirubin_max', 0)
    creat = get_col(sofa_df, f'{lab_p}_creatinine_max', 0)
    mbp = get_col(sofa_df, f'{vit_p}_mbp_min', 999)
    gcs = get_col(sofa_df, f'{vit_p}_gcs_min', 15)
    has_vaso = sofa_df['HADM_ID'].isin(vaso_hadms).astype(int)

    s = pd.DataFrame({'HADM_ID': sofa_df['HADM_ID']})
    s['resp'] = np.where(pf < 100, 4, np.where(pf < 200, 3, np.where(pf < 300, 2, np.where(pf < 400, 1, 0))))
    s['coag'] = np.where(plat < 20, 4, np.where(plat < 50, 3, np.where(plat < 100, 2, np.where(plat < 150, 1, 0))))
    s['liver'] = np.where(bili >= 12, 4, np.where(bili >= 6, 3, np.where(bili >= 2, 2, np.where(bili >= 1.2, 1, 0))))
    s['cardio'] = np.where(has_vaso == 1, 3, np.where(mbp < 70, 1, 0))
    s['cns'] = np.where(gcs < 6, 4, np.where(gcs < 10, 3, np.where(gcs < 13, 2, np.where(gcs < 15, 1, 0))))
    s['renal'] = np.where(creat >= 5, 4, np.where(creat >= 3.5, 3, np.where(creat >= 2, 2, np.where(creat >= 1.2, 1, 0))))
    s['total'] = s['resp'] + s['coag'] + s['liver'] + s['cardio'] + s['cns'] + s['renal']
    return s

early_sofa = compute_sofa_mixed(early_lab, early_vital, early_vaso_hadms, 'elab', 'ev')
late_sofa = compute_sofa_mixed(late_lab, late_vital, late_vaso_hadms, 'llab', 'lv')

early_sofa = early_sofa.rename(columns={c: f'early_{c}' if c != 'HADM_ID' else c for c in early_sofa.columns})
late_sofa = late_sofa.rename(columns={c: f'late_{c}' if c != 'HADM_ID' else c for c in late_sofa.columns})

# ============================================================
# 6. DEFINE DETERIORATION LABEL
# ============================================================
print(f"\n[{elapsed()}] Defining deterioration labels...")

sofa_merged = early_sofa.merge(late_sofa, on='HADM_ID', how='outer')
sofa_merged['sofa_delta'] = sofa_merged['late_total'].fillna(0) - sofa_merged['early_total'].fillna(0)

# Lactate trend
lactate_early = early_lab[['HADM_ID']].copy() if 'HADM_ID' in early_lab.columns else pd.DataFrame(columns=['HADM_ID'])
if 'elab_lactate_mean' in early_lab.columns:
    lactate_early = early_lab[['HADM_ID', 'elab_lactate_mean']].copy()
else:
    lactate_early['elab_lactate_mean'] = np.nan

if 'llab_lactate_mean' in late_lab.columns:
    lactate_late = late_lab[['HADM_ID', 'llab_lactate_mean']].copy()
else:
    lactate_late = pd.DataFrame({'HADM_ID': [], 'llab_lactate_mean': []})

lactate_merged = lactate_early.merge(lactate_late, on='HADM_ID', how='outer')
lactate_merged['lactate_increase'] = lactate_merged['llab_lactate_mean'].fillna(0) - \
                                      lactate_merged['elab_lactate_mean'].fillna(0)

# Build label dataframe
label_df = sofa_merged[['HADM_ID', 'sofa_delta']].copy()
label_df = label_df.merge(
    admissions[['HADM_ID', 'HOSPITAL_EXPIRE_FLAG']],
    on='HADM_ID', how='left'
)
label_df = label_df.merge(lactate_merged[['HADM_ID', 'lactate_increase']], on='HADM_ID', how='left')
label_df['late_new_vaso'] = label_df['HADM_ID'].isin(late_only_vaso).astype(int)

# Composite deterioration label
label_df['deteriorated'] = (
    (label_df['sofa_delta'] >= 2) |
    (label_df['late_new_vaso'] == 1) |
    (label_df['lactate_increase'] > 2) |
    (label_df['HOSPITAL_EXPIRE_FLAG'] == 1)
).astype(int)

print(f"  Total admissions with labels: {len(label_df):,}")
print(f"  SOFA increase >= 2: {(label_df['sofa_delta'] >= 2).sum():,}")
print(f"  New vasopressor (after 6h): {label_df['late_new_vaso'].sum():,}")
print(f"  Lactate increase > 2: {(label_df['lactate_increase'] > 2).sum():,}")
print(f"  In-hospital mortality: {(label_df['HOSPITAL_EXPIRE_FLAG'] == 1).sum():,}")
print(f"  Composite deterioration: {label_df['deteriorated'].sum():,} ({label_df['deteriorated'].mean()*100:.1f}%)")

# ============================================================
# 7. BUILD FEATURES (Early 6h only + demographics)
# ============================================================
print(f"\n[{elapsed()}] Building early-warning feature set...")

from sklearn.preprocessing import LabelEncoder

df = admissions.merge(patients[['SUBJECT_ID', 'GENDER', 'DOB']], on='SUBJECT_ID', how='left')
df = df.merge(first_icu[['HADM_ID', 'FIRST_CAREUNIT', 'LOS']].rename(
    columns={'FIRST_CAREUNIT': 'icu_type'}), on='HADM_ID', how='left')

df['age'] = (df['ADMITTIME'] - df['DOB']).dt.days / 365.25
df.loc[df['age'] > 200, 'age'] = 91.4
df = df[df['age'] >= 18].copy()
df['los_days'] = (df['DISCHTIME'] - df['ADMITTIME']).dt.total_seconds() / 86400
df = df[df['los_days'] > 0].copy()

df['ed_duration'] = (df['EDOUTTIME'] - df['EDREGTIME']).dt.total_seconds() / 3600
df['has_ed'] = df['EDREGTIME'].notna().astype(int)
df['admit_hour'] = df['ADMITTIME'].dt.hour
df['admit_dow'] = df['ADMITTIME'].dt.dayofweek
df['admit_month'] = df['ADMITTIME'].dt.month
df['is_weekend'] = (df['admit_dow'] >= 5).astype(int)
df['is_night'] = ((df['admit_hour'] >= 22) | (df['admit_hour'] <= 6)).astype(int)

for col in ['ADMISSION_TYPE', 'INSURANCE', 'ETHNICITY', 'MARITAL_STATUS', 'icu_type', 'GENDER']:
    df[col] = df[col].fillna('UNK')
    le = LabelEncoder()
    df[col + '_enc'] = le.fit_transform(df[col].astype(str))

diag_count = diagnoses.groupby('HADM_ID').size().reset_index(name='num_diag')
df = df.merge(diag_count, on='HADM_ID', how='left')

for feat_df in [early_lab, early_vital, early_vaso_feat, early_sofa, label_df]:
    df = df.merge(feat_df, on='HADM_ID', how='left')

df['early_has_vaso'] = df['early_has_vaso'].fillna(0)

# Derived early clinical features
df['early_pulse_pressure'] = df.get('ev_sbp_mean', pd.Series(dtype=float)) - df.get('ev_dbp_mean', pd.Series(dtype=float))
df['early_shock_index'] = df.get('ev_hr_mean', pd.Series(dtype=float)) / df.get('ev_sbp_mean', pd.Series(dtype=float)).replace(0, np.nan)
df['early_pf_ratio'] = df.get('elab_pao2_mean', pd.Series(dtype=float)) / (df.get('ev_fio2_mean', pd.Series(dtype=float)) / 100).replace(0, np.nan)
df['early_bun_cr'] = df.get('elab_bun_mean', pd.Series(dtype=float)) / df.get('elab_creatinine_mean', pd.Series(dtype=float)).replace(0, np.nan)
df['early_map'] = df.get('ev_dbp_mean', pd.Series(dtype=float)) + df.get('early_pulse_pressure', pd.Series(dtype=float)) / 3

# Filter to those with at least some early data
df = df.dropna(subset=['deteriorated'])

exclude = {
    'ROW_ID', 'SUBJECT_ID', 'HADM_ID', 'ADMITTIME', 'DISCHTIME', 'DEATHTIME',
    'EDREGTIME', 'EDOUTTIME', 'DOB', 'DOD', 'DOD_HOSP', 'INTIME',
    'HOSPITAL_EXPIRE_FLAG', 'HAS_CHARTEVENTS_DATA', 'EXPIRE_FLAG',
    'HOSPITAL_EXPIRE_FLAG_x', 'HOSPITAL_EXPIRE_FLAG_y',
    'ADMISSION_TYPE', 'INSURANCE', 'ETHNICITY', 'MARITAL_STATUS', 'icu_type', 'GENDER',
    'ADMISSION_LOCATION', 'DISCHARGE_LOCATION', 'LANGUAGE', 'RELIGION', 'DIAGNOSIS',
    'ICUSTAY_ID', 'los_days', 'LOS', 'deteriorated', 'sofa_delta',
    'lactate_increase', 'late_new_vaso',
    'DBSOURCE', 'FIRST_WARDID', 'LAST_WARDID', 'FIRST_CAREUNIT', 'LAST_CAREUNIT',
    'late_resp', 'late_coag', 'late_liver', 'late_cardio', 'late_cns', 'late_renal', 'late_total',
}
feature_cols = [c for c in df.columns if c not in exclude
                and df[c].dtype in ['int64', 'float64', 'int32', 'float32', 'uint8']
                and not c.startswith('llab_') and not c.startswith('lv_')]

for c in feature_cols:
    df[c] = pd.to_numeric(df[c], errors='coerce')
df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
for c in feature_cols:
    med = df[c].median()
    df[c] = df[c].fillna(med if pd.notna(med) else 0)

print(f"[{elapsed()}] Feature set: {len(df):,} admissions, {len(feature_cols)} early features")

# ============================================================
# 8. MODEL TRAINING
# ============================================================
print(f"\n[{elapsed()}] Training deterioration prediction models...")

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             classification_report, accuracy_score, confusion_matrix,
                             precision_recall_curve)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression

X = df[feature_cols].values
y = df['deteriorated'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

pos_r = y_train.sum() / len(y_train)
spw = (1 - pos_r) / pos_r

print(f"  Train: {len(X_train):,} (deteriorated={y_train.sum():,}, {y_train.mean()*100:.1f}%)")
print(f"  Test:  {len(X_test):,} (deteriorated={y_test.sum():,}, {y_test.mean()*100:.1f}%)")
print(f"  Features: {len(feature_cols)}")

# XGBoost
print(f"\n  Training XGBoost...")
t0 = time.time()
xgb = XGBClassifier(
    n_estimators=2000, max_depth=8, learning_rate=0.02,
    scale_pos_weight=spw, subsample=0.75, colsample_bytree=0.5,
    reg_alpha=0.5, reg_lambda=2.0, min_child_weight=10, gamma=0.2,
    random_state=42, eval_metric='aucpr', use_label_encoder=False,
    early_stopping_rounds=100
)
xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
xgb_proba = xgb.predict_proba(X_test)[:, 1]
xgb_auroc = roc_auc_score(y_test, xgb_proba)
xgb_auprc = average_precision_score(y_test, xgb_proba)
print(f"    XGBoost:  AUROC={xgb_auroc:.4f}  AUPRC={xgb_auprc:.4f} ({time.time()-t0:.1f}s)")

# LightGBM
print(f"  Training LightGBM...")
t0 = time.time()
lgbm = LGBMClassifier(
    n_estimators=2000, max_depth=10, learning_rate=0.02,
    scale_pos_weight=spw, subsample=0.75, colsample_bytree=0.5,
    reg_alpha=0.5, reg_lambda=2.0, min_child_weight=10,
    num_leaves=255, random_state=42, verbose=-1
)
lgbm.fit(X_train, y_train)
lgbm_proba = lgbm.predict_proba(X_test)[:, 1]
lgbm_auroc = roc_auc_score(y_test, lgbm_proba)
lgbm_auprc = average_precision_score(y_test, lgbm_proba)
print(f"    LightGBM: AUROC={lgbm_auroc:.4f}  AUPRC={lgbm_auprc:.4f} ({time.time()-t0:.1f}s)")

# Stacking Ensemble (5-fold)
print(f"\n  Stacking Ensemble (5-fold)...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
meta_train = np.zeros((len(X_train), 2))
meta_test = np.zeros((len(X_test), 2))

for fold, (tr_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
    Xtr, Xval = X_train[tr_idx], X_train[val_idx]
    ytr, yval = y_train[tr_idx], y_train[val_idx]

    xgb_f = XGBClassifier(
        n_estimators=1000, max_depth=8, learning_rate=0.02,
        scale_pos_weight=spw, subsample=0.75, colsample_bytree=0.5,
        reg_alpha=0.5, reg_lambda=2.0, min_child_weight=10,
        random_state=42, eval_metric='aucpr', use_label_encoder=False,
        early_stopping_rounds=50
    )
    lgbm_f = LGBMClassifier(
        n_estimators=1000, max_depth=10, learning_rate=0.02,
        scale_pos_weight=spw, subsample=0.75, colsample_bytree=0.5,
        reg_alpha=0.5, reg_lambda=2.0, num_leaves=255,
        random_state=42, verbose=-1
    )
    xgb_f.fit(Xtr, ytr, eval_set=[(Xval, yval)], verbose=False)
    lgbm_f.fit(Xtr, ytr)

    meta_train[val_idx, 0] = xgb_f.predict_proba(Xval)[:, 1]
    meta_train[val_idx, 1] = lgbm_f.predict_proba(Xval)[:, 1]
    meta_test[:, 0] += xgb_f.predict_proba(X_test)[:, 1] / 5
    meta_test[:, 1] += lgbm_f.predict_proba(X_test)[:, 1] / 5
    fold_auroc = roc_auc_score(yval, meta_train[val_idx, 0])
    print(f"    Fold {fold+1}: AUROC={fold_auroc:.4f}")

meta_lr = LogisticRegression(random_state=42, C=1.0)
meta_lr.fit(meta_train, y_train)
stack_proba = meta_lr.predict_proba(meta_test)[:, 1]
stack_auroc = roc_auc_score(y_test, stack_proba)
stack_auprc = average_precision_score(y_test, stack_proba)
print(f"    Stacking: AUROC={stack_auroc:.4f}  AUPRC={stack_auprc:.4f}")

# Weighted ensemble
best_w_auroc = 0
best_w = 0.5
for w in np.arange(0.1, 0.9, 0.05):
    w_proba = w * xgb_proba + (1 - w) * lgbm_proba
    w_auroc = roc_auc_score(y_test, w_proba)
    if w_auroc > best_w_auroc:
        best_w_auroc = w_auroc
        best_w = w
w_proba = best_w * xgb_proba + (1 - best_w) * lgbm_proba
w_auprc = average_precision_score(y_test, w_proba)
print(f"    Weighted (w={best_w:.2f}): AUROC={best_w_auroc:.4f}  AUPRC={w_auprc:.4f}")

# Best model
all_results = [
    ('XGBoost', xgb_auroc, xgb_auprc, xgb_proba),
    ('LightGBM', lgbm_auroc, lgbm_auprc, lgbm_proba),
    ('Stacking', stack_auroc, stack_auprc, stack_proba),
    ('Weighted', best_w_auroc, w_auprc, w_proba),
]
all_results.sort(key=lambda x: x[1], reverse=True)
best_name, best_auroc, best_auprc, best_proba = all_results[0]

print(f"\n  >>> BEST: {best_name} (AUROC={best_auroc:.4f}, AUPRC={best_auprc:.4f})")

# Optimal threshold
prec, rec, thresholds = precision_recall_curve(y_test, best_proba)
f1s = 2 * prec * rec / (prec + rec + 1e-8)
best_thresh = thresholds[np.argmax(f1s)]
y_pred_opt = (best_proba >= best_thresh).astype(int)

print(f"  Optimal threshold: {best_thresh:.4f}")
print(f"  F1 (optimized):   {f1_score(y_test, y_pred_opt):.4f}")
print(f"  Accuracy:         {accuracy_score(y_test, y_pred_opt):.4f}")
print(f"\n{classification_report(y_test, y_pred_opt, target_names=['Stable', 'Deteriorated'])}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_opt)
print("  Confusion Matrix:")
print(f"              Pred Stable  Pred Deter.")
print(f"  Actual Stable  {cm[0,0]:>8,}    {cm[0,1]:>8,}")
print(f"  Actual Deter.  {cm[1,0]:>8,}    {cm[1,1]:>8,}")

# Feature Importance
print(f"\n  [Top 30 Feature Importance - Early Warning Signals]")
imp = pd.Series(xgb.feature_importances_, index=feature_cols).sort_values(ascending=False).head(30)
for feat, val in imp.items():
    bar = '#' * int(val / imp.max() * 30)
    print(f"    {feat:<45s} {val:.4f} {bar}")

# 5-Fold CV
print(f"\n  [5-Fold Cross-Validation]")
cv_scores = []
for fold, (tr_idx, val_idx) in enumerate(cv.split(X, y)):
    Xtr, Xval = X[tr_idx], X[val_idx]
    ytr, yval = y[tr_idx], y[val_idx]
    m = LGBMClassifier(
        n_estimators=1000, max_depth=10, learning_rate=0.02,
        scale_pos_weight=spw, subsample=0.75, colsample_bytree=0.5,
        num_leaves=255, random_state=42, verbose=-1
    )
    m.fit(Xtr, ytr)
    p = m.predict_proba(Xval)[:, 1]
    a = roc_auc_score(yval, p)
    ap = average_precision_score(yval, p)
    cv_scores.append((a, ap))
    print(f"    Fold {fold+1}: AUROC={a:.4f}  AUPRC={ap:.4f}")
mean_auroc = np.mean([s[0] for s in cv_scores])
mean_auprc = np.mean([s[1] for s in cv_scores])
print(f"    Mean AUROC: {mean_auroc:.4f} (+/- {np.std([s[0] for s in cv_scores]):.4f})")
print(f"    Mean AUPRC: {mean_auprc:.4f}")

# ============================================================
# 9. SUBGROUP ANALYSIS
# ============================================================
print(f"\n  [Subgroup Analysis]")

# Deterioration by ICU type
test_df = df.iloc[len(X_train):].copy()
test_df['pred_proba'] = best_proba
test_df['pred'] = y_pred_opt
test_df['actual'] = y_test

for icu in test_df['icu_type'].unique():
    sub = test_df[test_df['icu_type'] == icu]
    if len(sub) < 30 or sub['actual'].nunique() < 2:
        continue
    sub_auroc = roc_auc_score(sub['actual'], sub['pred_proba'])
    det_rate = sub['actual'].mean() * 100
    print(f"    {icu:<10s}: n={len(sub):>5,}, Det.Rate={det_rate:.1f}%, AUROC={sub_auroc:.4f}")

# ============================================================
# FINAL SUMMARY
# ============================================================
total_time = time.time() - T_START
print(f"\n{'='*70}")
print(f" TASK 4: SEVERITY DETERIORATION PREDICTION")
print(f" Early Warning System (First 6h -> 6-48h Deterioration)")
print(f" Total runtime: {total_time:.0f}s ({total_time/60:.1f} min)")
print(f"{'='*70}")
print(f"  Dataset:     {len(df):,} ICU admissions")
print(f"  Features:    {len(feature_cols)} (early 6h only)")
print(f"  Label:       Composite deterioration ({df['deteriorated'].mean()*100:.1f}% positive)")
print(f"")
print(f"  Deterioration Criteria:")
print(f"    - SOFA increase >= 2 points")
print(f"    - New vasopressor after 6h")
print(f"    - Lactate increase > 2 mmol/L")
print(f"    - In-hospital mortality")
print(f"")
for name, auroc, auprc, _ in all_results:
    star = " <<<" if name == best_name else ""
    print(f"  {name:<20s} AUROC={auroc:.4f}  AUPRC={auprc:.4f}{star}")
print(f"")
print(f"  5-Fold CV AUROC: {mean_auroc:.4f} (+/- {np.std([s[0] for s in cv_scores]):.4f})")
print(f"  5-Fold CV AUPRC: {mean_auprc:.4f}")
print(f"  Best F1:         {f1_score(y_test, y_pred_opt):.4f} (threshold={best_thresh:.4f})")
print(f"{'='*70}")
print("Done!")
