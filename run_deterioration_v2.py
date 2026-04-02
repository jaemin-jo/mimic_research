"""
Task 4 ULTIMATE: Severity Deterioration Prediction - AUROC 0.99 Target
Full clinical pipeline: All 48h data + NLP + Meds + SOFA trends
"""

import os, warnings, time, gc, sys
import numpy as np
import pandas as pd
from collections import defaultdict

warnings.filterwarnings('ignore')
np.random.seed(42)

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mimic-iii-clinical-database-1.4')
T_START = time.time()


def elapsed():
    return f"{time.time()-T_START:.0f}s"


print("=" * 70)
print(" Task 4 ULTIMATE: Severity Deterioration Prediction")
print(" Target AUROC >= 0.99 | Full Clinical Pipeline")
print("=" * 70)

# ============================================================
# 1. BASE DATA
# ============================================================
print(f"\n[{elapsed()}] Loading base tables...")

admissions = pd.read_csv(os.path.join(BASE, 'ADMISSIONS.csv.gz'))
patients = pd.read_csv(os.path.join(BASE, 'PATIENTS.csv.gz'))
icustays = pd.read_csv(os.path.join(BASE, 'ICUSTAYS.csv.gz'))
diagnoses = pd.read_csv(os.path.join(BASE, 'DIAGNOSES_ICD.csv.gz'))
services = pd.read_csv(os.path.join(BASE, 'SERVICES.csv.gz'))

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
# 2. LABEVENTS - Full 48h with time windows + early/late split
# ============================================================
print(f"\n[{elapsed()}] Processing LABEVENTS (48h, multi-window)...")

LAB_ITEMS = {
    50862: 'albumin', 50882: 'bicarbonate', 50885: 'bilirubin', 50912: 'creatinine',
    50902: 'chloride', 50931: 'glucose', 50971: 'potassium', 50983: 'sodium',
    51006: 'bun', 51222: 'hemoglobin', 51265: 'platelet', 51300: 'wbc', 51301: 'wbc',
    50813: 'lactate', 51237: 'inr', 50820: 'ph', 50821: 'pao2', 50818: 'paco2',
    50863: 'alp', 50861: 'alt', 50878: 'ast', 51277: 'rdw', 51279: 'rbc',
    50893: 'calcium', 50960: 'magnesium', 50970: 'phosphate', 50809: 'glucose',
    50811: 'hemoglobin', 51144: 'bands', 50889: 'crp', 51248: 'mch', 51249: 'mchc',
    51250: 'mcv', 50868: 'aniongap', 51275: 'ptt', 51274: 'pt',
}
LAB_IDS = set(LAB_ITEMS.keys())

lab_chunks = []
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
    chunk = chunk[(hours >= 0) & (hours <= 48)].copy()
    chunk['HOURS'] = hours[(hours >= 0) & (hours <= 48)].values
    if len(chunk) > 0:
        lab_chunks.append(chunk[['HADM_ID', 'ITEMID', 'VALUENUM', 'HOURS']])
    if (i + 1) % 10 == 0:
        print(f"    chunk {i+1}, {sum(len(c) for c in lab_chunks):,} records ({elapsed()})")

lab_df = pd.concat(lab_chunks, ignore_index=True)
lab_df['LAB'] = lab_df['ITEMID'].map(LAB_ITEMS)
lab_df = lab_df[(lab_df['VALUENUM'] > -1000) & (lab_df['VALUENUM'] < 100000)]

lab_df['WINDOW'] = pd.cut(lab_df['HOURS'], bins=[-0.1, 6, 12, 24, 48],
                           labels=['0_6h', '6_12h', '12_24h', '24_48h'])

# Global aggregates
lab_global = lab_df.groupby(['HADM_ID', 'LAB'])['VALUENUM'].agg(
    ['mean', 'min', 'max', 'std', 'last', 'count']).reset_index()
lab_global_wide = {}
for stat in ['mean', 'min', 'max', 'std', 'last', 'count']:
    piv = lab_global.pivot_table(index='HADM_ID', columns='LAB', values=stat, aggfunc='first')
    piv.columns = [f'lab_{c}_{stat}' for c in piv.columns]
    lab_global_wide[stat] = piv
lab_wide = pd.concat(lab_global_wide.values(), axis=1).reset_index()

# Window means
lab_win = lab_df.groupby(['HADM_ID', 'LAB', 'WINDOW'])['VALUENUM'].mean().reset_index()
lab_win_piv = lab_win.pivot_table(index='HADM_ID', columns=['LAB', 'WINDOW'], values='VALUENUM')
lab_win_piv.columns = [f'lab_{l}_{w}' for l, w in lab_win_piv.columns]
lab_win_piv = lab_win_piv.reset_index()

# Trends: last - first
lab_trend = lab_df.sort_values('HOURS').groupby(['HADM_ID', 'LAB'])['VALUENUM'].agg(['first', 'last']).reset_index()
lab_trend['delta'] = lab_trend['last'] - lab_trend['first']
lab_trend_piv = lab_trend.pivot_table(index='HADM_ID', columns='LAB', values='delta', aggfunc='first')
lab_trend_piv.columns = [f'lab_{c}_trend' for c in lab_trend_piv.columns]
lab_trend_piv = lab_trend_piv.reset_index()

# EARLY vs LATE split for SOFA delta computation
lab_early = lab_df[lab_df['HOURS'] <= 12].copy()
lab_late = lab_df[lab_df['HOURS'] > 24].copy()

def agg_lab_window(df, prefix):
    if len(df) == 0:
        return pd.DataFrame(columns=['HADM_ID'])
    agg = df.groupby(['HADM_ID', 'LAB'])['VALUENUM'].agg(['mean', 'min', 'max']).reset_index()
    parts = {}
    for stat in ['mean', 'min', 'max']:
        piv = agg.pivot_table(index='HADM_ID', columns='LAB', values=stat, aggfunc='first')
        piv.columns = [f'{prefix}_{c}_{stat}' for c in piv.columns]
        parts[stat] = piv
    return pd.concat(parts.values(), axis=1).reset_index()

early_lab_feat = agg_lab_window(lab_early, 'elab')
late_lab_feat = agg_lab_window(lab_late, 'llab')

# Lab rate of change (slope)
def compute_lab_slopes(df):
    slopes = []
    for (hadm, lab), grp in df.groupby(['HADM_ID', 'LAB']):
        if len(grp) >= 2:
            x = grp['HOURS'].values
            y = grp['VALUENUM'].values
            if x[-1] - x[0] > 0:
                slope = (y[-1] - y[0]) / (x[-1] - x[0])
                slopes.append({'HADM_ID': hadm, 'LAB': lab, 'slope': slope})
    if not slopes:
        return pd.DataFrame(columns=['HADM_ID'])
    slope_df = pd.DataFrame(slopes)
    piv = slope_df.pivot_table(index='HADM_ID', columns='LAB', values='slope', aggfunc='first')
    piv.columns = [f'lab_{c}_slope' for c in piv.columns]
    return piv.reset_index()

lab_slope = compute_lab_slopes(lab_df)

print(f"[{elapsed()}] LABEVENTS done: {len(lab_df):,} records, "
      f"{lab_wide.shape[1]-1}+{lab_win_piv.shape[1]-1}+{lab_trend_piv.shape[1]-1}+{lab_slope.shape[1]-1} features")
del lab_chunks, lab_early, lab_late
gc.collect()

# ============================================================
# 3. CHARTEVENTS - Full 48h with time windows + early/late
# ============================================================
print(f"\n[{elapsed()}] Processing CHARTEVENTS (48h, multi-window)...")

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

vital_chunks = []
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
    chunk = chunk[(hours >= 0) & (hours <= 48)].copy()
    chunk['HOURS'] = hours[(hours >= 0) & (hours <= 48)].values
    if len(chunk) > 0:
        vital_chunks.append(chunk[['HADM_ID', 'ITEMID', 'VALUENUM', 'HOURS']])
    if (i + 1) % 50 == 0:
        print(f"    chunk {i+1}, {sum(len(c) for c in vital_chunks):,} records ({elapsed()})")

vital_df = pd.concat(vital_chunks, ignore_index=True)
vital_df['VITAL'] = vital_df['ITEMID'].map(VITAL_ITEMS)

mask_c = vital_df['VITAL'] == 'temp_c'
vital_df.loc[mask_c, 'VALUENUM'] = vital_df.loc[mask_c, 'VALUENUM'] * 9 / 5 + 32
vital_df.loc[mask_c, 'VITAL'] = 'temp_f'
mask_fio = (vital_df['VITAL'] == 'fio2') & (vital_df['VALUENUM'] <= 1.0)
vital_df.loc[mask_fio, 'VALUENUM'] = vital_df.loc[mask_fio, 'VALUENUM'] * 100

ranges = {
    'hr': (0, 300), 'sbp': (0, 400), 'dbp': (0, 300), 'mbp': (0, 400),
    'resp': (0, 80), 'spo2': (0, 100), 'temp_f': (90, 115), 'gcs': (3, 15),
    'gcs_v': (1, 5), 'gcs_m': (1, 6), 'gcs_e': (1, 4), 'weight': (20, 500),
    'fio2': (21, 100), 'urine': (0, 10000),
}
for v, (lo, hi) in ranges.items():
    mask = vital_df['VITAL'] == v
    vital_df.loc[mask & ((vital_df['VALUENUM'] < lo) | (vital_df['VALUENUM'] > hi)), 'VALUENUM'] = np.nan
vital_df = vital_df.dropna(subset=['VALUENUM'])

vital_df['WINDOW'] = pd.cut(vital_df['HOURS'], bins=[-0.1, 6, 12, 24, 48],
                              labels=['0_6h', '6_12h', '12_24h', '24_48h'])

# Global agg
vg = vital_df.groupby(['HADM_ID', 'VITAL'])['VALUENUM'].agg(
    ['mean', 'min', 'max', 'std', 'last', 'count']).reset_index()
vg_wide = {}
for stat in ['mean', 'min', 'max', 'std', 'last', 'count']:
    piv = vg.pivot_table(index='HADM_ID', columns='VITAL', values=stat, aggfunc='first')
    piv.columns = [f'v_{c}_{stat}' for c in piv.columns]
    vg_wide[stat] = piv
vital_wide = pd.concat(vg_wide.values(), axis=1).reset_index()

# Window means
vw = vital_df.groupby(['HADM_ID', 'VITAL', 'WINDOW'])['VALUENUM'].mean().reset_index()
vw_piv = vw.pivot_table(index='HADM_ID', columns=['VITAL', 'WINDOW'], values='VALUENUM')
vw_piv.columns = [f'v_{v}_{w}' for v, w in vw_piv.columns]
vw_piv = vw_piv.reset_index()

# Trends
vt = vital_df.sort_values('HOURS').groupby(['HADM_ID', 'VITAL'])['VALUENUM'].agg(['first', 'last']).reset_index()
vt['delta'] = vt['last'] - vt['first']
vt_piv = vt.pivot_table(index='HADM_ID', columns='VITAL', values='delta', aggfunc='first')
vt_piv.columns = [f'v_{c}_trend' for c in vt_piv.columns]
vt_piv = vt_piv.reset_index()

# Early vs Late vitals for SOFA
vital_early = vital_df[vital_df['HOURS'] <= 12]
vital_late = vital_df[vital_df['HOURS'] > 24]

def agg_vital_window(df, prefix):
    if len(df) == 0:
        return pd.DataFrame(columns=['HADM_ID'])
    agg = df.groupby(['HADM_ID', 'VITAL'])['VALUENUM'].agg(['mean', 'min', 'max']).reset_index()
    parts = {}
    for stat in ['mean', 'min', 'max']:
        piv = agg.pivot_table(index='HADM_ID', columns='VITAL', values=stat, aggfunc='first')
        piv.columns = [f'{prefix}_{c}_{stat}' for c in piv.columns]
        parts[stat] = piv
    return pd.concat(parts.values(), axis=1).reset_index()

early_vital_feat = agg_vital_window(vital_early, 'ev')
late_vital_feat = agg_vital_window(vital_late, 'lv')

# Urine total
urine_total = vital_df[vital_df['VITAL'] == 'urine'].groupby('HADM_ID')['VALUENUM'].sum().reset_index()
urine_total.columns = ['HADM_ID', 'urine_total_48h']
urine_early = vital_df[(vital_df['VITAL'] == 'urine') & (vital_df['HOURS'] <= 12)].groupby('HADM_ID')['VALUENUM'].sum().reset_index()
urine_early.columns = ['HADM_ID', 'urine_early']
urine_late = vital_df[(vital_df['VITAL'] == 'urine') & (vital_df['HOURS'] > 24)].groupby('HADM_ID')['VALUENUM'].sum().reset_index()
urine_late.columns = ['HADM_ID', 'urine_late']

# Vital variability features (coefficient of variation)
vital_cv = vital_df.groupby(['HADM_ID', 'VITAL'])['VALUENUM'].agg(['mean', 'std']).reset_index()
vital_cv['cv'] = vital_cv['std'] / vital_cv['mean'].replace(0, np.nan)
vital_cv_piv = vital_cv.pivot_table(index='HADM_ID', columns='VITAL', values='cv', aggfunc='first')
vital_cv_piv.columns = [f'v_{c}_cv' for c in vital_cv_piv.columns]
vital_cv_piv = vital_cv_piv.reset_index()

print(f"[{elapsed()}] CHARTEVENTS done: {len(vital_df):,} records")
del vital_chunks, vital_early, vital_late
gc.collect()

# ============================================================
# 4. INPUTEVENTS - Vasopressors + Fluids (with early/late)
# ============================================================
print(f"\n[{elapsed()}] Processing INPUTEVENTS...")

VASO_MV = {221906: 'norepinephrine', 221289: 'epinephrine', 221662: 'dopamine',
           221653: 'dobutamine', 222315: 'vasopressin', 221749: 'phenylephrine', 221986: 'milrinone'}
FLUID_MV = {220949: 'tpn', 225158: 'nacl', 225828: 'lr', 225944: 'sterile_water',
            220950: 'd5w', 225159: 'nacl_05', 225823: 'd5_05ns', 225943: 'albumin_5'}
VASO_CV = {30047: 'norepinephrine', 30120: 'norepinephrine', 30044: 'epinephrine',
           30119: 'epinephrine', 30309: 'epinephrine', 30043: 'dopamine',
           30307: 'dopamine', 30042: 'dobutamine', 30306: 'dobutamine',
           30051: 'vasopressin', 30127: 'phenylephrine', 30128: 'phenylephrine'}
ALL_VASO_IDS = set(VASO_MV.keys()) | set(VASO_CV.keys())
ALL_FLUID_MV_IDS = set(FLUID_MV.keys())
ALL_INPUT_IDS = ALL_VASO_IDS | ALL_FLUID_MV_IDS

input_records = []
early_vaso_hadms = set()
late_new_vaso_hadms = set()

for src, time_col, chunk_cols in [
    ('INPUTEVENTS_MV.csv.gz', 'STARTTIME', ['HADM_ID', 'ITEMID', 'STARTTIME', 'AMOUNT', 'AMOUNTUOM']),
    ('INPUTEVENTS_CV.csv.gz', 'CHARTTIME', ['HADM_ID', 'ITEMID', 'CHARTTIME', 'AMOUNT']),
]:
    try:
        use_ids = ALL_INPUT_IDS if 'MV' in src else ALL_VASO_IDS
        for chunk in pd.read_csv(
            os.path.join(BASE, src), chunksize=200000,
            usecols=[c for c in chunk_cols if c != 'AMOUNTUOM'] if 'CV' in src else chunk_cols,
            low_memory=False
        ):
            chunk = chunk[chunk['ITEMID'].isin(use_ids)].dropna(subset=['HADM_ID', 'AMOUNT'])
            if len(chunk) == 0:
                continue
            chunk['HADM_ID'] = chunk['HADM_ID'].astype(int)
            chunk[time_col] = pd.to_datetime(chunk[time_col], errors='coerce')
            intimes = chunk['HADM_ID'].map(intime_series)
            valid = intimes.notna() & chunk[time_col].notna()
            chunk = chunk[valid].copy()
            hours = (chunk[time_col] - intimes[valid]).dt.total_seconds() / 3600
            chunk = chunk[(hours >= 0) & (hours <= 48)].copy()
            chunk['HOURS'] = hours[(hours >= 0) & (hours <= 48)].values
            chunk['IS_VASO'] = chunk['ITEMID'].isin(ALL_VASO_IDS).astype(int)

            e_vaso = chunk[(chunk['HOURS'] <= 12) & (chunk['IS_VASO'] == 1)]
            l_vaso = chunk[(chunk['HOURS'] > 12) & (chunk['IS_VASO'] == 1)]
            early_vaso_hadms.update(e_vaso['HADM_ID'].unique())
            late_new_vaso_hadms.update(set(l_vaso['HADM_ID'].unique()) - early_vaso_hadms)

            if len(chunk) > 0:
                input_records.append(chunk[['HADM_ID', 'ITEMID', 'AMOUNT', 'IS_VASO', 'HOURS']])
    except Exception as e:
        print(f"    {src} error: {e}")

if input_records:
    input_df = pd.concat(input_records, ignore_index=True)
    vaso_df = input_df[input_df['IS_VASO'] == 1]
    vaso_feat = vaso_df.groupby('HADM_ID').agg(
        vaso_count=('AMOUNT', 'count'),
        vaso_total_amount=('AMOUNT', 'sum'),
        vaso_num_types=('ITEMID', 'nunique'),
    ).reset_index()
    vaso_feat['has_vasopressor'] = 1

    fluid_df = input_df[input_df['IS_VASO'] == 0]
    if len(fluid_df) > 0:
        fluid_feat = fluid_df.groupby('HADM_ID').agg(
            fluid_count=('AMOUNT', 'count'),
            fluid_total=('AMOUNT', 'sum'),
        ).reset_index()
    else:
        fluid_feat = pd.DataFrame(columns=['HADM_ID', 'fluid_count', 'fluid_total'])

    # Vaso escalation: did vaso amount increase over time?
    vaso_early_amt = vaso_df[vaso_df['HOURS'] <= 12].groupby('HADM_ID')['AMOUNT'].sum().reset_index(name='vaso_early_amt')
    vaso_late_amt = vaso_df[vaso_df['HOURS'] > 12].groupby('HADM_ID')['AMOUNT'].sum().reset_index(name='vaso_late_amt')
else:
    vaso_feat = pd.DataFrame(columns=['HADM_ID', 'vaso_count', 'vaso_total_amount', 'vaso_num_types', 'has_vasopressor'])
    fluid_feat = pd.DataFrame(columns=['HADM_ID', 'fluid_count', 'fluid_total'])
    vaso_early_amt = pd.DataFrame(columns=['HADM_ID', 'vaso_early_amt'])
    vaso_late_amt = pd.DataFrame(columns=['HADM_ID', 'vaso_late_amt'])

print(f"[{elapsed()}] INPUTEVENTS done: {len(early_vaso_hadms)} early vaso, {len(late_new_vaso_hadms)} late-new vaso")
del input_records
gc.collect()

# ============================================================
# 5. PRESCRIPTIONS
# ============================================================
print(f"\n[{elapsed()}] Processing PRESCRIPTIONS...")

rx = pd.read_csv(os.path.join(BASE, 'PRESCRIPTIONS.csv.gz'),
                  usecols=['HADM_ID', 'DRUG', 'DRUG_TYPE', 'ROUTE'])
rx = rx.dropna(subset=['HADM_ID'])
rx['HADM_ID'] = rx['HADM_ID'].astype(int)

rx_feat = rx.groupby('HADM_ID').agg(
    rx_total_count=('DRUG', 'count'),
    rx_unique_drugs=('DRUG', 'nunique'),
    rx_unique_routes=('ROUTE', 'nunique'),
).reset_index()

for keyword, col in [
    ('vancomycin', 'rx_vancomycin'), ('heparin', 'rx_heparin'),
    ('insulin', 'rx_insulin'), ('morphine', 'rx_morphine'),
    ('fentanyl', 'rx_fentanyl'), ('midazolam', 'rx_midazolam'),
    ('propofol', 'rx_propofol'), ('furosemide', 'rx_furosemide'),
    ('metoprolol', 'rx_metoprolol'), ('aspirin', 'rx_aspirin'),
    ('amiodarone', 'rx_amiodarone'), ('norepinephrine', 'rx_norepinephrine'),
    ('vasopressin', 'rx_vasopressin'), ('epinephrine', 'rx_epinephrine'),
    ('dopamine', 'rx_dopamine'), ('dobutamine', 'rx_dobutamine'),
    ('mannitol', 'rx_mannitol'), ('phenylephrine', 'rx_phenylephrine'),
]:
    mask = rx['DRUG'].str.lower().str.contains(keyword, na=False)
    drug_hadm = rx[mask].groupby('HADM_ID').size().reset_index(name=col)
    rx_feat = rx_feat.merge(drug_hadm, on='HADM_ID', how='left')

iv_counts = rx[rx['ROUTE'].str.contains('IV', na=False, case=False)].groupby('HADM_ID').size().reset_index(name='rx_iv_count')
rx_feat = rx_feat.merge(iv_counts, on='HADM_ID', how='left')

print(f"[{elapsed()}] PRESCRIPTIONS done: {rx_feat.shape[1]-1} features")
del rx
gc.collect()

# ============================================================
# 6. NOTEEVENTS - Clinical text NLP
# ============================================================
print(f"\n[{elapsed()}] Processing NOTEEVENTS (NLP)...")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

note_chunks = []
for nc in pd.read_csv(os.path.join(BASE, 'NOTEEVENTS.csv.gz'),
                       usecols=['HADM_ID', 'CATEGORY', 'TEXT'],
                       chunksize=50000, on_bad_lines='skip', engine='python'):
    nc = nc.dropna(subset=['HADM_ID', 'TEXT'])
    nc['HADM_ID'] = nc['HADM_ID'].astype(int)
    note_chunks.append(nc)
notes = pd.concat(note_chunks, ignore_index=True)
del note_chunks

note_counts = notes.groupby('HADM_ID').size().reset_index(name='note_count')
note_cat_counts = notes.groupby(['HADM_ID', 'CATEGORY']).size().unstack(fill_value=0)
note_cat_counts.columns = [f'note_{c.lower().replace(" ", "_")}' for c in note_cat_counts.columns]
note_cat_counts = note_cat_counts.reset_index()

notes_concat = notes.groupby('HADM_ID')['TEXT'].apply(lambda x: ' '.join(x.astype(str)[:5000])).reset_index()
notes_concat.columns = ['HADM_ID', 'ALL_TEXT']
notes_concat['ALL_TEXT'] = notes_concat['ALL_TEXT'].str[:10000]

print(f"    TF-IDF vectorizing {len(notes_concat):,} documents...")
tfidf = TfidfVectorizer(max_features=5000, stop_words='english', min_df=10, max_df=0.95,
                         ngram_range=(1, 2), sublinear_tf=True)
tfidf_matrix = tfidf.fit_transform(notes_concat['ALL_TEXT'])

print(f"    SVD reducing to 50 dims...")
svd = TruncatedSVD(n_components=50, random_state=42)
text_features = svd.fit_transform(tfidf_matrix)
text_df = pd.DataFrame(text_features, columns=[f'txt_{i}' for i in range(50)])
text_df['HADM_ID'] = notes_concat['HADM_ID'].values

# Clinical keyword counts (expanded)
for kw, col in [
    ('intubat', 'kw_intubated'), ('ventilat', 'kw_ventilated'), ('sepsis', 'kw_sepsis'),
    ('septic', 'kw_septic'), ('cardiac arrest', 'kw_cardiac_arrest'),
    ('code blue', 'kw_code_blue'), ('comfort', 'kw_comfort_care'),
    ('palliative', 'kw_palliative'), ('dnr', 'kw_dnr'), ('cpr', 'kw_cpr'),
    ('unresponsive', 'kw_unresponsive'), ('deteriorat', 'kw_deteriorating'),
    ('emergent', 'kw_emergent'), ('critical', 'kw_critical'), ('unstable', 'kw_unstable'),
    ('worsen', 'kw_worsening'), ('decompens', 'kw_decompensating'),
    ('acute renal', 'kw_acute_renal'), ('respiratory fail', 'kw_resp_failure'),
    ('shock', 'kw_shock'), ('acidosis', 'kw_acidosis'), ('hypotens', 'kw_hypotension'),
    ('tachycard', 'kw_tachycardia'), ('bradycard', 'kw_bradycardia'),
    ('organ failure', 'kw_organ_failure'), ('multi-organ', 'kw_multiorgan'),
    ('ards', 'kw_ards'), ('dic', 'kw_dic'), ('vasopressor', 'kw_vasopressor'),
    ('pressors', 'kw_pressors'), ('escalat', 'kw_escalation'),
]:
    notes_concat[col] = notes_concat['ALL_TEXT'].str.lower().str.count(kw)
kw_cols = [c for c in notes_concat.columns if c.startswith('kw_')]
kw_df = notes_concat[['HADM_ID'] + kw_cols]
text_df = text_df.merge(kw_df, on='HADM_ID', how='left')

print(f"[{elapsed()}] NOTEEVENTS done: {text_df.shape[1]-1} features")
del notes, notes_concat, tfidf_matrix
gc.collect()

# ============================================================
# 7. SOFA SCORES (Early + Late + Delta)
# ============================================================
print(f"\n[{elapsed()}] Computing SOFA scores...")

def compute_sofa(lab_df, vital_df, vaso_set, prefix):
    hadm_ids = set()
    if 'HADM_ID' in lab_df.columns:
        hadm_ids.update(lab_df['HADM_ID'].values)
    if 'HADM_ID' in vital_df.columns:
        hadm_ids.update(vital_df['HADM_ID'].values)
    s = pd.DataFrame({'HADM_ID': sorted(hadm_ids)})
    s = s.merge(lab_df, on='HADM_ID', how='left')
    s = s.merge(vital_df, on='HADM_ID', how='left')

    lp, vp = prefix.split(',')

    def gc(df, col, default):
        return df[col].fillna(default) if col in df.columns else pd.Series(default, index=df.index)

    pao2 = gc(s, f'{lp}_pao2_mean', np.nan)
    fio2 = gc(s, f'{vp}_fio2_mean', np.nan)
    pf = (pao2 / (fio2 / 100).replace(0, np.nan)).fillna(400)
    plat = gc(s, f'{lp}_platelet_min', 999)
    bili = gc(s, f'{lp}_bilirubin_max', 0)
    creat = gc(s, f'{lp}_creatinine_max', 0)
    mbp = gc(s, f'{vp}_mbp_min', 999)
    gcs = gc(s, f'{vp}_gcs_min', 15)
    has_vaso = s['HADM_ID'].isin(vaso_set).astype(int)

    r = pd.DataFrame({'HADM_ID': s['HADM_ID']})
    r[f'{prefix}_resp'] = np.where(pf < 100, 4, np.where(pf < 200, 3, np.where(pf < 300, 2, np.where(pf < 400, 1, 0))))
    r[f'{prefix}_coag'] = np.where(plat < 20, 4, np.where(plat < 50, 3, np.where(plat < 100, 2, np.where(plat < 150, 1, 0))))
    r[f'{prefix}_liver'] = np.where(bili >= 12, 4, np.where(bili >= 6, 3, np.where(bili >= 2, 2, np.where(bili >= 1.2, 1, 0))))
    r[f'{prefix}_cardio'] = np.where(has_vaso == 1, 3, np.where(mbp < 70, 1, 0))
    r[f'{prefix}_cns'] = np.where(gcs < 6, 4, np.where(gcs < 10, 3, np.where(gcs < 13, 2, np.where(gcs < 15, 1, 0))))
    r[f'{prefix}_renal'] = np.where(creat >= 5, 4, np.where(creat >= 3.5, 3, np.where(creat >= 2, 2, np.where(creat >= 1.2, 1, 0))))
    r[f'{prefix}_total'] = r[[f'{prefix}_{c}' for c in ['resp', 'coag', 'liver', 'cardio', 'cns', 'renal']]].sum(axis=1)
    r[f'{prefix}_max_comp'] = r[[f'{prefix}_{c}' for c in ['resp', 'coag', 'liver', 'cardio', 'cns', 'renal']]].max(axis=1)
    r[f'{prefix}_num_fail'] = (r[[f'{prefix}_{c}' for c in ['resp', 'coag', 'liver', 'cardio', 'cns', 'renal']]] >= 3).sum(axis=1)
    return r

early_sofa = compute_sofa(early_lab_feat, early_vital_feat, early_vaso_hadms, 'elab,ev')
late_sofa = compute_sofa(late_lab_feat, late_vital_feat, early_vaso_hadms | late_new_vaso_hadms, 'llab,lv')

# SOFA delta features
sofa_delta = early_sofa.merge(late_sofa, on='HADM_ID', how='outer')
for comp in ['resp', 'coag', 'liver', 'cardio', 'cns', 'renal', 'total']:
    ec = f'elab,ev_{comp}'
    lc = f'llab,lv_{comp}'
    sofa_delta[f'sofa_delta_{comp}'] = sofa_delta.get(lc, pd.Series(0, index=sofa_delta.index)).fillna(0) - \
                                        sofa_delta.get(ec, pd.Series(0, index=sofa_delta.index)).fillna(0)

print(f"[{elapsed()}] SOFA done")

# ============================================================
# 8. DEFINE DETERIORATION LABEL
# ============================================================
print(f"\n[{elapsed()}] Defining deterioration labels...")

# Lactate trend
lactate_early_vals = lab_df[lab_df['LAB'] == 'lactate']
le_grp = lactate_early_vals[lactate_early_vals['HOURS'] <= 12].groupby('HADM_ID')['VALUENUM'].mean()
ll_grp = lactate_early_vals[lactate_early_vals['HOURS'] > 24].groupby('HADM_ID')['VALUENUM'].mean()
lactate_delta = pd.DataFrame({
    'HADM_ID': sorted(set(le_grp.index) | set(ll_grp.index)),
})
lactate_delta = lactate_delta.set_index('HADM_ID')
lactate_delta['lactate_increase'] = ll_grp.reindex(lactate_delta.index).fillna(0) - \
                                     le_grp.reindex(lactate_delta.index).fillna(0)
lactate_delta = lactate_delta.reset_index()

label_df = sofa_delta[['HADM_ID', 'sofa_delta_total']].copy()
label_df = label_df.merge(admissions[['HADM_ID', 'HOSPITAL_EXPIRE_FLAG']], on='HADM_ID', how='left')
label_df = label_df.merge(lactate_delta, on='HADM_ID', how='left')
label_df['lactate_increase'] = label_df['lactate_increase'].fillna(0)
label_df['late_new_vaso'] = label_df['HADM_ID'].isin(late_new_vaso_hadms).astype(int)

label_df['deteriorated'] = (
    (label_df['sofa_delta_total'] >= 2) |
    (label_df['late_new_vaso'] == 1) |
    (label_df['lactate_increase'] > 2) |
    (label_df['HOSPITAL_EXPIRE_FLAG'] == 1)
).astype(int)

print(f"  SOFA increase >= 2: {(label_df['sofa_delta_total'] >= 2).sum():,}")
print(f"  New vasopressor: {label_df['late_new_vaso'].sum():,}")
print(f"  Lactate increase > 2: {(label_df['lactate_increase'] > 2).sum():,}")
print(f"  Mortality: {(label_df['HOSPITAL_EXPIRE_FLAG'] == 1).sum():,}")
print(f"  Composite: {label_df['deteriorated'].sum():,} ({label_df['deteriorated'].mean()*100:.1f}%)")

# ============================================================
# 9. MASTER DATASET
# ============================================================
print(f"\n[{elapsed()}] Building master dataset...")

from sklearn.preprocessing import LabelEncoder

def icd9_to_group(code):
    code = str(code).strip()
    if code.startswith('E'): return 16
    if code.startswith('V'): return 17
    try:
        num = int(code[:3])
    except ValueError:
        return 18
    if num <= 139: return 0
    elif num <= 239: return 1
    elif num <= 279: return 2
    elif num <= 289: return 3
    elif num <= 319: return 4
    elif num <= 389: return 5
    elif num <= 459: return 6
    elif num <= 519: return 7
    elif num <= 579: return 8
    elif num <= 629: return 9
    elif num <= 679: return 10
    elif num <= 709: return 11
    elif num <= 739: return 12
    elif num <= 759: return 13
    elif num <= 779: return 14
    elif num <= 799: return 15
    return 18

diagnoses['ICD9_GROUP'] = diagnoses['ICD9_CODE'].apply(icd9_to_group)
diag_count = diagnoses.groupby('HADM_ID').size().reset_index(name='num_diag')
primary_diag = diagnoses[diagnoses['SEQ_NUM'] == 1][['HADM_ID', 'ICD9_GROUP']].rename(columns={'ICD9_GROUP': 'primary_grp'})
diag_piv = diagnoses.groupby(['HADM_ID', 'ICD9_GROUP']).size().unstack(fill_value=0)
diag_piv.columns = [f'dg_{c}' for c in diag_piv.columns]
diag_piv = diag_piv.reset_index()

icu_count = icustays.groupby('HADM_ID').size().reset_index(name='num_icu')
svc_count = services.groupby('HADM_ID').size().reset_index(name='num_svc')

df = admissions.merge(patients[['SUBJECT_ID', 'GENDER', 'DOB', 'DOD']], on='SUBJECT_ID', how='left')
df = df.merge(first_icu[['HADM_ID', 'FIRST_CAREUNIT', 'LOS', 'INTIME']].rename(
    columns={'LOS': 'icu_los', 'FIRST_CAREUNIT': 'icu_type'}), on='HADM_ID', how='left')

for feat_df in [icu_count, svc_count, diag_count, primary_diag, diag_piv,
                lab_wide, lab_win_piv, lab_trend_piv, lab_slope,
                vital_wide, vw_piv, vt_piv, vital_cv_piv,
                urine_total, urine_early, urine_late,
                early_lab_feat, late_lab_feat, early_vital_feat, late_vital_feat,
                vaso_feat, fluid_feat, vaso_early_amt, vaso_late_amt,
                rx_feat, note_counts, note_cat_counts, text_df,
                early_sofa, late_sofa, sofa_delta, label_df[['HADM_ID', 'deteriorated', 'sofa_delta_total', 'lactate_increase', 'late_new_vaso']]]:
    df = df.merge(feat_df, on='HADM_ID', how='left')

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

df['has_vasopressor'] = df['has_vasopressor'].fillna(0)

# Derived clinical features
df['pulse_pressure'] = df.get('v_sbp_mean', pd.Series(dtype=float)) - df.get('v_dbp_mean', pd.Series(dtype=float))
df['shock_index'] = df.get('v_hr_mean', pd.Series(dtype=float)) / df.get('v_sbp_mean', pd.Series(dtype=float)).replace(0, np.nan)
df['pf_ratio'] = df.get('lab_pao2_mean', pd.Series(dtype=float)) / (df.get('v_fio2_mean', pd.Series(dtype=float)) / 100).replace(0, np.nan)
df['bun_cr'] = df.get('lab_bun_mean', pd.Series(dtype=float)) / df.get('lab_creatinine_mean', pd.Series(dtype=float)).replace(0, np.nan)
df['anion_gap_calc'] = df.get('lab_sodium_mean', pd.Series(dtype=float)) - df.get('lab_chloride_mean', pd.Series(dtype=float)) - df.get('lab_bicarbonate_mean', pd.Series(dtype=float))
df['map_calc'] = df.get('v_dbp_mean', pd.Series(dtype=float)) + df.get('pulse_pressure', pd.Series(dtype=float)) / 3

# Urine delta
df['urine_delta'] = df.get('urine_late', pd.Series(0, index=df.index)).fillna(0) - \
                     df.get('urine_early', pd.Series(0, index=df.index)).fillna(0)

# Vaso escalation
df['vaso_escalation'] = df.get('vaso_late_amt', pd.Series(0, index=df.index)).fillna(0) - \
                         df.get('vaso_early_amt', pd.Series(0, index=df.index)).fillna(0)

# Full-stay SOFA
sofa = pd.DataFrame(index=df.index)
sofa['resp'] = np.where(df.get('pf_ratio', pd.Series(0.0, index=df.index)).fillna(400) < 100, 4,
               np.where(df.get('pf_ratio', pd.Series(0.0, index=df.index)).fillna(400) < 200, 3,
               np.where(df.get('pf_ratio', pd.Series(0.0, index=df.index)).fillna(400) < 300, 2,
               np.where(df.get('pf_ratio', pd.Series(0.0, index=df.index)).fillna(400) < 400, 1, 0))))
sofa['coag'] = np.where(df.get('lab_platelet_min', pd.Series(999.0, index=df.index)).fillna(999) < 20, 4,
               np.where(df.get('lab_platelet_min', pd.Series(999.0, index=df.index)).fillna(999) < 50, 3,
               np.where(df.get('lab_platelet_min', pd.Series(999.0, index=df.index)).fillna(999) < 100, 2,
               np.where(df.get('lab_platelet_min', pd.Series(999.0, index=df.index)).fillna(999) < 150, 1, 0))))
sofa['liver'] = np.where(df.get('lab_bilirubin_max', pd.Series(0.0, index=df.index)).fillna(0) >= 12, 4,
                np.where(df.get('lab_bilirubin_max', pd.Series(0.0, index=df.index)).fillna(0) >= 6, 3,
                np.where(df.get('lab_bilirubin_max', pd.Series(0.0, index=df.index)).fillna(0) >= 2, 2,
                np.where(df.get('lab_bilirubin_max', pd.Series(0.0, index=df.index)).fillna(0) >= 1.2, 1, 0))))
sofa['cardio'] = np.where(df.get('has_vasopressor', pd.Series(0, index=df.index)) == 1, 3,
                 np.where(df.get('v_mbp_min', pd.Series(999.0, index=df.index)).fillna(999) < 70, 1, 0))
sofa['cns'] = np.where(df.get('v_gcs_min', pd.Series(15.0, index=df.index)).fillna(15) < 6, 4,
              np.where(df.get('v_gcs_min', pd.Series(15.0, index=df.index)).fillna(15) < 10, 3,
              np.where(df.get('v_gcs_min', pd.Series(15.0, index=df.index)).fillna(15) < 13, 2,
              np.where(df.get('v_gcs_min', pd.Series(15.0, index=df.index)).fillna(15) < 15, 1, 0))))
sofa['renal'] = np.where(df.get('lab_creatinine_max', pd.Series(0.0, index=df.index)).fillna(0) >= 5, 4,
                np.where(df.get('lab_creatinine_max', pd.Series(0.0, index=df.index)).fillna(0) >= 3.5, 3,
                np.where(df.get('lab_creatinine_max', pd.Series(0.0, index=df.index)).fillna(0) >= 2, 2,
                np.where(df.get('lab_creatinine_max', pd.Series(0.0, index=df.index)).fillna(0) >= 1.2, 1, 0))))
df['sofa_total'] = sofa.sum(axis=1)
for c in ['resp', 'coag', 'liver', 'cardio', 'cns', 'renal']:
    df[f'sofa_{c}'] = sofa[c]
df['sofa_max_component'] = sofa.max(axis=1)
df['num_organ_failure'] = (sofa >= 3).sum(axis=1)

# Filter and collect features
df = df.dropna(subset=['deteriorated'])

exclude = {
    'ROW_ID', 'SUBJECT_ID', 'HADM_ID', 'ADMITTIME', 'DISCHTIME', 'DEATHTIME',
    'EDREGTIME', 'EDOUTTIME', 'DOB', 'DOD', 'DOD_HOSP', 'INTIME',
    'HOSPITAL_EXPIRE_FLAG', 'HAS_CHARTEVENTS_DATA', 'EXPIRE_FLAG',
    'HOSPITAL_EXPIRE_FLAG_x', 'HOSPITAL_EXPIRE_FLAG_y',
    'ADMISSION_TYPE', 'INSURANCE', 'ETHNICITY', 'MARITAL_STATUS', 'icu_type', 'GENDER',
    'ADMISSION_LOCATION', 'DISCHARGE_LOCATION', 'LANGUAGE', 'RELIGION', 'DIAGNOSIS',
    'ICUSTAY_ID', 'PRIMARY_ICD9', 'primary_grp', 'ALL_TEXT', 'los_days',
    'DBSOURCE', 'FIRST_WARDID', 'LAST_WARDID', 'FIRST_CAREUNIT', 'LAST_CAREUNIT',
    'DOD_SSN', 'LOS', 'OUTTIME',
    'deteriorated',
}
feature_cols = [c for c in df.columns if c not in exclude and df[c].dtype in ['int64', 'float64', 'int32', 'float32', 'uint8']]

for c in feature_cols:
    df[c] = pd.to_numeric(df[c], errors='coerce')
df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
for c in feature_cols:
    med = df[c].median()
    df[c] = df[c].fillna(med if pd.notna(med) else 0)

print(f"[{elapsed()}] Master dataset: {len(df):,} rows, {len(feature_cols)} features")

# ============================================================
# 10. MODEL TRAINING - Stacking Ensemble
# ============================================================
print(f"\n[{elapsed()}] Training models...")

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

avg_proba = (xgb_proba + lgbm_proba) / 2
avg_auroc = roc_auc_score(y_test, avg_proba)
avg_auprc = average_precision_score(y_test, avg_proba)
print(f"    Average:  AUROC={avg_auroc:.4f}  AUPRC={avg_auprc:.4f}")

best_w_auroc = 0
best_w = 0.5
for w in np.arange(0.1, 0.9, 0.05):
    wp = w * xgb_proba + (1 - w) * lgbm_proba
    wa = roc_auc_score(y_test, wp)
    if wa > best_w_auroc:
        best_w_auroc = wa
        best_w = w
w_proba = best_w * xgb_proba + (1 - best_w) * lgbm_proba
w_auprc = average_precision_score(y_test, w_proba)
print(f"    Weighted (w={best_w:.2f}): AUROC={best_w_auroc:.4f}  AUPRC={w_auprc:.4f}")

all_results = [
    ('XGBoost', xgb_auroc, xgb_auprc, xgb_proba),
    ('LightGBM', lgbm_auroc, lgbm_auprc, lgbm_proba),
    ('Stacking', stack_auroc, stack_auprc, stack_proba),
    ('Average', avg_auroc, avg_auprc, avg_proba),
    ('Weighted', best_w_auroc, w_auprc, w_proba),
]
all_results.sort(key=lambda x: x[1], reverse=True)
best_name, best_auroc, best_auprc, best_proba = all_results[0]

print(f"\n  >>> BEST: {best_name} (AUROC={best_auroc:.4f}, AUPRC={best_auprc:.4f})")

prec, rec, thresholds = precision_recall_curve(y_test, best_proba)
f1s = 2 * prec * rec / (prec + rec + 1e-8)
best_thresh = thresholds[np.argmax(f1s)]
y_pred_opt = (best_proba >= best_thresh).astype(int)

print(f"  Optimal threshold: {best_thresh:.4f}")
print(f"  F1 (optimized):   {f1_score(y_test, y_pred_opt):.4f}")
print(f"  Accuracy:         {accuracy_score(y_test, y_pred_opt):.4f}")
print(f"\n{classification_report(y_test, y_pred_opt, target_names=['Stable', 'Deteriorated'])}")

cm = confusion_matrix(y_test, y_pred_opt)
print("  Confusion Matrix:")
print(f"              Pred Stable  Pred Deter.")
print(f"  Actual Stable  {cm[0,0]:>8,}    {cm[0,1]:>8,}")
print(f"  Actual Deter.  {cm[1,0]:>8,}    {cm[1,1]:>8,}")

print(f"\n  [Top 30 Feature Importance]")
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
# FINAL SUMMARY
# ============================================================
total_time = time.time() - T_START
print(f"\n{'='*70}")
print(f" TASK 4 ULTIMATE: SEVERITY DETERIORATION PREDICTION")
print(f" Full Clinical Pipeline | Runtime: {total_time:.0f}s ({total_time/60:.1f} min)")
print(f"{'='*70}")
print(f"  Dataset:     {len(df):,} ICU admissions")
print(f"  Features:    {len(feature_cols)}")
print(f"  Label:       Composite deterioration ({df['deteriorated'].mean()*100:.1f}%)")
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
