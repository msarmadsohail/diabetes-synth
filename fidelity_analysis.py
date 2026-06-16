import pandas as pd
import numpy as np


def pct(series, condition):
    return 100.0 * condition.sum() / len(series) if len(series) > 0 else float('nan')


def num_stats(series):
    return {'mean': series.mean(), 'median': series.median(), 'std': series.std()}


def cat_dist(series):
    return (series.value_counts(normalize=True) * 100).to_dict()


def compute_metrics(df):
    m = {}
    m['n'] = len(df)
    m['positive_rate'] = pct(df, df['diabetes'] == 1)
    m['age'] = num_stats(df['age'])
    m['bmi'] = num_stats(df['bmi'])
    m['hba1c'] = num_stats(df['HbA1c_level'])
    m['bgl'] = num_stats(df['blood_glucose_level'])
    m['hypertension'] = pct(df, df['hypertension'] == 1)
    m['heart_disease'] = pct(df, df['heart_disease'] == 1)
    m['gender'] = cat_dist(df['gender'])
    m['smoking'] = cat_dist(df['smoking_history'])
    return m


MODELS = {
    'M1': {'file': 'pool_m1.csv', 'filter_real': True,  'label': 'Diabetic-only model'},
    'M2': {'file': 'pool_m2.csv', 'filter_real': True,  'label': 'Balanced model (~50/50)'},
    'M3': {'file': 'pool_m3.csv', 'filter_real': True,  'label': 'Full-dist model (diabetic rows saved)'},
}

SMOKING_CATS = ['never', 'No Info', 'former', 'current', 'not current', 'ever']


def print_fold(fold, rm, sm, model):
    print(f"\n{'─'*72}")
    print(f"  {model} FOLD {fold}  |  Real: {rm['n']:,}  |  Synth: {sm['n']:,}")
    print(f"{'─'*72}")
    print(f"  {'Metric':<30} {'Real':>12} {'Synth':>12} {'Delta':>10}")
    print(f"  {'─'*30} {'─'*12} {'─'*12} {'─'*10}")

    def row(label, rv, sv, pct_sign=False):
        delta = sv - rv
        sign = '+' if delta >= 0 else ''
        suffix = '%' if pct_sign else ''
        print(f"  {label:<30} {rv:>11.2f}{suffix} {sv:>11.2f}{suffix} {sign}{delta:>8.2f}{suffix}")

    row('Age mean', rm['age']['mean'], sm['age']['mean'])
    row('Age median', rm['age']['median'], sm['age']['median'])
    row('Age std', rm['age']['std'], sm['age']['std'])
    row('BMI mean', rm['bmi']['mean'], sm['bmi']['mean'])
    row('BMI median', rm['bmi']['median'], sm['bmi']['median'])
    row('HbA1c mean', rm['hba1c']['mean'], sm['hba1c']['mean'])
    row('HbA1c median', rm['hba1c']['median'], sm['hba1c']['median'])
    row('Blood glucose mean', rm['bgl']['mean'], sm['bgl']['mean'])
    row('Blood glucose median', rm['bgl']['median'], sm['bgl']['median'])
    row('Hypertension (%)', rm['hypertension'], sm['hypertension'], True)
    row('Heart disease (%)', rm['heart_disease'], sm['heart_disease'], True)

    print(f"\n  Gender distribution:")
    print(f"  {'Category':<20} {'Real':>10} {'Synth':>10} {'Delta':>10}")
    print(f"  {'─'*20} {'─'*10} {'─'*10} {'─'*10}")
    for g in ['Female', 'Male', 'Other']:
        rv = rm['gender'].get(g, 0.0)
        sv = sm['gender'].get(g, 0.0)
        delta = sv - rv
        sign = '+' if delta >= 0 else ''
        print(f"  {g:<20} {rv:>9.2f}% {sv:>9.2f}% {sign}{delta:>8.2f}%")

    print(f"\n  Smoking history:")
    print(f"  {'Category':<20} {'Real':>10} {'Synth':>10} {'Delta':>10}")
    print(f"  {'─'*20} {'─'*10} {'─'*10} {'─'*10}")
    for s in SMOKING_CATS:
        rv = rm['smoking'].get(s, 0.0)
        sv = sm['smoking'].get(s, 0.0)
        delta = sv - rv
        sign = '+' if delta >= 0 else ''
        print(f"  {s:<20} {rv:>9.2f}% {sv:>9.2f}% {sign}{delta:>8.2f}%")


print("=" * 72)
print("Diabetes Synthetic Data Fidelity Analysis — All 3 Models, 5 Folds")
print("=" * 72)

for model_key, cfg in MODELS.items():
    all_real, all_synth = [], []

    print(f"\n{'#'*72}")
    print(f"  {model_key} — {cfg['label']}")
    print(f"{'#'*72}")

    for fold in range(5):
        real = pd.read_csv(f"/shared/diabetes-synth/data/splits/{fold}/train.csv")
        if cfg['filter_real']:
            real = real[real['diabetes'] == 1].reset_index(drop=True)

        synth = pd.read_csv(f"/shared/diabetes-synth/synthetic/fold_{fold}/{cfg['file']}")

        rm = compute_metrics(real)
        sm = compute_metrics(synth)
        all_real.append(rm)
        all_synth.append(sm)

        print_fold(fold, rm, sm, model_key)

    # Summary
    print(f"\n{'='*72}")
    print(f"  {model_key} SUMMARY — Average across 5 folds")
    print(f"{'='*72}")
    print(f"  {'Metric':<30} {'Avg Real':>12} {'Avg Synth':>12} {'Delta':>10}")
    print(f"  {'─'*30} {'─'*12} {'─'*12} {'─'*10}")

    def avg(key, sub=None):
        if sub:
            return np.mean([m[key][sub] for m in all_real]), np.mean([m[key][sub] for m in all_synth])
        return np.mean([m[key] for m in all_real]), np.mean([m[key] for m in all_synth])

    def srow(label, rv, sv, pct_sign=False):
        delta = sv - rv
        sign = '+' if delta >= 0 else ''
        suffix = '%' if pct_sign else ' '
        flag = ' ◄' if abs(delta) > (2 if pct_sign else 0.5) else ''
        print(f"  {label:<30} {rv:>11.2f}{suffix} {sv:>11.2f}{suffix} {sign}{delta:>8.2f}{suffix}{flag}")

    srow('Age mean',           *avg('age', 'mean'))
    srow('Age median',         *avg('age', 'median'))
    srow('BMI mean',           *avg('bmi', 'mean'))
    srow('BMI median',         *avg('bmi', 'median'))
    srow('HbA1c mean',         *avg('hba1c', 'mean'))
    srow('HbA1c median',       *avg('hba1c', 'median'))
    srow('Blood glucose mean',  *avg('bgl', 'mean'))
    srow('Blood glucose median',*avg('bgl', 'median'))
    srow('Hypertension (%)',    *avg('hypertension'), True)
    srow('Heart disease (%)',   *avg('heart_disease'), True)

    print(f"\n  Smoking history (avg %):")
    for s in SMOKING_CATS:
        rv = np.mean([m['smoking'].get(s, 0.0) for m in all_real])
        sv = np.mean([m['smoking'].get(s, 0.0) for m in all_synth])
        delta = sv - rv
        sign = '+' if delta >= 0 else ''
        flag = ' ◄' if abs(delta) > 2 else ''
        print(f"  {s:<30} {rv:>11.2f}% {sv:>11.2f}% {sign}{delta:>8.2f}%{flag}")

print()
