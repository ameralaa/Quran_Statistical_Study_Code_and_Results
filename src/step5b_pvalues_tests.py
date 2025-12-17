# compute_pvalues_and_tests.py
"""
Compute empirical p-values and statistical tests comparing real metrics to null model simulations.

Inputs:
  - results/null_models/all_simulations_metrics.csv   (one row per sim: model, sim, slope, r2, entropy_words, entropy_chars, autocorr_lag1, hurst)
  - results/advanced_summary.json  (contains real/reference metrics; or edit real_metrics dict below)

Outputs:
  - results/null_models/pvalues_and_tests.csv
  - results/null_models/pvalues_details.json
  - prints summary to console
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path('.')
NM = ROOT / 'final_results' / 'step5_null_models'
NM.mkdir(parents=True, exist_ok=True)

ALL = NM / 'all_simulations_metrics.csv'
ADV = ROOT / 'final_results' / 'step2_advanced_analysis' /'advanced_summary.json'  # fallback source for real values

if not ALL.exists():
    raise FileNotFoundError(f"{ALL} not found. Place all_simulations_metrics.csv in results/null_models/")

sim_df = pd.read_csv(ALL, encoding='utf-8')

# load real metrics (try advanced_summary.json)
real_metrics = {}
if ADV.exists():
    adv = json.loads(ADV.read_text(encoding='utf-8'))
    real_metrics = {
        'slope': adv['zipf_fit']['slope'],
        'r2': adv['zipf_fit']['r2'],
        'entropy_words': adv['entropy_words'],
        'entropy_chars': adv['entropy_chars'],
        'autocorr_lag1': adv['autocorr_first_10']['1'],
        'hurst': adv.get('hurst', adv.get('hurst', None))
    }
else:
    # If not available, edit here with known values
    real_metrics = {
        'slope': -1.0136338669595237,
        'r2': 0.9972353989298621,
        'entropy_words': 10.837507018290683,
        'entropy_chars': 4.5802240680011925,
        'autocorr_lag1': 0.47901673707753845,
        'hurst': 0.8841964440665105
    }

metrics = ['slope','r2','entropy_words','entropy_chars','autocorr_lag1','hurst']
models = sorted(sim_df['model'].unique())

rows = []
import math

for m in metrics:
    for model in models:
        vals = sim_df[sim_df['model']==model][m].dropna().values.astype(float)
        n = len(vals)
        if n == 0:
            continue

        real = float(real_metrics[m])
        # Empirical p-value: two-sided and one-sided
        # Decide direction: if greater-than is more extreme or less-than:
        # we'll compute two-sided empirical p-value = min( fraction(vals >= real), fraction(vals <= real) ) * 2
        frac_ge = np.sum(vals >= real) / n
        frac_le = np.sum(vals <= real) / n
        # two-sided
        emp_p_two = min(frac_ge, frac_le) * 2
        # one-sided (greater)
        emp_p_greater = frac_ge
        emp_p_less = frac_le

        # parametric t-test (assumes approx normal) — compare sim distribution to the single observed value:
        # perform one-sample t-test: H0 mean(vals) = real
        # Equivalent: compute t = (mean(vals)-real) / (std/sqrt(n)) ; p-value two-sided
        t_stat = (np.mean(vals) - real) / (np.std(vals, ddof=1) / math.sqrt(n)) if np.std(vals, ddof=1) > 0 else float('nan')
        t_df = n-1
        t_p_two = 2 * (1 - stats.t.cdf(abs(t_stat), df=t_df)) if not math.isnan(t_stat) else float('nan')

        # Mann-Whitney U test vs point-value approximate: create array with the single real value repeated k times? Not appropriate.
        # Instead compute KS test between sim distribution and delta at real: not meaningful.
        # Better: compare sim distribution to distribution of real via resampling: compute z-score
        sim_mean = float(np.mean(vals))
        sim_std = float(np.std(vals, ddof=1))
        z_score = (real - sim_mean) / (sim_std) if sim_std > 0 else float('nan')

        # KS test against normal fitted to sims (to test if real lies in same distribution)
        # We'll perform a one-sample z-test style via comparing CDF of normal fit
        # Or perform empirical CDF p: fraction(vals <= real) etc already given.

        # For more robust nonparametric test: compare distribution of sims of this model vs distribution of sims of 'char_shuffle' or other using KS/MW if needed.

        row = {
            'metric': m,
            'model': model,
            'n_sims': int(n),
            'sim_mean': sim_mean,
            'sim_std': sim_std,
            'real_value': real,
            'emp_p_ge': float(emp_p_greater),
            'emp_p_le': float(emp_p_less),
            'emp_p_two_sided': float(emp_p_two),
            't_stat_vs_real': float(t_stat) if not math.isnan(t_stat) else None,
            't_p_two_sided': float(t_p_two) if not math.isnan(t_p_two) else None,
            'z_vs_sim_mean': float(z_score) if not math.isnan(z_score) else None
        }
        rows.append(row)

# DataFrame and save
out_df = pd.DataFrame(rows)
out_df.to_csv(NM / 'pvalues_and_tests.csv', index=False, encoding='utf-8')
print("Saved pvalues_and_tests.csv ->", NM / 'pvalues_and_tests.csv')

# Also produce pairwise KS / Mann-Whitney between models per metric (optional)
pair_rows = []
for m in metrics:
    for i, a in enumerate(models):
        for b in models[i+1:]:
            va = sim_df[sim_df['model']==a][m].dropna().values
            vb = sim_df[sim_df['model']==b][m].dropna().values
            if len(va)>0 and len(vb)>0:
                ks_stat, ks_p = stats.ks_2samp(va, vb)
                mw_stat, mw_p = stats.mannwhitneyu(va, vb, alternative='two-sided')
                pair_rows.append({
                    'metric': m,
                    'model_a': a,
                    'model_b': b,
                    'ks_stat': float(ks_stat),
                    'ks_p': float(ks_p),
                    'mw_stat': float(mw_stat),
                    'mw_p': float(mw_p)
                })
pd.DataFrame(pair_rows).to_csv(NM / 'pairwise_model_tests.csv', index=False, encoding='utf-8')
print("Saved pairwise_model_tests.csv ->", NM / 'pairwise_model_tests.csv')

# summary JSON
summary = {
    'n_models': len(models),
    'metrics_tested': metrics,
    'n_rows_pvalues': len(rows)
}
with open(NM / 'pvalues_summary.json', 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print("Done. See results in:", NM)
