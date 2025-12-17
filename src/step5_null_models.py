# null_models_and_comparisons.py
"""
Null models and statistical comparisons for quran-stats project.

Generates:
  - char_shuffle model (shuffle characters preserving global char counts)
  - word_shuffle_preserve_lengths model (shuffle words but preserve per-verse word-length multiset)
  - markov_word_model (1st-order Markov on words)
  - poisson_wordcount_model (simulate n_words per verse with Poisson from empirical mean)

For each model:
  - produce N simulations (default N=30)
  - compute metrics per simulation: zipf slope & R2 (middle region), entropy words/chars, autocorr lag1, hurst
  - save results and comparison plots.

Inputs (from results/):
  - results/word_freq.csv
  - results/verse_metrics.csv
  - results/verses_structured.csv
  - results/quran_full_text.txt

Outputs (to results/null_models/):
  - simulated texts (.txt), metrics CSV per simulation, aggregated comparisons, PNG plots.
"""

import os
from pathlib import Path
import random
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from collections import Counter, defaultdict
from scipy import stats
import seaborn as sns

# ---------------- CONFIG ----------------
ROOT = Path('.')
RESULTS = ROOT / 'final_results'
OUT = RESULTS / 'step5_null_models'
OUT.mkdir(parents=True, exist_ok=True)

N_SIM = 30  # number of simulations per model (adjustable)
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# files
WORD_FREQ = RESULTS / 'step1_preprocess' / 'word_freq.csv'
VERSE_METRICS = RESULTS / 'step1_preprocess' / 'verse_metrics.csv'
VERSES_STRUCT = RESULTS / 'step1_preprocess' / 'verses_structured.csv'
FULL_TEXT = RESULTS / 'step1_preprocess' / 'quran_full_text.txt'

# metrics function utilities
def compute_zipf_slope_r2_from_word_counts(word_counts, lo_rank_exclude=20, hi_rank_limit=2000):
    """
    word_counts: array-like of counts sorted descending
    returns slope, intercept, r2 for fit on log-log between ranks lo..hi
    """
    counts = np.array(word_counts)
    ranks = np.arange(1, len(counts) + 1)
    # avoid zeros
    mask = counts > 0
    counts = counts[mask]
    ranks = ranks[mask]
    lo = lo_rank_exclude
    hi = min(hi_rank_limit, len(counts))
    if hi - lo < 10:
        return np.nan, np.nan, np.nan
    x = np.log(ranks[lo:hi]).reshape(-1,1)
    y = np.log(counts[lo:hi])
    model = LinearRegression().fit(x,y)
    slope = float(model.coef_[0])
    intercept = float(model.intercept_)
    r2 = float(model.score(x,y))
    return slope, intercept, r2

def shannon_entropy_from_counts(counts):
    p = np.array(counts) / float(np.sum(counts))
    p = p[p>0]
    return -np.sum(p * np.log2(p))

def autocorr_lag1(series):
    x = np.array(series)
    if len(x) < 2:
        return np.nan
    return float(np.corrcoef(x[:-1], x[1:])[0,1])

def hurst_exponent(ts):
    # simple R/S estimator using multiple window sizes
    ts = np.array(ts, dtype=float)
    N = len(ts)
    if N < 20:
        return np.nan
    sizes = np.unique(np.floor(np.logspace(np.log10(10), np.log10(N//4), num=10)).astype(int))
    rs = []
    ns = []
    for n in sizes:
        n = int(n)
        if n < 10: 
            continue
        # split into k segments of length n
        k = N // n
        if k < 2:
            continue
        rms = []
        for i in range(k):
            seg = ts[i*n:(i+1)*n]
            Y = np.cumsum(seg - np.mean(seg))
            R = np.max(Y) - np.min(Y)
            S = np.std(seg, ddof=1)
            if S>0:
                rms.append(R/S)
        if len(rms)>0:
            rs.append(np.mean(rms))
            ns.append(n)
    if len(rs) < 2:
        return np.nan
    lr = np.polyfit(np.log(ns), np.log(rs), 1)
    return float(lr[0])

# ---------------- Load data ----------------
print("Loading data...")
word_df = pd.read_csv(WORD_FREQ, encoding='utf-8')
verse_df = pd.read_csv(VERSE_METRICS, encoding='utf-8')
verses_df = pd.read_csv(VERSES_STRUCT, encoding='utf-8')
with open(FULL_TEXT, 'r', encoding='utf-8') as f:
    full_text = f.read()

# prepare tokens per verse
import re
token_re = re.compile(r'[\u0621-\u064A\u0660-\u0669]+')
tokens_by_verse = []
all_tokens = []
for _, row in verses_df.iterrows():
    text = str(row['text'])
    toks = token_re.findall(text)
    tokens_by_verse.append(toks)
    all_tokens.extend(toks)

print("Total tokens (words):", len(all_tokens))
v_word_counts = [len(t) for t in tokens_by_verse]

# global word list and counts
global_word_list = [w for w in all_tokens]
global_word_counts = Counter(global_word_list)
vocab = list(global_word_counts.keys())

# helper: write simulated text as sura|aya|text lines preserving structure (same sura/aya order)
sura_aya_pairs = list(zip(verses_df['sura'].astype(int).tolist(), verses_df['aya'].astype(int).tolist()))

def save_sim_text(lines, outpath):
    with open(outpath, 'w', encoding='utf-8') as f:
        for (s,a), text in zip(sura_aya_pairs, lines):
            f.write(f"{s}|{a}|{text}\n")

# compute real metrics for reference
print("Computing real metrics...")
real_word_counts = word_df['count'].values
real_slope, real_int, real_r2 = compute_zipf_slope_r2_from_word_counts(real_word_counts)
real_entropy_words = shannon_entropy_from_counts(real_word_counts)
char_counts_df = pd.read_csv(RESULTS / 'step1_preprocess' / 'char_freq_without_diacritics.csv', encoding='utf-8')
real_entropy_chars = shannon_entropy_from_counts(char_counts_df['count'].values)
real_autocorr = autocorr_lag1(verse_df['n_words'].values)
real_hurst = hurst_exponent(verse_df['n_words'].values)

ref_summary = {
    'zipf_slope': real_slope,
    'zipf_r2': real_r2,
    'entropy_words': real_entropy_words,
    'entropy_chars': real_entropy_chars,
    'autocorr_lag1': real_autocorr,
    'hurst': real_hurst
}
with open(OUT / 'real_metrics.json', 'w', encoding='utf-8') as f:
    json.dump(ref_summary, f, ensure_ascii=False, indent=2)

print("Real metrics:", ref_summary)

# ---------------- Model 1: char_shuffle ----------------
def model_char_shuffle():
    chars = [c for c in full_text if not c.isspace()]
    random.shuffle(chars)
    # rebuild verses by lengths of original verses (by n_chars_no_diac if available else n_chars)
    lengths = verse_df['n_chars_no_diac'].values if 'n_chars_no_diac' in verse_df.columns else verse_df['n_chars'].values
    lines = []
    idx = 0
    for L in lengths:
        seg = ''.join(chars[idx:idx+int(L)])
        idx += int(L)
        lines.append(seg)
    return lines

# ---------------- Model 2: word_shuffle_preserve_lengths (shuffle words globally but preserve per-verse word counts) ----------
def model_word_shuffle_preserve_counts():
    flat = list(global_word_list)
    random.shuffle(flat)
    lines = []
    idx = 0
    for cnt in v_word_counts:
        seg_words = flat[idx:idx+cnt]
        idx += cnt
        lines.append(' '.join(seg_words))
    return lines

# ---------------- Model 3: markov_word_model (1st order) ----------------
def build_word_markov():
    transitions = defaultdict(list)
    for toks in tokens_by_verse:
        for i in range(len(toks)-1):
            transitions[toks[i]].append(toks[i+1])
    # fallback to random vocab
    return transitions

def model_markov_words(transitions):
    lines = []
    vocab_keys = list(transitions.keys())
    for cnt in v_word_counts:
        if cnt == 0:
            lines.append('')
            continue
        # choose random start
        w = random.choice(vocab_keys)
        seq = [w]
        for _ in range(cnt-1):
            nxts = transitions.get(w)
            if nxts and len(nxts)>0:
                w = random.choice(nxts)
            else:
                w = random.choice(vocab_keys)
            seq.append(w)
        lines.append(' '.join(seq))
    return lines

# ---------------- Model 4: poisson_wordcount_model (simulate word counts per verse) ----------
def model_poisson_wordcount_simulate():
    mu = np.mean(v_word_counts)
    lines = []
    # sample word counts using Poisson around mu; then fill words randomly
    for _ in range(len(v_word_counts)):
        cnt = int(np.random.poisson(mu))
        if cnt < 0: cnt = 0
        seq = [random.choice(global_word_list) for _ in range(cnt)]
        lines.append(' '.join(seq))
    return lines

# ---------------- Evaluation pipeline ----------------
def evaluate_simulated_lines(lines, sim_name, sim_i):
    # save text
    fname = OUT / f"{sim_name}_sim_{sim_i:03d}.txt"
    save_sim_text(lines, fname)
    # compute word freq
    toks_all = []
    for line in lines:
        toks = token_re.findall(line)
        toks_all.extend(toks)
    wc = Counter(toks_all)
    word_counts_sorted = np.array([c for _,c in sorted(wc.items(), key=lambda x: -x[1])])
    # compute metrics
    slope, intercept, r2 = compute_zipf_slope_r2_from_word_counts(word_counts_sorted)
    ent_words = shannon_entropy_from_counts(word_counts_sorted) if word_counts_sorted.sum()>0 else np.nan
    # char counts
    chars = [c for line in lines for c in line if not c.isspace()]
    cc = Counter(chars)
    ent_chars = shannon_entropy_from_counts(np.array(list(cc.values()))) if sum(cc.values())>0 else np.nan
    # build verse-level n_words series
    verse_n_words = [len(token_re.findall(line)) for line in lines]
    ac1 = autocorr_lag1(verse_n_words)
    hurstv = hurst_exponent(verse_n_words)
    return {
        'sim_file': str(fname.name),
        'slope': slope,
        'r2': r2,
        'entropy_words': ent_words,
        'entropy_chars': ent_chars,
        'autocorr_lag1': ac1,
        'hurst': hurstv
    }

# run simulations
models = [
    ('char_shuffle', model_char_shuffle),
    ('word_shuffle', model_word_shuffle_preserve_counts),
    ('markov_word', None),  # will build transitions then call
    ('poisson_wordcounts', model_poisson_wordcount_simulate)
]

# prebuild markov transitions
markov_trans = build_word_markov()

all_results = []

for name, func in models:
    print("Running model:", name)
    model_out_dir = OUT / name
    model_out_dir.mkdir(exist_ok=True)
    for i in range(N_SIM):
        if name == 'markov_word':
            lines = model_markov_words(markov_trans)
        else:
            lines = func()
        res = evaluate_simulated_lines(lines, name, i+1)
        res['model'] = name
        res['sim'] = i+1
        all_results.append(res)
        # save per-sim metrics
        pd.DataFrame([res]).to_csv(model_out_dir / f"metrics_sim_{i+1:03d}.csv", index=False, encoding='utf-8')

# aggregate results
res_df = pd.DataFrame(all_results)
res_df.to_csv(OUT / 'all_simulations_metrics.csv', index=False, encoding='utf-8')

# ---------------- Compare distributions and plot ----------------

metrics = ['slope','r2','entropy_words','entropy_chars','autocorr_lag1','hurst']
summary_rows = []

for m in metrics:
    plt.figure(figsize=(9,5))

    plot_data = []
    plot_labels = []

    for name, _ in models:
        vals = res_df[res_df['model'] == name][m].dropna().values
        
        if len(vals) == 0:
            continue
        
        # Save summary
        summary_rows.append({
            'metric': m,
            'model': name,
            'mean': float(np.nanmean(vals)),
            'std': float(np.nanstd(vals)),
            'median': float(np.nanmedian(vals)),
            'n': int(len(vals))
        })

        # Add to plotting dataset
        plot_data.extend(vals)
        plot_labels.extend([name] * len(vals))

    # Build DataFrame for seaborn
    box_df = pd.DataFrame({'model': plot_labels, m: plot_data})

    # Plot boxplot or violinplot
    sns.boxplot(data=box_df, x='model', y=m)
    plt.xticks(rotation=45)
    plt.title(f"Comparison of {m} across models (Boxplot)")
    
    # Add real value
    real_val = ref_summary.get(m)
    if real_val is not None:
        plt.axhline(real_val, color='red', linestyle='--', linewidth=2, label='real')
        plt.legend()

    plt.tight_layout()
    plt.savefig(OUT / f"cmp_{m}.png")
    plt.close()

pd.DataFrame(summary_rows).to_csv(OUT / 'models_metrics_summary.csv', index=False, encoding='utf-8')

# Save aggregated res and summary
with open(OUT / 'null_models_config.json', 'w', encoding='utf-8') as f:
    json.dump({'N_SIM': N_SIM, 'seed': RANDOM_SEED}, f, ensure_ascii=False, indent=2)

print("Null models simulations complete. Results in:", OUT)
