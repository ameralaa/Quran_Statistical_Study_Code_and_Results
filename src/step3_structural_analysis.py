# structural_analysis.py
"""
Structural & Symmetry Analysis for quran-stats project
Outputs: results/structural/* (CSVs + PNGs)
Requirements: pandas, numpy, matplotlib, seaborn, scipy, scikit-learn
Run: python structural_analysis.py
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist, squareform
from math import log
import json
import warnings
warnings.filterwarnings("ignore")

ROOT = Path('.')
RESULTS = ROOT / 'final_results'
STRUCT_DIR = RESULTS / 'step3_structural'
STRUCT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Load data ----------
verse_metrics_path = RESULTS / 'step1_preprocess' / 'verse_metrics.csv'    # generated earlier
verses_structured_path = RESULTS / 'step1_preprocess' / 'verses_structured.csv'  # sura,aya,text
word_freq_path = RESULTS / 'step1_preprocess' / 'word_freq.csv'  # word,count

if not verse_metrics_path.exists() or not verses_structured_path.exists():
    raise FileNotFoundError("Expected results/verse_metrics.csv and results/verses_structured.csv. Run preprocessing first.")

verse_df = pd.read_csv(verse_metrics_path, encoding='utf-8')
verses_df = pd.read_csv(verses_structured_path, encoding='utf-8')

# ---------- 1) Sura-level metrics ----------
# compute for each sura:
# - n_verses
# - total_chars, mean_chars_per_verse
# - total_words, mean_words_per_verse
sura_stats = verse_df.groupby('sura').agg(
    n_verses=('aya','count'),
    total_chars=('n_chars','sum'),
    mean_chars_per_verse=('n_chars','mean'),
    total_chars_no_diac=('n_chars_no_diac','sum') if 'n_chars_no_diac' in verse_df.columns else ('n_chars','sum'),
    total_words=('n_words','sum'),
    mean_words_per_verse=('n_words','mean')
).reset_index().sort_values('sura')

sura_stats.to_csv(STRUCT_DIR / 'sura_metrics.csv', index=False, encoding='utf-8')

# Plot: total words per sura (bar)
plt.figure(figsize=(14,5))
plt.bar(sura_stats['sura'], sura_stats['total_words'])
plt.xlabel('Sura')
plt.ylabel('Total words')
plt.title('Total words per Sura')
plt.tight_layout()
plt.savefig(STRUCT_DIR / 'sura_total_words_bar.png', dpi=150)
plt.close()

# ---------- 2) Verse length profile (global) ----------
plt.figure(figsize=(12,4))
plt.plot(verse_df['n_words'].values, linewidth=0.6)
plt.xlabel('Verse index (ordered by file)')
plt.ylabel('Words per verse')
plt.title('Words per verse (sequence)')
plt.tight_layout()
plt.savefig(STRUCT_DIR / 'verse_words_sequence.png', dpi=150)
plt.close()

# ---------- 3) Sura × Sura correlation matrix (based on verse-length distributions) ----------
# Build matrix: for each sura, vector of verse lengths padded/truncated to same length (use median length or max length)
max_len = verse_df.groupby('sura')['aya'].count().max()
sura_vectors = {}
for s, group in verse_df.groupby('sura'):
    arr = group['n_words'].values
    # pad with mean of that sura to reach max_len (so vectors comparable)
    if len(arr) < max_len:
        pad = np.full(max_len - len(arr), arr.mean() if len(arr)>0 else 0)
        vec = np.concatenate([arr, pad])
    else:
        vec = arr[:max_len]
    sura_vectors[s] = vec

# Build DataFrame
suras = sorted(sura_vectors.keys())
matrix = np.vstack([sura_vectors[s] for s in suras])
corr = np.corrcoef(matrix)

corr_df = pd.DataFrame(corr, index=suras, columns=suras)
corr_df.to_csv(STRUCT_DIR / 'sura_length_correlation_matrix.csv', encoding='utf-8')

# Heatmap
plt.figure(figsize=(10,8))
sns.heatmap(corr_df, cmap='vlag', center=0, xticklabels=10, yticklabels=10)
plt.title('Sura×Sura Pearson correlation (verse-length profile)')
plt.tight_layout()
plt.savefig(STRUCT_DIR / 'sura_length_corr_heatmap.png', dpi=150)
plt.close()

# ---------- 4) Cosine similarity between suras using word-frequency vectors ----------
# Build per-sura word frequency vectors (top N words to limit memory)
word_df = pd.read_csv(word_freq_path, encoding='utf-8')
topN = 2000
top_words = word_df['word'].head(topN).tolist()

# initialize matrix: rows = suras, cols = top_words
sura_word_mat = np.zeros((len(suras), len(top_words)), dtype=float)
sura_index = {s: i for i,s in enumerate(suras)}
# Tokenize each verse's no-diac text if available in verses_df
# we have tokens in verse_metrics? not, so use verses_structured and simple tokenization
import re
token_re = re.compile(r'[\u0621-\u064A\u0660-\u0669]+')

for _, row in verses_df.iterrows():
    s = int(row['sura'])
    text = str(row['text'])
    # remove diacritics roughly (strip combining marks)
    # simple: keep only letters and spaces, then tokenize
    tokens = token_re.findall(text)
    idx = sura_index[s]
    for t in tokens:
        if t in top_words:
            j = top_words.index(t)
            sura_word_mat[idx, j] += 1

sura_word_df = pd.DataFrame(sura_word_mat, index=suras, columns=top_words)
sura_word_df.to_csv(STRUCT_DIR / 'sura_wordfreq_topN.csv', encoding='utf-8')

# Cosine similarity
cos_sim = cosine_similarity(sura_word_mat)
cos_df = pd.DataFrame(cos_sim, index=suras, columns=suras)
cos_df.to_csv(STRUCT_DIR / 'sura_word_cosine_similarity.csv', encoding='utf-8')

plt.figure(figsize=(10,8))
sns.heatmap(cos_df, cmap='viridis', vmin=0, vmax=1, xticklabels=10, yticklabels=10)
plt.title('Sura×Sura Cosine similarity (top words)')
plt.tight_layout()
plt.savefig(STRUCT_DIR / 'sura_word_cosine_heatmap.png', dpi=150)
plt.close()

# ---------- 5) Hierarchical clustering (dendrogram) on cosine distance ----------
dist = 1 - cos_sim
# convert to condensed form
condensed = squareform(dist, checks=False)
Z = linkage(condensed, method='average')
plt.figure(figsize=(12,6))
dendrogram(Z, labels=suras, leaf_rotation=90, leaf_font_size=6, color_threshold=None)
plt.title('Hierarchical clustering of Suras (based on cosine similarity of top words)')
plt.tight_layout()
plt.savefig(STRUCT_DIR / 'sura_dendrogram.png', dpi=150)
plt.close()

# ---------- 6) Palindromic / mirror-like pattern basic scan ----------
# We'll search for palindromic substrings of length >= L within verses (on no-diac text)
def strip_diacritics_simple(text):
    # remove Arabic diacritic blocks (basic)
    import regex as re2
    # Unicode combining marks block approx: \p{M}
    return re2.sub(r'\p{M}', '', text)

# Use verses_df text and scan for palindromic word sequences within an ayah (word-level)
pal_records = []
for _, row in verses_df.iterrows():
    s = int(row['sura']); a = int(row['aya']); text = str(row['text'])
    txt = strip_diacritics_simple(text)
    tokens = token_re.findall(txt)
    n = len(tokens)
    min_len = 3
    for i in range(n):
        for j in range(i+min_len, n):
            seq = tokens[i:j+1]
            if seq == seq[::-1]:
                pal_records.append({'sura': s, 'aya': a, 'start': i, 'end': j, 'length': len(seq), 'sequence': ' '.join(seq)})
# Save top palindromic sequences (long)
pal_df = pd.DataFrame(pal_records)
if not pal_df.empty:
    pal_df.sort_values('length', ascending=False).to_csv(STRUCT_DIR / 'palindromic_sequences_by_verse.csv', index=False, encoding='utf-8')

# ---------- 7) Box-counting (approx) to estimate fractal dimension on global verse-length signal ----------
# We'll apply a simple box-counting on the binary series after thresholding (multiple scales)
def box_count(signal, scale):
    # signal: 1D numeric array -> convert to binary presence above threshold of mean
    N = len(signal)
    step = int(scale)
    if step < 1: step = 1
    counts = 0
    for i in range(0, N, step):
        block = signal[i:i+step]
        if np.any(block):
            counts += 1
    return counts

# prepare binary signal: normalized absolute deviation from mean > t*std
sig = verse_df['n_words'].values
sig_norm = (sig - np.mean(sig)) / (np.std(sig) + 1e-9)
binary = np.abs(sig_norm) > 0.1  # threshold small to keep structure

scales = np.unique(np.logspace(0, np.log10(len(binary)//4 + 1), num=20, dtype=int))
scales = [s for s in scales if s>0]
counts = []
for s in scales:
    counts.append(box_count(binary, s))

# fit line log(counts) ~ -D * log(scale)
log_s = np.log(scales)
log_c = np.log(counts)
# filter zeros
mask = np.isfinite(log_c) & (log_c>0)
if mask.sum() >= 2:
    slope, intercept = np.polyfit(log_s[mask], log_c[mask], 1)
    fractal_dim = -slope
else:
    fractal_dim = float('nan')

# save boxcount results and plot
bc_df = pd.DataFrame({'scale': scales, 'counts': counts})
bc_df.to_csv(STRUCT_DIR / 'boxcount_results.csv', index=False)

plt.figure(figsize=(6,4))
plt.plot(log_s, log_c, 'o-')
plt.xlabel('log(scale)')
plt.ylabel('log(count)')
plt.title(f'Box-counting (approx). est. fractal_dim={fractal_dim:.3f}')
plt.tight_layout()
plt.savefig(STRUCT_DIR / 'boxcount_plot.png', dpi=150)
plt.close()

# ---------- 8) Save summary JSON ----------
summary_out = {
    'n_suras': len(suras),
    'sura_stats_path': str(STRUCT_DIR / 'sura_metrics.csv'),
    'corr_matrix_path': str(STRUCT_DIR / 'sura_length_correlation_matrix.csv'),
    'cosine_sim_path': str(STRUCT_DIR / 'sura_word_cosine_similarity.csv'),
    'fractal_dim_estimate': float(fractal_dim) if not np.isnan(fractal_dim) else None
}
with open(STRUCT_DIR / 'structural_summary.json', 'w', encoding='utf-8') as f:
    json.dump(summary_out, f, ensure_ascii=False, indent=2)

print("Structural analysis done. Outputs in:", STRUCT_DIR)
