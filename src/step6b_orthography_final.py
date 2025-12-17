#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone PART 3 fixed (safe parsing of diac_counts).
Reads Part1+Part2 outputs from results/orthography and produces the final visuals & summary.
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import ast
import re

ROOT = Path('.')
OUT = ROOT / "final_results" / "step6_orthography"
OUT.mkdir(parents=True, exist_ok=True)

print("Loading previously generated files...")

# Helper to safely parse the diac_counts field stored as string
def safe_parse_counts(obj):
    """
    Accepts:
     - dict object -> return as-is
     - string like "Counter({...})" -> extract {...} and literal_eval
     - string like "{'a':1,'b':2}" -> literal_eval
     - JSON string -> json.loads
    Returns dict or empty dict on failure.
    """
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if not isinstance(obj, str):
        return {}
    s = obj.strip()
    # if starts with "Counter(" remove wrapper
    if s.startswith("Counter(") and s.endswith(")"):
        inner = s[len("Counter("):-1].strip()
        s = inner
    # try literal_eval (safe)
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, dict):
            # keys might be strings representing codepoints; keep as-is
            return {str(k): int(v) for k,v in parsed.items()}
    except Exception:
        pass
    # try json
    try:
        parsed = json.loads(s)
        if isinstance(parsed, dict):
            return {str(k): int(v) for k,v in parsed.items()}
    except Exception:
        pass
    # fallback: attempt to parse simple "'ch':num" pairs with regex
    try:
        pairs = re.findall(r"['\"]([^'\"]+)['\"]\s*:\s*([0-9]+)", s)
        if pairs:
            return {k:int(v) for k,v in pairs}
    except Exception:
        pass
    return {}

# Load CSVs/JSON from PART1+PART2 with graceful error handling
def load_csv_safe(p):
    if p.exists():
        try:
            return pd.read_csv(p, encoding='utf-8')
        except Exception as e:
            print(f"Warning: failed to read {p}: {e}")
    else:
        print(f"Warning: {p} not found.")
    return None

verses = load_csv_safe(OUT/"verses_basic_summary.csv")
sura_sum = load_csv_safe(OUT/"sura_diac_summary.csv")
diac_df = load_csv_safe(OUT/"global_diac_counts.csv")
freq_by_norm = load_csv_safe(OUT/"freq_norm_tokens.csv")
orth_pairs_df = load_csv_safe(OUT/"orth_variant_pairs.csv")

# Load JSON results
def load_json_safe(p):
    if p.exists():
        try:
            return json.loads(p.read_text(encoding='utf-8'))
        except Exception as e:
            print(f"Warning: failed to parse {p}: {e}")
    else:
        print(f"Warning: {p} not found.")
    return None

part2 = load_json_safe(OUT/"orthography_partial2.json") or {}
perm_json = load_json_safe(OUT/"diac_shuffle_perm_test.json") or {}
markov_json = load_json_safe(OUT/"markov_summary.json") or {}
boot_json = load_json_safe(OUT/"bootstrap_summary.json") or {}

# If any critical file missing: exit with message
if verses is None or sura_sum is None or diac_df is None:
    raise SystemExit("Required input files missing in results/orthography. Run Part1+Part2 first.")

# Ensure numeric types
verses["n_diac"] = pd.to_numeric(verses["n_diac"], errors='coerce').fillna(0).astype(float)
verses["n_words"] = pd.to_numeric(verses["n_words"], errors='coerce').fillna(0).astype(float)
verses["n_chars_no_diac"] = pd.to_numeric(verses["n_chars_no_diac"], errors='coerce').fillna(0).astype(float)
verses["diac_per_word"] = verses["n_diac"] / (verses["n_words"] + 1e-9)

# parse diac_counts column safely into list of dicts
if "diac_counts" in verses.columns:
    parsed_counts = []
    for v in verses["diac_counts"].fillna("").tolist():
        d = safe_parse_counts(v)
        parsed_counts.append(d)
    verses["_diac_counts_parsed"] = parsed_counts
else:
    # try reading from file that might contain a different column
    verses["_diac_counts_parsed"] = [{} for _ in range(len(verses))]

perm_stats = None
perm_stats_path = OUT / "perm_stats.npy"
if perm_stats_path.exists():
    try:
        perm_stats = np.load(perm_stats_path)
    except Exception:
        perm_stats = None

markov_stats = None
markov_path = OUT / "markov_diac_stats.txt"
if markov_path.exists():
    try:
        arr = np.loadtxt(markov_path)
        markov_stats = arr if arr.size>0 else None
    except Exception:
        markov_stats = None

# Load other summary numbers
obs_stat = perm_json.get("obs_stat", part2.get("obs_stat_var_diac_per_sura") or perm_json.get("obs_stat") or 0.0)
p_emp = perm_json.get("p_emp", part2.get("perm_p_emp", None))
markov_pemp = markov_json.get("markov_pemp_vs_obs", None)

print("Loaded inputs. Proceeding with visualizations.")

# ---------------- Heatmap top diacritics per sura ----------------
top_diac = diac_df.head(10)["diac"].tolist() if diac_df is not None else []
heat_rows = []
for _, row in sura_sum.iterrows():
    s = int(row["sura"])
    block_idx = verses["sura"] == s
    block = verses[block_idx]
    counts = {d:0 for d in top_diac}
    # use parsed diac_counts
    for dct in block["_diac_counts_parsed"].tolist():
        if isinstance(dct, dict):
            for d in top_diac:
                counts[d] += int(dct.get(d, 0))
    denom = max(1, int(block["n_chars_no_diac"].sum()))
    rowvals = [counts[d] / denom * 1000.0 for d in top_diac]
    heat_rows.append([s] + rowvals)

heat_df = pd.DataFrame(heat_rows, columns=["sura"] + top_diac).set_index("sura")
plt.figure(figsize=(12,8))
sns.heatmap(heat_df, cmap="magma", cbar_kws={"label":"diacritics per 1000 chars_no_diac"})
plt.title("Heatmap: Top diacritics density per Surah")
plt.tight_layout()
plt.savefig(OUT/"standalone_heatmap_top10_diac.png", dpi=200)
plt.close()

# ---------------- Violin plot for representative suras ----------------
top_suras = sura_sum.sort_values("diac_per_char", ascending=False).head(10)["sura"].tolist()
bot_suras = sura_sum.sort_values("diac_per_char", ascending=True).head(10)["sura"].tolist()
sel_suras = top_suras + bot_suras
plot_df = verses[verses["sura"].isin(sel_suras)][["sura","diac_per_word"]].copy()
plot_df["sura"] = plot_df["sura"].astype(str)
plt.figure(figsize=(14,6))
sns.violinplot(data=plot_df, x="sura", y="diac_per_word")
plt.title("Violin plot of diac_per_word for selected suras")
plt.tight_layout()
plt.savefig(OUT/"standalone_violin_diac_per_word_selected.png", dpi=200)
plt.close()

# ---------------- Boxplot across all suras ----------------
plt.figure(figsize=(14,6))
sns.boxplot(data=verses, x="sura", y="diac_per_word")
plt.title("Boxplot of diac_per_word across all suras")
plt.xlabel("")
plt.xticks([],[])
plt.tight_layout()
plt.savefig(OUT/"standalone_boxplot_diac_all_suras.png", dpi=200)
plt.close()

# ---------------- PCA ----------------
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
feat_cols = ["n_chars_no_diac","n_diac","n_words","diac_per_word"]
feat_df = verses[feat_cols].fillna(0).astype(float)
Xs = StandardScaler().fit_transform(feat_df.values)
pca = PCA(n_components=2)
pcs = pca.fit_transform(Xs)
plt.figure(figsize=(7,6))
plt.scatter(pcs[:,0], pcs[:,1], s=6, alpha=0.4)
plt.title("PCA - Orthographic features (Standalone)")
plt.tight_layout()
plt.savefig(OUT/"standalone_pca.png", dpi=200)
plt.close()

# ---------------- Levenshtein distribution ----------------
if orth_pairs_df is not None and "lev_dist" in orth_pairs_df.columns:
    plt.figure(figsize=(6,4))
    sns.histplot(orth_pairs_df["lev_dist"].dropna().astype(int), bins=range(0, int(orth_pairs_df["lev_dist"].max())+2))
    plt.title("Levenshtein distance distribution for orthographic variant pairs")
    plt.tight_layout()
    plt.savefig(OUT/"standalone_levenshtein_dist.png", dpi=200)
    plt.close()

# ---------------- KS / Mann-Whitney between top/bottom suras ----------------
top_group = verses[verses["sura"].isin(top_suras)]["diac_per_word"].dropna().values
bot_group = verses[verses["sura"].isin(bot_suras)]["diac_per_word"].dropna().values
if len(top_group)>0 and len(bot_group)>0:
    ks_stat, ks_p = stats.ks_2samp(top_group, bot_group)
    mw_stat, mw_p = stats.mannwhitneyu(top_group, bot_group, alternative="two-sided")
else:
    ks_stat=ks_p=mw_stat=mw_p=None

with open(OUT/"standalone_pairwise_tests.json","w",encoding="utf-8") as f:
    json.dump({"top_suras": top_suras, "bot_suras": bot_suras, "ks_stat": ks_stat, "ks_p": ks_p, "mw_stat": mw_stat, "mw_p": mw_p}, f, ensure_ascii=False, indent=2)

# ---------------- Final summary ----------------
final_summary = {
    "status": "standalone ",
    "obs_stat": float(obs_stat) if obs_stat is not None else None,
    "perm_p_emp": float(p_emp) if p_emp is not None else None,
    "markov_pemp": float(markov_pemp) if markov_pemp is not None else None,
    "ks_p": float(ks_p) if ks_p is not None else None,
    "mw_p": float(mw_p) if mw_p is not None else None
}
with open(OUT/"section6_ready_for_paper.json","w",encoding="utf-8") as f:
    json.dump(final_summary, f, ensure_ascii=False, indent=2)

print("Standalone PART 3. Outputs written to:", OUT)
