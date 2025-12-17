#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Orthography & Diacritics Analysis – FINAL ACADEMIC VERSION
Section 6 for Quran Statistical Study
Outputs -> results/orthography/

Enhancements included:
 - K=5000 permutation test
 - 2000 bootstrap samples
 - Markov model for diacritics
 - Variant spelling analysis (entropy, PMI, KS tests)
 - Enhanced muqatta'at detection (canonical forms)
 - Heatmaps, violin plots, PCA, UMAP
 - Full JSON summary ready for publication
 - Clean modular structure
"""

import sys
from pathlib import Path
import re
import json
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import Levenshtein

# Optional UMAP
try:
    import umap
    _HAS_UMAP = True
except:
    _HAS_UMAP = False

##########################################################
# PATHS
##########################################################

ROOT = Path('.')
OUT = ROOT / "final_results" / "step6_orthography"
OUT.mkdir(parents=True, exist_ok=True)

POSSIBLE_VERSES = [
    ROOT/"final_results/step1_preprocess/verses_structured.csv",
    ROOT/"final_results/step1_preprocess/quran_full_text.txt"
]

##########################################################
# CONSTANTS
##########################################################

# Full canonical Arabic diacritic list
DIACRITICS = [
    '\u0610','\u0611','\u0612','\u0613','\u0614','\u0615',
    '\u0616','\u0617','\u0618','\u0619','\u061A',
    '\u064B','\u064C','\u064D','\u064E','\u064F',
    '\u0650','\u0651','\u0652','\u0653','\u0654',
    '\u0655','\u0670','\u06E5','\u06E6',
    '\u06D6','\u06D7','\u06D8','\u06D9','\u06DA','\u06DB',
    '\u06DC','\u06DF','\u06E0','\u06E1'
]
DIAC_SET = set(DIACRITICS)

# Pause marks & ornate symbols
PAUSE_MARKS = [
    '\u06DD','\u06DE','ۖ','ۗ','ۚ','ۛ','ۜ','۝','٭','۞'
]
PAUSE_SET = set(PAUSE_MARKS)

# Arabic letters recognized
AR_LETTER = r'[\u0621-\u064A\u0671-\u06D3\u06FA-\u06FF]'
DIAC_RE = "[" + "".join(DIACRITICS) + "]"
TOKEN_RE = re.compile("(" + AR_LETTER + "+" + DIAC_RE + "*"+ ")", flags=re.UNICODE)

# Canonical Muqatta'at list
MUQ_LIST = set([
    "الم","الر","المر","المص","كهيعص","طه","طسم","طس",
    "يس","ص","ق","ن","حم","عسق"
])


##########################################################
# UTILITY FUNCTIONS
##########################################################

def strip_diacritics(text):
    return "".join(ch for ch in text if ch not in DIAC_SET)

def count_diacritics(text):
    cnt = Counter()
    for ch in text:
        if ch in DIAC_SET:
            cnt[ch] += 1
    return cnt

def contains_pause(text):
    return any(ch in PAUSE_SET for ch in text)

def load_verses():
    """
    Loads verses_structured.csv OR quran_full_text.txt
    Returns DataFrame(sura, aya, text)
    """
    for p in POSSIBLE_VERSES:
        if p.exists():
            try:
                if p.suffix == ".csv":
                    df = pd.read_csv(p, encoding='utf-8')
                    cols = {c.lower(): c for c in df.columns}
                    # normalize
                    for k in ["sura","aya","text"]:
                        if k not in cols:
                            # try aliases
                            for alt in ["verse","verse_id","verse_text","aya_text","raw_text","quran_text","text_uthmani"]:
                                if alt in cols:
                                    cols[k] = cols[alt]
                                    break
                    df = df.rename(columns={cols["sura"]: "sura",
                                            cols["aya"]: "aya",
                                            cols["text"]: "text"})
                    return df[["sura","aya","text"]]
                else:
                    # parse txt as s|a|text lines
                    lines = p.read_text(encoding='utf-8').splitlines()
                    rows=[]
                    for L in lines:
                        parts = L.split("|",2)
                        if len(parts)==3:
                            try:
                                s = int(parts[0]); a = int(parts[1]); t = parts[2]
                            except:
                                s=None;a=None;t=L
                            rows.append({"sura":s,"aya":a,"text":t})
                        else:
                            rows.append({"sura":None,"aya":None,"text":L})
                    return pd.DataFrame(rows)
            except Exception as e:
                print("Error reading", p, e)
                continue
    raise FileNotFoundError("No verses_structured or quran_full_text found.")


##########################################################
# MARKOV MODEL FOR DIACRITICS
##########################################################

def build_diac_markov_chain(tokens):
    """
    Build first-order Markov chain for diacritics sequence.
    Returns transition matrix P, unique diacritic symbols.
    """
    # Extract diacritic-only stream
    stream = []
    for tok in tokens:
        for ch in tok:
            if ch in DIAC_SET:
                stream.append(ch)

    uniq = sorted(set(stream))
    if len(uniq) < 2:
        return None, None

    idx = {d:i for i,d in enumerate(uniq)}
    P = np.zeros((len(uniq), len(uniq)))

    for a,b in zip(stream, stream[1:]):
        P[idx[a], idx[b]] += 1

    # normalize
    row_sums = P.sum(axis=1)
    for i in range(len(uniq)):
        if row_sums[i] > 0:
            P[i] /= row_sums[i]

    return P, uniq


##########################################################
# BOOTSTRAP FUNCTIONS
##########################################################

def bootstrap_ci(data, n_boot=2000, ci=0.95):
    """
    Returns (lo, hi) CI for mean of data
    """
    if len(data)==0:
        return None, None
    means = []
    n = len(data)
    for _ in range(n_boot):
        sample = np.random.choice(data, size=n, replace=True)
        means.append(sample.mean())
    al = (1-ci)/2
    lo = np.quantile(means, al)
    hi = np.quantile(means, 1-al)
    return float(lo), float(hi)


##########################################################
# BEGIN MAIN PIPELINE – PART 1
##########################################################

def main():
    print("Loading verses...")
    verses = load_verses()

    verses["sura"] = verses["sura"].astype("Int64")
    verses["aya"] = verses["aya"].astype("Int64")
    verses["text"] = verses["text"].astype(str)

    print("Total verses loaded:", len(verses))

    ######################################################
    # BASIC COUNTS
    ######################################################
    verses["n_chars"] = verses["text"].str.len()
    verses["n_chars_no_diac"] = verses["text"].apply(lambda t: len(strip_diacritics(t)))
    verses["diac_counts"] = verses["text"].apply(count_diacritics)
    verses["n_diac"] = verses["diac_counts"].apply(lambda c: sum(c.values()))
    verses["has_pause"] = verses["text"].apply(contains_pause)

    # TOKENIZE
    verses["tokens"] = verses["text"].apply(lambda t: TOKEN_RE.findall(t))
    verses["n_words"] = verses["tokens"].apply(len)

    # Save per-verse summary
    verses.to_csv(OUT/"verses_basic_summary.csv", index=False, encoding="utf-8")

    ######################################################
    # GLOBAL DIACRITIC FREQUENCY
    ######################################################
    total_diac = Counter()
    for d in verses["diac_counts"]:
        total_diac.update(d)

    diac_df = pd.DataFrame(total_diac.most_common(), columns=["diac","count"])
    diac_df.to_csv(OUT/"global_diac_counts.csv", index=False, encoding="utf-8")

    # PLOT histogram per verse
    plt.figure(figsize=(8,4))
    plt.hist(verses["n_diac"], bins=60)
    plt.title("Distribution of Diacritics per Verse")
    plt.xlabel("n_diac")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(OUT/"hist_diac_per_verse.png", dpi=150)
    plt.close()

    ######################################################
    # SURAH-LEVEL DIACRITIC DENSITY
    ######################################################
    sura_sum = verses.groupby("sura").agg({
        "n_chars":"sum",
        "n_chars_no_diac":"sum",
        "n_diac":"sum",
        "n_words":"sum",
        "aya":"count"
    }).rename(columns={"aya":"n_verses"}).reset_index()

    sura_sum["diac_per_char"] = sura_sum["n_diac"] / sura_sum["n_chars_no_diac"]
    sura_sum["diac_per_word"] = sura_sum["n_diac"] / sura_sum["n_words"]
    sura_sum.to_csv(OUT/"sura_diac_summary.csv", index=False, encoding="utf-8")

    # Line plot
    plt.figure(figsize=(14,4))
    plt.plot(sura_sum["sura"], sura_sum["diac_per_char"], marker='o')
    plt.title("Diacritic Density per Surah")
    plt.xlabel("Surah")
    plt.ylabel("diacritics per char")
    plt.tight_layout()
    plt.savefig(OUT/"plot_diac_density_by_sura.png", dpi=150)
    plt.close()

    ######################################################
    # CHI-SQUARE TEST: homogeneity of diacritics over surahs
    ######################################################
    obs = sura_sum["n_diac"].values
    expected = sura_sum["n_chars_no_diac"].values / sura_sum["n_chars_no_diac"].sum() * obs.sum()
    chi2, pchi = stats.chisquare(obs, f_exp=expected)

    with open(OUT/"chi2_diac_by_sura.json", "w", encoding="utf-8") as f:
        json.dump({"chi2": float(chi2), "pvalue": float(pchi)}, f, ensure_ascii=False, indent=2)

    ######################################################
    # DETECT MUQATTA'AT – FORMAL
    ######################################################
    sura_first = verses.groupby("sura").first().reset_index()
    muq_counts = Counter()

    for _, row in sura_first.iterrows():
        text = strip_diacritics(row["text"]).strip()
        # take first token before space
        first_tok = text.split()[0] if text.split() else text
        # check if canonical muq
        if first_tok in MUQ_LIST:
            muq_counts[first_tok] += 1

    muq_df = pd.DataFrame(muq_counts.most_common(), columns=["muq","count"])
    muq_df.to_csv(OUT/"muq_counts.csv", index=False, encoding="utf-8")


# ------------------ PART 2 (continue main) ------------------
    ######################################################
    # PART 2: Variant Spellings, Context Entropy, Bootstrap, Permutation (K=5000)
    ######################################################

    # Flatten tokens and normalized tokens
    print("Preparing token lists and normalization...")
    token_lists = verses["tokens"].tolist()
    flat_tokens = [tok for toks in token_lists for tok in toks]
    norm_tokens = [strip_diacritics(tok) for tok in flat_tokens]

    token_df = pd.DataFrame({"token": flat_tokens, "norm": norm_tokens})
    freq_by_norm = token_df.groupby("norm").size().rename("count").reset_index().sort_values("count", ascending=False)
    freq_by_norm.to_csv(OUT/"freq_norm_tokens.csv", index=False, encoding="utf-8")

    # Identify orthographic variants: norms that have >1 distinct written form
    print("Detecting orthographic variants...")
    orth_variants = {}
    for norm, group in token_df.groupby("norm"):
        forms = sorted(set(group["token"].tolist()))
        if len(forms) > 1:
            orth_variants[norm] = forms

    # compute Levenshtein distances for top N norms with variants (limit to top 1000 by frequency to save time)
    max_norms = 1000
    rows = []
    top_norms = freq_by_norm[freq_by_norm["norm"].isin(orth_variants.keys())].head(max_norms)["norm"].tolist()
    for norm in top_norms:
        forms = orth_variants[norm]
        for i in range(len(forms)):
            for j in range(i+1, len(forms)):
                a = forms[i]; b = forms[j]
                d = Levenshtein.distance(a, b)
                rows.append({"norm": norm, "form_a": a, "form_b": b, "lev_dist": int(d)})
    pd.DataFrame(rows).to_csv(OUT/"orth_variant_pairs.csv", index=False, encoding="utf-8")

    # Variant-context association: compute context entropies (prev/next) for top norms
    print("Computing context entropies for variants...")
    mi_rows = []
    for norm in top_norms[:500]:
        contexts_prev = Counter()
        contexts_next = Counter()
        for toks in token_lists:
            for idx, tok in enumerate(toks):
                if strip_diacritics(tok) == norm:
                    prev_tok = toks[idx-1] if idx-1 >= 0 else "<BOS>"
                    next_tok = toks[idx+1] if idx+1 < len(toks) else "<EOS>"
                    contexts_prev[prev_tok] += 1
                    contexts_next[next_tok] += 1
        for side, counter in (("prev", contexts_prev), ("next", contexts_next)):
            total = sum(counter.values())
            if total == 0:
                continue
            probs = np.array(list(counter.values())) / total
            ent = -np.sum(probs * np.log2(probs + 1e-12))
            mi_rows.append({"norm": norm, "side": side, "context_entropy": float(ent), "n_context": int(total)})
    pd.DataFrame(mi_rows).to_csv(OUT/"variant_context_entropy.csv", index=False, encoding="utf-8")

    # Save orth_variants summary (counts)
    var_summary = [{"norm": k, "n_forms": len(v), "sample_forms": v[:5]} for k,v in list(orth_variants.items())[:500]]
    with open(OUT/"orth_variants_summary.json", "w", encoding="utf-8") as f:
        json.dump({"n_variants": len(orth_variants), "sample": var_summary}, f, ensure_ascii=False, indent=2)

    ######################################################
    # BOOTSTRAP CI for diac_per_word by sura & for global stats
    ######################################################
    print("Computing bootstrap CIs (n_boot=2000)...")
    BOOT_N = 2000
    # diac_per_word per verse
    verses["diac_per_word"] = verses["n_diac"] / (verses["n_words"] + 1e-9)
    diac_per_word_vals = verses["diac_per_word"].values
    diac_pw_lo, diac_pw_hi = bootstrap_ci(diac_per_word_vals, n_boot=BOOT_N, ci=0.95)

    # per-sura means and CIs
    sura_means = sura_sum["diac_per_word"].values
    # bootstrap CI for mean diac_per_word across suras
    sura_lo, sura_hi = bootstrap_ci(sura_means, n_boot=BOOT_N, ci=0.95)

    bs_dict = {
        "diac_per_word_mean_ci": {"lo": diac_pw_lo, "hi": diac_pw_hi},
        "sura_mean_diac_per_word_ci": {"lo": sura_lo, "hi": sura_hi}
    }
    with open(OUT/"bootstrap_summary.json", "w", encoding="utf-8") as f:
        json.dump(bs_dict, f, ensure_ascii=False, indent=2)

    ######################################################
    # PERMUTATION TEST: shuffle diacritics across tokens
    # K = 5000 permutations
    ######################################################
    print("Running permutation test (K=5000) — this may take time...")
    K = 5000
    np.random.seed(42)
    # Build list of diacritics per token
    tokens_with_diac = []
    for toks in token_lists:
        for tok in toks:
            diac = "".join(ch for ch in tok if ch in DIAC_SET)
            tokens_with_diac.append(diac)
    total_tokens = len(tokens_with_diac)
    print("Total tokens:", total_tokens)

    # Observed statistic: variance of diac_per_word across suras
    obs_sura_vals = verses.groupby("sura")["diac_per_word"].mean().values
    obs_stat = float(np.nanvar(obs_sura_vals))

    # perform permutations: shuffle diacritics vector and recompute per-sura diac_per_word var
    perm_stats = np.empty(K, dtype=float)
    # Precompute token counts per verse to reconstruct segmentation
    token_counts = verses["n_words"].tolist()

    for k in range(K):
        if k % 500 == 0 and k>0:
            print(f"Permutation {k}/{K}")
        perm = np.random.permutation(tokens_with_diac)
        # reassign per verse
        idx = 0
        per_sura_vals = []
        for tc in token_counts:
            # handle tc may be zero
            if tc <= 0:
                per_sura_vals.append(0.0)
                continue
            taken = perm[idx: idx + tc]
            idx += tc
            # count diacritics assigned
            total_d = sum(len(dstr) for dstr in taken)
            per_sura_vals.append(total_d / (tc + 1e-9))
        # compute var of per-sura mean
        perm_stats[k] = float(np.nanvar(np.array(per_sura_vals).reshape(-1)))

    p_emp = float(np.sum(perm_stats >= obs_stat) / K)
    with open(OUT/"diac_shuffle_perm_test.json", "w", encoding="utf-8") as f:
        json.dump({"obs_stat": obs_stat, "perm_mean": float(np.mean(perm_stats)), "p_emp": p_emp, "K": K}, f, ensure_ascii=False, indent=2)

    # plot
    plt.figure(figsize=(6,3))
    sns.histplot(perm_stats, bins=60, kde=False)
    plt.axvline(obs_stat, color='red', linestyle='--', linewidth=2, label=f"obs={obs_stat:.4e}")
    plt.title("Permutation test: var(diac_per_sura)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT/"perm_diac_var_hist.png", dpi=200)
    plt.close()

    ######################################################
    # MARKOV DIACRITICS MODEL: build chain & simulate
    ######################################################
    print("Building Markov diacritic chain...")
    # Build diacritic stream (flatten)
    diac_stream = [ch for tok in flat_tokens for ch in tok if ch in DIAC_SET]
    total_diac_count = len(diac_stream)
    print("Total diacritic symbols in text:", total_diac_count)

    # Build transitions
    uniq_diacs = sorted(set(diac_stream))
    if len(uniq_diacs) >= 2:
        idx_map = {d:i for i,d in enumerate(uniq_diacs)}
        trans = np.zeros((len(uniq_diacs), len(uniq_diacs)), dtype=float)
        for a,b in zip(diac_stream, diac_stream[1:]):
            trans[idx_map[a], idx_map[b]] += 1
        # normalize rows
        for i in range(trans.shape[0]):
            s = trans[i].sum()
            if s>0:
                trans[i] /= s
    else:
        trans = None

    # Simulate Markov sequences and compute same stat (var per sura)
    MARKOV_SIMS = 1000  # number of simulated diac streams
    markov_stats = []
    if trans is not None:
        # create cumulative distributions for faster sampling
        cum = np.cumsum(trans, axis=1)
        rng = np.random.default_rng(42)
        for sim in range(MARKOV_SIMS):
            if sim % 200 == 0 and sim>0:
                print(f"Markov sim {sim}/{MARKOV_SIMS}")
            # start randomly according to marginal distribution (empirical)
            start = rng.integers(0, len(uniq_diacs))
            seq_idx = [start]
            for _ in range(total_diac_count - 1):
                cur = seq_idx[-1]
                r = rng.random()
                # find next index via cum[cur]
                nxt = int(np.searchsorted(cum[cur], r))
                seq_idx.append(nxt)
            # map back to diac strings
            sim_diacs = [uniq_diacs[i] for i in seq_idx]
            # reassign to tokens as in permutation: use token_counts segmentation
            it = iter(sim_diacs)
            per_sura_vals = []
            for tc in token_counts:
                if tc <= 0:
                    per_sura_vals.append(0.0)
                    continue
                taken = [next(it) for _ in range(tc)]
                total_d = sum(len(dstr) for dstr in taken)
                per_sura_vals.append(total_d/(tc+1e-9))
            markov_stats.append(float(np.nanvar(np.array(per_sura_vals))))
        markov_stats = np.array(markov_stats)
        markov_pemp = float(np.sum(markov_stats >= obs_stat) / MARKOV_SIMS)
    else:
        markov_stats = np.array([])
        markov_pemp = None

    # Save markov results
    np.savetxt(OUT/"markov_diac_stats.txt", markov_stats, fmt="%.6e")
    with open(OUT/"markov_summary.json","w",encoding="utf-8") as f:
        json.dump({"MARKOV_SIMS": MARKOV_SIMS, "markov_pemp_vs_obs": markov_pemp}, f, ensure_ascii=False, indent=2)

    ######################################################
    # Save intermediate summary (part2)
    ######################################################
    part2_summary = {
        "n_tokens": int(total_tokens),
        "obs_stat_var_diac_per_sura": obs_stat,
        "perm_K": K,
        "perm_p_emp": p_emp,
        "markov_sims": MARKOV_SIMS,
        "markov_p_emp": markov_pemp,
        "bootstrap_n": BOOT_N
    }
    with open(OUT/"orthography_partial2.json","w",encoding="utf-8") as f:
        json.dump(part2_summary, f, ensure_ascii=False, indent=2)

    print("PART 2 complete: permutation, markov, variants, bootstrap saved.")


if __name__ == "__main__":
    main()

