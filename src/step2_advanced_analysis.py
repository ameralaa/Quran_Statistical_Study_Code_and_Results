# advanced_analysis_and_report.py
import math
import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from scipy import stats
from sklearn.linear_model import LinearRegression
import docx
from docx.shared import Inches

ROOT = Path('.')
RESULTS = ROOT / 'final_results' 
OUTPUT = ROOT / RESULTS / 'step2_advanced_analysis'
OUTPUT.mkdir(exist_ok=True)

# ---------- load summary ----------
with open(RESULTS / 'step1_preprocess' / 'summary.json', 'r', encoding='utf-8') as f:
    summary = json.load(f)

# ---------- load word freq ----------
word_df = pd.read_csv(RESULTS / 'step1_preprocess' / 'word_freq.csv', encoding='utf-8')  # columns: word,count
# Ensure sorted
word_df = word_df.sort_values('count', ascending=False).reset_index(drop=True)
word_df['rank'] = word_df.index + 1

# ---------- top 30 words ----------
top30 = word_df.head(30)
top30.to_csv(OUTPUT / 'top30_words.csv', index=False, encoding='utf-8')

# ---------- Zipf plot ----------
ranks = top30['rank']  # for plotting use full list though
freqs = word_df['count'].values
ranks_full = word_df['rank'].values

plt.figure(figsize=(8,6))
plt.loglog(ranks_full, freqs, marker='.', linestyle='none')
plt.xlabel('Rank (log)')
plt.ylabel('Frequency (log)')
plt.title('Zipf plot (full)')
plt.tight_layout()
plt.savefig(OUTPUT / 'zipf_full.png')
plt.close()

# ---------- Fit power-law (Zipf) on middle region ----------
# choose region: exclude top 20 (stopwords heavy) and tail after rank 10000
lo, hi = 20, min(2000, len(word_df))
x = np.log(word_df['rank'].iloc[lo:hi].values).reshape(-1,1)
y = np.log(word_df['count'].iloc[lo:hi].values)
model = LinearRegression().fit(x, y)
slope = model.coef_[0]
intercept = model.intercept_
r2 = model.score(x, y)
fit_summary = {'slope': float(slope), 'intercept': float(intercept), 'r2': float(r2), 'fit_reg_range': [int(lo+1), int(hi)]}

# residuals and KS test against power-law fit
pred = model.predict(x)
residuals = y - pred
# normality test of residuals (Shapiro or KS)
ks_stat, ks_p = stats.kstest(residuals, 'norm', args=(residuals.mean(), residuals.std()))

# ---------- Shannon entropy for words and chars ----------
counts_words = word_df['count'].values
p_words = counts_words / counts_words.sum()
entropy_words = -np.sum(p_words * np.log2(p_words))

char_df = pd.read_csv(RESULTS /  'step1_preprocess' / 'char_freq_without_diacritics.csv', encoding='utf-8')
counts_chars = char_df['count'].values
p_chars = counts_chars / counts_chars.sum()
entropy_chars = -np.sum(p_chars * np.log2(p_chars))

# ---------- Autocorrelation for token sequence lengths per verse ----------
verse_df = pd.read_csv(RESULTS /  'step1_preprocess' / 'verse_metrics.csv', encoding='utf-8')
series = verse_df['n_words'].values - verse_df['n_words'].mean()
def autocorr(x, lag):
    return np.corrcoef(x[:-lag], x[lag:])[0,1]
autocorrs = {lag: autocorr(series, lag) for lag in range(1, 31)}

# ---------- Hurst exponent (rescaled range) simple estimator ----------
def hurst_exponent(ts):
    N = len(ts)
    T = np.arange(1, N+1)
    Y = np.cumsum(ts - np.mean(ts))
    R = np.maximum.accumulate(Y) - np.minimum.accumulate(Y)
    S = pd.Series(ts).rolling(window=N).std().iloc[-1] if N>1 else 0
    # Use a very simple approximation via rescaled range over splits
    def rs(n):
        x = ts[:n]
        y = np.cumsum(x - x.mean())
        R = y.max() - y.min()
        S = x.std(ddof=1) if n>1 else 0
        return R / S if S!=0 else 0
    sizes = [int(N / k) for k in range(2, min(20,N//2))]
    rsvals = []
    sizes_clean = []
    for s in sizes:
        val = rs(s)
        if val>0:
            rsvals.append(val)
            sizes_clean.append(s)
    if len(rsvals) < 2:
        return float('nan')
    # fit log(R/S) ~ H * log(n)
    lr = LinearRegression().fit(np.log(sizes_clean).reshape(-1,1), np.log(rsvals))
    return float(lr.coef_[0])

hurst = hurst_exponent(verse_df['n_words'].values)

# ---------- Save numeric summary ----------
adv_summary = {
    'zipf_fit': fit_summary,
    'zipf_fit_ks_residuals': {'ks_stat': float(ks_stat), 'ks_p': float(ks_p)},
    'entropy_words': float(entropy_words),
    'entropy_chars': float(entropy_chars),
    'autocorr_first_10': {str(k): float(v) for k,v in list(autocorrs.items())[:10]},
    'hurst': hurst
}
with open(OUTPUT / 'advanced_summary.json', 'w', encoding='utf-8') as f:
    json.dump(adv_summary, f, ensure_ascii=False, indent=2)

# ---------- Plots for fit diagnostics ----------
# residuals histogram
plt.figure(figsize=(8,5))
plt.hist(residuals, bins=60)
plt.title('Residuals of Zipf log-log linear fit (middle region)')
plt.tight_layout()
plt.savefig(OUTPUT / 'zipf_residuals_hist.png')
plt.close()

# ---------- assemble docx report ----------
doc = docx.Document()
doc.add_heading('التحليل الإحصائي المتقدّم للنص القرآني', level=1)
doc.add_paragraph(f"المصدر: الملف المعالج في results/ (نُزِّلَ في هذه الجلسة). عدد الآيات: {summary['n_verses']}, عدد الكلمات: {summary['n_tokens']}.")
doc.add_heading('1. Zipf analysis', level=2)
doc.add_paragraph(f"تم ملاءمة نموذج خطي على لوغاريتمات الرتبة والتردد في النطاق {fit_summary['fit_reg_range']}. قيمة الميل (slope) = {fit_summary['slope']:.4f}, R^2 = {fit_summary['r2']:.4f}.")
doc.add_picture(str(OUTPUT / 'zipf_full.png'), width=Inches(6))
doc.add_heading('2. Entropy', level=2)
doc.add_paragraph(f"Shannon entropy (words): {entropy_words:.4f} bits; (chars): {entropy_chars:.4f} bits.")
doc.add_heading('3. Autocorrelation (عدد الكلمات في الآيات)', level=2)
doc.add_paragraph("autocorrelation for lags 1..10:")
for k in range(1,11):
    doc.add_paragraph(f"lag {k}: {autocorrs[k]:.4f}")
doc.add_heading('4. Hurst exponent (επισκόπηση)', level=2)
doc.add_paragraph(f"Estimated Hurst exponent (n_words series): {hurst:.4f}")
doc.add_heading('5. ملاحظات حول الانحرافات الإحصائية', level=2)
doc.add_paragraph("ملف advanced_summary.json يحتوي على النتائج الرقمية التفصيلية. انحرافات عن Zipf أو قيم غير اعتيادية ستُحلل لاحقًا في صفحة 'anomaly detection'.")
doc.save(OUTPUT / 'advanced_report.docx')

print("Done. Outputs in:", OUTPUT)
