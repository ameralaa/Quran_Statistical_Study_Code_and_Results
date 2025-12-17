# anomaly_detection.py
"""
Anomaly Detection for Quran Statistical Analysis
Outputs → results/anomalies/
Uses Isolation Forest, LOF, Z-score & PCA
Run: python anomaly_detection.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import json

ROOT = Path('.')
RESULTS = ROOT / 'final_results'
ANOM_DIR = RESULTS / 'step4_anomalies'
ANOM_DIR.mkdir(parents=True, exist_ok=True)

# ---------------- Load verse metrics ----------------
verse_path = RESULTS / 'step1_preprocess' / 'verse_metrics.csv'
if not verse_path.exists():
    raise FileNotFoundError("verse_metrics.csv not found. Run preprocessing first.")

df = pd.read_csv(verse_path, encoding='utf-8')

# Features to use for anomaly detection
features = [
    'n_chars',
    'n_chars_no_diac',
    'n_words',
]

# Remove non-numeric problems
df_clean = df.copy()
X = df_clean[features].values

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------- 1) Z-SCORE anomalies ----------------
df_clean['zscore_n_words'] = (df_clean['n_words'] - df_clean['n_words'].mean()) / df_clean['n_words'].std()
df_clean['zscore_n_chars'] = (df_clean['n_chars'] - df_clean['n_chars'].mean()) / df_clean['n_chars'].std()

z_thresh = 3
df_clean['is_z_anomaly'] = (
    (df_clean['zscore_n_words'].abs() > z_thresh) |
    (df_clean['zscore_n_chars'].abs() > z_thresh)
).astype(int)

df_clean[['sura', 'aya', 'n_words', 'n_chars', 'zscore_n_words','zscore_n_chars','is_z_anomaly']].to_csv(
    ANOM_DIR / 'zscore_anomalies.csv', index=False, encoding='utf-8'
)

# ---------------- 2) Isolation Forest ----------------
iso = IsolationForest(
    contamination=0.02,  # 2% expected anomalies
    n_estimators=500,
    random_state=42
)
iso.fit(X_scaled)
df_clean['iso_score'] = iso.decision_function(X_scaled)
df_clean['iso_label'] = iso.predict(X_scaled)  # -1 anomalous, 1 normal
df_clean['iso_is_anom'] = (df_clean['iso_label'] == -1).astype(int)

df_clean[['sura','aya','iso_score','iso_is_anom']].to_csv(
    ANOM_DIR / 'isolation_forest.csv', index=False, encoding='utf-8'
)

# ---------------- 3) Local Outlier Factor (LOF) ----------------
lof = LocalOutlierFactor(
    n_neighbors=20,
    contamination=0.02
)
lof_labels = lof.fit_predict(X_scaled)
df_clean['lof_score'] = lof.negative_outlier_factor_
df_clean['lof_is_anom'] = (lof_labels == -1).astype(int)

df_clean[['sura','aya','lof_score','lof_is_anom']].to_csv(
    ANOM_DIR / 'lof_anomalies.csv', index=False, encoding='utf-8'
)

# ---------------- 4) Combined anomaly score ----------------
# Normalize scores
iso_norm = (df_clean['iso_score'] - df_clean['iso_score'].min()) / (df_clean['iso_score'].max() - df_clean['iso_score'].min())
lof_norm = (df_clean['lof_score'] - df_clean['lof_score'].min()) / (df_clean['lof_score'].max() - df_clean['lof_score'].min())

df_clean['combined_score'] = (iso_norm + lof_norm) / 2

df_clean[['sura','aya','combined_score']].to_csv(
    ANOM_DIR / 'combined_anomaly_scores.csv', index=False, encoding='utf-8'
)

# ---------------- 5) PCA for visualization ----------------
pca = PCA(n_components=2)
pc = pca.fit_transform(X_scaled)
df_clean['PC1'] = pc[:,0]
df_clean['PC2'] = pc[:,1]

plt.figure(figsize=(8,6))
sns.scatterplot(
    x=df_clean['PC1'], y=df_clean['PC2'],
    hue=df_clean['combined_score'],
    palette='viridis', s=25
)
plt.title('PCA plot colored by anomaly score')
plt.tight_layout()
plt.savefig(ANOM_DIR / 'pca_anomaly_plot.png', dpi=150)
plt.close()

# ---------------- 6) Heatmap anomaly density per Sura ----------------
anom_by_sura = df_clean.groupby('sura')['combined_score'].mean()
plt.figure(figsize=(12,4))
plt.bar(anom_by_sura.index, anom_by_sura.values)
plt.title('Mean anomaly score per Sura')
plt.xlabel('Sura')
plt.ylabel('Mean anomaly score')
plt.tight_layout()
plt.savefig(ANOM_DIR / 'anomaly_score_per_sura.png', dpi=150)
plt.close()

anom_by_sura.to_csv(ANOM_DIR / 'anomaly_sura_mean.csv', encoding='utf-8')

# ---------------- 7) Full anomaly table ----------------
df_clean.to_csv(
    ANOM_DIR / 'all_anomaly_results.csv',
    index=False, encoding='utf-8'
)

# ---------------- 8) Summary JSON ----------------
summary = {
    'features_used': features,
    'zscore_threshold': z_thresh,
    'iso_contamination': 0.02,
    'lof_contamination': 0.02,
    'n_anomalies_zscore': int(df_clean['is_z_anomaly'].sum()),
    'n_anomalies_iso': int(df_clean['iso_is_anom'].sum()),
    'n_anomalies_lof': int(df_clean['lof_is_anom'].sum()),
}
with open(ANOM_DIR / 'anomaly_summary.json', 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print("Anomaly detection done. Outputs in:", ANOM_DIR)
