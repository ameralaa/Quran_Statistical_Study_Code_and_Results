# preprocess_and_basic_stats.py
import os
import re
import json
from pathlib import Path
from collections import Counter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------- إعداد المجلدات ----------
ROOT = Path('.')
DATA_DIR = ROOT / 'data'
RESULTS_DIR = ROOT / 'final_results' / 'step1_preprocess'
RESULTS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

INPUT_PATH = DATA_DIR / 'quran.txt'  # ضع ملفك هنا

# ---------- تعريف علامات التشكيل والرموز القرآنية ----------
# مجموعة شائعة لعلامات التشكيل والوقف (قابلة للتعديل/الإضافة حسب الحاجة)
ARABIC_DIACRITICS = [
    '\u0610','\u0611','\u0612','\u0613','\u0614','\u0615','\u0616','\u0617','\u0618','\u0619','\u061A',
    '\u064B','\u064C','\u064D','\u064E','\u064F','\u0650','\u0651','\u0652','\u0653','\u0654','\u0655',
    '\u0670','\u06D6','\u06D7','\u06D8','\u06D9','\u06DA','\u06DB','\u06DC','\u06DD','\u06DE','\u06DF',
    '\u06E0','\u06E1','\u06E2','\u06E3','\u06E4','\u06E5','\u06E6','\u06E7','\u06E8','\u06EA','\u06EB',
    '\u06EC','\u06ED'
]
ARABIC_DIACRITICS_RE = '[' + ''.join(ARABIC_DIACRITICS) + ']'

# علامات وقف/ترقيم قرآني شائعة (نماذج، أضف إذا احتجت)
QURAN_MARKERS = {
    'sajdah': '۩',      # علامة السجدة
    'ayah_sep': '۝',    # مثال - قد لا يوجد
    # علامات الوقف الشائعة في المصاحف:
    'waqf_small': ['ۖ','ۗ','ۚ','ۛ','ۜ','۞','۩','ﷺ','ٱ']  # اضف/احذف حسب الحاجة
}

# ---------- دوال مساعدة ----------
def strip_diacritics(text):
    """إزالة التشكيل/المدود/علامات آخَرَة حسب regex"""
    return re.sub(ARABIC_DIACRITICS_RE, '', text)

def parse_line(line):
    """يتوقع شكل: sura|aya|text  - يُعيد tuple (int(sura), int(aya), text)"""
    parts = line.rstrip('\n').split('|', 2)
    if len(parts) != 3:
        return None
    try:
        sura = int(parts[0])
        aya = int(parts[1])
        text = parts[2]
        return sura, aya, text
    except ValueError:
        return None

def char_counter(text, remove_whitespace=True):
    if remove_whitespace:
        chars = [c for c in text if not c.isspace()]
    else:
        chars = list(text)
    return Counter(chars)

def tokenize_words(text):
    # كلمة عربية = حروف عربية فقط أو أرقام عربية
    tokens = re.findall(r'[\u0621-\u064A\u0660-\u0669]+', text)
    return tokens


def count_special_markers(text):
    counts = {}
    for name, mark in QURAN_MARKERS.items():
        if isinstance(mark, list):
            total = sum(text.count(m) for m in mark)
        else:
            total = text.count(mark)
        counts[name] = total
    return counts

# ---------- التحميل والمعالجة ----------
def load_quran_file(path: Path):
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for lineno, line in enumerate(f, 1):
            parsed = parse_line(line)
            if parsed is None:
                # نتوافق: نتجاهل الأسطر الفارغة أو نطبع تحذير
                # print(f"Warning: couldn't parse line {lineno}: {line[:60]!r}")
                continue
            sura, aya, text = parsed
            rows.append({'sura': sura, 'aya': aya, 'text': text})
    return pd.DataFrame(rows)

def run_basic_analysis(df: pd.DataFrame):
    # 1) حفظ نسخة منظمة
    df_sorted = df.sort_values(['sura','aya']).reset_index(drop=True)
    df_sorted.to_csv(RESULTS_DIR / 'verses_structured.csv', index=False, encoding='utf-8')

    # 2) نص كامل
    full_text = '\n'.join(df_sorted['text'].tolist())
    with open(RESULTS_DIR / 'quran_full_text.txt', 'w', encoding='utf-8') as f:
        f.write(full_text)

    # 3) تكرار الحروف (مع تشكيل)
    char_freq_with = char_counter(full_text)
    char_df_with = pd.DataFrame(char_freq_with.most_common(), columns=['char','count'])
    char_df_with.to_csv(RESULTS_DIR / 'char_freq_with_diacritics.csv', index=False, encoding='utf-8')

    # 4) تكرار الحروف (بدون تشكيل)
    full_text_no_diac = strip_diacritics(full_text)
    char_freq_without = char_counter(full_text_no_diac)
    char_df_wo = pd.DataFrame(char_freq_without.most_common(), columns=['char','count'])
    char_df_wo.to_csv(RESULTS_DIR / 'char_freq_without_diacritics.csv', index=False, encoding='utf-8')

    # 5) كلمات وتكرار كلمات (نستخدم النص بدون تشكيل لتحليل الكلمات عادةً)
    all_words = []
    words_by_verse = []
    for _, row in df_sorted.iterrows():
        text = row['text']
        text_no_diac = strip_diacritics(text)
        tokens = tokenize_words(text_no_diac)
        all_words.extend(tokens)
        words_by_verse.append({'sura': row['sura'], 'aya': row['aya'], 'n_words': len(tokens), 'tokens': tokens})

    word_freq = Counter(all_words)
    word_df = pd.DataFrame(word_freq.most_common(), columns=['word','count'])
    word_df.to_csv(RESULTS_DIR / 'word_freq.csv', index=False, encoding='utf-8')

    # 6) توزيع طول الكلمات
    word_lengths = [len(w) for w in all_words]
    wl_series = pd.Series(word_lengths)
    wl_summary = {
        'n_tokens': len(all_words),
        'mean_len': wl_series.mean(),
        'median_len': wl_series.median(),
        'std_len': wl_series.std(),
    }
    pd.DataFrame([wl_summary]).to_csv(RESULTS_DIR / 'word_length_summary.csv', index=False, encoding='utf-8')

    # 7) طول الآيات (حروف/كلمات) + علامات خاصة
    verse_records = []
    for item in words_by_verse:
        s = item['sura']; a = item['aya']
        text = df_sorted[(df_sorted.sura==s) & (df_sorted.aya==a)]['text'].values[0]
        text_no_diac = strip_diacritics(text)
        n_chars = len([c for c in text if not c.isspace()])
        n_chars_no_diac = len([c for c in text_no_diac if not c.isspace()])
        n_words = item['n_words']
        markers = count_special_markers(text)
        verse_records.append({
            'sura': s, 'aya': a,
            'text': text,
            'n_chars': n_chars,
            'n_chars_no_diac': n_chars_no_diac,
            'n_words': n_words,
            **markers
        })
    verse_df = pd.DataFrame(verse_records)
    verse_df.to_csv(RESULTS_DIR / 'verse_metrics.csv', index=False, encoding='utf-8')

    # 8) رسم سريع: تكرار الحروف بدون تشكيل (Top 30)
    topN = 30
    top_chars = char_df_wo.head(topN)
    plt.figure(figsize=(10,6))
    plt.bar(top_chars['char'].astype(str), top_chars['count'])
    plt.title('Top {} characters (no diacritics)'.format(topN))
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'top_chars_no_diacritics.png')
    plt.close()

    # 9) رسم توزيع أطوال الكلمات
    plt.figure(figsize=(10,6))
    wl_series.hist(bins=range(1, max(word_lengths)+2))
    plt.title('Word length distribution')
    plt.xlabel('Length (chars)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'word_length_distribution.png')
    plt.close()

    # 10) حفظ بعض النتائج المفيدة كـ JSON
    summary = {
        'n_verses': len(df_sorted),
        'n_tokens': int(wl_summary['n_tokens']),
        'word_length_summary': wl_summary,
    }
    with open(RESULTS_DIR / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Done. Results in:", RESULTS_DIR)

# ---------- main ----------
def main():
    if not INPUT_PATH.exists():
        print(f"Input file {INPUT_PATH} not found. Please place quran_uthmani.txt in {DATA_DIR}")
        return
    df = load_quran_file(INPUT_PATH)
    print("Loaded verses:", len(df))
    run_basic_analysis(df)

if __name__ == '__main__':
    main()
