import subprocess
import sys
import time

STEPS = [
    ("Preprocessing", "src/step1_preprocess.py"),
    ("Advanced Statistical Analysis", "src/step2_advanced_analysis.py"),
    ("Structural Analysis", "src/step3_structural_analysis.py"),
    ("Anomaly Detection", "src/step4_anomaly_detection.py"),
    ("Null Models Simulation", "src/step5_null_models.py"),
    ("P-value & Statistical Tests", "src/step5b_pvalues_tests.py"),
    ("Orthography & Diacritics Analysis", "src/step6_orthography_analysis.py"),
    ("Orthography Final Summary", "src/step6b_orthography_final.py"),
]

def run_step(step_name, script_path):
    print("\n" + "=" * 70)
    print(f"▶ Running Step: {step_name}")
    print("=" * 70)

    start_time = time.time()

    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(f"❌ ERROR in {step_name}")
        print(result.stderr)
        sys.exit(1)

    print(result.stdout)
    elapsed = time.time() - start_time
    print(f"✅ Completed {step_name} in {elapsed:.2f} seconds")

def main():
    print("\n🔷 Quran Statistical Analysis Pipeline Started 🔷\n")

    for step_name, script in STEPS:
        run_step(step_name, script)

    print("\n🎉 Pipeline completed successfully!")
    print("All results are stored in the results/ directory.")

if __name__ == "__main__":
    main()
