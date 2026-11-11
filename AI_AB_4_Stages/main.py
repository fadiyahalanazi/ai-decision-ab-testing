from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
STAGES = [
    "01_question_and_design.py",
    "02_sample_size_and_duration.py",
    "03_analysis_plan_and_results.py",
    "04_report_and_presentation.py",
]

def main():
    for stage in STAGES:
        print(f"Running {stage}", flush=True)
        subprocess.run([sys.executable, str(ROOT / stage)], cwd=ROOT, check=True)
    print("All four stages completed successfully")

if __name__ == "__main__":
    main()
