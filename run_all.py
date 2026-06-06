"""Reproduce the whole project end to end: data -> NLP -> numeric -> evaluation."""
import subprocess, sys, os
SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
for step in ["data_prep.py", "eda.py", "train_nlp.py", "train_numeric.py", "evaluate.py", "nlp_qualitative.py"]:
    print(f"\n===== running {step} =====")
    subprocess.run([sys.executable, os.path.join(SRC, step)], check=True)
print("\nAll steps complete. Launch the app with:  python app.py")
