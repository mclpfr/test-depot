import os
import subprocess
import yaml
from pathlib import Path

def run_cmd(cmd, desc=""):
    print(f"[CMD] {desc} → {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[ERROR] {desc}:\n{result.stderr.strip()}")
        raise RuntimeError(f"Failed: {desc}")
    else:
        print(result.stdout.strip())
        return result.stdout.strip()

def file_has_changed(path):
    """Check if the .dvc file corresponding to a data file has changed"""
    result = subprocess.run(["git", "status", "--porcelain", f"{path}.dvc"], capture_output=True, text=True)
    return bool(result.stdout.strip())

def main():
    # 1. Load config file
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
        dagshub_cfg = config.get("dagshub", {})
        token = dagshub_cfg.get("token", "")

    # 2. Set DVC remote credentials
    run_cmd(["dvc", "remote", "modify", "origin", "--local", "access_key_id", token], "Set access_key_id")
    run_cmd(["dvc", "remote", "modify", "origin", "--local", "secret_access_key", token], "Set secret_access_key")

    # 3. Files to track
    files = [
        "data/raw/accidents_2023.csv",
        "data/processed/prepared_accidents_2023.csv",
        "models/rf_model_2023.joblib",
        "models/best_model_2023.joblib"
    ]

    updated_dvc_files = []

    for file in files:
        if not Path(file).exists():
            print(f"Missing file: {file}")
            continue

        print(f"Found file: {file}")
        # Check that the file is not dvc.lock or a config file
        if file.endswith('.dvc') or file.endswith('dvc.lock'):
            print(f"Skipping non-data file: {file}")
            continue
        try:
            run_cmd(["dvc", "commit", "-f", file], f"DVC commit {file}")
        except Exception as e:
            print(f"[ERROR] DVC commit failed for {file}: {e}")
            continue

        if file_has_changed(file):
            print(f"Change detected for {file}")
            updated_dvc_files.append(file)
        else:
            print(f"No changes for {file}")

    if updated_dvc_files:
        # 4. Git add, commit, push
        dvc_files = [f"{f}.dvc" for f in updated_dvc_files]
        run_cmd(["git", "add", ".gitignore"] + dvc_files, "Git add DVC files")
        run_cmd(["git", "commit", "-m", "Auto: update tracked data and models with DVC"], "Git commit")
        run_cmd(["git", "push"], "Git push")
    else:
        print("No Git commit necessary.")

    # 5. Final DVC push
    run_cmd(["dvc", "push"], "Final DVC push")

    print("All data and models are versioned and synced!")

if __name__ == "__main__":
    main()

