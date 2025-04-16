import os
import subprocess
import glob
import time

# Add and push a file to DVC and Git
def dvc_add_and_push(file_path):
    print(f"DVC add and push for: {file_path}")
    subprocess.run(["dvc", "add", file_path], check=True)
    subprocess.run(["git", "add", f"{file_path}.dvc"], check=True)
    subprocess.run(["git", "commit", "-m", f"Auto DVC add for {file_path}"], check=True)
    subprocess.run(["dvc", "push", file_path], check=True)
    subprocess.run(["git", "push"], check=True)

# Check if the file is already tracked by DVC
def is_tracked_by_dvc(file_path):
    dvc_file = f"{file_path}.dvc"
    return os.path.exists(dvc_file)

# Main logic: find files and add/push if not tracked
def main():
    # Wait for files to be generated (optional, adapt to your pipeline)
    time.sleep(10)
    data_files = glob.glob("data/**/*.csv", recursive=True)
    model_files = glob.glob("models/**/*.joblib", recursive=True)
    all_files = data_files + model_files

    for file_path in all_files:
        if not is_tracked_by_dvc(file_path):
            dvc_add_and_push(file_path)
        else:
            print(f"Already tracked by DVC: {file_path}")

if __name__ == "__main__":
    main() 