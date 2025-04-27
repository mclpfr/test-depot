import subprocess
import yaml
from pathlib import Path
import sys

def run(cmd):
    print(f"[CMD] {cmd}")
    result = subprocess.run(cmd, shell=True, text=True)
    if result.returncode != 0:
        print(f"❌ Command failed: {cmd}")
        sys.exit(1)

def load_credentials(config_path="config.yaml"):
    if not Path(config_path).exists():
        print(f"❌ Missing configuration file: {config_path}")
        sys.exit(1)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    try:
        user = config["dvc"]["dagshub_user"]
        token = config["dvc"]["dagshub_token"]
        return user, token
    except KeyError:
        print("❌ Invalid config.yaml format.")
        sys.exit(1)

def main():
    print("📥 Loading credentials from config.yaml...")
    user, token = load_credentials()

    print("🔧 Configuring DVC remote with provided credentials...")
    run(f'dvc remote modify origin --local access_key_id "{token}"')
    run(f'dvc remote modify origin --local secret_access_key "{token}"')

    print("⬇️ Pulling versioned data...")
    run("dvc pull")

    print("✅ Data successfully retrieved.")

if __name__ == "__main__":
    main()

