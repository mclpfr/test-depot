import pandas as pd
import os
import yaml
import numpy as np

# Essayez d'importer SDV, sinon fallback à du bruit simple
try:
    from sdv.tabular import CTGAN
    SDV_AVAILABLE = True
except ImportError:
    SDV_AVAILABLE = False
    print("SDV non disponible, fallback sur génération bruitée simple.")

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def generate_synthetic_data(real_data, n_samples):
    if SDV_AVAILABLE:
        model = CTGAN(epochs=10)
        model.fit(real_data)
        synth = model.sample(n_samples)
        return synth
    else:
        # Fallback : bruit simple sur les colonnes numériques, sampling sur les catégorielles
        synth = pd.DataFrame()
        for col in real_data.columns:
            if np.issubdtype(real_data[col].dtype, np.number):
                mean = real_data[col].mean()
                std = real_data[col].std()
                synth[col] = np.random.normal(mean, std, n_samples)
                # Arrondi si int
                if np.issubdtype(real_data[col].dtype, np.integer):
                    synth[col] = synth[col].round().astype(int)
            else:
                synth[col] = np.random.choice(real_data[col].dropna().unique(), n_samples)
        return synth

def main(config_path="config.yaml"):
    config = load_config(config_path)
    year = config["data_extraction"]["year"]
    input_path = os.path.join("data/raw", f"accidents_{year}.csv")
    output_path = os.path.join("data/raw", f"accidents_{year}_varied.csv")

    # Charger les données réelles
    data = pd.read_csv(input_path, low_memory=False)
    n = len(data)
    n_real = n // 2
    n_synth = n - n_real

    real_part = data.sample(n=n_real, random_state=42)
    synth_part = generate_synthetic_data(real_part, n_synth)

    # Concaténer et mélanger
    varied = pd.concat([real_part, synth_part], ignore_index=True)
    varied = varied.sample(frac=1, random_state=42).reset_index(drop=True)
    varied.to_csv(output_path, index=False)
    print(f"Fichier varié sauvegardé : {output_path} ({len(varied)} lignes)")

if __name__ == "__main__":
    main() 