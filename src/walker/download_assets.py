# download_assets.py
import os
import spacy
import subprocess
from pathlib import Path

from . import config, file_processor

from sentence_transformers import SentenceTransformer

def download_sentence_transformer(model_name: str, local_path: Path):
    """Downloads and saves a SentenceTransformer model."""
    print(f"Downloading SentenceTransformer model '{model_name}' to '{local_path}'...")
    if local_path.exists():
        print("...model already exists. Skipping.")
        return
    model = SentenceTransformer(model_name)
    model.save(str(local_path))
    print("...download complete!")

def download_spacy_model(model_full_name: str, model_short_name: str):
    """Downloads a spaCy model if it's not already installed."""
    print(f"Checking for spaCy model '{model_full_name}'...")
    if not spacy.util.is_package(model_full_name):
        print(f"...model not found. Downloading '{model_full_name}'...")
        spacy.cli.download(model_full_name)
        print("...download complete!")
    else:
        print("...model already installed. Skipping.")

def cache_tldextract_list(cache_dir: Path):
    """Caches the Public Suffix List for tldextract."""
    print(f"Caching tldextract list to '{cache_dir}'...")
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Set environment variable for the subprocess
    env = os.environ.copy()
    env["TLDEXTRACT_CACHE_DIR"] = str(cache_dir)
    subprocess.run(["python", "-c", "import tldextract; tldextract.TLDExtract()"], env=env, check=True)
    print("...caching complete!")

if __name__ == "__main__":
    app_config = config.load_config()

    # Define paths relative to this script's location for robustness.
    models_dir = Path(__file__).parent / "models"

    # Download sentence transformer model
    download_sentence_transformer('all-MiniLM-L6-v2', models_dir / 'all-MiniLM-L6-v2')
    # Download spaCy models for each configured PII language
    for lang in app_config.pii_languages:
        model_name = file_processor.get_spacy_model_name(lang)
        download_spacy_model(model_name, lang)
    cache_tldextract_list(models_dir / 'tldextract_cache')

    print("\nAll offline assets are ready.")
    print("\n--- IMPORTANT ---")
    print("The AI models used for text analysis are memory-intensive.")
    print("Each worker process can consume 1.5-2.5 GB of RAM.")
    print("On systems with limited memory (e.g., 16 GB or less), it is highly recommended to:")
    print("1. Set `workers = 1` or `workers = 2` in your 'walker.toml'.")
    print("2. Set `memory_limit_gb = 4.0` in your 'walker.toml' to prevent system freezes (on Linux/macOS).")