# download_assets.py
import os
import spacy
import subprocess
from pathlib import Path

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

def download_spacy_model(model_name: str):
    """Downloads a spaCy model if it's not already installed."""
    print(f"Checking for spaCy model '{model_name}'...")
    if not spacy.util.is_package(model_name):
        print(f"...model not found. Downloading '{model_name}'...")
        spacy.cli.download(model_name)
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
    download_sentence_transformer('all-MiniLM-L6-v2', Path('./models/all-MiniLM-L6-v2'))
    download_spacy_model('en_core_web_lg')
    cache_tldextract_list(Path('./models/tldextract_cache'))
    print("\nAll offline assets are ready.")