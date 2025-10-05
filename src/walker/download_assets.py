# download_assets.py
import os
import spacy
import subprocess
from pathlib import Path

from . import config

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

def download_spacy_model(model_name: str, model_short_name: str):
    """Downloads a spaCy model if it's not already installed."""
    print(f"Checking for spaCy model '{model_short_name}'...")
    if not spacy.util.is_package(model_short_name):
        print(f"...model not found. Downloading '{model_short_name}'...")
        spacy.cli.download(model_short_name)
        print("...download complete!")
    else:
        print("...model already installed. Skipping.")

def get_spacy_model_name(lang_code: str) -> str:
    """Gets the default spaCy model name for a given language code."""
    # This mapping can be expanded for more languages
    model_map = {
        "en": "en_core_web_lg",
        "es": "es_core_news_lg",
        "fr": "fr_core_news_lg",
    }
    return model_map.get(lang_code, f"{lang_code}_core_news_lg")

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

    # Download sentence transformer model
    download_sentence_transformer('all-MiniLM-L6-v2', Path('./models/all-MiniLM-L6-v2'))
    # Download spaCy models for each configured PII language
    for lang in app_config.pii_languages:
        model_name = get_spacy_model_name(lang)
        download_spacy_model(model_name, lang)
    cache_tldextract_list(Path('./src/walker/models/tldextract_cache'))
    print("\nAll offline assets are ready.")