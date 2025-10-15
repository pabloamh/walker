# download_assets.py
import os
import spacy
import subprocess
import urllib.request
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

def cache_fido_signatures(cache_dir: Path):
    """Downloads the latest PRONOM signature file for Fido."""
    print(f"Caching Fido signature file to '{cache_dir}'...")
    cache_dir.mkdir(parents=True, exist_ok=True)

    signature_file_url = "https://www.nationalarchives.gov.uk/pronom/latest/DROID_SignatureFile.xml"
    destination_path = cache_dir / "DROID_SignatureFile.xml"

    if destination_path.exists():
        print("...signature file already exists. Skipping.")
        return

    try:
        # Fido doesn't have a built-in update command. We must download it manually.
        print(f"Downloading from {signature_file_url}...")
        with urllib.request.urlopen(signature_file_url) as response, open(destination_path, 'wb') as out_file:
            out_file.write(response.read())
        print("...caching complete!")
    except Exception as e:
        print(f"...error: Failed to download Fido signature file: {e}")
        if destination_path.exists():
            destination_path.unlink() # Clean up partial download


def run_download():
    """Main function to download all required offline assets."""

    # Define paths relative to this script's location for robustness.
    script_dir = Path(__file__).parent
    models_dir = script_dir / "models"
    models_dir.mkdir(exist_ok=True)

    print("--- Starting asset download process ---")
    print(f"All assets will be saved in: {models_dir.resolve()}")

    # Download sentence transformer model
    download_sentence_transformer('all-MiniLM-L6-v2', models_dir)

    # Download spaCy models for each configured PII language
    app_config = config.load_config()
    print(f"\nPII detection is configured for: {app_config.pii_languages}")
    for lang in app_config.pii_languages:
        model_name = file_processor.get_spacy_model_name(lang)
        download_spacy_model(model_name, lang)

    cache_tldextract_list(models_dir / 'tldextract_cache')

    # Download Fido signatures if fido is configured to be used
    if app_config.use_fido:
        cache_fido_signatures(models_dir / 'fido_cache')

    print("\nAll offline assets are ready.")
    print("\n--- IMPORTANT ---")
    print("The AI models used for text analysis are memory-intensive.")
    print("Each worker process can consume 1.5-2.5 GB of RAM.")
    print("On systems with limited memory (e.g., 16 GB or less), it is highly recommended to:")
    print("1. Set `workers = 1` or `workers = 2` in your 'walker.toml'.")
    print("2. Set `memory_limit_gb = 4.0` in your 'walker.toml' to prevent system freezes (on Linux/macOS).")

if __name__ == "__main__":
    run_download()