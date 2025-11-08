# download_assets.py
import os
import spacy
import subprocess
import urllib.request
from pathlib import Path

from . import config, file_processor

from sentence_transformers import SentenceTransformer

def download_sentence_transformer(model_name: str, local_path: Path, progress_callback: callable = None):
    """Downloads and saves a SentenceTransformer model."""
    if progress_callback:
        progress_callback("embedding", f"Downloading model '{model_name}'...")
    else:
        print(f"Downloading SentenceTransformer model '{model_name}' to '{local_path}'...")

    if local_path.exists():
        if progress_callback: progress_callback("embedding", "Model already exists. Skipping.")
        else: print("...model already exists. Skipping.")
        return
    model = SentenceTransformer(model_name)
    model.save(str(local_path))
    if progress_callback: progress_callback("embedding", "Download complete!")
    else: print("...download complete!")

def download_spacy_model(model_full_name: str, model_short_name: str, progress_callback: callable = None):
    """Downloads a spaCy model if it's not already installed."""
    if progress_callback: progress_callback("pii", f"Checking for spaCy model '{model_full_name}'...")
    else: print(f"Checking for spaCy model '{model_full_name}'...")

    if not spacy.util.is_package(model_full_name):
        if progress_callback: progress_callback("pii", f"Model not found. Downloading '{model_full_name}'...")
        else: print(f"...model not found. Downloading '{model_full_name}'...")
        spacy.cli.download(model_full_name) # The extra argument was causing the issue.
        if progress_callback: progress_callback("pii", "Download complete!")
        else: print("...download complete!")
    else:
        if progress_callback: progress_callback("pii", "Model already installed. Skipping.")
        else: print("...model already installed. Skipping.")

def cache_tldextract_list(cache_dir: Path, progress_callback: callable = None):
    """Caches the Public Suffix List for tldextract."""
    if progress_callback: progress_callback("tldextract", f"Caching tldextract list to '{cache_dir}'...")
    else: print(f"Caching tldextract list to '{cache_dir}'...")

    cache_dir.mkdir(parents=True, exist_ok=True)
    # Set environment variable for the subprocess
    env = os.environ.copy()
    env["TLDEXTRACT_CACHE_DIR"] = str(cache_dir)
    try:
        # Redirect stderr to devnull to suppress tldextract's own logging during caching
        subprocess.run(["python", "-c", "import tldextract; tldextract.TLDExtract()"], env=env, check=True, stderr=subprocess.DEVNULL)
        if progress_callback: progress_callback("tldextract", "Caching complete!")
        else: print("...caching complete!")
    except Exception as e:
        if progress_callback: progress_callback("tldextract", f"Error caching tldextract: {e}")
        else: print(f"...error caching tldextract: {e}")

def cache_fido_signatures(cache_dir: Path, progress_callback: callable = None):
    """Downloads the latest PRONOM signature file for Fido."""
    import urllib.error

    if progress_callback: progress_callback("fido", f"Caching Fido signature file to '{cache_dir}'...")
    else: print(f"Caching Fido signature file to '{cache_dir}'...")

    cache_dir.mkdir(parents=True, exist_ok=True)
    destination_path = cache_dir / "DROID_SignatureFile.xml"

    if destination_path.exists():
        if progress_callback: progress_callback("fido", "Signature file already exists. Skipping.")
        else: print("...signature file already exists. Skipping.")
        return

    # The URL pattern for signature files seems to be versioned. We will
    # programmatically find the latest version.
    base_url = "https://cdn.nationalarchives.gov.uk/documents/DROID_SignatureFile_V{version}.xml"
    start_version = 120  # Start from a recent known version
    latest_found_version = 0

    headers = {'User-Agent': 'Mozilla/5.0'}

    if progress_callback: progress_callback("fido", "Finding latest PRONOM signature file version...")
    else: print("...finding latest PRONOM signature file version...")

    for version in range(start_version, start_version + 50): # Check the next 50 versions
        url = base_url.format(version=version)
        try:
            req = urllib.request.Request(url, method='HEAD', headers=headers)
            with urllib.request.urlopen(req):
                latest_found_version = version
        except urllib.error.HTTPError as e:
            if e.code == 404:
                # We've gone past the latest version.
                break
            else:
                if progress_callback: progress_callback("fido", f"Warning: Received unexpected HTTP status {e.code} for version {version}. Stopping search.")
                else: print(f"...warning: Received unexpected HTTP status {e.code} for version {version}. Stopping search.")
                break

    if latest_found_version == 0:
        if progress_callback: progress_callback("fido", "Error: Could not find a valid signature file.")
        else: print("...error: Could not find a valid signature file. Please check the URL pattern in the script.")
        return

    latest_url = base_url.format(version=latest_found_version)
    if progress_callback: progress_callback("fido", f"Latest version found is {latest_found_version}. Downloading from {latest_url}")
    else: print(f"...latest version found is {latest_found_version}. Downloading from {latest_url}")

    try:
        request = urllib.request.Request(latest_url, headers=headers)
        with urllib.request.urlopen(request) as response, open(destination_path, 'wb') as out_file:
            out_file.write(response.read())
        if progress_callback: progress_callback("fido", "Caching complete!")
        else: print("...caching complete!")
    except Exception as e:
        if progress_callback: progress_callback("fido", f"Error: Failed to download Fido signature file: {e}")
        else: print(f"...error: Failed to download Fido signature file: {e}")
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
    download_sentence_transformer('all-MiniLM-L6-v2', models_dir / 'all-MiniLM-L6-v2')

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