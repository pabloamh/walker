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

def download_droid(dest_dir: Path, progress_callback: callable = None):
    """Downloads and sets up the DROID binary and signature files."""
    droid_url = "https://cdn.nationalarchives.gov.uk/documents/droid-binary-6.8.1-bin.zip"
    
    if progress_callback: progress_callback("droid", f"Downloading DROID from {droid_url}...")
    else: print(f"Downloading DROID to '{dest_dir}'...")

    dest_dir.mkdir(exist_ok=True)
    zip_path = dest_dir / "droid.zip"

    if (dest_dir / "droid.sh").exists():
        if progress_callback: progress_callback("droid", "DROID already exists. Skipping.")
        else: print("...DROID already exists. Skipping.")
        return

    try:
        # Download the zip file
        with urllib.request.urlopen(droid_url) as response, open(zip_path, 'wb') as out_file:
            out_file.write(response.read())
        
        # Unzip and update signature file
        import zipfile
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dest_dir)
        
        # Ensure the droid shell script is executable
        droid_script_path = dest_dir / "droid.sh"
        if sys.platform != "win32":
            droid_script_path.chmod(droid_script_path.stat().st_mode | 0o111)

        # Run the signature update script
        if progress_callback: progress_callback("droid", "Updating DROID signature file...")
        else: print("...updating DROID signature file...")
        subprocess.run([str(droid_script_path), "-d"], check=True, capture_output=True)
        
        if progress_callback: progress_callback("droid", "DROID setup complete!")
        else: print("...DROID setup complete!")
    except subprocess.CalledProcessError as e:
        error_message = f"Error setting up DROID. It might require Java to be installed.\n"
        error_message += f"Stderr: {e.stderr.decode('utf-8', errors='ignore')}"
        if progress_callback: progress_callback("droid", error_message)
        else: print(f"...{error_message}")
    except Exception as e:
        if progress_callback: progress_callback("droid", f"An unexpected error occurred during DROID setup: {e}")
        else: print(f"...error setting up DROID: {e}")
    finally:
        if zip_path.exists():
            zip_path.unlink()


def run_download():
    """Main function to download all required offline assets."""

    # Define paths relative to this script's location for robustness.
    script_dir = Path(__file__).parent
    models_dir = script_dir / "droid"
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

    # Download DROID if it is configured to be used
    if app_config.use_droid:
        download_droid(script_dir / 'droid')

    print("\nAll offline assets are ready.")
    print("\n--- IMPORTANT ---")
    print("The AI models used for text analysis are memory-intensive.")
    print("Each worker process can consume 1.5-2.5 GB of RAM.")
    print("On systems with limited memory (e.g., 16 GB or less), it is highly recommended to:")
    print("1. Set `workers = 1` or `workers = 2` in your 'walker.toml'.")
    print("2. Set `memory_limit_gb = 4.0` in your 'walker.toml' to prevent system freezes (on Linux/macOS).")

if __name__ == "__main__":
    run_download()