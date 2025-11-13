# download_assets.py
import os
import platform
import spacy
import subprocess
import sys
import urllib.request
from pathlib import Path
import zipfile
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

def download_java(dest_dir: Path, progress_callback: callable = None):
    """Downloads and extracts a portable Java JRE for DROID."""
    java_dir = dest_dir / "java"
    java_executable_path_win = java_dir / "bin" / "java.exe"
    java_executable_path_unix = java_dir / "bin" / "java"

    if java_executable_path_win.exists() or java_executable_path_unix.exists():
        if progress_callback: progress_callback("java", "Java runtime already exists. Skipping.")
        else: print("...Java runtime already exists. Skipping.")
        return

    # Determine the correct download URL based on OS and architecture
    os_map = {"linux": "linux", "darwin": "macos", "win32": "win"}
    arch_map = {"amd64": "x64", "x86_64": "x64", "aarch64": "aarch64", "arm64": "aarch64"}
    
    os_key = os_map.get(sys.platform)
    arch_key = arch_map.get(platform.machine().lower())

    if not os_key or not arch_key:
        message = f"Unsupported OS/architecture for automatic Java download: {sys.platform}/{platform.machine()}"
        if progress_callback: progress_callback("java", message)
        else: print(message)
        return

    # Corrected URL pattern based on user feedback.
    # Format: zulu<zulu_version>-ca-jre<java_version>-<os>_<arch>.zip
    zulu_version = "21.46.19"
    java_version = "21.0.9"

    # Handle platform-specific naming conventions in the URL.
    if os_key == "macos" and arch_key == "aarch64":
        platform_str = "macosx_aarch64"
    else:
        platform_str = f"{os_key}_{arch_key}"

    base_url = "https://cdn.azul.com/zulu/bin/"
    file_name = f"zulu{zulu_version}-ca-jre{java_version}-{platform_str}.zip"
    java_url = f"{base_url}{file_name}"

    if progress_callback: progress_callback("java", f"Downloading Java from {java_url}...")
    else: print(f"Downloading Java JRE to '{java_dir}'...")

    java_dir.mkdir(exist_ok=True, parents=True)
    zip_path = java_dir / "java.zip"

    try:
        with urllib.request.urlopen(java_url) as response, open(zip_path, 'wb') as out_file:
            out_file.write(response.read())

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Extract and strip the top-level directory
            for member in zip_ref.infolist():
                # Remove the top-level folder from the path
                new_path = Path(*Path(member.filename).parts[1:])
                target_path = java_dir / new_path
                if member.is_dir():
                    target_path.mkdir(exist_ok=True, parents=True)
                else:
                    target_path.parent.mkdir(exist_ok=True, parents=True)
                    target_path.write_bytes(zip_ref.read(member.filename))
        
        if progress_callback: progress_callback("java", "Java setup complete!")
        else: print("...Java setup complete!")
    except Exception as e:
        if progress_callback: progress_callback("java", f"An unexpected error occurred during Java setup: {e}")
        else: print(f"...error setting up Java: {e}")
    finally:
        if zip_path.exists():
            zip_path.unlink()

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
        
        # --- CRITICAL FIX ---
        # Ensure the droid shell script is executable *before* trying to run it.
        droid_script_path = dest_dir / "droid.sh"
        if sys.platform != "win32" and droid_script_path.exists():
            droid_script_path.chmod(droid_script_path.stat().st_mode | 0o111)

        # Run the signature update script. This requires Java to be present.
        if progress_callback: progress_callback("droid", "Updating DROID signature file...")
        else: print("...updating DROID signature file...")

        # Set up the environment for the subprocess to use our portable Java
        java_home_path = dest_dir.parent / "java"
        env = os.environ.copy()
        if java_home_path.exists():
            env["JAVA_HOME"] = str(java_home_path.resolve())
        subprocess.run([str(droid_script_path), "-d"], check=True, capture_output=True, env=env, text=True)

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
    models_dir = script_dir / "assets"
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
        # Download Java first, as DROID's setup script requires it.
        download_java(script_dir)
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