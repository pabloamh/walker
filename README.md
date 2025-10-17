# Walker File Indexer

A powerful and efficient file indexer that recursively scans directories, extracts rich metadata from files, and stores it in a SQLite database for easy querying and analysis. Built with a multiprocessing architecture, `walker` is fast and suitable for large file collections.

## Features

- **Recursive Scanning**: Traverses entire directory trees to find all files.
- **Concurrent Processing**: Utilizes multiple worker processes to process files in parallel, significantly speeding up the indexing of large directories.
- **Rich Metadata Extraction**:
    - **Basic Info**: Path, filename, and size.
    - **Cryptographic Hash**: SHA-256 hash for file integrity and deduplication.
    - **MIME Type**: Identifies the file type using `libmagic`.
- **Specialized Content Extraction**:
    - **Images**: Generates perceptual hashes (p-hash) for similarity detection and extracts EXIF metadata.
    - **Documents**: Extracts text content from PDF (`.pdf`), Microsoft Word (`.docx`), and ODF (`.odt`) files.
    - **Videos**: Extracts media metadata like resolution, duration, and codecs.
    - **Audio**: Extracts metadata tags like artist, album, and title.
- **Deep Archive Scanning**: Extracts and individually processes files within archives (`.zip`, `.tar`, etc.), treating them as virtual folders.
- **Advanced File Identification**: Optionally uses **Fido** with **PRONOM** signatures for highly accurate file format identification, especially for files with generic MIME types.
- **Text Content Extraction**: Extracts readable text from plain text files (`.txt`, `.md`), HTML, and email (`.eml`) files.
- **Categorized PII Detection**: Scans for Personally Identifiable Information (PII) and reports the specific types found (e.g., `CREDIT_CARD_NUMBER`, `PHONE_NUMBER`).
- **Semantic Search**: Performs powerful, meaning-based searches on file content using AI embeddings.
- **Deferred Text Extraction**: Speed up initial scans by skipping text extraction, then process text content later with a dedicated command.
- **Configurable Archive Handling**: Exclude specific archive formats (like `.epub`, `.cbz`) from being extracted.
- **Incremental Updates**: On subsequent runs, only processes new or modified files, making updates very fast.
- **Scalable & Memory-Efficient**: Optimized to handle hundreds of thousands of files without running out of memory during reporting.
- **Powerful CLI**: Easy-to-use command-line interface built with Click for indexing and reporting.
- **Flexible Configuration**: Uses a `walker.toml` file for persistent settings and supports command-line overrides.
- **Automatic File Filtering**: Ignores common temporary and system files (e.g., `.swp`, `.tmp`, `.DS_Store`, `Thumbs.db`).

## Quick Start

1.  **Install prerequisites** (Python 3.11+, Poetry, `libmagic`, `mediainfo`).
2.  **Clone and install**:
    ```sh
    git clone <your-repo-url>
    cd walker
    poetry install
    ```
3.  **Download offline assets** (AI models, etc.):
    ```sh
    poetry run python -m walker.main download-assets
    ```
4.  **Configure `src/walker/walker.toml`** to set your scan directories.
5.  **Run the indexer**:
    ```sh
    poetry run python -m walker.main index
    ```

## Prerequisites

Before you begin, ensure you have the following installed:
-   **Python 3.11+**
-   **Poetry** for managing Python dependencies.
-   **`libmagic`**: Required by the `python-magic` library for MIME type detection.
-   **`mediainfo`**: Required for video metadata extraction.

### System Dependency Installation

**On Debian/Ubuntu:**
```bash
sudo apt-get update && sudo apt-get install libmagic1 mediainfo
```

**On macOS (using Homebrew):**
```bash
brew install libmagic mediainfo
```

## Installation

This project uses Poetry for dependency management.

1.  **Clone the Repository**:
    ```sh
    git clone <your-repo-url>
    cd walker
    ```

2.  **Install Python Dependencies**:
    From the root of the project, run the following command to create a virtual environment and install the required packages:
    ```sh
    poetry install
    ```

## Configuration

For convenience, you can define your default settings in a `walker.toml` file. The application will automatically look for this file in the `src/walker/` directory. This is the recommended way to set options you use frequently.

### Example `src/walker/walker.toml`

All settings for `walker` must be placed under the `[tool.walker]` section. Here is a comprehensive example:

```toml
[tool.walker]
# workers: Set the default number of worker threads for processing.
workers = 8

# db_batch_size: The number of file records to batch together before
# writing to the database.
db_batch_size = 1000

# scan_dirs: A list of directories to scan by default if no paths are
# provided on the command line.
# This supports user home directory expansion (e.g., "~/Documents").
# scan_dirs = ["~/Documents", "~/Pictures", "/media/archive/work"]
scan_dirs = []

# exclude_dirs: A list of directory names to always exclude from scanning.
# This is useful for ignoring common development, temporary, or cache folders.
# It supports simple names (e.g., "node_modules"), full paths
# (e.g., "/media/archive/do_not_scan"), and glob patterns (e.g., "*.egg-info").
# All checks are case-insensitive.
exclude_dirs = [
    "node_modules",
    "bower_components", # JS package manager
    "vendor",           # PHP package manager (Composer)
    # --- Python ---
    ".git",
    ".venv",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "build",
    "dist",
    "*.egg-info",
    # --- JS/Web Development ---
    ".idea",
    ".vscode",
    ".cache",
    ".next",  # Next.js build folder
    ".nuxt",  # Nuxt.js build folder
    # --- Other Build/Cache ---
    "target", # For Rust projects
    # --- Virtual/System Filesystems (especially for Linux) ---
    "proc",
    "sys",
    "dev",
    "run",
    # --- macOS Specific ---
    ".DocumentRevisions-V100",
    ".fseventsd",
    ".Spotlight-V100",
    ".Trashes",
    "private",
    # --- Windows/macOS User Dirs (use with caution if you want to scan these) ---
    # "AppData", # Windows AppData
    # "Library", # macOS Library
]

# pii_languages: A list of language codes (e.g., "en", "es", "fr") to use for
# PII (Personally Identifiable Information) detection. All specified languages
# will be used to scan each file.
pii_languages = ["en", "es"]

# memory_limit_gb: A soft memory limit (in Gigabytes) for each worker process.
# This is useful on systems with limited RAM to prevent the OS from freezing.
# For example, `4.0` would limit each worker to 4 GB of memory.
# This feature is only supported on Linux and macOS. It is ignored on Windows.
# memory_limit_gb = 4.0

# embedding_model_path: Path to a locally saved sentence-transformer model.
# If set, the application will not need internet access to download it.
# The path is resolved relative to the location of this `walker.toml` file.
# embedding_model_path = "models/all-MiniLM-L6-v2"

# use_fido: Enable Fido for more accurate file type identification.
# use_fido = true

# extract_text_on_scan: If set to false, the initial `index` command will
# skip the time-consuming process of extracting text content from documents.
# This allows for a much faster initial scan. You can then use the
# `refine-text` command later to process only the text-based files.
# extract_text_on_scan = false

# archive_exclude_extensions: A list of file extensions for archive-like
# files that should NOT be extracted. This is useful for formats like e-books
# that are technically zip files but should be treated as single items.
# Defaults include .epub, .cbz, .cbr.
# archive_exclude_extensions = [".epub", ".cbz", ".cbr", ".apk"]
```
## Offline Setup and Usage

The application uses several components that may require online access to download models or data on their first run. To use the application in a fully offline environment, you must pre-download these assets.

### Step 1: Download All Offline Assets

On a machine with internet access, run the provided `download_assets.py` script. This will download and cache all necessary models and data files into the `src/walker/models/` directory.

From the project's root directory, run:
```sh
poetry run python -m walker.main download-assets
```

This script will perform the following actions:
1.  **Download the `sentence-transformer` model** (`all-MiniLM-L6-v2`) and save it to `src/walker/models/all-MiniLM-L6-v2`.
2.  **Download the `spaCy` language model** (`en_core_web_lg`) required for PII detection.
3.  **Download `spaCy` language models** for all languages configured in `pii_languages` in your `walker.toml`.
4.  **Cache the Public Suffix List** used by `tldextract` (a dependency of the PII analyzer) and save it to `src/walker/models/tldextract_cache`.
5.  **Download the PRONOM signature file** used by `fido` if `use_fido = true` is set in your config.

### Step 2: Update Configuration for Offline Use

Update your `walker.toml` to point to the downloaded assets.

```toml
[tool.walker]
# ... other settings ...

# Path to the locally saved sentence-transformer model.
embedding_model_path = "models/all-MiniLM-L6-v2"

# Enable Fido if you downloaded its assets.
use_fido = true
```

### Step 3: Running Offline

When you deploy your application, ensure the `src/walker/models` directory (containing the downloaded assets) is included.

To ensure `tldextract` and `fido` use their local caches, you must **set environment variables** before running the application.

```bash
# Set environment variables to point to the cached assets.
# These paths should be absolute or relative to where you run the command.
export TLDEXTRACT_CACHE_DIR="$(pwd)/src/walker/models/tldextract_cache"

# The FIDO_SIG_FILE variable tells fido where to find its signature file.
export FIDO_SIG_FILE="$(pwd)/src/walker/models/fido_cache/DROID_SignatureFile.xml"

# Now run the indexer
poetry run python -m walker.main index
```

With these steps completed, the application will be fully functional without requiring any internet access.

## Usage

The application has multiple sub-commands.

### Indexing Files

To scan a directory and build or update your index, use the `index` command. The application will automatically change its working directory to `src/walker/` to ensure all paths are resolved correctly.

```bash
poetry run python -m walker.main index [ROOT_PATHS...] [OPTIONS]
```

**Arguments:**
-   `ROOT_PATH`: One or more directories to start scanning from.

**Options:**

### Refining Unknown Files

After an initial scan, you can run this command to re-process any files that were identified with a generic MIME type (like `application/octet-stream`). It uses `fido` to attempt a more accurate identification. This requires `use_fido = true` in your `walker.toml`.

```bash
poetry run python -m walker.main refine-unknowns
```

This is useful for improving your data quality without slowing down the initial indexing process.

### Reporting and Analysis

Once your index is built, you can run reports to find duplicates, analyze your data, and perform searches.

#### Find Identical Files

This command finds all files that are bit-for-bit identical by comparing their SHA-256 hashes. It will identify the oldest file in each set as the "Source".

```bash
poetry run python -m walker.main find-dupes
```

#### Find Similar Images

This command finds images that are visually identical or similar. By default, it finds exact duplicates. Use the `--threshold` to find similar images (a lower number is stricter).

```bash
# Find exact duplicates
poetry run python -m walker.main find-image-dupes

# Find very similar images (e.g., different resolutions or minor edits)
poetry run python -m walker.main find-image-dupes --threshold 4
```

#### Find Similar Text

This command uses AI model embeddings to find documents with similar text content. You can adjust the strictness with the `--threshold` option (1.0 is nearly identical, 0.8 is loosely related).

```bash
# Find documents that are at least 95% similar
poetry run python -m walker.main find-similar-text --threshold 0.95
```

#### Semantic Search

Performs a powerful semantic search across the content of all indexed text files. This finds files based on meaning, not just keywords.

```sh
# Search for a concept and get the top 5 results
poetry run python -m walker.main search "financial results for the last quarter" --limit 5
```

#### List Largest Files

This command lists the largest files in your index, helping you identify what is consuming the most disk space.

```sh
poetry run python -m walker.main largest-files --limit 25
```

#### Summarize by File Type

This command provides a summary of all indexed files, grouped by their type, showing the count and total size for each.

```bash
poetry run python -m walker.main type-summary
```

#### List Files with PII

This command lists all files flagged for containing PII and shows the specific categories of information found (e.g., `PERSON`, `PHONE_NUMBER`).

```bash
poetry run python -m walker.main list-pii-files
```