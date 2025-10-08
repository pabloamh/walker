# Walker File Indexer

A powerful and efficient file indexer that recursively scans a directory, extracts rich metadata from files, and stores it in a SQLite database for easy querying and analysis.
The application is built with a multiprocessing architecture to process files concurrently, making it fast and suitable for large collections of files.

## Features

- **Recursive Scanning**: Traverses entire directory trees to find all files.
- **Concurrent Processing**: Utilizes multiple worker threads to process files in parallel, significantly speeding up the indexing of large directories.
- **Rich Metadata Extraction**:
    - **Basic Info**: Path, filename, and size.
    - **Cryptographic Hash**: SHA-256 hash for file integrity and deduplication.
    - **MIME Type**: Identifies the file type using `libmagic`.
- **Specialized Content Extraction**:
    - **Images**: Generates perceptual hashes (p-hash) for similarity detection and extracts EXIF metadata.
    - **Documents**: Extracts text content from PDF (`.pdf`), Microsoft Word (`.docx`), and ODF (`.odt`) files.
    - **Videos**: Extracts media metadata like resolution, duration, and codecs.
    - **Audio**: Extracts metadata tags like artist, album, and title.
- **Text Content Extraction**: Extracts readable text from plain text files (`.txt`, `.md`), HTML, and email (`.eml`) files.
- **Archive Indexing**: Lists the contents of compressed files (`.zip`, `.tar`, `.tar.gz`, etc.).
- **Persistent Storage**: Saves all extracted metadata into a SQLite database (`file_indexer.db`).
- **Powerful Command-Line Interface**: Easy-to-use CLI built with Click for indexing and reporting.
- **Configurable Exclusions**: Smartly ignores system folders on Windows and allows users to specify custom directories to exclude.
- **Automatic File Filtering**: Ignores common temporary and system files (e.g., `.swp`, `.tmp`, `.DS_Store`, `Thumbs.db`).
- **Incremental Updates**: On subsequent runs, only processes new or modified files, making updates very fast.
- **Semantic Search**: Performs powerful, meaning-based searches on file content.
- **Memory-Efficient PII Detection**: Automatically scans text-based files for Personally Identifiable Information (PII), even on multi-gigabyte files.
- **High-Performance Reporting**: Includes highly optimized commands to find duplicate files, similar images, and textually similar documents.
- **Configuration File**: Uses a `walker.toml` file for persistent settings.

## Prerequisites

Before you begin, ensure you have the following installed on your system:

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
    ```bash
    git clone <your-repo-url>
    cd walker
    ```

2.  **Install Python Dependencies**:
    From the root of the project, run the following command to create a virtual environment and install the required packages:
    ```bash
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
# It supports glob patterns (e.g., "*.egg-info").
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
# PII (Personally Identifiable Information) detection. The default is English.
# The first language in the list is the primary one.
pii_languages = ["en", "es"]

# memory_limit_gb: A soft memory limit (in Gigabytes) for each worker process.
# This is useful on systems with limited RAM to prevent the OS from freezing.
# For example, `4.0` would limit each worker to 4 GB of memory.
# This feature is only supported on Linux and macOS. It is ignored on Windows.
# memory_limit_gb = 4.0

# embedding_model_path: Path to a locally saved sentence-transformer model.
# If set, the application will not need internet access to download it.
# The path is relative to the `src/walker/` directory.
# embedding_model_path = "models/all-MiniLM-L6-v2"
```

**Note**: Any options you provide on the command line will always take precedence over the settings in the `walker.toml` file.

## Offline Setup and Usage

The application uses several components that may require online access to download models or data on their first run. To use the application in a fully offline environment, you must pre-download these assets.

### Step 1: Download All Offline Assets

On a machine with internet access, run the provided `download_assets.py` script. This will download and cache all necessary models and data files into the `src/walker/models/` directory.

From your project's root directory, run:
```sh
poetry run python -m walker.main download-assets
```

This script will perform the following actions:
1.  **Download the `sentence-transformer` model** (`all-MiniLM-L6-v2`) and save it to `src/walker/models/all-MiniLM-L6-v2`.
2.  **Download the `spaCy` language model** (`en_core_web_lg`) required for PII detection.
3.  **Cache the Public Suffix List** used by `tldextract` (a dependency of the PII analyzer) and save it to `src/walker/models/tldextract_cache`.

### Step 2: Update Configuration for Offline Use

Uncomment and set the `embedding_model_path` in your `src/walker/walker.toml` file to point to the downloaded model directory.

```toml
[tool.walker]
# ... other settings ...

# Path to the locally saved sentence-transformer model.
embedding_model_path = "models/all-MiniLM-L6-v2"
```

### Step 3: Running Offline

When you deploy your application, ensure the `src/walker/models` directory (containing the downloaded assets) is included.

To ensure `tldextract` uses its local cache, you must **set an environment variable** before running the application.

```bash
# Set the cache directory path relative to where you run the command
export TLDEXTRACT_CACHE_DIR=./src/walker/models/tldextract_cache

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
-   `--workers INTEGER`: The number of worker threads to use for processing files.
-   `--exclude TEXT`: Directory name to exclude. Can be used multiple times.

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

```bash
# Search for a concept and get the top 5 results
poetry run python -m walker.main search "financial results for the last quarter" --limit 5
```

#### List Largest Files

This command lists the largest files in your index, helping you identify what is consuming the most disk space.

```bash
poetry run python -m walker.main largest-files --limit 25
```

#### Summarize by File Type

This command provides a summary of all indexed files, grouped by their type, showing the count and total size for each.

```bash
poetry run python -m walker.main type-summary
```

#### List Files with PII

This command lists all files that were flagged as potentially containing Personally Identifiable Information (PII) during the indexing process.

```bash
poetry run python -m walker.main list-pii-files
```