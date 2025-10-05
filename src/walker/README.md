# Walker File Indexer

A powerful and efficient file indexer that recursively scans a directory, extracts rich metadata from files, and stores it in a SQLite database for easy querying and analysis.

The application is built with a multi-threaded architecture to process files concurrently, making it fast and suitable for large collections of files.

## Features

- **Recursive Scanning**: Traverses entire directory trees to find all files.
- **Concurrent Processing**: Utilizes multiple worker threads to process files in parallel, significantly speeding up the indexing of large directories.
- **Rich Metadata Extraction**:
    - **Basic Info**: Path, filename, and size.
    - **Cryptographic Hash**: SHA-256 hash for file integrity and deduplication.
    - **MIME Type**: Identifies the file type using `libmagic`.
- **Specialized Content Extraction**:
    - **Images**: Generates perceptual hashes (p-hash) for similarity detection and extracts EXIF metadata.
    - **Documents**: Extracts text content from PDF (`.pdf`) and Microsoft Word (`.docx`) files.
    - **Videos**: Extracts media metadata like resolution, duration, and codecs.
    - **Audio**: Extracts metadata tags like artist, album, and title.
- **Text Content Extraction**: Extracts readable text from plain text files (`.txt`, `.md`), HTML, and email (`.eml`) files.
- **Archive Indexing**: Lists the contents of compressed files (`.zip`, `.tar`, `.tar.gz`, etc.).
- **Persistent Storage**: Saves all extracted metadata into a SQLite database (`file_indexer.db`).
- **Command-Line Interface**: Easy-to-use CLI built with Click.
- **Configurable Exclusions**: Smartly ignores system folders on Windows and allows users to specify custom directories to exclude.
- **Automatic File Filtering**: Ignores common temporary and system files (e.g., `.swp`, `.tmp`, `.DS_Store`, `Thumbs.db`).
- **Incremental Updates**: On subsequent runs, only processes new or modified files, making updates very fast.
- **Semantic Search**: Performs powerful, meaning-based searches on file content.
- **PII Detection**: Automatically scans text-based files for Personally Identifiable Information (PII).
- **Reporting**: Includes commands to find duplicate files, textually similar documents, and summarize disk usage.
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

For convenience, you can define your default settings in a `walker.toml` file. The application will automatically look for this file in the directory where you run the command. This is the recommended way to set options you use frequently.
The application will automatically look for this file in its own source directory (`src/walker/`). This is the recommended way to set options you use frequently.

### Example `walker.toml`

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
    "__pycache__",
    ".venv",
    ".git",
    "build",
    "dist",
    ".cache",
    "target", # For Rust projects
    "*.egg-info"
]
```

**Note**: Any options you provide on the command line will always take precedence over the settings in the `walker.toml` file.

## Offline Setup and Usage

The application uses several components that may require online access to download models or data on their first run. To use the application in a fully offline environment, you must pre-download these assets.

### Step 1: Download All Offline Assets

On a machine with internet access, run the provided `download_assets.py` script. This will download and cache all necessary models and data files into the `src/walker/` directory.

From your project's root directory, run:
```bash
poetry run python src/walker/download_assets.py
```

Run this script from your project's root directory:

```bash
poetry run python download_model.py
```

This will create a `models/all-MiniLM-L6-v2` directory containing the model files.

### Step 2: Update Configuration

Add the `embedding_model_path` setting to your `walker.toml` file to point to the downloaded model directory. This path should be relative to where you run the `walker` command.

```toml
[tool.walker]
# ... other settings ...

# Path to the locally saved sentence-transformer model.
embedding_model_path = "models/all-MiniLM-L6-v2"
```

When you deploy your application, simply include the `models` directory alongside it. The application will now load the model from this local path, removing the need for internet access.

## Usage

The application has multiple sub-commands.

### Indexing Files

To scan a directory and build or update your index, use the `index` command.

```bash
poetry run python -m walker.main index [ROOT_PATHS...] [OPTIONS]
```

**Arguments:**
-   `ROOT_PATH`: One or more directories to start scanning from.

**Options:**
-   `--workers INTEGER`: The number of worker threads to use for processing files.
-   `--exclude TEXT`: Directory name to exclude. Can be used multiple times.

**Examples:**

```bash
# Scan a directory using 8 worker threads
poetry run python -m walker.main index ~/Documents/my_files --workers 8

# Scan multiple directories at once
poetry run python -m walker.main index /path/to/photos /path/to/work-docs

# Scan a directory and exclude specific folders by name
# This will skip any folder named 'backups' or 'temp_folder' it encounters.
poetry run python -m walker.main index /media/archive --exclude backups --exclude temp_folder
```

### Reporting and Analysis

Once your index is built, you can run reports to find duplicate files and analyze your data.

#### Find Identical Files

This command finds all files that are bit-for-bit identical by comparing their SHA-256 hashes.

```bash
poetry run python -m walker.main find-dupes
```

#### Find Identical Images

This command finds images that are visually identical by comparing their perceptual hashes.

```bash
poetry run python -m walker.main find-image-dupes
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
