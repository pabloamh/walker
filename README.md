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

## Installation

This project uses Poetry for dependency management.

1.  **Prerequisites**:
    -   Python 3.10+
    -   Poetry
    -   `libmagic`: This is required by the `python-magic` library.
    -   `mediainfo`: This is required for video metadata extraction.
    -   A `spaCy` model for PII detection.

    On Debian/Ubuntu, you can install `libmagic` with:
    ```bash
    sudo apt-get update && sudo apt-get install libmagic1
    ```
    And `mediainfo` with:
    ```bash
    sudo apt-get update && sudo apt-get install mediainfo
    ```

    On macOS (using Homebrew):
    ```bash
    brew install libmagic mediainfo
    ```

    After installing the Python dependencies, you must also download the English language model for `spaCy`:
    ```bash
    poetry run python -m spacy download en_core_web_lg
    ```


2.  **Clone the repository**:
    ```bash
    git clone <your-repo-url>
    cd walker
    ```

3.  **Install dependencies**:
    From the root of the project, run the following command to create a virtual environment and install the required packages:
    ```bash
    poetry install
    ```

## Usage

The application has multiple sub-commands.

### Indexing Files

To scan a directory and build or update your index, use the `index` command.

```bash
poetry run python -m walker.main index <ROOT_PATH_1> [ROOT_PATH_2] ... [OPTIONS]
```

**Arguments:**
-   `ROOT_PATH`: One or more directories to start scanning from.

**Options:**
-   `--workers INTEGER`: The number of worker threads to use for processing files.
-   `--exclude TEXT`: Directory name to exclude. Can be used multiple times.

**Example:**
```bash
# Scan a directory using 8 worker threads
poetry run python -m walker.main index ~/Documents/my_files --workers 8
```

The application will create a `file_indexer.db` file in the project's root directory containing the metadata of all the processed files.

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

## Configuration

For convenience, you can define your default settings in a `walker.toml` file. The application will automatically look for this file in the directory where you run the command.

This is the recommended way to set options you use frequently, like the number of workers or a standard list of folders to exclude.

**Note**: Any options you provide on the command line will always take precedence over the settings in the `walker.toml` file.

### Example `walker.toml`

All settings for `walker` must be placed under the `[tool.walker]` section in the TOML file.

```toml
[tool.walker]
# Set the default number of worker threads for processing.
workers = 8

# Define a list of directory names to always exclude from scanning.
# This is useful for ignoring common development or temporary folders.
exclude_dirs = [
    "node_modules",
    "__pycache__",
    ".venv",
    ".git",
    "build",
    "dist",
]
```
