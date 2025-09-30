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
- **PII Detection**: Automatically scans text-based files for Personally Identifiable Information (PII).
- **Reporting**: Includes commands to find duplicate files, textually similar documents, and summarize disk usage.

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
