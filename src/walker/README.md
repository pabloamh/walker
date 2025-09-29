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
- **Persistent Storage**: Saves all extracted metadata into a SQLite database (`file_indexer.db`).
- **Command-Line Interface**: Easy-to-use CLI built with Click.

## Installation

This project uses Poetry for dependency management.

1.  **Prerequisites**:
    -   Python 3.10+
    -   Poetry
    -   `libmagic`: This is required by the `python-magic` library.

    On Debian/Ubuntu, you can install `libmagic` with:
    ```bash
    sudo apt-get update && sudo apt-get install libmagic1
    ```

    On macOS (using Homebrew):
    ```bash
    brew install libmagic
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

The application is run from the command line. You need to provide the path to the directory you want to scan.

```bash
poetry run python -m walker.main <ROOT_PATH> [OPTIONS]
```

### Arguments
-   `ROOT_PATH`: The directory to start scanning from.

### Options
-   `--workers INTEGER`: The number of worker threads to use for processing files. Defaults to 3.

### Example

To scan a directory named `~/Documents/my_files` using 8 worker threads:

```bash
poetry run python -m walker.main ~/Documents/my_files --workers 8
```

The application will create a `file_indexer.db` file in the project's root directory containing the metadata of all the processed files.
