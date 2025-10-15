# tests/test_file_processor.py
import hashlib
from walker.file_processor import FileProcessor

def test_process_text_file(test_env):
    """
    Tests basic metadata extraction from a simple text file.
    """
    test_file = test_env / "test_data" / "file1.txt"
    processor = FileProcessor(file_path=test_file)

    # process() is a generator, so we consume it into a list
    results = list(processor.process())

    # We expect one metadata object for this simple file
    assert len(results) == 1
    metadata = results[0]

    # Verify basic attributes
    assert metadata.filename == "file1.txt"
    assert metadata.mime_type == "text/plain"
    assert metadata.size_bytes == 23

    # Verify content and hash
    assert metadata.content == "This is the first file."
    expected_hash = hashlib.sha256(b"This is the first file.").hexdigest()
    assert metadata.crypto_hash == expected_hash
    assert metadata.content_embedding is not None
    assert metadata.pii_types is None # No PII in this file