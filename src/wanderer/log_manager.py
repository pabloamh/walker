# wanderer/log_manager.py
import logging
import re
from collections import defaultdict
import atexit

class DeduplicatingLogHandler(logging.FileHandler):
    """
    A custom log handler that prevents the same message from being logged
    repeatedly, which is useful for reducing log file size when many files
    fail with the same error.
    """
    def __init__(self, filename, mode='a', encoding=None, delay=False):
        super().__init__(filename, mode, encoding, delay)
        self.message_counts = defaultdict(int)
        self.suppressed_counts = defaultdict(int)
        # This regex is designed to find and remove file paths to better group errors.
        self.path_regex = re.compile(r"['\"]?([a-zA-Z]:\\|/)[^:,'\"\s]+['\"]?")
        atexit.register(self.log_summary)

    def emit(self, record: logging.LogRecord):
        """
        Emit a record.

        If a message has been seen more than a few times, it is suppressed.
        """
        # Create a more generic key by removing file paths from the message.
        original_message = record.getMessage()
        msg_key = self.path_regex.sub("<PATH>", original_message)

        self.message_counts[msg_key] += 1

        # Log the first 5 occurrences of any given message fully.
        # After that, it's considered a repetitive error and is suppressed.
        if self.message_counts[msg_key] <= 5:
            super().emit(record)
            self.flush()  # Force write to disk immediately for debugging crashes.
        else:
            self.suppressed_counts[msg_key] += 1

    def log_summary(self):
        """Logs a summary of suppressed messages at exit."""
        if not self.suppressed_counts:
            return
        
        summary_message = "\n--- Logging Summary ---\n"
        for msg_key, count in self.suppressed_counts.items():
            summary_message += f"Suppressed {count} instances of: {msg_key}\n"
        self.stream.write(summary_message)