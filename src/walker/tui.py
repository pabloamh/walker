# walker/tui.py
import os
from pathlib import Path

from textual.app import App, ComposeResult
from textual.containers import ScrollableContainer
from textual.widgets import Button, Footer, Header, Log, ProgressBar
from textual.worker import Worker

from . import config
from .indexer import Indexer


class WalkerApp(App):
    """A Textual app for the Walker file indexer."""

    CSS_PATH = "tui.css"
    BINDINGS = [("d", "toggle_dark", "Toggle dark mode")]

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Header()
        yield Button("Start Indexing", id="start_button", variant="primary")
        yield ProgressBar(id="progress_bar", show_eta=False)
        with ScrollableContainer(id="log_container"):
            yield Log(id="log_output")
        yield Footer()

    def on_mount(self) -> None:
        """Called when the app is first mounted."""
        # Change to the script's directory to find walker.toml
        script_dir = Path(__file__).parent
        os.chdir(script_dir)
        self.query_one("#log_output").write_line(f"Working directory: {script_dir}")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle the 'Start Indexing' button press."""
        if event.button.id == "start_button":
            log_widget = self.query_one("#log_output")
            log_widget.clear()
            log_widget.write_line("Starting indexing process...")
            event.button.disabled = True
            self.run_worker(self.run_indexing, thread=True)

    def run_indexing(self) -> None:
        """Runs the indexer in a background worker thread."""
        app_config = config.load_config()
        log_widget = self.query_one("#log_output")

        # Use call_from_thread to safely update the UI from the worker
        def progress_callback(message: str):
            self.call_from_thread(log_widget.write_line, message)

        indexer_instance = Indexer(
            root_paths=tuple(),  # Use paths from config
            workers=app_config.workers,
            exclude_paths=tuple(),
            progress_callback=progress_callback,
        )
        indexer_instance.run()

        # Re-enable the button when done
        self.call_from_thread(self.query_one("#start_button").__setattr__, "disabled", False)

    def action_toggle_dark(self) -> None:
        """An action to toggle dark mode."""
        self.dark = not self.dark


if __name__ == "__main__":
    app = WalkerApp()
    app.run()