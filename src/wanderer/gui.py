import flet as ft
from . import config
from . import indexer
import attrs
from . import database, models
import asyncio
import tomllib, spacy
from pathlib import Path
from . import download_assets
from datetime import datetime
from typing import Optional
import threading


class WandererGUI:
    def __init__(self, page: ft.Page):
        self.page = page
        self.app_config, self.config_path = config.load_config_with_path()
        self.page.title = "Wanderer"
        self.page.window_width = 1000
        self.page.window_height = 700
        self.page.vertical_alignment = ft.MainAxisAlignment.START
    
    async def start_scan_click(self, e):
        """Handles the 'Start Scan' button click."""
        self.page.run_task(self._set_scan_ui_state, is_scanning=True)

        # --- Backend Logic: Run indexer in a thread ---
        def scan_job(page):
            """The actual scanning function to run in a thread."""
            
            # This callback will be called from the indexer thread.
            # We use page.run_task to update the UI from the main thread.
            def progress_callback(value, total, description):
                async def update_ui():
                    self.scan_status_text.value = description
                    if total is not None and total > 1:
                        self.scan_progress.value = value / total
                    else:
                        self.scan_progress.value = None # Indeterminate
                    page.update()
                self.page.run_task(update_ui)

            # Get selected paths from the GUI
            selected_paths = [
                Path(cb.label) for cb in self.scan_targets_list.controls if isinstance(cb, ft.Checkbox) and cb.value
            ]

            # Create a temporary config object from the UI settings for this scan
            scan_config = attrs.evolve(
                self.app_config,
                extract_text_on_scan=self.scan_option_text.value,
                compute_perceptual_hash=self.scan_option_phash.value,
                use_fido=self.scan_option_fido.value,
            )
            
            idx = indexer.Indexer(root_paths=tuple(selected_paths), workers=self.app_config.workers, memory_limit_gb=self.app_config.memory_limit_gb, exclude_paths=(), app_config=scan_config, progress_callback=progress_callback)
            try:
                idx.run()
            finally:
                # Reset UI after scan and refresh history
                async def finish_scan():
                    await self.stop_scan_click(message="Scan finished.")
                    await self.refresh_scan_history(page)
                self.page.run_task(finish_scan)

        threading.Thread(target=scan_job, args=(self.page,), daemon=True).start()

    async def stop_scan_click(self, e=None, message: str | None = None):
        # In a future implementation, this could signal the scan thread to stop.
        await self._set_scan_ui_state(is_scanning=False, message=message or "Scan stopped by user.")

    async def _set_scan_ui_state(self, is_scanning: bool, message: Optional[str] = None):
        """Helper to toggle the UI between scanning and idle states."""
        self.start_scan_button.visible = not is_scanning
        self.stop_scan_button.visible = is_scanning
        self.scan_progress.visible = is_scanning

        if is_scanning:
            self.scan_progress.value = None  # Indeterminate
            self.scan_status_text.value = message or "Starting scan..."
        else:
            self.scan_status_text.value = message or "Idle."

        # Toggle controls
        for control in self.scan_targets_list.controls:
            if isinstance(control, ft.Checkbox):
                control.disabled = is_scanning
        self.scan_option_text.disabled = is_scanning
        self.scan_option_phash.disabled = is_scanning
        self.scan_option_fido.disabled = is_scanning
        
        self.page.update()
        
    async def refresh_scan_history(self, page: ft.Page):
        """Refreshes the scan history list by querying the database in a background thread."""
        def fetch_history_job():
            with database.get_session() as db:
                logs = db.query(models.ScanLog).order_by(models.ScanLog.start_time.desc()).limit(20).all()
            
            async def update_ui_with_history():
                self.scan_history_list.controls.clear()
                if not logs:
                    self.scan_history_list.controls.append(ft.Text("No scan history found."))
                else:
                    for log in logs:
                        status_color = {
                                "completed": ft.Colors.GREEN,
                                "failed": ft.Colors.RED,
                                "started": ft.Colors.ORANGE,
                            }.get(log.status, ft.Colors.GREY)

                        end_time_str = log.end_time.strftime('%Y-%m-%d %H:%M:%S') if log.end_time else "In Progress"
                        duration = (log.end_time - log.start_time) if log.end_time else "N/A"

                        self.scan_history_list.controls.append(
                                ft.ListTile(
                                    leading=ft.Icon(ft.Icons.HISTORY, color=status_color),
                                    title=ft.Text(f"Scan on {log.start_time.strftime('%Y-%m-%d %H:%M:%S')}"),
                                    subtitle=ft.Text(f"Status: {log.status} | Files: {log.files_scanned} | Duration: {duration}"),
                                )
                            )
                await page.update_async()
            self.page.run_task(update_ui_with_history)
        
        threading.Thread(target=fetch_history_job, daemon=True).start()

    def run_refine_job(self, e, refine_type: str):
        """Generic handler to run a refinement job in the background."""

        def refine_thread_job(page):
            """The actual refinement function to run in a thread."""
            # --- UI Update: Show processing state (from thread) ---
            async def set_processing_state():
                for btn in self.refine_buttons:
                    btn.disabled = True
                self.refine_progress.visible = True
                self.refine_progress.value = None  # Indeterminate
                self.refine_status_text.value = f"Starting {refine_type} refinement..."
                await page.update_async()
            page.run_task(set_processing_state)

            # Create a config object that reflects the main settings, but forces the correct
            # processing for the specific refinement job.
            refine_config = attrs.evolve(
                self.app_config,
                extract_text_on_scan=refine_type == "text",
                compute_perceptual_hash=False,  # Not used in these jobs
                use_fido=refine_type == "fido",
            )

            idx = indexer.Indexer(root_paths=(), workers=self.app_config.workers, memory_limit_gb=self.app_config.memory_limit_gb, exclude_paths=(), app_config=refine_config)
            
            if refine_type == "fido":
                idx.refine_unknown_files()
            elif refine_type == "text":
                # Ensure text extraction is enabled for this specific run
                idx.app_config.extract_text_on_scan = True
                idx.refine_text_content()
            # Add other refinement types here if needed in the future

            # --- UI Update: Reset to idle state ---
            async def finish_refine():
                for btn in self.refine_buttons:
                    btn.disabled = False
                self.refine_progress.visible = False
                self.refine_status_text.value = f"{refine_type.capitalize()} refinement finished."
                await page.update_async()
            
            page.run_task(finish_refine)

        threading.Thread(target=refine_thread_job, args=(self.page,), daemon=True).start()

    def show_report_results(self, report_name: str):
        """Placeholder function to display report results."""
        self.report_content_area.content = ft.Column([
            ft.IconButton(icon=ft.Icons.ARROW_BACK, on_click=self.show_report_list, tooltip="Back to reports list"),
            ft.Text(f"Results for: {report_name}", style=ft.TextThemeStyle.HEADLINE_SMALL),
            ft.Text("This feature is not yet implemented."),
        ])
        await self.page.update_async()

    def show_report_list(self, e=None):
        self.report_content_area.content = self.reports_list_view
        self.page.run_task(self.page.update)

    def remove_scan_dir(self, dir_to_remove: str):
        self.app_config.scan_dirs.remove(dir_to_remove)
        self.page.run_task(self.update_list_views)

    def remove_exclude_dir(self, dir_to_remove: str):
        self.app_config.exclude_dirs.remove(dir_to_remove)
        self.page.run_task(self.update_list_views)

    async def update_list_views(self):
        self.scan_dirs_list.controls.clear()
        for d in sorted(self.app_config.scan_dirs):
            self.scan_dirs_list.controls.append(
                ft.Row([ft.Text(d, expand=True), ft.IconButton(ft.Icons.DELETE_OUTLINE, on_click=lambda e, dir=d: self.remove_scan_dir(dir), tooltip="Remove")])
            )
        self.scan_targets_list.controls.clear() # Clear and re-add checkboxes for scan tab
        if self.app_config.scan_dirs:
            for d in self.app_config.scan_dirs:
                self.scan_targets_list.controls.append(ft.Checkbox(label=d, value=True))
        else:
            self.scan_targets_list.controls.append(ft.Text("No scan directories configured. Please add some in Settings."))
        self.exclude_dirs_list.controls.clear()
        for d in sorted(self.app_config.exclude_dirs):
            self.exclude_dirs_list.controls.append(
                ft.Row([ft.Text(d, expand=True), ft.IconButton(ft.Icons.DELETE_OUTLINE, on_click=lambda e, dir=d: self.remove_exclude_dir(dir), tooltip="Remove")])
            )
        self.page.update()

    async def on_directory_picked(self, e: ft.FilePickerResultEvent):
        if e.path:
            if e.path not in self.app_config.scan_dirs:
                self.app_config.scan_dirs.append(e.path)
                # Also update the scan targets list on the main scan page
                self.start_scan_button.disabled = False
                await self.update_list_views()

    def add_excluded_dir(self, e: ft.ControlEvent):
        e.control.disabled = True # Disable button to prevent multiple clicks
        dir_to_exclude = self.new_exclude_dir_field.value
        if dir_to_exclude and dir_to_exclude not in self.app_config.exclude_dirs:
            self.app_config.exclude_dirs.append(dir_to_exclude)
            self.new_exclude_dir_field.value = ""
            self.page.run_task(self.update_list_views)

    def save_settings_click(self, e):
        """Saves the current settings from the UI to the wanderer.toml file."""
        # --- UI Thread: Disable button and gather data ---
        e.control.disabled = True
        self.page.update()

        def save_job(page, button, settings_data):
            """The actual saving function to run in a thread."""
            try:
                # Update the app_config object with the data gathered from the UI thread
                attrs.evolve(self.app_config, **settings_data)
                if not self.config_path:
                    return

                with open(self.config_path, "rb") as f:
                    full_toml = tomllib.load(f)

                if "tool" not in full_toml:
                    full_toml["tool"] = {}
                if "wanderer" not in full_toml["tool"]:
                    full_toml["tool"]["wanderer"] = {}

                # Update the values
                full_toml["tool"]["wanderer"] = config.config_to_dict(self.app_config)
                config.save_config_to_path(full_toml, self.config_path)

                # Schedule UI updates on the main thread
                async def update_ui_after_save():
                    self.page.snack_bar = ft.SnackBar(content=ft.Text("Settings saved successfully!"), action="OK")
                    self.page.snack_bar.open = True
                    button.disabled = False # Re-enable button
                    await page.update_async()
                    # After saving, kick off the asset status check
                    self.update_asset_status(page)
                page.run_task(update_ui_after_save)

            except Exception as ex:
                async def show_error():
                    self.page.snack_bar = ft.SnackBar(content=ft.Text(f"Error saving config: {ex}"))
                    self.page.snack_bar.open = True
                    button.disabled = False # Re-enable button on error
                    await page.update_async()
                page.run_task(show_error)

        # Gather all data from UI controls *before* starting the thread.
        settings_from_ui = {
            "workers": int(self.workers_field.value),
            "db_batch_size": int(self.db_batch_size_field.value),
            "memory_limit_gb": float(self.memory_limit_field.value) if self.memory_limit_field.value else None,
            "use_fido": self.use_fido_switch.value,
            "extract_text_on_scan": self.extract_text_switch.value,
            "compute_perceptual_hash": self.phash_switch.value,
            "pii_languages": [lang.strip() for lang in self.pii_languages_field.value.split(',') if lang.strip()],
            "archive_exclude_extensions": [ext.strip() for ext in self.archive_excludes_field.value.split(',') if ext.strip()],
            "embedding_model_path": self.embedding_model_path_field.value or None,
        }

        threading.Thread(target=save_job, args=(self.page, e.control, settings_from_ui), daemon=True).start()

    def run_asset_download(self, e, asset_type: str):
        """Handles the download process for a given asset in a background thread."""

        def download_job(page, button):
            """The actual download function to run in a thread."""
            async def set_downloading_state():
                button.text = "Downloading..."
                button.icon = ft.ProgressRing(width=16, height=16, stroke_width=2)
                button.disabled = True
                await page.update_async()
            page.run_task(set_downloading_state)

            script_dir = Path(__file__).parent
            models_dir = script_dir / "models"
            if asset_type == "embedding":
                download_assets.download_sentence_transformer('all-MiniLM-L6-v2', models_dir / 'all-MiniLM-L6-v2')
            elif asset_type == "pii":
                for lang in self.app_config.pii_languages:
                    model_name = config.get_spacy_model_name(lang)
                    download_assets.download_spacy_model(model_name, lang)
            elif asset_type == "fido":
                download_assets.cache_fido_signatures(models_dir / 'fido_cache')
            
            # After download, re-run the status check
            self.update_asset_status(page)

        # Run the download in a separate thread to not block the UI
        threading.Thread(target=download_job, args=(self.page, e.control), daemon=True).start()

    def update_asset_status(self, page: ft.Page):
        """Checks for offline assets and updates the UI accordingly."""

        def check_assets_job(page):
            """The actual checking function to run in a thread."""
            async def set_checking_state():
                self.asset_status_progress.visible = True
                for btn in [self.download_embedding_button, self.download_pii_button, self.download_fido_button]:
                    btn.disabled = True
                await page.update_async()
            page.run_task(set_checking_state)

            script_dir = Path(__file__).parent
            models_dir = script_dir / "models"

            # 1. Check Embedding Model
            model_path = script_dir / (self.app_config.embedding_model_path or "models/all-MiniLM-L6-v2")
            embedding_ok = model_path.is_dir()

            # 2. Check PII (spaCy) Models
            pii_ok = True
            for lang in self.app_config.pii_languages:
                model_name = config.get_spacy_model_name(lang)
                if not spacy.util.is_package(model_name):
                    pii_ok = False
                    break
            
            # 3. Check Fido Signatures
            fido_sig_path = models_dir / "fido_cache" / "DROID_SignatureFile.xml"
            fido_ok = fido_sig_path.is_file()

            async def update_ui_after_check():
                # Embedding Model UI
                self.embedding_model_status.value = "Available" if embedding_ok else "Not Found"
                self.embedding_model_status.color = ft.Colors.GREEN if embedding_ok else ft.Colors.ORANGE
                self.download_embedding_button.disabled = embedding_ok
                self.download_embedding_button.text = "Downloaded" if embedding_ok else "Download"
                self.download_embedding_button.icon = ft.icons.CHECK if embedding_ok else ft.icons.DOWNLOAD

                # PII Models UI
                self.pii_model_status.value = "Available" if pii_ok else "Missing"
                self.pii_model_status.color = ft.Colors.GREEN if pii_ok else ft.Colors.ORANGE
                self.download_pii_button.disabled = pii_ok
                self.download_pii_button.text = "Downloaded" if pii_ok else "Download"
                self.download_pii_button.icon = ft.icons.CHECK if pii_ok else ft.icons.DOWNLOAD

                # Fido Signatures UI
                self.fido_status.value = "Available" if fido_ok else "Not Found"
                self.fido_status.color = ft.Colors.GREEN if fido_ok else ft.Colors.ORANGE
                self.download_fido_button.disabled = fido_ok or not self.app_config.use_fido
                self.download_fido_button.text = "Downloaded" if fido_ok else "Download"
                self.download_fido_button.icon = ft.icons.CHECK if fido_ok else ft.icons.DOWNLOAD

                self.asset_status_progress.visible = False
                await page.update_async()

            page.run_task(update_ui_after_check)

        threading.Thread(target=check_assets_job, args=(page,), daemon=True).start()

    async def navigate(self, e):
        """Handles navigation rail selection changes."""
        # Clear the main content area
        if e.control.selected_index == 0:
            self.main_content.content = self.view_scan
        elif e.control.selected_index == 1:
            self.main_content.content = self.view_search
        elif e.control.selected_index == 2:
            self.main_content.content = self.view_reports
        elif e.control.selected_index == 3:
            self.main_content.content = self.view_settings
        elif e.control.selected_index == 4:
            self.main_content.content = self.view_help
        elif e.control.selected_index == 5:
            self.main_content.content = self.view_about
        await self.page.update_async()

    async def window_event(self, e):
        if e.data == "close":
            self.page.window_destroy()

    async def build(self):
        # --- Scan View ---
        self.scan_targets_list = ft.ListView(expand=False, spacing=5, height=200)
        if self.app_config.scan_dirs:
            for d in self.app_config.scan_dirs:
                self.scan_targets_list.controls.append(ft.Checkbox(label=d, value=True)) # type: ignore
        else:
            self.scan_targets_list.controls.append(ft.Text("No scan directories configured. Please add some in Settings.")) # type: ignore

        self.scan_option_text = ft.Checkbox(label="Extract Text & PII", value=self.app_config.extract_text_on_scan)
        self.scan_option_phash = ft.Checkbox(label="Compute Perceptual Hashes", value=self.app_config.compute_perceptual_hash)
        self.scan_option_fido = ft.Checkbox(label="Enable Fido", value=self.app_config.use_fido)

        self.start_scan_button = ft.ElevatedButton("Start Scan", icon=ft.Icons.PLAY_ARROW_ROUNDED, on_click=self.start_scan_click, disabled=not self.app_config.scan_dirs)
        self.stop_scan_button = ft.ElevatedButton("Stop Scan", icon=ft.Icons.STOP_ROUNDED, on_click=self.stop_scan_click, visible=False)
        self.scan_progress = ft.ProgressBar(width=400, visible=False)
        self.scan_status_text = ft.Text("Idle.", italic=True)

        new_scan_view = ft.Column(
            controls=[
                ft.Text("Select directories to scan:"), self.scan_targets_list,
                ft.Text("Scan Options:"), ft.Row(controls=[self.scan_option_text, self.scan_option_phash, self.scan_option_fido]),
                ft.Row([self.start_scan_button, self.stop_scan_button]), self.scan_progress, self.scan_status_text],
            spacing=15,
        )

        # --- Scan History View ---
        self.scan_history_list = ft.ListView(expand=True, spacing=10)
        scan_history_view = ft.Column(
            controls=[
                ft.Row([ft.Text("Recent Scans"), ft.IconButton(icon=ft.Icons.REFRESH, on_click=lambda e: self.page.run_task(self.refresh_scan_history, self.page), tooltip="Refresh History")]),
                self.scan_history_list
            ], expand=True
        )

        # --- Refine View ---
        self.refine_status_text = ft.Text("Idle.", italic=True)
        self.refine_progress = ft.ProgressBar(width=400, visible=False)

        def refine_fido_click(e): self.run_refine_job(e, "fido")
        def refine_text_click(e): self.run_refine_job(e, "text")

        self.refine_buttons = [
            ft.ElevatedButton("Refine Unknown Files (Fido)", on_click=refine_fido_click, tooltip="Rescan files with generic types like 'application/octet-stream' using Fido for better accuracy.", disabled=not self.app_config.use_fido), # type: ignore
            ft.ElevatedButton("Refine Skipped Text", on_click=refine_text_click, tooltip="Extract text, PII, and embeddings for text-based files that were skipped in a fast initial scan."),
        ]

        refine_data_view = ft.Column(
            controls=[
                ft.Text("Run deep analysis on files already in the database. This is useful after a fast initial scan."),
                ft.Row(controls=self.refine_buttons),
                self.refine_progress,
                self.refine_status_text,
            ],
            spacing=15,
        )

        self.view_scan = ft.Column(
            controls=[
                ft.Text("Analysis & Refinement", style=ft.TextThemeStyle.HEADLINE_MEDIUM),
                ft.Tabs(
                    selected_index=0,
                    tabs=[
                        ft.Tab(text="New Scan", content=new_scan_view),
                        ft.Tab(text="Scan History", content=scan_history_view),
                        ft.Tab(text="Refine Data", content=refine_data_view),
                    ],
                ),
            ], expand=True,
            spacing=15,
        )

        # --- Search View ---
        search_explanation = """
This isn't your typical keyword search! It finds files based on the *meaning* of your query. Think of it like asking a question.

**For example, you could search for:**
*   "photos from my beach vacation"
*   "that chicken recipe my aunt sent me"
*   "summary of the project alpha meeting"

The search will find documents and notes that are conceptually related to what you're looking for.

**Note**: This feature requires an AI model. It will be downloaded automatically on first use (requires internet), or you can download it from the **Settings > Offline Assets** page for offline use.
"""
        self.view_search = ft.Column(
        controls=[
            ft.Text("Semantic Search", style=ft.TextThemeStyle.HEADLINE_MEDIUM),
            ft.Markdown(search_explanation),
            ft.TextField(label="Search for a concept or idea..."),
            ft.ElevatedButton("Search")
        ],
        spacing=15,
    )

        # --- Reports View ---
        reports_available = [
        {"name": "Find Identical Files", "description": "Finds files with identical content (SHA-256 hash).", "icon": ft.Icons.CONTENT_COPY},
        {"name": "Find Similar Images", "description": "Finds visually similar images using perceptual hashing.", "icon": ft.Icons.IMAGE_SEARCH},
        {"name": "Find Similar Text", "description": "Finds documents with similar meaning using AI embeddings.", "icon": ft.Icons.TEXT_SNIPPET},
        {"name": "List Largest Files", "description": "Lists the largest files in the index by size.", "icon": ft.Icons.FOLDER_ZIP_OUTLINED},
        {"name": "File Type Summary", "description": "Summarizes file counts and sizes by MIME type.", "icon": ft.Icons.PIE_CHART_OUTLINE},
        {"name": "PRONOM ID Summary", "description": "Summarizes file counts by their PRONOM signature.", "icon": ft.Icons.BAR_CHART},
        {"name": "List Unique Files", "description": "Lists files that have no content duplicates.", "icon": ft.Icons.DIAMOND_OUTLINED},
        {"name": "List Files with PII", "description": "Lists all files flagged for containing potential PII.", "icon": ft.Icons.PRIVACY_TIP_OUTLINED},
    ]

        self.reports_list_view = ft.ListView(expand=True, spacing=10)
        for report in reports_available:
            self.reports_list_view.controls.append(
            ft.ListTile(
                leading=ft.Icon(report["icon"]), # type: ignore
                title=ft.Text(report["name"]),
                subtitle=ft.Text(report["description"]),
                    on_click=lambda e, r=report["name"]: self.show_report_results(r), # type: ignore
            )
        )

        self.report_content_area = ft.Container(content=self.reports_list_view, expand=True)
        self.view_reports = ft.Column(
            [ft.Text("Reports", style=ft.TextThemeStyle.HEADLINE_MEDIUM), self.report_content_area], expand=True
        )

        # --- Settings View ---
        self.scan_dirs_list = ft.ListView(expand=True, spacing=5)
        self.exclude_dirs_list = ft.ListView(expand=True, spacing=5)

        directory_picker = ft.FilePicker(on_result=self.on_directory_picked)
        self.page.overlay.insert(0, directory_picker)

        self.new_exclude_dir_field = ft.TextField(label="Add exclusion (e.g., 'node_modules', '*.log')", expand=True, on_submit=self.add_excluded_dir)

        config_path_text = ft.Text(
            f"Editing: {self.config_path}" if self.config_path else "No wanderer.toml found. Using default settings.",
            italic=True
        )
        self.workers_field = ft.TextField(
            label="Worker Processes",
                value=str(self.app_config.workers),
            width=150,
            keyboard_type=ft.KeyboardType.NUMBER,
            text_align=ft.TextAlign.RIGHT,
        )
        self.memory_limit_field = ft.TextField(
            label="RAM per Worker (GB)",
                value=str(self.app_config.memory_limit_gb or ""),
            width=180,
            keyboard_type=ft.KeyboardType.NUMBER,
            hint_text="e.g., 4.0",
            text_align=ft.TextAlign.RIGHT,
        )
        self.db_batch_size_field = ft.TextField(
            label="DB Batch Size",
                value=str(self.app_config.db_batch_size),
            width=150,
            keyboard_type=ft.KeyboardType.NUMBER,
            text_align=ft.TextAlign.RIGHT,
        )
        self.use_fido_switch = ft.Switch(label="Use Fido", value=self.app_config.use_fido, label_position=ft.LabelPosition.LEFT)
        self.extract_text_switch = ft.Switch(
            label="Extract Text/PII",
                value=self.app_config.extract_text_on_scan,
            label_position=ft.LabelPosition.LEFT
        )
        self.phash_switch = ft.Switch(
            label="Compute Perceptual Hash",
                value=self.app_config.compute_perceptual_hash,
            label_position=ft.LabelPosition.LEFT,
        )
        self.pii_languages_field = ft.TextField(
            label="PII Languages (comma-separated, e.g., en,es)",
                value=",".join(self.app_config.pii_languages),
            expand=True,
        )
        self.archive_excludes_field = ft.TextField(
            label="Archive extensions to skip extracting (e.g., .epub,.cbz)",
                value=",".join(self.app_config.archive_exclude_extensions),
            expand=True,
        )
        text_fields_row = ft.Row(
                controls=[self.pii_languages_field, self.archive_excludes_field],
        )
    
        # --- Asset Management Controls & Logic ---
        self.embedding_model_status = ft.Text("Checking...", italic=True)
        self.pii_model_status = ft.Text("Checking...", italic=True)
        self.fido_status = ft.Text("Checking...", italic=True)
        self.asset_status_progress = ft.ProgressBar(visible=False)

        self.download_embedding_button = ft.ElevatedButton("Download", disabled=True, on_click=lambda e: self.run_asset_download(e, "embedding"))
        self.download_pii_button = ft.ElevatedButton("Download", disabled=True, on_click=lambda e: self.run_asset_download(e, "pii"))
        self.download_fido_button = ft.ElevatedButton("Download", disabled=True, on_click=lambda e: self.run_asset_download(e, "fido"))

        self.embedding_model_path_field = ft.TextField(
            label="Custom Embedding Model Path",
                value=self.app_config.embedding_model_path,
            hint_text="e.g., models/my-custom-model",
            expand=True,
        )

        asset_management_view = ft.Column([
            self.asset_status_progress,
            ft.Row([
                ft.Text("Default Semantic Search Model", expand=True),
                    self.embedding_model_status,
                    self.download_embedding_button
            ]),
            ft.Row([
                    ft.Text(f"PII Models ({','.join(self.app_config.pii_languages)})", expand=True),
                    self.pii_model_status,
                    self.download_pii_button
            ]),
            ft.Row([ft.Text("Fido/PRONOM Signatures", expand=True), self.fido_status, self.download_fido_button]),
            self.embedding_model_path_field,
            ft.Text("Note: Downloads can be large and take several minutes.", style=ft.TextThemeStyle.BODY_SMALL)
        ])

        self.view_settings = ft.Column(
            scroll=ft.ScrollMode.ADAPTIVE,
            controls=[ft.Container(
                padding=ft.padding.only(right=15),
                content=ft.Column(controls=[
                    ft.Text("Settings", style=ft.TextThemeStyle.HEADLINE_MEDIUM),
                    config_path_text,
                    ft.Divider(),
                    ft.Text("Performance", style=ft.TextThemeStyle.TITLE_MEDIUM),
                        ft.Row(controls=[self.workers_field, self.memory_limit_field, self.db_batch_size_field]),
                    ft.Text(
                        "Note: RAM limit is only supported on Linux and macOS.",
                        style=ft.TextThemeStyle.BODY_SMALL,
                    ),
                    ft.Divider(),
                    ft.Text("File Processing", style=ft.TextThemeStyle.TITLE_MEDIUM),
                        ft.Row(controls=[self.use_fido_switch, self.extract_text_switch, self.phash_switch], spacing=20),
                    ft.Container(content=text_fields_row, margin=ft.margin.only(top=10)),
                    ft.Divider(),
                    ft.Text("Scan Directories", style=ft.TextThemeStyle.TITLE_MEDIUM),
                    ft.Container(
                        content=self.scan_dirs_list,
                        border=ft.border.all(1, ft.Colors.OUTLINE),
                        border_radius=ft.border_radius.all(5),
                        padding=10,
                        height=150,
                    ),
                    ft.Row([ft.ElevatedButton("Add Directory", icon=ft.Icons.FOLDER_OPEN, on_click=lambda _: directory_picker.get_directory_path())]),
                    ft.Divider(),
                    ft.Text("Excluded Directories & Patterns", style=ft.TextThemeStyle.TITLE_MEDIUM),
                    ft.Text(
                        "Use folder names (e.g., 'node_modules') or glob patterns (e.g., '*.tmp') to exclude them everywhere. Full paths are not typically needed.",
                        style=ft.TextThemeStyle.BODY_SMALL,
                    ),
                    ft.Container(
                        content=self.exclude_dirs_list,
                        border=ft.border.all(1, ft.Colors.OUTLINE),
                        border_radius=ft.border_radius.all(5),
                        padding=10,
                        height=150,
                    ),
                    ft.Row([self.new_exclude_dir_field, ft.IconButton(icon=ft.Icons.ADD, on_click=self.add_excluded_dir, tooltip="Add")]),
                    ft.Divider(),
                    ft.Text("Offline Assets", style=ft.TextThemeStyle.TITLE_MEDIUM),
                    asset_management_view,
                    ft.Divider(),
                    ft.ElevatedButton(
                        "Save Settings", icon=ft.Icons.SAVE, on_click=self.save_settings_click,
                        disabled=not self.config_path, tooltip="Save settings to wanderer.toml"
                    ),
                ])
            )
            ]
        )
        self.view_about = ft.Column(
        [ft.Text("About Wanderer", style=ft.TextThemeStyle.HEADLINE_MEDIUM),
         ft.Text("Version, author, and other application information will go here.")]
    )
    
        # --- Help View ---
        help_content = """
# Wanderer Help

This guide explains the main features of the Wanderer application.

## 1. Scan

The **Scan** page is where you start the file indexing process.

- **Select Directories**: Choose which of your configured directories you want to scan. You can configure the master list of directories in the **Settings** page.
- **Scan Options**:
    - **Extract Text & PII**: When enabled, the scanner will read the content of documents (PDFs, DOCX, etc.) and check for Personally Identifiable Information (PII). This is required for semantic search.
    - **Compute Perceptual Hashes**: When enabled, the scanner will generate a "perceptual hash" for images, which allows you to find visually similar (but not identical) images.
- **Start Scan**: Begins the indexing process. This may take a long time for large directories.
- **Stop Scan**: Safely stops an ongoing scan.

## 2. Search

The **Search** page allows you to perform a powerful "semantic search" across all indexed text content. This means you can search for concepts and meanings, not just exact keywords.

For this feature to work, you must have run a scan with "Extract Text & PII" enabled.

## 3. Reports

The **Reports** page provides various analyses of your indexed files:

- **Find Identical Files**: Finds files that are bit-for-bit identical.
- **Find Similar Images**: Finds visually similar images. Requires "Compute Perceptual Hashes" to be enabled during scan.
- **Find Similar Text**: Finds documents with similar meaning. Requires "Extract Text & PII" to be enabled during scan.
- **List Largest Files**: Shows which files are taking up the most space.
- **File Type Summary**: Groups files by their type (e.g., 'image/jpeg') and shows counts and total sizes.
- **List Files with PII**: Lists all files that were flagged for containing sensitive information.

## 4. Settings

The **Settings** page allows you to configure how Wanderer operates. Changes made here are saved to your `wanderer.toml` file.

- **Performance**: Adjust the number of worker processes and memory limits to best suit your hardware.
- **File Processing**: Enable or disable features like Fido (for accurate file typing), text extraction, and perceptual hashing.
- **Scan Directories**: Manage the master list of directories that Wanderer can scan.
- **Excluded Directories & Patterns**: Configure a list of folders and patterns (like `node_modules` or `*.tmp`) to ignore during scans.
- **Offline Assets**: Manage the AI models and other data needed for features like semantic search and PII detection.

## Advanced Workflow: Staged Scanning

For very large collections, it's more efficient to perform a fast initial scan and then selectively "refine" the data.

1.  **Fast Initial Scan**: In **Settings**, disable "Extract Text/PII" and "Compute Perceptual Hash". Then run a scan from the **Scan** page. This will quickly index all files with basic metadata.
2.  **Selective Refinement**: Use the command line to run deeper analysis on specific folders. For example:
    - To analyze all documents in your `Documents` folder:
      `poetry run python -m wanderer.main refine-text-by-path /path/to/Documents`
    - To compute perceptual hashes for your `Pictures` folder:
      `poetry run python -m wanderer.main refine-images-by-path /path/to/Pictures`
    - To compute perceptual hashes for *only JPEGs* in your `Photos` folder:
      `poetry run python -m wanderer.main refine-images-by-mime-and-path --mime-type "image/jpeg" /path/to/Photos`
    - To force a high-accuracy Fido scan on an `Downloads` folder:
      `poetry run python -m wanderer.main refine-fido-by-path /path/to/Downloads`
"""
        self.view_help = ft.Column(
        scroll=ft.ScrollMode.ADAPTIVE,
        controls=[
            ft.Container(
                padding=ft.padding.only(right=15),
                content=ft.Markdown(
                    help_content,
                    extension_set=ft.MarkdownExtensionSet.GITHUB_WEB,
                )
            )
        ]
    )

        # --- Main Content Area ---
        self.main_content = ft.Container(content=self.view_scan, expand=True, padding=20)

        navigation_rail = ft.NavigationRail(
        selected_index=0,
        label_type=ft.NavigationRailLabelType.ALL,
        min_width=100,
        min_extended_width=400,
        group_alignment=-0.9,
        destinations=[
                ft.NavigationRailDestination(icon=ft.Icons.RADAR_OUTLINED, selected_icon=ft.Icons.RADAR, label="Scan"),
                ft.NavigationRailDestination(icon=ft.Icons.SEARCH_OUTLINED, selected_icon=ft.Icons.SEARCH, label="Search"),
                ft.NavigationRailDestination(icon=ft.Icons.INSERT_CHART_OUTLINED, selected_icon=ft.Icons.INSERT_CHART, label="Reports"),
                ft.NavigationRailDestination(icon=ft.Icons.SETTINGS_OUTLINED, selected_icon=ft.Icons.SETTINGS, label="Settings"),
                ft.NavigationRailDestination(icon=ft.Icons.HELP_OUTLINE, selected_icon=ft.Icons.HELP, label="Help"),
                ft.NavigationRailDestination(icon=ft.Icons.INFO_OUTLINE, selected_icon=ft.Icons.INFO, label="About"),
        ],
            on_change=self.navigate,
    )

        self.page.add(
            ft.Row([navigation_rail, ft.VerticalDivider(width=1), self.main_content], expand=True)
        )

        self.page.on_window_event = self.window_event
        
        # Run startup tasks after the page loads to improve perceived performance
        async def on_load_async(e=None):
            self.update_asset_status(self.page)
            await self.refresh_scan_history(self.page) # This is now async and calls a thread
        
        self.page.on_load = on_load_async
        await self.update_list_views()
        self.page.update()

async def main(page: ft.Page):
    """
    The main entry point for the Flet application.
    """
    gui_app = WandererGUI(page)
    await gui_app.build()