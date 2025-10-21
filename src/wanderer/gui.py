import flet as ft
from . import config
from . import indexer
from . import database, models
import asyncio
import tomllib, spacy
from pathlib import Path
from . import download_assets
from datetime import datetime
import threading


async def main(page: ft.Page):
    """
    The main async function to build and run the Flet GUI for Wanderer.
    """
    page.title = "Wanderer"
    page.window_width = 1000
    page.window_height = 700
    page.vertical_alignment = ft.MainAxisAlignment.START

    # Load application configuration
    app_config, config_path = config.load_config_with_path()

    # --- Scan View ---
    scan_targets_list = ft.ListView(expand=False, spacing=5, height=200)
    if app_config.scan_dirs:
        for d in app_config.scan_dirs:
            scan_targets_list.controls.append(ft.Checkbox(label=d, value=True))
    else:
        scan_targets_list.controls.append(ft.Text("No scan directories configured. Please add some in Settings."))

    scan_option_text = ft.Checkbox(
        label="Extract Text & PII",
        value=app_config.extract_text_on_scan
    )
    scan_option_phash = ft.Checkbox(
        label="Compute Perceptual Hashes",
        value=app_config.compute_perceptual_hash
    )
    scan_option_fido = ft.Checkbox(
        label="Enable Fido",
        value=app_config.use_fido
    )

    async def start_scan_click(e):
        """Handles the 'Start Scan' button click."""
        # --- UI Update: Show scanning state ---
        start_scan_button.visible = False
        stop_scan_button.visible = True
        scan_progress.visible = True
        scan_progress.value = None  # Indeterminate progress
        scan_status_text.value = "Starting scan..."
        
        # Disable controls
        for checkbox in scan_targets_list.controls:
            if isinstance(checkbox, ft.Checkbox):
                checkbox.disabled = True
        scan_option_text.disabled = True
        scan_option_phash.disabled = True
        scan_option_fido.disabled = True
        await page.update_async()

        # --- Backend Logic: Run indexer in a thread ---
        def scan_job():
            """The actual scanning function to run in a thread."""
            
            # This callback will be called from the indexer thread.
            # We use page.run_threadsafe to update the UI from the main thread.
            def progress_callback(value, total, description):
                async def update_ui():
                    scan_status_text.value = description
                    if total > 1:
                        scan_progress.value = value / total
                    else:
                        scan_progress.value = None # Indeterminate
                    await page.update_async()
                page.run_threadsafe(update_ui())

            # Get selected paths from the GUI
            selected_paths = [
                Path(cb.label) for cb in scan_targets_list.controls if isinstance(cb, ft.Checkbox) and cb.value
            ]
            
            idx = indexer.Indexer(root_paths=tuple(selected_paths), workers=app_config.workers, memory_limit_gb=app_config.memory_limit_gb, exclude_paths=(), progress_callback=progress_callback)
            idx.run()
            # Reset UI after scan and refresh history
            async def finish_scan():
                await stop_scan_click(None, "Scan finished.")
                await refresh_scan_history()
            page.run_threadsafe(finish_scan())

        threading.Thread(target=scan_job, daemon=True).start()
        await page.update_async()

    async def stop_scan_click(e, message: str | None = None):
        # This is where we would signal the scan thread to stop.
        start_scan_button.visible = True
        stop_scan_button.visible = False
        scan_progress.visible = False
        scan_status_text.value = message or "Scan stopped by user."
        # Re-enable controls
        for checkbox in scan_targets_list.controls: # type: ignore
            if isinstance(checkbox, ft.Checkbox):
                checkbox.disabled = False
        scan_option_text.disabled = False
        scan_option_phash.disabled = False
        scan_option_fido.disabled = False
        await page.update_async()

    start_scan_button = ft.ElevatedButton(
        "Start Scan",
        icon=ft.Icons.PLAY_ARROW_ROUNDED,
        on_click=start_scan_click,
        disabled=not app_config.scan_dirs
    )
    stop_scan_button = ft.ElevatedButton("Stop Scan", icon=ft.Icons.STOP_ROUNDED, on_click=stop_scan_click, visible=False)
    scan_progress = ft.ProgressBar(width=400, visible=False)
    scan_status_text = ft.Text("Idle.", italic=True)

    new_scan_view = ft.Column(
        controls=[
            ft.Text("Select directories to scan:"), scan_targets_list,
            ft.Text("Scan Options:"), ft.Row(controls=[scan_option_text, scan_option_phash, scan_option_fido]),
            ft.Row([start_scan_button, stop_scan_button]), scan_progress, scan_status_text],
        spacing=15,
    )

    # --- Scan History View ---
    scan_history_list = ft.ListView(expand=True, spacing=10)

    async def refresh_scan_history(e=None):
        scan_history_list.controls.clear()
        with database.get_session() as db:
            logs = db.query(models.ScanLog).order_by(models.ScanLog.start_time.desc()).limit(50).all()
            if not logs:
                scan_history_list.controls.append(ft.Text("No scan history found."))
            else:
                for log in logs:
                    status_color = {
                        "completed": ft.colors.GREEN,
                        "failed": ft.colors.RED,
                        "started": ft.colors.ORANGE,
                    }.get(log.status, ft.colors.GREY)
                    
                    end_time_str = log.end_time.strftime('%Y-%m-%d %H:%M:%S') if log.end_time else "In Progress"
                    duration = (log.end_time - log.start_time) if log.end_time else "N/A"

                    scan_history_list.controls.append(
                        ft.ListTile(
                            leading=ft.Icon(ft.Icons.HISTORY, color=status_color),
                            title=ft.Text(f"Scan on {log.start_time.strftime('%Y-%m-%d %H:%M:%S')}"),
                            subtitle=ft.Text(f"Status: {log.status} | Files: {log.files_scanned} | Duration: {duration}"),
                        )
                    )
        await scan_history_list.update_async()

    scan_history_view = ft.Column(
        controls=[ft.Row([ft.Text("Recent Scans"), ft.IconButton(icon=ft.Icons.REFRESH, on_click=refresh_scan_history, tooltip="Refresh History")]), scan_history_list], expand=True
    )

    # --- Refine View ---
    refine_status_text = ft.Text("Idle.", italic=True)
    refine_progress = ft.ProgressBar(width=400, visible=False)
    refine_buttons = [
        ft.ElevatedButton("Refine Unknown Files (Fido)", on_click=None, disabled=True),
        ft.ElevatedButton("Refine All Text Content", on_click=None, disabled=True),
        ft.ElevatedButton("Refine All Image Hashes", on_click=None, disabled=True),
    ]

    refine_data_view = ft.Column(
        controls=[
            ft.Text("Run deep analysis on files already in the database. This is useful after a fast initial scan."),
            ft.Row(controls=refine_buttons),
            refine_progress,
            refine_status_text,
        ],
        spacing=15,
    )

    view_scan = ft.Column(
        controls=[
            ft.Text("Scan Management", style=ft.TextThemeStyle.HEADLINE_MEDIUM),
            ft.Tabs(
                selected_index=0,
                tabs=[
                    ft.Tab(text="New Scan", content=new_scan_view),
                    ft.Tab(text="Scan History", content=scan_history_view),
                    ft.Tab(text="Refine Data", content=refine_data_view),
                ],
            ),
        ],
        spacing=15,
    )

    search_explanation = """
This isn't your typical keyword search! It finds files based on the *meaning* of your query. Think of it like asking a question.

**For example, you could search for:**
*   "photos from my beach vacation"
*   "that chicken recipe my aunt sent me"
*   "summary of the project alpha meeting"

The search will find documents and notes that are conceptually related to what you're looking for.

**Note**: This feature requires an AI model. It will be downloaded automatically on first use (requires internet), or you can download it from the **Settings > Offline Assets** page for offline use.
"""

    view_search = ft.Column(
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

    async def show_report_results(report_name: str):
        """Placeholder function to display report results."""
        report_content_area.content = ft.Column([
            ft.IconButton(icon=ft.Icons.ARROW_BACK, on_click=show_report_list, tooltip="Back to reports list"),
            ft.Text(f"Results for: {report_name}", style=ft.TextThemeStyle.HEADLINE_SMALL),
            ft.Text("Report results will be displayed here."),
        ])
        await page.update_async()

    reports_list_view = ft.ListView(expand=True, spacing=10)
    for report in reports_available:
        reports_list_view.controls.append(
            ft.ListTile(
                leading=ft.Icon(report["icon"]),
                title=ft.Text(report["name"]),
                subtitle=ft.Text(report["description"]),
                on_click=lambda e, r=report["name"]: asyncio.create_task(show_report_results(r)),
            )
        )

    report_content_area = ft.Container(content=reports_list_view, expand=True)
    async def show_report_list(e=None):
        report_content_area.content = reports_list_view
        await page.update_async()

    view_reports = ft.Column(
        [ft.Text("Reports", style=ft.TextThemeStyle.HEADLINE_MEDIUM), report_content_area], expand=True
    )

    # --- Directory Management for Settings (with improved removal logic) ---
    scan_dirs_list = ft.ListView(expand=True, spacing=5)
    exclude_dirs_list = ft.ListView(expand=True, spacing=5)

    async def remove_scan_dir(dir_to_remove: str):
        app_config.scan_dirs.remove(dir_to_remove)
        await update_list_views()

    async def remove_exclude_dir(dir_to_remove: str):
        app_config.exclude_dirs.remove(dir_to_remove)
        await update_list_views()

    async def update_list_views():
        scan_dirs_list.controls.clear()
        for d in sorted(app_config.scan_dirs):
            scan_dirs_list.controls.append(
                ft.Row([ft.Text(d, expand=True), ft.IconButton(ft.Icons.DELETE_OUTLINE, on_click=lambda _, dir=d: remove_scan_dir(dir), tooltip="Remove")])
            )
        
        exclude_dirs_list.controls.clear()
        for d in sorted(app_config.exclude_dirs):
            exclude_dirs_list.controls.append(
                ft.Row([ft.Text(d, expand=True), ft.IconButton(ft.Icons.DELETE_OUTLINE, on_click=lambda _, dir=d: remove_exclude_dir(dir), tooltip="Remove")])
            )
        await page.update_async()

    async def on_directory_picked(e: ft.FilePickerResultEvent):
        if e.path:
            if e.path not in app_config.scan_dirs:
                app_config.scan_dirs.append(e.path)
                # Also update the scan targets list on the main scan page
                scan_targets_list.controls.append(ft.Checkbox(label=e.path, value=True))
                start_scan_button.disabled = False
                await update_list_views()
                await scan_targets_list.update_async()
                await start_scan_button.update_async()

    async def add_excluded_dir(e):
        dir_to_exclude = new_exclude_dir_field.value
        if dir_to_exclude and dir_to_exclude not in app_config.exclude_dirs:
            app_config.exclude_dirs.append(dir_to_exclude)
            new_exclude_dir_field.value = ""
            await update_list_views()

    directory_picker = ft.FilePicker(on_result=on_directory_picked)
    page.overlay.append(directory_picker)

    new_exclude_dir_field = ft.TextField(
        label="Add exclusion (e.g., 'node_modules', '*.log')",
        expand=True,
        on_submit=add_excluded_dir,
    )

    async def save_settings_click(e):
        """Saves the current settings from the UI to the wanderer.toml file."""
        # Update the app_config object from the UI fields
        try:
            app_config.workers = int(workers_field.value)
            app_config.db_batch_size = int(db_batch_size_field.value)
            app_config.memory_limit_gb = float(memory_limit_field.value) if memory_limit_field.value else None
        except (ValueError, TypeError):
            # Handle potential conversion errors gracefully
            # In a real app, you'd show a dialog here.
            print("Invalid numeric value in settings.")
            return

        app_config.use_fido = use_fido_switch.value
        app_config.extract_text_on_scan = extract_text_switch.value
        app_config.compute_perceptual_hash = phash_switch.value
        app_config.pii_languages = [lang.strip() for lang in pii_languages_field.value.split(',') if lang.strip()]
        app_config.archive_exclude_extensions = [ext.strip() for ext in archive_excludes_field.value.split(',') if ext.strip()]
        app_config.embedding_model_path = embedding_model_path_field.value or None

        # The scan_dirs and exclude_dirs are already updated in real-time

        if config_path:
            try:
                # Read the existing toml file to preserve structure and comments
                with open(config_path, "rb") as f:
                    full_toml = tomllib.load(f)

                # Ensure the [tool.wanderer] section exists
                if "tool" not in full_toml:
                    full_toml["tool"] = {}
                if "wanderer" not in full_toml["tool"]:
                    full_toml["tool"]["wanderer"] = {}

                # Update the values
                full_toml["tool"]["wanderer"] = config.config_to_dict(app_config)
                config.save_config_to_path(full_toml, config_path)
            except Exception as ex:
                print(f"Error saving config: {ex}")

    # Populate initial lists
    await update_list_views()

    # --- Settings View ---
    config_path_text = ft.Text(
        f"Editing: {config_path}" if config_path else "No wanderer.toml found. Using default settings.",
        italic=True
    )
    workers_field = ft.TextField(
        label="Worker Processes",
        value=str(app_config.workers),
        width=150,
        keyboard_type=ft.KeyboardType.NUMBER,
        text_align=ft.TextAlign.RIGHT,
    )
    memory_limit_field = ft.TextField(
        label="RAM per Worker (GB)",
        value=str(app_config.memory_limit_gb or ""),
        width=180,
        keyboard_type=ft.KeyboardType.NUMBER,
        hint_text="e.g., 4.0",
        text_align=ft.TextAlign.RIGHT,
    )
    db_batch_size_field = ft.TextField(
        label="DB Batch Size",
        value=str(app_config.db_batch_size),
        width=150,
        keyboard_type=ft.KeyboardType.NUMBER,
        text_align=ft.TextAlign.RIGHT,
    )
    use_fido_switch = ft.Switch(label="Use Fido", value=app_config.use_fido, label_position=ft.LabelPosition.LEFT)
    extract_text_switch = ft.Switch(
        label="Extract Text/PII",
        value=app_config.extract_text_on_scan,
        label_position=ft.LabelPosition.LEFT
    )
    phash_switch = ft.Switch(
        label="Compute Perceptual Hash",
        value=app_config.compute_perceptual_hash,
        label_position=ft.LabelPosition.LEFT,
    )
    pii_languages_field = ft.TextField(
        label="PII Languages (comma-separated, e.g., en,es)",
        value=",".join(app_config.pii_languages),
        expand=True,
    )
    archive_excludes_field = ft.TextField(
        label="Archive extensions to skip extracting (e.g., .epub,.cbz)",
        value=",".join(app_config.archive_exclude_extensions),
        expand=True,
    )
    text_fields_row = ft.Row(
        controls=[pii_languages_field, archive_excludes_field],
    )
    
    # --- Asset Management Controls & Logic ---
    embedding_model_status = ft.Text("Checking...", italic=True)
    pii_model_status = ft.Text("Checking...", italic=True)
    fido_status = ft.Text("Checking...", italic=True)

    async def run_asset_download(e, asset_type: str):
        """Handles the download process for a given asset in a background thread."""
        e.control.disabled = True
        e.control.text = "Downloading..."
        progress_ring = ft.ProgressRing(width=16, height=16, stroke_width=2)
        e.control.icon = progress_ring
        await page.update_async()

        def download_job():
            """The actual download function to run in a thread."""
            script_dir = Path(__file__).parent
            models_dir = script_dir / "models"
            if asset_type == "embedding":
                download_assets.download_sentence_transformer('all-MiniLM-L6-v2', models_dir / 'all-MiniLM-L6-v2')
            elif asset_type == "pii":
                for lang in app_config.pii_languages:
                    model_name = config.get_spacy_model_name(lang)
                    download_assets.download_spacy_model(model_name, lang)
            elif asset_type == "fido":
                download_assets.cache_fido_signatures(models_dir / 'fido_cache')
            
            # After download, trigger a UI update from the main thread
            page.run_threadsafe(update_asset_status)

        # Run the download in a separate thread to not block the UI
        thread = threading.Thread(target=download_job, daemon=True)
        thread.start()

    download_embedding_button = ft.ElevatedButton(
        "Download", disabled=True, on_click=lambda e: run_asset_download(e, "embedding")
    )
    download_pii_button = ft.ElevatedButton(
        "Download", disabled=True, on_click=lambda e: run_asset_download(e, "pii")
    )
    download_fido_button = ft.ElevatedButton(
        "Download", disabled=True, on_click=lambda e: run_asset_download(e, "fido")
    )

    async def update_asset_status():
        """Checks for offline assets and updates the UI accordingly."""
        script_dir = Path(__file__).parent
        models_dir = script_dir / "models"

        # 1. Check Embedding Model
        model_path = script_dir / (app_config.embedding_model_path or "models/all-MiniLM-L6-v2")
        if model_path.is_dir():
            embedding_model_status.value = "Available"
            embedding_model_status.color = ft.colors.GREEN
            download_embedding_button.disabled = True
            download_embedding_button.text = "Downloaded"
            download_embedding_button.icon = ft.icons.CHECK
        else:
            embedding_model_status.value = "Not Found"
            embedding_model_status.color = ft.colors.ORANGE
            download_embedding_button.disabled = False
            download_embedding_button.text = "Download"
            download_embedding_button.icon = ft.icons.DOWNLOAD

        # 2. Check PII (spaCy) Models
        all_pii_models_found = True
        for lang in app_config.pii_languages:
            model_name = config.get_spacy_model_name(lang)
            if not spacy.util.is_package(model_name):
                all_pii_models_found = False
                break
        if all_pii_models_found:
            pii_model_status.value = "Available"
            pii_model_status.color = ft.colors.GREEN
            download_pii_button.disabled = True
            download_pii_button.text = "Downloaded"
            download_pii_button.icon = ft.icons.CHECK
        else:
            pii_model_status.value = "Missing"
            pii_model_status.color = ft.colors.ORANGE
            download_pii_button.disabled = False
            download_pii_button.text = "Download"
            download_pii_button.icon = ft.icons.DOWNLOAD

        # 3. Check Fido Signatures
        fido_sig_path = models_dir / "fido_cache" / "DROID_SignatureFile.xml"
        if fido_sig_path.is_file():
            fido_status.value = "Available"
            fido_status.color = ft.colors.GREEN
            download_fido_button.disabled = True
            download_fido_button.text = "Downloaded"
            download_fido_button.icon = ft.icons.CHECK
        else:
            fido_status.value = "Not Found"
            fido_status.color = ft.colors.ORANGE
            download_fido_button.disabled = app_config.use_fido is False
            download_fido_button.text = "Download"
            download_fido_button.icon = ft.icons.DOWNLOAD

        await page.update_async()

    embedding_model_path_field = ft.TextField(
        label="Custom Embedding Model Path",
        value=app_config.embedding_model_path,
        hint_text="e.g., models/my-custom-model",
        expand=True,
    )

    asset_management_view = ft.Column([
        ft.Row([
            ft.Text("Default Semantic Search Model", expand=True),
            embedding_model_status,
            download_embedding_button
        ]),
        ft.Row([
            ft.Text(f"PII Models ({','.join(app_config.pii_languages)})", expand=True),
            pii_model_status,
            download_pii_button
        ]),
        ft.Row([ft.Text("Fido/PRONOM Signatures", expand=True), fido_status, download_fido_button]),
        embedding_model_path_field,
        ft.Text("Note: Downloads can be large and take several minutes.", style=ft.TextThemeStyle.BODY_SMALL)
    ])

    view_settings = ft.Column(
        scroll=ft.ScrollMode.ADAPTIVE,
        controls=[ft.Container(
            padding=ft.padding.only(right=15),
            content=ft.Column(controls=[
                ft.Text("Settings", style=ft.TextThemeStyle.HEADLINE_MEDIUM),
                config_path_text,
                ft.Divider(),
                ft.Text("Performance", style=ft.TextThemeStyle.TITLE_MEDIUM),
                ft.Row(controls=[workers_field, memory_limit_field, db_batch_size_field]),
                ft.Text(
                    "Note: RAM limit is only supported on Linux and macOS.",
                    style=ft.TextThemeStyle.BODY_SMALL,
                ),
                ft.Divider(),
                ft.Text("File Processing", style=ft.TextThemeStyle.TITLE_MEDIUM),
                ft.Row(controls=[use_fido_switch, extract_text_switch, phash_switch], spacing=20),
                ft.Container(content=text_fields_row, margin=ft.margin.only(top=10)),
                ft.Divider(),
                ft.Text("Scan Directories", style=ft.TextThemeStyle.TITLE_MEDIUM),
                ft.Container(
                    content=scan_dirs_list,
                    border=ft.border.all(1, ft.Colors.OUTLINE),
                    border_radius=ft.border_radius.all(5),
                    padding=10,
                    height=150,
                ),
                ft.Row([ft.ElevatedButton("Add Directory", icon=ft.Icons.FOLDER_OPEN, on_click=lambda _: directory_picker.get_directory_path_async())]),
                ft.Divider(),
                ft.Text("Excluded Directories & Patterns", style=ft.TextThemeStyle.TITLE_MEDIUM),
                ft.Text(
                    "Use folder names (e.g., 'node_modules') or glob patterns (e.g., '*.tmp') to exclude them everywhere. Full paths are not typically needed.",
                    style=ft.TextThemeStyle.BODY_SMALL,
                ),
                 ft.Container(
                    content=exclude_dirs_list,
                    border=ft.border.all(1, ft.Colors.OUTLINE),
                    border_radius=ft.border_radius.all(5),
                    padding=10,
                    height=150,
                ),
                ft.Row([new_exclude_dir_field, ft.IconButton(icon=ft.Icons.ADD, on_click=add_excluded_dir, tooltip="Add")]),
                ft.Divider(),
                ft.Text("Offline Assets", style=ft.TextThemeStyle.TITLE_MEDIUM),
                asset_management_view,
                ft.Divider(),
                ft.ElevatedButton(
                    "Save Settings", icon=ft.Icons.SAVE, on_click=save_settings_click,
                    disabled=not config_path, tooltip="Save settings to wanderer.toml"
                ),
            ])
        )
        ]
    )
    view_about = ft.Column(
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
    view_help = ft.Column(
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
    # This control will display the currently selected view.
    main_content = ft.Container(content=view_scan, expand=True, padding=20)

    async def navigate(e):
        """Handles navigation rail selection changes."""
        # Clear the main content area
        if e.control.selected_index == 0:
            main_content.content = view_scan
        elif e.control.selected_index == 1:
            main_content.content = view_search
        elif e.control.selected_index == 2:
            main_content.content = view_reports
        elif e.control.selected_index == 3:
            main_content.content = view_settings
        elif e.control.selected_index == 4:
            main_content.content = view_help
        elif e.control.selected_index == 5:
            main_content.content = view_about
        await page.update_async()

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
        on_change=navigate,
    )

    page.add(
        ft.Row([navigation_rail, ft.VerticalDivider(width=1), main_content], expand=True)
    )

    await page.update_async()

    # --- Graceful Shutdown Handler ---
    # This handles the window closing event to prevent a common Flutter error on exit.
    async def window_event(e):
        if e.data == "close":
            await page.window_destroy_async()

    page.on_window_event = window_event

    # --- Initial UI Updates on Page Load ---
    async def on_page_load(e=None):
        await update_asset_status()
        await refresh_scan_history()

    page.on_load = on_page_load