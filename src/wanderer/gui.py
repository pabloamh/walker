import flet as ft
from . import config


def main(page: ft.Page):
    """
    The main function to build and run the Flet GUI for Wanderer.
    """
    page.title = "Wanderer"
    page.window_width = 1000
    page.window_height = 700
    page.vertical_alignment = ft.MainAxisAlignment.START

    # Load application configuration
    app_config, config_path = config.load_config_with_path()

    # --- Asset Management (for settings) ---
    # This is a placeholder for a more complex status check.
    # In a real implementation, you'd check if the model paths exist.

    # --- Placeholder Views ---

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

    def start_scan_click(e):
        # This is where the scan logic will be triggered in a thread.
        # For now, just update the UI to show the "scanning" state.
        start_scan_button.visible = False
        stop_scan_button.visible = True
        scan_progress.visible = True
        scan_status_text.value = "Starting scan..."
        for checkbox in scan_targets_list.controls:
            if isinstance(checkbox, ft.Checkbox):
                checkbox.disabled = True
        scan_option_text.disabled = True
        scan_option_phash.disabled = True
        scan_option_fido.disabled = True
        page.update()

    def stop_scan_click(e):
        # This is where we would signal the scan thread to stop.
        start_scan_button.visible = True
        stop_scan_button.visible = False
        scan_progress.visible = False
        scan_status_text.value = "Scan stopped by user."
        for checkbox in scan_targets_list.controls:
            if isinstance(checkbox, ft.Checkbox):
                checkbox.disabled = False
        scan_option_text.disabled = False
        scan_option_phash.disabled = False
        scan_option_fido.disabled = False
        page.update()

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
                tabs=[ft.Tab(text="New Scan", content=new_scan_view), ft.Tab(text="Refine Data", content=refine_data_view)],
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

    def show_report_results(report_name: str):
        """Placeholder function to display report results."""
        report_content_area.content = ft.Column([
            ft.IconButton(icon=ft.Icons.ARROW_BACK, on_click=lambda _: show_report_list(), tooltip="Back to reports list"),
            ft.Text(f"Results for: {report_name}", style=ft.TextThemeStyle.HEADLINE_SMALL),
            ft.Text("Report results will be displayed here."),
        ])
        page.update()

    reports_list_view = ft.ListView(expand=True, spacing=10)
    for report in reports_available:
        reports_list_view.controls.append(
            ft.ListTile(
                leading=ft.Icon(report["icon"]),
                title=ft.Text(report["name"]),
                subtitle=ft.Text(report["description"]),
                on_click=lambda _, r=report["name"]: show_report_results(r),
            )
        )

    report_content_area = ft.Container(content=reports_list_view, expand=True)
    def show_report_list():
        report_content_area.content = reports_list_view
        page.update()

    view_reports = ft.Column(
        [ft.Text("Reports", style=ft.TextThemeStyle.HEADLINE_MEDIUM), report_content_area], expand=True
    )

    # --- Directory Management for Settings ---
    scan_dirs_list = ft.ListView(expand=True, spacing=5)
    exclude_dirs_list = ft.ListView(expand=True, spacing=5)

    def update_list_views():
        scan_dirs_list.controls.clear()
        for d in sorted(app_config.scan_dirs):
            scan_dirs_list.controls.append(ft.Text(d))
        
        exclude_dirs_list.controls.clear()
        for d in sorted(app_config.exclude_dirs):
            exclude_dirs_list.controls.append(ft.Text(d))
        page.update()

    def on_directory_picked(e: ft.FilePickerResultEvent):
        if e.path:
            if e.path not in app_config.scan_dirs:
                app_config.scan_dirs.append(e.path)
                update_list_views()

    def add_excluded_dir(e):
        dir_to_exclude = new_exclude_dir_field.value
        if dir_to_exclude and dir_to_exclude not in app_config.exclude_dirs:
            app_config.exclude_dirs.append(dir_to_exclude)
            new_exclude_dir_field.value = ""
            update_list_views()

    def remove_selected_scan_dir(e):
        # This is a placeholder for a more complex selection mechanism
        if app_config.scan_dirs:
            app_config.scan_dirs.pop()
            update_list_views()

    def remove_selected_exclude_dir(e):
        # This is a placeholder for a more complex selection mechanism
        if app_config.exclude_dirs:
            app_config.exclude_dirs.pop()
            update_list_views()

    directory_picker = ft.FilePicker(on_result=on_directory_picked)
    page.overlay.append(directory_picker)

    new_exclude_dir_field = ft.TextField(
        label="Add exclusion (e.g., 'node_modules', '*.log')",
        expand=True,
        on_submit=add_excluded_dir,
    )

    # Populate initial lists
    update_list_views()

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
    
    # --- Asset Management Controls ---
    # In a real app, these would have functions to check status and trigger downloads.
    embedding_model_status = ft.Text("Status: Unknown", italic=True)
    pii_model_status = ft.Text("Status: Unknown", italic=True)
    fido_status = ft.Text("Status: Unknown", italic=True)
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
            ft.ElevatedButton("Download", disabled=True)
        ]),
        ft.Row([
            ft.Text(f"PII Models ({','.join(app_config.pii_languages)})", expand=True),
            pii_model_status,
            ft.ElevatedButton("Download", disabled=True)
        ]),
        ft.Row([ft.Text("Fido/PRONOM Signatures", expand=True), fido_status, ft.ElevatedButton("Download", disabled=True)]),
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
                ft.Row([ft.ElevatedButton("Add Directory", icon=ft.Icons.FOLDER_OPEN, on_click=lambda _: directory_picker.get_directory_path())]),
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
                ft.ElevatedButton("Save Settings", disabled=True), # Disabled for now
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

    def navigate(e):
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
        page.update()

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

    page.update()

    # --- Graceful Shutdown Handler ---
    # This handles the window closing event to prevent a common Flutter error on exit.
    def window_event(e):
        if e.data == "close":
            page.window_destroy()

    page.on_window_event = window_event