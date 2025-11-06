import sys
from pathlib import Path
from datetime import datetime
import multiprocessing
import tomllib, click

import spacy
from PySide6.QtCore import QObject, QThread, Signal, Slot, QFile, QSize, Qt
from PySide6.QtGui import QPalette, QIcon, QPixmap
from PySide6.QtUiTools import QUiLoader
from PySide6.QtWidgets import (QApplication, QCheckBox, QFormLayout,
                               QGroupBox, QHBoxLayout, QLabel, QLineEdit, QHeaderView, QTextBrowser,
                               QListWidget, QListWidgetItem, QMainWindow, QTreeWidget, QTreeWidgetItem,
                               QMessageBox, QProgressBar, QPushButton,
                               QSpinBox, QStackedWidget, QTabWidget, QSpacerItem, QSizePolicy,
                               QVBoxLayout, QWidget, QDoubleSpinBox, QTextEdit, QTableWidget, QTableWidgetItem,
                               QFileDialog)

import attrs
from . import config, database, download_assets, indexer, models
from .main import format_bytes
from .reporter import Reporter

#<editor-fold desc="Asset Check and Download Workers">
class AssetCheckWorker(QObject):
    """
    A worker that runs in a separate thread to check for assets without
    freezing the GUI. It communicates its findings via signals.
    """
    # Signal format: asset_name, status_text, color, is_downloaded
    checking_asset = Signal(str)
    status_updated = Signal(str, str, str, bool)
    finished = Signal()

    def __init__(self, app_config):
        super().__init__()
        self.app_config = app_config

    @Slot()
    def run(self):
        """The main work of the thread."""
        script_dir = Path(__file__).parent
        models_dir = script_dir / "models"

        # 1. Check Embedding Model
        self.checking_asset.emit("embedding")
        model_path = script_dir / (self.app_config.embedding_model_path or "models/all-MiniLM-L6-v2")
        embedding_ok = model_path.is_dir()
        self.status_updated.emit(
            "embedding", "Available" if embedding_ok else "Not Found",
            "green" if embedding_ok else "orange", embedding_ok
        )

        # 2. Check PII (spaCy) Models
        self.checking_asset.emit("pii")
        pii_ok = True
        for lang in self.app_config.pii_languages:
            model_name = config.get_spacy_model_name(lang)
            if not spacy.util.is_package(model_name):
                pii_ok = False
                break
        self.status_updated.emit(
            "pii", "Available" if pii_ok else "Not Found",
            "green" if pii_ok else "orange", pii_ok
        )

        # 3. Check Fido Signatures
        self.checking_asset.emit("fido")
        fido_sig_path = models_dir / "fido_cache" / "DROID_SignatureFile.xml"
        fido_ok = fido_sig_path.is_file()
        self.status_updated.emit(
            "fido", "Available" if fido_ok else "Not Found",
            "green" if fido_ok else "orange", fido_ok
        )

        self.finished.emit()

class AssetDownloadWorker(QObject):
    """
    A worker that runs in a separate thread to download a specific asset.
    """
    # Signal format: asset_name, progress_text
    # Signal format: asset_name, progress_text
    progress = Signal(str, str)
    finished = Signal()

    def __init__(self, app_config, asset_type: str):
        super().__init__()
        self.app_config = app_config
        self.asset_type = asset_type

    @Slot()
    def run(self):
        """The main work of the thread: download the specified asset."""
        script_dir = Path(__file__).parent
        models_dir = script_dir / "models"
        
        if self.asset_type == "embedding":
            download_assets.download_sentence_transformer('all-MiniLM-L6-v2', models_dir / 'all-MiniLM-L6-v2',
                                                          progress_callback=lambda asset, msg: self.progress.emit(asset, msg))
        elif self.asset_type == "pii":
            for lang in self.app_config.pii_languages:
                model_name = config.get_spacy_model_name(lang)
                download_assets.download_spacy_model(model_name, lang,
                                                     progress_callback=lambda asset, msg: self.progress.emit(asset, msg))
        elif self.asset_type == "fido":
            download_assets.cache_fido_signatures(models_dir / 'fido_cache',
                                                  progress_callback=lambda asset, msg: self.progress.emit(asset, msg))
        
        self.finished.emit()
#</editor-fold>

#<editor-fold desc="Settings and Asset Management Widget">
class OfflineAssetsWidget(QWidget):
    """
    A widget that displays the status of offline assets and allows downloading.
    This will be integrated into the full SettingsViewWidget.
    """
    def __init__(self, app_config: config.Config):
        super().__init__()
        self.app_config = app_config

        main_layout = QVBoxLayout(self)
        group_box = QGroupBox("Offline Assets") # Group box for this section
        main_layout.addWidget(group_box)

        form_layout = QFormLayout()
        group_box.setLayout(form_layout)

        # --- Controls ---
        controls_layout = QHBoxLayout()
        self.check_assets_button = QPushButton("Check Asset Status")
        self.check_assets_button.clicked.connect(self.run_asset_check)
        controls_layout.addWidget(self.check_assets_button)
        controls_layout.addStretch()
        main_layout.addLayout(controls_layout)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 3) # 3 assets to check
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)

        # --- Widgets for each asset ---
        self.embedding_status_label = QLabel("Unknown")
        self.embedding_download_button = QPushButton("Download")
        self.embedding_download_button.clicked.connect(lambda: self.start_download("embedding"))
        form_layout.addRow("Semantic Search Model:", self._create_asset_row(self.embedding_status_label, self.embedding_download_button))

        self.pii_status_label = QLabel("Unknown")
        self.pii_download_button = QPushButton("Download")
        self.pii_download_button.clicked.connect(lambda: self.start_download("pii"))
        form_layout.addRow(f"PII Models ({','.join(self.app_config.pii_languages)}):", self._create_asset_row(self.pii_status_label, self.pii_download_button))

        self.fido_status_label = QLabel("Unknown")
        self.fido_download_button = QPushButton("Download")
        self.fido_download_button.clicked.connect(lambda: self.start_download("fido"))
        form_layout.addRow("Fido/PRONOM Signatures:", self._create_asset_row(self.fido_status_label, self.fido_download_button))

        # Keep track of the currently running thread to prevent multiple operations
        self.active_thread = None
        self.worker = None

        # Set initial state
        self.on_check_finished()

    def _create_asset_row(self, status_label: QLabel, download_button: QPushButton) -> QWidget:
        """Helper to create a consistent layout for each asset row."""
        row_widget = QWidget()
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)
        status_label.setStyleSheet("font-style: italic;")
        row_layout.addWidget(status_label)
        row_layout.addStretch()
        row_layout.addWidget(download_button)
        return row_widget

    def run_asset_check(self):
        """
        Sets up and starts the background thread for checking asset status.
        """
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0) # Reset progress bar
        self.check_assets_button.setDisabled(True)

        for _, button in self._get_asset_widgets().values():
            button.setEnabled(False)

        # 1. Create a worker and a thread
        self.active_thread = QThread(self)
        self.worker = AssetCheckWorker(self.app_config)
        self.worker.moveToThread(self.active_thread)

        # 2. Connect signals from the worker to slots in this widget
        self.worker.checking_asset.connect(self.on_checking_asset)
        self.worker.status_updated.connect(self.on_status_updated)
        self.worker.finished.connect(self.worker.deleteLater)
        self.active_thread.finished.connect(self.active_thread.deleteLater)
        self.worker.finished.connect(self.on_check_finished) # Use a specific slot for check finished
        self.active_thread.started.connect(self.worker.run)

        # 3. Start the thread
        self.active_thread.start()

    def _get_asset_widgets(self):
        """Helper to get a fresh dictionary of asset widgets."""
        return {
            "embedding": (self.embedding_status_label, self.embedding_download_button),
            "pii": (self.pii_status_label, self.pii_download_button),
            "fido": (self.fido_status_label, self.fido_download_button),
        }

    @Slot(str)
    def on_checking_asset(self, asset_name: str):
        """Updates the UI to show which asset is currently being checked."""
        status_label, _ = self._get_asset_widgets()[asset_name]
        status_label.setText("Checking...")
        self.progress_bar.setValue(self.progress_bar.value() + 1)
        status_label.setStyleSheet("color: gray; font-style: italic;")

    @Slot(str, str, str, bool)
    def on_status_updated(self, asset_name: str, status_text: str, color: str, is_downloaded: bool):
        """
        This slot is executed on the main GUI thread when the worker emits a
        status_updated signal.
        """
        status_label, download_button = self._get_asset_widgets()[asset_name]
        status_label.setText(status_text)
        status_label.setStyleSheet(f"color: {color}; font-style: normal;")

        download_button.setText("Downloaded" if is_downloaded else "Download")
        download_button.setEnabled(not is_downloaded)

        # Special case for Fido button
        if asset_name == "fido" and not self.app_config.use_fido:
            download_button.setEnabled(False)

    @Slot(str, str)
    def on_download_progress(self, asset_name: str, progress_text: str):
        """Updates the status label for the specific asset during download."""
        status_label, _ = self._get_asset_widgets()[asset_name]
        status_label.setText(progress_text)
        status_label.setStyleSheet("font-style: italic; color: blue;")

    @Slot()
    def on_check_finished(self):
        """
        This slot is executed on the main GUI thread when the check worker is done.
        """
        self.check_assets_button.setDisabled(False)
        self.progress_bar.setVisible(False)
        self.progress_bar.setValue(0)
        # Re-enable download buttons for assets that are not downloaded
        for asset_name, (status_label, button) in self._get_asset_widgets().items():
            button.setEnabled(status_label.text() not in ["Available", "Downloaded"])

        self.active_thread = None
        self.worker = None

    def start_download(self, asset_type: str):
        """
        Sets up and starts a background thread for downloading an asset.
        """
        self.progress_bar.setVisible(True)
        for _, button in self._get_asset_widgets().values():
            button.setEnabled(False)

        # 1. Create a worker and a thread
        self.active_thread = QThread(self)
        self.worker = AssetDownloadWorker(self.app_config, asset_type)
        self.worker.moveToThread(self.active_thread)

        # 2. Connect signals
        self.worker.progress.connect(self.on_download_progress)
        self.worker.finished.connect(self.worker.deleteLater)
        self.active_thread.finished.connect(self.active_thread.deleteLater)
        # When download finishes, clean up and re-run the check.
        self.worker.finished.connect(self.on_check_finished) # Use the check finished slot for cleanup
        self.worker.finished.connect(self.run_asset_check)
        self.active_thread.started.connect(self.worker.run)

        # 3. Start the thread
        self.active_thread.start()
#</editor-fold>

#<editor-fold desc="Settings View Widget">
class SettingsViewWidget(QWidget):
    """
    A widget for managing application settings.
    """
    settings_saved = Signal()
    def __init__(self, app_config: config.Config, config_path: Path | None):
        super().__init__()
        self.app_config = app_config
        self.config_path = config_path

        # --- Load UI from .ui file ---
        ui_file_path = Path(__file__).parent / "settings_view.ui"
        loader = QUiLoader()
        # The first argument to load is the .ui file, the second is the parent widget (self)
        self.ui = loader.load(str(ui_file_path), self)

        # --- Set up layout ---
        layout = QVBoxLayout(self)
        layout.addWidget(self.ui)

        # --- Populate fields and connect signals ---
        self.ui.config_path_label.setText(f"Editing: {self.config_path}" if self.config_path else "No wanderer.toml found. Using default settings.")
        
        self.workers_field.setRange(1, multiprocessing.cpu_count() * 2)
        self.workers_field.setValue(self.app_config.workers)
        self.db_batch_size_field.setRange(100, 10000)
        self.db_batch_size_field.setSingleStep(100)
        self.db_batch_size_field.setValue(self.app_config.db_batch_size)
        self.memory_limit_field.setRange(0.0, 100.0)
        self.memory_limit_field.setSingleStep(0.5)
        self.memory_limit_field.setDecimals(1)
        self.memory_limit_field.setValue(self.app_config.memory_limit_gb or 0.0)

        self.use_fido_switch.setChecked(self.app_config.use_fido)
        self.extract_text_switch.setChecked(self.app_config.extract_text_on_scan)
        self.phash_switch.setChecked(self.app_config.compute_perceptual_hash)
        self.pii_languages_field.setText(",".join(self.app_config.pii_languages))
        self.archive_excludes_field.setText(",".join(self.app_config.archive_exclude_extensions))

        self.update_scan_dirs_list()
        self.update_exclude_dirs_list()
        
        self.ui.add_scan_dir_button.clicked.connect(self.add_scan_dir)
        self.ui.add_exclude_dir_button.clicked.connect(self.add_excluded_dir)
        self.ui.save_settings_button.clicked.connect(self.save_settings)
        
        # Offline Assets Widget
        self.offline_assets_widget = OfflineAssetsWidget(self.app_config)
        # Find the spacer and insert the widget before it
        spacer_item = self.ui.main_layout.itemAt(self.ui.main_layout.count() - 2)
        self.ui.main_layout.insertWidget(self.ui.main_layout.indexOf(spacer_item), self.offline_assets_widget)

    # --- Magic properties to access UI elements easily ---
    def __getattr__(self, name):
        # This allows you to access widgets from the .ui file as if they were
        # attributes of this class, e.g., self.workers_field
        widget = self.ui.findChild(QWidget, name)
        if widget:
            return widget
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def update_scan_dirs_list(self):
        self.scan_dirs_list.clear()
        for d in sorted(self.app_config.scan_dirs):
            item = QListWidgetItem(d)
            self.scan_dirs_list.addItem(item)

    def update_exclude_dirs_list(self):
        self.exclude_dirs_list.clear()
        for d in sorted(self.app_config.exclude_dirs):
            item = QListWidgetItem(d)
            self.exclude_dirs_list.addItem(item)

    def add_scan_dir(self):
        dialog = QFileDialog(self)
        dialog.setFileMode(QFileDialog.Directory)
        if dialog.exec():
            selected_dir = dialog.selectedFiles()[0]
            if selected_dir and selected_dir not in self.app_config.scan_dirs:
                self.app_config.scan_dirs.append(selected_dir)
                self.update_scan_dirs_list()

    def add_excluded_dir(self):
        dir_to_exclude = self.new_exclude_dir_field.text().strip()
        if dir_to_exclude and dir_to_exclude not in self.app_config.exclude_dirs:
            self.app_config.exclude_dirs.append(dir_to_exclude)
            self.new_exclude_dir_field.clear()
            self.update_exclude_dirs_list()

    def save_settings(self):
        # Update app_config from UI fields
        self.app_config.workers = self.workers_field.value()
        self.app_config.db_batch_size = self.db_batch_size_field.value()
        self.app_config.memory_limit_gb = self.memory_limit_field.value() if self.memory_limit_field.value() > 0 else None
        self.app_config.use_fido = self.use_fido_switch.isChecked()
        self.app_config.extract_text_on_scan = self.extract_text_switch.isChecked()
        self.app_config.compute_perceptual_hash = self.phash_switch.isChecked()
        self.app_config.pii_languages = [lang.strip() for lang in self.pii_languages_field.text().split(',') if lang.strip()]
        self.app_config.archive_exclude_extensions = [ext.strip() for ext in self.archive_excludes_field.text().split(',') if ext.strip()]

        config_to_save_path = self.config_path

        # If no config file exists, prompt the user to create one.
        if not config_to_save_path:
            dialog = QFileDialog()
            file_path, _ = dialog.getSaveFileName(self, "Save Configuration File", "wanderer.toml", "TOML Files (*.toml)")
            
            if not file_path:
                return  # User cancelled
            
            config_to_save_path = Path(file_path)
            self.config_path = config_to_save_path # Update path for this widget
            self.ui.config_path_label.setText(f"Editing: {self.config_path}")

        try:
            # If the config file doesn't exist, create it.
            if not config_to_save_path.exists():
                config_to_save_path.parent.mkdir(parents=True, exist_ok=True)
                full_toml = {} # Start with an empty TOML structure
            else:
                with open(config_to_save_path, "rb") as f:
                    full_toml = tomllib.load(f)
            
            if "tool" not in full_toml:
                full_toml["tool"] = {}
            if "wanderer" not in full_toml["tool"]:
                full_toml["tool"]["wanderer"] = {}
            
            full_toml["tool"]["wanderer"] = config.config_to_dict(self.app_config)
            config.save_config_to_path(full_toml, config_to_save_path)
            QMessageBox.information(self, "Settings Saved", "Settings have been saved successfully!")
            self.settings_saved.emit()
            self.offline_assets_widget.run_asset_check() # Re-check asset status after saving settings
        except Exception as ex:
            QMessageBox.critical(self, "Error Saving Settings", f"Failed to save settings: {ex}")


#</editor-fold>

#<editor-fold desc="Generic Worker">
class GenericWorker(QObject):
    """A generic worker that can run any function with arguments."""
    finished = Signal(object)  # Emits the return value of the function
    error = Signal(str)

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

    @Slot()
    def run(self):
        """Executes the function and emits the result or an error."""
        try:
            result = self.fn(*self.args, **self.kwargs)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))
#</editor-fold>

#<editor-fold desc="Scan and Refine Workers">
class ScanWorker(QObject):
    """Worker to run the main indexing job."""
    progress = Signal(int, int, str)
    finished = Signal(str)

    def __init__(self, scan_config, selected_paths):
        super().__init__()
        self.scan_config = scan_config
        self.selected_paths = selected_paths

    @Slot()
    def run(self):
        def progress_callback(value, total, description):
            self.progress.emit(value, total or 0, description)

        idx = indexer.Indexer(
            root_paths=tuple(self.selected_paths),
            workers=self.scan_config.workers,
            memory_limit_gb=self.scan_config.memory_limit_gb,
            exclude_paths=(),
            app_config=self.scan_config,
            progress_callback=progress_callback
        )
        try:
            idx.run()
            self.finished.emit("Scan finished successfully.")
        except Exception as e:
            self.finished.emit(f"Scan failed: {e}")

#</editor-fold>

class ScanViewWidget(QWidget):
    """The main view for scanning and refinement."""
    def __init__(self, app_config: config.Config):
        super().__init__()
        self.app_config = app_config
        self.worker = None
        self.active_thread = None

        # Main layout for the Scan View
        layout = QVBoxLayout(self)
        tab_widget = QTabWidget()
        layout.addWidget(tab_widget)

        # --- New Scan Tab ---
        new_scan_tab = QWidget()
        new_scan_layout = QVBoxLayout(new_scan_tab)

        # Scan Targets
        scan_targets_group = QGroupBox("Select directories to scan")
        self.scan_targets_layout = QVBoxLayout()
        scan_targets_group.setLayout(self.scan_targets_layout)
        self.scan_target_checkboxes = []
        # Populate scan directories from config
        if self.app_config.scan_dirs:
            for d in self.app_config.scan_dirs:
                cb = QCheckBox(d)
                cb.setChecked(True)
                self.scan_targets_layout.addWidget(cb)
                # Store checkbox references to read their state later
                self.scan_target_checkboxes.append(cb)
        else:
            self.scan_targets_layout.addWidget(QLabel("No scan directories configured in Settings."))
        new_scan_layout.addWidget(scan_targets_group)

        # Scan Options
        scan_options_group = QGroupBox("Scan Options")
        scan_options_layout = QHBoxLayout()
        scan_options_group.setLayout(scan_options_layout)
        self.scan_option_text = QCheckBox("Extract Text & PII")
        self.scan_option_text.setChecked(self.app_config.extract_text_on_scan)
        self.scan_option_phash = QCheckBox("Compute Perceptual Hashes")
        self.scan_option_phash.setChecked(self.app_config.compute_perceptual_hash)
        self.scan_option_fido = QCheckBox("Enable Fido for unknown types")
        self.scan_option_fido.setChecked(self.app_config.use_fido)
        scan_options_layout.addWidget(self.scan_option_text)
        scan_options_layout.addWidget(self.scan_option_phash)
        scan_options_layout.addWidget(self.scan_option_fido)
        new_scan_layout.addWidget(scan_options_group)

        # Scan Controls
        scan_controls_layout = QHBoxLayout()
        self.start_scan_button = QPushButton("Start Scan")
        self.start_scan_button.clicked.connect(self.start_scan)
        self.start_scan_button.setDisabled(not self.app_config.scan_dirs)
        
        self.stop_scan_button = QPushButton("Stop Scan")
        self.stop_scan_button.setVisible(False)
        scan_controls_layout.addWidget(self.start_scan_button)
        scan_controls_layout.addWidget(self.stop_scan_button)
        scan_controls_layout.addStretch()
        new_scan_layout.addLayout(scan_controls_layout)

        self.scan_progress = QProgressBar()
        self.scan_progress.setVisible(False)
        self.scan_status_label = QLabel("Idle.")
        new_scan_layout.addWidget(self.scan_progress)
        new_scan_layout.addWidget(self.scan_status_label)
        new_scan_layout.addStretch()

        # --- Scan History Tab ---
        scan_history_tab = QWidget()
        scan_history_layout = QVBoxLayout(scan_history_tab)
        self.scan_history_list = QListWidget()
        refresh_history_button = QPushButton("Refresh History")
        refresh_history_button.clicked.connect(self.refresh_scan_history)
        scan_history_layout.addWidget(refresh_history_button)
        scan_history_layout.addWidget(self.scan_history_list)

        # --- Refine Data Tab ---
        refine_data_tab = QWidget()
        refine_layout = QVBoxLayout(refine_data_tab)
        refine_layout.addWidget(QLabel("Run deep analysis on files already in the database. This is useful after a fast initial scan."))
        
        self.refine_fido_button = QPushButton("Refine Unknown Files (Fido)")
        self.refine_fido_button.clicked.connect(lambda: self.start_refine("fido"))
        self.refine_fido_button.setDisabled(not self.app_config.use_fido)
        refine_layout.addWidget(self.refine_fido_button)

        self.refine_text_button = QPushButton("Refine Skipped Text")
        self.refine_text_button.clicked.connect(lambda: self.start_refine("text"))
        refine_layout.addWidget(self.refine_text_button)

        refine_layout.addStretch()
        # ... (Refinement progress and status could be added here similar to scan)

        tab_widget.addTab(new_scan_tab, "New Scan")
        tab_widget.addTab(scan_history_tab, "Scan History")
        tab_widget.addTab(refine_data_tab, "Refine Data")

        self.refresh_scan_history()

    def refresh_view(self, new_config: config.Config):
        """
        Updates the entire scan view based on the current app_config.
        This is called when settings are saved elsewhere in the application.
        """
        self.app_config = new_config
        # Refresh scan option checkboxes

        self.scan_option_text.setChecked(self.app_config.extract_text_on_scan)
        self.scan_option_phash.setChecked(self.app_config.compute_perceptual_hash)
        self.scan_option_fido.setChecked(self.app_config.use_fido)
        self.refine_fido_button.setDisabled(not self.app_config.use_fido)

        # Refresh the list of scannable directories
        # Clear existing widgets from the layout
        while self.scan_targets_layout.count():
            child = self.scan_targets_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        
        self.scan_target_checkboxes = []
        if self.app_config.scan_dirs:
            for d in self.app_config.scan_dirs:
                cb = QCheckBox(d)
                cb.setChecked(True)
                self.scan_targets_layout.addWidget(cb)
                self.scan_target_checkboxes.append(cb)
            self.start_scan_button.setDisabled(False)
        else:
            self.scan_targets_layout.addWidget(QLabel("No scan directories configured in Settings."))
            self.start_scan_button.setDisabled(True)

    def set_scan_ui_state(self, is_scanning: bool, message: str = ""):
        """Toggles the UI between scanning and idle states."""
        self.start_scan_button.setVisible(not is_scanning)
        self.stop_scan_button.setVisible(is_scanning)
        self.scan_progress.setVisible(is_scanning)

        for cb in self.scan_target_checkboxes:
            cb.setDisabled(is_scanning)
        self.scan_option_text.setDisabled(is_scanning)
        self.scan_option_phash.setDisabled(is_scanning)
        self.scan_option_fido.setDisabled(is_scanning)

        self.scan_status_label.setText(message or ("Starting scan..." if is_scanning else "Idle."))
        if is_scanning:
            self.scan_progress.setRange(0, 0) # Indeterminate
            self.scan_progress.setValue(0)

    def start_scan(self):
        self.set_scan_ui_state(True)

        selected_paths = [Path(cb.text()) for cb in self.scan_target_checkboxes if cb.isChecked()]
        scan_config = config.Config(
            extract_text_on_scan=self.scan_option_text.isChecked(),
            compute_perceptual_hash=self.scan_option_phash.isChecked(),
            use_fido=self.scan_option_fido.isChecked(),
            # Inherit other settings from app_config
            workers=self.app_config.workers,
            db_batch_size=self.app_config.db_batch_size,
            exclude_dirs=self.app_config.exclude_dirs,
            scan_dirs=self.app_config.scan_dirs,
            pii_languages=self.app_config.pii_languages,
            memory_limit_gb=self.app_config.memory_limit_gb,
            embedding_model_path=self.app_config.embedding_model_path,
            archive_exclude_extensions=self.app_config.archive_exclude_extensions,
        )

        self.active_thread = QThread(self)
        self.worker = ScanWorker(scan_config, selected_paths)
        self.worker.moveToThread(self.active_thread)

        self.worker.progress.connect(self.on_scan_progress)
        self.stop_scan_button.clicked.connect(self.active_thread.requestInterruption) # Allow stopping scan
        self.worker.finished.connect(self.on_scan_finished)
        self.active_thread.started.connect(self.worker.run)
        self.active_thread.start()

    @Slot(int, int, str)
    def on_scan_progress(self, value, total, description):
        self.scan_status_label.setText(description)
        if total > 1:
            self.scan_progress.setRange(0, total)
            self.scan_progress.setValue(value)
        else:
            self.scan_progress.setRange(0, 0)

    @Slot(str)
    def on_scan_finished(self, message):
        self.set_scan_ui_state(False, message)
        self.refresh_scan_history()

        self.worker = None
        self.active_thread = None
    def start_refine(self, refine_type: str):
        # Disable buttons and show progress (similar to scan)
        self.refine_fido_button.setDisabled(True)
        self.refine_text_button.setDisabled(True)
        # Add a progress bar and status label for refinement if needed
        # For now, just a message box
        QMessageBox.information(self, "Refinement Started", f"{refine_type.capitalize()} refinement started in background.")

        # Create a temporary config object for refinement
        refine_config = attrs.evolve(
            self.app_config,
            extract_text_on_scan=refine_type == "text",
            compute_perceptual_hash=False,  # Not used in these jobs
            use_fido=refine_type == "fido",
        )

        # Define the function to be run in the background
        def refine_task(idx_instance, task_type):
            if task_type == "fido":
                idx_instance.refine_unknown_files()
                return "Fido refinement finished successfully."
            elif task_type == "text":
                idx_instance.refine_text_content()
                return "Text refinement finished successfully."
            return "Unknown refinement task."

        idx = indexer.Indexer(root_paths=(), workers=refine_config.workers, memory_limit_gb=refine_config.memory_limit_gb, exclude_paths=(), app_config=refine_config)

        self.active_thread = QThread(self)
        # Use the GenericWorker
        self.worker = GenericWorker(refine_task, idx, refine_type)
        self.worker.moveToThread(self.active_thread)

        def on_refine_finished(message):
            QMessageBox.information(self, "Refinement Status", message)
            self.refine_fido_button.setDisabled(not self.app_config.use_fido)
            self.refine_text_button.setDisabled(False)
            self.active_thread.quit()
            self.active_thread.wait() # type: ignore

        self.worker.finished.connect(on_refine_finished)
        self.worker.error.connect(lambda msg: QMessageBox.critical(self, "Refinement Error", msg))

        self.active_thread.started.connect(self.worker.run)
        self.active_thread.start()

    def refresh_scan_history(self):
        """Fetches scan history in a background thread to keep the GUI responsive."""
        def get_history():
            with database.get_session() as db:
                return db.query(models.ScanLog).order_by(models.ScanLog.start_time.desc()).limit(50).all()

        @Slot(object)
        def on_history_loaded(logs):
            self.scan_history_list.clear()
            if not logs:
                self.scan_history_list.addItem("No scan history found.")
            else:
                for log in logs:
                    end_time_str = log.end_time.strftime('%Y-%m-%d %H:%M:%S') if log.end_time else "In Progress"
                    item_text = f"[{log.status.upper()}] {log.start_time.strftime('%Y-%m-%d %H:%M:%S')} -> {end_time_str} ({log.files_scanned} files)"
                    self.scan_history_list.addItem(item_text)
            self.active_thread.quit()
            self.active_thread.wait()

        self.active_thread = QThread(self)
        self.worker = GenericWorker(get_history)
        self.worker.moveToThread(self.active_thread)
        self.worker.finished.connect(on_history_loaded)
        self.active_thread.started.connect(self.worker.run)
        self.active_thread.start()

class SearchViewWidget(QWidget):
    def __init__(self, app_config: config.Config):
        super().__init__()
        self.app_config = app_config
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Semantic Search (Not yet implemented)"))
        layout.addWidget(QLineEdit(placeholderText="Search for a concept or idea..."))
        layout.addWidget(QPushButton("Search"))
        layout.addStretch()


class ReportsViewWidget(QWidget):
    def __init__(self, app_config: config.Config):
        super().__init__()
        self.app_config = app_config
        self.worker = None
        self.active_thread = None

        layout = QVBoxLayout(self)
        tab_widget = QTabWidget()
        layout.addWidget(tab_widget)

        # --- Type Summary Tab ---
        type_summary_tab = QWidget()
        type_summary_layout = QVBoxLayout(type_summary_tab)
        self.run_type_summary_button = QPushButton("Run Type Summary Report")
        self.run_type_summary_button.clicked.connect(self.run_type_summary)
        self.type_summary_table = QTableWidget()
        self.type_summary_table.setColumnCount(3)
        self.type_summary_table.setHorizontalHeaderLabels(["MIME Type", "File Count", "Total Size"])
        self.type_summary_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.type_summary_table.setSortingEnabled(True)
        type_summary_layout.addWidget(self.run_type_summary_button)
        type_summary_layout.addWidget(self.type_summary_table)
        tab_widget.addTab(type_summary_tab, "File Type Summary")

        # --- Largest Files Tab ---
        largest_files_tab = QWidget()
        largest_files_layout = QVBoxLayout(largest_files_tab)
        largest_files_controls = QHBoxLayout()
        largest_files_controls.addWidget(QLabel("Number of files to show:"))
        self.largest_files_limit = QSpinBox()
        self.largest_files_limit.setRange(1, 1000)
        self.largest_files_limit.setValue(25)
        self.run_largest_files_button = QPushButton("Find Largest Files")
        self.run_largest_files_button.clicked.connect(self.run_largest_files)
        largest_files_controls.addWidget(self.largest_files_limit)
        largest_files_controls.addWidget(self.run_largest_files_button)
        largest_files_controls.addStretch()
        self.largest_files_table = QTableWidget()
        self.largest_files_table.setColumnCount(2)
        self.largest_files_table.setHorizontalHeaderLabels(["Size", "Path"])
        self.largest_files_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.largest_files_table.setSortingEnabled(True)
        largest_files_layout.addLayout(largest_files_controls)
        largest_files_layout.addWidget(self.largest_files_table)
        tab_widget.addTab(largest_files_tab, "Largest Files")

        # --- PII Files Tab ---
        pii_files_tab = QWidget()
        pii_files_layout = QVBoxLayout(pii_files_tab)
        self.run_pii_files_button = QPushButton("List Files with PII")
        self.run_pii_files_button.clicked.connect(self.run_pii_files)
        self.pii_files_table = QTableWidget()
        self.pii_files_table.setColumnCount(2)
        self.pii_files_table.setHorizontalHeaderLabels(["File Path", "PII Types Found"])
        self.pii_files_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.pii_files_table.setSortingEnabled(True)
        pii_files_layout.addWidget(self.run_pii_files_button)
        pii_files_layout.addWidget(self.pii_files_table)
        tab_widget.addTab(pii_files_tab, "PII Report")

        # --- Find Duplicates Tab ---
        dupes_tab = QWidget()
        self.dupes_layout = QVBoxLayout(dupes_tab)
        self.run_dupes_button = QPushButton("Find Duplicate Files")
        self.run_dupes_button.clicked.connect(self.run_dupes_report)
        self.dupes_tree = QTreeWidget()
        self.dupes_tree.setHeaderLabels(["File Path", "Size"])
        self.dupes_tree.header().setSectionResizeMode(0, QHeaderView.Stretch)
        self.dupes_layout.addWidget(self.run_dupes_button)
        self.dupes_layout.addWidget(self.dupes_tree)
        tab_widget.addTab(dupes_tab, "Duplicate Files")

        # --- PRONOM Summary Tab ---
        pronom_tab = QWidget()
        self.pronom_layout = QVBoxLayout(pronom_tab)
        self.run_pronom_button = QPushButton("Run PRONOM Summary Report")
        self.run_pronom_button.clicked.connect(self.run_pronom_report)
        self.pronom_table = QTableWidget()
        self.pronom_table.setColumnCount(3)
        self.pronom_table.setHorizontalHeaderLabels(["PRONOM ID", "File Count", "Total Size"])
        self.pronom_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.pronom_table.setSortingEnabled(True)
        self.pronom_layout.addWidget(self.run_pronom_button)
        self.pronom_layout.addWidget(self.pronom_table)
        tab_widget.addTab(pronom_tab, "PRONOM Summary")

        # --- Similar Images Tab ---
        similar_images_tab = QWidget()
        similar_images_layout = QVBoxLayout(similar_images_tab)
        similar_images_controls = QHBoxLayout()
        similar_images_controls.addWidget(QLabel("Similarity Threshold (0=exact, higher is less strict):"))
        self.image_threshold_spinbox = QSpinBox()
        self.image_threshold_spinbox.setRange(0, 64)
        self.image_threshold_spinbox.setValue(4)
        self.run_similar_images_button = QPushButton("Find Similar Images")
        self.run_similar_images_button.clicked.connect(self.run_similar_images_report)
        similar_images_controls.addWidget(self.image_threshold_spinbox)
        similar_images_controls.addWidget(self.run_similar_images_button)
        self.similar_images_status = QLabel("")
        similar_images_controls.addWidget(self.similar_images_status)
        self.similar_images_tree = QTreeWidget()
        self.similar_images_tree.setHeaderLabels(["File Path", ""])
        self.similar_images_tree.header().setSectionResizeMode(0, QHeaderView.Stretch)
        similar_images_layout.addLayout(similar_images_controls)
        similar_images_layout.addWidget(self.similar_images_tree)
        tab_widget.addTab(similar_images_tab, "Similar Images")

        # --- Similar Text Tab (Placeholder) ---
        similar_text_tab = QWidget()
        similar_text_layout = QVBoxLayout(similar_text_tab)
        similar_text_layout.addWidget(QLabel("Similar Text Report (Not yet implemented)"))
        tab_widget.addTab(similar_text_tab, "Similar Text")

    def _run_report(self, report_function, on_finish_slot, *args, **kwargs):
        """Generic method to run a report in a background thread."""
        self.active_thread = QThread(self)
        self.worker = GenericWorker(report_function, *args, **kwargs)
        self.worker.moveToThread(self.active_thread)
        self.worker.finished.connect(on_finish_slot)
        self.worker.finished.connect(self.active_thread.quit)
        self.worker.error.connect(lambda msg: QMessageBox.critical(self, "Report Error", msg))
        self.active_thread.started.connect(self.worker.run)
        self.active_thread.start()

    def run_type_summary(self):
        self.run_type_summary_button.setDisabled(True)
        self.type_summary_table.setRowCount(0)
        reporter = Reporter()
        self._run_report(reporter.type_summary, self.on_type_summary_finished, print_output=False)

    @Slot(object)
    def on_type_summary_finished(self, summary_data):
        self.type_summary_table.setRowCount(len(summary_data))
        for row, (mime_type, count, total_size) in enumerate(summary_data):
            self.type_summary_table.setItem(row, 0, QTableWidgetItem(mime_type))
            count_item = QTableWidgetItem()
            count_item.setData(0, count) # Store as number for sorting
            self.type_summary_table.setItem(row, 1, count_item) # type: ignore
            self.type_summary_table.setItem(row, 2, QTableWidgetItem(format_bytes(total_size or 0)))
        self.type_summary_table.resizeColumnsToContents()
        self.type_summary_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.run_type_summary_button.setDisabled(False)

    def run_largest_files(self):
        self.run_largest_files_button.setDisabled(True)
        self.largest_files_table.setRowCount(0)
        limit = self.largest_files_limit.value()
        reporter = Reporter()
        self._run_report(reporter.largest_files, self.on_largest_files_finished, limit, print_output=False)

    @Slot(object)
    def on_largest_files_finished(self, files_data):
        self.largest_files_table.setRowCount(len(files_data))
        for row, file in enumerate(files_data or []):
            size_item = QTableWidgetItem(format_bytes(file.size_bytes))
            size_item.setData(0, file.size_bytes) # Store raw bytes for sorting
            self.largest_files_table.setItem(row, 0, size_item)
            self.largest_files_table.setItem(row, 1, QTableWidgetItem(file.path))
        self.largest_files_table.resizeColumnsToContents()
        self.largest_files_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.run_largest_files_button.setDisabled(False)

    def run_pii_files(self):
        self.run_pii_files_button.setDisabled(True)
        self.pii_files_table.setRowCount(0)
        reporter = Reporter()
        self._run_report(reporter.list_pii_files, self.on_pii_files_finished, print_output=False)

    @Slot(object)
    def on_pii_files_finished(self, pii_data):
        self.pii_files_table.setRowCount(len(pii_data))
        for row, file in enumerate(pii_data or []):
            self.pii_files_table.setItem(row, 0, QTableWidgetItem(file.path))
            self.pii_files_table.setItem(row, 1, QTableWidgetItem(", ".join(file.pii_types)))
        self.pii_files_table.resizeColumnsToContents()
        self.pii_files_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.run_pii_files_button.setDisabled(False)

    def run_dupes_report(self):
        self.run_dupes_button.setDisabled(True)
        self.dupes_tree.clear()
        reporter = Reporter()
        self._run_report(reporter.find_dupes, self.on_dupes_finished, print_output=False)

    @Slot(object)
    def on_dupes_finished(self, dupes_data):
        for i, (hash_val, files) in enumerate(dupes_data):
            source_file = files[0]
            group_item = QTreeWidgetItem(self.dupes_tree, [f"Duplicate Set {i+1} ({len(files)} files, hash: {hash_val[:12]}...)", format_bytes(source_file.size_bytes)])
            
            source_item = QTreeWidgetItem(group_item, [source_file.path, "Source"])
            source_item.setForeground(0, QApplication.palette().color(QPalette.ColorRole.Link))

            for dupe_file in files[1:]:
                QTreeWidgetItem(group_item, [dupe_file.path, ""])
        
        self.dupes_tree.expandAll()
        self.dupes_tree.resizeColumnToContents(1)
        self.run_dupes_button.setDisabled(False)

    def run_pronom_report(self):
        self.run_pronom_button.setDisabled(True)
        self.pronom_table.setRowCount(0)
        reporter = Reporter()
        self._run_report(reporter.pronom_summary, self.on_pronom_summary_finished, print_output=False)

    @Slot(object)
    def on_pronom_summary_finished(self, summary_data):
        self.pronom_table.setRowCount(len(summary_data))
        for row, (pronom_id, count, total_size) in enumerate(summary_data or []):
            self.pronom_table.setItem(row, 0, QTableWidgetItem(pronom_id))
            self.pronom_table.setItem(row, 1, QTableWidgetItem(str(count)))
            self.pronom_table.setItem(row, 2, QTableWidgetItem(format_bytes(total_size or 0)))
        self.pronom_table.resizeColumnsToContents()
        self.pronom_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.run_pronom_button.setDisabled(False)

    def run_similar_images_report(self):
        self.run_similar_images_button.setDisabled(True)
        self.similar_images_tree.clear()
        self.similar_images_status.setText("Running report...")
        threshold = self.image_threshold_spinbox.value()
        reporter = Reporter()
        self._run_report(reporter.find_image_dupes, self.on_similar_images_finished, threshold, print_output=False)

    @Slot(object)
    def on_similar_images_finished(self, groups):
        self.similar_images_status.setText(f"Found {len(groups or [])} groups of similar images.")
        if not groups:
            self.run_similar_images_button.setDisabled(False)
            return

        self.similar_images_tree.setIconSize(QSize(128, 128))

        for i, group in enumerate(groups or []):
            group_item = QTreeWidgetItem(self.similar_images_tree, [f"Similar Group {i+1} ({len(group)} images)", ""])
            for path in group:
                item = QTreeWidgetItem(group_item, [path, ""])
                pixmap = QPixmap(path)
                if not pixmap.isNull():
                    item.setIcon(0, QIcon(pixmap.scaled(QSize(128, 128), Qt.KeepAspectRatio, Qt.SmoothTransformation)))
        self.similar_images_tree.expandAll()
        self.run_similar_images_button.setDisabled(False)

class HelpViewWidget(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        text_edit = QTextBrowser()
        text_edit.setReadOnly(True)
        text_edit.setOpenExternalLinks(True)
        layout.addWidget(text_edit)

        help_text = """
        <h2>Welcome to Wanderer!</h2>
        <p>Wanderer is a powerful tool for indexing and analyzing your local files. This guide will help you get started.</p>

        <h3>Getting Started</h3>
        <ol>
            <li><b>Configure Settings:</b> Go to the <b>Settings</b> tab to tell Wanderer where to find your files and how to process them.</li>
            <li><b>Check Offline Assets:</b> In the Settings tab, use the "Offline Assets" section to download the necessary AI models for semantic search and PII detection.</li>
            <li><b>Start a Scan:</b> Go to the <b>Scan</b> tab, select the directories you want to index, and click "Start Scan".</li>
            <li><b>Explore Reports:</b> Once the scan is complete, use the <b>Reports</b> tab to find duplicates, see summaries, and more.</li>
        </ol>

        <h3>Scanning and Refinement</h3>
        <p>The <b>Scan</b> tab is your main control center for indexing.</p>
        <ul>
            <li><b>New Scan:</b> Select directories and choose which processing steps (like text extraction) to run. For a faster initial scan, you can uncheck these options and run them later.</li>
            <li><b>Scan History:</b> View a log of past and ongoing scan jobs.</li>
            <li><b>Refine Data:</b> After a fast initial scan, you can come here to perform deeper analysis on files already in your database. For example, you can run "Refine Skipped Text" to process documents that were skipped during the first pass.</li>
        </ul>

        <h3>Reporting and Analysis</h3>
        <p>The <b>Reports</b> tab lets you gain insights from your indexed files.</p>
        <ul>
            <li><b>File Type Summary:</b> See a breakdown of all file types by MIME type.</li>
            <li><b>Largest Files:</b> Quickly find the biggest files in your index.</li>
            <li><b>PII Report:</b> List all files that have been flagged for containing Personally Identifiable Information.</li>
            <li><b>Duplicate Files:</b> Finds files that are bit-for-bit identical by comparing their cryptographic hashes.</li>
            <li><b>PRONOM Summary:</b> A more technical file type summary based on PRONOM IDs (requires Fido to be enabled in Settings).</li>
            <li><b>Similar Images:</b> Finds visually similar images using perceptual hashing. A threshold of 0 finds exact duplicates, while a small number like 4-5 will find resized or slightly edited copies.</li>
        </ul>

        <h3>Configuration File</h3>
        <p>Your settings are saved to a <code>wanderer.toml</code> file. You can edit this file directly for advanced configuration. The path to the file is shown at the top of the <b>Settings</b> tab.</p>
        """
        text_edit.setHtml(help_text)

class AboutViewWidget(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("About Wanderer (Not yet implemented)"))
        layout.addStretch()

class WandererQtGUI(QMainWindow):
    """The main application window."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Wanderer")
        self.setMinimumSize(1000, 700)

        self.app_config, self.config_path = config.load_config_with_path()

        # Main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Navigation
        self.nav_list = QListWidget()
        self.nav_list.setFixedWidth(150)
        self.nav_list.addItem("Scan")
        self.nav_list.addItem("Search")
        self.nav_list.addItem("Reports")
        self.nav_list.addItem("Settings")
        self.nav_list.addItem("Help")
        self.nav_list.addItem("About")
        main_layout.addWidget(self.nav_list)

        # Content area
        self.stack = QStackedWidget()
        main_layout.addWidget(self.stack)

        # --- Create and add views to the stacked widget ---
        self.scan_view = ScanViewWidget(self.app_config)
        self.stack.addWidget(self.scan_view)

        self.search_view = SearchViewWidget(self.app_config)
        self.stack.addWidget(self.search_view)
        self.reports_view = ReportsViewWidget(self.app_config)
        self.stack.addWidget(self.reports_view)
        self.settings_view = SettingsViewWidget(self.app_config, self.config_path)
        self.stack.addWidget(self.settings_view)
        self.help_view = HelpViewWidget()
        self.stack.addWidget(self.help_view)
        self.about_view = AboutViewWidget()
        self.stack.addWidget(self.about_view)

        # Connect navigation to stack
        self.settings_view.settings_saved.connect(self.on_settings_saved)
        self.nav_list.currentRowChanged.connect(self.stack.setCurrentIndex)
        self.nav_list.setCurrentRow(0)

    @Slot()
    def on_settings_saved(self):
        """Reloads the config and tells all relevant views to update."""
        self.app_config, self.config_path = config.load_config_with_path()
        self.settings_view.config_path = self.config_path # Ensure settings view has the updated path
        self.scan_view.refresh_view(self.app_config)
        # If other views need updating, they can be called here too.

def main_qt():
    """Entry point for the PySide6 GUI."""
    app = QApplication(sys.argv)
    database.init_db()
    window = WandererQtGUI()
    window.show()
    sys.exit(app.exec())