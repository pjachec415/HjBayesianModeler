import sys
import os
import time
import logging
import traceback
import multiprocessing
import subprocess


class _NullWriter:
    def write(self, _):
        return 0

    def flush(self):
        return None


def _configure_pytensor_for_frozen_runtime():
    """
    Configure PyTensor before importing pipeline/pymc.
    In a PyInstaller one-file app, the temporary _MEI folder can break C-linker builds.
    """
    if not getattr(sys, "frozen", False):
        return

    cache_root = os.path.join(
        os.environ.get("LOCALAPPDATA", os.path.expanduser("~")),
        "HjBM",
        "pytensor_cache",
    )
    os.makedirs(cache_root, exist_ok=True)

    # Keep values simple and deterministic for frozen Windows runtime.
    desired = {
        "base_compiledir": cache_root,
        "compiledir_format": "compiledir_%(platform)s-%(python_version)s-%(python_bitwidth)s",
        "linker": "py",
        "cxx": "",
        "mode": "FAST_COMPILE",
    }

    existing = os.environ.get("PYTENSOR_FLAGS", "")
    parsed = {}
    if existing.strip():
        for item in existing.split(","):
            if "=" in item:
                k, v = item.split("=", 1)
                parsed[k.strip()] = v.strip()
            elif item.strip():
                parsed[item.strip()] = ""

    parsed.update(desired)
    os.environ["PYTENSOR_FLAGS"] = ",".join(
        f"{k}={v}" if v != "" else k for k, v in parsed.items()
    )


_configure_pytensor_for_frozen_runtime()

if sys.stdout is None:
    sys.stdout = _NullWriter()
if sys.stderr is None:
    sys.stderr = _NullWriter()

from PySide6.QtWidgets import (
    QApplication, QWidget, QPushButton, QFileDialog, QVBoxLayout, QLabel,
    QLineEdit, QFormLayout, QMessageBox, QSpacerItem, QSizePolicy,
    QTabWidget, QHBoxLayout, QCheckBox, QTextEdit, QProgressBar,
    QGroupBox, QComboBox, QSplashScreen, QFrame, QScrollArea
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer, QSize
from PySide6.QtGui import QPixmap, QFont, QColor, QPainter, QTextCursor, QIcon

from pipeline import run_pipeline
from regression import run_poisson, run_negative_binomial, run_stepwise_nb


# ── Logging redirect ──────────────────────────────────────────────────────────

class QtLogHandler(logging.Handler):
    """Redirect Python logging to the in-app terminal."""
    def __init__(self, signal):
        super().__init__()
        self.signal = signal

    def emit(self, record):
        msg = self.format(record)
        self.signal.emit(msg)


class StreamRedirect:
    """Redirect stdout/stderr to a Qt signal."""
    def __init__(self, signal):
        self.signal = signal
        self._buf = ""

    def write(self, text):
        if text:
            self.signal.emit(text)

    def flush(self):
        pass


# ── Worker thread ─────────────────────────────────────────────────────────────

class Worker(QThread):
    log_signal   = Signal(str)
    done_signal  = Signal(bool, str)   # success, message

    def __init__(self, fn, *args, err_path=None, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.err_path = err_path

    def run(self):
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = StreamRedirect(self.log_signal)
        sys.stderr = StreamRedirect(self.log_signal)
        try:
            self.fn(*self.args, **self.kwargs)
            self.done_signal.emit(True, "Completed successfully.")
        except Exception as e:
            tb = traceback.format_exc()
            self.log_signal.emit(f"\n[ERROR] {e}\n{tb}")
            if self.err_path:
                try:
                    with open(self.err_path, "w", encoding="utf-8") as ef:
                        ef.write(f"{type(e).__name__}: {e}\n\n{tb}\n")
                except Exception:
                    pass
            self.done_signal.emit(False, str(e))
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


# -- Check for C++ Compiler ----------------------------------------------------
def _check_cxx_compiler():
    #Check if a C++ compiler is available for pytensor
    for compiler in ["g++", "cl"]:
        try:
            subprocess.run(
                [compiler, "--version"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False
            )
            return True
        except FileNotFoundError:
            continue
    return False
# ── Splash screen ─────────────────────────────────────────────────────────────

class SplashScreen(QSplashScreen):
    def __init__(self, image_path):
        if os.path.exists(image_path):
            logo_pixmap = QPixmap(image_path).scaled(
                540, 540, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        else:
            logo_pixmap = QPixmap(540, 540)
            logo_pixmap.fill(QColor("#1a1a2e"))
            painter = QPainter(logo_pixmap)
            painter.setPen(QColor("#e0e0e0"))
            font = QFont("Segoe UI", 28, QFont.Bold)
            painter.setFont(font)
            painter.drawText(logo_pixmap.rect(), Qt.AlignCenter, "HjBM")
            painter.end()

        footer_h = 24
        splash_pixmap = QPixmap(logo_pixmap.width(), logo_pixmap.height() + footer_h)
        splash_pixmap.fill(QColor("#11111b"))
        painter = QPainter(splash_pixmap)
        painter.drawPixmap(0, 0, logo_pixmap)
        painter.end()

        super().__init__(splash_pixmap)
        self.setWindowFlags(Qt.SplashScreen | Qt.FramelessWindowHint)

        # Progress bar in footer area (does not cover logo)
        self._progress = QProgressBar(self)
        self._progress.setGeometry(10, logo_pixmap.height() + 5, splash_pixmap.width() - 20, 12)
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setTextVisible(False)
        self._progress.setStyleSheet("""
            QProgressBar { background: #0b1020; border: 1px solid #1f2a44; border-radius: 4px; }
            QProgressBar::chunk { background: #4a90d9; }
        """)

        self._val = 0
        self._timer = QTimer()
        self._timer.timeout.connect(self._tick)

    def start_progress(self):
        self._timer.start(30)

    def _tick(self):
        self._val = min(self._val + 1, 95)
        self._progress.setValue(self._val)

    def finish_progress(self, widget):
        self._progress.setValue(100)
        time.sleep(0.1)
        self._timer.stop()
        self.finish(widget)


# ── Helper: output/log paths ──────────────────────────────────────────────────

def make_output_paths(output_dir, input_path, prefix):
    """
    Returns (artifact_stem, verbose_out_path, err_path, report_path).

    CSV and *_report.out live in output_dir. Verbose transcript (.out) and
    errors-only (.err) live in output_dir/logs/ (created if missing).
    """
    epoch = int(time.time())
    stem  = os.path.splitext(os.path.basename(input_path))[0] if input_path else "output"
    base  = f"{prefix}_{stem}_{epoch}"

    logs_dir = os.path.join(output_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    artifact_stem = os.path.join(output_dir, base)
    verbose_out_path = os.path.join(logs_dir, base + ".out")
    err_path = os.path.join(logs_dir, base + ".err")
    report_path = artifact_stem + "_report.out"
    return artifact_stem, verbose_out_path, err_path, report_path


# ── Main window ───────────────────────────────────────────────────────────────

class PHBayesGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HjBM | A Dedicated HBM GUI App")
        self.setMinimumSize(700, 860)
        self._worker = None

        # Window icon (title bar + taskbar)
        icon_path = os.path.join(os.path.dirname(__file__), "HjBM.ico")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))

        # Shared state
        self.data_path = ""
        self.geo_path  = ""
        self.output_dir = ""   # blank = same as data source

        self._build_ui()
        self._apply_style()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Title bar
        title_bar = QFrame()
        title_bar.setObjectName("titleBar")
        title_bar.setFixedHeight(48)
        tb_layout = QHBoxLayout(title_bar)
        tb_layout.setContentsMargins(16, 0, 16, 0)
        lbl = QLabel("HjBM  |  Hierarchical Bayesian Modeler")
        lbl.setObjectName("titleLabel")
        tb_layout.addWidget(lbl)
        tb_layout.addStretch()
        root.addWidget(title_bar)

        # Tab widget
        self.tabs = QTabWidget()
        self.tabs.setObjectName("mainTabs")
        root.addWidget(self.tabs, 1)

        self._build_settings_tab()
        self._build_regression_tab()
        self._build_hbm_tab()

        # Status bar
        status_bar = QFrame()
        status_bar.setObjectName("statusBar")
        status_bar.setFixedHeight(28)
        sb_layout = QHBoxLayout(status_bar)
        sb_layout.setContentsMargins(12, 0, 12, 0)
        self.status_lbl = QLabel("Ready")
        self.status_lbl.setObjectName("statusLabel")
        sb_layout.addWidget(self.status_lbl)
        sb_layout.addStretch()
        root.addWidget(status_bar)

        # Terminal
        term_group = QGroupBox("Output Terminal")
        term_group.setObjectName("termGroup")
        term_layout = QVBoxLayout(term_group)
        term_layout.setContentsMargins(6, 6, 6, 6)
        self.terminal = QTextEdit()
        self.terminal.setObjectName("terminal")
        self.terminal.setReadOnly(True)
        self.terminal.setMinimumHeight(160)
        self.terminal.setMaximumHeight(260)
        term_layout.addWidget(self.terminal)

        clr_btn = QPushButton("Clear")
        clr_btn.setFixedWidth(70)
        clr_btn.clicked.connect(self.terminal.clear)
        clr_btn.setObjectName("smallBtn")
        term_layout.addWidget(clr_btn, alignment=Qt.AlignRight)
        root.addWidget(term_group)

        # Legal
        legal = QLabel("© 2026 Harrison Jachec. For research use only; not for clinical decisions.")
        legal.setObjectName("legalLabel")
        legal.setAlignment(Qt.AlignCenter)
        root.addWidget(legal)

    # ── Tab 1: Settings ───────────────────────────────────────────────────────

    def _build_settings_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)

        # Input files group
        files_group = QGroupBox("Input Files")
        files_form  = QFormLayout(files_group)
        files_form.setSpacing(10)

        self.data_label = QLabel("No file selected")
        self.data_label.setObjectName("pathLabel")
        self.data_btn = QPushButton("Browse data file (CSV / XLSX)")
        self.data_btn.clicked.connect(self.browse_data)
        files_form.addRow("Data file:", self.data_btn)
        files_form.addRow("", self.data_label)

        self.geo_label = QLabel("No spatial file selected")
        self.geo_label.setObjectName("pathLabel")
        self.geo_btn = QPushButton("Browse spatial file (optional for regression)")
        self.geo_btn.clicked.connect(self.browse_geo)
        files_form.addRow("Spatial file:", self.geo_btn)
        files_form.addRow("", self.geo_label)

        layout.addWidget(files_group)

        # Output location group
        out_group = QGroupBox("Output Location")
        out_form  = QFormLayout(out_group)
        out_form.setSpacing(10)

        self.same_folder_chk = QCheckBox("Save outputs to same folder as source data (default)")
        self.same_folder_chk.setChecked(True)
        self.same_folder_chk.toggled.connect(self._toggle_output_dir)
        out_form.addRow(self.same_folder_chk)

        self.out_dir_label = QLabel("No folder selected")
        self.out_dir_label.setObjectName("pathLabel")
        self.out_dir_btn = QPushButton("Choose output folder")
        self.out_dir_btn.clicked.connect(self.browse_output_dir)
        self.out_dir_btn.setEnabled(False)
        self.out_dir_label.setEnabled(False)
        out_form.addRow("Output folder:", self.out_dir_btn)
        out_form.addRow("", self.out_dir_label)

        layout.addWidget(out_group)

        # Column settings group
        col_group = QGroupBox("Column Settings (shared across all workflows)")
        col_form  = QFormLayout(col_group)
        col_form.setSpacing(10)

        self.id_input       = QLineEdit("county_fips")
        self.outcome_input  = QLineEdit("cases")
        self.exposure_input = QLineEdit("population")
        self.exclude_input  = QLineEdit("")
        self.exclude_input.setPlaceholderText("col1, col2, col3  (comma-separated)")

        col_form.addRow("ID column:", self.id_input)
        col_form.addRow("Outcome / dependent variable:", self.outcome_input)
        col_form.addRow("Exposure column (HBM only):", self.exposure_input)
        col_form.addRow("Exclude columns:", self.exclude_input)

        layout.addWidget(col_group)
        layout.addStretch()

        self.tabs.addTab(tab, "Settings")

    def _toggle_output_dir(self, checked):
        self.out_dir_btn.setEnabled(not checked)
        self.out_dir_label.setEnabled(not checked)

    # ── Tab 2: Regression ────────────────────────────────────────────────────

    def _build_regression_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)

        info = QLabel(
            "Regression models run on the data file selected in Settings.\n"
            "No spatial file is required. Excluded columns are taken from Settings."
        )
        info.setObjectName("infoLabel")
        info.setWordWrap(True)
        layout.addWidget(info)

        model_group = QGroupBox("Model Selection")
        model_form  = QFormLayout(model_group)
        model_form.setSpacing(10)

        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "Poisson Regression",
            "Negative Binomial Regression",
            "Stepwise Backwards Negative Binomial",
        ])
        model_form.addRow("Model type:", self.model_combo)

        self.reg_dep_input = QLineEdit()
        self.reg_dep_input.setPlaceholderText("Uses 'Outcome / dependent variable' from Settings if blank")
        model_form.addRow("Dependent variable (override):", self.reg_dep_input)

        self.reg_exclude_input = QLineEdit()
        self.reg_exclude_input.setPlaceholderText("Additional columns to exclude (comma-separated)")
        model_form.addRow("Additional exclusions:", self.reg_exclude_input)

        layout.addWidget(model_group)

        self.run_reg_btn = QPushButton("▶  Run Regression")
        self.run_reg_btn.setObjectName("runBtn")
        self.run_reg_btn.clicked.connect(self.run_regression)
        layout.addWidget(self.run_reg_btn)
        layout.addStretch()

        self.tabs.addTab(tab, "Regression")

    # ── Tab 3: HBM ───────────────────────────────────────────────────────────

    def _build_hbm_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)

        info = QLabel(
            "Hierarchical Bayesian Model with spatial random effects.\n"
            "Requires both a data file and a spatial file (set in Settings)."
        )
        info.setObjectName("infoLabel")
        info.setWordWrap(True)
        layout.addWidget(info)

        param_group = QGroupBox("Model Parameters")
        param_form  = QFormLayout(param_group)
        param_form.setSpacing(10)

        self.min_input = QLineEdit("0")
        self.max_input = QLineEdit("1000000")
        self.tune_input = QLineEdit("2000")
        self.draws_input = QLineEdit("2000")
        self.target_accept_input = QLineEdit("0.95")
        param_form.addRow("Min outcome value:", self.min_input)
        param_form.addRow("Max outcome value:", self.max_input)
        param_form.addRow("Tune steps:", self.tune_input)
        param_form.addRow("Posterior draws:", self.draws_input)
        param_form.addRow("Target accept:", self.target_accept_input)

        layout.addWidget(param_group)

        self.run_hbm_btn = QPushButton("▶  Run HBM Pipeline")
        self.run_hbm_btn.setObjectName("runBtn")
        self.run_hbm_btn.clicked.connect(self.run_hbm)
        layout.addWidget(self.run_hbm_btn)
        layout.addStretch()

        self.tabs.addTab(tab, "HBM")

    # ── File browsers ─────────────────────────────────────────────────────────

    def browse_data(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select data file", "", "Excel/CSV (*.xlsx *.csv)"
        )
        if path:
            self.data_path = path
            self.data_label.setText(path)

    def browse_geo(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select spatial file", "", "GeoJSON/Shapefile (*.geojson *.shp)"
        )
        if path:
            self.geo_path = path
            self.geo_label.setText(path)

    def browse_output_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Select output folder")
        if path:
            self.output_dir = path
            self.out_dir_label.setText(path)

    # ── Resolve output dir ────────────────────────────────────────────────────

    def _resolve_output_dir(self):
        if self.same_folder_chk.isChecked() or not self.output_dir:
            if self.data_path:
                return os.path.dirname(self.data_path)
            return os.getcwd()
        return self.output_dir

    # ── Terminal helpers ──────────────────────────────────────────────────────

    def _append_terminal(self, text):
        cursor = self.terminal.textCursor()
        if '\r' in text:
            cursor.movePosition(QTextCursor.End)
            cursor.movePosition(QTextCursor.StartOfLine, QTextCursor.KeepAnchor)
            cursor.removeSelectedText()
            cursor.insertText(text.replace('\r', '').strip())
        else:
            cursor.movePosition(QTextCursor.End)
            self.terminal.setTextCursor(cursor)
            self.terminal.insertPlainText(text)
        self.terminal.ensureCursorVisible()

    def _set_status(self, text):
        self.status_lbl.setText(text)

    # ── Run guards ────────────────────────────────────────────────────────────

    def _lock_ui(self):
        self.run_reg_btn.setEnabled(False)
        self.run_hbm_btn.setEnabled(False)
        self._set_status("Working…")

    def _unlock_ui(self):
        self.run_reg_btn.setEnabled(True)
        self.run_hbm_btn.setEnabled(True)

    def _on_done(self, success, message, verbose_out_path, err_path):
        if success:
            self._set_status("Done")
            self._append_terminal(f"\n[DONE] {message}\n")
        else:
            self._set_status("Error — see terminal and logs .err")
            self._append_terminal(f"\n[FAILED] {message}\n")
        self._unlock_ui()

    # ── Regression runner ────────────────────────────────────────────────────

    def run_regression(self):
        if not self.data_path:
            QMessageBox.warning(self, "Missing input", "Please select a data file in the Settings tab.")
            return

        out_dir = self._resolve_output_dir()
        artifact_stem, verbose_out, err_path, report_path = make_output_paths(
            out_dir, self.data_path, "REG"
        )

        dep_var = self.reg_dep_input.text().strip() or self.outcome_input.text().strip()
        base_exclude = [c.strip() for c in self.exclude_input.text().split(",") if c.strip()]
        extra_exclude = [c.strip() for c in self.reg_exclude_input.text().split(",") if c.strip()]
        excluded = list(set(base_exclude + extra_exclude))

        model_idx = self.model_combo.currentIndex()
        model_map = {0: run_poisson, 1: run_negative_binomial, 2: run_stepwise_nb}
        fn = model_map[model_idx]

        params = {
            "data_path":   self.data_path,
            "dep_var":     dep_var,
            "exclude_cols": excluded,
            "output_path": report_path,
            "log_path":    verbose_out,
        }

        self._lock_ui()
        self._append_terminal(
            f"\n{'─'*60}\n[START] Regression → report: {report_path} | verbose: {verbose_out}\n{'─'*60}\n"
        )

        self._worker = Worker(fn, params, err_path=err_path)
        self._worker.log_signal.connect(self._append_terminal)
        self._worker.done_signal.connect(
            lambda ok, msg: self._on_done(ok, msg, verbose_out, err_path)
        )
        self._worker.start()

    # ── HBM runner ───────────────────────────────────────────────────────────

    def run_hbm(self):
        if not self.data_path:
            QMessageBox.warning(self, "Missing input", "Please select a data file in the Settings tab.")
            return
        if not self.geo_path:
            QMessageBox.warning(self, "Missing input", "Please select a spatial file in the Settings tab.")
            return

        try:
            min_val = float(self.min_input.text())
            max_val = float(self.max_input.text())
            tune_steps = int(self.tune_input.text())
            draw_steps = int(self.draws_input.text())
            target_accept = float(self.target_accept_input.text())
        except ValueError:
            QMessageBox.warning(
                self,
                "Invalid input",
                "Min/Max must be numeric. Tune/Draws must be integers. Target accept must be numeric.",
            )
            return
        if tune_steps <= 0 or draw_steps <= 0:
            QMessageBox.warning(self, "Invalid input", "Tune steps and posterior draws must be > 0.")
            return
        if not (0.0 < target_accept < 1.0):
            QMessageBox.warning(self, "Invalid input", "Target accept must be between 0 and 1 (exclusive).")
            return

        out_dir = self._resolve_output_dir()
        artifact_stem, verbose_out, err_path, report_path = make_output_paths(
            out_dir, self.data_path, "HBM"
        )

        excluded = [c.strip() for c in self.exclude_input.text().split(",") if c.strip()]

        params = {
            "data_path":        self.data_path,
            "geo_path":         self.geo_path,
            "id_col":           self.id_input.text().strip(),
            "outcome_col":      self.outcome_input.text().strip(),
            "exposure_col":     self.exposure_input.text().strip(),
            "min_val":          min_val,
            "max_val":          max_val,
            "tune_steps":       tune_steps,
            "draw_steps":       draw_steps,
            "target_accept":    target_accept,
            "artifact_stem":    artifact_stem,
            "verbose_log_path": verbose_out,
            "report_path":      report_path,
            "exclude_cols":     excluded,
        }

        self._lock_ui()
        self._append_terminal(
            f"\n{'─'*60}\n[START] HBM → CSV/report in {out_dir} | verbose log: {verbose_out}\n{'─'*60}\n"
        )

        self._worker = Worker(run_pipeline, params, err_path=err_path)
        self._worker.log_signal.connect(self._append_terminal)
        self._worker.done_signal.connect(
            lambda ok, msg: self._on_done(ok, msg, verbose_out, err_path)
        )
        self._worker.start()

    # ── Styling ───────────────────────────────────────────────────────────────

    def _apply_style(self):
        self.setStyleSheet("""
            QWidget {
                background-color: #1e1e2e;
                color: #cdd6f4;
                font-family: 'Segoe UI', sans-serif;
                font-size: 10pt;
            }
            #titleBar {
                background-color: #181825;
                border-bottom: 1px solid #313244;
            }
            #titleLabel {
                font-size: 12pt;
                font-weight: bold;
                color: #89b4fa;
                letter-spacing: 1px;
            }
            QTabWidget::pane {
                border: 1px solid #313244;
                background: #1e1e2e;
            }
            QTabBar::tab {
                background: #181825;
                color: #a6adc8;
                padding: 8px 20px;
                border: 1px solid #313244;
                border-bottom: none;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background: #1e1e2e;
                color: #89b4fa;
                border-bottom: 2px solid #89b4fa;
            }
            QGroupBox {
                border: 1px solid #313244;
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 10px;
                font-weight: bold;
                color: #89b4fa;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
            }
            QPushButton {
                background-color: #313244;
                color: #cdd6f4;
                border: 1px solid #45475a;
                border-radius: 5px;
                padding: 6px 14px;
            }
            QPushButton:hover {
                background-color: #45475a;
                border-color: #89b4fa;
            }
            QPushButton:disabled {
                background-color: #1e1e2e;
                color: #585b70;
                border-color: #313244;
            }
            #runBtn {
                background-color: #1e66f5;
                color: #ffffff;
                font-weight: bold;
                font-size: 11pt;
                padding: 10px;
                border: none;
                border-radius: 6px;
            }
            #runBtn:hover  { background-color: #2979ff; }
            #runBtn:disabled { background-color: #313244; color: #585b70; }
            #smallBtn {
                padding: 3px 10px;
                font-size: 9pt;
            }
            QLineEdit, QComboBox {
                background-color: #181825;
                border: 1px solid #313244;
                border-radius: 4px;
                padding: 5px 8px;
                color: #cdd6f4;
            }
            QLineEdit:focus, QComboBox:focus {
                border-color: #89b4fa;
            }
            QComboBox::drop-down { border: none; }
            QCheckBox { spacing: 8px; }
            QCheckBox::indicator {
                width: 16px; height: 16px;
                border: 1px solid #45475a;
                border-radius: 3px;
                background: #181825;
            }
            QCheckBox::indicator:checked {
                background: #1e66f5;
                border-color: #1e66f5;
            }
            #terminal {
                background-color: #11111b;
                color: #a6e3a1;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 9pt;
                border: 1px solid #313244;
                border-radius: 4px;
            }
            #termGroup {
                margin: 0;
                border-top: 1px solid #313244;
                border-left: none;
                border-right: none;
                border-bottom: none;
                border-radius: 0;
                padding: 8px;
            }
            #statusBar {
                background-color: #181825;
                border-top: 1px solid #313244;
            }
            #statusLabel {
                color: #a6adc8;
                font-size: 9pt;
            }
            #pathLabel {
                color: #6c7086;
                font-size: 9pt;
                font-style: italic;
            }
            #infoLabel {
                color: #a6adc8;
                font-size: 9pt;
                background: #181825;
                border: 1px solid #313244;
                border-radius: 4px;
                padding: 8px;
            }
            #legalLabel {
                font-size: 8pt;
                color: #45475a;
                font-style: italic;
                padding: 4px;
            }
        """)


# ── Entry point ───────────────────────────────────────────────────────────────

def _set_windows_app_user_model_id():
    """
    Windows groups taskbar / Start pins by AppUserModelID. PyInstaller (especially
    one-file) often needs this set before QApplication so Pin to taskbar targets
    this app instead of python.exe or a generic bootstrap entry.
    """
    if sys.platform != "win32":
        return
    try:
        import ctypes
        # Fixed ID for this product; change only if you ship a distinct app build.
        appid = "HjBM.HierarchicalBayesianModeler.GUI.1"
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(appid)
    except Exception:
        pass


if __name__ == "__main__":
    multiprocessing.freeze_support()
    _set_windows_app_user_model_id()
    if (not getattr(sys, "frozen", False)) and (not _check_cxx_compiler()):
        app = QApplication(sys.argv)
        QMessageBox.warning(
            None,
            "Missing C++ Compiler",
            "No C++ compiler was detected on your system.\n\n"
            "The HBM pipeline requires a C++ compiler to run efficiently.\n\n"
            "Please install Microsoft Visual C++ Build Tools:\n"
            "https://visualstudio.microsoft.com/visual-cpp-build-tools/\n\n"
            "Select 'Desktop development with C++' during installation.\n"
            "Then restart HjBM.",
        )
        sys.exit(1)
    app = QApplication(sys.argv)
    app.setApplicationName("HjBM")
    app.setOrganizationName("HjBM")
    if hasattr(app, "setApplicationDisplayName"):
        app.setApplicationDisplayName("HjBM")

    # App-level icon (taskbar + alt-tab)
    icon_path = os.path.join(os.path.dirname(__file__), "HjBM.ico")
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))

    # Splash
    splash_img = os.path.join(os.path.dirname(__file__), "splash.png")
    splash = SplashScreen(splash_img)
    splash.show()
    splash.start_progress()
    app.processEvents()

    # Simulate load time (imports are heavy)
    for _ in range(60):
        time.sleep(0.10)
        app.processEvents()

    window = PHBayesGUI()
    splash.finish_progress(window)
    window.show()
    sys.exit(app.exec())