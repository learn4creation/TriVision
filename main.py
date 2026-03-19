"""
TriVision — Main Application
The unified OpenCV + CVIPtools2 + scikit-image workbench.

python main.py
"""

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))


import sys
import os
# Ensure the TriVision root is on the path so all subpackages resolve correctly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import time
import numpy as np
import cv2
from pathlib import Path
from typing import Optional, Callable, Any

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QSplitter, QTabWidget, QTreeWidget, QTreeWidgetItem, QLabel,
    QPushButton, QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox,
    QGroupBox, QFormLayout, QFileDialog, QMessageBox, QScrollArea,
    QStatusBar, QProgressBar, QTextEdit, QLineEdit, QListWidget,
    QListWidgetItem, QSizePolicy, QFrame, QDialog, QDialogButtonBox,
    QMenu, QAbstractItemView
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize
from PyQt6.QtGui import QImage, QPixmap, QColor, QPalette, QFont, QPainter, QPen, QIcon, QAction

# ── TriVision imports ──────────────────────────────────────────────────────────
import core.algorithms_opencv as _ocv
import core.algorithms_skimage as _ski
import core.algorithms_cvip_fusion as _cvf
from core.registry import REGISTRY, Lib, ReturnType
from pipeline.engine import Pipeline
from plugins.sdk import load_plugins


def _bootstrap():
    """Register all built-in algorithms and load plugins."""
    _ocv.register_all()
    _ski.register_all()
    _cvf.register_all()
    loaded = load_plugins("plugins")
    print(f"TriVision: {len(REGISTRY)} algorithms registered, {len(loaded)} plugins loaded.")

_bootstrap()


# ─── Library badge colors ─────────────────────────────────────────────────────
LIB_COLORS = {
    Lib.OPENCV:    "#1a4a8a",
    Lib.CVIP:      "#1a6a4a",
    Lib.SKIMAGE:   "#5a2a8a",
    Lib.TRIVISION: "#8a3a1a",
}
LIB_LABELS = {
    Lib.OPENCV:    "CV",
    Lib.CVIP:      "CL",
    Lib.SKIMAGE:   "SK",
    Lib.TRIVISION: "TV",
}


# ═══════════════════════════════════════════════════════════════════════════════
# Image Viewer
# ═══════════════════════════════════════════════════════════════════════════════

class ImageViewer(QLabel):
    def __init__(self, hint="", parent=None):
        super().__init__(parent)
        self._img: Optional[np.ndarray] = None
        self._hint = hint
        self.setMinimumSize(320, 240)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background:#060810; border:1px solid #1a1f2e;")
        self._placeholder()

    def _placeholder(self):
        self.setText(f"<span style='color:#252a3a;font-family:monospace;font-size:11px'>{self._hint}</span>")

    def set_image(self, img: Optional[np.ndarray]):
        self._img = img.copy() if img is not None else None
        if img is None:
            self._placeholder(); return
        self._render()

    def get_image(self): return self._img

    def _render(self):
        img = self._img
        if img is None: return
        if len(img.shape) == 2:
            rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        else:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        q = QImage(rgb.data, w, h, 3*w, QImage.Format.Format_RGB888)
        px = QPixmap.fromImage(q).scaled(self.size(),
             Qt.AspectRatioMode.KeepAspectRatio,
             Qt.TransformationMode.SmoothTransformation)
        self.setPixmap(px)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self._img is not None: self._render()


# ═══════════════════════════════════════════════════════════════════════════════
# Histogram Widget
# ═══════════════════════════════════════════════════════════════════════════════

class HistogramWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(60); self.setMaximumHeight(75)
        self._ch = []; self.setStyleSheet("background:#030508;")

    def set_image(self, img):
        if img is None: self._ch=[]; self.update(); return
        self._ch=[]
        if len(img.shape)==2:
            self._ch=[((160,160,180), cv2.calcHist([img],[0],None,[64],[0,256]).flatten())]
        else:
            for i,c in enumerate([(80,80,220),(80,200,80),(220,80,80)]):
                self._ch.append((c,cv2.calcHist([img],[i],None,[64],[0,256]).flatten()))
        self.update()

    def paintEvent(self,e):
        p=QPainter(self); p.fillRect(self.rect(),QColor(3,5,8))
        if not self._ch: return
        w,h=self.width(),self.height()
        mx=max(ch[1].max() for ch in self._ch) or 1
        for col,hist in self._ch:
            r,g,b=col; p.setPen(QPen(QColor(r,g,b,130),1))
            n=len(hist); bw=(w-4)/n
            for i,v in enumerate(hist):
                bh=int((v/mx)*(h-4)); x=int(2+i*bw)
                p.drawLine(x,h-2,x,h-2-bh)
        p.end()


# ═══════════════════════════════════════════════════════════════════════════════
# Processing Worker
# ═══════════════════════════════════════════════════════════════════════════════

class Worker(QThread):
    done = pyqtSignal(object, float, str)  # result, elapsed_ms, message

    def __init__(self, fn, img, kwargs):
        super().__init__()
        self._fn=fn; self._img=img; self._kw=kwargs

    def run(self):
        t0=time.perf_counter()
        try:
            result=self._fn(self._img,**self._kw)
            ms=(time.perf_counter()-t0)*1000
            if isinstance(result,tuple) and len(result)==3:
                img,ratio,bpp=result
                self.done.emit(img,ms,f"Ratio {ratio:.2f}:1  •  {bpp:.2f} bpp")
            elif isinstance(result,dict):
                self.done.emit(result,ms,"Features extracted")
            else:
                self.done.emit(result,ms,"Done")
        except Exception as ex:
            ms=(time.perf_counter()-t0)*1000
            self.done.emit(None,ms,f"Error: {ex}")


# ═══════════════════════════════════════════════════════════════════════════════
# Dynamic Parameter Panel
# ═══════════════════════════════════════════════════════════════════════════════

class ParamPanel(QWidget):
    changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._form = QFormLayout(self)
        self._form.setSpacing(5)
        self._getters: dict[str, Callable] = {}

    def load_spec(self, spec):
        # Clear
        while self._form.count():
            item = self._form.takeAt(0)
            if item.widget(): item.widget().deleteLater()
        self._getters.clear()

        if spec is None:
            return

        for p in spec.params:
            if p.kind == "int":
                w = QSpinBox()
                w.setRange(int(p.lo), int(p.hi))
                w.setValue(int(p.default))
                w.setSingleStep(int(p.step))
                w.valueChanged.connect(self.changed)
                self._getters[p.name] = w.value
            elif p.kind == "float":
                w = QDoubleSpinBox()
                w.setRange(float(p.lo), float(p.hi))
                w.setValue(float(p.default))
                w.setSingleStep(float(p.step))
                dec = max(2, len(str(p.step).rstrip("0").split(".")[-1]))
                w.setDecimals(dec)
                w.valueChanged.connect(self.changed)
                self._getters[p.name] = w.value
            elif p.kind == "bool":
                w = QCheckBox()
                w.setChecked(bool(p.default))
                w.toggled.connect(self.changed)
                self._getters[p.name] = w.isChecked
            elif p.kind == "choice":
                w = QComboBox()
                w.addItems(p.choices)
                if p.default in p.choices:
                    w.setCurrentText(p.default)
                w.currentIndexChanged.connect(self.changed)
                self._getters[p.name] = w.currentText
            else:
                continue
            self._form.addRow(f"{p.label}:", w)

    def get_kwargs(self) -> dict:
        return {k: g() for k, g in self._getters.items()}


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline Panel (right side of pipeline tab)
# ═══════════════════════════════════════════════════════════════════════════════

class PipelinePanel(QWidget):
    run_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)

        hdr = QHBoxLayout()
        hdr.addWidget(QLabel("Pipeline"))
        self.clear_btn = QPushButton("✕ Clear")
        self.clear_btn.setFixedWidth(70)
        self.save_btn = QPushButton("💾")
        self.save_btn.setFixedWidth(32)
        self.load_btn = QPushButton("📂")
        self.load_btn.setFixedWidth(32)
        for b in [self.clear_btn, self.save_btn, self.load_btn]:
            b.setStyleSheet("QPushButton{background:#111;color:#aaa;border:1px solid #222;padding:2px 4px;font-size:11px;}"
                            "QPushButton:hover{background:#1a2030;}")
        hdr.addWidget(self.clear_btn)
        hdr.addWidget(self.save_btn)
        hdr.addWidget(self.load_btn)
        layout.addLayout(hdr)

        self.node_list = QListWidget()
        self.node_list.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.node_list.setStyleSheet("""
            QListWidget { background:#06080e; border:1px solid #1a1f2e; font-size:11px; }
            QListWidget::item { padding:4px 6px; border-bottom:1px solid #0e1018; }
            QListWidget::item:selected { background:#1a2a50; color:#8af; }
            QListWidget::item:hover { background:#0e1828; }
        """)
        layout.addWidget(self.node_list)

        preset_label = QLabel("Presets:")
        preset_label.setStyleSheet("color:#555; font-size:11px; margin-top:6px;")
        layout.addWidget(preset_label)

        for name, fn in [
            ("Edge Detection", Pipeline.edge_detection_pipeline),
            ("Denoise + Enhance", Pipeline.denoise_and_enhance),
            ("Segmentation", Pipeline.segmentation_pipeline),
            ("Feature Extraction", Pipeline.feature_extraction_pipeline),
        ]:
            b = QPushButton(name)
            b.setStyleSheet("QPushButton{background:#0a0f1a;color:#6af;border:1px solid #1a2a4a;"
                            "font-size:11px;padding:4px;text-align:left;padding-left:8px;}"
                            "QPushButton:hover{background:#101828;}")
            b.clicked.connect(lambda checked, f=fn: self.run_requested.emit() or self._load_preset(f))
            layout.addWidget(b)

        self.run_btn = QPushButton("▶  Run Pipeline")
        self.run_btn.setMinimumHeight(34)
        self.run_btn.setStyleSheet(
            "QPushButton{background:#1a3a6a;color:#c0d8ff;font-weight:bold;border-radius:3px;}"
            "QPushButton:hover{background:#2050a0;}")
        layout.addWidget(self.run_btn)

        self._pipeline = Pipeline()
        self._preset_cb: Optional[Callable] = None

    def set_preset_callback(self, cb): self._preset_cb = cb

    def _load_preset(self, fn):
        if self._preset_cb: self._preset_cb(fn())

    def set_pipeline(self, p: Pipeline):
        self._pipeline = p
        self._refresh_list()

    def get_pipeline(self) -> Pipeline: return self._pipeline

    def add_algo(self, algo_key: str):
        spec = REGISTRY.get(algo_key)
        if spec is None: return
        # Find last node to connect to
        upstream = self._pipeline.nodes[-1].node_id if self._pipeline.nodes else None
        node = self._pipeline.add_node(algo_key, upstream_id=upstream)
        col = LIB_COLORS.get(spec.lib, "#444")
        tag = LIB_LABELS.get(spec.lib, "?")
        item = QListWidgetItem(f"[{tag}] {spec.label}")
        item.setData(Qt.ItemDataRole.UserRole, node.node_id)
        item.setForeground(QColor(col))
        self.node_list.addItem(item)

    def remove_selected(self):
        item = self.node_list.currentItem()
        if not item: return
        nid = item.data(Qt.ItemDataRole.UserRole)
        self._pipeline.remove_node(nid)
        self.node_list.takeItem(self.node_list.row(item))

    def clear(self):
        self._pipeline.clear()
        self.node_list.clear()

    def _refresh_list(self):
        self.node_list.clear()
        for node in self._pipeline.nodes:
            spec = REGISTRY.get(node.algo_key)
            if spec is None: continue
            col = LIB_COLORS.get(spec.lib, "#444")
            tag = LIB_LABELS.get(spec.lib, "?")
            item = QListWidgetItem(f"[{tag}] {spec.label}")
            item.setData(Qt.ItemDataRole.UserRole, node.node_id)
            item.setForeground(QColor(col))
            self.node_list.addItem(item)


# ═══════════════════════════════════════════════════════════════════════════════
# Feature Display Widget
# ═══════════════════════════════════════════════════════════════════════════════

class FeatureDisplay(QTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setMaximumHeight(180)
        self.setStyleSheet(
            "QTextEdit{background:#030508;color:#60c060;font-family:monospace;"
            "font-size:11px;border:none;border-top:1px solid #1a1f2e;}")
        self.setPlaceholderText("Feature extraction results appear here…")

    def show_features(self, d: dict):
        lines = []
        for k, v in d.items():
            if isinstance(v, float):
                lines.append(f"  {k:<32} {v:.6f}")
            else:
                lines.append(f"  {k:<32} {v}")
        self.setText("\n".join(lines))


# ═══════════════════════════════════════════════════════════════════════════════
# Batch Tab
# ═══════════════════════════════════════════════════════════════════════════════

class BatchTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # Input/Output dirs
        for attr, label in [("_in_edit","Input folder:"),("_out_edit","Output folder:")]:
            row = QHBoxLayout()
            lbl = QLabel(label); lbl.setFixedWidth(90)
            lbl.setStyleSheet("color:#888;font-size:11px;")
            edit = QLineEdit(); edit.setStyleSheet(
                "QLineEdit{background:#06080e;color:#aaa;border:1px solid #1a1f2e;padding:3px;font-size:11px;}")
            setattr(self, attr, edit)
            btn = QPushButton("…")
            btn.setFixedWidth(28)
            btn.setStyleSheet("QPushButton{background:#111;color:#888;border:1px solid #222;}")
            a = attr
            btn.clicked.connect(lambda _, a=a: self._browse(a))
            row.addWidget(lbl); row.addWidget(edit); row.addWidget(btn)
            layout.addLayout(row)

        # Algorithm selection
        algo_row = QHBoxLayout()
        algo_lbl = QLabel("Algorithm:"); algo_lbl.setFixedWidth(90)
        algo_lbl.setStyleSheet("color:#888;font-size:11px;")
        self._algo_combo = QComboBox()
        self._algo_combo.setStyleSheet(
            "QComboBox{background:#06080e;color:#bbb;border:1px solid #1a1f2e;padding:3px;font-size:11px;}")
        for spec in sorted(REGISTRY.all(), key=lambda s: s.label):
            self._algo_combo.addItem(f"[{LIB_LABELS.get(spec.lib,'?')}] {spec.label}", spec.key)
        algo_row.addWidget(algo_lbl); algo_row.addWidget(self._algo_combo)
        layout.addLayout(algo_row)

        # Options
        opts = QHBoxLayout()
        self._save_check = QCheckBox("Save outputs")
        self._save_check.setChecked(True)
        self._save_check.setStyleSheet("color:#888;font-size:11px;")
        opts.addWidget(self._save_check)
        opts.addStretch()
        layout.addLayout(opts)

        # Run button
        self._run_btn = QPushButton("▶  Run Batch")
        self._run_btn.setMinimumHeight(34)
        self._run_btn.setStyleSheet(
            "QPushButton{background:#1a3a6a;color:#c0d8ff;font-weight:bold;border-radius:3px;}"
            "QPushButton:hover{background:#2050a0;}")
        self._run_btn.clicked.connect(self._run_batch)
        layout.addWidget(self._run_btn)

        # Progress
        self._progress = QProgressBar()
        self._progress.setStyleSheet(
            "QProgressBar{background:#060810;border:1px solid #1a1f2e;color:#aaa;font-size:10px;}"
            "QProgressBar::chunk{background:#1a4a8a;}")
        self._progress.setMaximum(100); self._progress.setValue(0)
        layout.addWidget(self._progress)

        # Results
        self._results = QTextEdit()
        self._results.setReadOnly(True)
        self._results.setStyleSheet(
            "QTextEdit{background:#030508;color:#6af;font-family:monospace;font-size:10px;border:none;}")
        layout.addWidget(self._results, 1)

        # Export
        export_row = QHBoxLayout()
        self._export_csv = QPushButton("Export CSV")
        self._export_json = QPushButton("Export JSON")
        for b in [self._export_csv, self._export_json]:
            b.setStyleSheet("QPushButton{background:#111;color:#888;border:1px solid #222;font-size:11px;padding:4px;}"
                            "QPushButton:hover{background:#1a2030;color:#aaa;}")
            export_row.addWidget(b)
        self._export_csv.clicked.connect(self._export_to_csv)
        self._export_json.clicked.connect(self._export_to_json)
        layout.addLayout(export_row)

        self._report = None
        self._worker_thread = None

    def _browse(self, attr):
        d = QFileDialog.getExistingDirectory(self, "Select folder")
        if d: getattr(self, attr).setText(d)

    def _run_batch(self):
        from batch.processor import BatchProcessor
        in_dir = self._in_edit.text().strip()
        out_dir = self._out_edit.text().strip()
        if not in_dir:
            QMessageBox.warning(self,"Batch","Please select an input folder."); return
        if not out_dir and self._save_check.isChecked():
            QMessageBox.warning(self,"Batch","Please select an output folder."); return

        key = self._algo_combo.currentData()
        self._run_btn.setEnabled(False)
        self._progress.setValue(0)
        self._results.clear()

        processor = BatchProcessor(algo_key=key)
        self._report = None

        def progress_cb(cur, total, fname):
            pct = int(cur/max(total,1)*100)
            self._progress.setValue(pct)
            self._results.append(f"  [{cur}/{total}] {fname}")

        import threading
        def do_batch():
            report = processor.run(in_dir, out_dir or ".",
                                    save_outputs=self._save_check.isChecked(),
                                    progress_cb=progress_cb)
            self._report = report
            summary = report.summary()
            lines = ["\n─── Batch Complete ───"]
            for k,v in summary.items(): lines.append(f"  {k}: {v}")
            self._results.append("\n".join(lines))
            self._run_btn.setEnabled(True)

        t = threading.Thread(target=do_batch, daemon=True)
        t.start()

    def _export_to_csv(self):
        if not self._report:
            QMessageBox.information(self,"Export","Run a batch first."); return
        fn,_ = QFileDialog.getSaveFileName(self,"Save CSV","batch_report.csv","CSV (*.csv)")
        if fn: self._report.to_csv(fn)

    def _export_to_json(self):
        if not self._report:
            QMessageBox.information(self,"Export","Run a batch first."); return
        fn,_ = QFileDialog.getSaveFileName(self,"Save JSON","batch_report.json","JSON (*.json)")
        if fn: self._report.to_json(fn)


# ═══════════════════════════════════════════════════════════════════════════════
# Main Window
# ═══════════════════════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TriVision — Unified Image Science Workbench")
        self.setMinimumSize(1500, 900)
        self._input_img: Optional[np.ndarray] = None
        self._current_spec = None
        self._worker: Optional[Worker] = None
        self._auto = True
        self._pipeline = Pipeline()
        self._setup_ui()
        self._setup_menu()
        self._create_default_image()

    # ─── UI ──────────────────────────────────────────────────────────

    def _setup_ui(self):
        root = QWidget(); self.setCentralWidget(root)
        rl = QHBoxLayout(root); rl.setContentsMargins(4,4,4,4); rl.setSpacing(4)

        outer = QSplitter(Qt.Orientation.Horizontal)

        # ── LEFT: Algorithm Tree ──────────────────────────────────────
        left = QWidget(); ll = QVBoxLayout(left); ll.setContentsMargins(0,0,0,0)

        # Search bar
        self._search = QLineEdit()
        self._search.setPlaceholderText("Search algorithms…")
        self._search.setStyleSheet(
            "QLineEdit{background:#06080e;color:#aaa;border:1px solid #1a1f2e;"
            "padding:4px 6px;font-size:11px;border-radius:3px;}")
        self._search.textChanged.connect(self._filter_tree)
        ll.addWidget(self._search)

        # Library filter
        lib_row = QHBoxLayout()
        self._lib_btns: dict[str, QPushButton] = {}
        for lib, label, col in [("ALL","All","#444"),("OpenCV","CV","#1a4a8a"),
                                  ("CVIPtools2","CL","#1a6a4a"),
                                  ("scikit-image","SK","#5a2a8a"),
                                  ("TriVision","TV","#8a3a1a")]:
            b = QPushButton(label)
            b.setFixedHeight(22)
            b.setStyleSheet(f"QPushButton{{background:{col};color:#ddf;border:none;font-size:10px;font-weight:bold;}}"
                            f"QPushButton:hover{{background:{col};opacity:0.8;}}")
            b.clicked.connect(lambda _, l=lib: self._filter_by_lib(l))
            lib_row.addWidget(b)
            self._lib_btns[lib] = b
        ll.addLayout(lib_row)

        self._tree = QTreeWidget()
        self._tree.setHeaderHidden(True)
        self._tree.setStyleSheet("""
            QTreeWidget { background:#04060c; border:none; font-size:11px; color:#bbb; }
            QTreeWidget::item { padding:2px 4px; }
            QTreeWidget::item:hover { background:#0d1220; }
            QTreeWidget::item:selected { background:#162240; color:#7af; }
            QTreeWidget::branch { background:#04060c; }
        """)
        self._tree.itemClicked.connect(self._on_tree_click)
        self._tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._tree.customContextMenuRequested.connect(self._tree_context_menu)
        self._build_tree()
        ll.addWidget(self._tree, 1)

        # Algorithm count
        count_lbl = QLabel(f"{len(REGISTRY)} algorithms")
        count_lbl.setStyleSheet("color:#333;font-size:10px;padding:2px;")
        ll.addWidget(count_lbl)

        outer.addWidget(left)

        # ── CENTRE: Image panels + tabs ───────────────────────────────
        centre = QWidget(); cl = QVBoxLayout(centre); cl.setSpacing(4)

        # Viewers row
        viewers = QHBoxLayout()
        for attr, hint, hist_attr in [
            ("_input_viewer","INPUT","_input_hist"),
            ("_output_viewer","OUTPUT","_output_hist")
        ]:
            grp = QGroupBox(hint)
            grp.setStyleSheet(
                "QGroupBox{font-size:10px;font-weight:bold;color:#3a5080;border:1px solid #101828;margin-top:8px;}"
                "QGroupBox::title{left:6px;}")
            gl = QVBoxLayout(grp)
            v = ImageViewer(hint.lower())
            setattr(self, attr, v)
            gl.addWidget(v)
            h = HistogramWidget()
            setattr(self, hist_attr, h)
            gl.addWidget(h)
            viewers.addWidget(grp)
        cl.addLayout(viewers, 4)

        # Controls row
        ctrl = QHBoxLayout()
        self._process_btn = QPushButton("▶  Process")
        self._process_btn.setMinimumHeight(36)
        self._process_btn.setStyleSheet(
            "QPushButton{background:#102060;color:#a0c0ff;font-weight:bold;font-size:13px;border-radius:3px;letter-spacing:1px;}"
            "QPushButton:hover{background:#1a3a90;}"
            "QPushButton:disabled{background:#0a0e18;color:#252a40;}")
        self._process_btn.clicked.connect(self._process)
        self._auto_check = QCheckBox("Auto")
        self._auto_check.setChecked(True)
        self._auto_check.setStyleSheet("color:#555;font-size:11px;")
        self._auto_check.toggled.connect(lambda v: setattr(self,'_auto',v))
        self._copy_btn = QPushButton("→ Use as Input")
        self._diff_btn = QPushButton("⊕ Diff")
        self._ab_btn = QPushButton("A/B Compare")
        self._reset_btn = QPushButton("↺ Reset")
        for b in [self._copy_btn, self._diff_btn, self._ab_btn, self._reset_btn]:
            b.setStyleSheet(
                "QPushButton{background:#06080e;color:#6080a0;border:1px solid #1a1f2e;padding:4px 10px;font-size:11px;}"
                "QPushButton:hover{background:#101828;color:#80a0d0;}")
        self._copy_btn.clicked.connect(self._copy_to_input)
        self._diff_btn.clicked.connect(self._show_diff)
        self._ab_btn.clicked.connect(self._show_ab)
        self._reset_btn.clicked.connect(self._create_default_image)
        for w in [self._process_btn, self._auto_check, self._copy_btn,
                   self._diff_btn, self._ab_btn, self._reset_btn]:
            ctrl.addWidget(w)
        cl.addLayout(ctrl)

        # Feature display
        self._feat_display = FeatureDisplay()
        cl.addWidget(self._feat_display)

        # Metrics row
        self._metric_lbl = QLabel("PSNR: —  RMSE: —  SSIM: —  Sharpness: —")
        self._metric_lbl.setStyleSheet("color:#3a5a3a;font-family:monospace;font-size:11px;padding:3px 4px;"
                                        "background:#030508;border-top:1px solid #0e1018;")
        cl.addWidget(self._metric_lbl)

        outer.addWidget(centre)

        # ── RIGHT: Tabs (Params / Pipeline / Batch) ───────────────────
        right_tabs = QTabWidget()
        right_tabs.setMinimumWidth(300)
        right_tabs.setStyleSheet("""
            QTabWidget::pane{border:1px solid #1a1f2e;background:#04060c;}
            QTabBar::tab{background:#06080e;color:#555;border:1px solid #1a1f2e;
                         padding:5px 12px;font-size:11px;}
            QTabBar::tab:selected{background:#0d1828;color:#8af;border-bottom:none;}
            QTabBar::tab:hover{color:#6a8aaa;}
        """)

        # Parameters tab
        param_widget = QWidget()
        pw = QVBoxLayout(param_widget); pw.setContentsMargins(4,4,4,4)
        self._algo_label = QLabel("Select an algorithm")
        self._algo_label.setStyleSheet("font-weight:bold;font-size:12px;color:#5a8aff;padding:4px;")
        self._lib_badge = QLabel("")
        self._lib_badge.setStyleSheet("font-size:10px;color:#666;padding:2px 4px;")
        self._algo_desc = QLabel("")
        self._algo_desc.setWordWrap(True)
        self._algo_desc.setStyleSheet("color:#444;font-size:10px;padding:2px 4px;")
        pw.addWidget(self._algo_label)
        pw.addWidget(self._lib_badge)
        pw.addWidget(self._algo_desc)
        self._param_panel = ParamPanel()
        self._param_panel.changed.connect(self._on_param_changed)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea{border:none;background:#04060c;}")
        scroll.setWidget(self._param_panel)
        pw.addWidget(scroll, 1)
        # I/O buttons
        io_grp = QGroupBox("Image I/O")
        io_grp.setStyleSheet("QGroupBox{font-size:10px;color:#3a5080;border:1px solid #101828;margin-top:8px;}"
                              "QGroupBox::title{left:6px;}")
        ig = QVBoxLayout(io_grp)
        for lbl, slot in [("📂 Load Image",self._load_image),
                           ("💾 Save Output",self._save_image),
                           ("📷 Webcam",self._capture_webcam),
                           ("📊 Extract All Features",self._extract_features)]:
            b = QPushButton(lbl)
            b.setStyleSheet("QPushButton{background:#06080e;color:#6080a0;border:1px solid #1a1f2e;"
                            "padding:4px;font-size:11px;text-align:left;padding-left:6px;}"
                            "QPushButton:hover{background:#0d1828;color:#80a0d0;}")
            b.clicked.connect(slot)
            ig.addWidget(b)
        pw.addWidget(io_grp)
        right_tabs.addTab(param_widget, "Parameters")

        # Pipeline tab
        pipeline_widget = QWidget()
        plw = QVBoxLayout(pipeline_widget); plw.setContentsMargins(4,4,4,4)
        self._pipeline_panel = PipelinePanel()
        self._pipeline_panel.set_pipeline(self._pipeline)
        self._pipeline_panel.run_btn.clicked.connect(self._run_pipeline)
        self._pipeline_panel.clear_btn.clicked.connect(self._pipeline_panel.clear)
        self._pipeline_panel.save_btn.clicked.connect(self._save_pipeline)
        self._pipeline_panel.load_btn.clicked.connect(self._load_pipeline)
        self._pipeline_panel.set_preset_callback(self._load_preset_pipeline)
        plw.addWidget(self._pipeline_panel)
        right_tabs.addTab(pipeline_widget, "Pipeline")

        # Batch tab
        self._batch_tab = BatchTab()
        right_tabs.addTab(self._batch_tab, "Batch")

        outer.addWidget(right_tabs)
        outer.setSizes([230, 1000, 310])
        rl.addWidget(outer)

        # Status bar
        sb = QStatusBar()
        sb.setStyleSheet("QStatusBar{color:#3a5060;font-size:10px;background:#030508;}"
                          "QStatusBar::item{border:none;}")
        self.setStatusBar(sb)
        self._prog = QProgressBar()
        self._prog.setMaximumWidth(200); self._prog.setMaximumHeight(14)
        self._prog.setRange(0,0); self._prog.setVisible(False)
        self._prog.setStyleSheet("QProgressBar{background:#060810;border:1px solid #1a2030;}"
                                  "QProgressBar::chunk{background:#1a4080;}")
        sb.addPermanentWidget(self._prog)
        self.statusBar().showMessage("TriVision ready — load an image to begin")

    def _build_tree(self, filter_text="", filter_lib="ALL"):
        self._tree.clear()
        tree_data = REGISTRY.by_category()
        for cat, subcats in sorted(tree_data.items()):
            cat_item = QTreeWidgetItem([cat])
            cat_item.setFont(0, QFont("", 11, QFont.Weight.Bold))
            cat_item.setForeground(0, QColor("#3a6080"))
            has_visible = False
            for subcat, specs in sorted(subcats.items()):
                sub_item = QTreeWidgetItem([subcat])
                sub_item.setForeground(0, QColor("#2a4060"))
                sub_item.setFont(0, QFont("", 10))
                has_sub = False
                for spec in sorted(specs, key=lambda s: s.label):
                    # Apply filters
                    if filter_text and filter_text.lower() not in spec.label.lower() \
                       and filter_text.lower() not in spec.description.lower():
                        continue
                    if filter_lib != "ALL" and spec.lib.value != filter_lib:
                        continue
                    tag = LIB_LABELS.get(spec.lib, "?")
                    col = LIB_COLORS.get(spec.lib, "#444")
                    leaf = QTreeWidgetItem([f"[{tag}] {spec.label}"])
                    leaf.setData(0, Qt.ItemDataRole.UserRole, spec.key)
                    leaf.setForeground(0, QColor(col))
                    sub_item.addChild(leaf)
                    has_sub = True
                if has_sub:
                    cat_item.addChild(sub_item)
                    has_visible = True
            if has_visible:
                self._tree.addTopLevelItem(cat_item)
        if filter_text or filter_lib != "ALL":
            self._tree.expandAll()
        else:
            for i in range(self._tree.topLevelItemCount()):
                self._tree.topLevelItem(i).setExpanded(True)

    def _filter_tree(self, text):
        self._build_tree(filter_text=text)

    def _filter_by_lib(self, lib):
        self._build_tree(filter_lib=lib)

    def _tree_context_menu(self, pos):
        item = self._tree.itemAt(pos)
        if not item: return
        key = item.data(0, Qt.ItemDataRole.UserRole)
        if not key: return
        menu = QMenu(self)
        menu.setStyleSheet("QMenu{background:#06080e;color:#aaa;border:1px solid #1a1f2e;font-size:11px;}"
                           "QMenu::item:selected{background:#1a2840;}")
        a1 = menu.addAction("▶ Process now")
        a2 = menu.addAction("➕ Add to pipeline")
        action = menu.exec(self._tree.viewport().mapToGlobal(pos))
        if action == a1:
            self._activate_algo(key); self._process()
        elif action == a2:
            self._pipeline_panel.add_algo(key)

    def _setup_menu(self):
        mb = self.menuBar()
        mb.setStyleSheet("QMenuBar{background:#030508;color:#6080a0;font-size:11px;}"
                          "QMenuBar::item:selected{background:#0d1828;}")
        fm = mb.addMenu("File")
        for lbl, sc, fn in [("Open Image…","Ctrl+O",self._load_image),
                              ("Save Output…","Ctrl+S",self._save_image),
                              (None,None,None),("Exit","Ctrl+Q",self.close)]:
            if lbl is None: fm.addSeparator(); continue
            a = QAction(lbl,self); a.setShortcut(sc); a.triggered.connect(fn); fm.addAction(a)

        vm = mb.addMenu("View")
        a = QAction("Extract All Features",self); a.triggered.connect(self._extract_features); vm.addAction(a)
        a2 = QAction("A/B Compare",self); a2.triggered.connect(self._show_ab); vm.addAction(a2)

        pm = mb.addMenu("Pipeline")
        for lbl, fn in [("Run Pipeline",self._run_pipeline),
                          ("Save Pipeline…",self._save_pipeline),
                          ("Load Pipeline…",self._load_pipeline)]:
            a = QAction(lbl,self); a.triggered.connect(fn); pm.addAction(a)

        hm = mb.addMenu("Help")
        a = QAction("About TriVision",self); a.triggered.connect(self._about); hm.addAction(a)

    # ─── Tree → activate algorithm ────────────────────────────────────

    def _on_tree_click(self, item, col):
        key = item.data(0, Qt.ItemDataRole.UserRole)
        if not key: return
        self._activate_algo(key)
        if self._auto and self._input_img is not None:
            self._process()

    def _activate_algo(self, key: str):
        spec = REGISTRY.get(key)
        if spec is None: return
        self._current_spec = spec
        self._algo_label.setText(spec.label)
        col = LIB_COLORS.get(spec.lib, "#444")
        self._lib_badge.setText(f"Library: {spec.lib.value}")
        self._lib_badge.setStyleSheet(f"font-size:10px;color:{col};padding:2px 4px;")
        self._algo_desc.setText(spec.description)
        self._param_panel.load_spec(spec)
        self._param_panel.changed.connect(self._on_param_changed)

    def _on_param_changed(self):
        if self._auto: self._process()

    # ─── Processing ───────────────────────────────────────────────────

    def _process(self):
        if self._current_spec is None or self._input_img is None: return
        if self._worker and self._worker.isRunning(): return
        kwargs = self._param_panel.get_kwargs()
        self._process_btn.setEnabled(False)
        self._prog.setVisible(True)
        self.statusBar().showMessage(f"Processing: {self._current_spec.label}…")
        self._worker = Worker(self._current_spec.fn, self._input_img, kwargs)
        self._worker.done.connect(self._on_done)
        self._worker.start()

    def _on_done(self, result, elapsed_ms, message):
        self._process_btn.setEnabled(True)
        self._prog.setVisible(False)
        if result is None:
            self.statusBar().showMessage(message); return

        if isinstance(result, dict):
            self._feat_display.show_features(result)
            self.statusBar().showMessage(f"{message}  •  {elapsed_ms:.0f}ms")
            return

        self._output_viewer.set_image(result)
        self._output_hist.set_image(result)
        self._compute_metrics(result)
        h,w = result.shape[:2]
        ch = result.shape[2] if len(result.shape)==3 else 1
        self.statusBar().showMessage(
            f"{self._current_spec.label}  •  {message}  •  {w}×{h}  •  {elapsed_ms:.0f}ms")

    def _compute_metrics(self, result):
        inp = self._input_img
        if inp is None or not isinstance(result, np.ndarray): return
        try:
            def gray(x): return cv2.cvtColor(x,cv2.COLOR_BGR2GRAY) if len(x.shape)==3 else x
            g1 = gray(inp).astype(np.float64)
            g2 = gray(result).astype(np.float64)
            if g1.shape != g2.shape:
                g2 = cv2.resize(g2.astype(np.uint8),(g1.shape[1],g1.shape[0])).astype(np.float64)
            mse = np.mean((g1-g2)**2); rmse = np.sqrt(mse)
            psnr = 10*np.log10(255**2/(mse+1e-10))
            sharp = cv2.Laplacian(result,cv2.CV_64F).var()
            ssim_s = "—"
            try:
                from skimage.metrics import structural_similarity as ssim
                ssim_s = f"{ssim(g1,g2,data_range=255.0):.4f}"
            except ImportError: pass
            self._metric_lbl.setText(
                f"PSNR: {psnr:.1f}dB  •  RMSE: {rmse:.2f}  •  SSIM: {ssim_s}  •  Sharpness: {sharp:.0f}")
        except Exception: pass

    # ─── Pipeline ─────────────────────────────────────────────────────

    def _run_pipeline(self):
        if self._input_img is None: return
        p = self._pipeline_panel.get_pipeline()
        if not p.nodes:
            QMessageBox.information(self,"Pipeline","Add algorithms to the pipeline first."); return
        self.statusBar().showMessage(f"Running pipeline: {p.name}…")
        result = p.final_output(self._input_img)
        if isinstance(result, np.ndarray):
            self._output_viewer.set_image(result)
            self._output_hist.set_image(result)
            self._compute_metrics(result)
        elif isinstance(result, dict):
            self._feat_display.show_features(result)
        self.statusBar().showMessage(f"Pipeline '{p.name}' complete  •  {len(p.nodes)} steps")

    def _save_pipeline(self):
        p = self._pipeline_panel.get_pipeline()
        fn,_ = QFileDialog.getSaveFileName(self,"Save Pipeline","pipeline.json","JSON (*.json)")
        if fn: p.save(fn)

    def _load_pipeline(self):
        fn,_ = QFileDialog.getOpenFileName(self,"Load Pipeline","","JSON (*.json)")
        if not fn: return
        p = Pipeline.load(fn)
        self._pipeline = p
        self._pipeline_panel.set_pipeline(p)

    def _load_preset_pipeline(self, pipeline: Pipeline):
        self._pipeline = pipeline
        self._pipeline_panel.set_pipeline(pipeline)

    # ─── I/O ──────────────────────────────────────────────────────────

    def _create_default_image(self):
        img = np.zeros((420, 640, 3), np.uint8)
        # Background grid
        for i in range(0, 640, 40):
            cv2.line(img, (i,0), (i,420), (8,10,18), 1)
        for j in range(0, 420, 40):
            cv2.line(img, (0,j), (640,j), (8,10,18), 1)
        cv2.rectangle(img,(30,30),(220,200),(0,180,120),2)
        cv2.circle(img,(460,130),85,(180,80,220),-1)
        cv2.ellipse(img,(320,310),(135,65),20,0,360,(0,160,255),3)
        cv2.line(img,(30,370),(610,380),(220,200,0),3)
        cv2.putText(img,"TriVision",(155,265),cv2.FONT_HERSHEY_DUPLEX,1.6,(240,240,255),2)
        cv2.putText(img,"OpenCV · CVIPtools2 · scikit-image",(105,300),cv2.FONT_HERSHEY_SIMPLEX,0.6,(100,130,180),1)
        self._set_input(img)

    def _set_input(self, img: np.ndarray):
        self._input_img = img
        self._input_viewer.set_image(img)
        self._input_hist.set_image(img)
        h,w = img.shape[:2]
        ch = img.shape[2] if len(img.shape)==3 else 1
        self.statusBar().showMessage(f"Input: {w}×{h}  ch={ch}")

    def _load_image(self):
        fn,_ = QFileDialog.getOpenFileName(self,"Open Image","",
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.tif *.webp *.pgm *.ppm);;All (*)")
        if fn:
            img = cv2.imread(fn)
            if img is not None: self._set_input(img)
            else: QMessageBox.warning(self,"Error",f"Cannot read: {fn}")

    def _save_image(self):
        out = self._output_viewer.get_image()
        if out is None: QMessageBox.information(self,"Save","No output to save."); return
        fn,_ = QFileDialog.getSaveFileName(self,"Save Output","","PNG (*.png);;JPEG (*.jpg);;BMP (*.bmp)")
        if fn: cv2.imwrite(fn, out)

    def _copy_to_input(self):
        out = self._output_viewer.get_image()
        if out is not None: self._set_input(out)

    def _show_diff(self):
        inp = self._input_img; out = self._output_viewer.get_image()
        if inp is None or out is None:
            QMessageBox.information(self,"Diff","Need both input and output."); return
        try:
            r = cv2.resize(out,(inp.shape[1],inp.shape[0]))
            diff = cv2.convertScaleAbs(cv2.absdiff(inp,r),alpha=3.0)
            self._output_viewer.set_image(diff)
            self.statusBar().showMessage("Showing ×3 amplified diff")
        except Exception as e:
            QMessageBox.warning(self,"Diff",str(e))

    def _show_ab(self):
        inp = self._input_img; out = self._output_viewer.get_image()
        if inp is None or out is None:
            QMessageBox.information(self,"A/B","Need both input and output."); return
        bgr_in = cv2.cvtColor(inp,cv2.COLOR_GRAY2BGR) if len(inp.shape)==2 else inp
        bgr_out = cv2.cvtColor(out,cv2.COLOR_GRAY2BGR) if len(out.shape)==2 else out
        h = max(bgr_in.shape[0],bgr_out.shape[0])
        a = cv2.resize(bgr_in,(bgr_in.shape[1],h))
        b = cv2.resize(bgr_out,(bgr_out.shape[1],h))
        div = np.full((h,4,3),[40,160,255],np.uint8)
        combined = np.hstack([a,div,b])
        cv2.putText(combined,"INPUT",(6,22),cv2.FONT_HERSHEY_DUPLEX,0.6,(40,200,255),1)
        cv2.putText(combined,"OUTPUT",(a.shape[1]+10,22),cv2.FONT_HERSHEY_DUPLEX,0.6,(40,200,255),1)
        self._output_viewer.set_image(combined)
        self._output_hist.set_image(combined)

    def _capture_webcam(self):
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret,frame = cap.read(); cap.release()
            if ret: self._set_input(frame); return
        QMessageBox.warning(self,"Webcam","Cannot access webcam.")

    def _extract_features(self):
        if self._input_img is None: return
        spec = REGISTRY.get("tv_all_features")
        if spec:
            result = spec.fn(self._input_img)
            if isinstance(result, dict):
                self._feat_display.show_features(result)

    def _about(self):
        QMessageBox.about(self,"About TriVision",
            "<b>TriVision</b><br>"
            "Unified Computer Vision &amp; Image Processing Workbench<br><br>"
            f"<b>{len(REGISTRY)} algorithms</b> from three libraries:<br>"
            "  OpenCV — speed, camera I/O, DNN inference<br>"
            "  CVIPtools2 — classical CVIP algorithms<br>"
            "  scikit-image — research-grade algorithms<br>"
            "  TriVision fusion — cross-library composites<br><br>"
            "Features: visual pipeline builder, A/B compare, batch processing,<br>"
            "comprehensive feature extraction, plugin SDK, quality metrics.")


# ═══════════════════════════════════════════════════════════════════════════════
# Dark palette + launch
# ═══════════════════════════════════════════════════════════════════════════════

def _dark_palette(app):
    p = QPalette(); c = QColor
    p.setColor(QPalette.ColorRole.Window,         c(4,6,12))
    p.setColor(QPalette.ColorRole.WindowText,      c(180,190,210))
    p.setColor(QPalette.ColorRole.Base,            c(3,5,9))
    p.setColor(QPalette.ColorRole.AlternateBase,   c(6,9,16))
    p.setColor(QPalette.ColorRole.Text,            c(180,190,210))
    p.setColor(QPalette.ColorRole.Button,          c(8,12,20))
    p.setColor(QPalette.ColorRole.ButtonText,      c(160,180,210))
    p.setColor(QPalette.ColorRole.Highlight,       c(20,60,140))
    p.setColor(QPalette.ColorRole.HighlightedText, c(200,220,255))
    app.setPalette(p)


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    _dark_palette(app)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
