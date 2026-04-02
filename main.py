"""
TriVision — Main Application
The unified OpenCV + CVIPtools2 + scikit-image workbench.

python main.py
"""

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json, time
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

import core.algorithms_opencv as _ocv
import core.algorithms_skimage as _ski
import core.algorithms_cvip_fusion as _cvf
from core.registry import REGISTRY, Lib, ReturnType
from pipeline.engine import Pipeline
from plugins.sdk import load_plugins


def _bootstrap():
    _ocv.register_all()
    _ski.register_all()
    _cvf.register_all()
    loaded = load_plugins("plugins")
    print(f"TriVision: {len(REGISTRY)} algorithms registered, {len(loaded)} plugins loaded.")

_bootstrap()


# ═══════════════════════════════════════════════════════════════════════════════
# Theme System
# ═══════════════════════════════════════════════════════════════════════════════

class Theme:
    DARK = "dark"
    LIGHT = "light"
    _current = DARK

    DARK_COLORS = {
        "bg_window":       "#04060c", "bg_base":        "#03050a",
        "bg_panel":        "#06080e", "bg_deep":        "#030508",
        "bg_hover":        "#0d1220", "bg_selected":    "#162240",
        "bg_button":       "#0a0f1a", "bg_action":      "#102060",
        "bg_action_hover": "#1a3a90", "bg_preset":      "#0a0f1a",
        "bg_hist":         "#030508", "border":         "#1a1f2e",
        "border_strong":   "#101828", "border_focus":   "#2a4070",
        "text_primary":    "#b4beef", "text_secondary": "#6a7a90",
        "text_muted":      "#3a4a60", "text_accent":    "#7aafff",
        "text_action":     "#a0c0ff", "text_green":     "#60c060",
        "viewer_bg":       "#060810", "viewer_hint":    "#252a3a",
        "status_bg":       "#030508", "status_text":    "#3a5060",
        "metric_bg":       "#030508", "metric_text":    "#3a5a3a",
        "metric_border":   "#0e1018", "tab_bg":         "#06080e",
        "tab_selected_bg": "#0d1828", "tab_text":       "#55667a",
        "tab_selected_text":"#7aafff","cat_text":       "#3a6080",
        "subcat_text":     "#2a4060", "menu_bg":        "#06080e",
        "menu_text":       "#aaaaaa", "menu_selected":  "#1a2840",
        "menubar_bg":      "#030508", "menubar_text":   "#6080a0",
        "scroll_bg":       "#04060c", "progress_bg":    "#060810",
        "progress_chunk":  "#1a4080",
    }
    LIGHT_COLORS = {
        "bg_window":       "#f0f2f7", "bg_base":        "#ffffff",
        "bg_panel":        "#f8f9fc", "bg_deep":        "#eef0f5",
        "bg_hover":        "#e4e8f2", "bg_selected":    "#d0dff5",
        "bg_button":       "#eceef5", "bg_action":      "#2563eb",
        "bg_action_hover": "#1d4ed8", "bg_preset":      "#eff2fb",
        "bg_hist":         "#f5f6fa", "border":         "#d0d5e8",
        "border_strong":   "#c0c8de", "border_focus":   "#6090d0",
        "text_primary":    "#1a2035", "text_secondary": "#4a5570",
        "text_muted":      "#8090aa", "text_accent":    "#2563eb",
        "text_action":     "#ffffff", "text_green":     "#16a34a",
        "viewer_bg":       "#e8eaf2", "viewer_hint":    "#b0b8cc",
        "status_bg":       "#ffffff", "status_text":    "#4a6080",
        "metric_bg":       "#f0f4ff", "metric_text":    "#2a6040",
        "metric_border":   "#c8d0e0", "tab_bg":         "#eceef5",
        "tab_selected_bg": "#ffffff", "tab_text":       "#7080a0",
        "tab_selected_text":"#2563eb","cat_text":       "#1a5090",
        "subcat_text":     "#3060a0", "menu_bg":        "#ffffff",
        "menu_text":       "#1a2035", "menu_selected":  "#dde8fa",
        "menubar_bg":      "#f0f2f7", "menubar_text":   "#3060a0",
        "scroll_bg":       "#f0f2f7", "progress_bg":    "#e0e4f0",
        "progress_chunk":  "#2563eb",
    }

    LIB_COLORS_DARK  = {Lib.OPENCV:"#1a4a8a", Lib.CVIP:"#1a6a4a",
                         Lib.SKIMAGE:"#5a2a8a", Lib.TRIVISION:"#8a3a1a"}
    LIB_COLORS_LIGHT = {Lib.OPENCV:"#1e56b0", Lib.CVIP:"#1a7a50",
                         Lib.SKIMAGE:"#6a30aa", Lib.TRIVISION:"#b04020"}
    LIB_LABELS       = {Lib.OPENCV:"CV", Lib.CVIP:"CL",
                         Lib.SKIMAGE:"SK", Lib.TRIVISION:"TV"}
    LIB_BTN_DARK  = {"ALL":"#2a3040","OpenCV":"#1a4a8a","CVIPtools2":"#1a6a4a",
                      "scikit-image":"#5a2a8a","TriVision":"#8a3a1a"}
    LIB_BTN_LIGHT = {"ALL":"#6080b0","OpenCV":"#2060c0","CVIPtools2":"#1a8a55",
                      "scikit-image":"#7040b0","TriVision":"#c04820"}

    @classmethod
    def is_dark(cls): return cls._current == cls.DARK

    @classmethod
    def toggle(cls): cls._current = cls.LIGHT if cls._current == cls.DARK else cls.DARK

    @classmethod
    def c(cls, key):
        return (cls.DARK_COLORS if cls.is_dark() else cls.LIGHT_COLORS).get(key, "#ff00ff")

    @classmethod
    def lib_color(cls, lib):
        return (cls.LIB_COLORS_DARK if cls.is_dark() else cls.LIB_COLORS_LIGHT).get(lib, "#888")

    @classmethod
    def lib_btn_color(cls, name):
        return (cls.LIB_BTN_DARK if cls.is_dark() else cls.LIB_BTN_LIGHT).get(name, "#888")

    @classmethod
    def apply_palette(cls, app):
        p = QPalette(); c = QColor
        if cls.is_dark():
            p.setColor(QPalette.ColorRole.Window,         c(4,6,12))
            p.setColor(QPalette.ColorRole.WindowText,      c(180,190,210))
            p.setColor(QPalette.ColorRole.Base,            c(3,5,9))
            p.setColor(QPalette.ColorRole.AlternateBase,   c(6,9,16))
            p.setColor(QPalette.ColorRole.Text,            c(180,190,210))
            p.setColor(QPalette.ColorRole.Button,          c(8,12,20))
            p.setColor(QPalette.ColorRole.ButtonText,      c(160,180,210))
            p.setColor(QPalette.ColorRole.Highlight,       c(20,60,140))
            p.setColor(QPalette.ColorRole.HighlightedText, c(200,220,255))
        else:
            p.setColor(QPalette.ColorRole.Window,         c(240,242,247))
            p.setColor(QPalette.ColorRole.WindowText,      c(26,32,53))
            p.setColor(QPalette.ColorRole.Base,            c(255,255,255))
            p.setColor(QPalette.ColorRole.AlternateBase,   c(244,246,252))
            p.setColor(QPalette.ColorRole.Text,            c(26,32,53))
            p.setColor(QPalette.ColorRole.Button,          c(236,238,245))
            p.setColor(QPalette.ColorRole.ButtonText,      c(26,32,53))
            p.setColor(QPalette.ColorRole.Highlight,       c(37,99,235))
            p.setColor(QPalette.ColorRole.HighlightedText, c(255,255,255))
        app.setPalette(p)

T = Theme


# ═══════════════════════════════════════════════════════════════════════════════
# Theme Toggle Button
# ═══════════════════════════════════════════════════════════════════════════════

class ThemeToggleButton(QPushButton):
    theme_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(80, 26)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip("Toggle light / dark mode  (Ctrl+Shift+T)")
        self._update_label()
        self.clicked.connect(self._on_click)

    def _update_label(self):
        self.setText("☀  Light" if T.is_dark() else "🌙  Dark")
        self._refresh_style()

    def _refresh_style(self):
        if T.is_dark():
            self.setStyleSheet(
                "QPushButton{background:#1a2a50;color:#c0d8ff;border:1px solid #2a3a70;"
                "border-radius:13px;font-size:11px;font-weight:500;}"
                "QPushButton:hover{background:#243560;border-color:#4a6aaa;}"
            )
        else:
            self.setStyleSheet(
                "QPushButton{background:#e0eaf8;color:#1a3060;border:1px solid #a0c0e0;"
                "border-radius:13px;font-size:11px;font-weight:500;}"
                "QPushButton:hover{background:#ccddf5;border-color:#80a0c8;}"
            )

    def _on_click(self):
        T.toggle()
        self._update_label()
        self.theme_changed.emit()


# ═══════════════════════════════════════════════════════════════════════════════
# Image Viewer
# ═══════════════════════════════════════════════════════════════════════════════

class ImageViewer(QLabel):
    def __init__(self, hint="", parent=None):
        super().__init__(parent)
        self._img = None
        self._hint = hint
        self.setMinimumSize(320, 240)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.apply_theme()
        self._placeholder()

    def apply_theme(self):
        self.setStyleSheet(
            f"background:{T.c('viewer_bg')};border:1px solid {T.c('border')};border-radius:5px;")
        if self._img is None:
            self._placeholder()

    def _placeholder(self):
        hint_col = T.c("viewer_hint")
        self.setText(
            f"<span style='color:{hint_col};font-family:monospace;font-size:11px'>"
            f"{self._hint}</span>")

    def set_image(self, img):
        self._img = img.copy() if img is not None else None
        if img is None: self._placeholder(); return
        self._render()

    def get_image(self): return self._img

    def _render(self):
        img = self._img
        if img is None: return
        if len(img.shape) == 2: rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4: rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        else: rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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
        self.setMinimumHeight(60); self.setMaximumHeight(72)
        self._ch = []
        self.apply_theme()

    def apply_theme(self):
        self.setStyleSheet(
            f"background:{T.c('bg_hist')};border-radius:0 0 4px 4px;")
        self.update()

    def set_image(self, img):
        if img is None: self._ch=[]; self.update(); return
        self._ch=[]
        if len(img.shape)==2:
            self._ch=[((160,160,180), cv2.calcHist([img],[0],None,[64],[0,256]).flatten())]
        else:
            for i,c in enumerate([(80,80,220),(80,200,80),(220,80,80)]):
                self._ch.append((c, cv2.calcHist([img],[i],None,[64],[0,256]).flatten()))
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(T.c("bg_hist")))
        if not self._ch: return
        w, h = self.width(), self.height()
        mx = max(ch[1].max() for ch in self._ch) or 1
        for col, hist in self._ch:
            r,g,b = col
            p.setPen(QPen(QColor(r,g,b, 160 if T.is_dark() else 200), 1))
            n=len(hist); bw=(w-4)/n
            for i,v in enumerate(hist):
                bh=int((v/mx)*(h-4)); x=int(2+i*bw)
                p.drawLine(x, h-2, x, h-2-bh)
        p.end()


# ═══════════════════════════════════════════════════════════════════════════════
# Worker Thread
# ═══════════════════════════════════════════════════════════════════════════════

class Worker(QThread):
    done = pyqtSignal(object, float, str)

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
# Param Panel
# ═══════════════════════════════════════════════════════════════════════════════

class ParamPanel(QWidget):
    changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._form = QFormLayout(self)
        self._form.setSpacing(6)
        self._form.setContentsMargins(4,4,4,4)
        self._getters: dict[str, Callable] = {}

    def _style_input(self, w):
        bg=T.c("bg_base"); bd=T.c("border"); txt=T.c("text_primary"); foc=T.c("border_focus")
        base = (f"background:{bg};color:{txt};border:1px solid {bd};"
                f"padding:3px;border-radius:3px;font-size:11px;")
        if isinstance(w, (QSpinBox, QDoubleSpinBox)):
            w.setStyleSheet(f"QSpinBox,QDoubleSpinBox{{{base}}}"
                            f"QSpinBox:focus,QDoubleSpinBox:focus{{border-color:{foc};}}")
        elif isinstance(w, QComboBox):
            w.setStyleSheet(f"QComboBox{{{base}}}QComboBox:focus{{border-color:{foc};}}")
        elif isinstance(w, QCheckBox):
            w.setStyleSheet(f"QCheckBox{{color:{txt};font-size:11px;}}")

    def apply_theme(self):
        for i in range(self._form.rowCount()):
            lbl = self._form.itemAt(i, QFormLayout.ItemRole.LabelRole)
            fld = self._form.itemAt(i, QFormLayout.ItemRole.FieldRole)
            if lbl and lbl.widget():
                lbl.widget().setStyleSheet(
                    f"QLabel{{color:{T.c('text_secondary')};font-size:11px;}}")
            if fld and fld.widget():
                self._style_input(fld.widget())

    def load_spec(self, spec):
        while self._form.count():
            item = self._form.takeAt(0)
            if item.widget(): item.widget().deleteLater()
        self._getters.clear()
        if spec is None: return
        for p in spec.params:
            if p.kind == "int":
                w = QSpinBox()
                w.setRange(int(p.lo), int(p.hi)); w.setValue(int(p.default)); w.setSingleStep(int(p.step))
                w.valueChanged.connect(self.changed); self._getters[p.name] = w.value
            elif p.kind == "float":
                w = QDoubleSpinBox()
                w.setRange(float(p.lo), float(p.hi)); w.setValue(float(p.default)); w.setSingleStep(float(p.step))
                dec = max(2, len(str(p.step).rstrip("0").split(".")[-1])); w.setDecimals(dec)
                w.valueChanged.connect(self.changed); self._getters[p.name] = w.value
            elif p.kind == "bool":
                w = QCheckBox(); w.setChecked(bool(p.default))
                w.toggled.connect(self.changed); self._getters[p.name] = w.isChecked
            elif p.kind == "choice":
                w = QComboBox(); w.addItems(p.choices)
                if p.default in p.choices: w.setCurrentText(p.default)
                w.currentIndexChanged.connect(self.changed); self._getters[p.name] = w.currentText
            else: continue
            lbl = QLabel(f"{p.label}:")
            lbl.setStyleSheet(f"QLabel{{color:{T.c('text_secondary')};font-size:11px;}}")
            self._style_input(w)
            self._form.addRow(lbl, w)

    def get_kwargs(self): return {k: g() for k, g in self._getters.items()}


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline Panel
# ═══════════════════════════════════════════════════════════════════════════════

class PipelinePanel(QWidget):
    run_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0,0,0,0); layout.setSpacing(5)

        hdr = QHBoxLayout()
        self._hdr_lbl = QLabel("Pipeline")
        self._hdr_lbl.setStyleSheet("font-weight:bold;font-size:12px;")
        hdr.addWidget(self._hdr_lbl)
        self.clear_btn = QPushButton("✕ Clear"); self.clear_btn.setFixedWidth(70)
        self.save_btn  = QPushButton("💾");       self.save_btn.setFixedWidth(32)
        self.load_btn  = QPushButton("📂");       self.load_btn.setFixedWidth(32)
        self._ctrl = [self.clear_btn, self.save_btn, self.load_btn]
        for b in self._ctrl: hdr.addWidget(b)
        layout.addLayout(hdr)

        self.node_list = QListWidget()
        self.node_list.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.node_list.setMinimumHeight(120)
        layout.addWidget(self.node_list)

        self._preset_lbl = QLabel("Presets:")
        layout.addWidget(self._preset_lbl)

        self._preset_btns = []
        for name, fn in [
            ("Edge Detection", Pipeline.edge_detection_pipeline),
            ("Denoise + Enhance", Pipeline.denoise_and_enhance),
            ("Segmentation", Pipeline.segmentation_pipeline),
            ("Feature Extraction", Pipeline.feature_extraction_pipeline),
        ]:
            b = QPushButton(name)
            b.clicked.connect(lambda checked, f=fn: self._load_preset(f))
            layout.addWidget(b); self._preset_btns.append(b)

        self.run_btn = QPushButton("▶  Run Pipeline")
        self.run_btn.setMinimumHeight(34)
        layout.addWidget(self.run_btn)

        self._pipeline = Pipeline()
        self._preset_cb = None
        self.apply_theme()

    def apply_theme(self):
        bg=T.c("bg_base"); bd=T.c("border"); txt=T.c("text_secondary")
        hover=T.c("bg_hover"); bg_btn=T.c("bg_button"); acc=T.c("text_accent")
        preset_bg=T.c("bg_preset"); sel=T.c("bg_selected")
        act_bg=T.c("bg_action"); act_hover=T.c("bg_action_hover"); act_txt=T.c("text_action")

        self._hdr_lbl.setStyleSheet(f"font-weight:bold;font-size:12px;color:{T.c('text_primary')};")
        for b in self._ctrl:
            b.setStyleSheet(
                f"QPushButton{{background:{bg_btn};color:{txt};border:1px solid {bd};"
                f"padding:2px 4px;font-size:11px;border-radius:3px;}}"
                f"QPushButton:hover{{background:{hover};}}")

        self.node_list.setStyleSheet(
            f"QListWidget{{background:{bg};border:1px solid {bd};font-size:11px;border-radius:4px;}}"
            f"QListWidget::item{{padding:5px 8px;border-bottom:1px solid {bd};}}"
            f"QListWidget::item:selected{{background:{sel};color:{acc};}}"
            f"QListWidget::item:hover{{background:{hover};}}")

        self._preset_lbl.setStyleSheet(
            f"color:{txt};font-size:11px;margin-top:4px;font-weight:500;")
        for b in self._preset_btns:
            b.setStyleSheet(
                f"QPushButton{{background:{preset_bg};color:{acc};border:1px solid {bd};"
                f"font-size:11px;padding:5px;text-align:left;padding-left:10px;border-radius:4px;}}"
                f"QPushButton:hover{{background:{hover};}}")

        self.run_btn.setStyleSheet(
            f"QPushButton{{background:{act_bg};color:{act_txt};font-weight:bold;"
            f"border-radius:5px;border:none;font-size:12px;}}"
            f"QPushButton:hover{{background:{act_hover};}}")

    def set_preset_callback(self, cb): self._preset_cb = cb
    def _load_preset(self, fn):
        if self._preset_cb: self._preset_cb(fn())
    def set_pipeline(self, p):
        self._pipeline = p; self._refresh_list()
    def get_pipeline(self): return self._pipeline

    def add_algo(self, algo_key):
        spec = REGISTRY.get(algo_key)
        if spec is None: return
        upstream = self._pipeline.nodes[-1].node_id if self._pipeline.nodes else None
        node = self._pipeline.add_node(algo_key, upstream_id=upstream)
        col = T.lib_color(spec.lib); tag = T.LIB_LABELS.get(spec.lib,"?")
        item = QListWidgetItem(f"[{tag}] {spec.label}")
        item.setData(Qt.ItemDataRole.UserRole, node.node_id)
        item.setForeground(QColor(col)); self.node_list.addItem(item)

    def remove_selected(self):
        item = self.node_list.currentItem()
        if not item: return
        self._pipeline.remove_node(item.data(Qt.ItemDataRole.UserRole))
        self.node_list.takeItem(self.node_list.row(item))

    def clear(self):
        self._pipeline.clear(); self.node_list.clear()

    def _refresh_list(self):
        self.node_list.clear()
        for node in self._pipeline.nodes:
            spec = REGISTRY.get(node.algo_key)
            if spec is None: continue
            col = T.lib_color(spec.lib); tag = T.LIB_LABELS.get(spec.lib,"?")
            item = QListWidgetItem(f"[{tag}] {spec.label}")
            item.setData(Qt.ItemDataRole.UserRole, node.node_id)
            item.setForeground(QColor(col)); self.node_list.addItem(item)


# ═══════════════════════════════════════════════════════════════════════════════
# Feature Display
# ═══════════════════════════════════════════════════════════════════════════════

class FeatureDisplay(QTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True); self.setMaximumHeight(180)
        self.setPlaceholderText("Feature extraction results appear here…")
        self.apply_theme()

    def apply_theme(self):
        self.setStyleSheet(
            f"QTextEdit{{background:{T.c('bg_deep')};color:{T.c('text_green')};"
            f"font-family:monospace;font-size:11px;border:none;"
            f"border-top:1px solid {T.c('border')};}}")

    def show_features(self, d):
        lines = []
        for k,v in d.items():
            lines.append(f"  {k:<32} {v:.6f}" if isinstance(v,float) else f"  {k:<32} {v}")
        self.setText("\n".join(lines))


# ═══════════════════════════════════════════════════════════════════════════════
# Batch Tab
# ═══════════════════════════════════════════════════════════════════════════════

class BatchTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setSpacing(8); layout.setContentsMargins(6,8,6,6)

        self._row_lbls = {}; self._edits = {}; self._browse_btns = {}

        for attr, label in [("_in_edit","Input folder:"),("_out_edit","Output folder:")]:
            row = QHBoxLayout()
            lbl = QLabel(label); lbl.setFixedWidth(95)
            self._row_lbls[attr] = lbl
            edit = QLineEdit(); setattr(self, attr, edit); self._edits[attr] = edit
            btn = QPushButton("…"); btn.setFixedWidth(28); self._browse_btns[attr] = btn
            a = attr; btn.clicked.connect(lambda _,a=a: self._browse(a))
            row.addWidget(lbl); row.addWidget(edit); row.addWidget(btn)
            layout.addLayout(row)

        algo_row = QHBoxLayout()
        self._algo_lbl = QLabel("Algorithm:"); self._algo_lbl.setFixedWidth(95)
        self._algo_combo = QComboBox()
        for spec in sorted(REGISTRY.all(), key=lambda s: s.label):
            self._algo_combo.addItem(f"[{T.LIB_LABELS.get(spec.lib,'?')}] {spec.label}", spec.key)
        algo_row.addWidget(self._algo_lbl); algo_row.addWidget(self._algo_combo)
        layout.addLayout(algo_row)

        opts = QHBoxLayout()
        self._save_check = QCheckBox("Save outputs"); self._save_check.setChecked(True)
        opts.addWidget(self._save_check); opts.addStretch(); layout.addLayout(opts)

        self._run_btn = QPushButton("▶  Run Batch")
        self._run_btn.setMinimumHeight(34); self._run_btn.clicked.connect(self._run_batch)
        layout.addWidget(self._run_btn)

        self._progress = QProgressBar()
        self._progress.setMaximum(100); self._progress.setValue(0)
        layout.addWidget(self._progress)

        self._results = QTextEdit(); self._results.setReadOnly(True)
        layout.addWidget(self._results, 1)

        export_row = QHBoxLayout()
        self._export_csv  = QPushButton("Export CSV")
        self._export_json = QPushButton("Export JSON")
        self._exp_btns = [self._export_csv, self._export_json]
        for b in self._exp_btns: export_row.addWidget(b)
        self._export_csv.clicked.connect(self._export_to_csv)
        self._export_json.clicked.connect(self._export_to_json)
        layout.addLayout(export_row)
        self._report = None
        self.apply_theme()

    def apply_theme(self):
        bg=T.c("bg_base"); bd=T.c("border"); txt=T.c("text_primary")
        txt2=T.c("text_secondary"); hover=T.c("bg_hover"); bg_btn=T.c("bg_button")
        acc=T.c("text_accent"); deep=T.c("bg_deep"); foc=T.c("border_focus")
        act_bg=T.c("bg_action"); act_hover=T.c("bg_action_hover"); act_txt=T.c("text_action")
        prog_bg=T.c("progress_bg"); prog_chunk=T.c("progress_chunk")

        for lbl in list(self._row_lbls.values()) + [self._algo_lbl]:
            lbl.setStyleSheet(f"color:{txt2};font-size:11px;")
        for edit in self._edits.values():
            edit.setStyleSheet(
                f"QLineEdit{{background:{bg};color:{txt};border:1px solid {bd};"
                f"padding:4px 6px;border-radius:3px;font-size:11px;}}"
                f"QLineEdit:focus{{border-color:{foc};}}")
        for btn in self._browse_btns.values():
            btn.setStyleSheet(
                f"QPushButton{{background:{bg_btn};color:{txt2};border:1px solid {bd};"
                f"border-radius:3px;}}"
                f"QPushButton:hover{{background:{hover};}}")
        self._algo_combo.setStyleSheet(
            f"QComboBox{{background:{bg};color:{txt};border:1px solid {bd};"
            f"padding:4px;border-radius:3px;font-size:11px;}}")
        self._save_check.setStyleSheet(f"QCheckBox{{color:{txt2};font-size:11px;}}")
        self._run_btn.setStyleSheet(
            f"QPushButton{{background:{act_bg};color:{act_txt};font-weight:bold;"
            f"border-radius:5px;border:none;font-size:12px;}}"
            f"QPushButton:hover{{background:{act_hover};}}")
        self._progress.setStyleSheet(
            f"QProgressBar{{background:{prog_bg};border:1px solid {bd};color:{txt2};"
            f"font-size:10px;border-radius:3px;}}"
            f"QProgressBar::chunk{{background:{prog_chunk};border-radius:3px;}}")
        self._results.setStyleSheet(
            f"QTextEdit{{background:{deep};color:{acc};font-family:monospace;"
            f"font-size:10px;border:1px solid {bd};border-radius:4px;}}")
        for b in self._exp_btns:
            b.setStyleSheet(
                f"QPushButton{{background:{bg_btn};color:{txt2};border:1px solid {bd};"
                f"font-size:11px;padding:5px;border-radius:4px;}}"
                f"QPushButton:hover{{background:{hover};color:{txt};}}")

    def _browse(self, attr):
        d = QFileDialog.getExistingDirectory(self, "Select folder")
        if d: getattr(self, attr).setText(d)

    def _run_batch(self):
        from batch.processor import BatchProcessor
        in_dir = self._in_edit.text().strip(); out_dir = self._out_edit.text().strip()
        if not in_dir: QMessageBox.warning(self,"Batch","Please select an input folder."); return
        if not out_dir and self._save_check.isChecked():
            QMessageBox.warning(self,"Batch","Please select an output folder."); return
        key = self._algo_combo.currentData()
        self._run_btn.setEnabled(False); self._progress.setValue(0); self._results.clear()
        processor = BatchProcessor(algo_key=key); self._report = None
        def progress_cb(cur, total, fname):
            self._progress.setValue(int(cur/max(total,1)*100))
            self._results.append(f"  [{cur}/{total}] {fname}")
        import threading
        def do_batch():
            report = processor.run(in_dir, out_dir or ".",
                                    save_outputs=self._save_check.isChecked(),
                                    progress_cb=progress_cb)
            self._report = report
            lines = ["\n─── Batch Complete ───"]
            for k,v in report.summary().items(): lines.append(f"  {k}: {v}")
            self._results.append("\n".join(lines)); self._run_btn.setEnabled(True)
        threading.Thread(target=do_batch, daemon=True).start()

    def _export_to_csv(self):
        if not self._report: QMessageBox.information(self,"Export","Run a batch first."); return
        fn,_ = QFileDialog.getSaveFileName(self,"Save CSV","batch_report.csv","CSV (*.csv)")
        if fn: self._report.to_csv(fn)

    def _export_to_json(self):
        if not self._report: QMessageBox.information(self,"Export","Run a batch first."); return
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
        self._input_img = None; self._current_spec = None
        self._worker = None; self._auto = True
        self._pipeline = Pipeline()
        self._setup_ui(); self._setup_menu(); self._create_default_image()

    def _setup_ui(self):
        root = QWidget(); self.setCentralWidget(root)
        rl = QHBoxLayout(root); rl.setContentsMargins(4,4,4,4); rl.setSpacing(4)
        outer = QSplitter(Qt.Orientation.Horizontal)

        # ── LEFT ──────────────────────────────────────────────────────
        left = QWidget(); ll = QVBoxLayout(left)
        ll.setContentsMargins(0,0,0,0); ll.setSpacing(5)

        top_bar = QHBoxLayout()
        self._logo_lbl = QLabel("⬡ TriVision")
        top_bar.addWidget(self._logo_lbl); top_bar.addStretch()
        self._theme_btn = ThemeToggleButton()
        self._theme_btn.theme_changed.connect(self._apply_theme_all)
        top_bar.addWidget(self._theme_btn); ll.addLayout(top_bar)

        self._search = QLineEdit()
        self._search.setPlaceholderText("Search algorithms…")
        self._search.textChanged.connect(self._filter_tree)
        ll.addWidget(self._search)

        lib_row = QHBoxLayout(); lib_row.setSpacing(3)
        self._lib_btns = {}
        for lib, label in [("ALL","All"),("OpenCV","CV"),("CVIPtools2","CL"),
                            ("scikit-image","SK"),("TriVision","TV")]:
            b = QPushButton(label); b.setFixedHeight(22)
            b.clicked.connect(lambda _, l=lib: self._filter_by_lib(l))
            lib_row.addWidget(b); self._lib_btns[lib] = b
        ll.addLayout(lib_row)

        self._tree = QTreeWidget(); self._tree.setHeaderHidden(True)
        self._tree.itemClicked.connect(self._on_tree_click)
        self._tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._tree.customContextMenuRequested.connect(self._tree_context_menu)
        self._build_tree(); ll.addWidget(self._tree, 1)

        self._count_lbl = QLabel(f"{len(REGISTRY)} algorithms")
        ll.addWidget(self._count_lbl)
        outer.addWidget(left)

        # ── CENTRE ────────────────────────────────────────────────────
        centre = QWidget(); cl = QVBoxLayout(centre)
        cl.setSpacing(6); cl.setContentsMargins(4,2,4,2)

        viewers = QHBoxLayout(); viewers.setSpacing(8)
        for attr, hint, hist_attr in [("_input_viewer","INPUT","_input_hist"),
                                       ("_output_viewer","OUTPUT","_output_hist")]:
            grp = QGroupBox(hint); grp.setObjectName(f"grp_{attr}")
            gl = QVBoxLayout(grp); gl.setSpacing(2); gl.setContentsMargins(6,16,6,6)
            v = ImageViewer(hint.lower()); setattr(self, attr, v); gl.addWidget(v)
            h = HistogramWidget(); setattr(self, hist_attr, h); gl.addWidget(h)
            viewers.addWidget(grp)
        cl.addLayout(viewers, 4)

        ctrl = QHBoxLayout(); ctrl.setSpacing(6)
        self._process_btn = QPushButton("▶  Process"); self._process_btn.setMinimumHeight(36)
        self._process_btn.clicked.connect(self._process)
        self._auto_check = QCheckBox("Auto"); self._auto_check.setChecked(True)
        self._auto_check.toggled.connect(lambda v: setattr(self,'_auto',v))
        self._copy_btn = QPushButton("→ Use as Input")
        self._diff_btn = QPushButton("⊕ Diff")
        self._ab_btn   = QPushButton("A/B Compare")
        self._reset_btn= QPushButton("↺ Reset")
        self._action_btns = [self._copy_btn, self._diff_btn, self._ab_btn, self._reset_btn]
        self._copy_btn.clicked.connect(self._copy_to_input)
        self._diff_btn.clicked.connect(self._show_diff)
        self._ab_btn.clicked.connect(self._show_ab)
        self._reset_btn.clicked.connect(self._create_default_image)
        for w in [self._process_btn, self._auto_check] + self._action_btns: ctrl.addWidget(w)
        cl.addLayout(ctrl)

        self._feat_display = FeatureDisplay(); cl.addWidget(self._feat_display)
        self._metric_lbl = QLabel("PSNR: —  RMSE: —  SSIM: —  Sharpness: —")
        cl.addWidget(self._metric_lbl)
        outer.addWidget(centre)

        # ── RIGHT ─────────────────────────────────────────────────────
        self._right_tabs = QTabWidget(); self._right_tabs.setMinimumWidth(300)

        # Params tab
        param_widget = QWidget()
        pw = QVBoxLayout(param_widget); pw.setContentsMargins(6,6,6,6); pw.setSpacing(4)
        self._algo_label = QLabel("Select an algorithm")
        self._lib_badge  = QLabel("")
        self._algo_desc  = QLabel(""); self._algo_desc.setWordWrap(True)
        pw.addWidget(self._algo_label); pw.addWidget(self._lib_badge); pw.addWidget(self._algo_desc)
        self._param_panel = ParamPanel(); self._param_panel.changed.connect(self._on_param_changed)
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(self._param_panel)
        pw.addWidget(scroll, 1)
        io_grp = QGroupBox("Image I/O"); io_grp.setObjectName("io_grp")
        ig = QVBoxLayout(io_grp); ig.setSpacing(4)
        self._io_buttons = []
        for lbl_txt, slot in [("📂 Load Image",self._load_image),
                               ("💾 Save Output",self._save_image),
                               ("📷 Webcam",self._capture_webcam),
                               ("📊 Extract All Features",self._extract_features)]:
            b = QPushButton(lbl_txt); b.clicked.connect(slot); ig.addWidget(b)
            self._io_buttons.append(b)
        pw.addWidget(io_grp)
        self._right_tabs.addTab(param_widget, "Parameters")

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
        self._right_tabs.addTab(pipeline_widget, "Pipeline")

        # Batch tab
        self._batch_tab = BatchTab()
        self._right_tabs.addTab(self._batch_tab, "Batch")

        outer.addWidget(self._right_tabs)
        outer.setSizes([240, 1000, 320]); rl.addWidget(outer)

        # Status bar
        sb = QStatusBar(); self.setStatusBar(sb)
        self._prog = QProgressBar()
        self._prog.setMaximumWidth(200); self._prog.setMaximumHeight(14)
        self._prog.setRange(0,0); self._prog.setVisible(False)
        sb.addPermanentWidget(self._prog)
        self.statusBar().showMessage("TriVision ready — load an image to begin")

        self._apply_theme_all()

    def _apply_theme_all(self):
        app = QApplication.instance()
        T.apply_palette(app)

        bg=T.c("bg_window"); base=T.c("bg_base"); panel=T.c("bg_panel")
        bd=T.c("border"); txt=T.c("text_primary"); txt2=T.c("text_secondary")
        txt_m=T.c("text_muted"); acc=T.c("text_accent"); hover=T.c("bg_hover")
        bg_btn=T.c("bg_button"); sel=T.c("bg_selected"); foc=T.c("border_focus")
        act_bg=T.c("bg_action"); act_hover=T.c("bg_action_hover"); act_txt=T.c("text_action")
        deep=T.c("bg_deep"); prog_bg=T.c("progress_bg"); prog_chunk=T.c("progress_chunk")

        self.setStyleSheet(f"QMainWindow{{background:{bg};}}")

        self._logo_lbl.setStyleSheet(
            f"font-weight:bold;font-size:13px;color:{acc};letter-spacing:0.5px;")

        self._search.setStyleSheet(
            f"QLineEdit{{background:{base};color:{txt};border:1px solid {bd};"
            f"padding:5px 8px;font-size:11px;border-radius:5px;}}"
            f"QLineEdit:focus{{border-color:{foc};}}")

        for lib, b in self._lib_btns.items():
            col = T.lib_btn_color(lib)
            b.setStyleSheet(
                f"QPushButton{{background:{col};color:#ffffff;border:none;"
                f"font-size:10px;font-weight:bold;border-radius:3px;padding:2px;}}"
                f"QPushButton:hover{{background:{col};}}")

        self._tree.setStyleSheet(
            f"QTreeWidget{{background:{base};border:1px solid {bd};font-size:11px;"
            f"color:{txt};border-radius:4px;}}"
            f"QTreeWidget::item{{padding:3px 4px;}}"
            f"QTreeWidget::item:hover{{background:{hover};}}"
            f"QTreeWidget::item:selected{{background:{sel};color:{acc};}}"
            f"QTreeWidget::branch{{background:{base};}}")
        self._count_lbl.setStyleSheet(f"color:{txt_m};font-size:10px;padding:2px;")

        grp_style = (
            f"QGroupBox{{font-size:10px;font-weight:bold;color:{txt2};"
            f"border:1px solid {bd};margin-top:12px;border-radius:5px;}}"
            f"QGroupBox::title{{left:8px;top:-7px;background:{bg_btn};"
            f"padding:0 5px;border-radius:3px;}}")
        for attr in ["_input_viewer","_output_viewer"]:
            grp = self.findChild(QGroupBox, f"grp_{attr}")
            if grp: grp.setStyleSheet(grp_style)
        for obj in [self._input_viewer, self._output_viewer,
                    self._input_hist, self._output_hist,
                    self._feat_display, self._param_panel,
                    self._pipeline_panel, self._batch_tab]:
            obj.apply_theme()

        self._process_btn.setStyleSheet(
            f"QPushButton{{background:{act_bg};color:{act_txt};font-weight:bold;"
            f"font-size:13px;border-radius:5px;border:none;letter-spacing:0.5px;}}"
            f"QPushButton:hover{{background:{act_hover};}}"
            f"QPushButton:disabled{{background:{deep};color:{txt_m};}}")
        self._auto_check.setStyleSheet(f"QCheckBox{{color:{txt2};font-size:11px;}}")
        for b in self._action_btns:
            b.setStyleSheet(
                f"QPushButton{{background:{bg_btn};color:{txt2};border:1px solid {bd};"
                f"padding:5px 10px;font-size:11px;border-radius:4px;}}"
                f"QPushButton:hover{{background:{hover};color:{txt};}}")

        self._metric_lbl.setStyleSheet(
            f"color:{T.c('metric_text')};font-family:monospace;font-size:11px;"
            f"padding:4px 6px;background:{T.c('metric_bg')};"
            f"border-top:1px solid {T.c('metric_border')};border-radius:0 0 4px 4px;")

        self._right_tabs.setStyleSheet(
            f"QTabWidget::pane{{border:1px solid {bd};background:{base};"
            f"border-radius:0 0 5px 5px;}}"
            f"QTabBar::tab{{background:{T.c('tab_bg')};color:{T.c('tab_text')};"
            f"border:1px solid {bd};padding:6px 14px;font-size:11px;"
            f"border-radius:4px 4px 0 0;}}"
            f"QTabBar::tab:selected{{background:{T.c('tab_selected_bg')};"
            f"color:{T.c('tab_selected_text')};border-bottom:none;font-weight:500;}}"
            f"QTabBar::tab:hover{{color:{acc};}}")

        self._algo_label.setStyleSheet(
            f"font-weight:bold;font-size:13px;color:{acc};padding:4px 2px;")
        self._lib_badge.setStyleSheet(f"font-size:10px;color:{txt2};padding:2px 4px;")
        self._algo_desc.setStyleSheet(f"color:{txt_m};font-size:10px;padding:2px 4px;")

        scroll = self._right_tabs.widget(0).findChild(QScrollArea)
        if scroll: scroll.setStyleSheet(f"QScrollArea{{border:none;background:{base};}}")

        io_grp = self._right_tabs.widget(0).findChild(QGroupBox, "io_grp")
        if io_grp: io_grp.setStyleSheet(grp_style)
        for b in self._io_buttons:
            b.setStyleSheet(
                f"QPushButton{{background:{bg_btn};color:{txt2};border:1px solid {bd};"
                f"padding:5px;font-size:11px;text-align:left;padding-left:8px;border-radius:4px;}}"
                f"QPushButton:hover{{background:{hover};color:{txt};}}")

        self.statusBar().setStyleSheet(
            f"QStatusBar{{color:{T.c('status_text')};font-size:10px;"
            f"background:{T.c('status_bg')};}}"
            f"QStatusBar::item{{border:none;}}")
        self._prog.setStyleSheet(
            f"QProgressBar{{background:{prog_bg};border:1px solid {bd};border-radius:3px;}}"
            f"QProgressBar::chunk{{background:{prog_chunk};border-radius:3px;}}")

        mb = self.menuBar()
        mb.setStyleSheet(
            f"QMenuBar{{background:{T.c('menubar_bg')};color:{T.c('menubar_text')};font-size:11px;}}"
            f"QMenuBar::item:selected{{background:{hover};}}"
            f"QMenu{{background:{T.c('menu_bg')};color:{T.c('menu_text')};"
            f"border:1px solid {bd};font-size:11px;}}"
            f"QMenu::item:selected{{background:{T.c('menu_selected')};}}")

        self._build_tree(filter_text=self._search.text())
        if self._current_spec:
            col = T.lib_color(self._current_spec.lib)
            self._lib_badge.setStyleSheet(f"font-size:10px;color:{col};padding:2px 4px;")

    # ─── Tree ────────────────────────────────────────────────────────

    def _build_tree(self, filter_text="", filter_lib="ALL"):
        self._tree.clear()
        for cat, subcats in sorted(REGISTRY.by_category().items()):
            cat_item = QTreeWidgetItem([cat])
            cat_item.setFont(0, QFont("",11,QFont.Weight.Bold))
            cat_item.setForeground(0, QColor(T.c("cat_text")))
            has_visible = False
            for subcat, specs in sorted(subcats.items()):
                sub_item = QTreeWidgetItem([subcat])
                sub_item.setForeground(0, QColor(T.c("subcat_text")))
                sub_item.setFont(0, QFont("",10)); has_sub = False
                for spec in sorted(specs, key=lambda s: s.label):
                    if filter_text and filter_text.lower() not in spec.label.lower() \
                       and filter_text.lower() not in spec.description.lower(): continue
                    if filter_lib != "ALL" and spec.lib.value != filter_lib: continue
                    tag = T.LIB_LABELS.get(spec.lib,"?"); col = T.lib_color(spec.lib)
                    leaf = QTreeWidgetItem([f"[{tag}] {spec.label}"])
                    leaf.setData(0, Qt.ItemDataRole.UserRole, spec.key)
                    leaf.setForeground(0, QColor(col))
                    sub_item.addChild(leaf); has_sub = True
                if has_sub: cat_item.addChild(sub_item); has_visible = True
            if has_visible: self._tree.addTopLevelItem(cat_item)
        expand = bool(filter_text or filter_lib != "ALL")
        if expand: self._tree.expandAll()
        else:
            for i in range(self._tree.topLevelItemCount()):
                self._tree.topLevelItem(i).setExpanded(True)

    def _filter_tree(self, text): self._build_tree(filter_text=text)
    def _filter_by_lib(self, lib): self._build_tree(filter_lib=lib)

    def _tree_context_menu(self, pos):
        item = self._tree.itemAt(pos)
        if not item: return
        key = item.data(0, Qt.ItemDataRole.UserRole)
        if not key: return
        menu = QMenu(self)
        menu.setStyleSheet(
            f"QMenu{{background:{T.c('menu_bg')};color:{T.c('menu_text')};"
            f"border:1px solid {T.c('border')};font-size:11px;border-radius:4px;}}"
            f"QMenu::item:selected{{background:{T.c('menu_selected')};}}")
        a1 = menu.addAction("▶ Process now")
        a2 = menu.addAction("➕ Add to pipeline")
        action = menu.exec(self._tree.viewport().mapToGlobal(pos))
        if action == a1: self._activate_algo(key); self._process()
        elif action == a2: self._pipeline_panel.add_algo(key)

    def _setup_menu(self):
        mb = self.menuBar()
        fm = mb.addMenu("File")
        for lbl,sc,fn in [("Open Image…","Ctrl+O",self._load_image),
                           ("Save Output…","Ctrl+S",self._save_image),
                           (None,None,None),("Exit","Ctrl+Q",self.close)]:
            if lbl is None: fm.addSeparator(); continue
            a = QAction(lbl,self); a.setShortcut(sc); a.triggered.connect(fn); fm.addAction(a)

        vm = mb.addMenu("View")
        a = QAction("Extract All Features",self); a.triggered.connect(self._extract_features); vm.addAction(a)
        a2 = QAction("A/B Compare",self); a2.triggered.connect(self._show_ab); vm.addAction(a2)
        vm.addSeparator()
        a3 = QAction("Toggle Light/Dark Mode",self); a3.setShortcut("Ctrl+Shift+T")
        a3.triggered.connect(lambda: (T.toggle(), self._theme_btn._update_label(), self._apply_theme_all()))
        vm.addAction(a3)

        pm = mb.addMenu("Pipeline")
        for lbl,fn in [("Run Pipeline",self._run_pipeline),
                        ("Save Pipeline…",self._save_pipeline),
                        ("Load Pipeline…",self._load_pipeline)]:
            a = QAction(lbl,self); a.triggered.connect(fn); pm.addAction(a)

        hm = mb.addMenu("Help")
        a = QAction("About TriVision",self); a.triggered.connect(self._about); hm.addAction(a)

    # ─── Algo ────────────────────────────────────────────────────────

    def _on_tree_click(self, item, col):
        key = item.data(0, Qt.ItemDataRole.UserRole)
        if not key: return
        self._activate_algo(key)
        if self._auto and self._input_img is not None: self._process()

    def _activate_algo(self, key):
        spec = REGISTRY.get(key)
        if spec is None: return
        self._current_spec = spec
        self._algo_label.setText(spec.label)
        col = T.lib_color(spec.lib)
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
        self._process_btn.setEnabled(False); self._prog.setVisible(True)
        self.statusBar().showMessage(f"Processing: {self._current_spec.label}…")
        self._worker = Worker(self._current_spec.fn, self._input_img, kwargs)
        self._worker.done.connect(self._on_done); self._worker.start()

    def _on_done(self, result, elapsed_ms, message):
        self._process_btn.setEnabled(True); self._prog.setVisible(False)
        if result is None: self.statusBar().showMessage(message); return
        if isinstance(result, dict):
            self._feat_display.show_features(result)
            self.statusBar().showMessage(f"{message}  •  {elapsed_ms:.0f}ms"); return
        self._output_viewer.set_image(result); self._output_hist.set_image(result)
        self._compute_metrics(result)
        h,w = result.shape[:2]
        self.statusBar().showMessage(
            f"{self._current_spec.label}  •  {message}  •  {w}x{h}  •  {elapsed_ms:.0f}ms")

    def _compute_metrics(self, result):
        inp = self._input_img
        if inp is None or not isinstance(result, np.ndarray): return
        try:
            def gray(x): return cv2.cvtColor(x,cv2.COLOR_BGR2GRAY) if len(x.shape)==3 else x
            g1 = gray(inp).astype(np.float64); g2 = gray(result).astype(np.float64)
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
            self._output_viewer.set_image(result); self._output_hist.set_image(result)
            self._compute_metrics(result)
        elif isinstance(result, dict): self._feat_display.show_features(result)
        self.statusBar().showMessage(f"Pipeline '{p.name}' complete  •  {len(p.nodes)} steps")

    def _save_pipeline(self):
        fn,_ = QFileDialog.getSaveFileName(self,"Save Pipeline","pipeline.json","JSON (*.json)")
        if fn: self._pipeline_panel.get_pipeline().save(fn)

    def _load_pipeline(self):
        fn,_ = QFileDialog.getOpenFileName(self,"Load Pipeline","","JSON (*.json)")
        if not fn: return
        p = Pipeline.load(fn); self._pipeline = p; self._pipeline_panel.set_pipeline(p)

    def _load_preset_pipeline(self, pipeline):
        self._pipeline = pipeline; self._pipeline_panel.set_pipeline(pipeline)

    # ─── I/O ──────────────────────────────────────────────────────────

    def _create_default_image(self):
        img = np.zeros((420, 640, 3), np.uint8)
        for i in range(0,640,40): cv2.line(img,(i,0),(i,420),(8,10,18),1)
        for j in range(0,420,40): cv2.line(img,(0,j),(640,j),(8,10,18),1)
        cv2.rectangle(img,(30,30),(220,200),(0,180,120),2)
        cv2.circle(img,(460,130),85,(180,80,220),-1)
        cv2.ellipse(img,(320,310),(135,65),20,0,360,(0,160,255),3)
        cv2.line(img,(30,370),(610,380),(220,200,0),3)
        cv2.putText(img,"TriVision",(155,265),cv2.FONT_HERSHEY_DUPLEX,1.6,(240,240,255),2)
        cv2.putText(img,"OpenCV · CVIPtools2 · scikit-image",(105,300),cv2.FONT_HERSHEY_SIMPLEX,0.6,(100,130,180),1)
        self._set_input(img)

    def _set_input(self, img):
        self._input_img = img; self._input_viewer.set_image(img); self._input_hist.set_image(img)
        h,w = img.shape[:2]; ch = img.shape[2] if len(img.shape)==3 else 1
        self.statusBar().showMessage(f"Input: {w}x{h}  ch={ch}")

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
            self.statusBar().showMessage("Showing x3 amplified diff")
        except Exception as e: QMessageBox.warning(self,"Diff",str(e))

    def _show_ab(self):
        inp = self._input_img; out = self._output_viewer.get_image()
        if inp is None or out is None:
            QMessageBox.information(self,"A/B","Need both input and output."); return
        bgr_in  = cv2.cvtColor(inp,cv2.COLOR_GRAY2BGR)  if len(inp.shape)==2 else inp
        bgr_out = cv2.cvtColor(out,cv2.COLOR_GRAY2BGR)  if len(out.shape)==2 else out
        h = max(bgr_in.shape[0],bgr_out.shape[0])
        a = cv2.resize(bgr_in,(bgr_in.shape[1],h)); b = cv2.resize(bgr_out,(bgr_out.shape[1],h))
        div = np.full((h,4,3),[40,160,255],np.uint8); combined = np.hstack([a,div,b])
        cv2.putText(combined,"INPUT",(6,22),cv2.FONT_HERSHEY_DUPLEX,0.6,(40,200,255),1)
        cv2.putText(combined,"OUTPUT",(a.shape[1]+10,22),cv2.FONT_HERSHEY_DUPLEX,0.6,(40,200,255),1)
        self._output_viewer.set_image(combined); self._output_hist.set_image(combined)

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
            if isinstance(result, dict): self._feat_display.show_features(result)

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
            "comprehensive feature extraction, plugin SDK, quality metrics.<br><br>"
            "<i>Press Ctrl+Shift+T or click ☀/🌙 to toggle light/dark mode.</i>")


# ═══════════════════════════════════════════════════════════════════════════════
# Launch
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    T.apply_palette(app)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
