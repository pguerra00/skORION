#!/usr/bin/env python3
"""Desktop analysis workspace. Launch with: python orion_gui.py"""
import argparse
from copy import deepcopy
from datetime import datetime
import json
import math
from pathlib import Path
import sys
import tempfile

try:
    from PySide6.QtCore import Qt, QProcess, QSize, QThread, QTimer, Signal, QUrl
    from PySide6.QtGui import QColor, QDesktopServices, QFontDatabase, QIcon, QImage, QPainter, QPixmap
    from PySide6.QtWidgets import (
        QApplication, QAbstractItemView, QCheckBox, QColorDialog, QComboBox, QDoubleSpinBox,
        QFileDialog, QFormLayout, QGraphicsScene, QGraphicsView, QGridLayout, QGroupBox,
        QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMainWindow, QMessageBox,
        QPlainTextEdit, QProgressBar, QPushButton, QScrollArea, QSizePolicy, QSpinBox, QSplitter,
        QTabBar, QTabWidget, QTableWidget, QTableWidgetItem, QToolButton, QTreeWidget,
        QTreeWidgetItem, QVBoxLayout, QWidget,
    )
except ImportError:
    raise SystemExit('The desktop GUI needs Qt. Install it with: python -m pip install -r requirements.txt')

try:
    import numpy as np
    import pandas as pd
    from gui_analysis import (FLOOR_KEYS, analyze_preview, default_display_colors, list_images,
                              render_image, validate_parameters, validate_run_paths)
    from image_io import inspect_image
    from gui_identity import apply_application_identity, prepare_taskbar
    from progress_reporting import ProgressStream, RunProgress, STAGE_LABELS
    from gui_theme import (APP_STYLE, TOKENS, DecoForm, DecoHeader, DecoImageFrame, DecoInspectionDock,
                           DecoPanel, DecoSurface, apply_theme, button_variant,
                           layout_metrics, text_label, tracked_font)
except ImportError as exc:
    raise SystemExit('Analysis dependencies are missing. Using Python 3.12, install '
                     'requirements.txt in your environment: python -m pip install -r requirements.txt') from exc

ROOT = Path(__file__).resolve().parent
BASIC_KEYS = {'group_names', 'nuclear_channel', 'foci_channels', 'nuclear.fixed_threshold_adu',
              'qc_output', 'rgb_channel_order', 'gui_display', 'colocalization'}
# These strings describe fixed algorithms; editing them would not change execution.
METHOD_KEYS = {
    'analysis_version', 'input_axis_order', 'analysis_unit', 'statistical_testing',
    'nuclear.threshold_rule', 'foci.background_method', 'foci.noise_method',
    'foci.threshold_rule', 'foci.post_watershed_rule', 'foci.contrast_filter',
    'foci.annulus_exclusions', 'foci.annulus_fallback', 'foci.nuclear_edge_rule',
    'foci.sensitivity_rule', 'qc.interpretation',
}


def label_for(key):
    text = key.replace('_um2', ' (µm²)').replace('_adu', ' (ADU)').replace('_px', ' (px)').replace('_', ' ')
    return text[:1].upper() + text[1:]


def set_path(parameters, path, value):
    obj = parameters
    for key in path[:-1]:
        obj = obj[key]
    obj[path[-1]] = value


class ParameterEditor(QTreeWidget):
    changed = Signal()

    def __init__(self):
        super().__init__()
        self.setHeaderLabels(['Parameter', 'Value'])
        self.setIndentation(12)
        self.setMinimumHeight(330)
        self.setColumnWidth(0, 170)
        self.header().setStretchLastSection(True)
        self.editors = {}

    def load(self, parameters):
        self.clear()
        self.editors.clear()

        def add(obj, parent=None, prefix=()):
            for key, value in obj.items():
                path = (*prefix, key)
                full = '.'.join(path)
                if full in BASIC_KEYS or (len(path) == 3 and path[0] == 'foci'
                                         and path[1].startswith('channel_') and path[2] in FLOOR_KEYS):
                    continue
                item = QTreeWidgetItem([label_for(key), ''])
                item.setToolTip(0, full)
                if parent is None:
                    self.addTopLevelItem(item)
                else:
                    parent.addChild(item)
                if isinstance(value, dict):
                    add(value, item, path)
                else:
                    if isinstance(value, bool):
                        editor = QCheckBox()
                        editor.setChecked(value)
                        editor.toggled.connect(self.changed)
                    else:
                        editor = QLineEdit(value if isinstance(value, str) else json.dumps(value))
                        editor.setReadOnly(full in METHOD_KEYS or path[0] == 'measurements')
                        editor.textEdited.connect(self.changed)
                    editor.setAccessibleName(full)
                    editor.setToolTip(full + (' · method description; read only' if
                        isinstance(editor, QLineEdit) and editor.isReadOnly() else ''))
                    self.setItemWidget(item, 1, editor)
                    item.setSizeHint(1, QSize(0, editor.sizeHint().height() + TOKENS.gap))
                    self.editors[path] = (editor, value)
        add(parameters)

    def apply(self, parameters):
        for path, (editor, original) in self.editors.items():
            if isinstance(editor, QCheckBox):
                value = editor.isChecked()
            elif isinstance(original, str):
                value = editor.text()
            else:
                try:
                    value = json.loads(editor.text(), parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s)))
                except (ValueError, json.JSONDecodeError):
                    raise ValueError(f'{".".join(path)} must contain a valid JSON value.') from None
                if isinstance(original, list) and not isinstance(value, list):
                    raise ValueError(f'{".".join(path)} must be a JSON list.')
                if type(original) in (int, float) and (type(value) not in (int, float)
                                                       or not math.isfinite(value)):
                    raise ValueError(f'{".".join(path)} must be a finite number.')
            set_path(parameters, path, value)


class ImageView(QGraphicsView):
    picked = Signal(int, int)

    def __init__(self):
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.setBackgroundBrush(QColor(TOKENS.image_background))
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setMinimumSize(240, 200)
        self.setAccessibleName('Image viewer: wheel to zoom, drag to pan, click to inspect a nucleus')
        self.pixmap = None
        self.press_point = None

    def drawForeground(self, painter, rect):
        if self.pixmap is not None:
            return
        painter.save()
        painter.resetTransform()
        bounds = self.viewport().rect()
        painter.setPen(QColor(TOKENS.gold))
        font = self.font()
        font.setFamily(TOKENS.heading_font)
        font.setPixelSize(20)
        painter.setFont(font)
        painter.drawText(bounds.adjusted(12, -16, -12, -16), Qt.AlignmentFlag.AlignCenter,
                         'YOUR FIELD OF VIEW')
        font.setFamily(TOKENS.body_font)
        font.setPixelSize(13)
        painter.setFont(font)
        painter.setPen(QColor(TOKENS.muted))
        painter.drawText(bounds.adjusted(12, 35, -12, 35), Qt.AlignmentFlag.AlignCenter,
                         'Load TIFF images to begin')
        painter.restore()

    def show_rgb(self, rgb, metadata, fit=False):
        h, w = rgb.shape[:2]
        image = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(image)
        # Draw a calibrated scale bar in image coordinates, not invented units.
        sx = metadata['pixel_size_x_um']
        choices = [v for v in (.1, .2, .5, 1, 2, 5, 10, 20, 50, 100, 200, 500) if v / sx <= w * .2]
        if choices and w >= 128 and h >= 80:
            length = choices[-1]
            painter = QPainter(pixmap)
            font = painter.font()
            font.setPixelSize(max(11, int(w / 80)))
            painter.setFont(font)
            painter.fillRect(8, h - 42, int(length / sx) + 20, 36, QColor(8, 15, 29, 210))
            painter.setPen(QColor('#ffffff'))
            painter.drawLine(16, h - 30, 16 + int(length / sx), h - 30)
            painter.drawText(16, h - 14, f'{length:g} µm')
            painter.end()
        if self.pixmap is None:
            self.pixmap = self.scene().addPixmap(pixmap)
            fit = True
        else:
            self.pixmap.setPixmap(pixmap)
        self.scene().setSceneRect(0, 0, w, h)
        if fit:
            self.fit()

    def fit(self):
        if self.pixmap is not None:
            self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def wheelEvent(self, event):
        self.scale(1.2 if event.angleDelta().y() > 0 else 1 / 1.2,
                   1.2 if event.angleDelta().y() > 0 else 1 / 1.2)
        event.accept()

    def mousePressEvent(self, event):
        self.press_point = event.position().toPoint()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if (event.button() == Qt.MouseButton.LeftButton and self.press_point is not None
                and (event.position().toPoint() - self.press_point).manhattanLength() < 5):
            point = self.mapToScene(event.position().toPoint())
            self.picked.emit(int(point.x()), int(point.y()))
        super().mouseReleaseEvent(event)


class PreviewWorker(QThread):
    completed = Signal(object)
    failed = Signal(str)

    def __init__(self, path, parameters):
        super().__init__()
        self.path, self.parameters = path, deepcopy(parameters)

    def run(self):
        try:
            self.completed.emit(analyze_preview(self.path, self.parameters))
        except Exception as exc:
            self.failed.emit(f'{type(exc).__name__}: {exc}')


class OrionWindow(QMainWindow):
    def __init__(self, parameter_file=None, input_dir=None, output_dir=None):
        apply_theme(QApplication.instance())
        super().__init__()
        self.setWindowIcon(apply_application_identity(QApplication.instance()))
        self.setWindowTitle(' · scikitORION · ')
        self.resize(1320, 950)
        self.parameters = json.loads((ROOT / 'parameters.template.json').read_text(encoding='utf-8'))
        self.parameters.setdefault('rgb_channel_order', None)
        self.parameters.setdefault('pixel_size_um', None)
        self.images, self.preview, self.pixels, self.metadata = [], None, None, None
        self.selected = 0
        self.busy = False
        self.worker = None
        self.run_temp = None
        self.run_output = None
        self.table_origin = None
        self.layer_checks = {}
        self.signal_colors, self.outline_colors, self.color_buttons = {}, {}, {}
        self.foci_checks, self.floors = {}, {}
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.process.readyReadStandardOutput.connect(self.read_process_output)
        self.process.finished.connect(self.process_finished)
        self.process.errorOccurred.connect(self.process_error)
        self.process_stream = ProgressStream()
        self.run_progress = None
        self.progress_detail = ''
        self.preview_error = False
        self.cancel_requested = False
        self.stage = ''
        self.build_ui()
        self.load_controls()
        if parameter_file:
            self.parameter_dir.setText(str(Path(parameter_file).expanduser().resolve().parent))
            self.parameter_name.setText(Path(parameter_file).name)
            self.load_parameters()
        if input_dir:
            self.input_dir.setText(str(Path(input_dir).expanduser().resolve()))
        if output_dir:
            self.output_dir.setText(str(Path(output_dir).expanduser().resolve()))

    def build_ui(self):
        shell = DecoSurface()
        shell_layout = layout_metrics(QVBoxLayout(shell), margin=0, spacing=0)
        shell_layout.addWidget(DecoHeader())
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(1)
        shell_layout.addWidget(splitter, 1)
        self.setCentralWidget(shell)
        left = QWidget()
        left.setObjectName('setupColumn')
        left_layout = layout_metrics(QVBoxLayout(left))
        self.sidebar = QWidget()
        self.sidebar.setObjectName('setupFields')
        setup = layout_metrics(QVBoxLayout(self.sidebar), margin=0, spacing=TOKENS.space)
        directories = DecoPanel('I · Directories')
        paths = DecoForm(directories)
        self.input_dir = self.path_field(paths, 'Input directory', '')
        default_output = self.input_dir.text() or Path.home()
        self.output_dir = self.path_field(paths, 'Output directory', str(default_output))
        self.parameter_dir = self.path_field(paths, 'Parameters directory', self.input_dir.text() or str(default_output))
        self.parameter_name = QLineEdit('parameters.json')
        paths.addRow('Parameter file', self.parameter_name)
        self.parameter_name.setAccessibleName('Parameter file name')
        parameter_buttons = QHBoxLayout()
        self.add_button(parameter_buttons, 'Load JSON', self.load_parameters)
        self.add_button(parameter_buttons, 'Save JSON…', self.save_parameters)
        paths.addRow(parameter_buttons)
        self.add_button(paths, 'Load images', self.load_images)
        setup.addWidget(directories)
        basic = DecoPanel('II · Basic parameters')
        form = DecoForm(basic)
        self.groups = QLineEdit()
        self.groups.setPlaceholderText('Control, Treatment')
        form.addRow('Group names', self.groups)
        self.groups.textEdited.connect(self.invalidate_preview)
        group_help = text_label('Comma-separated names matched to filenames. Leave blank to use the existing filename parser.')
        form.addRow(group_help)
        self.rgb_enabled = QCheckBox('Map RGB TIFF colors to channels')
        self.rgb_enabled.setAccessibleName('Enable RGB TIFF channel mapping')
        self.rgb_enabled.setToolTip('Required only for RGB TIFF exports. Channel stacks keep their original channel order.')
        form.addRow(self.rgb_enabled)
        self.rgb_container = QWidget()
        rgb_form = DecoForm(self.rgb_container)
        rgb_form.setContentsMargins(0, 0, 0, 0)
        self.rgb_colors = []
        for channel in range(1, 4):
            color = QComboBox()
            color.addItem('Choose color…', None)
            for name in ('Red', 'Green', 'Blue'):
                color.addItem(name, name.lower())
            color.setAccessibleName(f'Channel {channel} RGB color')
            color.currentIndexChanged.connect(self.rgb_changed)
            rgb_form.addRow(f'Channel {channel}', color)
            self.rgb_colors.append(color)
        self.rgb_help = text_label('For RGB exports, assign red, green, and blue once each. '
                          'Choose nuclear and foci roles independently. '
                          'For example: Channel 1 = Blue, Channel 2 = Green, Channel 3 = Red. '
                          'Channel stacks use their original channel order.')
        form.addRow(self.rgb_container)
        form.addRow(self.rgb_help)
        self.rgb_enabled.toggled.connect(self.rgb_changed)
        self.nuclear = QComboBox()
        self.nuclear.currentIndexChanged.connect(self.channel_changed)
        form.addRow('Nuclear channel', self.nuclear)
        self.foci_container = QWidget()
        self.foci_layout = QVBoxLayout(self.foci_container)
        self.foci_layout.setContentsMargins(0, 0, 0, 0)
        form.addRow('Foci channels', self.foci_container)
        self.coloc_enabled = QCheckBox('Analyze colocalization')
        self.coloc_enabled.setAccessibleName('Analyze colocalization')
        form.addRow(self.coloc_enabled)
        self.coloc_container = QWidget()
        coloc_form = DecoForm(self.coloc_container)
        coloc_form.setContentsMargins(0, 0, 0, 0)
        self.coloc_a, self.coloc_b = QComboBox(), QComboBox()
        for name, combo in [('Marker A channel', self.coloc_a), ('Marker B channel', self.coloc_b)]:
            combo.setAccessibleName(name)
            combo.currentIndexChanged.connect(self.invalidate_preview)
            coloc_form.addRow(name, combo)
        self.coloc_minimum = QSpinBox()
        self.coloc_minimum.setRange(1, 2147483647)
        self.coloc_minimum.setValue(1)
        self.coloc_minimum.setAccessibleName('Minimum shared pixels')
        self.coloc_minimum.valueChanged.connect(self.invalidate_preview)
        coloc_form.addRow('Minimum shared pixels', self.coloc_minimum)
        coloc_form.addRow(text_label('Shared pixels establish an association. All qualifying partners are retained. '
                                     'In the Colocalization view, bright objects have partners; dim objects do not. Yellow marks qualifying shared pixels.'))
        form.addRow(self.coloc_container)
        self.coloc_container.hide()
        self.coloc_enabled.toggled.connect(self.colocalization_changed)
        self.threshold = QComboBox()
        self.threshold.addItems(['Automatic · median / MAD', 'Manual · ADU range'])
        form.addRow('Nuclear thresholding', self.threshold)
        self.lower, self.upper = self.number(), self.number()
        self.lower.setAccessibleName('Lower nuclear ADU')
        self.upper.setAccessibleName('Upper nuclear ADU')
        bounds = QHBoxLayout()
        bounds.addWidget(QLabel('Lower'))
        bounds.addWidget(self.lower)
        bounds.addWidget(QLabel('Upper'))
        bounds.addWidget(self.upper)
        form.addRow('Range · ADU', bounds)
        self.threshold_help = QLabel()
        self.threshold_help.setWordWrap(True)
        form.addRow(self.threshold_help)
        self.threshold.currentIndexChanged.connect(self.threshold_changed)
        self.lower.valueChanged.connect(self.invalidate_preview)
        self.upper.valueChanged.connect(self.invalidate_preview)
        self.floor_container = QWidget()
        self.floor_layout = QVBoxLayout(self.floor_container)
        self.floor_layout.setContentsMargins(0, 0, 0, 0)
        form.addRow(self.floor_container)
        floors_help = text_label('ADU floors are stored per channel. New channels start at 0 ADU; noise-based thresholds still apply.')
        form.addRow(floors_help)
        self.qc_mode = QComboBox()
        self.qc_mode.addItems(['minimal', 'detailed', 'off'])
        form.addRow('QC output', self.qc_mode)
        setup.addWidget(basic)
        self.advanced_toggle = QToolButton()
        self.advanced_toggle.setText('III · Advanced parameters')
        self.advanced_toggle.setAccessibleName('Advanced parameters')
        self.advanced_toggle.setCheckable(True)
        self.advanced_toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.advanced_toggle.setArrowType(Qt.ArrowType.RightArrow)
        setup.addWidget(self.advanced_toggle)
        self.advanced = ParameterEditor()
        self.advanced.hide()
        self.advanced.changed.connect(self.invalidate_preview)
        self.advanced_toggle.toggled.connect(self.toggle_advanced)
        setup.addWidget(self.advanced)
        setup.addStretch()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.sidebar)
        scroll.setMinimumWidth(340)
        left_layout.addWidget(scroll)
        actions = QGridLayout()
        actions.setSpacing(TOKENS.gap)
        self.run_button = button_variant(QPushButton('Run analysis'), 'primary')
        self.run_button.clicked.connect(self.start_analysis)
        self.stop_button = button_variant(QPushButton('Stop'), 'quiet')
        self.stop_button.clicked.connect(self.stop_analysis)
        self.preview_button = button_variant(QPushButton('Preview image'))
        self.preview_button.clicked.connect(self.start_preview)
        actions.addWidget(self.run_button, 0, 0)
        actions.addWidget(self.stop_button, 0, 1)
        actions.addWidget(self.preview_button, 1, 0, 1, 2)
        actions.setColumnStretch(0, 1)
        self.stop_button.setEnabled(False)
        left_layout.addLayout(actions)
        splitter.addWidget(left)
        workspace = DecoSurface(patterned=True)
        layout = layout_metrics(QVBoxLayout(workspace))
        workspace_title = QHBoxLayout()
        workspace_title.addWidget(text_label('Observation room', 'heading'))
        workspace_title.addStretch()
        self.image_info = text_label('Load TIFF images to begin', 'metadata')
        self.image_info.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        workspace_title.addWidget(self.image_info)
        layout.addLayout(workspace_title)
        image_row = QHBoxLayout()
        image_label = text_label('Image', 'fieldLabel')
        image_row.addWidget(image_label)
        self.image_picker = QComboBox()
        self.image_picker.setMinimumWidth(150)
        self.image_picker.currentIndexChanged.connect(self.image_changed)
        image_label.setBuddy(self.image_picker)
        image_row.addWidget(self.image_picker, 1)
        button_variant(self.add_button(image_row, '−', lambda: self.viewer.scale(1 / 1.25, 1 / 1.25)), 'tool').setAccessibleName('Zoom out')
        button_variant(self.add_button(image_row, 'Fit', lambda: self.viewer.fit()), 'tool')
        button_variant(self.add_button(image_row, '+', lambda: self.viewer.scale(1.25, 1.25)), 'tool').setAccessibleName('Zoom in')
        layout.addLayout(image_row)
        self.view_tabs = QTabBar()
        for name in ('Raw image', 'Nuclei', 'Foci', 'Overlay'):
            self.view_tabs.addTab(name)
        self.view_tabs.setCurrentIndex(3)
        self.view_tabs.setAccessibleName('Image view tabs')
        self.view_tabs.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        tracked_font(self.view_tabs, .7)
        self.view_tabs.currentChanged.connect(self.draw)
        layout.addWidget(self.view_tabs)
        self.raw_controls = QWidget()
        view_row = layout_metrics(QHBoxLayout(self.raw_controls), margin=0)
        raw_label = text_label('Display channel', 'fieldLabel')
        view_row.addWidget(raw_label)
        self.raw_channel = QComboBox()
        self.raw_channel.setAccessibleName('Display channel')
        self.raw_channel.setEnabled(False)
        self.raw_channel.currentIndexChanged.connect(self.draw)
        raw_label.setBuddy(self.raw_channel)
        view_row.addWidget(self.raw_channel)
        view_row.addStretch()
        self.raw_controls.hide()
        layout.addWidget(self.raw_controls)
        self.viewer = ImageView()
        self.viewer.picked.connect(self.pick_nucleus)
        layout.addWidget(DecoImageFrame(self.viewer), 1)
        self.layers = QWidget()
        self.layers_layout = QHBoxLayout(self.layers)
        self.layers_layout.setContentsMargins(0, 0, 0, 0)
        self.colors_toggle = QToolButton()
        self.colors_toggle.setText('Channel colors')
        self.colors_toggle.setCheckable(True)
        self.colors_toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.colors_toggle.setArrowType(Qt.ArrowType.RightArrow)
        layer_row = QHBoxLayout()
        layer_row.addWidget(self.layers, 1)
        layer_row.addWidget(self.colors_toggle)
        layout.addLayout(layer_row)
        self.colors_panel = QWidget()
        self.colors_layout = QGridLayout(self.colors_panel)
        self.colors_layout.setContentsMargins(0, 0, 0, 0)
        self.colors_panel.hide()
        self.colors_toggle.toggled.connect(self.toggle_colors)
        dock = QWidget()
        dock_layout = layout_metrics(QVBoxLayout(dock), margin=0)
        dock_layout.addWidget(self.colors_panel)
        self.inspection_dock = DecoInspectionDock(dock)
        self.tabs = QTabWidget()
        self.tabs.setMaximumHeight(200)
        self.inspector = QWidget()
        inspection = QVBoxLayout(self.inspector)
        picker = QHBoxLayout()
        picker.addWidget(QLabel('Inspect nucleus'))
        self.nucleus_picker = QComboBox()
        self.nucleus_picker.currentIndexChanged.connect(self.select_nucleus)
        picker.addWidget(self.nucleus_picker, 1)
        inspection.addLayout(picker)
        self.nucleus_info = QLabel('Preview an image to inspect nuclei, foci, and applied thresholds.')
        self.nucleus_info.setWordWrap(True)
        self.nucleus_info.setTextFormat(Qt.TextFormat.PlainText)
        self.nucleus_info.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.nucleus_info.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.nucleus_scroll = QScrollArea()
        self.nucleus_scroll.setWidgetResizable(True)
        self.nucleus_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self.nucleus_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.nucleus_scroll.setWidget(self.nucleus_info)
        inspection.addWidget(self.nucleus_scroll, 1)
        self.tabs.addTab(self.inspector, 'Nucleus inspector')
        results = QWidget()
        result_layout = QVBoxLayout(results)
        self.table_label = QLabel('No measurements yet')
        result_layout.addWidget(self.table_label)
        self.table = QTableWidget()
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.setShowGrid(False)
        result_layout.addWidget(self.table)
        self.tabs.addTab(results, 'Results')
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
        self.log.setMaximumBlockCount(10000)
        self.tabs.addTab(self.log, 'Analysis log')
        dock_layout.addWidget(self.tabs)
        layout.addWidget(self.inspection_dock)
        bottom = QHBoxLayout()
        progress_rows = QVBoxLayout()
        progress_rows.setSpacing(4)
        self.progress = QProgressBar()
        self.stage_progress = QProgressBar()
        self.overall_label = text_label('Overall', 'metadata')
        self.stage_label = text_label('Current stage', 'metadata')
        self.overall_percent = text_label('0%', 'metadata')
        self.stage_percent = text_label('0%', 'metadata')
        for bar, name, value in [(self.progress, self.overall_label, self.overall_percent),
                                 (self.stage_progress, self.stage_label, self.stage_percent)]:
            row = QHBoxLayout()
            row.setSpacing(TOKENS.gap)
            name.setWordWrap(False)
            name.setFixedWidth(112)
            value.setWordWrap(False)
            value.setFixedWidth(36)
            value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            bar.setRange(0, 100)
            bar.setProperty('variant', 'compact')
            bar.setValue(0)
            bar.setTextVisible(False)
            bar.setFixedHeight(6)
            bar.setMinimumWidth(60)
            row.addWidget(name)
            row.addWidget(bar, 1)
            row.addWidget(value)
            progress_rows.addLayout(row)
        self.progress.setAccessibleName('Overall analysis progress')
        self.stage_progress.setAccessibleName('Current stage progress')
        bottom.addLayout(progress_rows, 1)
        button_variant(self.add_button(bottom, 'Load saved results', self.load_saved_results), 'quiet')
        self.report_button = button_variant(self.add_button(bottom, 'Open report', self.open_report), 'quiet')
        self.report_button.setEnabled(False)
        layout.addLayout(bottom)
        splitter.addWidget(workspace)
        splitter.setSizes([360, 960])
        splitter.setStretchFactor(1, 1)
        self.statusBar().showMessage('Ready · previews do not write output files')
        self.input_dir.textChanged.connect(self.input_changed)

    @staticmethod
    def add_button(layout, text, callback):
        button = button_variant(QPushButton(text))
        button.clicked.connect(callback)
        if isinstance(layout, QFormLayout):
            layout.addRow(button)
        else:
            layout.addWidget(button)
        return button

    def path_field(self, form, text, value):
        row = QHBoxLayout()
        field = QLineEdit(value)
        field.setAccessibleName(text)
        field.setMinimumWidth(120)
        row.addWidget(field, 1)
        browse = self.add_button(row, '…', lambda: self.browse_directory(field))
        button_variant(browse, 'icon')
        browse.setText('')
        browse.setIcon(QIcon(str(ROOT / 'assets/icons/folder.svg')))
        browse.setFixedWidth(44)
        browse.setAccessibleName('Browse ' + text.lower())
        form.addRow(text, row)
        form.labelForField(row).setBuddy(field)
        return field

    @staticmethod
    def number(value=0):
        field = QDoubleSpinBox()
        field.setDecimals(4)
        field.setRange(0, 1e12)
        field.setValue(value)
        field.setKeyboardTracking(False)
        field.setMinimumWidth(70)
        field.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        return field

    @staticmethod
    def clear_layout(layout):
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                OrionWindow.clear_layout(item.layout())

    def browse_directory(self, field):
        chosen = QFileDialog.getExistingDirectory(self, 'Choose directory', field.text())
        if chosen:
            field.setText(chosen)

    def parameter_path(self):
        name = self.parameter_name.text().strip()
        if not name or Path(name).name != name or not name.lower().endswith('.json'):
            raise ValueError('Parameter file must be a JSON filename, without a directory path.')
        if not self.parameter_dir.text().strip():
            raise ValueError('Choose a parameters directory.')
        return Path(self.parameter_dir.text()).expanduser().resolve() / name

    def load_controls(self):
        p = self.parameters
        self.signal_colors, self.outline_colors = self.read_display_colors(p)
        self.groups.setText(', '.join(p.get('group_names', [])))
        order = p.get('rgb_channel_order')
        self.rgb_enabled.blockSignals(True)
        self.rgb_enabled.setChecked(order is not None)
        self.rgb_enabled.blockSignals(False)
        for index, color in enumerate(self.rgb_colors):
            color.blockSignals(True)
            color.setCurrentIndex(color.findData(order[index] if order else None))
            color.blockSignals(False)
        self.sync_rgb_controls()
        bounds = p['nuclear'].get('fixed_threshold_adu')
        self.threshold.blockSignals(True)
        self.threshold.setCurrentIndex(0 if bounds is None else 1)
        self.threshold.blockSignals(False)
        self.lower.blockSignals(True)
        self.upper.blockSignals(True)
        self.lower.setValue(bounds[0] if bounds else 0)
        self.upper.setValue(bounds[1] if bounds else 65535)
        self.lower.blockSignals(False)
        self.upper.blockSignals(False)
        self.threshold_changed(invalidate=False)
        self.qc_mode.setCurrentText(p.get('qc_output', 'minimal'))
        count = max(3, p.get('nuclear_channel', 1), *p.get('foci_channels', [2, 3]),
                    *self.signal_colors, *self.outline_colors,
                    *[int(k) for k in p.get('channels', {}) if k.isdecimal()],
                    *[int(k[8:]) for k in p['foci'] if k.startswith('channel_') and k[8:].isdecimal()])
        self.build_channels(count)
        coloc = p.get('colocalization', {})
        self.sync_colocalization_channels(coloc.get('channels', [2, 3]))
        self.coloc_minimum.setValue(coloc.get('minimum_overlap_pixels', 1))
        self.coloc_enabled.blockSignals(True)
        self.coloc_enabled.setChecked(coloc.get('enabled', False))
        self.coloc_enabled.blockSignals(False)
        self.colocalization_changed(invalidate=False)
        self.advanced.load(p)

    def build_channels(self, count):
        p = self.parameters
        self.nuclear.blockSignals(True)
        self.nuclear.clear()
        self.clear_layout(self.foci_layout)
        self.clear_layout(self.floor_layout)
        self.foci_checks, self.floors = {}, {}
        for channel in range(1, count + 1):
            self.nuclear.addItem(f'Channel {channel}', channel)
            check = QCheckBox(f'Channel {channel}')
            check.setChecked(channel in p['foci_channels'])
            check.toggled.connect(self.channel_changed)
            self.foci_layout.addWidget(check)
            self.foci_checks[channel] = check
            box = DecoPanel(f'Channel {channel} · foci ADU floors')
            form = DecoForm(box)
            values = p['foci'].get(f'channel_{channel}', {})
            controls = {}
            for key in FLOOR_KEYS:
                spin = self.number(values.get(key, 0))
                spin.setAccessibleName(f'Channel {channel} {label_for(key)}')
                spin.valueChanged.connect(self.invalidate_preview)
                form.addRow(label_for(key).replace(' (ADU)', '').replace(' floor', ''), spin)
                controls[key] = spin
            self.floor_layout.addWidget(box)
            box.setVisible(check.isChecked())
            self.floors[channel] = (box, controls)
        self.nuclear.setCurrentIndex(self.nuclear.findData(p['nuclear_channel']))
        self.nuclear.blockSignals(False)
        self.rebuild_layers()
        self.rebuild_color_controls()
        self.sync_colocalization_channels()

    def sync_colocalization_channels(self, choices=None):
        if choices is None:
            choices = [combo.currentData() for combo in (self.coloc_a, self.coloc_b)]
        selected = [c for c, check in self.foci_checks.items() if check.isChecked()]
        for combo, channel in zip((self.coloc_a, self.coloc_b), choices):
            combo.blockSignals(True)
            combo.clear()
            combo.addItem('Choose channel…', None)
            for c in selected:
                combo.addItem(f'Channel {c}', c)
            combo.setCurrentIndex(max(0, combo.findData(channel)))
            combo.blockSignals(False)

    def colocalization_changed(self, *_, invalidate=True):
        enabled = self.coloc_enabled.isChecked()
        self.coloc_container.setVisible(enabled)
        index = next((i for i in range(self.view_tabs.count())
                      if self.view_tabs.tabText(i) == 'Colocalization'), -1)
        if enabled and index < 0:
            self.view_tabs.addTab('Colocalization')
        elif not enabled and index >= 0:
            if self.view_tabs.currentIndex() == index:
                self.view_tabs.setCurrentIndex(3)
            self.view_tabs.removeTab(index)
        if invalidate:
            self.invalidate_preview()

    @staticmethod
    def read_display_colors(parameters):
        settings = parameters.get('gui_display', {})
        if not isinstance(settings, dict) or not isinstance(settings.get('channels', {}), dict):
            raise ValueError('gui_display must contain a channels object.')
        signals, outlines = {}, {}
        for channel, values in settings.get('channels', {}).items():
            if not isinstance(channel, str) or not channel.isdecimal() or int(channel) < 1 or not isinstance(values, dict):
                raise ValueError('Display colors must use positive channel numbers and color objects.')
            for key, palette in [('signal', signals), ('outline', outlines)]:
                if key in values:
                    value = values[key]
                    if not isinstance(value, str) or not QColor(value).isValid():
                        raise ValueError(f'Channel {channel} {key} color must be a valid color name or hex value.')
                    palette[int(channel)] = QColor(value).name()
        return signals, outlines

    def rebuild_color_controls(self):
        self.clear_layout(self.colors_layout)
        self.color_buttons = {}
        for column, name in enumerate(('Channel', 'Signal / mask', 'Outline')):
            self.colors_layout.addWidget(QLabel(name), 0, column)
        n = self.nuclear.currentData()
        for channel in self.foci_checks:
            signal, outline = default_display_colors(channel, n)
            self.signal_colors.setdefault(channel, QColor(*signal).name())
            self.outline_colors.setdefault(channel, QColor(*outline).name())
            self.colors_layout.addWidget(QLabel(f'Channel {channel}'), channel, 0)
            for column, kind in enumerate(('signal', 'outline'), start=1):
                button = button_variant(QPushButton(), 'color')
                button.setAccessibleName(f'Channel {channel} {kind} color')
                button.clicked.connect(lambda _=False, c=channel, k=kind: self.choose_display_color(c, k))
                self.colors_layout.addWidget(button, channel, column)
                self.color_buttons[channel, kind] = button
                self.update_color_button(channel, kind)
        reset = button_variant(QPushButton('Reset colors'), 'quiet')
        reset.clicked.connect(self.reset_display_colors)
        self.colors_layout.addWidget(reset, len(self.foci_checks) + 1, 0)
        help_text = text_label('Display only · colors stay with their channel and are saved with Save JSON…')
        self.colors_layout.addWidget(help_text, len(self.foci_checks) + 1, 1, 1, 2)

    def toggle_colors(self, checked):
        self.colors_panel.setVisible(checked)
        self.colors_toggle.setArrowType(Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow)
        if checked:
            self.inspection_dock.verticalScrollBar().setValue(0)

    def update_color_button(self, channel, kind):
        palette = self.signal_colors if kind == 'signal' else self.outline_colors
        color = QColor(palette[channel])
        swatch = QPixmap(18, 18)
        swatch.fill(color)
        button = self.color_buttons[channel, kind]
        button.setIcon(QIcon(swatch))
        button.setText(color.name().upper())

    def choose_display_color(self, channel, kind):
        palette = self.signal_colors if kind == 'signal' else self.outline_colors
        color = QColorDialog.getColor(QColor(palette[channel]), self, f'Channel {channel} · {kind} color')
        if color.isValid():
            self.set_display_color(channel, kind, color)

    def set_display_color(self, channel, kind, color):
        color = QColor(color)
        if not color.isValid() or kind not in {'signal', 'outline'}:
            raise ValueError('Choose a valid signal or outline color.')
        palette = self.signal_colors if kind == 'signal' else self.outline_colors
        palette[channel] = color.name()
        if (channel, kind) in self.color_buttons:
            self.update_color_button(channel, kind)
        self.draw()

    def reset_display_colors(self):
        self.signal_colors, self.outline_colors = {}, {}
        self.rebuild_color_controls()
        self.draw()

    def channel_changed(self, *_):
        for channel, (box, _) in self.floors.items():
            box.setVisible(self.foci_checks[channel].isChecked())
        self.rebuild_layers()
        self.sync_colocalization_channels()
        self.invalidate_preview()

    def rgb_changed(self, *_):
        self.sync_rgb_controls()
        # A color reassignment changes the source pixels for each channel.
        self.pixels, self.metadata = None, None
        self.viewer.scene().clear()
        self.viewer.pixmap = None
        self.invalidate_preview()
        self.statusBar().showMessage('RGB mapping changed · preview again to reload channel pixels')

    def sync_rgb_controls(self):
        enabled = self.rgb_enabled.isChecked()
        self.rgb_container.setEnabled(enabled)
        self.rgb_container.setVisible(enabled)
        self.rgb_help.setVisible(enabled)

    def prepare_input_parameters(self, paths):
        """Enable color mapping only when this input batch contains RGB exports.

        The temporary order is used for metadata inspection only. It never
        becomes a scientific channel assignment or reads/reorders image pixels.
        A user's saved color choices remain available when the mapping is off.
        """
        metadata = [inspect_image(path, {'rgb_channel_order': ['red', 'green', 'blue']},
                                  require_calibration=False) for path in paths]
        first_rgb = next((path for path, md in zip(paths, metadata)
                          if md['input_format'] == 'rgb_export'), None)
        self.rgb_enabled.setChecked(first_rgb is not None)
        if first_rgb is not None and set(color.currentData() for color in self.rgb_colors) != {'red', 'green', 'blue'}:
            self.inspect_input(first_rgb, {'rgb_channel_order': None})
        return self.collect_parameters(), metadata[0]

    def inspect_input(self, path, parameters):
        # A temporary order permits metadata inspection only; no pixels are
        # read or analyzed until the user supplies an explicit RGB mapping.
        inspection = dict(parameters)
        if inspection.get('rgb_channel_order') is None:
            inspection['rgb_channel_order'] = ['red', 'green', 'blue']
        md = inspect_image(path, inspection, require_calibration=False)
        if md['input_format'] == 'rgb_export' and parameters.get('rgb_channel_order') is None:
            self.rgb_enabled.setChecked(True)
            self.rgb_colors[0].setFocus()
            raise ValueError(f'{Path(path).name} is an RGB TIFF. In Basic parameters, '
                             'choose the RGB color for Channels 1, 2, and 3, using '
                             'red, green, and blue once each. Then preview or run again.')
        return md

    def rebuild_layers(self):
        self.clear_layout(self.layers_layout)
        self.layer_checks = {}
        n = self.nuclear.currentData()
        channels = list(dict.fromkeys([n, *[c for c, check in self.foci_checks.items() if check.isChecked()]]))
        for channel in channels:
            if channel is None:
                continue
            role = 'nuclear' if channel == n else 'foci'
            check = QCheckBox(f'Channel {channel} · {role}')
            check.setChecked(True)
            check.toggled.connect(self.draw)
            self.layers_layout.addWidget(check)
            self.layer_checks[channel] = check
        self.outlines = QCheckBox('Outlines')
        self.outlines.setChecked(True)
        self.outlines.toggled.connect(self.draw)
        self.layers_layout.addWidget(self.outlines)
        self.layers_layout.addStretch()
        self.raw_channel.blockSignals(True)
        self.raw_channel.clear()
        for channel in range(1, len(self.foci_checks) + 1):
            self.raw_channel.addItem(f'Channel {channel}', channel)
        self.raw_channel.setCurrentIndex(self.raw_channel.findData(n))
        self.raw_channel.blockSignals(False)

    def threshold_changed(self, *_, invalidate=True):
        manual = self.threshold.currentIndex() == 1
        self.lower.setEnabled(manual)
        self.upper.setEnabled(manual)
        self.threshold_help.setText('Inclusive ADU range on original pixels.' if manual else
                                    'Automatic cutoff estimated separately for each image.')
        if invalidate:
            self.invalidate_preview()

    def toggle_advanced(self, checked):
        self.advanced.setVisible(checked)
        self.advanced_toggle.setArrowType(Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow)

    def collect_parameters(self):
        p = deepcopy(self.parameters)
        self.advanced.apply(p)
        display = p.setdefault('gui_display', {})
        display['channels'] = {str(channel): {'signal': self.signal_colors[channel],
                                             'outline': self.outline_colors[channel]}
                               for channel in self.signal_colors}
        p['rgb_channel_order'] = None
        if self.rgb_enabled.isChecked():
            order = [color.currentData() for color in self.rgb_colors]
            if set(order) != {'red', 'green', 'blue'}:
                raise ValueError('RGB mapping: choose a color for Channels 1, 2, and 3 '
                                 'in Basic parameters. Use red, green, and blue once each.')
            p['rgb_channel_order'] = order
        p['group_names'] = [name.strip() for name in self.groups.text().split(',')] if self.groups.text().strip() else []
        p['nuclear_channel'] = self.nuclear.currentData()
        p['foci_channels'] = [c for c, check in self.foci_checks.items() if check.isChecked()]
        p['colocalization'] = {'enabled': self.coloc_enabled.isChecked(),
                               'channels': [self.coloc_a.currentData(), self.coloc_b.currentData()],
                               'minimum_overlap_pixels': self.coloc_minimum.value()}
        p['nuclear']['fixed_threshold_adu'] = ([self.lower.value(), self.upper.value()]
                                             if self.threshold.currentIndex() == 1 else None)
        for channel, (_, controls) in self.floors.items():
            key = f'channel_{channel}'
            if channel in p['foci_channels'] or key in p['foci']:
                values = p['foci'].setdefault(key, {})
                values.update({name: spin.value() for name, spin in controls.items()})
        p['qc_output'] = self.qc_mode.currentText()
        return validate_parameters(p)

    def load_parameters(self):
        if self.busy:
            return
        try:
            path = self.parameter_path()
            p = json.loads(path.read_text(encoding='utf-8'))
            if not isinstance(p, dict):
                raise ValueError('The parameter file must contain a JSON object.')
            p.setdefault('rgb_channel_order', None)
            p.setdefault('pixel_size_um', None)
            p = validate_parameters(p)
            self.read_display_colors(p)
            self.parameters = p
            self.invalidate_preview()
            self.pixels, self.metadata = None, None
            self.viewer.scene().clear()
            self.viewer.pixmap = None
            self.load_controls()
            self.statusBar().showMessage(f'Loaded settings: {path.name}')
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self.show_error(str(exc))

    def save_parameters(self):
        if self.busy:
            return
        try:
            p = self.collect_parameters()
            default = self.parameter_path()
            name, _ = QFileDialog.getSaveFileName(self, 'Save experiment parameters', str(default), 'JSON files (*.json)')
            if not name:
                return
            path = Path(name).resolve()
            if path.suffix.lower() != '.json':
                path = path.with_suffix('.json')
            if self.input_dir.text().strip():
                raw = Path(self.input_dir.text()).expanduser().resolve()
                if path == raw or raw in path.parents:
                    raise ValueError('Save parameter files outside the raw input directory.')
            path.parent.mkdir(parents=True, exist_ok=True)
            # Atomic replacement protects a prior configuration from interrupted writes.
            with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, suffix='.tmp', delete=False) as tmp:
                temporary = Path(tmp.name)
                json.dump(p, tmp, indent=2, allow_nan=False)
                tmp.write('\n')
            try:
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
            self.parameter_dir.setText(str(path.parent))
            self.parameter_name.setText(path.name)
            self.statusBar().showMessage(f'Saved parameters: {path}')
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self.show_error(str(exc))

    def load_images(self):
        if self.busy:
            return
        try:
            if not self.input_dir.text().strip():
                raise ValueError('Choose an input directory first.')
            paths = list_images(self.input_dir.text())
            p, md = self.prepare_input_parameters(paths)
            self.parameters = p
            self.images = paths
            self.build_channels(max(md['shape'][0], len(self.foci_checks)))
            self.image_picker.blockSignals(True)
            self.image_picker.clear()
            for path in paths:
                self.image_picker.addItem(path.name)
            self.image_picker.blockSignals(False)
            self.image_info.setText(f'{len(paths)} image(s) · {md["shape"][0]} channels')
            self.invalidate_preview()
            self.start_preview()
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self.show_error(str(exc))

    def image_changed(self, *_):
        if not self.busy and 0 <= self.image_picker.currentIndex() < len(self.images):
            self.pixels, self.metadata = None, None
            self.viewer.scene().clear()
            self.viewer.pixmap = None
            self.invalidate_preview()
            self.start_preview()

    def input_changed(self, *_):
        self.images = []
        self.image_picker.blockSignals(True)
        self.image_picker.clear()
        self.image_picker.blockSignals(False)
        self.pixels, self.metadata = None, None
        self.viewer.scene().clear()
        self.viewer.pixmap = None
        self.image_info.setText('Load TIFF images to begin')
        self.invalidate_preview()

    def invalidate_preview(self, *_):
        self.preview, self.selected = None, 0
        if hasattr(self, 'nucleus_picker'):
            self.nucleus_picker.blockSignals(True)
            self.nucleus_picker.clear()
            self.nucleus_picker.blockSignals(False)
            self.nucleus_info.setText('Settings changed. Preview this image to inspect updated masks and measurements.')
        if self.table_origin == 'preview':
            self.table.setRowCount(0)
            self.table_label.setText('Preview settings changed; refresh the preview')
        if self.pixels is not None:
            self.statusBar().showMessage('Settings changed · refresh the preview before inspecting masks')
            self.draw()

    def set_busy(self, busy, processing=False):
        self.busy = busy
        self.sidebar.setEnabled(not busy)
        self.image_picker.setEnabled(not busy)
        self.preview_button.setEnabled(not busy)
        self.run_button.setEnabled(not busy)
        self.stop_button.setEnabled(busy and processing)

    def update_run_progress(self):
        if self.run_progress is None:
            return
        self.progress.setRange(0, 100)
        self.stage_progress.setRange(0, 100)
        self.progress.setValue(self.run_progress.overall_percent)
        self.stage_progress.setValue(self.run_progress.stage_percent)
        self.overall_percent.setText(f'{self.progress.value()}%')
        self.stage_percent.setText(f'{self.stage_progress.value()}%')
        name = STAGE_LABELS[self.run_progress.stage]
        self.stage_label.setText(name)
        self.stage_progress.setAccessibleName(f'{name} progress')

    def start_preview(self):
        if self.busy:
            return
        if not self.images:
            self.load_images()
            return
        try:
            index = self.image_picker.currentIndex()
            if not 0 <= index < len(self.images):
                raise ValueError('Choose an image to preview.')
            p, _ = self.prepare_input_parameters(self.images)
            self.run_progress = None
            self.preview_error = False
            self.progress.setRange(0, 100)
            self.progress.setValue(0)
            self.overall_percent.setText('—')
            self.stage_label.setText('Previewing image')
            self.stage_progress.setAccessibleName('Image preview progress')
            self.stage_progress.setRange(0, 0)
            self.stage_percent.setText('—')
            self.set_busy(True)
            self.statusBar().showMessage(f'Previewing {self.images[index].name}…')
            self.worker = PreviewWorker(self.images[index], p)
            self.worker.completed.connect(self.preview_complete)
            self.worker.failed.connect(self.preview_failed)
            self.worker.finished.connect(self.preview_finished)
            self.worker.start()
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self.show_error(str(exc))

    def preview_finished(self):
        self.set_busy(False)
        self.stage_progress.setRange(0, 100)
        self.stage_progress.setValue(0 if self.preview_error else 100)
        self.stage_percent.setText('0%' if self.preview_error else '100%')
        self.stage_label.setText('Preview failed' if self.preview_error else 'Preview complete')
        self.worker.deleteLater()
        self.worker = None

    def preview_failed(self, message):
        self.preview_error = True
        self.show_error(message)

    def preview_complete(self, preview):
        self.preview = preview
        self.pixels, self.metadata = preview.pixels, preview.metadata
        self.selected = 0
        self.image_info.setText(f'{preview.pixels.shape[0]} channels · {preview.pixels.shape[2]} × {preview.pixels.shape[1]} px')
        self.nucleus_picker.blockSignals(True)
        self.nucleus_picker.clear()
        self.nucleus_picker.addItem('Select a nucleus', 0)
        for rec in preview.nuclear_audit:
            label = rec['nucleus_label']
            self.nucleus_picker.addItem(f'Nucleus {label}' + ('' if rec['included'] else ' · excluded'), label)
        self.nucleus_picker.blockSignals(False)
        self.nucleus_info.setText(f'{len(preview.cells)} usable nuclei · {len(preview.nuclear_audit) - len(preview.cells)} excluded.\n'
                                 'Click a nucleus in the image or select it above.')
        rows = [{key: value for key, value in r.items() if key not in
                 {'nuclear_mean_DAPI_adu', 'nuclear_median_DAPI_adu', 'DAPI_threshold_adu',
                  'DAPI_background_smoothed_adu', 'DAPI_background_smoothed_mad_adu'}} for r in preview.cells]
        self.show_table(pd.DataFrame(rows), f'Preview measurements · {preview.path.name} · no files saved', 'preview')
        self.draw(fit=True)
        message = f'Preview ready · nuclear Channel {preview.parameters["nuclear_channel"]} · {len(preview.cells)} usable nuclei'
        self.statusBar().showMessage(message)
        self.log.appendPlainText(message)

    def draw(self, *_, fit=False):
        mode = self.view_tabs.tabText(self.view_tabs.currentIndex())
        self.raw_channel.setEnabled(mode == 'Raw image')
        self.raw_controls.setVisible(mode == 'Raw image')
        if self.pixels is None:
            return
        # A previous image may have fewer channels than newly chosen controls.
        n = self.nuclear.currentData() or 1
        foci = [c for c, check in self.foci_checks.items() if check.isChecked()]
        if max([n, *foci]) > self.pixels.shape[0]:
            return
        p = dict(nuclear_channel=n, foci_channels=foci)
        if self.preview is None and mode in ('Nuclei', 'Foci', 'Colocalization'):
            rgb = np.zeros((*self.pixels.shape[1:], 3), dtype=np.uint8)
        else:
            raw_channel = self.raw_channel.currentData() or n
            if raw_channel > self.pixels.shape[0]:
                return
            rgb = render_image(self.pixels, p, self.preview, mode, raw_channel,
                               [c for c, check in self.layer_checks.items() if check.isChecked()],
                               self.outlines.isChecked(), self.selected,
                               {c: QColor(color).getRgb()[:3] for c, color in self.signal_colors.items()},
                               {c: QColor(color).getRgb()[:3] for c, color in self.outline_colors.items()})
        self.viewer.show_rgb(rgb, self.metadata, fit)

    def pick_nucleus(self, x, y):
        if self.preview is None:
            return
        mask = self.preview.candidates
        if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
            self.nucleus_picker.setCurrentIndex(self.nucleus_picker.findData(int(mask[y, x])))

    def select_nucleus(self, *_):
        if self.preview is None:
            return
        self.selected = self.nucleus_picker.currentData() or 0
        rec = next((r for r in self.preview.nuclear_audit if r['nucleus_label'] == self.selected), None)
        if rec is None:
            self.nucleus_info.setText('Select a nucleus to inspect its measurements.')
        else:
            lines = [f'Nucleus {self.selected} · {"included" if rec["included"] else "excluded"}',
                     f'Area: {rec["nuclear_area_um2"]:.3g} µm² · Nuclear Channel {self.preview.parameters["nuclear_channel"]}']
            if rec['exclusion_reason']:
                lines.append('Exclusion: ' + rec['exclusion_reason'])
            if rec['nuclear_shape_warning']:
                lines.append('Shape: ' + rec['nuclear_shape_warning'])
            lines.append(f'Nuclear cutoff: {rec["DAPI_threshold_adu"]:.4g} ADU' +
                         (f' to {rec["nuclear_threshold_upper_adu"]:g} ADU' if rec['nuclear_threshold_upper_adu'] is not None else ' · automatic'))
            for c in self.preview.parameters['foci_channels']:
                rows = [r for r in self.preview.foci if r['nucleus_label'] == self.selected and r['channel'] == c]
                line = f'Channel {c}: {len(rows)} foci'
                t = next((r for r in self.preview.thresholds if r['channel'] == c and r['nucleus_label'] == self.selected), None)
                if t:
                    line += ' · ADU support / seed / prominence: ' + ' / '.join(
                        f'{t[key]:.3g}' for key in ('support_threshold_adu', 'seed_threshold_adu', 'prominence_threshold_adu'))
                lines.append(line)
            if self.preview.colocalization is not None:
                table = self.preview.colocalization.nuclei
                rows = table[table.nucleus_label == self.selected]
                if len(rows):
                    r = rows.iloc[0]
                    lines.append(f'Colocalization: {int(r.pair_count)} relationships · '
                                 f'A {int(r.A_colocalized_foci)}/{int(r.A_total_foci)} · '
                                 f'B {int(r.B_colocalized_foci)}/{int(r.B_total_foci)}')
                    coefficient = lambda field: (f'{r[field]:.4g}' if np.isfinite(r[field]) else
                                                  f'undefined ({r[field+"_status"].replace("_", " ")})')
                    lines.append(f'Pearson raw / corrected: {coefficient("pearson_raw")} / {coefficient("pearson_corrected")}')
                    value_text = lambda key: f'{r[key]:.3g}' if np.isfinite(r[key]) else 'undefined'
                    for role in ('A', 'B'):
                        lines.append(f'{role}: mean partners (all / associated) '
                                     f'{value_text(role+"_mean_partner_count_all")} / '
                                     f'{value_text(role+"_mean_partner_count_colocalized")}; '
                                     f'mean coverage {value_text(role+"_mean_coverage_fraction_colocalized")}')
            self.nucleus_info.setText('\n'.join(lines))
        self.draw()

    def show_table(self, frame, caption, origin):
        self.table_origin = origin
        preferred = ['image_id', 'image_name', 'treatment_group', 'nucleus_id', 'nucleus_label',
                     'nuclear_channel', 'nuclear_area_um2']
        cols = [c for c in preferred if c in frame] + [c for c in frame if c not in preferred]
        visible = frame.loc[:, cols].head(2000)
        self.table.clear()
        self.table.setRowCount(len(visible))
        self.table.setColumnCount(len(cols))
        self.table.setHorizontalHeaderLabels(cols)
        for r, row in enumerate(visible.itertuples(index=False, name=None)):
            for c, value in enumerate(row):
                text = '' if pd.isna(value) else f'{value:.5g}' if isinstance(value, float) else str(value)
                self.table.setItem(r, c, QTableWidgetItem(text))
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.table_label.setText(caption + (f' · first 2,000 of {len(frame):,} rows; complete data in CSV' if len(frame) > 2000 else ''))

    def show_error(self, message):
        self.statusBar().showMessage(message)
        self.log.appendPlainText('Error: ' + message)
        self.tabs.setCurrentWidget(self.log)
        self.inspection_dock.ensureWidgetVisible(self.tabs)
        QMessageBox.warning(self, 'scikitORION', message)

    def start_analysis(self):
        if self.busy:
            return
        try:
            if not self.input_dir.text().strip() or not self.output_dir.text().strip():
                raise ValueError('Choose input and output directories.')
            paths = list_images(self.input_dir.text())
            source, output = validate_run_paths(self.input_dir.text(), self.output_dir.text())
            p, _ = self.prepare_input_parameters(paths)
            self.run_temp = tempfile.TemporaryDirectory(prefix='orion-gui-')
            snapshot = Path(self.run_temp.name) / 'parameters.json'
            snapshot.write_text(json.dumps(p, indent=2, allow_nan=False), encoding='utf-8')
            self.run_output = output
            self.run_parameters = deepcopy(p)
            self.cancel_requested = False
            self.run_progress = RunProgress(outputs=p['qc_output'] != 'off')
            self.update_run_progress()
            self.set_busy(True, processing=True)
            self.tabs.setCurrentWidget(self.log)
            self.log.appendPlainText(f'Analyzing {len(paths)} image(s)\nOutput: {output}\n'
                                     f'Nuclear: Channel {p["nuclear_channel"]}; foci: {p["foci_channels"]}\n'
                                     'Current settings are saved with this run; the loaded JSON is unchanged.')
            self.report_button.setEnabled(False)
            self.stage = 'analysis'
            # Keep each stage in its own QProcess so Stop also stops report
            # generation instead of leaving the CLI's renderer child running.
            self.start_process('analyze_foci.py', ['--input-dir', str(source), '--output-dir', str(output),
                                                 '--parameters', str(snapshot), '--qc-output', 'off'])
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self.finish_run(False)
            self.show_error(str(exc))

    def start_process(self, script, args):
        self.process_stream = ProgressStream()
        self.progress_detail = ''
        self.run_progress.begin(self.stage)
        self.update_run_progress()
        self.process.setWorkingDirectory(str(ROOT))
        self.process.start(sys.executable, ['-X', 'utf8', '-u', str(ROOT / script), *args, '--progress-json'])
        self.statusBar().showMessage({'analysis': 'Analyzing images…', 'outputs': 'Generating QC and reports…',
                                     'validation': 'Validating saved results…'}[self.stage])

    def read_process_output(self):
        self.consume_process_output(self.process_stream.feed(bytes(self.process.readAllStandardOutput())))

    def consume_process_output(self, records):
        ordinary = []
        for record, line in records:
            if record is None:
                ordinary.append(line)
            elif (self.run_progress is not None and not self.cancel_requested
                  and self.run_progress.accept(record)):
                self.progress_detail = record['detail']
                self.update_run_progress()
                self.stage_progress.setAccessibleDescription(self.progress_detail)
                self.statusBar().showMessage(self.progress_detail)
        if ordinary:
            self.log.moveCursor(self.log.textCursor().MoveOperation.End)
            self.log.insertPlainText(''.join(ordinary))
            self.log.ensureCursorVisible()
            lines = [line.strip() for line in ordinary if line.strip()]
            if lines and not self.progress_detail and not self.cancel_requested:
                self.statusBar().showMessage(lines[-1])

    def process_finished(self, code, exit_status):
        self.read_process_output()
        self.consume_process_output(self.process_stream.feed(final=True))
        if self.cancel_requested:
            self.log.appendPlainText('Analysis stopped. Partial files may remain in the output directory.')
            self.finish_run(False)
            self.statusBar().showMessage('Stopped · choose a new output directory before another run')
        elif code != 0 or exit_status != QProcess.ExitStatus.NormalExit:
            self.finish_run(False)
            self.show_error(f'{self.stage.capitalize()} failed (exit {code}). See the analysis log. Completed files are preserved.')
        else:
            self.run_progress.complete_stage()
            self.update_run_progress()
            if self.stage == 'analysis' and self.run_parameters['qc_output'] != 'off':
                self.stage = 'outputs'
                self.start_process('make_outputs.py', ['--output-dir', str(self.run_output),
                                                      '--qc-output', self.run_parameters['qc_output']])
            elif self.stage in ('analysis', 'outputs'):
                self.stage = 'validation'
                self.start_process('validate_results.py', ['--output-dir', str(self.run_output)])
            else:
                self.finish_run(True)
                self.load_saved_results(self.run_output)
                self.statusBar().showMessage(f'Analysis complete and validated · {self.run_output}')

    def process_error(self, error):
        if error == QProcess.ProcessError.FailedToStart:
            self.finish_run(False)
            self.show_error('Could not start the analysis process: ' + self.process.errorString())

    def finish_run(self, success):
        self.set_busy(False)
        if self.run_progress is not None and self.run_progress.active:
            self.run_progress.finish(success)
            self.update_run_progress()
        if self.run_temp:
            self.run_temp.cleanup()
            self.run_temp = None
        if self.run_output:
            self.report_button.setEnabled((self.run_output / 'Review.html').is_file())

    def stop_analysis(self):
        if not self.busy or self.process.state() == QProcess.ProcessState.NotRunning:
            return
        self.cancel_requested = True
        self.statusBar().showMessage(f'Stopping · {STAGE_LABELS.get(self.stage, self.stage)}')
        self.stop_button.setEnabled(False)
        self.process.terminate()
        QTimer.singleShot(2000, self.kill_cancelled_process)

    def kill_cancelled_process(self):
        if self.cancel_requested and self.process.state() != QProcess.ProcessState.NotRunning:
            self.process.kill()

    def load_saved_results(self, directory=None):
        # QPushButton.clicked supplies a bool, not a directory.
        if self.busy:
            return
        try:
            output = Path(directory if directory and not isinstance(directory, bool) else self.output_dir.text()).expanduser().resolve()
            frame = pd.read_csv(output / 'Results' / 'Cell_Summary.csv')
            parameters = json.loads((output / 'Methods' / 'Effective_Parameters.json').read_text(encoding='utf-8'))
            if parameters.get('colocalization', {}).get('enabled', False):
                coloc = pd.read_csv(output / 'Results' / 'Colocalization_Nuclei.csv')
                columns = ['nucleus_id'] + [c for c in coloc if c not in {
                    'image_id', 'image_name', 'treatment_group', 'group_names', 'replicate_id', 'nucleus_id', 'nucleus_label'}]
                coloc = coloc[columns].rename(columns={c: f'coloc_{c}' for c in columns if c != 'nucleus_id'})
                frame = frame.merge(coloc, on='nucleus_id', how='left', validate='one_to_one')
            channels = ', '.join(f'Channel {c}' for c in parameters['foci_channels'])
            self.show_table(frame, f'Saved run · {output.name} · nuclear Channel {parameters["nuclear_channel"]} · foci {channels}', 'saved')
            self.run_output = output
            self.report_button.setEnabled((output / 'Review.html').is_file())
            self.tabs.setCurrentIndex(1)
        except (OSError, ValueError, KeyError) as exc:
            self.show_error(str(exc))

    def open_report(self):
        if self.run_output and (self.run_output / 'Review.html').is_file():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.run_output / 'Review.html')))

    def closeEvent(self, event):
        if self.busy:
            event.ignore()
            self.statusBar().showMessage('Wait for the preview to finish, or stop the running analysis, before closing.')
        else:
            event.accept()


def main():
    parser = argparse.ArgumentParser(description='scikitORION desktop image analysis workspace')
    parser.add_argument('--parameters', type=Path, help='Load an experiment JSON file')
    parser.add_argument('--input-dir', type=Path, help='Initial input directory')
    parser.add_argument('--output-dir', type=Path, help='Initial output directory')
    args = parser.parse_args()
    prepare_taskbar()
    app = QApplication(sys.argv[:1])
    apply_theme(app)
    window = OrionWindow(args.parameters, args.input_dir, args.output_dir)
    window.show()
    return app.exec()


if __name__ == '__main__':
    sys.exit(main())
