#!/usr/bin/env python3
"""Desktop analysis workspace. Launch with: python orion_gui.py"""
import argparse
import codecs
from copy import deepcopy
from datetime import datetime
import json
import math
from pathlib import Path
import sys
import tempfile

try:
    from PySide6.QtCore import Qt, QProcess, QThread, QTimer, Signal, QUrl
    from PySide6.QtGui import QColor, QDesktopServices, QImage, QPainter, QPixmap
    from PySide6.QtWidgets import (
        QApplication, QAbstractItemView, QCheckBox, QComboBox, QDoubleSpinBox,
        QFileDialog, QFormLayout, QGraphicsScene, QGraphicsView, QGroupBox,
        QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMainWindow, QMessageBox,
        QPlainTextEdit, QProgressBar, QPushButton, QScrollArea, QSizePolicy, QSplitter,
        QTabWidget, QTableWidget, QTableWidgetItem, QToolButton, QTreeWidget,
        QTreeWidgetItem, QVBoxLayout, QWidget,
    )
except ImportError:
    raise SystemExit('The desktop GUI needs Qt. Install it with: python -m pip install -r requirements-gui.txt')

try:
    import numpy as np
    import pandas as pd
    from gui_analysis import (FLOOR_KEYS, analyze_preview, list_images,
                              render_image, validate_parameters, validate_run_paths)
    from image_io import inspect_image
except ImportError as exc:
    raise SystemExit('Analysis dependencies are missing. Using Python 3.12, install '
                     'requirements-gui.txt in your environment: python -m pip install -r requirements-gui.txt') from exc

ROOT = Path(__file__).resolve().parent
APP_STYLE = '''
    QWidget { font-size: 13px; }
    QMainWindow { background: palette(window); }
    QGroupBox { background: palette(base); border: 1px solid palette(mid);
                border-radius: 8px; margin-top: 18px; padding: 12px 8px 8px; }
    QGroupBox::title { subcontrol-origin: margin; left: 10px; font-weight: 600; }
    QLineEdit, QComboBox, QDoubleSpinBox { padding: 5px; }
    QPushButton { padding: 6px 10px; }
    QScrollArea { border: none; }
    QTabWidget::pane { border: 1px solid palette(mid); }
'''
BASIC_KEYS = {'group_names', 'nuclear_channel', 'foci_channels', 'nuclear.fixed_threshold_adu',
              'qc_output', 'rgb_channel_order'}
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
        self.setBackgroundBrush(QColor('#080f1d'))
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setMinimumSize(300, 300)
        self.setAccessibleName('Image viewer: wheel to zoom, drag to pan, click to inspect a nucleus')
        self.pixmap = None
        self.press_point = None

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
        super().__init__()
        self.setWindowTitle('scikitORION · Nuclear & foci analysis')
        self.resize(1320, 950)
        self.parameters = json.loads((ROOT / 'parameters.template.json').read_text())
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
        self.foci_checks, self.floors = {}, {}
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.process.readyReadStandardOutput.connect(self.read_process_output)
        self.process.finished.connect(self.process_finished)
        self.process.errorOccurred.connect(self.process_error)
        self.decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
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
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.setCentralWidget(splitter)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        self.sidebar = QWidget()
        setup = QVBoxLayout(self.sidebar)
        title = QLabel('scikitORION')
        title.setStyleSheet('font-size: 23px; font-weight: 600;')
        setup.addWidget(title)
        setup.addWidget(QLabel('Nuclear & foci analysis'))
        directories = QGroupBox('Analysis directories')
        paths = QFormLayout(directories)
        paths.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
        self.input_dir = self.path_field(paths, 'Input directory', '')
        default_output = ROOT / 'Analyses' / datetime.now().strftime('%Y-%m-%d_%H%M%S_analysis')
        self.output_dir = self.path_field(paths, 'Output directory', str(default_output))
        self.parameter_dir = self.path_field(paths, 'Parameters directory', str(ROOT))
        self.parameter_name = QLineEdit('parameters.template.json')
        paths.addRow('Parameter file', self.parameter_name)
        self.parameter_name.setAccessibleName('Parameter file name')
        parameter_buttons = QHBoxLayout()
        self.add_button(parameter_buttons, 'Load JSON', self.load_parameters)
        self.add_button(parameter_buttons, 'Save JSON…', self.save_parameters)
        paths.addRow(parameter_buttons)
        self.add_button(paths, 'Load images', self.load_images)
        setup.addWidget(directories)
        basic = QGroupBox('Basic parameters')
        form = QFormLayout(basic)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
        self.groups = QLineEdit()
        self.groups.setPlaceholderText('Control, Treatment')
        form.addRow('Group names', self.groups)
        self.groups.textEdited.connect(self.invalidate_preview)
        group_help = QLabel('Comma-separated names matched to filenames. Leave blank to use the existing filename parser.')
        group_help.setWordWrap(True)
        form.addRow(group_help)
        self.rgb_enabled = QCheckBox('Map RGB TIFF colors to channels')
        form.addRow(self.rgb_enabled)
        self.rgb_container = QWidget()
        rgb_form = QFormLayout(self.rgb_container)
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
        form.addRow(self.rgb_container)
        rgb_help = QLabel('For RGB exports, assign red, green, and blue once each. '
                          'Then choose the nuclear and foci channels below. '
                          'For example: Channel 1 = Blue, Channel 2 = Green, Channel 3 = Red. '
                          'Channel stacks use their original channel order.')
        rgb_help.setWordWrap(True)
        form.addRow(rgb_help)
        self.rgb_enabled.toggled.connect(self.rgb_changed)
        self.nuclear = QComboBox()
        self.nuclear.currentIndexChanged.connect(self.channel_changed)
        form.addRow('Nuclear channel', self.nuclear)
        self.foci_container = QWidget()
        self.foci_layout = QVBoxLayout(self.foci_container)
        self.foci_layout.setContentsMargins(0, 0, 0, 0)
        form.addRow('Foci channels', self.foci_container)
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
        floors_help = QLabel('ADU floors are stored per channel. New channels start at 0 ADU; noise-based thresholds still apply.')
        floors_help.setWordWrap(True)
        form.addRow(floors_help)
        self.qc_mode = QComboBox()
        self.qc_mode.addItems(['minimal', 'detailed', 'off'])
        form.addRow('QC output', self.qc_mode)
        setup.addWidget(basic)
        self.advanced_toggle = QToolButton()
        self.advanced_toggle.setText('Advanced parameters')
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
        actions = QHBoxLayout()
        self.preview_button = self.add_button(actions, 'Preview image', self.start_preview)
        self.run_button = self.add_button(actions, 'Run analysis', self.start_analysis)
        self.stop_button = self.add_button(actions, 'Stop', self.stop_analysis)
        self.stop_button.setEnabled(False)
        left_layout.addLayout(actions)
        splitter.addWidget(left)
        workspace = QWidget()
        layout = QVBoxLayout(workspace)
        image_row = QHBoxLayout()
        image_row.addWidget(QLabel('Image'))
        self.image_picker = QComboBox()
        self.image_picker.setMinimumWidth(150)
        self.image_picker.currentIndexChanged.connect(self.image_changed)
        image_row.addWidget(self.image_picker, 1)
        self.image_info = QLabel('Load TIFF images to begin')
        image_row.addWidget(self.image_info)
        layout.addLayout(image_row)
        view_row = QHBoxLayout()
        self.mode = QComboBox()
        self.mode.addItems(['Raw image', 'Nuclei', 'Foci', 'Overlay'])
        self.mode.setCurrentText('Overlay')
        self.mode.setAccessibleName('Image view')
        self.mode.currentIndexChanged.connect(self.draw)
        view_row.addWidget(self.mode)
        self.raw_channel = QComboBox()
        self.raw_channel.setAccessibleName('Display channel')
        self.raw_channel.currentIndexChanged.connect(self.draw)
        view_row.addWidget(self.raw_channel)
        view_row.addStretch()
        self.add_button(view_row, '−', lambda: self.viewer.scale(1 / 1.25, 1 / 1.25)).setAccessibleName('Zoom out')
        self.add_button(view_row, 'Fit', lambda: self.viewer.fit())
        self.add_button(view_row, '+', lambda: self.viewer.scale(1.25, 1.25)).setAccessibleName('Zoom in')
        layout.addLayout(view_row)
        self.viewer = ImageView()
        self.viewer.picked.connect(self.pick_nucleus)
        layout.addWidget(self.viewer, 1)
        self.layers = QWidget()
        self.layers_layout = QHBoxLayout(self.layers)
        self.layers_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.layers)
        self.tabs = QTabWidget()
        self.tabs.setMaximumHeight(300)
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
        inspection.addWidget(self.nucleus_info)
        inspection.addStretch()
        self.tabs.addTab(self.inspector, 'Nucleus inspector')
        results = QWidget()
        result_layout = QVBoxLayout(results)
        self.table_label = QLabel('No measurements yet')
        result_layout.addWidget(self.table_label)
        self.table = QTableWidget()
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        result_layout.addWidget(self.table)
        self.tabs.addTab(results, 'Results')
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(10000)
        self.tabs.addTab(self.log, 'Analysis log')
        layout.addWidget(self.tabs)
        bottom = QHBoxLayout()
        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        bottom.addWidget(self.progress, 1)
        self.add_button(bottom, 'Load saved results', self.load_saved_results)
        self.report_button = self.add_button(bottom, 'Open report', self.open_report)
        self.report_button.setEnabled(False)
        layout.addLayout(bottom)
        splitter.addWidget(workspace)
        splitter.setSizes([390, 930])
        splitter.setStretchFactor(1, 1)
        self.statusBar().showMessage('Ready · previews do not write output files')
        self.input_dir.textChanged.connect(self.input_changed)

    @staticmethod
    def add_button(layout, text, callback):
        button = QPushButton(text)
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
        browse.setFixedWidth(32)
        browse.setAccessibleName('Browse ' + text.lower())
        form.addRow(text, row)
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
        self.groups.setText(', '.join(p.get('group_names', [])))
        order = p.get('rgb_channel_order')
        self.rgb_enabled.blockSignals(True)
        self.rgb_enabled.setChecked(order is not None)
        self.rgb_enabled.blockSignals(False)
        for index, color in enumerate(self.rgb_colors):
            color.blockSignals(True)
            color.setCurrentIndex(color.findData(order[index] if order else None))
            color.blockSignals(False)
        self.rgb_container.setEnabled(order is not None)
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
                    *[int(k) for k in p.get('channels', {}) if k.isdecimal()],
                    *[int(k[8:]) for k in p['foci'] if k.startswith('channel_') and k[8:].isdecimal()])
        self.build_channels(count)
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
            box = QGroupBox(f'Channel {channel} · foci ADU floors')
            form = QFormLayout(box)
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

    def channel_changed(self, *_):
        for channel, (box, _) in self.floors.items():
            box.setVisible(self.foci_checks[channel].isChecked())
        self.rebuild_layers()
        self.invalidate_preview()

    def rgb_changed(self, *_):
        self.rgb_container.setEnabled(self.rgb_enabled.isChecked())
        # A color reassignment changes the source pixels for each channel.
        self.pixels, self.metadata = None, None
        self.viewer.scene().clear()
        self.viewer.pixmap = None
        self.invalidate_preview()
        self.statusBar().showMessage('RGB mapping changed · preview again to reload channel pixels')

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
            p = json.loads(path.read_text())
            if not isinstance(p, dict):
                raise ValueError('The parameter file must contain a JSON object.')
            p.setdefault('rgb_channel_order', None)
            p.setdefault('pixel_size_um', None)
            p = validate_parameters(p)
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
            p = self.collect_parameters()
            paths = list_images(self.input_dir.text())
            md = self.inspect_input(paths[0], p)
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
        self.progress.setRange(0, 0 if busy else 1)
        self.progress.setValue(0)

    def start_preview(self):
        if self.busy:
            return
        if not self.images:
            self.load_images()
            return
        try:
            p = self.collect_parameters()
            index = self.image_picker.currentIndex()
            if not 0 <= index < len(self.images):
                raise ValueError('Choose an image to preview.')
            self.inspect_input(self.images[index], p)
            self.set_busy(True)
            self.statusBar().showMessage(f'Previewing {self.images[index].name}…')
            self.worker = PreviewWorker(self.images[index], p)
            self.worker.completed.connect(self.preview_complete)
            self.worker.failed.connect(self.show_error)
            self.worker.finished.connect(self.preview_finished)
            self.worker.start()
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self.show_error(str(exc))

    def preview_finished(self):
        self.set_busy(False)
        self.worker.deleteLater()
        self.worker = None

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
        if self.pixels is None:
            return
        # A previous image may have fewer channels than newly chosen controls.
        n = self.nuclear.currentData() or 1
        foci = [c for c, check in self.foci_checks.items() if check.isChecked()]
        if max([n, *foci]) > self.pixels.shape[0]:
            return
        p = dict(nuclear_channel=n, foci_channels=foci)
        mode = self.mode.currentText()
        self.raw_channel.setEnabled(mode == 'Raw image')
        if self.preview is None and mode in ('Nuclei', 'Foci'):
            rgb = np.zeros((*self.pixels.shape[1:], 3), dtype=np.uint8)
        else:
            raw_channel = self.raw_channel.currentData() or n
            if raw_channel > self.pixels.shape[0]:
                return
            rgb = render_image(self.pixels, p, self.preview, mode, raw_channel,
                               [c for c, check in self.layer_checks.items() if check.isChecked()],
                               self.outlines.isChecked(), self.selected)
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
        QMessageBox.warning(self, 'scikitORION', message)

    def start_analysis(self):
        if self.busy:
            return
        try:
            p = self.collect_parameters()
            if not self.input_dir.text().strip() or not self.output_dir.text().strip():
                raise ValueError('Choose input and output directories.')
            paths = list_images(self.input_dir.text())
            source, output = validate_run_paths(self.input_dir.text(), self.output_dir.text())
            for path in paths:
                self.inspect_input(path, p)
            self.run_temp = tempfile.TemporaryDirectory(prefix='orion-gui-')
            snapshot = Path(self.run_temp.name) / 'parameters.json'
            snapshot.write_text(json.dumps(p, indent=2, allow_nan=False))
            self.run_output = output
            self.run_parameters = deepcopy(p)
            self.cancel_requested = False
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
        self.decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
        self.process.setWorkingDirectory(str(ROOT))
        self.process.start(sys.executable, ['-u', str(ROOT / script), *args])
        self.statusBar().showMessage({'analysis': 'Analyzing images…', 'outputs': 'Generating QC and reports…',
                                     'validation': 'Validating saved results…'}[self.stage])

    def read_process_output(self):
        text = self.decoder.decode(bytes(self.process.readAllStandardOutput()))
        self.log.moveCursor(self.log.textCursor().MoveOperation.End)
        self.log.insertPlainText(text)
        self.log.ensureCursorVisible()
        lines = [line for line in text.splitlines() if line.strip()]
        if lines:
            self.statusBar().showMessage(lines[-1])

    def process_finished(self, code, exit_status):
        self.read_process_output()
        tail = self.decoder.decode(b'', final=True)
        if tail:
            self.log.insertPlainText(tail)
        if self.cancel_requested:
            self.log.appendPlainText('Analysis stopped. Partial files may remain in the output directory.')
            self.finish_run(False)
            self.statusBar().showMessage('Stopped · choose a new output directory before another run')
        elif code != 0 or exit_status != QProcess.ExitStatus.NormalExit:
            self.finish_run(False)
            self.show_error(f'{self.stage.capitalize()} failed (exit {code}). See the analysis log. Completed files are preserved.')
        elif self.stage == 'analysis' and self.run_parameters['qc_output'] != 'off':
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
        self.progress.setValue(1 if success else 0)
        if self.run_temp:
            self.run_temp.cleanup()
            self.run_temp = None
        if self.run_output:
            self.report_button.setEnabled((self.run_output / 'Review.html').is_file())

    def stop_analysis(self):
        if not self.busy or self.process.state() == QProcess.ProcessState.NotRunning:
            return
        self.cancel_requested = True
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
            parameters = json.loads((output / 'Methods' / 'Effective_Parameters.json').read_text())
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
    app = QApplication(sys.argv[:1])
    app.setApplicationName('scikitORION')
    app.setStyleSheet(APP_STYLE)
    window = OrionWindow(args.parameters, args.input_dir, args.output_dir)
    window.show()
    return app.exec()


if __name__ == '__main__':
    sys.exit(main())
