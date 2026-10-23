"""Regression checks for GUI channel assignment and preview/CLI agreement."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import tifffile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gui_analysis import analyze_preview, default_display_colors, list_images, render_image, validate_run_paths
from analyze_foci import nuclear_segmentation

QT_AVAILABLE = importlib.util.find_spec('PySide6') is not None
if QT_AVAILABLE:
    os.environ.setdefault('QT_QPA_PLATFORM', 'windows' if sys.platform == 'win32' else 'offscreen')
    from PySide6.QtWidgets import QApplication, QPushButton, QScrollArea
    from PySide6.QtGui import QColor, QFontDatabase, QFontInfo, QFontMetricsF
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtTest import QTest
    from orion_gui import OrionWindow, format_elapsed
    from gui_theme import DecoImageFrame, TOKENS
    from progress_reporting import RunProgress, PREFIX

PARAMETERS = json.loads((ROOT / 'parameters.template.json').read_text(encoding='utf-8'))


class ControlledClock:
    def __init__(self):
        self.milliseconds = 0
        self.starts = 0

    def start(self):
        self.milliseconds = 0
        self.starts += 1

    def elapsed(self):
        return self.milliseconds

    def isValid(self):
        return self.starts > 0


def fixture(directory, nuclear=3):
    inputs = directory / 'raw'
    inputs.mkdir()
    yy, xx = np.mgrid[:128, :128]
    nucleus = 10 + 500 * ((yy - 64) ** 2 + (xx - 64) ** 2 < 26 ** 2)
    signal = np.full((128, 128), 20.0)
    for y, x in [(54, 55), (65, 73), (76, 57)]:
        signal += 250 * np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / 8)
    arr = np.array([signal, signal, signal], dtype=np.uint16)
    arr[nuclear - 1] = nucleus
    path = inputs / 'Control_01.TIF'
    tifffile.imwrite(path, arr, imagej=True, resolution=(5, 5), metadata={'unit': 'micron', 'axes': 'CYX'})
    p = copy.deepcopy(PARAMETERS)
    p['group_names'] = ['Control']
    p['nuclear_channel'] = nuclear
    p['foci_channels'] = [c for c in (1, 2, 3) if c != nuclear]
    p['foci']['channel_1'] = copy.deepcopy(p['foci']['channel_2'])
    p['qc_output'] = 'off'
    return inputs, path, p


class PreviewTests(unittest.TestCase):
    def test_source_aware_palette_and_exact_raw_preview_colors(self):
        for nuclear in (1, 2, 3):
            others = [c for c in (1, 2, 3) if c != nuclear]
            self.assertEqual(default_display_colors(others[0], nuclear), ((0, 244, 8),) * 2)
            self.assertEqual(default_display_colors(others[1], nuclear), ((153, 29, 33),) * 2)
            self.assertEqual(default_display_colors(nuclear, nuclear), ((0, 44, 236), (105, 215, 245)))
            for colors in [('green', 'red'), ('red', 'green')]:
                order = ['blue'] * 3
                for channel, source in zip(others, colors):
                    order[channel - 1] = source
                pixels = np.tile(np.arange(16, dtype=np.uint16).reshape(4, 4), (3, 1, 1))
                p = dict(nuclear_channel=nuclear, foci_channels=others, rgb_channel_order=order)
                for channel in others:
                    expected = (0, 244, 8) if order[channel - 1] == 'green' else (153, 29, 33)
                    self.assertEqual(default_display_colors(channel, nuclear, order), (expected,) * 2)
                    rendered = render_image(pixels, p, mode='Raw image', raw_channels=[channel])
                    self.assertEqual(rendered[-1, -1].tolist(), list(expected))
        self.assertEqual(default_display_colors(4, 1), ((250, 189, 90),) * 2)

    def test_raw_multichannel_blending_legacy_empty_and_channel_validation(self):
        pixels=np.tile(np.arange(9,dtype=np.uint16),(4,9,1));before=pixels.copy()
        p={'nuclear_channel':3,'foci_channels':[1,2]}
        colors={1:(255,0,0),2:(0,255,0),3:(0,0,255),4:(255,0,0)}
        raw=render_image(pixels,p,mode='Raw image',raw_channels=[1,2],signal_colors=colors)
        self.assertEqual(raw[4,8].tolist(),[255,255,0])
        single=render_image(pixels,p,mode='Raw image',raw_channels=[1],signal_colors=colors)
        np.testing.assert_array_equal(single,render_image(pixels,p,mode='Raw image',raw_channel=1,signal_colors=colors))
        self.assertFalse(render_image(pixels,p,mode='Raw image',raw_channels=[]).any())
        clipped=render_image(pixels,p,mode='Raw image',raw_channels=[1,4],signal_colors=colors)
        self.assertEqual(clipped[4,8].tolist(),[255,0,0])
        with self.assertRaises(ValueError):render_image(pixels,p,mode='Raw image',raw_channels=[5])
        np.testing.assert_array_equal(pixels,before)

    def test_pseudocolors_outlines_and_visibility_do_not_modify_analysis_data(self):
        pixels = np.tile(np.arange(9, dtype=np.uint16), (3, 9, 1))
        nuclei = np.zeros((9, 9), dtype=np.int32)
        nuclei[1:8, 1:8] = 1
        foci = np.zeros_like(nuclei)
        foci[3:6, 3:6] = 1
        preview = SimpleNamespace(nuclei=nuclei, candidates=nuclei, foci_masks={1: foci})
        original = [array.copy() for array in (pixels, nuclei, foci)]
        p = {'nuclear_channel': 3, 'foci_channels': [1]}
        options = {'signal_colors': {1: (255, 0, 0), 3: (0, 0, 200)},
                   'outline_colors': {1: (0, 255, 0), 3: (255, 255, 0)}}
        raw = render_image(pixels, p, preview, mode='Raw image', raw_channel=1, selected=1, **options)
        self.assertEqual(raw[4, 8].tolist(), [255, 0, 0])
        np.testing.assert_array_equal(raw, render_image(pixels, p, mode='Raw image', raw_channel=1, **options))
        nuclear = render_image(pixels, p, preview, mode='Nuclei', **options)
        self.assertEqual(nuclear[2, 2].tolist(), [0, 0, 90])
        self.assertEqual(nuclear[1, 2].tolist(), [255, 255, 0])
        hidden = render_image(pixels, p, preview, mode='Nuclei', visible_channels=[1], **options)
        self.assertFalse(hidden.any())
        focus = render_image(pixels, p, preview, mode='Foci', **options)
        self.assertEqual(focus[4, 4].tolist(), [255, 0, 0])
        self.assertEqual(focus[3, 4].tolist(), [0, 255, 0])
        no_outline = render_image(pixels, p, preview, mode='Foci', outlines=False, **options)
        self.assertEqual(no_outline[3, 4].tolist(), [255, 0, 0])
        overlay = render_image(pixels, p, preview, **options)
        self.assertEqual(overlay[1, 2].tolist(), [255, 255, 0])
        self.assertEqual(overlay[3, 4].tolist(), [0, 255, 0])
        hidden = render_image(pixels, p, preview, visible_channels=[], **options)
        self.assertFalse(hidden.any())
        no_outline = render_image(pixels, p, preview, outlines=False, **options)
        self.assertEqual(no_outline[3, 4, 1], 0)
        for before, after in zip(original, (pixels, nuclei, foci)):
            np.testing.assert_array_equal(before, after)

    def test_fill_holes_setting_changes_the_mask(self):
        yy, xx = np.mgrid[:64, :64]
        radius = (yy - 32) ** 2 + (xx - 32) ** 2
        raw = np.where((radius < 20 ** 2) & (radius > 8 ** 2), 500, 10).astype(np.uint16)
        p = copy.deepcopy(PARAMETERS['nuclear'])
        p.update(fixed_threshold_adu=[100, 600], closing_disk_radius_px=0,
                 candidate_min_area_px=100, minimum_seed_distance_to_edge_px=1,
                 distance_h_maxima_px=1, accepted_min_area_um2=0, accepted_max_area_um2=10000)
        p['fill_holes'] = False
        _, without, _ = nuclear_segmentation(raw, p, 1, 1)
        p['fill_holes'] = True
        _, filled, _ = nuclear_segmentation(raw, p, 1, 1)
        self.assertEqual(without[32, 32], 0)
        self.assertGreater(filled[32, 32], 0)

    def test_preview_matches_cli_for_reassigned_nuclear_channels(self):
        for nuclear in (2, 3):
            with self.subTest(nuclear=nuclear), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                inputs, path, p = fixture(root, nuclear)
                if nuclear == 2:
                    p['nuclear']['fixed_threshold_adu'] = [100, 600]
                before = path.read_bytes()
                preview = analyze_preview(path, p)
                self.assertEqual(path.read_bytes(), before)
                self.assertEqual(set(root.iterdir()), {inputs})  # preview has no output side effects
                config = root / 'parameters.json'
                config.write_text(json.dumps(p))
                output = root / 'out'
                result = subprocess.run([sys.executable, str(ROOT / 'analyze_foci.py'),
                                         '--input-dir', str(inputs), '--output-dir', str(output),
                                         '--parameters', str(config)], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                np.testing.assert_array_equal(preview.nuclei, tifffile.imread(output / 'Masks/I01_nuclei.tif'))
                cells = pd.read_csv(output / 'Results/Cell_Summary.csv')
                self.assertEqual(len(preview.cells), 1)
                self.assertEqual(cells.nuclear_channel.tolist(), [nuclear])
                for channel in p['foci_channels']:
                    np.testing.assert_array_equal(preview.foci_masks[channel],
                                                  tifffile.imread(output / f'Masks/I01_C{channel}_foci.tif'))
                    self.assertEqual(preview.cells[0][f'C{channel}_foci_count'], cells[f'C{channel}_foci_count'].iloc[0])
                self.assertEqual(path.read_bytes(), before)
                rgb = render_image(preview.pixels, p, preview, selected=1)
                self.assertEqual(rgb.shape, (128, 128, 3))
                self.assertEqual(rgb.dtype, np.uint8)

    def test_paths_and_tiff_discovery_protect_raw_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs, path, _ = fixture(root)
            (inputs / '._ignored.TIF').write_text('metadata')
            self.assertEqual(list_images(inputs), [path.resolve()])
            for output in (inputs, inputs / 'Results', root):
                with self.assertRaises(ValueError):
                    validate_run_paths(inputs, output)
            with self.assertRaises(ValueError):
                validate_run_paths(inputs, root / 'out', inputs / 'config.json')
            output = root / 'out'
            output.mkdir()
            (output / 'old.csv').write_text('existing output')
            with self.assertRaises(ValueError):
                validate_run_paths(inputs, output)


@unittest.skipUnless(QT_AVAILABLE, 'Install requirements-gui.txt to test desktop controls')
class DesktopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def make_window(self):
        window = OrionWindow()
        self.addCleanup(window.close)
        errors = []
        window.show_error = errors.append
        window.test_errors = errors
        return window

    def wait_until_idle(self, window, timeout=45):
        deadline = time.monotonic() + timeout
        while window.busy and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(.01)
        self.app.processEvents()
        self.assertFalse(window.busy, 'The background operation did not complete')
        self.assertEqual(window.test_errors, [])

    def test_nucleus_name_backups_survive_role_changes_and_json_reload(self):
        window = self.make_window()
        window.parameters['channels'] = {'1': 'DAPI', '2': 'γH2AX', '3': '53BP1'}
        window.load_controls()
        self.assertEqual(window.channel_names[1].text(), 'Nucleus')
        self.assertFalse(window.channel_names[1].isEnabled())
        window.nuclear_checks[3].setChecked(True)
        self.assertEqual(window.channel_names[1].text(), 'DAPI')
        self.assertEqual(window.channel_names[3].text(), 'Nucleus')
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / 'names.json'
            with patch('orion_gui.QFileDialog.getSaveFileName', return_value=(str(config), 'JSON files (*.json)')):
                window.save_parameters()
            saved = json.loads(config.read_text())
            self.assertEqual(saved['channels']['3'], 'Nucleus')
            self.assertEqual(saved['gui_display']['channels']['3']['nonnuclear_name'], '53BP1')
            window.load_parameters()
            window.nuclear_checks[1].setChecked(True)
            self.assertEqual(window.channel_names[3].text(), '53BP1')
            self.assertEqual(window.channel_names[1].text(), 'Nucleus')
            self.assertEqual(window.collect_parameters()['channels']['1'], 'Nucleus')
        legacy = copy.deepcopy(PARAMETERS)
        legacy['channels']['1'] = 'Nucleus'
        window.parameters = legacy
        window.load_controls()
        window.nuclear_checks[3].setChecked(True)
        self.assertEqual(window.channel_names[1].text(), 'Channel 1')

    def test_automatic_colors_follow_mapping_with_independent_persistent_overrides(self):
        window = self.make_window()
        self.assertEqual(window.signal_colors[2], '#00f408')
        self.assertEqual(window.signal_colors[3], '#991d21')
        window.rgb_enabled.setChecked(True)
        self.assertEqual(window.outline_colors[2], '#00f408')  # incomplete mapping uses the fallback
        with self.assertRaisesRegex(ValueError, 'RGB mapping'):
            window.collect_parameters()
        for color, value in zip(window.rgb_colors, ['blue', 'green', 'red']):
            color.setCurrentIndex(color.findData(value))
        window.set_display_color(2, 'signal', '#aabbcc')
        for color, value in zip(window.rgb_colors, ['blue', 'red', 'green']):
            color.setCurrentIndex(color.findData(value))
        self.assertEqual(window.signal_colors[2], '#aabbcc')
        self.assertEqual(window.outline_colors[2], '#991d21')
        self.assertEqual(window.signal_colors[3], '#00f408')
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / 'palette.json'
            with patch('orion_gui.QFileDialog.getSaveFileName', return_value=(str(config), 'JSON files (*.json)')):
                window.save_parameters()
            saved = json.loads(config.read_text())
            self.assertFalse(saved['gui_display']['channels']['2']['signal_auto'])
            self.assertTrue(saved['gui_display']['channels']['2']['outline_auto'])
            window.load_parameters()
            for color, value in zip(window.rgb_colors, ['blue', 'green', 'red']):
                color.setCurrentIndex(color.findData(value))
            self.assertEqual(window.signal_colors[2], '#aabbcc')
            self.assertEqual(window.outline_colors[2], '#00f408')
            self.assertEqual(window.signal_colors[3], '#991d21')
            window.reset_display_colors()
            self.assertEqual(window.signal_colors[2], '#00f408')
            self.assertTrue(all(window.signal_auto.values()))
            self.assertTrue(all(window.outline_auto.values()))
        window.rgb_enabled.setChecked(False)
        window.nuclear_checks[3].setChecked(True)
        self.assertEqual(window.signal_colors[1], '#00f408')
        self.assertEqual(window.signal_colors[2], '#991d21')
        self.assertEqual(window.signal_colors[3], '#002cec')

    def test_legacy_color_overrides_remain_explicit_until_reset(self):
        window = self.make_window()
        p = copy.deepcopy(PARAMETERS)
        p['gui_display'] = {'channels': {'2': {'signal': '#aabbcc'}, '3': {'outline': '#ccbbaa'}}}
        window.parameters = p
        window.load_controls()
        self.assertEqual(window.signal_colors[2], '#aabbcc')
        self.assertFalse(window.signal_auto[2])
        self.assertTrue(window.outline_auto[2])
        window.rgb_enabled.setChecked(True)
        window.rgb_colors[1].setCurrentIndex(window.rgb_colors[1].findData('red'))
        window.rgb_colors[2].setCurrentIndex(window.rgb_colors[2].findData('green'))
        self.assertEqual(window.signal_colors[2], '#aabbcc')
        self.assertEqual(window.outline_colors[2], '#991d21')
        self.assertEqual(window.outline_colors[3], '#ccbbaa')
        with tempfile.TemporaryDirectory() as tmp:
            inputs, _, _ = fixture(Path(tmp), nuclear=1)
            window.input_dir.setText(str(inputs))
            window.load_images()
            self.wait_until_idle(window)
            self.assertFalse(window.rgb_enabled.isChecked())
            self.assertFalse(window.signal_auto[2])
            self.assertFalse(window.outline_auto[3])
            self.assertTrue(window.outline_auto[2])
            self.assertEqual(window.signal_colors[2], '#aabbcc')
            window.rgb_enabled.setChecked(True)
        window.reset_display_colors()
        self.assertEqual(window.signal_colors[2], '#991d21')
        self.assertEqual(window.outline_colors[3], '#00f408')
        p['gui_display']['channels']['2']['signal_auto'] = 'yes'
        with self.assertRaisesRegex(ValueError, 'signal_auto'):
            window.read_display_colors(p)

    def test_elapsed_clock_runs_across_all_stages_and_freezes_on_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs, _, p = fixture(root)
            p['qc_output'] = 'minimal'
            window = self.make_window()
            window.parameters = p
            window.load_controls()
            window.input_dir.setText(str(inputs))
            window.output_dir.setText(str(root / 'output'))
            clock = window.analysis_clock = ControlledClock()
            stages = []

            def stage(script, args):
                stages.append(script)
                window.run_progress.begin(window.stage)
                window.update_run_progress()

            window.start_process = stage
            window.read_process_output = lambda: None
            self.addCleanup(lambda: window.finish_run(False))
            with patch.object(window, 'load_saved_results'):
                window.start_analysis()
                self.assertEqual(clock.starts, 1)
                self.assertTrue(window.elapsed_timer.isActive())
                clock.milliseconds = 61_234
                window.elapsed_timer.timeout.emit()
                self.assertEqual(window.elapsed_label.text(), 'Elapsed 00:01:01')
                window.process_finished(0, window.process.ExitStatus.NormalExit)
                self.assertEqual(window.stage, 'outputs')
                clock.milliseconds = 3_600_000
                window.elapsed_timer.timeout.emit()
                window.process_finished(0, window.process.ExitStatus.NormalExit)
                self.assertEqual(window.stage, 'validation')
                self.assertEqual(clock.starts, 1)
                clock.milliseconds = 3_601_999
                window.process_finished(0, window.process.ExitStatus.NormalExit)
            self.assertEqual(stages, ['analyze_foci.py', 'make_outputs.py', 'validate_results.py'])
            self.assertFalse(window.elapsed_timer.isActive())
            self.assertEqual(window.elapsed_label.text(), 'Elapsed 01:00:01')
            clock.milliseconds += 100_000
            window.update_elapsed_time()
            self.assertEqual(window.elapsed_label.text(), 'Elapsed 01:00:01')
            self.assertEqual(window.progress.value(), 100)
            self.assertNotIn('elapsed', json.dumps(window.collect_parameters()))
            window.start_analysis()
            self.assertEqual(clock.starts, 2)
            self.assertEqual(window.elapsed_label.text(), 'Elapsed 00:00:00')
            clock.milliseconds = 3456
            window.process_finished(2, window.process.ExitStatus.NormalExit)
            self.assertFalse(window.elapsed_timer.isActive())
            self.assertEqual(window.elapsed_label.text(), 'Elapsed 00:00:03')
            window.input_dir.setText('')
            window.start_analysis()
            self.assertEqual(clock.starts, 2)
            self.assertEqual(window.elapsed_label.text(), 'Elapsed 00:00:03')

    def test_elapsed_time_format_includes_hours_without_a_limit(self):
        for milliseconds, text in [(0, '00:00:00'), (59_999, '00:00:59'), (60_000, '00:01:00'),
                                   (3_600_000, '01:00:00'), (360_119_999, '100:01:59')]:
            self.assertEqual(format_elapsed(milliseconds), text)

    def test_neutral_channels_advanced_roundtrip_and_role_selection(self):
        window = self.make_window()
        self.assertEqual([field.text() for field in window.channel_names.values()], ['Nucleus', 'Channel 2', 'Channel 3'])
        roundtrip = window.collect_parameters()
        expected = copy.deepcopy(PARAMETERS)
        expected['channels']['1'] = 'Nucleus'
        for key, value in expected.items():
            self.assertEqual(roundtrip[key], value, key)
        window.nuclear_checks[3].setChecked(True)
        window.foci_checks[3].setChecked(False)
        window.foci_checks[1].setChecked(True)
        window.floors[1][1]['seed_floor_adu'].setValue(42)
        p = window.collect_parameters()
        self.assertEqual(p['nuclear_channel'], 3)
        self.assertEqual(p['foci_channels'], [1, 2])
        self.assertEqual(p['foci']['channel_1']['seed_floor_adu'], 42)
        editor = window.advanced.editors[('nuclear', 'accepted_max_area_um2')][0]
        editor.setText('275')
        self.assertEqual(window.collect_parameters()['nuclear']['accepted_max_area_um2'], 275)
        editor.setText('NaN')
        with self.assertRaises(ValueError):
            window.collect_parameters()

    def test_channel_roles_transfer_without_reassigning_old_nuclei_to_foci(self):
        window = self.make_window()
        colors = dict(window.signal_colors), dict(window.outline_colors)
        for kind, palette in zip(('signal', 'outline'), colors):
            for channel, color in palette.items():
                window.set_display_color(channel, kind, color)
        for nuclear in (3, 2, 1):
            previous = window.nuclear_channel
            window.nuclear_checks[nuclear].setChecked(True)
            self.assertEqual([c for c, check in window.nuclear_checks.items() if check.isChecked()], [nuclear])
            self.assertFalse(window.foci_checks[nuclear].isChecked())
            self.assertFalse(window.foci_checks[nuclear].isEnabled())
            self.assertFalse(window.channel_names[nuclear].isEnabled())
            self.assertTrue(window.channel_names[previous].isEnabled())
            self.assertFalse(window.foci_checks[previous].isChecked())
            window.nuclear_checks[nuclear].setChecked(False)
            self.assertTrue(window.nuclear_checks[nuclear].isChecked())
        self.assertEqual((window.signal_colors, window.outline_colors), colors)
        with self.assertRaisesRegex(ValueError, 'nonempty|at least one'):
            window.collect_parameters()
        window.foci_checks[2].setChecked(True)
        self.assertEqual(window.collect_parameters()['foci_channels'], [2])

    def test_names_refresh_labels_and_roundtrip_without_recomputing_preview(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, path, p = fixture(root)
            p['channels']['3'] = 'Hoechst'
            preview = analyze_preview(path, p)
            window = self.make_window()
            window.parameters = p
            window.load_controls()
            window.preview_complete(preview)
            window.nucleus_picker.setCurrentIndex(1)
            window.raw_checks[1].setChecked(True)
            window.layer_checks[1].setChecked(False)
            window.outlines.setChecked(False)
            window.coloc_a.setCurrentIndex(window.coloc_a.findData(1))
            window.coloc_b.setCurrentIndex(window.coloc_b.findData(2))
            window.set_display_color(1, 'signal', '#ab23cd')
            window.floors[1][1]['seed_floor_adu'].setValue(42)
            # Floor edits invalidate segmentation; restore the completed preview
            # before verifying that metadata edits do not invalidate it.
            window.preview_complete(preview)
            window.nucleus_picker.setCurrentIndex(1)
            selected = window.selected
            pixels, masks, cells = window.pixels, preview.nuclei, preview.cells
            window.viewer.scale(2, 2)
            zoom = window.viewer.transform().m11()
            window.channel_names[1].setText('γH2AX')
            window.channel_names[2].setText('53BP1')
            self.assertIs(window.preview, preview)
            self.assertIs(window.pixels, pixels)
            self.assertIs(window.preview.nuclei, masks)
            self.assertIs(window.preview.cells, cells)
            self.assertEqual(window.selected, selected)
            self.assertEqual(window.viewer.transform().m11(), zoom)
            self.assertTrue(window.raw_checks[1].isChecked())
            self.assertFalse(window.layer_checks[1].isChecked())
            self.assertFalse(window.outlines.isChecked())
            self.assertEqual(window.raw_checks[1].text(), 'Channel 1 · γH2AX')
            self.assertEqual(window.layer_checks[1].text(), 'Channel 1 · γH2AX · foci')
            self.assertEqual(window.color_labels[1].text(), 'Channel 1 · γH2AX')
            self.assertEqual(window.floor_labels[1].text(), 'Channel 1 · γH2AX')
            self.assertEqual(window.coloc_a.currentText(), 'Channel 1 · γH2AX')
            self.assertEqual(window.coloc_b.currentData(), 2)
            self.assertIn('Channel 1 · γH2AX:', window.nucleus_info.text())
            self.assertIn('Channel 3 · Nucleus', window.nucleus_info.text())
            self.assertFalse(any(key[0] == 'channels' for key in window.advanced.editors))
            config = root / 'named.json'
            with patch('orion_gui.QFileDialog.getSaveFileName', return_value=(str(config), 'JSON files (*.json)')):
                window.save_parameters()
            window.load_parameters()
            self.assertEqual(window.channel_names[1].text(), 'γH2AX')
            self.assertEqual(window.channel_names[3].text(), 'Nucleus')
            self.assertEqual(window.nonnuclear_names[3], 'Hoechst')
            self.assertFalse(window.channel_names[3].isEnabled())
            self.assertEqual(window.signal_colors[1], '#ab23cd')
            self.assertEqual(window.floors[1][1]['seed_floor_adu'].effective_value(), 42)
            window.channel_names[1].setText('  ')
            self.assertEqual(window.collect_parameters()['channels']['1'], 'Channel 1')
            self.assertEqual(window.test_errors, [])

    def test_legacy_roles_and_rgb_mapping_are_normalized_without_editing_source_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, _, p = fixture(root, nuclear=1)
            p['foci_channels'] = [1, 2, 3]
            p['channels']['1'] = 'DAPI'
            p['rgb_channel_order'] = ['red', 'blue', 'green']
            config = root / 'legacy.json'
            config.write_text(json.dumps(p))
            original = config.read_bytes()
            window = self.make_window()
            window.parameter_dir.setText(str(root))
            window.parameter_name.setText(config.name)
            window.load_parameters()
            effective = window.collect_parameters()
            self.assertEqual(effective['nuclear_channel'], 1)
            self.assertEqual(effective['foci_channels'], [2, 3])
            self.assertEqual(effective['channels']['1'], 'Nucleus')
            self.assertEqual(effective['gui_display']['channels']['1']['nonnuclear_name'], 'DAPI')
            self.assertEqual(effective['rgb_channel_order'], ['blue', 'red', 'green'])
            self.assertIn('foci role was cleared', window.channel_notice.text())
            self.assertIn('uses Blue', window.channel_notice.text())
            self.assertFalse(window.channel_notice.isHidden())
            self.assertEqual(config.read_bytes(), original)
            self.assertEqual(window.test_errors, [])

    def test_rgb_nuclear_swaps_preserve_display_colors_and_clear_cached_pixels(self):
        window = self.make_window()
        window.rgb_enabled.setChecked(True)
        for color, value in zip(window.rgb_colors, ['blue', 'green', 'red']):
            color.setCurrentIndex(color.findData(value))
        colors = dict(window.signal_colors), dict(window.outline_colors)
        for kind, palette in zip(('signal', 'outline'), colors):
            for channel, color in palette.items():
                window.set_display_color(channel, kind, color)
        for nuclear, expected in [(2, ['green', 'blue', 'red']), (3, ['green', 'red', 'blue']),
                                  (1, ['blue', 'red', 'green'])]:
            window.pixels = np.zeros((3, 8, 8), dtype=np.uint16)
            window.nuclear_checks[nuclear].setChecked(True)
            window.foci_checks[next(c for c in (1, 2, 3) if c != nuclear)].setChecked(True)
            self.assertEqual(window.collect_parameters()['rgb_channel_order'], expected)
            self.assertIsNone(window.pixels)
            self.assertFalse(window.rgb_colors[nuclear - 1].isEnabled())
            self.assertEqual((window.signal_colors, window.outline_colors), colors)
            self.assertEqual(window.channel_notice.text().count('uses Blue'), 1)
            self.assertIn(f'{window.channel_label(nuclear)} uses Blue', window.channel_notice.text())
        window.rgb_enabled.setChecked(False)
        self.assertTrue(window.channel_notice.isHidden())
        window.nuclear_checks[2].setChecked(True)
        self.assertEqual([c.currentData() for c in window.rgb_colors], ['blue', 'red', 'green'])
        window.rgb_enabled.setChecked(True)
        self.assertEqual([c.currentData() for c in window.rgb_colors], ['red', 'blue', 'green'])
        window.parameters['nuclear_channel'] = 4
        window.build_channels(4)
        with self.assertRaisesRegex(ValueError, 'only Channels 1, 2, and 3'):
            window.collect_parameters()

    def test_raw_checkboxes_are_independent_preserve_selection_zoom_and_preview(self):
        with tempfile.TemporaryDirectory() as tmp:
            _,path,p=fixture(Path(tmp),nuclear=3)
            preview=analyze_preview(path,p)
            window=self.make_window();window.parameters=p;window.load_controls();window.preview_complete(preview)
            window.view_tabs.setCurrentIndex(0)
            self.assertEqual([c for c,check in window.raw_checks.items() if check.isChecked()],[3])
            layout=window.raw_controls.parentWidget().layout()
            self.assertEqual(layout.indexOf(window.raw_controls),layout.indexOf(window.view_tabs)+1)
            window.viewer.scale(2,2);zoom=window.viewer.transform().m11()
            original_pixels=preview.pixels.copy();original_mask=preview.nuclei.copy()
            layer_state={c:check.isChecked() for c,check in window.layer_checks.items()}
            with patch.object(window.viewer,'show_rgb') as show:
                window.raw_checks[1].setChecked(True)
                expected=render_image(window.pixels,p,preview,mode='Raw image',raw_channels=[1,3],
                                      signal_colors={c:QColor(color).getRgb()[:3] for c,color in window.signal_colors.items()})
                np.testing.assert_array_equal(show.call_args.args[0],expected)
                for check in window.raw_checks.values():check.setChecked(False)
                self.assertFalse(show.call_args.args[0].any())
                self.assertFalse(window.raw_empty.isHidden())
                window.raw_checks[2].setChecked(True)
                window.view_tabs.setCurrentIndex(3);window.view_tabs.setCurrentIndex(0)
                window.rebuild_layers()
                self.assertEqual([c for c,check in window.raw_checks.items() if check.isChecked()],[2])
                self.assertEqual(layer_state,{c:check.isChecked() for c,check in window.layer_checks.items()})
                # A channel may be inspected even when it is not an analyzed foci channel.
                window.pixels=np.concatenate([preview.pixels,preview.pixels[:1]])
                window.draw();self.assertEqual(set(window.raw_checks),{1,2,3,4})
                window.raw_checks[4].setChecked(True)
                window.pixels=preview.pixels;window.draw()
                self.assertEqual(set(window.raw_checks),{1,2,3})
                self.assertTrue(window.raw_checks[2].isChecked())
            self.assertIs(window.preview,preview)
            self.assertEqual(window.viewer.transform().m11(),zoom)
            np.testing.assert_array_equal(preview.pixels,original_pixels)
            np.testing.assert_array_equal(preview.nuclei,original_mask)

    def test_floor_tenths_preserve_untouched_imported_precision_and_json(self):
        window=self.make_window();p=copy.deepcopy(PARAMETERS)
        p['foci']['channel_2']['support_floor_adu']=15.123456789
        window.parameters=p;window.load_controls()
        spin=window.floors[2][1]['support_floor_adu']
        self.assertEqual(spin.decimals(),1);self.assertEqual(spin.singleStep(),.1)
        self.assertTrue(spin.text().endswith('15.1'))
        self.assertEqual(window.collect_parameters()['foci']['channel_2']['support_floor_adu'],15.123456789)
        # Merely committing/focusing the rounded display does not edit its source value.
        spin.interpretText()
        self.assertEqual(window.collect_parameters()['foci']['channel_2']['support_floor_adu'],15.123456789)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'parameters.json'
            with patch('orion_gui.QFileDialog.getSaveFileName',return_value=(str(path),'JSON files (*.json)')):
                window.save_parameters()
            self.assertEqual(json.loads(path.read_text())['foci']['channel_2']['support_floor_adu'],15.123456789)
        spin.lineEdit().textEdited.emit(spin.text())
        self.assertEqual(window.collect_parameters()['foci']['channel_2']['support_floor_adu'],15.1)
        spin.setValue(15.2)
        self.assertEqual(window.collect_parameters()['foci']['channel_2']['support_floor_adu'],15.2)
        self.assertEqual(window.lower.decimals(),4)

    def test_nuclear_mean_uses_supported_property_without_deprecation(self):
        import warnings
        yy,xx=np.mgrid[:128,:128]
        raw=np.array(10+500*((yy-64)**2+(xx-64)**2<26**2),dtype=np.uint16)
        with warnings.catch_warnings():
            warnings.filterwarnings('error',message='.*RegionProperties.mean_intensity.*',category=FutureWarning)
            _,candidates,audit=nuclear_segmentation(raw,PARAMETERS['nuclear'],.2,.2)
        self.assertTrue(audit)
        self.assertAlmostEqual(audit[0]['nuclear_mean_DAPI_adu'],float(raw[candidates==audit[0]['nucleus_label']].mean()))

    def test_studio_layout_fonts_and_color_drawer_fit_a_smaller_window(self):
        window = self.make_window()
        self.assertIn('Marcellus', QFontDatabase.families())
        self.assertIn('Josefin Sans', QFontDatabase.families())
        self.assertTrue(self.app.property('orionStudioTheme'))
        self.assertEqual(TOKENS.space, 12)
        window.show()
        window.resize(1060, 800)
        self.app.processEvents()
        window.colors_toggle.setChecked(True)
        window.advanced_toggle.setChecked(True)
        self.app.processEvents()
        self.assertEqual((window.width(), window.height()), (1060, 800))
        self.assertGreaterEqual(window.viewer.width(), 240)
        self.assertGreaterEqual(window.viewer.height(), 200)
        for area in window.findChildren(QScrollArea):
            self.assertEqual(area.horizontalScrollBar().maximum(), 0)
        self.assertIsNotNone(window.findChild(DecoImageFrame))
        self.assertFalse(window.rgb_enabled.isChecked())
        self.assertTrue(all(field.isHidden() for field in window.rgb_fields.values()))
        self.assertTrue(window.rgb_help.isHidden())
        self.assertTrue(window.elapsed_label.isVisible())
        self.assertEqual(window.elapsed_label.text(), 'Elapsed 00:00:00')
        window.rgb_enabled.setChecked(True)
        window.channel_names[2].setText('Phosphorylated H2AX')
        window.channel_names[3].setText('DNA repair marker 53BP1')
        window.coloc_enabled.setChecked(True)
        self.app.processEvents()
        self.assertTrue(all(not field.isHidden() for field in window.rgb_fields.values()))
        self.assertEqual((window.width(), window.height()), (1060, 800))
        for area in window.findChildren(QScrollArea):
            self.assertEqual(area.horizontalScrollBar().maximum(), 0)
        for check in window.layer_checks.values():
            self.assertGreaterEqual(check.width(), check.sizeHint().width())
        self.assertGreater(window.outlines.y(), 0)
        window.view_tabs.setCurrentIndex(0)
        self.app.processEvents()
        for check in window.raw_checks.values():
            self.assertGreaterEqual(check.width(), check.sizeHint().width())
        self.assertEqual(window.size().toTuple(), (1060, 800))

    def test_analysis_log_remains_monospaced_after_theme_polish(self):
        window = self.make_window()
        window.show()
        window.tabs.setCurrentWidget(window.log)
        self.app.setStyleSheet(self.app.styleSheet())
        self.app.processEvents()
        for font in [window.log.font(), window.log.document().defaultFont()]:
            self.assertTrue(QFontInfo(font).fixedPitch())
            metrics = QFontMetricsF(font)
            widths = [metrics.horizontalAdvance(character) for character in 'iW0. ']
            self.assertLess(max(widths) - min(widths), .01)
        self.assertEqual(QFontInfo(window.groups.font()).family(), 'Josefin Sans')
        self.assertTrue(window.log.isReadOnly())
        self.assertEqual(window.log.maximumBlockCount(), 10000)

    def test_panels_resize_collapse_and_restore_independently(self):
        window = self.make_window()
        window.show()
        self.app.processEvents()
        horizontal = window.horizontal_splitter
        vertical = window.vertical_splitter
        original_width = horizontal.sizes()[0]
        handle = horizontal.handle(1)
        point = QPoint(handle.width() // 2, handle.height() // 4)
        QTest.mousePress(handle, Qt.MouseButton.LeftButton, pos=point)
        QTest.mouseMove(handle, point + QPoint(60, 0))
        QTest.mouseRelease(handle, Qt.MouseButton.LeftButton, pos=point + QPoint(60, 0))
        self.app.processEvents()
        self.assertGreater(horizontal.sizes()[0], original_width)
        handle = vertical.handle(1)
        point = QPoint(handle.width() // 5, handle.height() // 2)
        QTest.mousePress(handle, Qt.MouseButton.LeftButton, pos=point)
        QTest.mouseMove(handle, point - QPoint(0, 80))
        QTest.mouseRelease(handle, Qt.MouseButton.LeftButton, pos=point - QPoint(0, 80))
        self.app.processEvents()
        self.assertGreater(vertical.sizes()[1], 210)
        self.assertGreater(window.tabs.height(), 200)
        width, height = horizontal.sizes()[0], vertical.sizes()[1]
        settings = window.collect_parameters()
        window.sidebar_toggle.click()
        self.app.processEvents()
        self.assertTrue(window.sidebar_panel.isHidden())
        self.assertFalse(window.console_collapsed)
        self.assertEqual(window.sidebar_toggle.arrowType(), Qt.ArrowType.RightArrow)
        window.console_toggle.click()
        self.app.processEvents()
        self.assertTrue(window.console_panel.isHidden())
        self.assertEqual(window.console_toggle.arrowType(), Qt.ArrowType.UpArrow)
        self.assertEqual(horizontal.sizes()[0], 0)
        self.assertEqual(vertical.sizes()[1], 0)
        for button in [window.run_button, window.preview_button, window.stop_button,
                       window.results_button, window.report_button, window.console_toggle]:
            self.assertTrue(button.isVisible())
        for label in [window.overall_percent, window.stage_percent]:
            self.assertTrue(label.isVisible())
        window.console_toggle.click()
        window.sidebar_toggle.click()
        self.app.processEvents()
        self.assertAlmostEqual(horizontal.sizes()[0], width, delta=1)
        self.assertAlmostEqual(vertical.sizes()[1], height, delta=1)
        self.assertEqual(window.collect_parameters(), settings)
        window.set_sidebar_collapsed(True)
        window.set_console_collapsed(True)
        window.resize(1060, 800)
        self.app.processEvents()
        self.assertTrue(window.sidebar_collapsed)
        self.assertTrue(window.console_collapsed)
        window.set_sidebar_collapsed(False)
        window.set_console_collapsed(False)
        self.app.processEvents()
        self.assertEqual(window.size().toTuple(), (1060, 800))
        self.assertLessEqual(vertical.sizes()[1], height)
        self.assertGreaterEqual(window.viewer.width(), 240)
        self.assertGreaterEqual(window.viewer.height(), 200)

    def test_divider_grips_follow_handles_and_remain_clickable_when_collapsed(self):
        window = self.make_window()
        window.show()
        self.app.processEvents()

        def check_position(splitter, grip):
            self.assertEqual(splitter.count(), 2)
            self.assertIs(grip.parentWidget(), window.centralWidget())
            handle = splitter.handle(1)
            expected = handle.mapTo(grip.parentWidget(), handle.rect().center())
            self.assertLessEqual((grip.geometry().center() - expected).manhattanLength(), 2)
            target = window.centralWidget().childAt(grip.geometry().center())
            self.assertIs(target, grip)

        for splitter, grip in [(window.horizontal_splitter, window.sidebar_toggle),
                               (window.vertical_splitter, window.console_toggle)]:
            check_position(splitter, grip)
        self.assertGreater(window.sidebar_toggle.height(), window.sidebar_toggle.width())
        self.assertGreater(window.console_toggle.width(), window.console_toggle.height())
        self.assertFalse(window.footer.isAncestorOf(window.console_toggle))
        # Use hit-testing to exercise the overlay rather than directly invoking
        # a button that might be obscured by another widget.
        for grip in [window.sidebar_toggle, window.console_toggle]:
            target = window.centralWidget().childAt(grip.geometry().center())
            QTest.mouseClick(target, Qt.MouseButton.LeftButton)
            self.app.processEvents()
        self.assertTrue(window.sidebar_collapsed)
        self.assertTrue(window.console_collapsed)
        window.resize(1060, 800)
        self.app.processEvents()
        for grip in [window.console_toggle, window.sidebar_toggle]:
            self.assertTrue(window.centralWidget().rect().contains(grip.geometry()))
            target = window.centralWidget().childAt(grip.geometry().center())
            self.assertIs(target, grip)
            QTest.mouseClick(target, Qt.MouseButton.LeftButton)
            self.app.processEvents()
        self.assertFalse(window.sidebar_collapsed)
        self.assertFalse(window.console_collapsed)
        for splitter, grip in [(window.horizontal_splitter, window.sidebar_toggle),
                               (window.vertical_splitter, window.console_toggle)]:
            check_position(splitter, grip)

    def test_panel_toggles_preserve_preview_zoom_selections_and_work_during_processing(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, path, p = fixture(Path(tmp))
            preview = analyze_preview(path, p)
            window = self.make_window()
            window.parameters = p
            window.load_controls()
            window.show()
            self.app.processEvents()
            window.preview_complete(preview)
            window.nucleus_picker.setCurrentIndex(1)
            window.layer_checks[1].setChecked(False)
            window.raw_checks[2].setChecked(True)
            window.tabs.setCurrentWidget(window.log)
            window.viewer.scale(3, 3)
            window.viewer.centerOn(64, 64)
            zoom = window.viewer.transform()
            center = window.viewer.mapToScene(window.viewer.viewport().rect().center())
            selected = window.selected
            with patch.object(window, 'draw') as draw:
                window.set_busy(True, processing=True)
                try:
                    for button in [window.sidebar_toggle, window.console_toggle]:
                        self.assertTrue(button.isEnabled())
                        button.click()
                    self.app.processEvents()
                    self.assertTrue(window.stop_button.isEnabled())
                    self.assertTrue(window.stop_button.isVisible())
                    self.assertFalse(window.run_button.isEnabled())
                    self.assertFalse(window.preview_button.isEnabled())
                    window.console_toggle.click()
                    window.sidebar_toggle.click()
                    self.app.processEvents()
                    draw.assert_not_called()
                finally:
                    window.set_busy(False)
            self.assertIs(window.preview, preview)
            self.assertIs(window.pixels, preview.pixels)
            self.assertEqual(window.viewer.transform(), zoom)
            restored_center = window.viewer.mapToScene(window.viewer.viewport().rect().center())
            self.assertLess((center - restored_center).manhattanLength(), 2)
            self.assertEqual(window.selected, selected)
            self.assertFalse(window.layer_checks[1].isChecked())
            self.assertTrue(window.raw_checks[2].isChecked())
            self.assertIs(window.tabs.currentWidget(), window.log)

    def test_panel_focus_and_keyboard_disclosure(self):
        window = self.make_window()
        window.show()
        self.app.processEvents()
        window.groups.setFocus()
        window.set_sidebar_collapsed(True)
        self.assertIs(self.app.focusWidget(), window.sidebar_toggle)
        QTest.keyClick(window.sidebar_toggle, Qt.Key.Key_Space)
        self.assertFalse(window.sidebar_collapsed)
        self.assertEqual(window.sidebar_toggle.accessibleName(), 'Hide settings panel')
        window.tabs.setCurrentWidget(window.log)
        window.log.setFocus()
        window.set_console_collapsed(True)
        self.assertIs(self.app.focusWidget(), window.console_toggle)
        QTest.keyClick(window.console_toggle, Qt.Key.Key_Space)
        self.assertFalse(window.console_collapsed)
        self.assertEqual(window.console_toggle.arrowType(), Qt.ArrowType.DownArrow)
        window.console_panel.setFocus()
        window.set_console_collapsed(True)
        self.assertIs(self.app.focusWidget(), window.console_toggle)

    def test_console_reveals_only_for_errors_and_explicit_panel_actions(self):
        window = self.make_window()
        window.show()
        self.app.processEvents()
        window.set_console_collapsed(True)
        window.consume_process_output([(None, 'Ordinary processing output\n')])
        self.assertTrue(window.console_collapsed)
        with patch('orion_gui.QMessageBox.warning'):
            OrionWindow.show_error(window, 'Example failure')
        self.assertFalse(window.console_collapsed)
        self.assertIs(window.tabs.currentWidget(), window.log)
        self.assertIn('Example failure', window.log.toPlainText())
        window.set_console_collapsed(True)
        window.colors_toggle.setChecked(True)
        self.assertFalse(window.console_collapsed)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'Results').mkdir()
            (root / 'Methods').mkdir()
            pd.DataFrame([{'nucleus_id': 'I01_N1'}]).to_csv(root / 'Results/Cell_Summary.csv', index=False)
            (root / 'Methods/Effective_Parameters.json').write_text(json.dumps(PARAMETERS))
            window.set_console_collapsed(True)
            window.load_saved_results(root, reveal=False)
            self.assertTrue(window.console_collapsed)
            window.load_saved_results(root)
            self.assertFalse(window.console_collapsed)
            self.assertEqual(window.tabs.currentIndex(), 1)

    def test_colocalization_box_retains_visible_disabled_inputs(self):
        window = self.make_window()
        self.assertEqual(window.coloc_box.title(), 'COLOCALIZATION')
        self.assertTrue(window.coloc_box.isAncestorOf(window.coloc_enabled))
        self.assertTrue(window.coloc_box.isAncestorOf(window.coloc_minimum))
        self.assertFalse(window.coloc_container.isHidden())
        self.assertFalse(window.coloc_container.isEnabled())
        self.assertFalse(any(button.text().lower() == 'load images' for button in window.findChildren(QPushButton)))
        window.coloc_enabled.setChecked(True)
        window.coloc_minimum.setValue(7)
        choices = [window.coloc_a.currentData(), window.coloc_b.currentData()]
        window.coloc_enabled.setChecked(False)
        self.assertFalse(window.coloc_container.isHidden())
        self.assertFalse(window.coloc_a.isEnabled())
        window.coloc_enabled.setChecked(True)
        self.assertTrue(window.coloc_minimum.isEnabled())
        self.assertEqual(window.coloc_minimum.value(), 7)
        self.assertEqual([window.coloc_a.currentData(), window.coloc_b.currentData()], choices)

    def test_channel_keyboard_navigation_follows_row_order(self):
        window = self.make_window()
        window.show()
        window.rgb_enabled.setChecked(True)
        self.app.processEvents()
        window.channel_names[2].setFocus()
        self.assertIs(self.app.focusWidget(), window.channel_names[2])
        for target in [window.foci_checks[2], window.nuclear_checks[2], window.rgb_colors[1],
                       window.channel_names[3], window.foci_checks[3], window.nuclear_checks[3],
                       window.rgb_colors[2], window.coloc_enabled]:
            QTest.keyClick(self.app.focusWidget(), Qt.Key.Key_Tab)
            self.assertIs(self.app.focusWidget(), target)
            self.assertTrue(target.accessibleName())

    def test_rgb_mapping_only_activates_when_the_input_batch_needs_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs, path, p = fixture(root)
            order = ['green', 'red', 'blue']
            p['rgb_channel_order'] = ['green', 'blue', 'red']
            window = self.make_window()
            window.parameters = p
            window.load_controls()
            self.assertTrue(window.rgb_enabled.isChecked())  # saved RGB experiment
            window.input_dir.setText(str(inputs))
            window.output_dir.setText(str(root / 'output'))
            before = path.read_bytes()
            window.load_images()
            self.wait_until_idle(window)
            self.assertFalse(window.rgb_enabled.isChecked())
            self.assertTrue(all(field.isHidden() for field in window.rgb_fields.values()))
            self.assertIsNone(window.preview.parameters['rgb_channel_order'])
            np.testing.assert_array_equal(window.preview.pixels, tifffile.imread(path))
            self.assertEqual([combo.currentData() for combo in window.rgb_colors], order)
            window.start_analysis()
            self.wait_until_idle(window)
            effective = json.loads((root / 'output/Methods/Effective_Parameters.json').read_text())
            self.assertIsNone(effective['rgb_channel_order'])
            self.assertEqual(path.read_bytes(), before)
            # A mixed batch still needs the remembered mapping, even if its
            # first image is a channel stack. Nuclear Blue stays assigned.
            cyx = tifffile.imread(path)
            rgb_path = inputs / 'Control_02.TIF'
            tifffile.imwrite(rgb_path, np.moveaxis(cyx[[1, 0, 2]], 0, 2),
                             photometric='rgb', resolution=(127000, 127000), resolutionunit='INCH')
            mixed, _ = window.prepare_input_parameters(list_images(inputs))
            self.assertTrue(window.rgb_enabled.isChecked())
            self.assertTrue(all(not field.isHidden() for field in window.rgb_fields.values()))
            self.assertEqual(mixed['rgb_channel_order'], order)

    def test_view_tabs_color_dialog_and_saved_display_preferences(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, path, p = fixture(root, nuclear=3)
            preview = analyze_preview(path, p)
            window = self.make_window()
            window.parameters = p
            window.load_controls()
            window.preview_complete(preview)
            pixels, nuclei = preview.pixels.copy(), preview.nuclei.copy()
            self.assertEqual([window.view_tabs.tabText(i) for i in range(window.view_tabs.count())],
                             ['Raw image', 'Nuclei', 'Foci', 'Overlay'])
            window.colors_toggle.setChecked(True)
            self.assertFalse(window.colors_panel.isHidden())
            with patch('orion_gui.QColorDialog.getColor', return_value=QColor('#ff0000')):
                window.color_buttons[1, 'signal'].click()
            with patch('orion_gui.QColorDialog.getColor', return_value=QColor('#00ff00')):
                window.color_buttons[1, 'outline'].click()
            self.assertIs(window.preview, preview)
            self.assertEqual(window.color_buttons[1, 'signal'].text(), '#FF0000')
            self.assertEqual(window.color_buttons[1, 'outline'].text(), '#00FF00')
            with patch('orion_gui.QColorDialog.getColor', return_value=QColor()):
                window.color_buttons[1, 'signal'].click()
            self.assertEqual(window.signal_colors[1], '#ff0000')
            window.viewer.scale(2, 2)
            zoom = window.viewer.transform().m11()
            for index in range(4):
                window.view_tabs.setCurrentIndex(index)
                self.assertEqual(window.raw_controls.isEnabled(), index == 0)
                self.assertEqual(window.viewer.transform().m11(), zoom)
                self.assertIs(window.preview, preview)
            np.testing.assert_array_equal(preview.pixels, pixels)
            np.testing.assert_array_equal(preview.nuclei, nuclei)
            config = root / 'colors.json'
            with patch('orion_gui.QFileDialog.getSaveFileName', return_value=(str(config), 'JSON files (*.json)')):
                window.save_parameters()
            saved = json.loads(config.read_text())
            self.assertEqual(saved['gui_display']['channels']['1'], {
                'signal': '#ff0000', 'outline': '#00ff00', 'signal_auto': False, 'outline_auto': False,
                'nonnuclear_name': 'Channel 1'})
            window.reset_display_colors()
            self.assertIs(window.preview, preview)
            self.assertNotEqual(window.signal_colors[1], '#ff0000')
            window.load_parameters()
            self.assertEqual(window.signal_colors[1], '#ff0000')
            self.assertEqual(window.outline_colors[1], '#00ff00')
            self.assertNotIn(('gui_display', 'channels', '1', 'signal'), window.advanced.editors)
            window.nuclear_checks[1].setChecked(True)
            window.foci_checks[1].setChecked(False)
            self.assertEqual(window.signal_colors[1], '#ff0000')
            self.assertEqual(window.outline_colors[1], '#00ff00')
            self.assertEqual(window.test_errors, [])

    def test_background_preview_and_complete_analysis_with_channel_three_nuclei(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'Experiment Δ with spaces'
            root.mkdir()
            inputs, path, p = fixture(root)
            p['channels'] = {'1': 'γH2AX', '2': '53BP1', '3': 'Hoechst'}
            path = path.rename(inputs / 'Control_Δ.TIF')
            config = root / 'config.json'
            config.write_text(json.dumps(p))
            window = self.make_window()
            window.parameter_dir.setText(str(root))
            window.parameter_name.setText(config.name)
            window.load_parameters()
            window.input_dir.setText(str(inputs))
            window.output_dir.setText(str(root / 'output'))
            before = config.read_bytes(), path.read_bytes()
            window.preview_button.click()
            self.wait_until_idle(window)
            self.assertEqual(window.preview.parameters['nuclear_channel'], 3)
            self.assertEqual(set(window.preview.foci_masks), {1, 2})
            window.nucleus_picker.setCurrentIndex(1)
            self.assertIn('Nuclear Channel 3', window.nucleus_info.text())
            self.assertIn('Channel 1 · γH2AX:', window.nucleus_info.text())
            window.advanced.editors[('nuclear', 'fill_holes')][0].setChecked(False)
            for key, value in {
                ('nuclear', 'accepted_max_area_um2'): '275',
                ('foci', 'gaussian_sigma_px'): '1.25',
                ('foci', 'support_noise_multiplier'): '4',
                ('foci', 'post_watershed_peak_fraction'): '0.3',
                ('foci', 'annulus_min_pixels'): '10',
                ('qc', 'extranuclear_background_exclusion_dilation_px'): '11',
            }.items():
                window.advanced.editors[key][0].setText(value)
            window.qc_mode.setCurrentText('minimal')
            overall_values = []
            stage_values = []
            window.progress.valueChanged.connect(overall_values.append)
            window.stage_progress.valueChanged.connect(stage_values.append)
            window.set_console_collapsed(True)
            window.start_analysis()
            self.wait_until_idle(window)
            self.assertTrue(window.console_collapsed)
            self.assertEqual(window.progress.value(), 100)
            self.assertEqual(window.stage_progress.value(), 100)
            self.assertEqual(window.overall_percent.text(), '100%')
            self.assertEqual(window.stage_percent.text(), '100%')
            self.assertEqual(overall_values, sorted(overall_values))
            self.assertTrue(any(0 < v < 100 for v in overall_values))
            self.assertTrue(any(0 < v < 100 for v in stage_values))
            self.assertNotIn(PREFIX, window.log.toPlainText())
            self.assertEqual(window.table_origin, 'saved')
            self.assertTrue((root / 'output/Methods/Validation_Report.json').is_file())
            self.assertTrue((root / 'output/Review.html').is_file())
            self.assertIn('Control_Δ.TIF', window.log.toPlainText())
            output_settings = json.loads((root / 'output/Methods/Output_Parameters.json').read_text(encoding='utf-8'))
            self.assertEqual(output_settings['qc_output'], 'minimal')
            methods = (root / 'output/Methods/Methods.md').read_text(encoding='utf-8')
            for description in ['internal holes are retained', '25 to 275 square microns',
                                'Gaussian sigma 1.25 px', '4 x sigma_smooth',
                                '30% of its corrected peak', 'At least 10 annular pixels',
                                'after 11 px dilation']:
                self.assertIn(description, methods)
            effective = json.loads((root / 'output/Methods/Effective_Parameters.json').read_text(encoding='utf-8'))
            self.assertEqual(effective['nuclear_channel'], 3)
            self.assertEqual(effective['foci_channels'], [1, 2])
            self.assertEqual(effective['channels'], {**p['channels'], '3': 'Nucleus'})
            self.assertEqual(set(pd.read_csv(root / 'output/Results/Cell_Summary.csv').nuclear_marker), {'Nucleus'})
            self.assertEqual(set(pd.read_csv(root / 'output/Results/Focus_Data.csv').marker), {'γH2AX', '53BP1'})
            self.assertIn('Channel 1 · γH2AX', window.table_label.text())
            self.assertEqual((config.read_bytes(), path.read_bytes()), before)
            window.groups.setText('Different')
            window.invalidate_preview()
            self.assertIsNone(window.preview)
            window.input_dir.setText(str(root / 'another_raw_folder'))
            self.assertEqual(window.images, [])
            self.assertIsNone(window.pixels)

    def test_more_than_three_channels_and_manual_threshold(self):
        window = self.make_window()
        window.parameters['foci']['channel_4'] = dict(zip(('support_floor_adu', 'seed_floor_adu', 'prominence_floor_adu'), [5, 10, 5]))
        window.parameters['nuclear_channel'] = 4
        window.load_controls()
        self.assertTrue(window.nuclear_checks[4].isChecked())
        self.assertFalse(window.channel_names[4].isEnabled())
        window.threshold.setCurrentIndex(1)
        self.assertTrue(window.lower.isEnabled())
        window.lower.setValue(200)
        window.upper.setValue(100)
        with self.assertRaises(ValueError):
            window.collect_parameters()
        window.upper.setValue(300)
        self.assertEqual(window.collect_parameters()['nuclear']['fixed_threshold_adu'], [200, 300])
        window.threshold.setCurrentIndex(0)
        self.assertIsNone(window.collect_parameters()['nuclear']['fixed_threshold_adu'])

    def test_rgb_mapping_controls_save_reload_and_reject_invalid_choices(self):
        window = self.make_window()
        self.assertNotIn(('rgb_channel_order',), window.advanced.editors)
        window.rgb_enabled.setChecked(True)
        self.assertEqual(window.rgb_colors[0].currentData(), 'blue')
        self.assertFalse(window.rgb_colors[0].isEnabled())
        with self.assertRaisesRegex(ValueError, 'RGB mapping'):
            window.collect_parameters()
        for color, value in zip(window.rgb_colors, ['blue', 'green', 'red']):
            color.setCurrentIndex(color.findData(value))
        self.assertEqual(window.collect_parameters()['rgb_channel_order'], ['blue', 'green', 'red'])
        window.rgb_colors[2].setCurrentIndex(window.rgb_colors[2].findData('green'))
        with self.assertRaisesRegex(ValueError, 'once each'):
            window.collect_parameters()
        window.rgb_colors[2].setCurrentIndex(window.rgb_colors[2].findData('red'))
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / 'rgb.json'
            with patch('orion_gui.QFileDialog.getSaveFileName', return_value=(str(config), 'JSON files (*.json)')):
                window.save_parameters()
            window.rgb_enabled.setChecked(False)
            self.assertIsNone(window.collect_parameters()['rgb_channel_order'])
            window.load_parameters()
            self.assertTrue(window.rgb_enabled.isChecked())
            self.assertEqual(window.collect_parameters()['rgb_channel_order'], ['blue', 'green', 'red'])

    def test_rgb_preview_and_full_run_with_channel_two_nuclei(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs, path, p = fixture(root, nuclear=2)
            baseline = analyze_preview(path, p)
            cyx = tifffile.imread(path)
            # C1 is green, C2 blue, C3 red. Roles remain independent of color.
            rgb = np.moveaxis(cyx[[2, 0, 1]], 0, 2)
            tifffile.imwrite(path, rgb, photometric='rgb', resolution=(127000, 127000),
                             resolutionunit='INCH')
            before = path.read_bytes()
            window = self.make_window()
            window.parameters = p
            window.load_controls()
            window.input_dir.setText(str(inputs))
            window.output_dir.setText(str(root / 'output'))
            window.load_images()
            self.assertIn('Basic parameters', window.test_errors.pop())
            self.assertTrue(window.rgb_enabled.isChecked())
            self.assertIsNone(window.worker)
            for color, value in zip(window.rgb_colors, ['green', 'blue', 'red']):
                color.setCurrentIndex(color.findData(value))
            window.load_images()
            self.wait_until_idle(window)
            np.testing.assert_array_equal(window.preview.pixels, cyx)
            np.testing.assert_array_equal(window.preview.nuclei, baseline.nuclei)
            for channel in p['foci_channels']:
                np.testing.assert_array_equal(window.preview.foci_masks[channel], baseline.foci_masks[channel])
            window.start_analysis()
            self.wait_until_idle(window)
            effective = json.loads((root / 'output/Methods/Effective_Parameters.json').read_text(encoding='utf-8'))
            self.assertEqual(effective['rgb_channel_order'], ['green', 'blue', 'red'])
            self.assertEqual(effective['nuclear_channel'], 2)
            np.testing.assert_array_equal(tifffile.imread(root / 'output/Masks/I01_nuclei.tif'), baseline.nuclei)
            cells = pd.read_csv(root / 'output/Results/Cell_Summary.csv')
            for channel in p['foci_channels']:
                self.assertEqual(cells[f'C{channel}_foci_count'].iloc[0], baseline.cells[0][f'C{channel}_foci_count'])
            self.assertEqual(path.read_bytes(), before)
            window.rgb_colors[0].setCurrentIndex(window.rgb_colors[0].findData('red'))
            self.assertIsNone(window.preview)
            self.assertIsNone(window.pixels)
            self.assertIsNone(window.viewer.pixmap)

    def test_rgb_without_mapping_blocks_run_before_output_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs, path, p = fixture(root)
            # A later RGB image must be checked even when the first is a stack.
            tifffile.imwrite(inputs / 'Control_02.TIF', np.moveaxis(tifffile.imread(path), 0, 2),
                             photometric='rgb', resolution=(127000, 127000), resolutionunit='INCH')
            window = self.make_window()
            window.parameters = p
            window.load_controls()
            window.input_dir.setText(str(inputs))
            output = root / 'out'
            window.output_dir.setText(str(output))
            stages = []
            window.start_process = lambda script, args: stages.append(script)
            window.start_analysis()
            self.assertEqual(stages, [])
            self.assertFalse(output.exists())
            self.assertIsNone(window.run_temp)
            self.assertIn('Control_02.TIF is an RGB TIFF', window.test_errors[-1])
            self.assertTrue(window.rgb_enabled.isChecked())

    def test_stop_does_not_start_another_stage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs, _, p = fixture(root)
            window = self.make_window()
            window.parameters = p
            window.load_controls()
            window.input_dir.setText(str(inputs))
            window.output_dir.setText(str(root / 'out'))
            stages = []
            clock = window.analysis_clock = ControlledClock()

            def slow_stage(script, args):
                stages.append(script)
                window.process.start(sys.executable, ['-c', 'import time; time.sleep(20)'])

            window.start_process = slow_stage
            window.start_analysis()
            deadline = time.monotonic() + 5
            while window.process.state() == window.process.ProcessState.Starting and time.monotonic() < deadline:
                self.app.processEvents()
                time.sleep(.01)
            window.consume_process_output([(dict(stage='analysis', completed=2, total=4,
                                                  detail='Half of mask work complete'), '')])
            before = window.progress.value(), window.stage_progress.value()
            clock.milliseconds = 2100
            window.stop_analysis()
            self.assertTrue(window.elapsed_timer.isActive())
            clock.milliseconds = 4100
            self.wait_until_idle(window)
            self.assertFalse(window.elapsed_timer.isActive())
            self.assertEqual(window.elapsed_label.text(), 'Elapsed 00:00:04')
            self.assertEqual((window.progress.value(), window.stage_progress.value()), before)
            self.assertEqual(stages, ['analyze_foci.py'])
            self.assertEqual(window.process.state(), window.process.ProcessState.NotRunning)
            self.assertIsNone(window.run_temp)

    def test_failure_preserves_progress_and_next_run_resets_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs, _, p = fixture(root)
            window = self.make_window()
            window.parameters = p
            window.load_controls()
            window.input_dir.setText(str(inputs))
            window.output_dir.setText(str(root / 'out'))
            states = []

            def fail_stage(script, args):
                states.append((window.progress.value(), window.stage_progress.value()))
                record = PREFIX + json.dumps(dict(stage='analysis', completed=1, total=2, detail='Image Δ'))
                window.process.start(sys.executable, ['-c', 'import sys; print(sys.argv[1], flush=True); sys.exit(2)', record])

            window.start_process = fail_stage
            window.start_analysis()
            deadline = time.monotonic() + 10
            while window.busy and time.monotonic() < deadline:
                self.app.processEvents()
                time.sleep(.01)
            self.assertFalse(window.busy)
            self.assertEqual((window.progress.value(), window.stage_progress.value()), (45, 50))
            self.assertIn('Analysis failed', window.test_errors[-1])
            window.start_analysis()
            deadline = time.monotonic() + 10
            while window.busy and time.monotonic() < deadline:
                self.app.processEvents()
                time.sleep(.01)
            self.assertFalse(window.busy)
            self.assertEqual(states, [(0, 0), (0, 0)])
            # A preflight error is not a new run and retains the failed run's values.
            before = window.progress.value(), window.stage_progress.value()
            window.input_dir.setText('')
            window.start_analysis()
            self.assertEqual((window.progress.value(), window.stage_progress.value()), before)
            window.input_dir.setText(str(inputs))
            window.start_process = lambda script, args: window.process.start(str(root / 'missing-executable'))
            window.start_analysis()
            deadline = time.monotonic() + 10
            while window.busy and time.monotonic() < deadline:
                self.app.processEvents()
                time.sleep(.01)
            self.assertFalse(window.busy)
            self.assertEqual(window.progress.value(), 0)
            self.assertIn('Could not start the analysis process', window.test_errors[-1])
            self.assertFalse(window.elapsed_timer.isActive())

    def test_preview_has_no_overall_percentage(self):
        with tempfile.TemporaryDirectory() as directory:
            inputs, _, p = fixture(Path(directory))
            window = self.make_window()
            window.parameters = p
            window.load_controls()
            window.input_dir.setText(str(inputs))
            clock = window.analysis_clock = ControlledClock()
            window.start_elapsed_time()
            clock.milliseconds = 75_000
            window.finish_run(True)
            window.load_images()
            self.assertEqual(window.overall_percent.text(), '—')
            self.assertEqual(window.stage_progress.maximum(), 0)
            self.wait_until_idle(window)
            self.assertEqual(window.overall_percent.text(), '—')
            self.assertEqual(window.stage_percent.text(), '100%')
            self.assertEqual(window.stage_label.text(), 'Preview complete')
            self.assertEqual(window.elapsed_label.text(), 'Elapsed 00:01:15')
            self.assertFalse(window.elapsed_timer.isActive())
            self.assertEqual(clock.starts, 1)

    def test_explicit_parameter_save_and_raw_directory_protection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs, _, _ = fixture(root)
            window = self.make_window()
            window.input_dir.setText(str(inputs))
            window.groups.setText('Control, Treatment')
            destination = root / 'experiment.json'
            with patch('orion_gui.QFileDialog.getSaveFileName', return_value=(str(destination), 'JSON files (*.json)')):
                window.save_parameters()
            saved = json.loads(destination.read_text(encoding='utf-8'))
            self.assertEqual(saved['group_names'], ['Control', 'Treatment'])
            self.assertEqual(window.parameter_name.text(), 'experiment.json')
            forbidden = inputs / 'config.json'
            with patch('orion_gui.QFileDialog.getSaveFileName', return_value=(str(forbidden), 'JSON files (*.json)')):
                window.save_parameters()
            self.assertFalse(forbidden.exists())
            self.assertIn('outside the raw input directory', window.test_errors[-1])


if __name__ == '__main__':
    unittest.main()
