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
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import tifffile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gui_analysis import analyze_preview, list_images, render_image, validate_run_paths
from analyze_foci import nuclear_segmentation

QT_AVAILABLE = importlib.util.find_spec('PySide6') is not None
if QT_AVAILABLE:
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from PySide6.QtWidgets import QApplication
    from orion_gui import OrionWindow

PARAMETERS = json.loads((ROOT / 'parameters.template.json').read_text())


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

    def test_neutral_channels_advanced_roundtrip_and_role_selection(self):
        window = self.make_window()
        self.assertEqual([window.nuclear.itemText(i) for i in range(3)], ['Channel 1', 'Channel 2', 'Channel 3'])
        roundtrip = window.collect_parameters()
        for key, value in PARAMETERS.items():
            self.assertEqual(roundtrip[key], value, key)
        window.nuclear.setCurrentIndex(2)
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

    def test_background_preview_and_complete_analysis_with_channel_three_nuclei(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs, path, p = fixture(root)
            config = root / 'config.json'
            config.write_text(json.dumps(p))
            window = self.make_window()
            window.parameter_dir.setText(str(root))
            window.parameter_name.setText(config.name)
            window.load_parameters()
            window.input_dir.setText(str(inputs))
            window.output_dir.setText(str(root / 'output'))
            before = config.read_bytes(), path.read_bytes()
            window.load_images()
            self.wait_until_idle(window)
            self.assertEqual(window.preview.parameters['nuclear_channel'], 3)
            self.assertEqual(set(window.preview.foci_masks), {1, 2})
            window.nucleus_picker.setCurrentIndex(1)
            self.assertIn('Nuclear Channel 3', window.nucleus_info.text())
            self.assertIn('Channel 1:', window.nucleus_info.text())
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
            window.start_analysis()
            self.wait_until_idle(window)
            self.assertEqual(window.progress.value(), 1)
            self.assertEqual(window.table_origin, 'saved')
            self.assertTrue((root / 'output/Methods/Validation_Report.json').is_file())
            self.assertTrue((root / 'output/Review.html').is_file())
            output_settings = json.loads((root / 'output/Methods/Output_Parameters.json').read_text())
            self.assertEqual(output_settings['qc_output'], 'minimal')
            methods = (root / 'output/Methods/Methods.md').read_text()
            for description in ['internal holes are retained', '25 to 275 square microns',
                                'Gaussian sigma 1.25 px', '4 x sigma_smooth',
                                '30% of its corrected peak', 'At least 10 annular pixels',
                                'after 11 px dilation']:
                self.assertIn(description, methods)
            effective = json.loads((root / 'output/Methods/Effective_Parameters.json').read_text())
            self.assertEqual(effective['nuclear_channel'], 3)
            self.assertEqual(effective['foci_channels'], [1, 2])
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
        self.assertEqual(window.nuclear.currentText(), 'Channel 4')
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
        with self.assertRaisesRegex(ValueError, 'RGB mapping'):
            window.collect_parameters()
        for color, value in zip(window.rgb_colors, ['green', 'blue', 'red']):
            color.setCurrentIndex(color.findData(value))
        self.assertEqual(window.collect_parameters()['rgb_channel_order'], ['green', 'blue', 'red'])
        window.rgb_colors[2].setCurrentIndex(window.rgb_colors[2].findData('blue'))
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
            self.assertEqual(window.collect_parameters()['rgb_channel_order'], ['green', 'blue', 'red'])

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
            effective = json.loads((root / 'output/Methods/Effective_Parameters.json').read_text())
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

            def slow_stage(script, args):
                stages.append(script)
                window.process.start(sys.executable, ['-c', 'import time; time.sleep(20)'])

            window.start_process = slow_stage
            window.start_analysis()
            deadline = time.monotonic() + 5
            while window.process.state() == window.process.ProcessState.Starting and time.monotonic() < deadline:
                self.app.processEvents()
                time.sleep(.01)
            window.stop_analysis()
            self.wait_until_idle(window)
            self.assertEqual(stages, ['analyze_foci.py'])
            self.assertEqual(window.process.state(), window.process.ProcessState.NotRunning)
            self.assertIsNone(window.run_temp)

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
            saved = json.loads(destination.read_text())
            self.assertEqual(saved['group_names'], ['Control', 'Treatment'])
            self.assertEqual(window.parameter_name.text(), 'experiment.json')
            forbidden = inputs / 'config.json'
            with patch('orion_gui.QFileDialog.getSaveFileName', return_value=(str(forbidden), 'JSON files (*.json)')):
                window.save_parameters()
            self.assertFalse(forbidden.exists())
            self.assertIn('outside the raw input directory', window.test_errors[-1])


if __name__ == '__main__':
    unittest.main()
