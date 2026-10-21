"""End-to-end colocalization exports, validation, reports and GUI settings."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from html.parser import HTMLParser

import numpy as np
import pandas as pd
import tifffile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gui_analysis import analyze_preview, render_image
from progress_reporting import parse_progress
from colocalization import SCHEMAS

PARAMETERS = json.loads((ROOT / 'parameters.template.json').read_text(encoding='utf-8'))


def fixture(root, rgb=False):
    inputs = root / 'input'; inputs.mkdir()
    yy, xx = np.mgrid[:128, :128]
    nucleus = 10 + 180 * ((yy - 64) ** 2 + (xx - 64) ** 2 < 26 ** 2)
    signal = np.full((128, 128), 10.)
    for y, x in [(54, 55), (65, 73), (76, 57)]:
        signal += 200 * np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / 8)
    pixels = np.array([nucleus, signal, signal], dtype=np.uint8)
    p = copy.deepcopy(PARAMETERS)
    p['colocalization']['enabled'] = True
    p['group_names'] = ['Control']
    p['channels'] = {'1': 'Nuclei', '2': 'Marker <A>', '3': 'Marker B'}
    p['rgb_channel_order'] = ['blue', 'green', 'red'] if rgb else None
    for name, array in [('Control_01.TIF', pixels),
                        ('Control_02.TIF', np.stack([pixels[0], pixels[1] * 0, pixels[2] * 0])),
                        ('Control_03.TIF', pixels * 0)]:
        if rgb:
            tifffile.imwrite(inputs / name, np.moveaxis(array[[2, 1, 0]], 0, 2),
                             photometric='rgb', resolution=(127000, 127000), resolutionunit='INCH')
        else:
            tifffile.imwrite(inputs / name, array, imagej=True, resolution=(5, 5),
                             metadata={'unit': 'micron', 'axes': 'CYX'})
    return inputs, p


class Links(HTMLParser):
    def __init__(self):
        super().__init__(); self.links = []

    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if key in ('href', 'src') and not value.startswith(('http', '#')):
                self.links.append(value.split('#')[0])


class PipelineTests(unittest.TestCase):
    def run_script(self, script, *args, success=True):
        result = subprocess.run([sys.executable, '-X', 'utf8', str(ROOT / script), *map(str, args)],
                                capture_output=True, text=True)
        if success:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0)
        return result

    def test_preview_exports_report_and_independent_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            inputs, p = fixture(root)
            config = root / 'parameters.json'; config.write_text(json.dumps(p), encoding='utf-8')
            original = {path.name: path.read_bytes() for path in inputs.iterdir()}
            out = root / 'output'
            run = self.run_script('analyze_foci.py', '--input-dir', inputs, '--output-dir', out,
                                  '--parameters', config, '--progress-json')
            events = [parse_progress(line) for line in run.stdout.splitlines()]
            self.assertEqual([e for e in events if e][-1]['completed'], 13)
            tables = {name: pd.read_csv(out / 'Results' / f'Colocalization_{name}.csv') for name in SCHEMAS}
            self.assertEqual(len(tables['Images']), 3)
            self.assertEqual(tables['Images'].pair_count.tolist(), [3, 0, 0])
            self.assertEqual(len(tables['Foci']), 6)
            self.assertTrue(tables['Foci'].is_colocalized.all())
            self.assertTrue((tables['Foci'].overlap_fraction == 1).all())
            self.assertEqual(tables['Images'].iloc[2].analysis_status, 'no_usable_nuclei')
            preview = analyze_preview(inputs / 'Control_01.TIF', p)
            for name, frame in preview.colocalization.tables().items():
                saved = tables[name][tables[name].image_id == 'I01']
                cols = [c for c in frame.select_dtypes(include='number') if c != 'replicate_id']
                np.testing.assert_allclose(frame[cols].to_numpy(float), saved[cols].to_numpy(float), rtol=1e-9, atol=1e-9)
            display = render_image(preview.pixels, p, preview, mode='Colocalization')
            self.assertEqual(display.shape, (128, 128, 3))
            self.run_script('validate_results.py', '--output-dir', out)
            generated = self.run_script('make_outputs.py', '--output-dir', out, '--qc-output', 'minimal', '--progress-json')
            last = [parse_progress(line) for line in generated.stdout.splitlines() if parse_progress(line)][-1]
            self.assertEqual(last['completed'], last['total'])
            self.run_script('validate_results.py', '--output-dir', out)
            review = (out / 'Review.html').read_text(encoding='utf-8')
            self.assertLess(review.index('<h2>Files and reproducibility'), review.index('<h2>Colocalization analysis'))
            self.assertLess(review.index('<h2>Colocalization analysis'), review.index('<h2>Inspect every image'))
            self.assertNotIn('No cross-channel colocalization', review)
            self.assertIn('Marker &lt;A&gt;', review)
            links = Links(); links.feed(review)
            self.assertTrue(all((out / link).exists() for link in links.links),
                            [link for link in links.links if not (out / link).exists()])
            dictionary = pd.read_csv(out / 'Methods/Data_Dictionary.csv')
            self.assertTrue(dictionary.table.eq('Colocalization_Pairs').any())
            methods = (out / 'Methods/Methods.md').read_text(encoding='utf-8')
            self.assertIn('float32', methods)
            self.assertIn('MatCol', methods)
            self.assertNotIn('No cross-channel colocalization', methods)
            manifest = json.loads((out / 'Methods/Run_Manifest.json').read_text())
            self.assertEqual(manifest['colocalization'], p['colocalization'])
            self.assertEqual(len(manifest['colocalization_source_sha256']), 64)
            # Validation must detect corrupt pair geometry, not just reconcile totals.
            corrupt = tables['Pairs'].copy(); corrupt.loc[0, 'overlap_area_px'] += 1
            corrupt.to_csv(out / 'Results/Colocalization_Pairs.csv', index=False)
            failure = self.run_script('validate_results.py', '--output-dir', out, success=False)
            self.assertIn('pair areas', failure.stderr)
            self.assertEqual({path.name: path.read_bytes() for path in inputs.iterdir()}, original)

    def test_rgb_equivalence_and_disabled_results(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            original_tables = None
            for rgb in (False, True):
                workspace = root / str(rgb); workspace.mkdir()
                inputs, p = fixture(workspace, rgb)
                preview = analyze_preview(inputs / 'Control_01.TIF', p)
                if original_tables is None:
                    original_tables = preview.colocalization.tables()
                else:
                    for name, table in preview.colocalization.tables().items():
                        pd.testing.assert_frame_equal(original_tables[name], table)
                on = workspace / 'on'; off = workspace / 'off'
                config = workspace / 'parameters.json'
                for enabled, output in [(True, on), (False, off)]:
                    p['colocalization']['enabled'] = enabled
                    config.write_text(json.dumps(p), encoding='utf-8')
                    self.run_script('analyze_foci.py', '--input-dir', inputs, '--output-dir', output, '--parameters', config)
                    self.run_script('validate_results.py', '--output-dir', output)
                for name in ['Focus_Data', 'Cell_Summary', 'Image_Summary', 'QC_Summary']:
                    pd.testing.assert_frame_equal(pd.read_csv(on / 'Results' / f'{name}.csv'),
                                                  pd.read_csv(off / 'Results' / f'{name}.csv'))
                self.assertEqual(list((off / 'Results').glob('Colocalization*')), [])

    def test_reassigned_channels_and_empty_association_report(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            inputs, p = fixture(root)
            for path in inputs.iterdir():
                array = tifffile.imread(path)
                tifffile.imwrite(path, array[[2, 0, 1]], imagej=True, resolution=(5, 5),
                                 metadata={'unit': 'micron', 'axes': 'CYX'})
            p['nuclear_channel'] = 2
            p['foci_channels'] = [3, 1]
            p['foci']['channel_1'] = copy.deepcopy(p['foci']['channel_3'])
            p['foci']['channel_3'] = copy.deepcopy(p['foci']['channel_2'])
            p['colocalization']['channels'] = [3, 1]
            p['colocalization']['minimum_overlap_pixels'] = 100000
            config = root / 'parameters.json'; config.write_text(json.dumps(p), encoding='utf-8')
            output = root / 'output'
            self.run_script('analyze_foci.py', '--input-dir', inputs, '--parameters', config,
                            '--output-dir', output, '--qc-output', 'minimal')
            self.run_script('validate_results.py', '--output-dir', output)
            pairs = pd.read_csv(output / 'Results/Colocalization_Pairs.csv')
            foci = pd.read_csv(output / 'Results/Colocalization_Foci.csv')
            self.assertTrue(pairs.empty)
            self.assertEqual(set(foci.channel), {1, 3})
            self.assertFalse(foci.is_colocalized.any())
            self.assertTrue((foci.partner_count == 0).all())
            self.assertTrue((foci.nearest_opposite_distance_um == 0).all())
            self.assertTrue((output / 'Plots/Colocalization/Coverage_Distances.png').exists())

    def test_entire_runs_without_foci_or_accepted_nuclei(self):
        for source_name, nuclei_expected in [('Control_02.TIF', 1), ('Control_03.TIF', 0)]:
            with self.subTest(source=source_name), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                inputs, p = fixture(root)
                for path in inputs.iterdir():
                    if path.name != source_name:
                        path.unlink()
                config = root / 'parameters.json'; config.write_text(json.dumps(p), encoding='utf-8')
                output = root / 'output'
                self.run_script('analyze_foci.py', '--input-dir', inputs, '--parameters', config,
                                '--output-dir', output, '--qc-output', 'minimal')
                self.run_script('validate_results.py', '--output-dir', output)
                self.assertTrue(pd.read_csv(output / 'Results/Colocalization_Foci.csv').empty)
                self.assertTrue(pd.read_csv(output / 'Results/Colocalization_Pairs.csv').empty)
                image = pd.read_csv(output / 'Results/Colocalization_Images.csv').iloc[0]
                self.assertEqual(image.nuclei_analyzed, nuclei_expected)
                self.assertEqual(image.pair_count, 0)
                self.assertTrue(pd.isna(image.A_colocalized_fraction))
                self.assertTrue((output / 'QC/Colocalization/I01_Colocalization.png').is_file())


@unittest.skipUnless(importlib.util.find_spec('PySide6') is not None, 'Install GUI dependencies to test desktop controls')
class GuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault('QT_QPA_PLATFORM', 'windows' if sys.platform == 'win32' else 'offscreen')
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def test_optional_controls_persistence_preview_and_channel_errors(self):
        from orion_gui import OrionWindow
        window = OrionWindow()
        self.addCleanup(window.close)
        self.assertFalse(window.coloc_enabled.isChecked())
        self.assertTrue(window.coloc_container.isHidden())
        self.assertEqual([window.coloc_a.currentData(), window.coloc_b.currentData()], [2, 3])
        window.coloc_enabled.setChecked(True)
        self.assertFalse(window.coloc_container.isHidden())
        self.assertEqual(window.view_tabs.tabText(window.view_tabs.count() - 1), 'Colocalization')
        window.coloc_b.setCurrentIndex(window.coloc_b.findData(2))
        with self.assertRaises(ValueError):
            window.collect_parameters()
        window.coloc_b.setCurrentIndex(window.coloc_b.findData(3))
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            inputs, p = fixture(root)
            config = root / 'saved.json'
            window.coloc_minimum.setValue(2)
            with patch('orion_gui.QFileDialog.getSaveFileName', return_value=(str(config), 'JSON files (*.json)')):
                window.save_parameters()
            window.coloc_enabled.setChecked(False)
            window.parameter_dir.setText(str(root)); window.parameter_name.setText(config.name)
            window.load_parameters()
            self.assertTrue(window.coloc_enabled.isChecked())
            self.assertEqual(window.coloc_minimum.value(), 2)
            window.parameters = p; window.load_controls()
            preview = analyze_preview(inputs / 'Control_01.TIF', p)
            window.preview_complete(preview)
            window.nucleus_picker.setCurrentIndex(window.nucleus_picker.findData(1))
            self.assertIn('Colocalization: 3 relationships', window.nucleus_info.text())
            self.assertIn('Pearson raw / corrected', window.nucleus_info.text())
            window.show()
            window.resize(1060, 800)
            self.app.processEvents()
            self.assertEqual(window.nucleus_scroll.horizontalScrollBar().maximum(), 0)
            self.assertGreater(window.nucleus_scroll.verticalScrollBar().maximum(), 0)
            self.assertTrue(any(window.table.horizontalHeaderItem(i).text() == 'coloc_pair_count'
                                for i in range(window.table.columnCount())))
            window.coloc_minimum.setValue(3)
            self.assertIsNone(window.preview)
            window.foci_checks[3].setChecked(False)
            self.assertIsNone(window.coloc_b.currentData())
            with self.assertRaises(ValueError):
                window.collect_parameters()
            window.coloc_enabled.setChecked(False)
            self.assertTrue(window.coloc_container.isHidden())
            self.assertFalse(window.collect_parameters()['colocalization']['enabled'])
            snapshot = root / 'analysis_parameters.json'
            snapshot.write_text(json.dumps(p), encoding='utf-8')
            output = root / 'saved_run'
            run = subprocess.run([sys.executable, str(ROOT / 'analyze_foci.py'), '--input-dir', str(inputs),
                                  '--parameters', str(snapshot), '--output-dir', str(output)], capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            window.load_saved_results(output)
            self.assertTrue(any(window.table.horizontalHeaderItem(i).text() == 'coloc_pair_count'
                                for i in range(window.table.columnCount())))


if __name__ == '__main__':
    unittest.main()
