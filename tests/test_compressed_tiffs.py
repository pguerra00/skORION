"""LZW decoding preserves native pixels and GUI/CLI analysis results."""
import copy
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import tifffile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gui_analysis import analyze_preview
from image_io import read_image


class CompressedTIFFTests(unittest.TestCase):
    def test_lzw_preserves_stack_and_rgb_pixels_and_calibration(self):
        rng = np.random.default_rng(42)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stack = rng.integers(0, 65536, (3, 32, 40), dtype=np.uint16)
            path = root / 'channels.TIF'
            tifffile.imwrite(path, stack, imagej=True, compression='lzw', predictor=True,
                             resolution=(5, 4), metadata={'axes': 'CYX', 'unit': 'micron'})
            before = path.read_bytes()
            pixels, metadata = read_image(path)
            np.testing.assert_array_equal(pixels, stack)
            self.assertEqual(pixels.dtype, np.uint16)
            self.assertEqual(metadata['pixel_size_x_um'], .2)
            self.assertEqual(metadata['pixel_size_y_um'], .25)
            self.assertEqual(path.read_bytes(), before)

            rgb = rng.integers(0, 256, (32, 40, 3), dtype=np.uint8)
            for planar in ('contig', 'separate'):
                with self.subTest(planar=planar):
                    path = root / f'{planar}.TIF'
                    stored = rgb if planar == 'contig' else np.moveaxis(rgb, 2, 0)
                    tifffile.imwrite(path, stored, photometric='rgb', planarconfig=planar,
                                     compression='lzw', predictor=True,
                                     resolution=(50000, 40000), resolutionunit='CENTIMETER')
                    before = path.read_bytes()
                    pixels, metadata = read_image(path, {'rgb_channel_order': ['blue', 'green', 'red']})
                    np.testing.assert_array_equal(pixels, np.moveaxis(rgb, 2, 0)[[2, 1, 0]])
                    self.assertEqual(pixels.dtype, np.uint8)
                    self.assertEqual(metadata['pixel_size_x_um'], .2)
                    self.assertEqual(metadata['pixel_size_y_um'], .25)
                    self.assertEqual(path.read_bytes(), before)

    def test_compressed_analysis_matches_uncompressed_and_records_decoder(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            yy, xx = np.mgrid[:128, :128]
            nuclear = 10 + 160 * ((yy - 64) ** 2 + (xx - 64) ** 2 < 26 ** 2)
            signal = np.full((128, 128), 20.)
            for y, x in ((54, 55), (65, 73), (76, 57)):
                signal += 180 * np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / 8)
            stack = np.array([nuclear, signal, signal * .6], dtype=np.uint8)
            rgb = np.moveaxis(stack[[2, 1, 0]], 0, 2)
            parameters = copy.deepcopy(json.loads((ROOT / 'parameters.template.json').read_text(encoding='utf-8')))
            parameters.update(rgb_channel_order=['blue', 'green', 'red'], qc_output='off')
            config = root / 'parameters.json'
            config.write_text(json.dumps(parameters))
            plain = root / 'plain.TIF'
            inputs = root / 'inputs'
            inputs.mkdir()
            compressed = inputs / 'compressed.TIF'
            for path, compression in ((plain, None), (compressed, 'lzw')):
                tifffile.imwrite(path, rgb, photometric='rgb', compression=compression,
                                 resolution=(50000, 50000), resolutionunit='CENTIMETER')
            before = compressed.read_bytes()
            baseline = analyze_preview(plain, parameters)
            preview = analyze_preview(compressed, parameters)
            np.testing.assert_array_equal(preview.pixels, baseline.pixels)
            np.testing.assert_array_equal(preview.nuclei, baseline.nuclei)
            for channel in parameters['foci_channels']:
                np.testing.assert_array_equal(preview.foci_masks[channel], baseline.foci_masks[channel])
            pd.testing.assert_frame_equal(pd.DataFrame(preview.cells), pd.DataFrame(baseline.cells))
            pd.testing.assert_frame_equal(pd.DataFrame(preview.foci), pd.DataFrame(baseline.foci))
            self.assertGreater(len(preview.foci), 0)

            output = root / 'output'
            for script, args in (
                ('analyze_foci.py', ['--input-dir', inputs, '--parameters', config, '--output-dir', output]),
                ('make_outputs.py', ['--output-dir', output, '--qc-output', 'minimal']),
                ('validate_results.py', ['--output-dir', output]),
            ):
                result = subprocess.run([sys.executable, str(ROOT / script), *map(str, args)],
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            np.testing.assert_array_equal(tifffile.imread(output / 'Masks/I01_nuclei.tif'), baseline.nuclei)
            for channel in parameters['foci_channels']:
                np.testing.assert_array_equal(tifffile.imread(output / f'Masks/I01_C{channel}_foci.tif'),
                                              baseline.foci_masks[channel])
            manifest = json.loads((output / 'Methods/Run_Manifest.json').read_text(encoding='utf-8'))
            self.assertEqual(manifest['imagecodecs'], version('imagecodecs'))
            requirements = (output / 'Methods/requirements.txt').read_text(encoding='utf-8').splitlines()
            self.assertIn(f'imagecodecs=={version("imagecodecs")}', requirements)
            self.assertEqual(compressed.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
