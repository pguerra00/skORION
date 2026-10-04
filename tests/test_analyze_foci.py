"""Channel routing regression checks using small calibrated synthetic TIFFs.

Run: python -m unittest discover -s Scripts/tests -p 'test_analyze_foci.py'
"""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analyze_foci import select_channels, validate_group_names, metadata_row


SCRIPT = Path(__file__).resolve().parents[1] / 'analyze_foci.py'
PARAMETERS = json.loads(SCRIPT.with_name('parameters.template.json').read_text())


class ChannelSelectionTests(unittest.TestCase):
    def test_configured_groups_and_legacy_parser(self):
        md={'shape':[3,128,128],'bit_depth':16,'dtype':'uint16',
            'pixel_size_x_um':.2,'pixel_size_y_um':.2,'imagej_metadata':{}}
        groups=['siHELQ','MMC','PDS']
        validate_group_names(groups)
        with tempfile.TemporaryDirectory() as tmp:
            for name,expected in [('field_SIhelq_100mmc_01.tif',['siHELQ','MMC']),
                                  ('field_siHELQ_20PDS_02.tif',['siHELQ','PDS']),
                                  ('unmatched.tif',[])]:
                path=Path(tmp)/name
                path.write_bytes(b'fixture')
                row=metadata_row(path,1,md,groups)
                self.assertEqual(json.loads(row['group_names']),expected)
                self.assertEqual(row['treatment_group'],' + '.join(expected) if expected else 'AMBIGUOUS')
                self.assertIsNone(row['siRNA'])
            legacy=Path(tmp)/'63x_TP53KO_488_gH2AX_59453BP1_siCon_ART558_20PDS_2026-08-13_12_MAX.tif'
            legacy.write_bytes(b'fixture')
            self.assertEqual(metadata_row(legacy,1,md)['treatment_group'],'ART558_20PDS')
            self.assertEqual(metadata_row(legacy,1,md,[])['treatment_group'],'ART558_20PDS')
            # Explicit names override even a recognized legacy filename.
            self.assertEqual(metadata_row(legacy,1,md,['MMC'])['treatment_group'],'AMBIGUOUS')
        for invalid in [None,'MMC',{},[''],[' MMC'],[1],['MMC','mmc'],['AMBIGUOUS']]:
            with self.subTest(invalid=invalid),self.assertRaises(ValueError):
                validate_group_names(invalid)

    def test_parameter_defaults_overrides_and_errors(self):
        legacy = copy.deepcopy(PARAMETERS)
        legacy.pop('nuclear_channel')
        legacy.pop('foci_channels')
        self.assertEqual(select_channels(legacy), (1, [2, 3]))
        self.assertEqual(select_channels(PARAMETERS, 3, [2]), (3, [2]))
        for nuclear, foci in [(0, [2]), (1, []), (1, [2, 2]), (1, [-1]), (1, [1])]:
            with self.subTest(nuclear=nuclear, foci=foci), self.assertRaises(ValueError):
                select_channels(PARAMETERS, nuclear, foci)

    def test_channel_routing_and_empty_results(self):
        yy, xx = np.mgrid[:128, :128]
        nuclear = 10 + 500 * ((yy - 64)**2 + (xx - 64)**2 < 26**2)
        signal = np.full((128, 128), 20.0)
        for y, x in [(54, 55), (65, 73), (76, 57)]:
            signal += 250 * np.exp(-((yy - y)**2 + (xx - x)**2) / 8)
        original = np.array([nuclear, signal, signal * 0.5], dtype=np.uint16)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            def run(name, data, options=(), parameters=None, success=True):
                folder = root / name
                inputs = folder / 'input'
                inputs.mkdir(parents=True)
                tifffile.imwrite(inputs / 'synthetic.tif', data, imagej=True,
                                 resolution=(5, 5), metadata={'unit': 'micron', 'axes': 'CYX'})
                config = folder / 'parameters.json'
                config.write_text(json.dumps(parameters or PARAMETERS))
                output = folder / 'output'
                result = subprocess.run([sys.executable, str(SCRIPT), '--input-dir', str(inputs),
                                         '--output-dir', str(output), '--parameters', str(config),
                                         *options], capture_output=True, text=True)
                if success:
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                else:
                    self.assertNotEqual(result.returncode, 0)
                return output, result

            baseline, result = run('baseline', original)
            self.assertIn('Analysis: 1/1 images processed; 0 remaining',result.stdout)
            self.assertFalse((baseline / 'Review.html').exists())
            self.assertFalse((baseline / 'QC/Native_Overlays').exists())
            basecells = pd.read_csv(baseline / 'Results/Cell_Summary.csv')
            self.assertEqual(len(basecells), 1)
            self.assertGreater(basecells.C2_foci_count.iloc[0], 0)

            # Reorder the nuclear and C3 planes and select only the new C1 signal.
            p = copy.deepcopy(PARAMETERS)
            p['channels'] = {'1': '53BP1', '2': 'gH2AX', '3': 'DAPI'}
            p['foci']['channel_1'] = p['foci']['channel_3']
            swapped, _ = run('swapped', original[[2, 1, 0]],
                             ['--nuclear-channel', '3', '--foci-channels', '1'], p)
            for before, after in [('I01_nuclei.tif', 'I01_nuclei.tif'),
                                  ('I01_C3_foci.tif', 'I01_C1_foci.tif')]:
                np.testing.assert_array_equal(tifffile.imread(baseline / 'Masks' / before),
                                              tifffile.imread(swapped / 'Masks' / after))
            cells = pd.read_csv(swapped / 'Results/Cell_Summary.csv')
            self.assertEqual(cells.C1_foci_count.iloc[0], basecells.C3_foci_count.iloc[0])
            self.assertEqual(cells.nuclear_channel.iloc[0], 3)
            self.assertFalse(any(c.startswith(('C2_', 'C3_')) for c in cells))
            self.assertEqual(set(pd.read_csv(swapped / 'Results/Focus_Data.csv').channel), {1})
            self.assertFalse((swapped / 'Masks/I01_C2_foci.tif').exists())
            manifest = json.loads((swapped / 'Methods/Run_Manifest.json').read_text())
            self.assertEqual(manifest['foci_channels'], [1])
            self.assertEqual(manifest['nuclear_channel'], 3)
            effective = json.loads((swapped / 'Methods/Effective_Parameters.json').read_text())
            self.assertEqual(effective['foci_channels'], [1])

            # More than three planes, configured through JSON instead of CLI.
            p = copy.deepcopy(PARAMETERS)
            p['foci_channels'] = [4, 2]
            p['foci']['channel_4'] = p['foci']['channel_2']
            four, _ = run('four', np.concatenate([original, original[1:2]]), parameters=p)
            fourcells = pd.read_csv(four / 'Results/Cell_Summary.csv')
            self.assertEqual(fourcells.C4_foci_count.iloc[0], basecells.C2_foci_count.iloc[0])
            self.assertEqual(pd.read_csv(four / 'Methods/Treatment_Group_Mapping.csv').n_channels.iloc[0], 4)

            for name, data in [('no_foci', np.array([nuclear, signal * 0 + 20], dtype=np.uint16)),
                               ('no_nuclei', np.zeros((2, 128, 128), dtype=np.uint16))]:
                empty, _ = run(name, data, ['--foci-channels', '2'])
                self.assertTrue(pd.read_csv(empty / 'Results/Focus_Data.csv').empty)
                self.assertEqual(pd.read_csv(empty / 'Results/Image_Summary.csv').C2_total_foci.iloc[0], 0)

            invalid, result = run('invalid', original, ['--nuclear-channel', '4'], success=False)
            self.assertIn('channel 4 is unavailable', result.stderr)
            self.assertFalse(invalid.exists())
            p=copy.deepcopy(PARAMETERS)
            p['group_names']='MMC'
            invalid,result=run('invalid_groups',original,parameters=p,success=False)
            self.assertIn('group_names must be a list',result.stderr)
            self.assertFalse(invalid.exists())
            invalid,result=run('invalid_qc',original,['--qc-output','unknown'],success=False)
            self.assertIn('invalid choice',result.stderr)
            self.assertFalse(invalid.exists())


if __name__ == '__main__':
    unittest.main()
