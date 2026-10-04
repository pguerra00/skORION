"""End-to-end checks for channel-aware plots, reports, and validation."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from channel_config import resolve_channels
from make_outputs import expand_groups

METHODS = Path(__file__).resolve().parents[1]
PARAMETERS = json.loads((METHODS / 'parameters.template.json').read_text())


class DownstreamChannelTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def run_script(self, script, *options, success=True):
        env = dict(os.environ, MPLCONFIGDIR=str(self.root / 'mpl'))
        result = subprocess.run([sys.executable, str(METHODS / script), *map(str, options)],
                                capture_output=True, text=True, env=env)
        if success:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0)
        return result

    def fixture(self, *, no_nuclei=False, channels=(1, 4), qc_output='off',
                filenames=('synthetic.tif',), group_names=None):
        inputs = self.root / 'input'
        inputs.mkdir()
        yy, xx = np.mgrid[:128, :128]
        nucleus = 10 + 500 * ((yy - 64)**2 + (xx - 64)**2 < 26**2)
        signal = np.full((128, 128), 20.0)
        for y, x in [(54, 55), (65, 73), (76, 57)]:
            signal += 250 * np.exp(-((yy - y)**2 + (xx - x)**2) / 8)
        arr = np.array([signal, signal * 0 + 20, nucleus, signal * 0 + 20], dtype=np.uint16)
        if no_nuclei:
            arr[:] = 20
        for name in filenames:
            tifffile.imwrite(inputs / name, arr, imagej=True, resolution=(5, 5),
                             metadata={'unit': 'micron', 'axes': 'CYX'})
        p = copy.deepcopy(PARAMETERS)
        p['nuclear_channel'] = 3
        p['foci_channels'] = list(channels)
        p['channels'] = {'1': 'Focus A', '3': 'Nuclear stain', '4': 'Focus B'}
        p['foci']['channel_1'] = p['foci']['channel_2']
        p['foci']['channel_4'] = p['foci']['channel_2']
        if group_names is not None:p['group_names']=group_names
        config = self.root / 'config.json'
        config.write_text(json.dumps(p))
        out = self.root / 'output'
        self.analysis_result=self.run_script('analyze_foci.py', '--input-dir', inputs, '--output-dir', out,
                                             '--parameters', config, '--qc-output', qc_output)
        return out

    def test_automatic_minimal_qc_and_overlapping_groups(self):
        out=self.fixture(channels=(1,),qc_output='minimal',group_names=['siHELQ','MMC','PDS'],
                         filenames=('field_siHELQ_MMC.tif','field_siHELQ_PDS.tif'))
        self.assertFalse((out/'QC/Native_Overlays').exists())
        self.assertFalse((out/'QC/Nucleus_Crops').exists())
        for name in ['QC/I01_Annotated_QC.png','QC/I02_Annotated_QC.png','QC/C1_Shared_Contrast.png',
                     'Plots/Channel_1/Image_Level_Endpoints.svg','Methods/workbook_data.json','Report.md']:
            self.assertTrue((out/name).exists(),name)
        review=(out/'Review.html').read_text()
        self.assertNotIn('Native_Overlays',review)
        self.assertNotIn('Nucleus_Crops',review)
        self.assertIn('2 images · 3 groups · 2 usable nuclei',review)
        methods=(out/'Methods/Methods.md').read_text()
        self.assertIn('Minimal QC was requested',methods)
        self.assertNotIn('Nucleus_Crops contains',methods)
        for label in ['Analysis','QC']:
            self.assertIn(f'{label}: 1/2 images processed; 1 remaining',self.analysis_result.stdout)
            self.assertIn(f'{label}: 2/2 images processed; 0 remaining',self.analysis_result.stdout)
        mapping=pd.read_csv(out/'Methods/Treatment_Group_Mapping.csv')
        self.assertEqual(mapping.group_names.map(json.loads).tolist(),[['siHELQ','MMC'],['siHELQ','PDS']])
        cells=pd.read_csv(out/'Results/Cell_Summary.csv')
        self.assertEqual(len(cells),2)
        self.assertEqual(expand_groups(cells).groupby('treatment_group').size().to_dict(),
                         {'MMC':1,'PDS':1,'siHELQ':2})
        self.assertEqual(cells.nucleus_id.nunique(),2)
        focus=pd.read_csv(out/'Results/Focus_Data.csv')
        self.assertEqual(focus.focus_id.nunique(),len(focus))
        svg=(out/'Plots/Channel_1/Focus_Positive_Nuclei.svg').read_text()
        self.assertIn('I01: 1/1 nuclei',svg)
        self.assertNotIn('2/1 nuclei',svg)
        for group in ['siHELQ','MMC','PDS']:self.assertIn(group,svg)
        self.assertEqual(json.loads((out/'Methods/Output_Parameters.json').read_text())['qc_output'],'minimal')
        self.run_script('validate_results.py','--output-dir',out)

    def test_automatic_detailed_qc(self):
        out=self.fixture(channels=(1,),qc_output='detailed')
        self.assertTrue((out/'QC/Native_Overlays/I01_C1_foci_outlines.png').exists())
        self.assertTrue((out/'QC/Native_Overlays/I01_C3_nuclear_outlines.png').exists())
        self.assertTrue((out/'QC/Nucleus_Crops/I01_N001.png').exists())
        self.assertIn('QC/Nucleus_Crops/I01_N001.png',(out/'Review.html').read_text())
        self.assertEqual(json.loads((out/'Methods/Run_Manifest.json').read_text())['qc_output'],'detailed')
        # Switching to minimal hides previous detailed files without deleting them.
        self.run_script('make_outputs.py','--output-dir',out,'--qc-output','minimal')
        self.assertNotIn('Native_Overlays',(out/'Review.html').read_text())
        self.assertNotIn('Nucleus_Crops',(out/'Review.html').read_text())
        self.assertTrue((out/'QC/Nucleus_Crops/I01_N001.png').exists())

    def test_group_expansion_legacy_unmatched_and_empty(self):
        legacy=pd.DataFrame({'treatment_group':['old']})
        pd.testing.assert_frame_equal(expand_groups(legacy),legacy)
        unmatched=pd.DataFrame({'treatment_group':['AMBIGUOUS'],'group_names':['[]']})
        self.assertEqual(expand_groups(unmatched).treatment_group.tolist(),['AMBIGUOUS'])
        pd.testing.assert_frame_equal(expand_groups(unmatched.iloc[:0]),unmatched.iloc[:0])

    def test_saved_selection_and_explicit_subset(self):
        out = self.fixture()
        self.run_script('validate_results.py', '--output-dir', out)
        report = json.loads((out / 'Methods/Validation_Report.json').read_text())
        self.assertEqual(report['nuclear_channel'], 3)
        self.assertEqual(report['foci_channels'], [1, 4])
        self.assertEqual(report['counts']['channel_4_foci'], 0)
        self.assertGreater(report['counts']['channel_1_foci'], 0)
        self.run_script('make_outputs.py', '--output-dir', out)
        text = (out / 'Review.html').read_text()
        self.assertIn('C3 (Nuclear stain)', text)
        self.assertIn('Channel 1 · Focus A', text)
        self.assertIn('Channel 4 · Focus B', text)
        self.assertNotIn('Channel_2', text)
        self.assertTrue((out / 'QC/Native_Overlays/I01_C3_nuclear_outlines.png').exists())
        self.assertTrue((out / 'Plots/Channel_4/Focus_Properties.svg').exists())
        ranges = pd.read_csv(out / 'QC/QC_Display_Ranges.csv')
        self.assertEqual(set(ranges.channel), {1, 3, 4})
        methods = (out / 'Methods/Methods.md').read_text()
        self.assertIn('Nuclear segmentation uses C3 (Nuclear stain)', methods)
        self.assertNotIn('C2 floors', methods)
        self.run_script('make_outputs.py', '--output-dir', out, '--nuclear-channel', 3, '--foci-channels', 1)
        self.run_script('validate_results.py', '--output-dir', out, '--nuclear-channel', 3, '--foci-channels', 1)
        self.assertNotIn('Channel_4', (out / 'Review.html').read_text())
        self.assertEqual(json.loads((out / 'Methods/Validation_Report.json').read_text())['foci_channels'], [1])
        for script in ['make_outputs.py', 'validate_results.py']:
            result = self.run_script(script, '--output-dir', out, '--nuclear-channel', 2, success=False)
            self.assertIn('Saved nuclear masks use channel 3', result.stderr)
            result = self.run_script(script, '--output-dir', out, '--foci-channels', 2, success=False)
            self.assertIn('were not analyzed', result.stderr)

    def test_empty_analysis_and_shared_channel_roles(self):
        out = self.fixture(no_nuclei=True, channels=(3,))
        self.run_script('validate_results.py', '--output-dir', out)
        self.run_script('make_outputs.py', '--output-dir', out)
        native = out / 'QC/Native_Overlays'
        self.assertTrue((native / 'I01_C3_nuclear_outlines.png').exists())
        self.assertTrue((native / 'I01_C3_foci_outlines.png').exists())
        self.assertEqual(json.loads((out / 'Methods/Validation_Report.json').read_text())['counts']['nuclei'], 0)

    def test_configuration_precedence_and_legacy(self):
        folder = self.root / 'Methods'
        folder.mkdir()
        self.assertEqual(resolve_channels(self.root)['foci_channels'], [2, 3])
        (folder / 'parameters.json').write_text(json.dumps({'nuclear_channel': 1, 'foci_channels': [2, 3]}))
        (folder / 'Effective_Parameters.json').write_text(json.dumps({'nuclear_channel': 4, 'foci_channels': [1, 2]}))
        self.assertEqual(resolve_channels(self.root)['nuclear_channel'], 4)
        self.assertEqual(resolve_channels(self.root, 4, [2])['foci_channels'], [2])
        for kwargs in [{'nuclear_channel': 0}, {'foci_channels': [2, 2]}, {'foci_channels': []}]:
            with self.assertRaises(ValueError):
                resolve_channels(self.root, **kwargs)


if __name__ == '__main__':
    unittest.main()
