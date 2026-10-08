"""End-to-end checks for channel-aware plots, reports, and validation."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from channel_config import resolve_channels
import make_outputs as outputs

METHODS = Path(__file__).resolve().parents[1]
PARAMETERS = json.loads((METHODS / 'parameters.template.json').read_text())


def svg_text(path):
    return [''.join(node.itertext()).strip() for node in ET.parse(path).iter('{http://www.w3.org/2000/svg}text')]


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

    def test_automatic_minimal_qc_and_combined_treatment_groups(self):
        out=self.fixture(channels=(1,),qc_output='minimal',group_names=['siControl','100MMC','20PDS'],
                         filenames=('field_siControl_100MMC.tif','field_siControl_20PDS.tif'))
        self.assertFalse((out/'QC/Native_Overlays').exists())
        self.assertFalse((out/'QC/Nucleus_Crops').exists())
        for name in ['QC/I01_Annotated_QC.png','QC/I02_Annotated_QC.png','QC/C1_Shared_Contrast.png',
                     'Plots/Channel_1/Image_Level_Endpoints.svg','Methods/workbook_data.json','Report.md']:
            self.assertTrue((out/name).exists(),name)
        review=(out/'Review.html').read_text()
        self.assertNotIn('Native_Overlays',review)
        self.assertNotIn('Nucleus_Crops',review)
        self.assertIn('2 images · 2 groups · 2 usable nuclei',review)
        self.assertIn('Comparisons use combined treatment groups',review)
        self.assertNotIn('Groups can overlap',review)
        methods=(out/'Methods/Methods.md').read_text()
        self.assertIn('Minimal QC was requested',methods)
        self.assertNotIn('Nucleus_Crops contains',methods)
        for index,name in enumerate(['field_siControl_100MMC.tif','field_siControl_20PDS.tif'],1):
            self.assertIn(f'Analyzing image {index}/2: {name}',self.analysis_result.stdout)
            self.assertIn(f'QC: {index}/2 images processed (I{index:02d})',self.analysis_result.stdout)
        self.assertIn('Analysis complete: 2/2 images processed; 0 remaining',self.analysis_result.stdout)
        mapping=pd.read_csv(out/'Methods/Treatment_Group_Mapping.csv')
        self.assertEqual(mapping.group_names.map(json.loads).tolist(),[['siControl','100MMC'],['siControl','20PDS']])
        groups=['siControl + 100MMC','siControl + 20PDS']
        self.assertEqual(mapping.treatment_group.tolist(),groups)
        cells=pd.read_csv(out/'Results/Cell_Summary.csv')
        self.assertEqual(len(cells),2)
        self.assertEqual(cells.groupby('treatment_group').size().to_dict(),
                         {'siControl + 100MMC':1,'siControl + 20PDS':1})
        self.assertEqual(cells.nucleus_id.nunique(),2)
        focus=pd.read_csv(out/'Results/Focus_Data.csv')
        self.assertEqual(focus.focus_id.nunique(),len(focus))
        # Inspect rendered categories and annotations, rather than just CSV rows.
        for path in (out/'Plots/Channel_1').glob('*.svg'):
            labels=svg_text(path)
            for group in groups:
                self.assertTrue(group in labels or any(label.startswith(f'{group} (n=') for label in labels),path.name)
            for component in ['siControl','100MMC','20PDS']:
                self.assertFalse(component in labels or any(label.startswith(f'{component} (n=') for label in labels),path.name)
        positive=svg_text(out/'Plots/Channel_1/Focus_Positive_Nuclei.svg')
        for im in ['I01','I02']:self.assertEqual(positive.count(f'{im}: 1/1 nuclei'),1)
        endpoints=svg_text(out/'Plots/Channel_1/Image_Level_Endpoints.svg')
        for im in ['I01','I02']:self.assertEqual(endpoints.count(im),16)
        ecdf=svg_text(out/'Plots/Channel_1/Distribution_ECDF.svg')
        for group in groups:
            self.assertEqual(ecdf.count(f'{group} (n=1)'),1)
            n=len(focus[focus.treatment_group==group])
            self.assertEqual(ecdf.count(f'{group} (n={n})'),2)
        for index,group in enumerate(groups,1):
            stem=f'C1_Input_Intensity_Group_{index:03d}'
            histogram=svg_text(out/'QC'/f'{stem}.svg')
            self.assertIn(f'{group} · 1 image(s)',histogram)
            self.assertEqual(histogram.count('Individual images (n=1)'),1)
            self.assertEqual(histogram.count('Average distribution'),1)
            self.assertEqual(sum(label.startswith('Mean intensity:') for label in histogram),1)
            self.assertIn(f'href="QC/{stem}.png"',review)
            self.assertIn(f'src="QC/{stem}.png"',review)
            self.assertIn(f'href="QC/{stem}.svg"',review)
        self.assertNotIn('Input_Intensity_Histogram',review)
        self.assertIn('Each image receives equal weight in both averages',methods)
        self.assertIn('comparison uses treatment_group',methods)
        self.assertNotIn('Overlapping groups',methods)
        self.assertIn('Comparisons use combined treatment_group labels',(out/'Report.md').read_text())
        self.assertEqual(json.loads((out/'Methods/Output_Parameters.json').read_text())['qc_output'],'minimal')
        self.run_script('validate_results.py','--output-dir',out)

    def test_automatic_detailed_qc(self):
        out=self.fixture(channels=(1,),qc_output='detailed',filenames=('synthetic.TIF',))
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

    def test_legacy_tables_without_group_names(self):
        out=self.fixture(channels=(1,),filenames=(
            '63x_TP53KO_488_gH2AX_59453BP1_siCon_20PDS_2026-08-13_12_MAX.tif',))
        tables=[out/'Methods/Treatment_Group_Mapping.csv',*(out/'Results').glob('*.csv')]
        for path in tables:
            table=pd.read_csv(path)
            table.drop(columns='group_names',errors='ignore').to_csv(path,index=False)
        before={path:path.read_bytes() for path in tables}
        old_histogram=out/'QC/C1_Input_Intensity_Histogram.svg'
        old_histogram.write_text('Previous combined histogram')
        self.run_script('make_outputs.py','--output-dir',out,'--qc-output','minimal')
        self.assertIn('20PDS',svg_text(out/'Plots/Channel_1/Focus_Positive_Nuclei.svg'))
        self.assertIn('1 images · 1 groups · 1 usable nuclei',(out/'Review.html').read_text())
        for path,content in before.items():self.assertEqual(path.read_bytes(),content)
        self.assertEqual(old_histogram.read_text(),'Previous combined histogram')
        self.assertNotIn(old_histogram.name,(out/'Review.html').read_text())
        self.run_script('validate_results.py','--output-dir',out)

    def test_small_scatter_markers_and_rotated_treatment_labels(self):
        self.addCleanup(outputs.plt.close,'all')
        out=self.fixture(channels=(1,),group_names=['siControl','100MMC','20PDS'],
                         filenames=('field_siControl_100MMC.tif','field_siControl_20PDS.tif'))
        cells=pd.read_csv(out/'Results/Cell_Summary.csv')
        focus=pd.read_csv(out/'Results/Focus_Data.csv')
        images=pd.read_csv(out/'Results/Image_Summary.csv')
        groups=['siControl + 100MMC','siControl + 20PDS']
        # Exercise both sides of the previous 100-point size cutoff.
        cells=pd.concat([cells[cells.treatment_group==groups[0]],
                         *[cells[cells.treatment_group==groups[1]]]*101],ignore_index=True)
        focus=pd.concat([focus[focus.treatment_group==groups[0]],
                         *[focus[focus.treatment_group==groups[1]]]*101],ignore_index=True)
        names=[]

        def inspect(fig,folder,name):
            names.append(name)
            for ax in fig.axes:
                for collection in ax.collections:
                    np.testing.assert_array_equal(collection.get_sizes(),[8])
                if name=='Distribution_ECDF':
                    self.assertTrue(all(tick.get_rotation()==0 for tick in ax.get_xticklabels()))
                    continue
                self.assertEqual([tick.get_text() for tick in ax.get_xticklabels()],groups)
                for tick in ax.get_xticklabels():
                    self.assertEqual(tick.get_rotation(),45)
                    self.assertEqual(tick.get_ha(),'right')
                    self.assertEqual(tick.get_rotation_mode(),'anchor')
                if (name.startswith('Cellular_') or name=='Focus_Properties') and len(ax.collections):
                    self.assertEqual(ax.collections[0].get_alpha(),.65)
                    self.assertEqual(ax.collections[1].get_alpha(),.18)
            outputs.plt.close(fig)

        with patch.object(outputs,'GROUPS',groups),patch.object(outputs,'LABELS',groups), \
                patch.object(outputs,'COLORS',['#0072B2','#D55E00']), \
                patch.object(outputs,'MARKERS',{'I01':'s','I02':'o'}), \
                patch.object(outputs,'figsave',side_effect=inspect), \
                patch.object(outputs,'make_input_intensity_plots'):
            outputs.make_plots(out,cells,focus,images,resolve_channels(out))
        self.assertEqual(len(names),7)
        self.assertEqual(outputs.plt.get_fignums(),[])

    def test_single_combined_and_unmatched_groups_with_empty_measurements(self):
        out=self.fixture(no_nuclei=True,channels=(1,),qc_output='minimal',
                         group_names=['siControl','100MMC'],
                         filenames=('field_siControl.tif','field_siControl_100MMC.tif','unmatched.tif'))
        groups=['siControl','siControl + 100MMC','AMBIGUOUS']
        self.assertIn('3 images · 3 groups · 0 usable nuclei',(out/'Review.html').read_text())
        self.assertTrue(pd.read_csv(out/'Results/Cell_Summary.csv').empty)
        self.assertTrue(pd.read_csv(out/'Results/Focus_Data.csv').empty)
        labels=svg_text(out/'Plots/Channel_1/Image_Level_Endpoints.svg')
        for group in groups:self.assertIn(group,labels)
        self.assertNotIn('100MMC',labels)
        ecdf=svg_text(out/'Plots/Channel_1/Distribution_ECDF.svg')
        for group in groups:self.assertEqual(ecdf.count(f'{group} (n=0)'),3)

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
