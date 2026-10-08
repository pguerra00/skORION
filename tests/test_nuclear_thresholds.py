"""Fixed nuclear ranges use original intensities, with auditable bounds."""
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

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from analyze_foci import nuclear_segmentation, validate_nuclear_threshold_range

SCRIPTS=Path(__file__).resolve().parents[1]
PARAMETERS=json.loads((SCRIPTS/'parameters.template.json').read_text())


class NuclearThresholdTests(unittest.TestCase):
    def test_range_validation(self):
        self.assertIsNone(validate_nuclear_threshold_range({}))
        self.assertIsNone(validate_nuclear_threshold_range({'fixed_threshold_adu':None}))
        for bounds in [[131,255],[131,131],[0,65535]]:
            self.assertEqual(validate_nuclear_threshold_range({'fixed_threshold_adu':bounds}),tuple(bounds))
        for bounds in [131,'131-255',{},[],[131],[131,255,256],[255,131],[-1,255],
                       [True,255],[131,False],[131,'255'],[float('nan'),255],[131,float('inf')]]:
            with self.subTest(bounds=bounds),self.assertRaisesRegex(ValueError,'nuclear.fixed_threshold_adu'):
                validate_nuclear_threshold_range({'fixed_threshold_adu':bounds})

    def test_inclusive_raw_bounds_bypass_intensity_smoothing_and_background_offset(self):
        raw=np.full((100,260),130,dtype=np.uint16)
        for j,value in enumerate([130,131,255,256]):
            raw[20:60,20+60*j:60+60*j]=value
        p=copy.deepcopy(PARAMETERS['nuclear'])
        p['fixed_threshold_adu']=[131,255]
        p['closing_disk_radius_px']=0
        expected=(raw>=131)&(raw<=255)
        original=raw.copy()
        first=None
        for sigma in [2,7]:
            p['gaussian_sigma_px']=sigma
            usable,candidates,audit=nuclear_segmentation(raw,p,.2,.2)
            np.testing.assert_array_equal(usable>0,expected)
            np.testing.assert_array_equal(candidates>0,expected)
            np.testing.assert_array_equal(raw,original)
            self.assertEqual(len(audit),2)
            self.assertTrue(all(row['included'] for row in audit))
            self.assertTrue(all(row['nuclear_threshold_method']=='fixed_raw' for row in audit))
            self.assertTrue(all(row['DAPI_threshold_adu']==131 and row['nuclear_threshold_upper_adu']==255 for row in audit))
            if first is not None:np.testing.assert_array_equal(usable,first)
            first=usable

    def test_missing_and_null_ranges_preserve_adaptive_masks(self):
        yy,xx=np.mgrid[:128,:128]
        raw=np.uint16(10+500*((yy-64)**2+(xx-64)**2<26**2))
        p=copy.deepcopy(PARAMETERS['nuclear'])
        p.pop('fixed_threshold_adu',None)
        legacy=nuclear_segmentation(raw,p,.2,.2)
        p['fixed_threshold_adu']=None
        current=nuclear_segmentation(raw,p,.2,.2)
        np.testing.assert_array_equal(current[0],legacy[0])
        np.testing.assert_array_equal(current[1],legacy[1])
        self.assertEqual(current[2],legacy[2])
        self.assertTrue(all(row['nuclear_threshold_method']=='adaptive' and row['nuclear_threshold_upper_adu'] is None for row in current[2]))

    def test_rgb_analysis_records_fixed_range_and_reports_active_method(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);inputs=root/'input';inputs.mkdir()
            yy,xx=np.mgrid[:128,:128]
            blue=np.uint8(30+180*((yy-64)**2+(xx-64)**2<26**2))
            green=np.uint8(20+100*np.exp(-((yy-60)**2+(xx-60)**2)/8))
            rgb=np.stack([np.full_like(blue,20),green,blue],axis=-1)
            path=inputs/'field_NT.TIF'
            tifffile.imwrite(path,rgb,photometric='rgb',resolution=(127000,127000),resolutionunit='INCH')
            original=path.read_bytes()
            p=copy.deepcopy(PARAMETERS)
            p['nuclear']['fixed_threshold_adu']=[131,255]
            p['group_names']=['NT']
            p['rgb_channel_order']=['blue','green','red']
            config=root/'parameters.json';config.write_text(json.dumps(p))
            out=root/'output'
            env=dict(os.environ,MPLCONFIGDIR=str(root/'mpl'))

            def run(script,*args):
                result=subprocess.run([sys.executable,str(SCRIPTS/script),*map(str,args)],
                                      capture_output=True,text=True,env=env)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)
                return result

            run('analyze_foci.py','--input-dir',inputs,'--parameters',config,'--output-dir',out,'--qc-output','minimal')
            run('validate_results.py','--output-dir',out)
            self.assertEqual(path.read_bytes(),original)
            audit=pd.read_csv(out/'QC/Nucleus_Inclusion_Audit.csv')
            cells=pd.read_csv(out/'Results/Cell_Summary.csv')
            self.assertEqual(len(cells),1)
            for table in [audit,cells]:
                self.assertEqual(table.nuclear_threshold_method.unique().tolist(),['fixed_raw'])
                self.assertTrue(table.nuclear_threshold_adu.eq(131).all())
                self.assertTrue(table.nuclear_threshold_upper_adu.eq(255).all())
            self.assertEqual(cells.nuclear_stain_mean_adu.iloc[0],210)
            saved=json.loads((out/'Methods/Effective_Parameters.json').read_text())
            self.assertEqual(saved['nuclear']['fixed_threshold_adu'],[131,255])
            self.assertEqual(saved['foci'],p['foci'])
            methods=(out/'Methods/Methods.md').read_text()
            self.assertIn('fixed inclusive range 131–255 ADU',methods)
            self.assertIn('Gaussian intensity smoothing and background subtraction are not used for foreground selection',methods)
            self.assertNotIn('foreground threshold is the full-image smoothed',methods)
            dictionary=pd.read_csv(out/'Methods/Data_Dictionary.csv').set_index('column')
            self.assertIn('Inclusive upper bound',dictionary.loc['nuclear_threshold_upper_adu','definition'])

            p['nuclear']['fixed_threshold_adu']=[255,131]
            config.write_text(json.dumps(p))
            invalid=root/'invalid_output'
            result=subprocess.run([sys.executable,str(SCRIPTS/'analyze_foci.py'),
                                   '--input-dir',str(inputs),'--parameters',str(config),'--output-dir',str(invalid)],
                                  capture_output=True,text=True,env=env)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('nuclear.fixed_threshold_adu',result.stderr)
            self.assertFalse(invalid.exists())


if __name__=='__main__':
    unittest.main()
