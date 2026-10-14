"""Adaptive seed pruning preserves small objects and genuinely separated peaks."""
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

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from analyze_foci import detect_nucleus,select_channels
from gui_analysis import analyze_preview,validate_parameters

ROOT=Path(__file__).resolve().parents[1]
PARAMETERS=json.loads((ROOT/'parameters.template.json').read_text(encoding='utf-8'))


class AdaptiveSplittingTests(unittest.TestCase):
    def setUp(self):
        self.p=copy.deepcopy(PARAMETERS['foci'])
        self.corr=np.zeros((80,120),dtype=float)
        # Broad bright region with a shallow secondary peak: two seeds -> one.
        self.corr[10:28,10:40]=80
        self.corr[15:20,14:20]=120
        self.corr[15:20,27:33]=100
        # Two faint isolated foci stay separate and keep their original masks.
        self.corr[45:49,10:14]=14
        self.corr[45:49,18:22]=18
        # Two peaks connected through a deep valley remain separate.
        self.corr[10:28,65:105]=7
        self.corr[15:20,70:76]=120
        self.corr[15:20,90:96]=100
        self.raw=self.corr+100
        self.bg=np.full_like(self.raw,100)
        self.nucleus=np.ones_like(self.raw,dtype=bool)

    def detect(self,p=None,channel=3,pixel_area=.01):
        return detect_nucleus(self.raw,self.bg,self.corr,self.nucleus,channel,
                              self.p if p is None else p,pixel_area)

    def enable(self):
        self.p['channel_3'].update(adaptive_split_prominence_fraction=.3,
                                   adaptive_split_minimum_peak_adu=100,
                                   adaptive_split_minimum_area_um2=2)

    def test_large_shallow_peaks_merge_small_and_deep_valley_peaks_survive(self):
        original,_,_,_=self.detect()
        self.assertEqual(original.max(),6)
        self.enable()
        tuned,_,_,thresholds=self.detect()
        self.assertEqual(tuned.max(),5)
        self.assertEqual(tuned[16,16],tuned[16,29])
        self.assertNotEqual(tuned[16,72],tuned[16,92])
        self.assertNotEqual(tuned[46,11],tuned[46,19])
        np.testing.assert_array_equal(original[40:]>0,tuned[40:]>0)
        self.assertEqual(thresholds['prominence_threshold_adu'],6)
        self.assertEqual(thresholds['adaptive_split_suppressed_seeds'],1)
        self.assertEqual(thresholds['adaptive_split_maximum_tested_prominence_adu'],30)

    def test_disabled_rule_is_identical_and_selected_channel_only(self):
        legacy=copy.deepcopy(self.p)
        for key in list(legacy['channel_3']):
            if key.startswith('adaptive_split_'):legacy['channel_3'].pop(key)
        original,_,_,applied=self.detect(legacy)
        disabled,_,_,disabled_applied=self.detect()
        np.testing.assert_array_equal(original,disabled)
        self.assertEqual(applied,disabled_applied)
        before,_,_,before_applied=self.detect(channel=2)
        self.enable()
        after,_,_,after_applied=self.detect(channel=2)
        np.testing.assert_array_equal(before,after)
        self.assertEqual(before_applied,after_applied)

    def test_area_brightness_and_strongest_seed_guards(self):
        original,_,_,_=self.detect()
        self.enable()
        self.p['channel_3']['adaptive_split_minimum_peak_adu']=150
        np.testing.assert_array_equal(original,self.detect()[0])
        self.p['channel_3']['adaptive_split_minimum_peak_adu']=100
        # At a smaller calibrated pixel area, the support regions are below 2 µm².
        np.testing.assert_array_equal(original,self.detect(pixel_area=.001)[0])
        self.p['channel_3']['adaptive_split_prominence_fraction']=1
        self.assertGreater(self.detect()[0][16,16],0)
        empty=np.zeros_like(self.corr)
        result=detect_nucleus(self.bg,self.bg,empty,self.nucleus,3,self.p,.01)
        self.assertEqual(result[0].max(),0)
        self.assertEqual(result[3]['adaptive_split_suppressed_seeds'],0)

    def test_merged_basin_trim_override_leaves_other_objects_identical(self):
        # Dim edge lies below the default 25% cutoff of the merged peak.
        self.corr[10:28,8:10]=20
        self.raw=self.corr+100
        self.enable()
        original,_,_,_=self.detect()
        self.p['channel_3']['adaptive_split_post_watershed_peak_fraction']=.15
        tuned,_,_,applied=self.detect()
        self.assertEqual(original.max(),tuned.max())
        self.assertGreater(np.count_nonzero(tuned[:30,:45]),np.count_nonzero(original[:30,:45]))
        np.testing.assert_array_equal(original[40:],tuned[40:])
        np.testing.assert_array_equal(original[:30,60:],tuned[:30,60:])
        self.assertEqual(applied['adaptive_split_post_watershed_peak_fraction'],.15)

    def test_validation_cli_and_gui(self):
        p=copy.deepcopy(PARAMETERS)
        for key,values in {
            'adaptive_split_prominence_fraction':[-.1,1.01,None,True,float('nan')],
            'adaptive_split_minimum_peak_adu':[-1,None,'100',float('inf')],
            'adaptive_split_minimum_area_um2':[-1,None,False],
            'adaptive_split_post_watershed_peak_fraction':[-1,1.1,None,False],
        }.items():
            for value in values:
                invalid=copy.deepcopy(p);invalid['foci']['channel_3'][key]=value
                with self.subTest(key=key,value=value),self.assertRaisesRegex(ValueError,key):
                    select_channels(invalid)
                with self.assertRaisesRegex(ValueError,key):validate_parameters(invalid)

    def test_saved_audit_and_preview_match_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);inputs=root/'raw';inputs.mkdir()
            yy,xx=np.mgrid[:128,:128]
            nucleus=np.where((yy-64)**2+(xx-64)**2<42**2,500,100)
            focus=np.zeros_like(nucleus,dtype=float)
            focus[45:65,40:80]=80
            focus[50:56,45:51]=120;focus[50:56,66:72]=100
            focus[80:84,53:57]=14;focus[80:84,62:66]=18
            path=inputs/'test.TIF'
            tifffile.imwrite(path,np.array([nucleus,focus+100,focus+100],dtype=np.uint16),
                             imagej=True,resolution=(10,10),metadata={'axes':'CYX','unit':'micron'})
            p=copy.deepcopy(PARAMETERS)
            p['foci_channels']=[3]
            p['foci']['channel_3'].update(adaptive_split_prominence_fraction=.3,
                                          adaptive_split_minimum_peak_adu=100,
                                          adaptive_split_minimum_area_um2=2,
                                          adaptive_split_post_watershed_peak_fraction=.15)
            preview=analyze_preview(path,p)
            config=root/'parameters.json';config.write_text(json.dumps(p))
            before=path.read_bytes()
            result=subprocess.run([sys.executable,str(ROOT/'analyze_foci.py'),
                                   '--input-dir',str(inputs),'--output-dir',str(root/'out'),
                                   '--parameters',str(config),'--qc-output','minimal'],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            np.testing.assert_array_equal(preview.foci_masks[3],tifffile.imread(root/'out/Masks/I01_C3_foci.tif'))
            applied=pd.read_csv(root/'out/Methods/Applied_Thresholds.csv')
            self.assertEqual(applied.adaptive_split_prominence_fraction.tolist(),[.3])
            self.assertEqual(applied.adaptive_split_post_watershed_peak_fraction.tolist(),[.15])
            self.assertEqual(applied.adaptive_split_suppressed_seeds.tolist(),
                             [preview.thresholds[0]['adaptive_split_suppressed_seeds']])
            self.assertEqual(path.read_bytes(),before)
            methods=(root/'out/Methods/Methods.md').read_text(encoding='utf-8')
            self.assertIn('Adaptive splitting is enabled for C3: fraction 0.3',methods)
            self.assertIn('minimum corrected peak 100 ADU',methods)
            self.assertIn('merged-basin trimming at 15%',methods)
            self.assertIn('Each support region retains its strongest original seed',methods)


if __name__=='__main__':unittest.main()
