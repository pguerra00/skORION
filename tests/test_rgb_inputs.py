"""RGB exports preserve pixels, calibration, measurements, and report provenance."""
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
from image_io import read_image, inspect_image, validate_rgb_channel_order

SCRIPTS=Path(__file__).resolve().parents[1]
ORDER={'rgb_channel_order':['blue','green','red']}


class RGBInputTests(unittest.TestCase):
    def test_color_mapping_and_physical_calibration(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            rgb=np.zeros((16,20,3),dtype=np.uint8)
            rgb[...,0]=11;rgb[...,1]=22;rgb[...,2]=33
            for planar in ['contig','separate']:
                with self.subTest(planar=planar):
                    path=folder/f'{planar}.TIF'
                    stored=rgb if planar=='contig' else np.moveaxis(rgb,2,0)
                    tifffile.imwrite(path,stored,photometric='rgb',planarconfig=planar,
                                     resolution=(127000,63500),resolutionunit='INCH')
                    raw,md=read_image(path,ORDER)
                    self.assertEqual(raw.dtype,np.uint8)
                    self.assertEqual(raw.shape,(3,16,20))
                    np.testing.assert_array_equal(raw[:,0,0],[33,22,11])
                    self.assertAlmostEqual(md['pixel_size_x_um'],.2)
                    self.assertAlmostEqual(md['pixel_size_y_um'],.4)
                    self.assertEqual(md['bit_depth'],8)
                    self.assertEqual(md['saturation_value'],255)
                    self.assertEqual(md['input_format'],'rgb_export')
                    self.assertEqual(md['rgb_channel_order'],ORDER['rgb_channel_order'])
                    with self.assertRaisesRegex(ValueError,'--rgb-channel-order'):
                        inspect_image(path)
            path=folder/'uncalibrated.TIF'
            tifffile.imwrite(path,rgb,photometric='rgb')
            with self.assertRaisesRegex(ValueError,'Unknown spatial calibration'):
                inspect_image(path,ORDER)
            for order in [['blue','blue','red'],['red','green'],['cyan','green','blue'],'blue green red']:
                with self.subTest(order=order),self.assertRaises(ValueError):
                    validate_rgb_channel_order(order)

    def test_rgb_matches_equivalent_cyx_and_generates_detailed_qc(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            yy,xx=np.mgrid[:128,:128]
            nuclear=10+160*((yy-64)**2+(xx-64)**2<26**2)
            signal=np.full((128,128),20.)
            for y,x in [(54,55),(65,73),(76,57)]:
                signal+=180*np.exp(-((yy-y)**2+(xx-x)**2)/8)
            signal[53:56,54:57]=255
            cyx=np.array([nuclear,signal,signal*.6],dtype=np.uint8)
            rgb=np.moveaxis(cyx[[2,1,0]],0,2)
            params=copy.deepcopy(json.loads((SCRIPTS/'parameters.template.json').read_text()))
            params['channels']={'1':'Nuclear stain','2':'gH2AX','3':'RAD51'}
            config=root/'parameters.json';config.write_text(json.dumps(params))

            def run(script,*args):
                result=subprocess.run([sys.executable,str(SCRIPTS/script),*map(str,args)],
                                      capture_output=True,text=True)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)

            outputs={}
            for layout in ['cyx','rgb']:
                inputs=root/layout;inputs.mkdir()
                path=inputs/'synthetic.TIF'
                if layout=='cyx':
                    tifffile.imwrite(path,cyx,imagej=True,resolution=(5,5),
                                     metadata={'axes':'CYX','unit':'micron'})
                    options=[]
                else:
                    tifffile.imwrite(path,rgb,photometric='rgb',resolution=(127000,127000),resolutionunit='INCH')
                    (inputs/'._synthetic.TIF').write_bytes(b'\x00\x05\x16\x07'+b'\x00'*22)
                    options=['--rgb-channel-order',*ORDER['rgb_channel_order']]
                before=path.read_bytes()
                out=root/f'{layout}_output';outputs[layout]=out
                run('analyze_foci.py','--input-dir',inputs,'--parameters',config,'--output-dir',out,*options)
                run('validate_results.py','--output-dir',out)
                self.assertEqual(path.read_bytes(),before)
            for name in ['Image_Summary','Cell_Summary','Focus_Data','QC_Summary']:
                baseline=pd.read_csv(outputs['cyx']/f'Results/{name}.csv')
                measured=pd.read_csv(outputs['rgb']/f'Results/{name}.csv')
                pd.testing.assert_frame_equal(measured,baseline)
            focus=pd.read_csv(outputs['rgb']/'Results/Focus_Data.csv')
            self.assertGreater(len(focus),0)
            self.assertGreater(focus.saturated_pixels.sum(),0)
            self.assertLessEqual(focus.raw_max_adu.max(),255)
            qc=pd.read_csv(outputs['rgb']/'Results/QC_Summary.csv')
            self.assertGreater(qc.percent_saturated_pixels_image.max(),0)
            mapping=pd.read_csv(outputs['rgb']/'Methods/Treatment_Group_Mapping.csv')
            self.assertEqual(mapping.bit_depth.tolist(),[8])
            self.assertEqual(mapping.saturation_value.tolist(),[255])
            self.assertEqual(mapping.input_format.tolist(),['rgb_export'])
            self.assertEqual(json.loads(mapping.rgb_channel_order.iloc[0]),ORDER['rgb_channel_order'])
            run('make_outputs.py','--output-dir',outputs['rgb'],'--qc-output','detailed')
            for name in ['QC/I01_Annotated_QC.png','QC/C2_Shared_Contrast.png',
                         'QC/Native_Overlays/I01_C1_nuclear_outlines.png',
                         'QC/Native_Overlays/I01_C2_foci_outlines.png','QC/Nucleus_Crops/I01_N001.png']:
                self.assertTrue((outputs['rgb']/name).exists(),name)
            methods=(outputs['rgb']/'Methods/Methods.md').read_text()
            self.assertIn('blue, green, red',methods)
            self.assertIn('0–255',methods)
            self.assertIn('do not recover original detector ADU',methods)
            self.assertIn('RGB export analysis',(outputs['rgb']/'Review.html').read_text())
            manifest=json.loads((outputs['rgb']/'Methods/Run_Manifest.json').read_text())
            self.assertEqual(manifest['rgb_channel_order'],ORDER['rgb_channel_order'])
            self.assertIn('image_reader_sha256',manifest)


if __name__=='__main__':
    unittest.main()
