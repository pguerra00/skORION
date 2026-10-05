"""OME physical pixel sizes and explicit overrides preserve calibrated analysis."""
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
from image_io import inspect_image, validate_pixel_size

SCRIPTS=Path(__file__).resolve().parents[1]


class CalibrationTests(unittest.TestCase):
    def test_ome_units_and_override_priority(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'calibrated.ome.tif'
            tifffile.imwrite(path,np.zeros((3,16,20),dtype=np.uint16),ome=True,
                             photometric='minisblack',resolution=(72,72),resolutionunit='INCH',
                             metadata={'axes':'CYX','PhysicalSizeX':125,'PhysicalSizeXUnit':'nm',
                                       'PhysicalSizeY':.25,'PhysicalSizeYUnit':'µm'})
            before=path.read_bytes()
            md=inspect_image(path)
            self.assertAlmostEqual(md['pixel_size_x_um'],.125)
            self.assertAlmostEqual(md['pixel_size_y_um'],.25)
            self.assertIn('OME',md['calibration_source'])
            override=inspect_image(path,{'pixel_size_um':[.3,.4]})
            self.assertAlmostEqual(override['pixel_size_x_um'],.3)
            self.assertAlmostEqual(override['pixel_size_y_um'],.4)
            self.assertIn('Explicit',override['calibration_source'])
            self.assertEqual(path.read_bytes(),before)
            defaults=Path(tmp)/'default-units.ome.tif'
            tifffile.imwrite(defaults,np.zeros((3,16,20),dtype=np.uint16),ome=True,
                             photometric='minisblack',metadata={'axes':'CYX','PhysicalSizeX':.2,'PhysicalSizeY':.3})
            md=inspect_image(defaults)
            self.assertAlmostEqual(md['pixel_size_x_um'],.2)
            self.assertAlmostEqual(md['pixel_size_y_um'],.3)

    def test_incomplete_ome_and_invalid_overrides_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'incomplete.ome.tif'
            tifffile.imwrite(path,np.zeros((3,16,20),dtype=np.uint16),ome=True,
                             photometric='minisblack',metadata={'axes':'CYX','PhysicalSizeX':.2})
            with self.assertRaisesRegex(ValueError,'missing PhysicalSizeY'):
                inspect_image(path)
            # Known acquisition calibration can explicitly replace incomplete metadata.
            self.assertEqual(inspect_image(path,{'pixel_size_um':[.2,.2]})['pixel_size_y_um'],.2)
        for size in [[0,.2],[-.1,.2],[float('nan'),.2],[.2,float('inf')],
                     [.2],.2,['.2','.2'],[True,.2]]:
            with self.subTest(size=size),self.assertRaises(ValueError):
                validate_pixel_size(size)

    def test_missing_ome_scale_requires_and_records_explicit_calibration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            inputs=root/'input';inputs.mkdir()
            yy,xx=np.mgrid[:128,:128]
            nucleus=10+500*((yy-64)**2+(xx-64)**2<26**2)
            raw=np.array([nucleus,np.full((128,128),20),np.full((128,128),20)],dtype=np.uint16)
            path=inputs/'synthetic.ome.tif'
            tifffile.imwrite(path,raw,ome=True,photometric='minisblack',metadata={'axes':'CYX'})
            config=root/'parameters.json'
            config.write_bytes((SCRIPTS/'parameters.template.json').read_bytes())
            out=root/'output'
            command=[sys.executable,str(SCRIPTS/'analyze_foci.py'),'--input-dir',str(inputs),
                     '--parameters',str(config),'--output-dir',str(out)]
            before=path.read_bytes()
            result=subprocess.run(command,capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('--pixel-size-um X Y',result.stderr)
            self.assertFalse(out.exists())
            result=subprocess.run([*command,'--pixel-size-um','.2','.25','--foci-channels','2'],
                                  capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            mapping=pd.read_csv(out/'Methods/Treatment_Group_Mapping.csv')
            self.assertEqual(mapping.pixel_size_x_um.tolist(),[.2])
            self.assertEqual(mapping.pixel_size_y_um.tolist(),[.25])
            self.assertIn('Explicit',mapping.calibration_source.iloc[0])
            cells=pd.read_csv(out/'Results/Cell_Summary.csv')
            self.assertEqual(len(cells),1)
            np.testing.assert_allclose(cells.nuclear_area_um2,cells.nuclear_area_px*.2*.25)
            for name in ['Effective_Parameters','Run_Manifest']:
                saved=json.loads((out/f'Methods/{name}.json').read_text())
                self.assertEqual(saved['pixel_size_um'],[.2,.25])
            result=subprocess.run([sys.executable,str(SCRIPTS/'validate_results.py'),'--output-dir',str(out)],
                                  capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertEqual(path.read_bytes(),before)


if __name__=='__main__':
    unittest.main()
