"""Check histogram averaging, bounded legends, and large-folder rendering."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import warnings
import weakref

import numpy as np
import pandas as pd
from PIL import Image
import tifffile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import make_outputs as outputs


class InputIntensityTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.qdir=self.root/'QC'
        self.qdir.mkdir()
        self.params={'foci_channels':[1],'nuclear_channel':1,'channels':{'1':'Signal'}}

    def mapping(self,arrays,groups):
        rows=[]
        for i,(array,group) in enumerate(zip(arrays,groups),1):
            path=self.root/f'field_{i}.TIF'
            tifffile.imwrite(path,array[None],photometric='minisblack',metadata={'axes':'CYX'})
            rows.append({'image_id':f'I{i:02d}','treatment_group':group,'source_path':str(path)})
        return pd.DataFrame(rows)

    def test_equal_image_weight_zero_normalization_and_shared_axes(self):
        groups=['siControl + 100MMC','all / zero','siControl + 20PDS']
        arrays=[np.array([[0,0],[10,10]],dtype=np.uint16),np.full((3,4),100,dtype=np.uint16),
                np.zeros((2,3),dtype=np.uint16),np.full((3,3),50,dtype=np.uint16)]
        mapping=self.mapping(arrays,[groups[0],groups[0],groups[1],groups[2]])
        bins,records=outputs.input_intensity_distributions(mapping,1,self.params)
        widths=np.diff(bins)
        ten=np.searchsorted(bins,10,side='right')-1
        hundred=np.searchsorted(bins,100,side='right')-1
        expected=np.zeros(len(widths))
        expected[ten]=.5/widths[ten]
        np.testing.assert_allclose(records[0]['density'],expected)
        self.assertAlmostEqual(np.sum(records[0]['density']*widths),.5)
        self.assertAlmostEqual(np.sum(records[1]['density']*widths),1)
        self.assertEqual(records[0]['mean_intensity'],5)
        self.assertEqual(records[1]['mean_intensity'],100)
        self.assertTrue(records[2]['all_zero'])
        np.testing.assert_array_equal(records[2]['density'],0)
        average=np.zeros(len(widths))
        average[ten]=.25/widths[ten]
        average[hundred]=.5/widths[hundred]
        colors=['#0072B2','#D55E00','#009E73']
        seen={}

        def inspect(fig,folder,name):
            ax=fig.axes[0]
            j=int(name[-3:])-1
            self.assertEqual(folder,self.qdir)
            self.assertIn(groups[j],ax.get_title())
            self.assertEqual(ax.get_xscale(),'log')
            self.assertEqual(ax.get_yscale(),'log')
            self.assertEqual(len(ax.get_legend().get_texts()),3)
            self.assertEqual(ax.get_legend().get_texts()[1].get_text(),'Average distribution')
            for individual in ax.patches[:-1]:
                self.assertEqual(individual.get_alpha(),.25)
                np.testing.assert_allclose(individual.get_edgecolor()[:3],outputs.matplotlib.colors.to_rgb(colors[j]))
            np.testing.assert_allclose(ax.patches[-1].get_edgecolor()[:3],outputs.matplotlib.colors.to_rgb(colors[j]))
            self.assertGreater(ax.patches[-1].get_linewidth(),ax.patches[0].get_linewidth())
            if j==0:
                self.assertEqual(len(ax.patches),3)
                np.testing.assert_allclose(ax.patches[-1].get_data().values,average)
                self.assertEqual(ax.lines[0].get_linestyle(),'--')
                np.testing.assert_allclose(ax.lines[0].get_xdata(),[52.5,52.5])
                # Pooled pixels would give 76.25, so this proves equal image weight.
                self.assertIn('52.5',ax.get_legend().get_texts()[2].get_text())
            elif j==1:
                self.assertEqual(len(ax.lines),0)
                self.assertTrue(any('only zero-intensity pixels' in text.get_text() for text in ax.texts))
            else:
                np.testing.assert_allclose(ax.patches[0].get_data().values,ax.patches[-1].get_data().values)
                np.testing.assert_allclose(ax.lines[0].get_xdata(),[50,50])
            seen[name]=(ax.get_xlim(),ax.get_ylim(),ax.patches[-1].get_data().edges.copy())

        with patch.object(outputs,'COLORS',colors),patch.object(outputs,'figsave',side_effect=inspect), \
                warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            outputs.make_input_intensity_plots(self.qdir,mapping,self.params)
        self.assertFalse(caught,[str(w.message) for w in caught])
        self.assertEqual(len(seen),3)
        first=seen['C1_Input_Intensity_Group_001']
        for limits in seen.values():
            self.assertEqual(limits[:2],first[:2])
            np.testing.assert_array_equal(limits[2],first[2])
        self.assertEqual(outputs.plt.get_fignums(),[])

    def test_zero_image_contributes_to_group_average(self):
        arrays=[np.full((2,2),80,dtype=np.uint16),np.zeros((3,5),dtype=np.uint16)]
        mapping=self.mapping(arrays,['Mixed','Mixed'])

        def inspect(fig,*args):
            ax=fig.axes[0]
            average=ax.patches[-1].get_data()
            self.assertAlmostEqual(np.sum(average.values*np.diff(average.edges)),.5)
            np.testing.assert_allclose(ax.lines[0].get_xdata(),[40,40])
            self.assertTrue(any('1 all-zero image(s)' in text.get_text() for text in ax.texts))

        with patch.object(outputs,'figsave',side_effect=inspect):
            outputs.make_input_intensity_plots(self.qdir,mapping,self.params)

    def test_rgb_and_native_uint8_use_identical_distributions(self):
        rgb=np.zeros((4,5,3),dtype=np.uint8)
        rgb[...,0]=np.arange(20).reshape(4,5)*10
        rgb[...,1]=50
        rgb[...,2]=150
        rgb_path=self.root/'rgb.TIF'
        tifffile.imwrite(rgb_path,rgb,photometric='rgb')
        native_path=self.root/'native.TIF'
        tifffile.imwrite(native_path,np.moveaxis(rgb[..., [2,1,0]],-1,0),
                         photometric='minisblack',metadata={'axes':'CYX'})
        params={**self.params,'foci_channels':[1,2,3],'rgb_channel_order':['blue','green','red']}
        rows=[{'image_id':'RGB','treatment_group':'RGB','source_path':str(rgb_path)},
              {'image_id':'Native','treatment_group':'Native','source_path':str(native_path)}]
        mapping=pd.DataFrame(rows)
        for c,mean in [(1,150),(2,50),(3,95)]:
            bins,records=outputs.input_intensity_distributions(mapping,c,params)
            self.assertEqual(bins[-1],256)
            np.testing.assert_array_equal(records[0]['density'],records[1]['density'])
            self.assertEqual(records[0]['mean_intensity'],mean)
            self.assertEqual(records[1]['mean_intensity'],mean)
        outputs.make_input_intensity_plots(self.qdir,mapping,params)
        self.assertEqual(len(list(self.qdir.glob('*.png'))),6)
        self.assertEqual(len(list(self.qdir.glob('*.svg'))),6)

    def test_75_images_render_without_retaining_raw_arrays(self):
        groups=['siControl + 100MMC','siControl + 20PDS','siHELQ + 100MMC']
        arrays=[(np.arange(256).reshape(16,16)+i*50).astype(np.uint16) for i in range(75)]
        mapping=self.mapping(arrays,[groups[i%3] for i in range(75)])
        references=[]
        read=outputs.read_channels

        def checked_read(*args):
            self.assertTrue(all(ref() is None for ref in references),'A previous raw image was retained')
            raw=read(*args)
            references.append(weakref.ref(raw))
            return raw

        save=outputs.figsave
        rendered=[]

        def checked_save(fig,folder,name):
            ax=fig.axes[0]
            self.assertEqual(len(ax.patches),26)  # 25 images and one average.
            self.assertEqual(len(ax.get_legend().get_texts()),3)
            self.assertIn('25 image(s)',ax.get_title())
            rendered.append(name)
            return save(fig,folder,name)

        with patch.object(outputs,'read_channels',side_effect=checked_read), \
                patch.object(outputs,'figsave',side_effect=checked_save),warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            outputs.make_input_intensity_plots(self.qdir,mapping,self.params)
        self.assertFalse(caught,[str(w.message) for w in caught])
        self.assertEqual(len(references),150)
        self.assertTrue(all(ref() is None for ref in references))
        self.assertEqual(rendered,[f'C1_Input_Intensity_Group_{i:03d}' for i in range(1,4)])
        for name in rendered:
            with Image.open(self.qdir/f'{name}.png') as image:
                self.assertLess(image.width,3000)
                self.assertLess(image.height,1800)
            self.assertTrue((self.qdir/f'{name}.svg').exists())
        self.assertEqual(outputs.plt.get_fignums(),[])

    def test_all_zero_channel_renders_without_logarithmic_warnings(self):
        mapping=self.mapping([np.zeros((4,5),dtype=np.uint16)],['All zero'])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            outputs.make_input_intensity_plots(self.qdir,mapping,self.params)
        self.assertFalse(caught,[str(w.message) for w in caught])
        self.assertTrue((self.qdir/'C1_Input_Intensity_Group_001.png').exists())
        self.assertIn('Mean intensity: 0 ADU', (self.qdir/'C1_Input_Intensity_Group_001.svg').read_text())


if __name__=='__main__':
    unittest.main()
