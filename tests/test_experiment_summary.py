"""Check summary observation units, channel pairing, exports, and layout."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import warnings
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import make_outputs as outputs


class ExperimentSummaryTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(outputs.plt.close,'all')
        self.groups=['siHELQ + 100MMC','siControl + 20PDS']
        for name,value in [('GROUPS',self.groups),('LABELS',self.groups),
                           ('COLORS',list(outputs.TREATMENT_PALETTE[:2]))]:
            patcher=patch.object(outputs,name,value);patcher.start();self.addCleanup(patcher.stop)

    def fixture(self,channels=(4,1)):
        params={'foci_channels':list(channels),'channels':{'4':'Focus B','1':'Focus A','2':'Focus C'},
                'random_seed_for_plot_jitter':42}
        mapping=pd.DataFrame({'image_id':['I01','I02','I03'],
                              'treatment_group':[self.groups[0],self.groups[0],self.groups[1]]})
        images=mapping.copy();images['nuclei_analyzed']=[4,1,1]
        cells=pd.DataFrame({'image_id':['I01']*4+['I02','I03'],
                            'nucleus_id':[f'N{i}' for i in range(6)],
                            'treatment_group':[self.groups[0]]*5+[self.groups[1]]})
        foci=[]
        for c in channels:
            images[f'C{c}_mean_foci_per_nucleus']=[1,9,0]
            images[f'C{c}_mean_focus_area_um2']=[2,8,np.nan]
            images[f'C{c}_mean_focus_intensity_adu']=[20,80,np.nan]
            images[f'C{c}_percent_focus_positive_nuclei']=[25,100,0]
            images[f'C{c}_total_foci']=[4,9,0]
            cells[f'C{c}_foci_count']=[0,0,0,4,9,0] if c==4 else [0,1,2,3,9,0]
            for im,areas in [('I01',[1,2,2,3]),('I02',[8]*9)]:
                for area in areas:
                    foci.append({'image_id':im,'treatment_group':self.groups[0],'channel':c,
                                 'focus_area_um2':area,'raw_mean_adu':10*area})
        focus=pd.DataFrame(foci,columns=['image_id','treatment_group','channel','focus_area_um2','raw_mean_adu'])
        return mapping,images,cells,focus,params

    def draw(self,data):
        figures=[]
        def inspect(fig,folder,name):
            self.assertEqual(name,'Experiment_Summary')
            fig.canvas.draw();figures.append(fig)
        with patch.object(outputs,'figsave',side_effect=inspect),warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            outputs.make_experiment_summary(Path('/private/tmp/example_experiment'),*data)
        self.assertFalse(caught,[str(w.message) for w in caught])
        self.assertEqual(outputs.plt.get_fignums(),[])
        return figures[0]

    def test_image_endpoints_ecdfs_and_same_nucleus_pairing(self):
        data=self.fixture();fig=self.draw(data)
        self.assertEqual(len(fig.axes),16)  # Header, seven per channel, one paired scatter.
        # Unequal nucleus counts: endpoints contain image means [1,9], never pooled nuclei.
        np.testing.assert_array_equal(fig.axes[1].collections[0].get_offsets()[:,1],[1,9])
        np.testing.assert_array_equal(np.unique(fig.axes[1].patches[0].get_path().vertices[:,1]),[3,7])
        np.testing.assert_array_equal(fig.axes[2].collections[0].get_offsets()[:,1],[2,8])
        np.testing.assert_array_equal(fig.axes[3].collections[0].get_offsets()[:,1],[20,80])
        np.testing.assert_array_equal(fig.axes[4].collections[0].get_offsets()[:,1],[25,100])
        # Nucleus count ECDF includes all three zero-focus nuclei.
        np.testing.assert_array_equal(fig.axes[5].lines[0].get_xdata(),[0,0,0,4,9])
        np.testing.assert_allclose(fig.axes[5].lines[0].get_ydata(),np.arange(1,6)/5)
        np.testing.assert_array_equal(fig.axes[6].lines[0].get_xdata(),[1,2,2,3]+[8]*9)
        np.testing.assert_allclose(fig.axes[6].lines[0].get_ydata(),np.arange(1,14)/13)
        np.testing.assert_array_equal(fig.axes[7].lines[0].get_xdata(),[10,20,20,30]+[80]*9)
        for index in [6,7,13,14]:self.assertEqual(fig.axes[index].get_xscale(),'log')
        scatter=fig.axes[-1]
        np.testing.assert_array_equal(scatter.collections[0].get_offsets(),[[0,0],[0,1],[0,2],[4,3],[9,9]])
        np.testing.assert_array_equal(scatter.collections[1].get_offsets(),[[0,0]])
        self.assertEqual(len(scatter.lines),1)
        np.testing.assert_array_equal(scatter.lines[0].get_xdata(),[0,9])
        np.testing.assert_allclose(scatter.lines[0].get_ydata(),[3-2.6*54/63.2,3+(9-2.6)*54/63.2])
        legend=[text.get_text() for text in fig.axes[0].get_legend().get_texts()]
        self.assertIn('2 images · 5 nuclei',legend[0])
        self.assertIn('1 images · 1 nuclei',legend[1])
        self.assertTrue(any('No trend fit' in text.get_text() for text in scatter.texts))
        self.assertIn('C4 Focus B',scatter.get_xlabel());self.assertIn('C1 Focus A',scatter.get_ylabel())
        for collection in scatter.collections:
            self.assertTrue(collection.get_rasterized())
            np.testing.assert_array_equal(collection.get_sizes(),[8])
        repeated=self.draw(data)
        np.testing.assert_array_equal(fig.axes[1].collections[0].get_offsets(),
                                      repeated.axes[1].collections[0].get_offsets())

    def test_channel_subsets_and_nonfinite_values(self):
        for channels in [(1,),(4,1,2),(2,4)]:
            with self.subTest(channels=channels):
                data=self.fixture(channels);fig=self.draw(data)
                self.assertEqual(len(fig.axes),1+7*len(channels)+(len(channels)==2))
        mapping,images,cells,focus,p=self.fixture()
        cells[['C4_foci_count','C1_foci_count']]=cells[['C4_foci_count','C1_foci_count']].astype(float)
        focus['focus_area_um2']=focus.focus_area_um2.astype(float)
        cells.loc[0,'C4_foci_count']=np.nan
        cells.loc[1,'C1_foci_count']=np.inf
        focus.loc[0,'focus_area_um2']=np.inf
        images.loc[0,'C4_mean_focus_area_um2']=np.inf
        fig=self.draw((mapping,images,cells,focus,p))
        np.testing.assert_array_equal(fig.axes[2].collections[0].get_offsets()[:,1],[8])
        self.assertEqual(len(fig.axes[6].lines[0].get_xdata()),12)
        np.testing.assert_array_equal(fig.axes[-1].collections[0].get_offsets(),[[0,2],[4,3],[9,9]])

    def test_empty_results_and_all_zero_axes(self):
        mapping,images,cells,focus,p=self.fixture()
        for c in p['foci_channels']:
            for key in ['mean_foci_per_nucleus','mean_focus_area_um2','mean_focus_intensity_adu','percent_focus_positive_nuclei']:
                images[f'C{c}_{key}']=np.nan
        fig=self.draw((mapping,images,cells.iloc[:0],focus.iloc[:0],p))
        for ax in fig.axes[1:]:
            self.assertIn('No measurements',[text.get_text() for text in ax.texts])
            self.assertTrue(np.isfinite(ax.get_xlim()).all());self.assertTrue(np.isfinite(ax.get_ylim()).all())
        for c in p['foci_channels']:cells[f'C{c}_foci_count']=0
        focus['focus_area_um2']=0;focus['raw_mean_adu']=0
        fig=self.draw((mapping,images,cells,focus,p))
        self.assertEqual(fig.axes[6].get_xscale(),'linear')
        self.assertEqual(fig.axes[7].get_xscale(),'linear')
        self.assertEqual(sum(len(collection.get_offsets()) for collection in fig.axes[-1].collections),6)

    def test_known_trends_constant_y_and_legend_counts_ignore_missing_pairs(self):
        mapping,images,cells,focus,p=self.fixture()
        cells['C4_foci_count']=[0.,2.,4.,np.nan,np.inf,0.]
        cells['C1_foci_count']=[1.,5.,9.,7.,8.,0.]
        fig=self.draw((mapping,images,cells,focus,p))
        scatter=fig.axes[-1]
        np.testing.assert_array_equal(scatter.lines[0].get_xdata(),[0,4])
        np.testing.assert_allclose(scatter.lines[0].get_ydata(),[1,9])
        legend=fig.axes[0].get_legend().get_texts()[0].get_text()
        self.assertIn('2 images · 5 nuclei',legend)
        cells['C1_foci_count']=3.
        fig=self.draw((mapping,images,cells,focus,p))
        np.testing.assert_array_equal(fig.axes[-1].lines[0].get_ydata(),[3,3])

    def test_exports_long_labels_and_failure_cleanup(self):
        data=self.fixture()
        long_names=['siHELQ + 100MMC + extended combination of experimental treatment labels',
                    'siControl + 20PDS + additional extended experimental condition']
        for table in data[:4]:table['treatment_group']=table.treatment_group.replace(dict(zip(self.groups,long_names)))
        with patch.object(outputs,'GROUPS',long_names),patch.object(outputs,'LABELS',long_names),tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'example_experiment';out.mkdir()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                outputs.make_experiment_summary(out,*data)
            self.assertFalse(caught,[str(w.message) for w in caught])
            with Image.open(out/'Experiment_Summary.png') as image:
                self.assertGreaterEqual(image.width,5300)
                self.assertAlmostEqual(image.info['dpi'][0],300,delta=.1)
            svg=ET.parse(out/'Experiment_Summary.svg')
            self.assertTrue(list(svg.iter('{http://www.w3.org/2000/svg}image')))
            text=' '.join(''.join(node.itertext()) for node in svg.iter('{http://www.w3.org/2000/svg}text'))
            self.assertIn('example_experiment',text);self.assertIn('Channel 4',text);self.assertIn('Channel 1',text)
            (out/'Experiment_Summary.svg').write_text('old summary')
            outputs.make_experiment_summary(out,*data)
            ET.parse(out/'Experiment_Summary.svg')
        with patch.object(outputs,'figsave',side_effect=OSError('disk full')):
            with self.assertRaisesRegex(OSError,'disk full'):
                outputs.make_experiment_summary(Path('/private/tmp'),*self.fixture())
        self.assertEqual(outputs.plt.get_fignums(),[])


if __name__=='__main__':unittest.main()
