"""Representative scoring, review fences, display names, and adaptive panels."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import warnings

import numpy as np
import pandas as pd
import tifffile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from image_review import ENDPOINTS, image_display_name, representative_grid, review_images
import make_outputs as outputs


class ImageReviewTests(unittest.TestCase):
    def fixture(self,values,group='Treatment',channel=2):
        mapping=pd.DataFrame([dict(image_id=f'I{i+1:02d}',image_name=f'field_{i+1}.tif',treatment_group=group)
                              for i in range(len(values))])
        images=mapping.copy();images['nuclei_analyzed']=10
        for key,multiplier in zip(ENDPOINTS,[1,10,100,2]):images[f'C{channel}_{key}']=np.array(values,dtype=float)*multiplier
        return mapping,images,{'foci_channels':[channel]}

    def test_means_include_outliers_and_ranking_is_scale_independent(self):
        mapping,images,p=self.fixture([1,2,3,4,50])
        original=images.copy(deep=True)
        review,selection=review_images(mapping,images,p)
        self.assertEqual(set(review.loc[review.outlier,'image_id']),{'I05'})
        flags=review[(review.metric=='C2_mean_foci_per_nucleus')&review.outlier].iloc[0]
        self.assertEqual((flags.q1,flags.q3,flags.lower_fence,flags.upper_fence),(2,4,-1,7))
        selected=selection[selection.selected].sort_values('rank')
        self.assertEqual(selected.image_id.tolist(),['I04','I03'])
        self.assertFalse(selection.loc[selection.image_id=='I05','eligible'].iloc[0])
        expected=(12-4)/np.std([1,2,3,4,50],ddof=1)
        self.assertAlmostEqual(selected.score.iloc[0],expected)
        self.assertIn('"C2_mean_foci_per_nucleus": 12.0',selected.group_means.iloc[0])
        pd.testing.assert_frame_equal(images,original)
        shuffled=images.sample(frac=1,random_state=9)
        _,again=review_images(mapping,shuffled,p)
        self.assertEqual(again[again.selected].sort_values('rank').image_id.tolist(),['I04','I03'])

    def test_fences_boundaries_small_groups_zero_iqr_and_missing(self):
        # Q1=1,Q3=3,IQR=2; the boundary at 6 must not be flagged.
        mapping,images,p=self.fixture([0,1,2,3,6])
        review,_=review_images(mapping,images,p)
        self.assertFalse(review.outlier.any())
        for values,status in [([1,2,3,100],'insufficient images'),([1]*5+[2],'zero IQR')]:
            mapping,images,p=self.fixture(values)
            review,_=review_images(mapping,images,p)
            self.assertEqual(set(review.assessment_status),{status})
            self.assertFalse(review.outlier.any())
        mapping,images,p=self.fixture([1,2,3,4,50,np.nan,np.inf])
        review,_=review_images(mapping,images,p)
        self.assertEqual(set(review[review.image_id.isin(['I06','I07'])].assessment_status),{'missing value'})
        self.assertEqual(set(review.finite_images),{5})

    def test_multiple_channels_groups_coverage_and_constant_ties(self):
        mapping,images,p=self.fixture([0,5,10])
        images['C2_mean_focus_area_um2']=[1,np.nan,3]
        _,selection=review_images(mapping,images,p)
        self.assertEqual(set(selection.loc[selection.selected,'image_id']),{'I01','I03'})
        self.assertTrue(selection.loc[selection.image_id=='I02','partial_metrics'].iloc[0])
        mapping,images,p=self.fixture([7,7,7])
        mapping['image_name']=['z.tif','a.tif','b.tif'];images['image_name']=mapping.image_name
        _,selection=review_images(mapping,images,p)
        self.assertEqual(selection[selection.selected].sort_values('rank').image_id.tolist(),['I02','I03'])
        self.assertTrue((selection.score==0).all())
        # Repeated floating values can yield a tiny numerical std despite no variation.
        float_mapping,float_images,float_p=self.fixture([.1]*6)
        _,constant=review_images(float_mapping,float_images,float_p)
        self.assertTrue((constant.metrics_available==0).all())
        images['nuclei_analyzed']=[0,10,0]
        _,selection=review_images(mapping,images,p)
        self.assertEqual(selection[selection.selected].image_id.tolist(),['I02'])
        first,first_images,p=self.fixture([1,2,3,4,50])
        second,second_images,_=self.fixture([100,110,120,130,140],group='Other')
        second['image_id']='O'+second.image_id;second_images['image_id']=second.image_id
        mapping=pd.concat([first,second],ignore_index=True);images=pd.concat([first_images,second_images],ignore_index=True)
        for key in ENDPOINTS:images[f'C4_{key}']=images[f'C2_{key}']
        # Flagged on channel 2, but not 4: eligibility still rejects the image.
        for key in ENDPOINTS:images.loc[images.image_id=='I05',f'C4_{key}']=2
        review,selection=review_images(mapping,images,{'foci_channels':[2,4]})
        self.assertEqual(set(review.loc[review.outlier,'image_id']),{'I05'})
        self.assertFalse(selection.loc[selection.image_id=='I05','selected'].iloc[0])
        self.assertEqual(set(selection.metrics_available),{8})
        _,subset=review_images(mapping,images,{'foci_channels':[4]})
        self.assertTrue(subset.loc[subset.image_id=='I05','eligible'].iloc[0])

    def test_grid_and_legacy_display_names(self):
        for count,expected in [(1,(1,1)),(2,(1,2)),(3,(2,2)),(7,(3,3)),(9,(3,3)),(10,(2,5)),(26,(5,6))]:
            self.assertEqual(representative_grid(count),expected)
        self.assertEqual(image_display_name({'image_name':'true <name>.tif','image_id':'I01'}),'true <name>.tif')
        self.assertEqual(image_display_name({'image_name':np.nan,'source_path':'/raw/legacy.TIF','image_id':'I01'}),'legacy.TIF')

    def test_render_shared_ranges_full_fields_and_missing_slots(self):
        self.addCleanup(outputs.plt.close,'all')
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);(out/'QC').mkdir()
            mapping,images,p=self.fixture([1,2,3])
            p.update(nuclear_channel=1,channels={'1':'Nuclear','2':'Foci'})
            for i,row in mapping.iterrows():
                raw=np.stack([np.arange(48*64).reshape(48,64)+100*i]*2).astype(np.uint16)
                path=out/row.image_name
                tifffile.imwrite(path,raw,photometric='minisblack',metadata={'axes':'CYX'})
                mapping.loc[i,'source_path']=str(path)
            mapping['pixel_size_x_um']=.1;mapping['pixel_size_y_um']=.2
            _,selection=review_images(mapping,images,p)
            ranges=outputs.shared_display_ranges(mapping,[1,2],p)
            seen=[]
            def inspect(fig,folder,name):
                self.assertEqual(name,'Representative_Images');fig.canvas.draw()
                panels=[ax for ax in fig.axes if ax.images]
                self.assertEqual(len(panels),4)
                for ax in panels:
                    np.testing.assert_allclose(ax.images[0].get_extent(),[0,6.4,9.6,0])
                    np.testing.assert_allclose(ax.images[0].get_clim(),ranges[1])
                    self.assertEqual(ax.get_aspect(),1)
                    # Adaptive bar is a quarter of the calibrated field width.
                    np.testing.assert_allclose(np.diff(ax.lines[0].get_xdata()),[1.6])
                titles=' '.join(ax.get_title() for ax in fig.axes)
                self.assertIn('field_2.tif',titles);self.assertNotIn('I02',titles)
                seen.append(fig)
            with patch.object(outputs,'figsave',side_effect=inspect),warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                outputs.make_representative_images(out,mapping,selection,p,ranges)
            self.assertFalse(caught,[str(w.message) for w in caught])
            self.assertTrue((out/'QC/Representative_Display_Ranges.csv').exists())
            p['nuclear_channel']=2
            with patch.object(outputs,'figsave') as save:
                outputs.make_representative_images(out,mapping,selection,p,ranges)
                self.assertEqual(len([ax for ax in save.call_args.args[0].axes if ax.images]),2)
            selection['selected']=False
            with patch.object(outputs,'figsave') as save:
                outputs.make_representative_images(out,mapping,selection,p,ranges)
                fig=save.call_args.args[0]
                self.assertIn('Fewer than two eligible images',' '.join(text.get_text() for ax in fig.axes for text in ax.texts))
            self.assertEqual(outputs.plt.get_fignums(),[])

    def test_downsampling_preserves_odd_sized_field_calibration(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);(out/'QC').mkdir()
            mapping,images,p=self.fixture([1])
            path=out/'large.tif'
            tifffile.imwrite(path,np.ones((2,1101,1401),dtype=np.uint16),photometric='minisblack',metadata={'axes':'CYX'})
            mapping['source_path']=str(path);mapping['pixel_size_x_um']=.1;mapping['pixel_size_y_um']=.2
            p.update(nuclear_channel=1)
            _,selection=review_images(mapping,images,p)
            with patch.object(outputs,'figsave') as save:
                outputs.make_representative_images(out,mapping,selection,p,{1:[0,2],2:[0,2]})
                axes=[ax for ax in save.call_args.args[0].axes if ax.images]
                for ax in axes:
                    self.assertLessEqual(max(ax.images[0].get_array().shape),1024)
                    np.testing.assert_allclose(ax.images[0].get_extent(),[0,140.1,220.2,0])

    def test_html_filenames_flags_and_unchanged_links(self):
        mapping,images,p=self.fixture([1,2,3,4,50]);p['nuclear_channel']=1
        mapping.loc[4,'image_name']='true <outlier> & sample.tif'
        cells=pd.DataFrame(columns=['image_id','nucleus_id']);focus=pd.DataFrame(columns=['channel'])
        qc=pd.DataFrame([dict(image_id='I05',channel=2,number_usable_nuclei=10,number_excluded_nuclei=0,
                             manual_inspection_warnings='Existing QC warning')])
        images['C2_total_foci']=1
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            outputs.report(out,mapping,images,cells,focus,qc,p)
            text=(out/'Review.html').read_text()
            self.assertIn('1 image(s) flagged',text)
            self.assertIn('true &lt;outlier&gt; &amp; sample.tif',text)
            self.assertIn('Existing QC warning',text)
            self.assertIn('href="QC/I05_Annotated_QC.png"',text)
            self.assertIn('1 flagged for review',text)
            self.assertNotIn('<h3>I05',text)
            self.assertIn('Representative_Images.png',text)


if __name__=='__main__':unittest.main()
