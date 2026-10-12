"""Numerical and presentation checks for boxes built from saved image values."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from matplotlib.markers import MarkerStyle

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import make_outputs as outputs


class ImageBoxplotTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(outputs.plt.close,'all')
        self.groups=['siControl + 100MMC','single image','missing measurements','empty group']
        self.colors=['#0072B2','#D55E00','#009E73','#CC79A7']
        for name,value in [('GROUPS',self.groups),('LABELS',self.groups),('COLORS',self.colors)]:
            patcher=patch.object(outputs,name,value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def draw(self,data):
        fig,ax=outputs.plt.subplots(layout='constrained')
        outputs.stripbox(ax,data,'value','Image endpoint',np.random.default_rng(42),point_alpha=.25)
        fig.canvas.draw()
        return ax

    def test_known_statistics_single_image_and_invalid_values(self):
        data=pd.DataFrame({'treatment_group':[self.groups[0]]*8+[self.groups[1]]+[self.groups[2]]*3,
                           'value':[1,2,3,4,20,None,np.inf,-np.inf,7,None,np.nan,np.inf]})
        ax=self.draw(data)
        self.assertEqual(len(ax.patches),2)
        # [1,2,3,4,20]: Q1=2, median=3, Q3=4; 20 is outside the upper whisker.
        np.testing.assert_array_equal(np.unique(ax.patches[0].get_path().vertices[:,1]),[2,4])
        np.testing.assert_array_equal(ax.lines[0].get_ydata(),[2,1])
        np.testing.assert_array_equal(ax.lines[1].get_ydata(),[4,4])
        np.testing.assert_array_equal(ax.lines[4].get_ydata(),[3,3])
        self.assertEqual(len(ax.lines),10)  # Two whiskers, two caps, one median per box; no fliers.
        np.testing.assert_array_equal(np.unique(ax.patches[1].get_path().vertices[:,1]),[7])
        self.assertEqual(len(ax.collections),2)
        circle=MarkerStyle('o')
        circle_path=circle.get_path().transformed(circle.get_transform())
        for j,(collection,expected) in enumerate(zip(ax.collections,[[1,2,3,4,20],[7]])):
            offsets=collection.get_offsets()
            np.testing.assert_array_equal(offsets[:,1],expected)
            self.assertTrue(np.all(np.abs(offsets[:,0]-j)<=.16))
            np.testing.assert_array_equal(collection.get_sizes(),[8])
            self.assertEqual(collection.get_alpha(),.25)
            np.testing.assert_array_equal(collection.get_paths()[0].vertices,circle_path.vertices)
            np.testing.assert_allclose(collection.get_facecolors()[0],
                                       outputs.matplotlib.colors.to_rgba(self.colors[j],.25))
            np.testing.assert_allclose(ax.patches[j].get_edgecolor(),
                                       outputs.matplotlib.colors.to_rgba(self.colors[j]))
        self.assertEqual(len(ax.texts),0)
        self.assertEqual([tick.get_text() for tick in ax.get_xticklabels()],self.groups)
        for tick in ax.get_xticklabels():
            self.assertEqual(tick.get_rotation(),45)
            self.assertEqual(tick.get_ha(),'right')
        repeated=self.draw(data)
        for first,second in zip(ax.collections,repeated.collections):
            np.testing.assert_array_equal(first.get_offsets(),second.get_offsets())

    def test_empty_and_all_nonfinite_metrics_keep_categories(self):
        for data in [pd.DataFrame({'treatment_group':[],'value':[]}),
                     pd.DataFrame({'treatment_group':self.groups,'value':[None,np.nan,np.inf,-np.inf]})]:
            with self.subTest(data=data):
                ax=self.draw(data)
                self.assertEqual(len(ax.collections),0)
                self.assertEqual(len(ax.patches),0)
                self.assertEqual(len(ax.texts),0)
                self.assertEqual([tick.get_text() for tick in ax.get_xticklabels()],self.groups)
                self.assertTrue(np.isfinite(ax.get_ylim()).all())


if __name__=='__main__':unittest.main()
