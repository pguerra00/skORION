"""Partner-count lines preserve discrete probability mass and both directions."""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import make_outputs as outputs
from colocalization_outputs import partner_count_lines


class PartnerProbabilityTests(unittest.TestCase):
    def test_known_probabilities_zero_counts_above_three_and_shared_axes(self):
        self.addCleanup(outputs.plt.close,'all')
        records=[]
        for role,group,counts in [('A','First',[0,0,1,4]),('A','Second',[1]*8),
                                   ('B','First',[0,2,6]),('B','Second',[0])]:
            records.extend(dict(marker_role=role,treatment_group=group,partner_count=v) for v in counts)
        objects=pd.DataFrame(records)
        fig,axes=outputs.plt.subplots(1,2)
        for ax,role in zip(axes,['A','B']):
            partner_count_lines(ax,objects,role,['First','Second','Empty'],['red','blue','green'],6)
            self.assertEqual(len(ax.lines),2)
            self.assertFalse(ax.patches)
            self.assertEqual(ax.get_ylabel(),'Fraction of foci')
            for line in ax.lines:
                np.testing.assert_array_equal(line.get_xdata(),np.arange(7))
                self.assertAlmostEqual(sum(line.get_ydata()),1)
        np.testing.assert_allclose(axes[0].lines[0].get_ydata(),[.5,.25,0,0,.25,0,0])
        np.testing.assert_allclose(axes[0].lines[1].get_ydata(),[0,1,0,0,0,0,0])
        np.testing.assert_allclose(axes[1].lines[0].get_ydata(),[1/3,0,1/3,0,0,0,1/3])
        self.assertEqual(axes[0].get_xlim(),axes[1].get_xlim())
        self.assertEqual(axes[1].lines[1].get_marker(),'o')

    def test_empty_and_single_count_distributions(self):
        self.addCleanup(outputs.plt.close,'all')
        objects=pd.DataFrame([dict(marker_role='A',treatment_group='Only',partner_count=0)])
        fig,axes=outputs.plt.subplots(1,2)
        partner_count_lines(axes[0],objects,'A',['Only'],['red'],0)
        np.testing.assert_array_equal(axes[0].lines[0].get_ydata(),[1])
        self.assertEqual(axes[0].lines[0].get_marker(),'o')
        partner_count_lines(axes[1],objects,'B',['Only'],['red'],0)
        self.assertFalse(axes[1].lines)
        self.assertIn('No defined observations',[text.get_text() for text in axes[1].texts])


if __name__=='__main__':unittest.main()
