"""Large folders retain every field and use consistent contrast across pages."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import make_outputs as outputs
from matplotlib.figure import Figure


class SharedContrastTests(unittest.TestCase):
    def test_page_boundaries(self):
        for count, sizes in [(0, []), (1, [1]), (12, [12]), (13, [12, 1]),
                             (75, [12] * 6 + [3])]:
            with self.subTest(count=count):
                mapping = pd.DataFrame({'image_id': range(count)}, index=range(100, 100 + count))
                pages = list(outputs.shared_contrast_pages(mapping, 4))
                self.assertEqual([len(fields) for _, fields in pages], sizes)
                if pages:
                    self.assertEqual(pages[0][0], 'C4_Shared_Contrast.png')
                    self.assertEqual(pd.concat([fields for _, fields in pages]).image_id.tolist(), list(range(count)))

    def test_75_images_render_and_report_all_pages(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            qdir = out / 'QC'
            qdir.mkdir()
            records = []
            samples = {1: [], 2: []}
            for i in range(75):
                raw = np.stack([np.arange(256).reshape(16, 16) + 100 * i,
                                np.arange(256).reshape(16, 16) + 200 * i]).astype(np.uint16)
                path = out / f'field_{i}.tif'
                tifffile.imwrite(path, raw, metadata={'axes': 'CYX'}, photometric='minisblack')
                records.append({'image_id': f'I{i + 1:02d}', 'source_path': str(path),
                                'pixel_size_x_um': 1., 'treatment_group': 'Example', 'image_name': path.name})
                for c in samples:
                    samples[c].append(raw[c - 1, ::4, ::4].ravel())
            mapping = pd.DataFrame(records)
            params = {'foci_channels': [1, 2], 'nuclear_channel': 1,
                      'channels': {'1': 'A', '2': 'B'}}
            expected_limits = {c: np.percentile(np.concatenate(values), [1, 99.9])
                               for c, values in samples.items()}
            seen = {1: [], 2: []}
            savefig = Figure.savefig

            def checked_save(fig, filename, **kwargs):
                c = int(filename.name[1])
                width, height = fig.get_size_inches() * kwargs['dpi']
                self.assertLessEqual(width, 4000)
                self.assertLessEqual(height, 3081)
                panels = [ax for ax in fig.axes if ax.images]
                self.assertLessEqual(len(panels), 12)
                for ax in panels:
                    seen[c].append(ax.get_title().split(' · ')[1])
                    np.testing.assert_allclose(ax.images[0].get_clim(), expected_limits[c])
                return savefig(fig, filename, **kwargs)

            with patch.object(Figure, 'savefig', checked_save):
                outputs.make_shared_contrast(qdir, mapping, params)
            self.assertEqual(outputs.plt.get_fignums(), [])
            for c in samples:
                self.assertEqual(seen[c], mapping.image_id.tolist())
                for filename, _ in outputs.shared_contrast_pages(mapping, c):
                    with Image.open(qdir / filename) as image:
                        self.assertLessEqual(image.width, 4000)
                        self.assertLessEqual(image.height, 3081)

            images = mapping[['image_id', 'treatment_group']].copy()
            images['nuclei_analyzed'] = 0
            for c in samples:
                images[f'C{c}_total_foci'] = 0
                images[f'C{c}_mean_foci_per_nucleus'] = 0
            cells = pd.DataFrame(columns=['image_id', 'nucleus_id'])
            focus = pd.DataFrame(columns=['channel'])
            qc = pd.DataFrame(columns=['image_id', 'channel', 'number_usable_nuclei',
                                       'number_excluded_nuclei', 'manual_inspection_warnings'])
            outputs.report(out, mapping, images, cells, focus, qc, params)
            review = (out / 'Review.html').read_text()
            for path in qdir.glob('*.png'):
                self.assertIn(f'href="QC/{path.name}"', review)
                self.assertIn(f'src="QC/{path.name}"', review)
            self.assertEqual(len(list(qdir.glob('*.png'))), 14)

    def test_figure_closed_when_saving_fails(self):
        mapping = pd.DataFrame([{'source_path': 'unused.tif', 'image_id': 'I01',
                                 'treatment_group': 'Example', 'pixel_size_x_um': 1}])
        with patch.object(outputs.tifffile, 'imread', return_value=np.zeros((1, 16, 16))), \
                patch.object(Figure, 'savefig', side_effect=OSError('disk full')):
            with self.assertRaisesRegex(OSError, 'disk full'):
                outputs.make_shared_contrast(Path('.'), mapping, {'foci_channels': [1]})
        self.assertEqual(outputs.plt.get_fignums(), [])


if __name__ == '__main__':
    unittest.main()
