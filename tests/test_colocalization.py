"""Known-answer colocalization geometry and missing-data conventions."""
import copy
from io import StringIO
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import tifffile
from skimage.measure import regionprops

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from colocalization import calculate, settings, pearson
from colocalization_outputs import overlay_rgb
from colocalization_validation import validate_image


def measure_masks(a, b, *, minimum=1, sx=.2, sy=.3, nuclei=None, pixels=None, backgrounds=None, channels=(2, 3)):
    if nuclei is None:
        nuclei = np.ones_like(a)
    if pixels is None:
        signal = np.arange(a.size, dtype=float).reshape(a.shape)
        pixels = np.stack([np.ones_like(signal), signal, signal * 2 + 10])
    params = {'foci_channels': list(channels), 'colocalization': {
        'enabled': True, 'channels': list(channels), 'minimum_overlap_pixels': minimum}}
    masks = dict(zip(channels, (a, b)))
    if backgrounds is None:
        backgrounds = {c: np.zeros_like(a, dtype=float) for c in channels}
    rows = []
    for channel, mask in masks.items():
        for r in regionprops(mask):
            rows.append({'channel': channel, 'nucleus_label': int(nuclei[r.coords[0, 0], r.coords[0, 1]]),
                         'focus_label': r.label, 'focus_area_px': r.area, 'focus_area_um2': r.area * sx * sy,
                         'centroid_y_px': r.centroid[0], 'centroid_x_px': r.centroid[1],
                         'raw_mean_adu': 20 + r.label, 'local_corrected_mean_adu': 10 + r.label,
                         'local_corrected_integrated_adu_px': (10 + r.label) * r.area})
    return calculate(pixels, nuclei, masks, backgrounds, rows, params, sx, sy,
                     {'image_id': 'I01', 'image_name': 'fixture.tif', 'treatment_group': 'Control'})


class ColocalizationTests(unittest.TestCase):
    def test_three_to_one_counts_and_union_coverage(self):
        a = np.zeros((12, 14), int)
        a[2:4, 2:4], a[5:7, 2:4], a[8:10, 2:4] = 1, 2, 3
        a[2:4, 10:12] = 4  # Unassociated focus.
        b = np.zeros_like(a)
        b[1:11, 1:5] = 1
        result = measure_masks(a, b)
        image = result.images.iloc[0]
        self.assertEqual(image.pair_count, 3)
        self.assertEqual(image.A_colocalized_foci, 3)
        self.assertEqual(image.B_colocalized_foci, 1)
        self.assertEqual(image.A_colocalized_fraction, .75)
        self.assertEqual(image.B_colocalized_fraction, 1)
        aa = result.foci[result.foci.marker_role == 'A']
        bb = result.foci[result.foci.marker_role == 'B'].iloc[0]
        self.assertEqual(aa.partner_count.tolist(), [1, 1, 1, 0])
        self.assertEqual(aa.overlap_fraction.tolist(), [1, 1, 1, 0])
        self.assertEqual(bb.partner_count, 3)
        self.assertAlmostEqual(bb.overlap_fraction, 12 / 40)
        self.assertAlmostEqual(image.A_mean_partner_count_all, .75)
        self.assertAlmostEqual(image.B_mean_partner_count_all, 3)
        self.assertAlmostEqual(image.A_colocalized_mean_focus_area_um2, .24)
        self.assertAlmostEqual(image.A_noncolocalized_mean_focus_area_um2, .24)
        self.assertEqual(result.pairs.focus_id_b.nunique(), 1)
        swapped = measure_masks(b, a)
        self.assertEqual(swapped.images.iloc[0].A_colocalized_foci, 1)
        self.assertEqual(swapped.images.iloc[0].B_colocalized_foci, 3)

    def test_partial_overlap_threshold_and_anisotropic_distance(self):
        a = np.zeros((6, 6), int); b = np.zeros_like(a)
        a[1:3, 1:3] = 1; b[2:4, 2:4] = 1
        result = measure_masks(a, b, sx=2, sy=3)
        pair = result.pairs.iloc[0]
        self.assertEqual(pair.overlap_area_px, 1)
        self.assertEqual(pair.overlap_area_um2, 6)
        self.assertEqual(pair.overlap_fraction_a, .25)
        self.assertAlmostEqual(pair.jaccard, 1 / 7)
        self.assertAlmostEqual(pair.centroid_distance_um, np.sqrt(13))
        excluded = measure_masks(a, b, minimum=2)
        self.assertTrue(excluded.pairs.empty)
        self.assertFalse(excluded.foci.is_colocalized.any())
        self.assertTrue((excluded.foci.overlap_fraction == 0).all())
        self.assertTrue((excluded.foci.nearest_opposite_distance_um > 0).all())
        self.assertAlmostEqual(excluded.nuclei.iloc[0].jaccard, 1 / 7)
        adjacent = np.zeros_like(a); adjacent[1:3, 3:5] = 1
        self.assertTrue(measure_masks(a, adjacent).pairs.empty)

    def test_many_to_many_and_disconnected_intersections(self):
        a = np.zeros((8, 8), int); b = np.zeros_like(a)
        a[1:3, 1:7] = 1; a[5:7, 1:7] = 2
        b[1:7, 1:3] = 1; b[1:7, 5:7] = 2
        result = measure_masks(a, b)
        self.assertEqual(len(result.pairs), 4)
        self.assertTrue((result.foci.partner_count == 2).all())
        self.assertTrue(np.allclose(result.foci.overlap_fraction, 8 / 12))
        # Multiple disconnected pieces of the same label pair are one relationship.
        b[b == 2] = 1
        result = measure_masks(a, b)
        self.assertEqual(len(result.pairs), 2)
        self.assertTrue((result.pairs.overlap_area_px == 8).all())

    def test_empty_masks_nuclei_and_constant_pearson(self):
        blank = np.zeros((4, 4), int)
        result = measure_masks(blank, blank, pixels=np.ones((3, 4, 4)))
        row = result.nuclei.iloc[0]
        self.assertEqual(row.pearson_raw_status, 'constant_channel')
        self.assertTrue(np.isnan(row.pearson_raw))
        self.assertTrue(np.isnan(row.A_colocalized_fraction))
        self.assertEqual(row.A_colocalized_fraction_status, 'no_foci')
        self.assertEqual(row.pair_count, 0)
        self.assertEqual(row.jaccard_status, 'empty_masks')
        result = measure_masks(blank, blank, nuclei=blank)
        self.assertTrue(result.nuclei.empty)
        self.assertTrue(result.foci.empty)
        self.assertTrue(result.pairs.empty)
        self.assertEqual(result.images.iloc[0].analysis_status, 'no_usable_nuclei')
        a = blank.copy(); a[1:3, 1:3] = 1
        result = measure_masks(a, blank)
        self.assertEqual(result.foci.iloc[0].nearest_opposite_status, 'no_opposite_foci')
        self.assertEqual(result.images.iloc[0].A_colocalized_fraction, 0)
        self.assertTrue(np.isnan(result.images.iloc[0].B_colocalized_fraction))

    def test_signed_correction_manders_and_pearson(self):
        a = np.zeros((3, 3), int); a[:2, :2] = 1
        b = np.zeros_like(a); b[1:, 1:] = 1
        signal = np.arange(9).reshape(3, 3)
        pixels = np.stack([signal * 0, signal, signal * 2 + 1]).astype(float)
        bg = {2: np.full((3, 3), 5.), 3: np.arange(9).reshape(3, 3)[::-1].astype(float)}
        result = measure_masks(a, b, pixels=pixels, backgrounds=bg)
        row = result.nuclei.iloc[0]
        self.assertAlmostEqual(row.pearson_raw, 1)
        expected = np.corrcoef((pixels[1] - bg[2]).ravel(), (pixels[2] - bg[3]).ravel())[0, 1]
        self.assertAlmostEqual(row.pearson_corrected, expected)
        positive = np.maximum(pixels[1] - bg[2], 0)
        self.assertAlmostEqual(row.manders_a, positive[b > 0].sum() / positive.sum())
        self.assertNotAlmostEqual(row.pearson_corrected,
                                 np.corrcoef(positive.ravel(), np.maximum(pixels[2] - bg[3], 0).ravel())[0, 1])
        zero = measure_masks(a, b, pixels=pixels, backgrounds={2: pixels[1] + 1, 3: pixels[2] + 1})
        self.assertEqual(zero.nuclei.iloc[0].manders_a_status, 'no_positive_corrected_signal')
        self.assertTrue(np.isnan(zero.nuclei.iloc[0].manders_a))

    def test_channel_permutation_nuclear_boundaries_and_display(self):
        a = np.zeros((6, 8), int); b = np.zeros_like(a)
        a[1:3, 1:3] = 1; b[1:3, 1:3] = 1
        result = measure_masks(a, b)
        original = [x.copy() for x in (a, b)]
        rgb = overlay_rgb(np.ones_like(a), {2: a, 3: b}, result.foci, result.pairs)
        np.testing.assert_allclose(rgb[2, 2], [1, .88, .12], atol=1e-7)
        np.testing.assert_array_equal(a, original[0]); np.testing.assert_array_equal(b, original[1])
        permuted = measure_masks(a, b, channels=(3, 1))
        np.testing.assert_allclose(permuted.pairs.centroid_distance_um, result.pairs.centroid_distance_um)
        nuclei = np.ones_like(a); nuclei[:, 4:] = 2
        a[1:3, 5:7] = 2
        separated = measure_masks(a, b, nuclei=nuclei)
        self.assertEqual(len(separated.nuclei), 2)
        self.assertEqual(separated.foci.iloc[-1].nearest_opposite_status, 'no_opposite_foci')

    def test_settings_validation_and_defaults(self):
        self.assertFalse(settings({})['enabled'])
        self.assertFalse(settings({'foci_channels': [1]})['enabled'])
        valid = {'foci_channels': [1, 3], 'colocalization': {'enabled': True, 'channels': [3, 1]}}
        self.assertEqual(settings(valid)['channels'], [3, 1])
        for overrides in [{'channels': [1, 1]}, {'channels': [2, 3]}, {'channels': [None, 1]},
                          {'enabled': 1}, {'minimum_overlap_pixels': 0}, {'minimum_overlap_pixels': True},
                          {'minimum_overlap_pixels': 1.5}]:
            p = copy.deepcopy(valid); p['colocalization'].update(overrides)
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                settings(p)
        with self.assertRaises(ValueError):
            settings(valid, channel_count=2)
        self.assertEqual(pearson([1], [2])[1], 'insufficient_pixels')

    def test_validation_recomputes_short_distances_from_masks_not_rounded_csv(self):
        a = np.zeros((5, 2050), int); b = np.zeros_like(a)
        a[1:3, 2000] = 1; a[1, 2001] = 1
        b[1:3, 2001] = 1; b[1, 2000] = 1
        yy, xx = np.indices(a.shape)
        raw = np.stack([xx * 0, xx + yy, (xx + yy) * 2 + 10]).astype(float)
        result = measure_masks(a, b, pixels=raw)
        original = result.foci.copy()
        original['centroid_x_px'] = original.centroid_x_um / .2
        original['centroid_y_px'] = original.centroid_y_um / .3
        rounded = pd.read_csv(StringIO(original.to_csv(index=False, float_format='%.10g')))
        ra, rb = rounded.iloc[0], rounded.iloc[1]
        rounded_distance = np.hypot((ra.centroid_x_px - rb.centroid_x_px) * .2,
                                    (ra.centroid_y_px - rb.centroid_y_px) * .3)
        self.assertGreater(abs(rounded_distance - result.pairs.iloc[0].centroid_distance_um), 1e-8)
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder); (out / 'Masks').mkdir()
            for c, mask in [(2, a), (3, b)]:
                tifffile.imwrite(out / 'Masks' / f'I01_C{c}_foci.tif', mask.astype(np.uint32))
                tifffile.imwrite(out / 'Masks' / f'I01_C{c}_background_model.tif', np.zeros_like(a, dtype=np.float32))
            parameters = {'foci_channels': [2, 3], 'colocalization': {'enabled': True}}
            mapping = pd.Series({'image_id': 'I01', 'image_name': 'fixture.tif', 'treatment_group': 'Control',
                                 'pixel_size_x_um': .2, 'pixel_size_y_um': .3})
            def check(condition, description):
                self.assertTrue(condition, description)
            validate_image(out, parameters, mapping, raw, np.ones_like(a), result.nuclei,
                           rounded, result.tables(), check)


if __name__ == '__main__':
    unittest.main()
