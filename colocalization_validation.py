"""Independent mask/pixel/table reconciliation for optional colocalization."""
import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi
from scipy.spatial.distance import cdist

from colocalization import settings, PROPERTIES


def validate_image(out, parameters, mapping, raw, nuclei, cells, focus, tables, check):
    """Recompute geometry and correlations without calling the analysis engine."""
    config = settings(parameters)
    a, b = config['channels']
    image_id = mapping.image_id
    prefix = f'{image_id} colocalization'
    pp = tables['Pairs'].loc[tables['Pairs'].image_id == image_id].copy()
    ff = tables['Foci'].loc[tables['Foci'].image_id == image_id].copy()
    nn = tables['Nuclei'].loc[tables['Nuclei'].image_id == image_id].copy()
    ii = tables['Images'].loc[tables['Images'].image_id == image_id]

    def close(actual, expected, title):
        check(np.allclose(np.asarray(actual, dtype=float), np.asarray(expected, dtype=float),
                          rtol=1e-8, atol=1e-8, equal_nan=True), prefix + ': ' + title)

    def average(series, method='mean'):
        return getattr(pd.Series(series, dtype=float), method)()

    check(len(ii) == 1, prefix + ': exactly one image summary')
    check(not nn.nucleus_id.duplicated().any() and set(nn.nucleus_id) == set(cells.nucleus_id),
          prefix + ': every accepted nucleus appears once, including zero-focus nuclei')
    selected = focus[focus.channel.isin([a, b])].copy()
    check(ff.marker_role.isin(['A', 'B']).all(), prefix + ': every focus has a valid A/B role')
    check(not ff.focus_id.duplicated().any() and set(ff.focus_id) == set(selected.focus_id),
          prefix + ': every selected focus appears once, including unassociated foci')
    check(not pp.duplicated(['focus_id_a', 'focus_id_b']).any(), prefix + ': pair IDs are unique')
    for table in (pp, ff, nn, ii):
        check(table.treatment_group.eq(mapping.treatment_group).all(), prefix + ': treatment identities preserved')
        check(table.image_name.eq(mapping.image_name).all(), prefix + ': original filenames preserved')
        if 'channel_a' in table:
            check(table.channel_a.eq(a).all() and table.channel_b.eq(b).all(), prefix + ': ordered channels preserved')
    ma, mb = [tifffile.imread(out / 'Masks' / f'{image_id}_C{c}_foci.tif') for c in (a, b)]
    bg_a, bg_b = [tifffile.imread(out / 'Masks' / f'{image_id}_C{c}_background_model.tif') for c in (a, b)]
    sx, sy = mapping.pixel_size_x_um, mapping.pixel_size_y_um
    # Focus_Data uses ten significant digits. Subtracting rounded coordinates
    # can distort short distances, so independently recover geometric centroids
    # from mask pixels instead of relaxing the distance reconciliation tolerance.
    for channel, mask in [(a, ma), (b, mb)]:
        rows = selected[selected.channel == channel]
        if rows.empty:
            continue
        yy, xx = np.nonzero(mask)
        labels = mask[yy, xx].astype(int)
        areas = np.bincount(labels)
        x_sums = np.bincount(labels, weights=xx)
        y_sums = np.bincount(labels, weights=yy)
        ids = rows.focus_label.to_numpy(int)
        cx, cy = x_sums[ids] / areas[ids], y_sums[ids] / areas[ids]
        close(rows.centroid_x_px, cx, 'original centroid X reconciles to mask pixels')
        close(rows.centroid_y_px, cy, 'original centroid Y reconciles to mask pixels')
        selected.loc[rows.index, 'centroid_x_px'] = cx
        selected.loc[rows.index, 'centroid_y_px'] = cy
        selected.loc[rows.index, 'centroid_x_um'] = cx * sx
        selected.loc[rows.index, 'centroid_y_um'] = cy * sy
    lookup_a = selected[selected.channel == a].set_index('focus_label')
    lookup_b = selected[selected.channel == b].set_index('focus_label')
    shared = (ma > 0) & (mb > 0)
    radix = int(mb.max()) + 1
    codes, counts = np.unique(ma[shared].astype(np.int64) * radix + mb[shared], return_counts=True)
    keep = counts >= config['minimum_overlap_pixels']
    la, lb, counts = codes[keep] // radix, codes[keep] % radix, counts[keep]
    expected = {(int(x), int(y)): int(n) for x, y, n in zip(la, lb, counts)}
    actual = {(int(r.focus_label_a), int(r.focus_label_b)): r for r in pp.itertuples()}
    check(set(expected) == set(actual) and len(pp) == len(expected), prefix + ': complete qualifying pixel-intersection pairs')
    expected_values, actual_values = [], []
    for (x, y), count in expected.items():
        ra, rb, pair = lookup_a.loc[x], lookup_b.loc[y], actual[x, y]
        check(pair.focus_id_a == ra.focus_id and pair.focus_id_b == rb.focus_id
              and pair.nucleus_id == ra.nucleus_id == rb.nucleus_id
              and pair.nucleus_label == ra.nucleus_label == rb.nucleus_label,
              prefix + ': pair IDs and parents agree with source focus rows')
        expected_values.extend([count, count * sx * sy, count / ra.focus_area_px,
                                count / rb.focus_area_px, count / (ra.focus_area_px + rb.focus_area_px - count),
                                np.hypot((ra.centroid_x_px - rb.centroid_x_px) * sx,
                                         (ra.centroid_y_px - rb.centroid_y_px) * sy)])
        actual_values.extend([pair.overlap_area_px, pair.overlap_area_um2, pair.overlap_fraction_a,
                              pair.overlap_fraction_b, pair.jaccard, pair.centroid_distance_um])
    close(actual_values, expected_values, 'pair areas, directional fractions, Jaccard and calibrated distances')

    for role, channel, other, original in [('A', a, b, lookup_a), ('B', b, a, lookup_b)]:
        records = ff[ff.marker_role == role].sort_values('focus_label')
        check(records.channel.eq(channel).all() and records.partner_channel.eq(other).all(),
              prefix + f': {role} focus roles and partner channels agree')
        if records.empty:
            continue
        rows = original.reindex(records.focus_label)
        check(records.focus_id.tolist() == rows.focus_id.tolist() and records.nucleus_id.tolist() == rows.nucleus_id.tolist()
              and records.nucleus_label.tolist() == rows.nucleus_label.tolist(),
              prefix + f': {role} focus identity and nucleus assignment reconcile')
        values = []
        for label in records.focus_label:
            hits = [n for (x, y), n in expected.items() if (x if role == 'A' else y) == label]
            values.append((len(hits), sum(hits)))
        degree = np.asarray([v[0] for v in values], dtype=int)
        covered = np.asarray([v[1] for v in values], dtype=int)
        close(records.partner_count, degree, f'{role} distinct partner counts')
        check(np.array_equal(records.is_colocalized.to_numpy(), degree > 0), prefix + f': {role} association flags')
        close(records.covered_area_px, covered, f'{role} union coverage pixels')
        close(records.covered_area_um2, covered * sx * sy, f'{role} physical covered area')
        close(records.overlap_fraction, covered / rows.focus_area_px.to_numpy(), f'{role} coverage fractions')
        close(records.focus_area_px, rows.focus_area_px, f'{role} original focus pixel areas')
        for prop in PROPERTIES:
            if prop in rows:
                close(records[prop], rows[prop], f'{role} original {prop}')
        close(records.centroid_x_um, rows.centroid_x_px * sx, f'{role} centroid X calibration')
        close(records.centroid_y_um, rows.centroid_y_px * sy, f'{role} centroid Y calibration')
        for label, group in records.groupby('nucleus_label'):
            opposite = selected[(selected.nucleus_label == label) & (selected.channel == other)]
            if opposite.empty:
                check(group.nearest_opposite_distance_um.isna().all() and group.nearest_opposite_focus_id.isna().all()
                      and group.nearest_opposite_status.eq('no_opposite_foci').all(), prefix + ': missing nearest partners explained')
            else:
                coords = group[['centroid_x_um', 'centroid_y_um']].to_numpy()
                distances = cdist(coords, opposite[['centroid_x_um', 'centroid_y_um']].to_numpy())
                close(group.nearest_opposite_distance_um, distances.min(axis=1), f'{role} nearest opposite centroids')
                check(group.nearest_opposite_status.eq('ok').all() and set(group.nearest_opposite_focus_id) <= set(opposite.focus_id),
                      prefix + ': nearest focus IDs belong to the same nucleus and opposite channel')
                nearest = opposite.set_index('focus_id').loc[group.nearest_opposite_focus_id]
                close(np.linalg.norm(coords - nearest[['centroid_x_um', 'centroid_y_um']].to_numpy(), axis=1),
                      distances.min(axis=1), 'nearest focus IDs actually achieve nearest distance')

    def summary(row, objects, pairs, title):
        wanted = {'pair_count': len(pairs),
                  'mean_centroid_distance_um': average(pairs.centroid_distance_um),
                  'median_centroid_distance_um': average(pairs.centroid_distance_um, 'median')}
        check(row.pair_distance_status == ('ok' if len(pairs) else 'no_qualifying_pairs'), prefix + ': ' + title + ' pair status')
        for role in ('A', 'B'):
            rr = objects[objects.marker_role == role]
            coloc = rr[rr.is_colocalized == True]
            noncoloc = rr[rr.is_colocalized == False]
            keys = {'total_foci': len(rr), 'colocalized_foci': len(coloc),
                    'colocalized_fraction': len(coloc) / len(rr) if len(rr) else np.nan,
                    'mean_partner_count_all': average(rr.partner_count),
                    'mean_partner_count_colocalized': average(coloc.partner_count),
                    'mean_coverage_fraction_colocalized': average(coloc.overlap_fraction),
                    'median_coverage_fraction_colocalized': average(coloc.overlap_fraction, 'median'),
                    'mean_nearest_opposite_distance_um': average(rr.nearest_opposite_distance_um),
                    'median_nearest_opposite_distance_um': average(rr.nearest_opposite_distance_um, 'median')}
            for category, subset in [('colocalized', coloc), ('noncolocalized', noncoloc)]:
                for prop in PROPERTIES:
                    for method in ('mean', 'median'):
                        keys[f'{category}_{method}_{prop}'] = average(subset[prop], method)
                check(row[f'{role}_{category}_status'] == ('ok' if len(subset) else f'no_{category}_foci'),
                      prefix + ': ' + title + f' {role} {category} missing-data status')
            check(row[f'{role}_colocalized_fraction_status'] == ('ok' if len(rr) else 'no_foci')
                  and row[f'{role}_all_foci_status'] == ('ok' if len(rr) else 'no_foci'),
                  prefix + ': ' + title + f' {role} all-focus status')
            wanted.update({f'{role}_{key}': value for key, value in keys.items()})
        close([row[key] for key in wanted], list(wanted.values()), title + ' counts, ratios and property summaries')

    totals = dict.fromkeys(['shared_mask_area_px', 'union_mask_area_px', 'signal_a_total', 'signal_a_in_b',
                           'signal_b_total', 'signal_b_in_a'], 0.)
    raw_coefficients, corrected_coefficients = [], []
    slices = ndi.find_objects(nuclei)
    for row in nn.to_dict('records'):
        r = pd.Series(row)
        label = int(r.nucleus_label)
        crop = slices[label - 1]
        nm = nuclei[crop] == label
        am, bm = ma[crop][nm] > 0, mb[crop][nm] > 0
        xa, xb = raw[a - 1][crop][nm].astype(float), raw[b - 1][crop][nm].astype(float)
        ca, cb = xa - bg_a[crop][nm], xb - bg_b[crop][nm]
        for basis, x, y, collector in [('pearson_raw', xa, xb, raw_coefficients),
                                        ('pearson_corrected', ca, cb, corrected_coefficients)]:
            if len(x) < 2:
                expected_r, status = np.nan, 'insufficient_pixels'
            elif np.ptp(x) == 0 or np.ptp(y) == 0:
                expected_r, status = np.nan, 'constant_channel'
            else:
                expected_r, status = np.corrcoef(x, y)[0, 1], 'ok'
                collector.append(expected_r)
            close(r[basis], expected_r, f'N{label} {basis} from source pixels')
            check(r[f'{basis}_status'] == status, prefix + ': Pearson undefined reason reconciles')
        pa, pb = np.clip(ca, 0, None), np.clip(cb, 0, None)
        values = {'shared_mask_area_px': int((am & bm).sum()), 'union_mask_area_px': int((am | bm).sum()),
                  'signal_a_total': pa.sum(), 'signal_a_in_b': pa[bm].sum(),
                  'signal_b_total': pb.sum(), 'signal_b_in_a': pb[am].sum()}
        close([r[key] for key in values], list(values.values()), f'N{label} mask and positive corrected signal sums')
        close(r.nuclear_pixels, nm.sum(), f'N{label} Pearson pixel count')
        for key in totals:
            totals[key] += values[key]
        validate_signal(r, values, close, check, prefix)
        summary(r, ff[ff.nucleus_label == label], pp[pp.nucleus_label == label], f'N{label}')
    ir = ii.iloc[0]
    summary(ir, ff, pp, 'image')
    close(ir.nuclei_analyzed, len(cells), 'image nucleus count')
    check(ir.analysis_status == ('ok' if len(cells) else 'no_usable_nuclei'), prefix + ': image analysis status')
    for basis, values in [('pearson_raw', raw_coefficients), ('pearson_corrected', corrected_coefficients)]:
        close([ir[f'{basis}_mean'], ir[f'{basis}_median'], ir[f'{basis}_n_assessed']],
              [average(values), average(values, 'median'), len(values)], f'{basis} image aggregate')
        check(ir[f'{basis}_status'] == ('ok' if values else 'no_valid_nuclei'), prefix + ': image Pearson status')
    close([ir[key] for key in totals], list(totals.values()), 'image pooled signal and mask sums')
    validate_signal(ir, totals, close, check, prefix)


def validate_signal(row, sums, close, check, prefix):
    union = sums['union_mask_area_px']
    close(row.jaccard, sums['shared_mask_area_px'] / union if union else np.nan, 'mask Jaccard')
    check(row.jaccard_status == ('ok' if union else 'empty_masks'), prefix + ': Jaccard status')
    for role, opposite in [('a', 'b'), ('b', 'a')]:
        total = sums[f'signal_{role}_total']
        close(row[f'manders_{role}'], sums[f'signal_{role}_in_{opposite}'] / total if total else np.nan,
              f'Manders {role} from corrected source pixels')
        check(row[f'manders_{role}_status'] == ('ok' if total else 'no_positive_corrected_signal'), prefix + ': Manders status')
