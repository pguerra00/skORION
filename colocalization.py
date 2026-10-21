"""Calibrated, many-to-many colocalization of accepted 2D nuclear foci.

This module never segments images or changes masks. Pearson uses all accepted
nuclear pixels. Object associations use only qualifying shared label pixels.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.spatial import cKDTree


DEFAULTS = {'enabled': False, 'channels': [2, 3], 'minimum_overlap_pixels': 1}
VERSION = '1.0'
IDENTITY = ['image_id', 'image_name', 'treatment_group', 'group_names', 'replicate_id']
NUCLEUS = ['nucleus_id', 'nucleus_label']
PROPERTIES = ['focus_area_um2', 'raw_mean_adu', 'local_corrected_mean_adu',
              'local_corrected_integrated_adu_px']
PAIR_COLUMNS = IDENTITY + NUCLEUS + [
    'channel_a', 'channel_b', 'focus_id_a', 'focus_label_a', 'focus_id_b', 'focus_label_b',
    'overlap_area_px', 'overlap_area_um2', 'overlap_fraction_a', 'overlap_fraction_b',
    'jaccard', 'centroid_distance_um']
FOCUS_COLUMNS = IDENTITY + NUCLEUS + [
    'marker_role', 'channel', 'partner_channel', 'focus_id', 'focus_label', 'focus_area_px',
    *PROPERTIES, 'centroid_x_um', 'centroid_y_um', 'is_colocalized', 'partner_count',
    'covered_area_px', 'covered_area_um2', 'overlap_fraction',
    'nearest_opposite_focus_id', 'nearest_opposite_distance_um', 'nearest_opposite_status']
SUMMARY_COLUMNS = ['pair_count', 'mean_centroid_distance_um', 'median_centroid_distance_um',
                   'pair_distance_status']
for _role in ('A', 'B'):
    SUMMARY_COLUMNS += [f'{_role}_{key}' for key in (
        'total_foci', 'colocalized_foci', 'colocalized_fraction', 'colocalized_fraction_status',
        'mean_partner_count_all', 'mean_partner_count_colocalized',
        'mean_coverage_fraction_colocalized', 'median_coverage_fraction_colocalized',
        'mean_nearest_opposite_distance_um', 'median_nearest_opposite_distance_um',
        'all_foci_status', 'colocalized_status', 'noncolocalized_status')]
    SUMMARY_COLUMNS += [f'{_role}_{category}_{stat}_{prop}'
                        for category in ('colocalized', 'noncolocalized')
                        for prop in PROPERTIES for stat in ('mean', 'median')]
SIGNAL_COLUMNS = ['shared_mask_area_px', 'union_mask_area_px', 'jaccard', 'jaccard_status',
                  'signal_a_total', 'signal_a_in_b', 'signal_b_total', 'signal_b_in_a',
                  'manders_a', 'manders_a_status', 'manders_b', 'manders_b_status']
NUCLEUS_COLUMNS = IDENTITY + NUCLEUS + ['channel_a', 'channel_b', 'nuclear_pixels',
    'pearson_raw', 'pearson_raw_status', 'pearson_corrected', 'pearson_corrected_status',
    *SIGNAL_COLUMNS, *SUMMARY_COLUMNS]
IMAGE_COLUMNS = IDENTITY + ['channel_a', 'channel_b', 'nuclei_analyzed', 'analysis_status',
    *[f'{basis}_{suffix}' for basis in ('pearson_raw', 'pearson_corrected')
      for suffix in ('mean', 'median', 'n_assessed', 'status')], *SIGNAL_COLUMNS, *SUMMARY_COLUMNS]
SCHEMAS = {'Pairs': PAIR_COLUMNS, 'Foci': FOCUS_COLUMNS,
           'Nuclei': NUCLEUS_COLUMNS, 'Images': IMAGE_COLUMNS}


def settings(parameters, foci_channels=None, channel_count=None):
    """Resolve defaults; inactive channel choices need not be available."""
    supplied = parameters.get('colocalization', {})
    if not isinstance(supplied, dict):
        raise ValueError('colocalization must be a JSON object')
    config = {**DEFAULTS, **supplied}
    config['channels'] = list(config['channels']) if isinstance(config['channels'], list) else config['channels']
    if type(config['enabled']) is not bool:
        raise ValueError('colocalization.enabled must be true or false')
    if not isinstance(config['channels'], list) or len(config['channels']) != 2:
        raise ValueError('colocalization.channels must contain Marker A and Marker B selections')
    minimum = config['minimum_overlap_pixels']
    if type(minimum) is not int or minimum < 1:
        raise ValueError('colocalization.minimum_overlap_pixels must be a positive integer')
    if config['enabled']:
        channels = config['channels']
        if (not isinstance(channels, list) or len(channels) != 2
                or any(type(c) is not int or c < 1 for c in channels) or channels[0] == channels[1]):
            raise ValueError('Colocalization requires two distinct, one-based Marker A/B channels')
        selected = foci_channels if foci_channels is not None else parameters.get('foci_channels', [2, 3])
        if not set(channels) <= set(selected):
            raise ValueError('Both colocalization channels must be selected for foci analysis')
        if channel_count is not None and max(channels) > channel_count:
            raise ValueError('A selected colocalization channel is unavailable in this image')
    return config


def pearson(x, y):
    """Coefficient only, with explicit undefined states; no significance test."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.size < 2:
        return np.nan, 'insufficient_pixels'
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        return np.nan, 'nonfinite_pixels'
    x, y = x - x.mean(), y - y.mean()
    nx, ny = np.linalg.norm(x), np.linalg.norm(y)
    if nx == 0 or ny == 0:
        return np.nan, 'constant_channel'
    return float(np.clip(np.dot(x / nx, y / ny), -1, 1)), 'ok'


def _stat(rows, key, stat='mean'):
    values = np.asarray([r.get(key, np.nan) for r in rows], dtype=float)
    values = values[np.isfinite(values)]
    return float(getattr(np, stat)(values)) if len(values) else np.nan


def summarize(foci, pairs):
    result = {'pair_count': len(pairs),
              'mean_centroid_distance_um': _stat(pairs, 'centroid_distance_um'),
              'median_centroid_distance_um': _stat(pairs, 'centroid_distance_um', 'median'),
              'pair_distance_status': 'ok' if pairs else 'no_qualifying_pairs'}
    for role in ('A', 'B'):
        all_rows = [r for r in foci if r['marker_role'] == role]
        coloc = [r for r in all_rows if r['is_colocalized']]
        other = [r for r in all_rows if not r['is_colocalized']]
        values = {'total_foci': len(all_rows), 'colocalized_foci': len(coloc),
                  'colocalized_fraction': len(coloc) / len(all_rows) if all_rows else np.nan,
                  'colocalized_fraction_status': 'ok' if all_rows else 'no_foci',
                  'mean_partner_count_all': _stat(all_rows, 'partner_count'),
                  'mean_partner_count_colocalized': _stat(coloc, 'partner_count'),
                  'mean_coverage_fraction_colocalized': _stat(coloc, 'overlap_fraction'),
                  'median_coverage_fraction_colocalized': _stat(coloc, 'overlap_fraction', 'median'),
                  'mean_nearest_opposite_distance_um': _stat(all_rows, 'nearest_opposite_distance_um'),
                  'median_nearest_opposite_distance_um': _stat(all_rows, 'nearest_opposite_distance_um', 'median'),
                  'all_foci_status': 'ok' if all_rows else 'no_foci',
                  'colocalized_status': 'ok' if coloc else 'no_colocalized_foci',
                  'noncolocalized_status': 'ok' if other else 'no_noncolocalized_foci'}
        for category, rows in [('colocalized', coloc), ('noncolocalized', other)]:
            for prop in PROPERTIES:
                for stat in ('mean', 'median'):
                    values[f'{category}_{stat}_{prop}'] = _stat(rows, prop, stat)
        result.update({f'{role}_{key}': value for key, value in values.items()})
    return result


def _signal_summary(values):
    result = dict(values)
    union = values['union_mask_area_px']
    result.update(jaccard=values['shared_mask_area_px'] / union if union else np.nan,
                  jaccard_status='ok' if union else 'empty_masks')
    for role, opposite in [('a', 'b'), ('b', 'a')]:
        total = values[f'signal_{role}_total']
        result[f'manders_{role}'] = values[f'signal_{role}_in_{opposite}'] / total if total else np.nan
        result[f'manders_{role}_status'] = 'ok' if total else 'no_positive_corrected_signal'
    return result


@dataclass
class ColocalizationResult:
    pairs: pd.DataFrame
    foci: pd.DataFrame
    nuclei: pd.DataFrame
    images: pd.DataFrame

    def tables(self):
        return {name: getattr(self, name.lower()) for name in SCHEMAS}


def calculate(pixels, nuclei, masks, backgrounds, focus_rows, parameters, sx, sy, identity=None):
    """Measure one image using accepted masks, source pixels, and saved backgrounds.

    Quantize background models to their saved float32 representation so previews,
    full analyses and independent reconciliation use exactly the same correction.
    """
    config = settings(parameters, channel_count=len(pixels))
    if not config['enabled']:
        return ColocalizationResult(*(pd.DataFrame(columns=s) for s in SCHEMAS.values()))
    a, b = config['channels']
    identity = {key: (identity or {}).get(key) for key in IDENTITY}
    image_id = identity['image_id'] or 'preview'
    identity['image_id'] = image_id
    pair_rows, object_rows, nucleus_rows = [], [], []
    source_rows = {}
    for row in focus_rows:
        if row['channel'] in (a, b):
            source_rows.setdefault((row['nucleus_label'], row['channel']), []).append(row)
    for c in (a, b):
        if masks[c].shape != nuclei.shape or backgrounds[c].shape != nuclei.shape:
            raise ValueError('Colocalization masks and backgrounds must match the nuclear image')
        if np.any((masks[c] > 0) & (nuclei == 0)):
            raise ValueError('Colocalization focus pixels must belong to accepted nuclei')
    for label, crop in enumerate(ndi.find_objects(nuclei), 1):
        if crop is None:
            continue
        nuclear = nuclei[crop] == label
        mask_a = np.where(nuclear, masks[a][crop], 0)
        mask_b = np.where(nuclear, masks[b][crop], 0)
        rows_a = sorted(source_rows.get((label, a), []), key=lambda r: r['focus_label'])
        rows_b = sorted(source_rows.get((label, b), []), key=lambda r: r['focus_label'])
        for mask, rows in [(mask_a, rows_a), (mask_b, rows_b)]:
            if set(np.unique(mask)) - {0} != {r['focus_label'] for r in rows}:
                raise ValueError('Colocalization focus rows disagree with their nuclear mask labels')
        nid = f'{image_id}_N{label:03d}'
        base = {**identity, 'nucleus_id': nid, 'nucleus_label': label}
        indexes = [{r['focus_label']: r for r in rows} for rows in (rows_a, rows_b)]
        shared = (mask_a > 0) & (mask_b > 0)
        labels, counts = np.unique(np.column_stack((mask_a[shared], mask_b[shared])),
                                   axis=0, return_counts=True)
        partners = {a: {}, b: {}}
        these_pairs = []
        for (la, lb), count in zip(labels, counts):
            if count < config['minimum_overlap_pixels']:
                continue
            ra, rb = indexes[0][la], indexes[1][lb]
            area_a, area_b = ra['focus_area_px'], rb['focus_area_px']
            pair = {**base, 'channel_a': a, 'channel_b': b,
                    'focus_id_a': ra.get('focus_id', f'{image_id}_C{a}_F{la:05d}'),
                    'focus_id_b': rb.get('focus_id', f'{image_id}_C{b}_F{lb:05d}'),
                    'focus_label_a': int(la), 'focus_label_b': int(lb),
                    'overlap_area_px': int(count), 'overlap_area_um2': float(count * sx * sy),
                    'overlap_fraction_a': count / area_a, 'overlap_fraction_b': count / area_b,
                    'jaccard': count / (area_a + area_b - count),
                    'centroid_distance_um': float(np.hypot(
                        (ra['centroid_x_px'] - rb['centroid_x_px']) * sx,
                        (ra['centroid_y_px'] - rb['centroid_y_px']) * sy))}
            these_pairs.append(pair)
            partners[a].setdefault(la, []).append(int(count))
            partners[b].setdefault(lb, []).append(int(count))
        these_objects = []
        for role, c, other, rows, opposite in [('A', a, b, rows_a, rows_b), ('B', b, a, rows_b, rows_a)]:
            if rows and opposite:
                coords = lambda rr: np.asarray([[r['centroid_x_px'] * sx, r['centroid_y_px'] * sy] for r in rr])
                distances, nearest = cKDTree(coords(opposite)).query(coords(rows))
            else:
                distances, nearest = np.full(len(rows), np.nan), [None] * len(rows)
            for row, distance, nearest_index in zip(rows, distances, nearest):
                focus_label = row['focus_label']
                overlaps = partners[c].get(focus_label, [])
                covered = sum(overlaps)  # Same-channel label masks are disjoint.
                nearest_row = opposite[int(nearest_index)] if nearest_index is not None else None
                record = {**base, 'marker_role': role, 'channel': c, 'partner_channel': other,
                          'focus_id': row.get('focus_id', f'{image_id}_C{c}_F{focus_label:05d}'),
                          'focus_label': focus_label, 'focus_area_px': row['focus_area_px'],
                          **{prop: row.get(prop, np.nan) for prop in PROPERTIES},
                          'centroid_x_um': row['centroid_x_px'] * sx,
                          'centroid_y_um': row['centroid_y_px'] * sy,
                          'is_colocalized': bool(overlaps), 'partner_count': len(overlaps),
                          'covered_area_px': covered, 'covered_area_um2': covered * sx * sy,
                          'overlap_fraction': covered / row['focus_area_px'],
                          'nearest_opposite_focus_id': (nearest_row.get('focus_id',
                              f'{image_id}_C{other}_F{nearest_row["focus_label"]:05d}') if nearest_row else None),
                          'nearest_opposite_distance_um': float(distance),
                          'nearest_opposite_status': 'ok' if nearest_row else 'no_opposite_foci'}
                these_objects.append(record)
        raw_a = pixels[a - 1][crop][nuclear].astype(float)
        raw_b = pixels[b - 1][crop][nuclear].astype(float)
        corrected_a = raw_a - np.asarray(backgrounds[a][crop], dtype=np.float32)[nuclear]
        corrected_b = raw_b - np.asarray(backgrounds[b][crop], dtype=np.float32)[nuclear]
        positive_a, positive_b = np.maximum(corrected_a, 0), np.maximum(corrected_b, 0)
        raw_r, raw_status = pearson(raw_a, raw_b)
        corrected_r, corrected_status = pearson(corrected_a, corrected_b)
        signal = _signal_summary({
            'shared_mask_area_px': int(shared.sum()),
            'union_mask_area_px': int(((mask_a > 0) | (mask_b > 0)).sum()),
            'signal_a_total': float(positive_a.sum()), 'signal_a_in_b': float(positive_a[(mask_b > 0)[nuclear]].sum()),
            'signal_b_total': float(positive_b.sum()), 'signal_b_in_a': float(positive_b[(mask_a > 0)[nuclear]].sum())})
        nucleus_rows.append({**base, 'channel_a': a, 'channel_b': b, 'nuclear_pixels': int(nuclear.sum()),
                             'pearson_raw': raw_r, 'pearson_raw_status': raw_status,
                             'pearson_corrected': corrected_r, 'pearson_corrected_status': corrected_status,
                             **signal, **summarize(these_objects, these_pairs)})
        object_rows.extend(these_objects)
        pair_rows.extend(these_pairs)
    image = {**identity, 'channel_a': a, 'channel_b': b, 'nuclei_analyzed': len(nucleus_rows),
             'analysis_status': 'ok' if nucleus_rows else 'no_usable_nuclei',
             **summarize(object_rows, pair_rows)}
    for basis in ('pearson_raw', 'pearson_corrected'):
        valid = [r for r in nucleus_rows if np.isfinite(r[basis])]
        image.update({f'{basis}_mean': _stat(valid, basis), f'{basis}_median': _stat(valid, basis, 'median'),
                      f'{basis}_n_assessed': len(valid), f'{basis}_status': 'ok' if valid else 'no_valid_nuclei'})
    totals = {key: sum(r[key] for r in nucleus_rows) for key in
              ('shared_mask_area_px', 'union_mask_area_px', 'signal_a_total', 'signal_a_in_b',
               'signal_b_total', 'signal_b_in_a')}
    image.update(_signal_summary(totals))
    return ColocalizationResult(pd.DataFrame(pair_rows, columns=PAIR_COLUMNS),
                                pd.DataFrame(object_rows, columns=FOCUS_COLUMNS),
                                pd.DataFrame(nucleus_rows, columns=NUCLEUS_COLUMNS),
                                pd.DataFrame([image], columns=IMAGE_COLUMNS))


def save_tables(out, results):
    for name, columns in SCHEMAS.items():
        frames = [result.tables()[name] for result in results if not result.tables()[name].empty]
        table = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)
        table.to_csv(out / 'Results' / f'Colocalization_{name}.csv', index=False, float_format='%.12g')


def read_tables(out, parameters):
    if not settings(parameters)['enabled']:
        return {}
    tables = {}
    for name, columns in SCHEMAS.items():
        path = out / 'Results' / f'Colocalization_{name}.csv'
        if not path.exists():
            raise ValueError(f'Missing {path.name}; rerun analysis with colocalization enabled in a new output folder')
        table = pd.read_csv(path)
        if set(columns) - set(table):
            raise ValueError(f'Incomplete colocalization table: {path.name}')
        tables[name] = table
    return tables
