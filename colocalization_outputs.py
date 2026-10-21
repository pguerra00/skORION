"""Display-only colocalization figures and report content from saved tables."""
import html

import numpy as np
import pandas as pd
import tifffile
from skimage.segmentation import find_boundaries

from colocalization import settings, IDENTITY, SCHEMAS
from progress_reporting import advance


FIGURES = {
    'Correlations': 'Nuclear intensity correlation',
    'Associations': 'Directional focus associations',
    'Focus_Properties': 'Colocalized and non-colocalized focus properties',
    'Coverage_Distances': 'Coverage and spatial distances',
    'Partner_Counts': 'Number of opposite-channel partners',
    'Signal_Overlap': 'Directional signal fractions and mask overlap',
}
A_COLOR, B_COLOR = np.array([.96, .28, .68]), np.array([.20, .86, .60])


def overlay_rgb(nuclei, masks, foci, pairs, visible_channels=None, *, nuclear_visible=True, outlines=True):
    """Show accepted labels only; yellow pixels belong to qualifying pairs."""
    canvas = np.zeros((*nuclei.shape, 3), dtype=np.float32)
    if nuclear_visible:
        canvas[nuclei > 0] = [.025, .04, .06]
    roles = {}
    for role, color in [('A', A_COLOR), ('B', B_COLOR)]:
        rows = foci[foci.marker_role == role]
        if rows.empty:
            continue
        channel = int(rows.channel.iloc[0])
        roles[role] = channel
        if visible_channels is not None and channel not in visible_channels:
            continue
        mask = masks[channel]
        canvas[mask > 0] = color * .28
        labels = rows.loc[rows.is_colocalized == True, 'focus_label'].to_numpy(int)
        canvas[np.isin(mask, labels)] = color
    if len(pairs) and len(roles) == 2 and (visible_channels is None or set(roles.values()) <= set(visible_channels)):
        ma, mb = masks[roles['A']], masks[roles['B']]
        overlap = (ma > 0) & (mb > 0)
        # Encoded label pairs avoid a full image scan for every relationship.
        radix = int(mb.max()) + 1
        codes = ma[overlap].astype(np.int64) * radix + mb[overlap]
        qualifying = pairs.focus_label_a.to_numpy(np.int64) * radix + pairs.focus_label_b.to_numpy(np.int64)
        canvas[overlap] = np.where(np.isin(codes, qualifying)[:, None], [1, .88, .12], canvas[overlap])
    if nuclear_visible and outlines:
        canvas[find_boundaries(nuclei, mode='inner')] = [.35, .70, .95]
    return canvas


def methods_text(parameters):
    config = settings(parameters)
    if not config['enabled']:
        return ''
    a, b = config['channels']
    return f'''
## Colocalization analysis
Marker A is Channel {a}; Marker B is Channel {b}. A relationship requires at least
{config['minimum_overlap_pixels']} shared segmented pixels within the same accepted nucleus.
Touching objects without a common pixel do not qualify. Every qualifying pair is
retained; no one-to-one assignment, mask dilation, registration or new segmentation
is applied. Distinct associated foci and pair counts are separate: three A foci
overlapping one B focus yield three associated A foci, one associated B focus and
three relationships. Per-focus partner counts include zeros. Their averages are
reported both over all foci and only over associated foci.

Pair overlap fractions divide shared area by each respective focus area. Total
focus coverage uses the union of qualifying partners, counting each focus once.
Partner masks within a channel are disjoint. Areas use X times Y calibration;
geometric-centroid distances scale X and Y separately. Nearest opposite-channel
centroid distance includes nonassociated foci and never establishes an association.

Pearson coefficients use every pixel inside each accepted nucleus, independently
of focus detection. Raw coefficients use original unsmoothed pixels. Corrected
coefficients subtract the existing opening background in its saved float32
representation, preserving negative values. No display contrast, clipping or
significance test is used for Pearson. Image Pearson values are arithmetic means
and medians of finite nuclear coefficients, with assessed-nucleus counts; they
are not correlations of pixels pooled across nuclei.

Directional Manders coefficients divide the sum of positive corrected channel A
signal inside accepted B focus masks by all positive corrected A signal in the
same nucleus (and conversely). Negative corrected values are clipped only for
these signal-fraction measurements. Jaccard is mask intersection divided by union.
Manders and nuclear/image Jaccard measure complete accepted masks independently
of the minimum-pixel object-association rule. Image Manders and Jaccard use pooled
signal sums and pixel areas, respectively.

Image association fractions divide pooled unique-focus counts; focus properties
have equal weight per focus, while pair distances have equal weight per relationship.
Focus properties compare area, original mean intensity, local-corrected mean and
local-corrected integrated intensity, reusing the original focus measurements.
Treatment boxplots show image observations with equal image weight, median/IQR,
1.5-IQR whiskers and individual image points. ECDFs pool focus/pair observations
within treatments and are descriptive distributions. Partner-count bars show
pooled focus fractions for zero, one, two and three-or-more partners.

Counts remain zero when no foci/pairs exist. Empty-denominator fractions,
empty-category means, unavailable nearest neighbors and constant-input Pearson
coefficients are missing, with status fields identifying the cause. These are
descriptive 2D measurements; no hypothesis tests or biological replicate claims
are made. Colocalization tables and source-code/version hashes are saved with the run.

References: [MatCol](https://www.nature.com/articles/s41598-017-08786-1),
[scikit-image colocalization metrics](https://scikit-image.org/docs/stable/auto_examples/applications/plot_colocalization_metrics.html),
and [Manders coefficient implementation](https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.manders_coloc_coeff).
'''


def definition(column):
    """Colocalization-specific definitions take precedence over generic ones."""
    exact = {
        'marker_role': 'A or B according to the saved ordered colocalization channel pair.',
        'channel_a': 'One-based Marker A channel in the saved colocalization settings.',
        'channel_b': 'One-based Marker B channel in the saved colocalization settings.',
        'is_colocalized': 'True when this focus has at least one qualifying opposite-channel partner in its own nucleus.',
        'partner_count': 'Number of distinct qualifying opposite-channel focus partners; includes zero.',
        'overlap_fraction': 'Fraction of this focus covered by the union of all qualifying partner masks; each pixel counted once.',
        'overlap_fraction_a': 'Pair shared pixel area divided by the area of Marker A focus.',
        'overlap_fraction_b': 'Pair shared pixel area divided by the area of Marker B focus.',
        'overlap_area_px': 'Number of shared mask pixels for this qualifying A-B pair.',
        'overlap_area_um2': 'Pair shared pixel count multiplied by X and Y pixel sizes in µm.',
        'covered_area_px': 'Number of this focus\'s pixels covered by qualifying partners, without double counting.',
        'covered_area_um2': 'Covered pixel count multiplied by X and Y pixel sizes in µm.',
        'centroid_distance_um': 'Distance between unweighted pair centroids, scaling X and Y separately.',
        'nearest_opposite_distance_um': 'Nearest opposite-channel centroid distance within the same nucleus, including nonassociated foci.',
        'nearest_opposite_focus_id': 'ID of a nearest opposite-channel focus in the same nucleus; missing if none exists.',
        'jaccard': 'Intersection / union of segmented areas; pair masks for pairs, all accepted focus masks for nucleus/image rows.',
        'manders_a': 'Positive corrected A intensity inside accepted B focus masks / total positive corrected A nuclear intensity.',
        'manders_b': 'Positive corrected B intensity inside accepted A focus masks / total positive corrected B nuclear intensity.',
        'pearson_raw': 'Pearson coefficient of original unsmoothed values across every accepted nuclear pixel; no p-value.',
        'pearson_corrected': 'Pearson of original values minus saved float32 opening backgrounds; signed values preserved.',
        'pair_count': 'Number of qualifying A-B relationships; distinct-focus counts can differ in either direction.',
        'nuclear_pixels': 'Number of accepted nuclear pixels used for both Pearson coefficients.',
    }
    if column in exact:
        return exact[column]
    if column.endswith('_status') or column == 'analysis_status':
        return 'Assessment status: ok, or an explicit reason the corresponding measurement/category is undefined.'
    if column.startswith('pearson_'):
        return 'Image summary of finite per-nucleus Pearson coefficients: arithmetic mean, median or number assessed, as named; not a pooled-pixel correlation.'
    if column.startswith(('A_', 'B_')):
        role, rest = column.split('_', 1)
        if rest == 'colocalized_fraction':
            return f'Marker {role}: distinct foci with at least one qualifying partner / all {role} foci; undefined if there are no {role} foci.'
        if rest in ('total_foci', 'colocalized_foci'):
            return f'Marker {role}: count of distinct '+('accepted foci.' if rest == 'total_foci' else 'foci with at least one qualifying partner; each focus counted once.')
        if 'coverage_fraction_colocalized' in rest:
            return f'Marker {role}: mean or median total partner-covered fraction among associated foci only; equal weight per focus, undefined for an empty category.'
        if 'partner_count' in rest:
            return f'Marker {role}: arithmetic mean of distinct partner counts '+('over all foci, including zero-partner foci.' if rest.endswith('_all') else 'over associated foci only.')+' Undefined for an empty category.'
        if 'nearest_opposite_distance_um' in rest:
            return f'Marker {role}: mean or median finite nearest-opposite centroid distance in µm, including associated and unassociated foci; missing partners are omitted.'
        category, stat, prop = rest.split('_', 2)
        population = 'associated' if category == 'colocalized' else 'unassociated'
        description = {'focus_area_um2': 'segmented focus area in µm²',
                       'raw_mean_adu': 'original unsmoothed focus mean intensity',
                       'local_corrected_mean_adu': 'local-background-corrected focus mean intensity',
                       'local_corrected_integrated_adu_px': 'local-background-corrected focus integrated intensity (stored units × pixel)'}.get(prop, prop)
        return f'Marker {role}: {stat} {description} among {population} foci, equally weighted per focus; undefined for an empty category.'
    if column.startswith('signal_'):
        return 'Sum of max(original intensity - saved opening background, 0), in the named channel/partner mask; image rows pool nuclear sums.'
    if column in ('shared_mask_area_px', 'union_mask_area_px'):
        return 'Intersection or union pixel count of complete accepted A/B focus masks; independent of minimum overlap threshold.'
    if column in ('mean_centroid_distance_um', 'median_centroid_distance_um'):
        return 'Mean or median calibrated distance over qualifying pair rows, giving every relationship equal weight.'
    return None


def make_figures(out, mapping, parameters, tables, groups, colors, progress=None):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    folder = out / 'Plots' / 'Colocalization'
    folder.mkdir(parents=True, exist_ok=True)
    images, objects, pairs = tables['Images'], tables['Foci'], tables['Pairs']
    config = settings(parameters)
    a, b = config['channels']
    marker = lambda c: parameters.get('channels', {}).get(str(c), f'Channel {c}')
    title = f'A: C{a} {marker(a)} · B: C{b} {marker(b)}'
    rng = np.random.default_rng(parameters.get('random_seed_for_plot_jitter', 20261001))

    def save(fig, name):
        fig.suptitle(title + '\n' + FIGURES[name], fontsize=12)
        fig.savefig(folder / f'{name}.png', dpi=200, bbox_inches='tight')
        fig.savefig(folder / f'{name}.svg', bbox_inches='tight')
        plt.close(fig)
        advance(progress, 'Colocalization · ' + FIGURES[name])

    def boxes(ax, columns, ylabel):
        finite_count = 0
        for j, group in enumerate(groups):
            for k, col in enumerate(columns):
                vals = pd.to_numeric(images.loc[images.treatment_group == group, col], errors='coerce').to_numpy(float)
                vals = vals[np.isfinite(vals)]
                finite_count += len(vals)
                if not len(vals):
                    continue
                x = j + (k - (len(columns) - 1) / 2) * .32
                ax.boxplot([vals], positions=[x], widths=.24 if len(columns) > 1 else .4,
                           showfliers=False, patch_artist=True,
                           boxprops={'facecolor': colors[j] + ('40' if k == 0 else '12'), 'edgecolor': colors[j],
                                     'linestyle': '-' if k == 0 else '--'},
                           medianprops={'color': '#222222'}, whiskerprops={'color': colors[j]}, capprops={'color': colors[j]})
                ax.scatter(x + rng.uniform(-.06, .06, len(vals)), vals, s=8, alpha=.25,
                           color=colors[j], edgecolors='none', rasterized=True, zorder=3)
        ax.set_xticks(range(len(groups)), [g.replace(' + ', '\n') for g in groups], rotation=45, ha='right', fontsize=8)
        ax.set_ylabel(ylabel)
        ax.grid(axis='y', alpha=.16)
        if not finite_count:
            ax.text(.5, .5, 'No defined observations', transform=ax.transAxes, ha='center')

    fig, axs = plt.subplots(1, 2, figsize=(13, 5), layout='constrained')
    for ax, basis, label in zip(axs, ['pearson_raw_mean', 'pearson_corrected_mean'], ['Raw Pearson', 'Background-corrected Pearson']):
        boxes(ax, [basis], label + '\nMean of valid nuclear coefficients')
        ax.set_ylim(-1.05, 1.05)
    save(fig, 'Correlations')

    fig, axs = plt.subplots(3, 2, figsize=(13, 12), layout='constrained')
    for j, role in enumerate(('A', 'B')):
        for i, (suffix, label) in enumerate([
                ('colocalized_fraction', 'Fraction of foci with partners'),
                ('mean_partner_count_all', 'Mean partners / focus (including zeros)'),
                ('mean_partner_count_colocalized', 'Mean partners / associated focus')]):
            boxes(axs[i, j], [f'{role}_{suffix}'], f'{role}: {label}')
            if i == 0:
                axs[i, j].set_ylim(-.03, 1.05)
    save(fig, 'Associations')

    fig, axs = plt.subplots(4, 2, figsize=(13, 15), layout='constrained')
    for i, (prop, label) in enumerate([
            ('focus_area_um2', 'Mean area (µm²)'), ('raw_mean_adu', 'Mean raw focus intensity'),
            ('local_corrected_mean_adu', 'Mean local-corrected intensity'),
            ('local_corrected_integrated_adu_px', 'Mean local-corrected integrated intensity')]):
        for j, role in enumerate(('A', 'B')):
            boxes(axs[i, j], [f'{role}_{category}_mean_{prop}' for category in ('colocalized', 'noncolocalized')], f'{role}: {label}')
            axs[i, j].set_title('Left/solid: colocalized · right/faint: non-colocalized', fontsize=8)
    save(fig, 'Focus_Properties')

    def ecdf(ax, table, column, xlabel):
        count = 0
        for group, color in zip(groups, colors):
            vals = pd.to_numeric(table.loc[table.treatment_group == group, column], errors='coerce').to_numpy(float)
            vals = np.sort(vals[np.isfinite(vals)])
            if len(vals):
                ax.step(vals, np.arange(1, len(vals) + 1) / len(vals), where='post', color=color, label=group)
                count += len(vals)
        ax.set(xlabel=xlabel, ylabel='Cumulative fraction', ylim=(0, 1.03))
        if count:
            ax.legend(fontsize=7)
        else:
            ax.text(.5, .5, 'No defined observations', transform=ax.transAxes, ha='center')

    fig, axs = plt.subplots(2, 2, figsize=(13, 9), layout='constrained')
    for ax, role in zip(axs[0], ('A', 'B')):
        ecdf(ax, objects[(objects.marker_role == role) & (objects.is_colocalized == True)],
             'overlap_fraction', f'{role}: fraction covered by all qualifying partners')
        ax.set_xlim(0, 1)
    ecdf(axs[1, 0], pairs, 'centroid_distance_um', 'Qualifying pair centroid distance (µm)')
    ecdf(axs[1, 1], objects, 'nearest_opposite_distance_um', 'Nearest opposite-channel centroid (µm), all foci')
    save(fig, 'Coverage_Distances')

    fig, axs = plt.subplots(1, 2, figsize=(13, 5), layout='constrained')
    for ax, role in zip(axs, ('A', 'B')):
        for j, (group, color) in enumerate(zip(groups, colors)):
            counts = objects.loc[(objects.marker_role == role) & (objects.treatment_group == group), 'partner_count'].to_numpy(int)
            if len(counts):
                frequencies = np.bincount(np.minimum(counts, 3), minlength=4) / len(counts)
                width = .8 / max(len(groups), 1)
                ax.bar(np.arange(4) - .4 + width * (j + .5), frequencies, width, color=color, label=group)
        ax.set_xticks(range(4), ['0', '1', '2', '3+'])
        ax.set(xlabel=f'{role}: number of opposite-channel partners', ylabel='Fraction of foci', ylim=(0, 1.05))
        if ax.patches:
            ax.legend(fontsize=7)
    save(fig, 'Partner_Counts')

    fig, axs = plt.subplots(1, 3, figsize=(16, 5), layout='constrained')
    for ax, col, label in zip(axs, ['manders_a', 'manders_b', 'jaccard'],
                             ['A signal inside B masks (Manders)', 'B signal inside A masks (Manders)', 'Accepted mask Jaccard']):
        boxes(ax, [col], label)
        ax.set_ylim(-.03, 1.05)
    save(fig, 'Signal_Overlap')

    qdir = out / 'QC' / 'Colocalization'
    qdir.mkdir(parents=True, exist_ok=True)
    legend = [Line2D([], [], marker='s', linestyle='', color=color, label=label) for color, label in
              [(A_COLOR, 'A associated'), (A_COLOR * .28, 'A unassociated'),
               (B_COLOR, 'B associated'), (B_COLOR * .28, 'B unassociated'),
               ((1, .88, .12), 'Qualifying shared pixels'), ((.35, .70, .95), 'Accepted nucleus')]]
    for _, row in mapping.iterrows():
        nuclei = tifffile.imread(out / 'Masks' / f'{row.image_id}_nuclei.tif')
        masks = {c: tifffile.imread(out / 'Masks' / f'{row.image_id}_C{c}_foci.tif') for c in (a, b)}
        rgb = overlay_rgb(nuclei, masks, objects[objects.image_id == row.image_id], pairs[pairs.image_id == row.image_id])
        step = max(1, int(np.ceil(max(nuclei.shape) / 1600)))
        fig, ax = plt.subplots(figsize=(9, 8), layout='constrained')
        ax.imshow(rgb[::step, ::step], extent=(0, nuclei.shape[1] * row.pixel_size_x_um,
                  nuclei.shape[0] * row.pixel_size_y_um, 0), interpolation='nearest')
        ax.set(xlabel='X (µm)', ylabel='Y (µm)', title=str(row.image_name) + '\n' + title)
        ax.legend(handles=legend, loc='upper center', bbox_to_anchor=(.5, -.08), ncol=3, fontsize=7)
        fig.savefig(qdir / f'{row.image_id}_Colocalization.png', dpi=200, bbox_inches='tight')
        plt.close(fig)
        advance(progress, f'Colocalization overlay · {row.image_name}')


def report_section(out, parameters, tables):
    if not tables:
        return ''
    esc = lambda value: html.escape(str(value))
    config = settings(parameters)
    a, b = config['channels']
    marker = lambda c: parameters.get('channels', {}).get(str(c), f'Channel {c}')
    images = tables['Images']
    body = '<section id="colocalization"><h2>Colocalization analysis</h2>'
    body += f'<p>Marker A: Channel {a} · {esc(marker(a))}. Marker B: Channel {b} · {esc(marker(b))}. '
    body += f'At least {config["minimum_overlap_pixels"]} shared pixels within one accepted nucleus establish an association. '
    body += 'All qualifying relationships are retained; focus counts and relationship counts are separate.</p>'
    body += '<p>Three A foci overlapping one B focus count as three associated A foci, one associated B focus, and three relationships. '
    body += 'Coverage is the fraction of each focus covered by all qualifying partners together. '
    body += 'Pair CSVs retain each individual overlap fraction and center distance.</p><ul>'
    for name in SCHEMAS:
        body += f'<li><a href="Results/Colocalization_{name}.csv">Colocalization {name.lower()} CSV</a></li>'
    body += '</ul>'
    summary = images.groupby('treatment_group', sort=False).agg(
        Images=('image_id', 'size'), Nuclei=('nuclei_analyzed', 'sum'),
        Relationships=('pair_count', 'sum'), A_foci=('A_total_foci', 'sum'),
        A_associated=('A_colocalized_foci', 'sum'), B_foci=('B_total_foci', 'sum'),
        B_associated=('B_colocalized_foci', 'sum'),
        Mean_image_A_fraction=('A_colocalized_fraction', 'mean'),
        Mean_image_B_fraction=('B_colocalized_fraction', 'mean'),
        Mean_image_raw_Pearson=('pearson_raw_mean', 'mean'),
        Mean_image_corrected_Pearson=('pearson_corrected_mean', 'mean')).reset_index()
    summary = summary.rename(columns={
        'treatment_group': 'Treatment', 'A_foci': 'A foci', 'A_associated': 'A with partners',
        'B_foci': 'B foci', 'B_associated': 'B with partners',
        'Mean_image_A_fraction': 'Mean image A fraction', 'Mean_image_B_fraction': 'Mean image B fraction',
        'Mean_image_raw_Pearson': 'Mean image raw Pearson',
        'Mean_image_corrected_Pearson': 'Mean image corrected Pearson'})
    body += '<div style="overflow-x:auto">' + summary.to_html(index=False, border=0, float_format=lambda x: f'{x:.4g}', na_rep='Undefined') + '</div>'
    body += '<p>Boxes summarize image observations with equal image weight; points are individual images. '
    body += 'Image fractions pool unique-focus counts, focus-property averages weight each focus equally, and pair distances weight each relationship equally. '
    body += 'Image Pearson values average valid per-nucleus coefficients over all nuclear pixels; raw and background-corrected values are separate. '
    body += 'Corrected Pearson retains negative values. Positive corrected signal is used for directional Manders coefficients. '
    body += 'Mask Jaccard and Manders use complete accepted masks independently of the association threshold. '
    body += 'Zero counts are retained; undefined measurements are missing with reasons in status columns. No statistical tests are performed.</p>'
    for filename, title in FIGURES.items():
        body += f'<article><h3>{esc(title)}</h3><img loading="lazy" src="Plots/Colocalization/{filename}.png" alt="{esc(title)}">'
        body += f'<p><a href="Plots/Colocalization/{filename}.png">PNG</a> · <a href="Plots/Colocalization/{filename}.svg">Editable SVG</a></p></article>'
    body += '<p>Coverage/distance ECDFs pool focus or relationship observations within each treatment. '
    body += 'Partner bars include zero-partner foci; exact counts remain in the focus CSV. '
    body += 'Nearest opposite-channel distance includes non-colocalized foci and does not establish an association.</p>'
    body += '<details><summary>Inspect colocalization in every image</summary><div class="gallery">'
    for _, row in images.iterrows():
        path = f'QC/Colocalization/{row.image_id}_Colocalization.png'
        body += f'<a href="{path}"><img loading="lazy" src="{path}" alt="{esc(row.image_name)}"><small>{esc(row.image_name)}</small></a>'
    body += '</div></details><p>Methods references: <a href="https://www.nature.com/articles/s41598-017-08786-1">MatCol</a> and '
    body += '<a href="https://scikit-image.org/docs/stable/auto_examples/applications/plot_colocalization_metrics.html">scikit-image colocalization metrics</a>. '
    body += '<a href="Methods/Methods.md">Full definitions and aggregation rules</a>.</p></section>'
    return body
