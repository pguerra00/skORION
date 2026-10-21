"""Read-only GUI previews using the same segmentation and measurements as the CLI."""
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from skimage import measure, morphology, segmentation

from analyze_foci import (detect_nucleus, focus_measurements, nuclear_segmentation,
                          select_channels, summarize_cell, validate_group_names,
                          validate_nuclear_threshold_range)
from image_io import read_image, validate_pixel_size, validate_rgb_channel_order
from colocalization import calculate as calculate_colocalization, settings as colocalization_settings

FLOOR_KEYS = ('support_floor_adu', 'seed_floor_adu', 'prominence_floor_adu')


def list_images(directory):
    directory = Path(directory).expanduser().resolve()
    if not directory.is_dir():
        raise ValueError('Choose an existing input directory.')
    paths = sorted(p for p in directory.iterdir()
                   if p.is_file() and not p.name.startswith('._')
                   and p.suffix.lower() in {'.tif', '.tiff'})
    if not paths:
        raise ValueError('No TIFF images were found in the input directory.')
    return paths


def validate_parameters(parameters, channel_count=None):
    """Validate GUI settings before starting a worker or writing run outputs."""
    p = deepcopy(parameters)
    nuclear, foci = select_channels(p)
    p['colocalization'] = colocalization_settings(p, foci, channel_count)
    # The engine permits measuring the nuclear stain for foci as well. Keep the
    # selections independent rather than silently reassigning any channel.
    validate_nuclear_threshold_range(p['nuclear'])
    validate_group_names(p.get('group_names', []))
    validate_rgb_channel_order(p.get('rgb_channel_order'))
    validate_pixel_size(p.get('pixel_size_um'))
    if channel_count is not None and max(nuclear, *foci) > channel_count:
        raise ValueError(f'This image has {channel_count} channels. Select available channels.')
    for section in ('nuclear', 'foci', 'qc'):
        if not isinstance(p.get(section), dict):
            raise ValueError(f'{section} must be a JSON object.')
        for key, value in p[section].items():
            if type(value) in (int, float) and (not np.isfinite(value) or value < 0):
                raise ValueError(f'{section}.{key} must be finite and nonnegative.')
    for key in ('gaussian_sigma_px',):
        if p['foci'][key] <= 0:
            raise ValueError(f'foci.{key} must be greater than zero.')
    for section, keys in {
        'nuclear': ('closing_disk_radius_px', 'candidate_min_area_px', 'image_edge_buffer_px'),
        'foci': ('background_disk_radius_px', 'minimum_area_px', 'annulus_min_pixels'),
    }.items():
        for key in keys:
            if type(p[section][key]) is not int:
                raise ValueError(f'{section}.{key} must be an integer.')
    if p['nuclear']['accepted_min_area_um2'] > p['nuclear']['accepted_max_area_um2']:
        raise ValueError('Minimum nuclear area must not exceed maximum nuclear area.')
    for key in ('connectivity', 'watershed_connectivity'):
        if p['foci'][key] not in (1, 2):
            raise ValueError(f'foci.{key} must be 1 or 2 for a 2D image.')
    if p['foci']['background_disk_decomposition'] not in (None, 'sequence', 'crosses'):
        raise ValueError('Foci background disk decomposition must be sequence, crosses, or null.')
    factors = p['foci']['sensitivity_threshold_factors']
    if (not isinstance(factors, list) or not factors
            or any(type(v) not in (int, float) or not np.isfinite(v) or v <= 0 for v in factors)):
        raise ValueError('Sensitivity threshold factors must be a nonempty list of positive numbers.')
    if not {0.8, 1.2}.issubset(factors):
        raise ValueError('Sensitivity factors must include 0.8 and 1.2, which the saved QC columns require. Additional factors are allowed.')
    return p


@dataclass
class Preview:
    path: Path
    parameters: dict
    pixels: np.ndarray
    metadata: dict
    nuclei: np.ndarray
    candidates: np.ndarray
    nuclear_audit: list
    foci_masks: dict
    cells: list
    foci: list
    thresholds: list
    colocalization: object = None


def analyze_preview(path, parameters):
    """Compute one image without saving files; mirror CLI cropping and labeling."""
    p = validate_parameters(parameters)
    arr, md = read_image(path, p)
    validate_parameters(p, arr.shape[0])
    sx, sy = md['pixel_size_x_um'], md['pixel_size_y_um']
    usable, candidates, audit = nuclear_segmentation(arr[p['nuclear_channel'] - 1], p['nuclear'], sx, sy)
    cells = {r['nucleus_label']: dict(r) for r in audit if r['included']}
    masks, all_foci, thresholds, backgrounds = {}, [], [], {}
    saturation = md['saturation_value'] if md['saturation_value'] is not None else np.inf
    for channel in p['foci_channels']:
        raw = arr[channel - 1].astype(float)
        smoothed = ndi.gaussian_filter(raw, p['foci']['gaussian_sigma_px'], mode='reflect')
        background = morphology.opening(smoothed, morphology.disk(
            p['foci']['background_disk_radius_px'], decomposition=p['foci']['background_disk_decomposition']))
        corrected = smoothed - background
        mask = np.zeros_like(usable)
        global_label = 0
        for region in measure.regionprops(usable):
            y0, x0 = max(0, region.bbox[0] - 10), max(0, region.bbox[1] - 10)
            y1, x1 = min(raw.shape[0], region.bbox[2] + 10), min(raw.shape[1], region.bbox[3] + 10)
            crop = (slice(y0, y1), slice(x0, x1))
            nucleus = usable[crop] == region.label
            labels, extra, _, applied = detect_nucleus(
                raw[crop], background[crop], corrected[crop], nucleus, channel, p['foci'], sx * sy)
            rows = focus_measurements(labels, raw[crop], background[crop], extra, nucleus,
                                      (y0, x0), region.centroid, sx, sy, saturation)
            for row in rows:
                local_label = row.pop('local_focus_label')
                global_label += 1
                mask[crop][labels == local_label] = global_label
                row.update(nucleus_label=region.label, channel=channel, focus_label=global_label)
            all_foci.extend(rows)
            summary = summarize_cell(rows, region.area, region.area * sx * sy)
            cells[region.label].update({f'C{channel}_{key}': value for key, value in summary.items()})
            thresholds.append(dict(nucleus_label=region.label, channel=channel, **applied))
        masks[channel] = mask
        if p['colocalization']['enabled'] and channel in p['colocalization']['channels']:
            backgrounds[channel] = background.astype(np.float32)
    colocalization = None
    if p['colocalization']['enabled']:
        colocalization = calculate_colocalization(arr, usable, masks, backgrounds, all_foci, p, sx, sy,
                                                 {'image_id': 'preview', 'image_name': Path(path).name})
        for record in colocalization.nuclei.to_dict('records'):
            cells[record['nucleus_label']].update({f'coloc_{key}': value for key, value in record.items()
                                                  if key not in ('nucleus_label', 'nucleus_id')})
    return Preview(Path(path), p, arr, md, usable, candidates, audit, masks,
                   list(cells.values()), all_foci, thresholds, colocalization)


def channel_color(channel):
    return np.array(((100, 235, 165), (247, 146, 210), (90, 210, 245),
                     (250, 189, 90), (194, 155, 248))[(channel - 1) % 5], dtype=float)


def default_display_colors(channel, nuclear_channel):
    """Initial signal and outline colors; GUI choices stay with channel numbers."""
    if channel == nuclear_channel:
        return (80, 120, 240), (105, 215, 245)
    color = tuple(int(value) for value in channel_color(channel))
    return color, color


def display_signal(raw):
    """Display-only percentile contrast; never modifies analysis pixel values."""
    low, high = np.percentile(raw, [1, 99.8])
    if high <= low:
        high = float(np.max(raw))
    if high <= low:
        return np.zeros(raw.shape, dtype=float)
    return np.clip((raw.astype(float) - low) / (high - low), 0, 1)


def render_image(pixels, parameters, preview=None, mode='Overlay', raw_channel=None,
                 visible_channels=None, outlines=True, selected=0,
                 signal_colors=None, outline_colors=None):
    """Render display-only pseudocolors, without changing source pixels or masks."""
    n = parameters['nuclear_channel']
    foci = parameters['foci_channels']
    visible = set([n, *foci] if visible_channels is None else visible_channels)

    def color(channel, outline=False):
        palette = (outline_colors if outline else signal_colors) or {}
        defaults = default_display_colors(channel, n)
        return np.asarray(palette.get(channel, defaults[1 if outline else 0]), dtype=float)

    canvas = np.zeros((*pixels.shape[1:], 3), dtype=float)
    if mode == 'Raw image':
        channel = raw_channel or n
        signal = display_signal(pixels[channel - 1])
        canvas[:] = signal[..., None] * color(channel)
    elif mode == 'Overlay':
        if n in visible:
            canvas += display_signal(pixels[n - 1])[..., None] * color(n)
        for channel in foci:
            if channel in visible and channel != n:
                canvas += display_signal(pixels[channel - 1])[..., None] * color(channel) * .65
    if preview is not None:
        if mode == 'Colocalization' and getattr(preview, 'colocalization', None) is not None:
            from colocalization_outputs import overlay_rgb
            canvas = overlay_rgb(preview.nuclei, preview.foci_masks,
                                 preview.colocalization.foci, preview.colocalization.pairs,
                                 visible_channels=visible, nuclear_visible=n in visible, outlines=outlines) * 255
        if mode == 'Nuclei' and n in visible:
            canvas[preview.nuclei > 0] = color(n) * .45
        if mode in ('Overlay', 'Nuclei') and outlines and n in visible:
            canvas[segmentation.find_boundaries(preview.nuclei, mode='inner')] = color(n, outline=True)
        if mode in ('Overlay', 'Foci'):
            for channel, mask in preview.foci_masks.items():
                if channel in visible:
                    if mode == 'Foci':
                        canvas[mask > 0] = color(channel)
                    if outlines:
                        canvas[segmentation.find_boundaries(mask, mode='inner')] = color(channel, outline=True)
        if selected and mode != 'Raw image':
            canvas[segmentation.find_boundaries(preview.candidates == selected, mode='inner')] = [255, 219, 100]
    return np.ascontiguousarray(np.clip(canvas, 0, 255).astype(np.uint8))


def validate_run_paths(input_dir, output_dir, parameter_file=None):
    source, output = Path(input_dir).expanduser().resolve(), Path(output_dir).expanduser().resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError('Input and output directories must be separate and must not contain each other.')
    if output == Path(__file__).resolve().parent:
        raise ValueError('Choose an analysis output directory separate from the application folder.')
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError('Choose a new or empty output directory for this analysis.')
    if parameter_file:
        config = Path(parameter_file).expanduser().resolve()
        if config == source or source in config.parents:
            raise ValueError('Save parameter files outside the raw input directory.')
    return source, output
