"""Channel selection shared by plotting and result validation."""
import json
import re

from image_io import inspect_image


def add_channel_arguments(parser):
    parser.add_argument('--nuclear-channel', type=int,
                        help='One-based nuclear channel; defaults to the saved analysis selection')
    parser.add_argument('--foci-channels', type=int, nargs='+',
                        help='One or more analyzed foci channels, e.g. 2 3; defaults to the saved selection')


def marker_name(parameters, channel):
    return parameters.get('channels', {}).get(str(channel), f'Channel {channel}')


def resolve_channels(out, nuclear_channel=None, foci_channels=None):
    """Prefer effective run parameters, then manifest, then legacy parameters.

    Explicit foci subsets are permitted. Existing nuclear masks cannot be
    relabeled as another channel: rerun analysis to change their source.
    """
    methods = out / 'Methods'
    parameters = {}
    for name in ['parameters.json', 'Effective_Parameters.json']:
        path = methods / name
        if path.exists():
            parameters.update(json.loads(path.read_text()))
    manifest_path = methods / 'Run_Manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for key in ['nuclear_channel', 'foci_channels', 'rgb_channel_order', 'pixel_size_um']:
        if key in manifest:
            parameters[key] = manifest[key]
    recorded_nuclear = parameters.get('nuclear_channel')
    recorded_foci = parameters.get('foci_channels')
    nuclear = nuclear_channel if nuclear_channel is not None else parameters.get('nuclear_channel', 1)
    foci = foci_channels if foci_channels is not None else parameters.get('foci_channels', [2, 3])
    if type(nuclear) is not int or nuclear < 1:
        raise ValueError('The nuclear channel must be a positive, one-based number')
    if not isinstance(foci, list) or not foci or any(type(c) is not int or c < 1 for c in foci):
        raise ValueError('Foci channels must be a nonempty list of positive, one-based numbers')
    if len(set(foci)) != len(foci):
        raise ValueError('Foci channels must not contain duplicates')
    if recorded_nuclear is not None and nuclear != recorded_nuclear:
        raise ValueError(f'Saved nuclear masks use channel {recorded_nuclear}, not {nuclear}. '
                         'Rerun analyze_foci.py to change the nuclear channel')
    if recorded_foci is not None and not set(foci) <= set(recorded_foci):
        raise ValueError(f'Requested foci channels were not analyzed. Saved channels: {recorded_foci}. '
                         'Rerun analyze_foci.py for additional channels')
    parameters.update(nuclear_channel=nuclear, foci_channels=list(foci))
    return parameters


def check_channel_inputs(out, parameters, mapping, images, cells, qc):
    nuclear = parameters['nuclear_channel']
    foci = parameters['foci_channels']
    for table in [images, cells, qc]:
        if 'nuclear_channel' in table and not table.nuclear_channel.dropna().eq(nuclear).all():
            raise ValueError('Selected nuclear channel disagrees with the saved result tables; rerun analysis')
    for c in foci:
        if f'C{c}_foci_count' not in cells or f'C{c}_total_foci' not in images:
            raise ValueError(f'No saved summary for foci channel {c}; rerun analyze_foci.py for this channel')
    for _, row in mapping.iterrows():
        md = inspect_image(row.source_path, parameters, require_calibration=False)
        if max([nuclear, *foci]) > md['shape'][0]:
            raise ValueError(f"{row.image_id}: requested channel is unavailable; source has {md['shape'][0]} channels")
        paths = [out / 'Masks' / f'{row.image_id}_nuclei.tif']
        paths += [out / 'Masks' / f'{row.image_id}_C{c}_foci.tif' for c in foci]
        for path in paths:
            if not path.exists():
                raise ValueError(f'Missing analysis mask: {path.name}; rerun analyze_foci.py')
        for c in foci:
            if len(qc[(qc.image_id == row.image_id) & (qc.channel == c)]) != 1:
                raise ValueError(f'{row.image_id}: expected one QC summary for channel {c}')


def selected_columns(table, channels):
    """Keep identifiers and measurements for the requested foci channels."""
    return table[[col for col in table if not (match := re.match(r'^C(\d+)_', col))
                  or int(match[1]) in channels]].copy()
