# Selecting channels in analyze_foci

Channel numbers start at **1** and refer to the channel planes in the TIFF. The
input can be a calibrated, multichannel 2D ImageJ TIFF in CYX order or an RGB
TIFF with an explicit color mapping. Channel stacks are not restricted to three.

## RGB TIFF exports (including EVOS)

For blue nuclei, green γH2AX, and red RAD51, add this option to analysis:

```sh
python analyze_foci.py --input-dir /path/to/tiffs --parameters /path/to/parameters.json --output-dir /path/to/results --rgb-channel-order blue green red --qc-output detailed
```

This assigns C1 = blue, C2 = green, and C3 = red. Alternatively save
`"rgb_channel_order": ["blue", "green", "red"]` in the experiment parameters.
Each color must appear exactly once; the program does not guess stain identity.
Plotting and validation reuse the saved mapping automatically. The option has
no effect on existing CYX stacks.

RGB exports must contain one 2D image with exactly three color components and
physical calibration in OME metadata, ImageJ micron metadata, TIFF inch/centimeter
resolution tags, or an explicit pixel-size override. EVOS files with physical resolution tags are supported.
macOS `._` companions are ignored. Keep `image_io.py` beside the other scripts.

Stored component values keep their native dtype and range: 8-bit RGB stays
0–255. Threshold floors apply on that scale; saturation/clipping is counted
at 255 for uint8 and 65535 for uint16. RGB export values can reflect display
adjustments and do not recover original detector measurements. Review the
segmentation QC and treat RGB results as exploratory. Source layout, mapping,
bit depth, calibration, and this limitation are recorded in the generated outputs.

## Pixel calibration

Pixel sizes are read in this order: an explicit override, OME `PhysicalSizeX`
and `PhysicalSizeY` (converted to µm), ImageJ micron metadata, then TIFF
inch/centimeter resolution. Missing OME physical-size units default to µm
according to the [OME schema](https://www.openmicroscopy.org/Schemas/Documentation/Generated/OME-2016-06/ome_xsd.html).
A file with no physical calibration requires
pixel sizes from the original acquisition settings; objective magnification
alone is insufficient.

Supply both X and Y sizes with `--pixel-size-um X Y` (µm per pixel), or save
`"pixel_size_um": [X, Y]` in the parameters JSON. Replace X and Y with actual
positive numbers; use the same value twice for square pixels. This explicitly
overrides stored calibration for every image in that run. The applied scale
and its source are recorded in the image mapping, metadata, and run parameters.
OME channel stacks use the stored plane order; the RGB option only applies to
RGB exports.

## Selecting planes

From the Foci_Analysis folder, analyze only channel 2 using channel 1 for nuclei:

```sh
python Scripts/analyze_foci.py --input-dir /path/to/tiffs --parameters /path/to/parameters.json --output-dir /path/to/new_results --nuclear-channel 1 --foci-channels 2
```

To analyze both channels 2 and 3, use `--foci-channels 2 3`. To segment nuclei
from a different plane, change `--nuclear-channel` to its channel number.
The nuclear channel can also be included in the foci list if that is intended.

Save selections in the parameter file for your experiment (each run saves a copy at `Methods/parameters.json` inside its analysis folder):

```json
"nuclear_channel": 1,
"foci_channels": [2, 3]
```

Command-line options override these settings. Older parameter files without
these keys continue to use nuclear channel 1 and foci channels 2 and 3.

Each selected foci channel needs its own threshold floors under `foci.channel_N`
in the parameters file. The reusable `parameters.template.json` has example settings for channels 2 and 3.
For a new foci channel, add its `support_floor_adu`, `seed_floor_adu`, and
`prominence_floor_adu` settings; values should be chosen for that signal.
When rearranging planes, move the threshold settings and the marker names in
`channels` to match the new channel numbers. The script does not infer marker
identity or transfer thresholds from another channel. Unknown marker names
appear as `Channel N`.

Foci masks, measurements, sensitivity analysis, and QC summaries use only the
selected foci channels. Input intensity distributions still describe all input
channels. Nuclear measurements identify the selected channel and marker;
legacy DAPI columns are retained when the configured nuclear marker is DAPI.
`Methods/Run_Manifest.json` records the selected channels and their focus counts.
`Methods/Effective_Parameters.json` saves the parameters with command-line
channel overrides applied.

## Plots, reports, and validation

`make_outputs.py` and `validate_results.py` accept the same channel options:

```sh
python Scripts/make_outputs.py --output-dir /path/to/new_results --nuclear-channel 1 --foci-channels 2
python Scripts/validate_results.py --output-dir /path/to/new_results --nuclear-channel 1 --foci-channels 2
```

You can omit both options to reuse the analysis selection. These scripts load
`Effective_Parameters.json`, use the recorded channel selection from
`Run_Manifest.json` when available, and fall back to `parameters.json` for older
runs. Legacy runs without channel settings retain C1 nuclei and C2/C3 foci.
Marker labels come from the saved `channels` mapping, with `Channel N` as the
fallback. Explicit foci selections may be a subset of the analyzed channels.
Changing the nuclear channel or adding an unanalyzed foci channel requires
rerunning `analyze_foci.py`; plotting and validation do not redo segmentation.

QC panel counts, nuclear overlays, plots, report tables, and validation counts
follow the selection. A channel may have both nuclear and foci roles, with
separate QC rows. Nuclear overlay filenames are `Ixx_CN_nuclear_outlines.png`.
`Output_Parameters.json` records the report selection; `Validation_Report.json`
records the validation selection. The validation checks nuclear mean and median
intensities against the selected source plane as well as reconciling foci.

Use a new output directory for a different analysis: existing files from a
previous run are not removed. The workbook renderer `make_workbook.mjs` still
has column-order assumptions for C2/C3 and has not been updated here.

See [README.md](README.md) for the directory layout and starting a new analysis.
