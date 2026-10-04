# Selecting channels in analyze_foci

Channel numbers start at **1** and refer to the channel planes in the TIFF. The
input must be a calibrated, multichannel 2D ImageJ TIFF in CYX order. The number
of channels is no longer restricted to three.

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
