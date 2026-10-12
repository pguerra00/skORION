# scikitORION

Reproducible, calibrated 2D nuclear-foci analysis for multichannel ImageJ TIFF
images and explicitly mapped RGB TIFF exports. scikitORION segments nuclei, detects foci in one or more selected
channels, records measurement provenance, produces quality-control material and
reports, and validates the generated result tables.

The pipeline can be used from the command line or an optional desktop GUI. The
analysis scripts can be copied into an analysis workspace without package
installation or a project-specific runtime. Raw
images are read-only inputs and are never copied into an analysis output
directory.

## Features

- Calibration-aware processing of 2D `CYX` ImageJ TIFF files.
- Input extensions `.tif` and `.tiff` are accepted in any letter case, including
  `.TIF` and `.TIFF`; files keep their original names. macOS `._` metadata
  companions on external drives are ignored.
- LZW-compressed TIFFs are decoded through the included `imagecodecs` dependency.
- Configurable nuclear segmentation and per-channel foci detection.
- One-based command-line channel selection with saved run manifests.
- Per-image, per-nucleus, and per-focus measurement tables.
- Masks, annotated QC images, shared-contrast figures, plots, Markdown/HTML
  review material, methods records, and validation reports.
- SHA-256 source-image hashes and version/parameter records for provenance.
- Regression tests using synthetic TIFF data.

## Pipeline layout

The repository contains the reusable scripts, configuration template, and
tests. A typical analysis output is separate from this repository:

```text
scikitORION/
├── analyze_foci.py
├── make_outputs.py
├── validate_results.py
├── channel_config.py
├── image_io.py
├── make_workbook.mjs
├── parameters.template.json
├── requirements.txt
└── tests/

Analyses/my_experiment/
├── Results/
├── QC/
├── Plots/
├── Masks/
├── Methods/
├── Report.md
└── Review.html
```

Keep the scripts together when copying them into an existing analysis
workspace. The generated output records the script version and effective
parameters used for that run.

Keep `analyze_foci.py`, `make_outputs.py`, `validate_results.py`,
`channel_config.py`, and `image_io.py` together. You can copy this folder's contents into your
working directory. No installation as a Python package is required; install
the dependencies in `requirements.txt` in your Python environment.

`parameters.template.json` is a starting template with example numerical
settings. Make a separate copy for each experiment and review channel labels,
threshold floors, nuclear size limits, and experimental metadata before running.
Existing experiment settings are stored with that experiment's outputs, not here.

## Start a named analysis

From the repository root (or from a copied scripts directory):

```sh
mkdir -p Analyses/my_experiment/Methods
cp parameters.template.json Analyses/my_experiment/Methods/parameters.json
```

Edit `Analyses/my_experiment/Methods/parameters.json` for your images, then run:

```sh
python analyze_foci.py --input-dir /path/to/tiffs --parameters Analyses/my_experiment/Methods/parameters.json --output-dir Analyses/my_experiment --nuclear-channel 1 --foci-channels 2 3
python make_outputs.py --output-dir Analyses/my_experiment
python validate_results.py --output-dir Analyses/my_experiment
```

Paths are resolved from your current working directory, not from the location
of the scripts.

The last two commands reuse the analysis's saved channel selections. See
[CHANNELS.md](CHANNELS.md) for overrides, marker labels, and selecting subsets.

## Requirements and installation

- Python 3.12 is the validated runtime for the pinned dependency set.
- A calibrated multichannel ImageJ/OME TIFF with `CYX` axes, or a calibrated 2D RGB
  TIFF with an explicit color-to-channel mapping, is required.
- If physical pixel size was lost during export, supply the acquisition
  calibration with `--pixel-size-um X Y` in µm per pixel. See
  [CHANNELS.md](CHANNELS.md) for calibration sources and overrides.
- Node.js is optional and is only needed for workbook generation.

Install the Python dependencies in a new Python 3.12 virtual environment:

```sh
python3.12 -m venv .venv
. .venv/bin/activate       # Windows PowerShell: .venv\Scripts\Activate.ps1
python --version          # Must report Python 3.12 or newer
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows, create the environment with `py -3.12 -m venv .venv`.
If `python3.12` is unavailable but Anaconda provides Python 3.12, use
`/opt/anaconda3/bin/python3.12` on macOS.

An error such as "No matching distribution found for numpy==2.5.3" with
"Requires-Python >=3.12" means the active interpreter is too old. Check
`python --version`; the terminal's `(.venv)` prompt does not identify its
Python version. Recreating an environment over an older environment can leave
its old Python executable in place. Use a new directory instead, or switch to
the Python 3.12 GUI environment below if it is already installed:

```sh
deactivate
. .venv-gui/bin/activate
python --version
```

The pinned versions in `requirements.txt` are deliberately recorded with each
analysis output. Review and validate any dependency changes before using them
for a production comparison.

`imagecodecs` provides decoding for LZW-compressed TIFFs and is included in both
the command-line and GUI dependency lists. If an existing environment reports
"COMPRESSION.LZW requires the 'imagecodecs' package", reinstall the updated
requirements in that environment and restart the GUI:

```sh
.venv-gui/bin/python -m pip install -r requirements-gui.txt
.venv-gui/bin/python orion_gui.py
```

The compression format is independent of the filename's letter case. No TIFF
conversion or renaming is needed. New runs record the decoder version in
`Methods/Run_Manifest.json` and the generated `Methods/requirements.txt`.

## Desktop GUI

The optional Qt desktop application provides a resizable image workspace and a
scrolling setup sidebar. Use Python 3.12 and a separate environment:

```sh
python3.12 -m venv .venv-gui
.venv-gui/bin/python -m pip install -r requirements-gui.txt
.venv-gui/bin/python orion_gui.py
```

On macOS, after this setup, double-click `launch_gui.command` to open the GUI.
On Windows, activate the environment and run `python orion_gui.py`.
You can also start with `--input-dir`, `--output-dir`, and `--parameters` options.

1. Enter or browse to an input directory, an output directory, and a parameters
   directory. **Load JSON** reads the named experiment file; **Save JSON…** saves
   your edited configuration separately. The bundled template is loaded initially.
2. Choose the nuclear channel and any foci channels independently. All selectors
   use neutral labels (`Channel 1`, `Channel 2`, etc.); there is no fixed stain
   assignment. For example, choose nuclear Channel 3 and foci Channels 1 and 2.
   Images with more than three channels expand the available choices. Foci ADU
   floors belong to their physical channel and appear for each selected channel.
   A newly configured channel starts with zero floors, so noise thresholds apply
   until you enter experiment-specific floors. Review these settings before a run.
   For RGB TIFF exports, enable **Map RGB TIFF colors to channels** in Basic
   parameters and choose a color for Channels 1, 2, and 3, using red, green, and
   blue once each. For blue nuclei, green Channel 2, and red Channel 3, choose
   Blue / Green / Red and nuclear Channel 1. Other assignments are equally valid;
   nuclear and foci roles are selected separately. This mapping is saved as
   `rgb_channel_order` in JSON and is used by both previews and full runs.
   Channel stacks retain their original channel order.
3. Choose automatic nuclear thresholding (the default) or an inclusive manual
   ADU range. Expand **Advanced parameters** to edit the remaining numeric,
   Boolean, metadata, and JSON-list settings. Fixed method descriptions remain
   visible and read-only because changing that text would not change the algorithm.
   `pixel_size_um` accepts a JSON list or `null`, matching the command-line
   calibration override. Sensitivity factors
   must retain 0.8 and 1.2 for the existing QC summary columns; extra factors may
   be added. Generated methods describe the effective numerical settings.
4. **Load images** discovers TIFF files and previews the first image. **Preview
   image** computes real masks and measurements for the selected image without
   saving output files. Switch between raw image, nuclei, foci, and overlay views;
   toggle channels and outlines, zoom with the mouse wheel or buttons, and drag
   to pan. Click a nucleus or use the inspector selector to see its area,
   exclusions, foci counts, and applied thresholds. Display contrast changes only
   the view, not the pixels used for detection. Edits invalidate old preview masks.
5. **Run analysis** processes the whole input directory, generates the selected
   QC/report detail, and validates the results in background processes. The log
   shows progress and failures; **Stop** stops the active stage and preserves
   partial output. A new run requires a new or empty output directory, separate
   from the raw input directory. The loaded parameter file and raw TIFFs are
   unchanged: the current settings are captured with the run in `Methods`.

The GUI runs analysis, report generation, and validation as separate stages so
each can be stopped. `Effective_Parameters.json` records analysis with automatic
report generation off; `Output_Parameters.json` records the separately generated
QC detail. **Results** displays cell measurements (up to 2,000 rows; full tables
remain in CSV). **Load saved results** reads the output directory's saved table,
and **Open report** opens its generated review in your browser. Saving parameters
is always an explicit action.

This first desktop version launches from Python; it is not yet packaged as a
standalone installer. The original command-line workflow remains available
without the Qt dependency.

## Fixed nuclear thresholds

To reproduce a manually chosen nuclear intensity range, add
`"fixed_threshold_adu": [131, 255]` inside the existing `nuclear` object in your
experiment's parameters. Keep the object's other settings. The bounds are
inclusive and apply to the original nuclear-channel pixels, before intensity
smoothing or background subtraction. For RGB TIFFs, these are the values of
the individual color component selected by your explicit RGB channel mapping.

The range replaces the adaptive median/MAD cutoff for nuclear foreground
selection. Closing, hole filling, watershed splitting, and nucleus size/border
filters still follow it, so the final masks can differ from FIJI's initial
threshold selection. Foci threshold settings are independent of this option.
The nucleus audit and cell table record the threshold method and both bounds;
the generated methods describe the active rule. Set the entry to `null`, or
remove it, to use the original adaptive method. The `threshold_rule` string is
descriptive text, not an executable expression.

Changing nuclear thresholding requires rerunning `analyze_foci.py` to create
new masks and measurements. Regenerating plots alone does not rerun detection.

## Optional adaptive focus splitting

When large bright foci split into multiple objects while higher blur or
prominence loses small foci, enable adaptive splitting for the affected
channel. Add these entries inside its existing `foci.channel_N` object:

```json
"adaptive_split_prominence_fraction": 0.30,
"adaptive_split_minimum_peak_adu": 400.0,
"adaptive_split_minimum_area_um2": 2.0,
"adaptive_split_post_watershed_peak_fraction": 0.10
```

These are starting values tuned on the supplied IFPG46 channel-3 image, not
universal defaults. The fraction is dimensionless and must be between 0 and
1. It defaults to 0 (disabled), preserving the original segmentation. Missing
brightness and area entries default to 100 ADU and 2 µm².

The usual support, seed and absolute prominence thresholds first detect
candidate peaks. Within a connected support region at least the configured
area, secondary peaks at least the configured brightness must also survive
a prominence threshold of `max(absolute prominence, fraction × that peak's
background-corrected intensity)`. The region's strongest seed is retained.
Faint peaks below the brightness cutoff and smaller support regions retain
the original splitting rule. Thus the extra criterion targets shallow peaks
within broad bright regions without increasing the global detection floor
or smoothing. Separate bright peaks with sufficiently deep valleys remain
separate; overlapping projected signals still require visual judgment.

The optional `adaptive_split_post_watershed_peak_fraction` applies only to
basins that receive a suppressed secondary peak. Lowering it preserves dimmer
edges of a merged large focus without changing the trimming of other foci.
It must be between 0 and 1; if omitted, the normal
`foci.post_watershed_peak_fraction` is used. Its support-threshold floor and
connected-component rule remain in effect. The tuned value above is 0.10;
ordinary unmerged basins keep their original 0.25 setting.

The GUI's **Advanced parameters** exposes these channel-specific entries
when present in the loaded file. GUI previews and CLI runs use the same rule.
`Methods/Applied_Thresholds.csv` and cell summaries record the enabled settings,
number of secondary peaks tested/suppressed and maximum tested prominence.
The existing `prominence_threshold_adu` remains the absolute baseline.
Generated methods describe enabled channels and numerical settings. Threshold
sensitivity varies the original absolute thresholds while adaptive settings
stay fixed. Rerun segmentation after changing these parameters.

## Automatic QC and progress

Add `--qc-output minimal` or `--qc-output detailed` to the analysis command to
automatically run `make_outputs.py` after analysis succeeds:

```sh
python analyze_foci.py --input-dir /path/to/tiffs --parameters /path/to/parameters.json --qc-output minimal
```

- `off` (default): run analysis as before, including measurement tables, masks,
  and numerical QC; do not automatically generate figures or reports.
- `minimal`: also generate annotated image QC, shared-contrast figures,
  comparison plots, methods, and reports. Skip separate native overlays and
  individual nucleus crops.
- `detailed`: generate all of the above, including `QC/Native_Overlays/` and
  `QC/Nucleus_Crops/`.

Standalone `make_outputs.py` also accepts `--qc-output minimal|detailed` and
defaults to `detailed` to preserve its previous behavior. Minimal reports omit
links to skipped outputs. Existing files from a previous detailed run are left
in place when reusing its output directory; use a fresh folder for a minimal-only
output set. If automatic output generation fails, the completed analysis remains
available for a standalone `make_outputs.py` retry.

The terminal shows each image as it starts, then reports completed/total images
and how many remain after all its channels finish. QC rendering has its own image
counter, followed by status messages for shared-contrast figures, plots, and reports.

## Assign images to groups

Set the top-level `group_names` list in your experiment's `parameters.json`:

```json
"group_names": ["siHELQ", "siCon", "MMC", "PDS"]
```

Matching uses case-insensitive literal substrings of each filename, excluding its
extension, and retains **every match** in configuration order. For example:

| Filename | Matched labels | Treatment group used for plotting |
| --- | --- | --- |
| `field_siHELQ_100MMC_01.tif` | siHELQ, MMC | siHELQ + MMC |
| `field_siHELQ_20PDS_02.tif` | siHELQ, PDS | siHELQ + PDS |

Choose specific names to avoid unintended substring matches: `PDS` also matches
`20PDS`, and configuring both assigns both groups. Names must be nonempty and
unique regardless of case; `AMBIGUOUS` is reserved for unmatched images. Missing
or empty `group_names` retains the legacy parser. Unmatched images remain in the
analysis as `AMBIGUOUS` with a terminal warning.

Measurement and mapping CSVs store all assignments in a `group_names` column as
a JSON list. `treatment_group` shows the combined label (such as `siHELQ + MMC`).
Each image, nucleus, and focus is measured and stored once. Comparison plots
use `treatment_group`: each observation contributes once to its combined
category. For example, `siControl + 100MMC` is one category on the graph.
Image summaries retain one row per image; `number_images_in_treatment` counts
images with the same full
combination of group labels.

For existing analyses, rerun `python make_outputs.py --output-dir /path/to/analysis`
to rebuild the plots and reports using the saved combined treatment groups.
Segmentation and measurements do not need to be repeated.

## Automatically create a run folder

Omit `--output-dir` when you want a new folder each time:

```sh
python Scripts/analyze_foci.py --input-dir /path/to/tiffs --parameters /path/to/experiment_parameters.json
```

The script prints the output path, such as
`Analyses/2026-10-02_120000_tiffs_ab12cd34/`. Use that path as `--output-dir` for
plotting and validation. Automatic names are unique, including runs started in
the same second. An explicit output directory reuses that folder; choose a fresh
name for a different analysis to avoid mixing it with older outputs.

## What each analysis contains

- `Results/`: CSV measurement tables and, if generated, the workbook.
- `QC/`: overlays, nucleus crops, sensitivity tables, and inspection summaries.
- `Plots/`: figures for the selected foci channels.
- `Masks/`: nuclear masks, focus masks, and background models.
- `Methods/parameters.json`: exact copy of the parameter file supplied to analysis.
- `Methods/Effective_Parameters.json`: parameters with command-line channel overrides applied.
- `Methods/Output_Parameters.json`: settings used to generate the plots and reports.
- `Methods/`: image metadata, treatment mapping, applied thresholds, methods text,
  data dictionary, workbook input, version records, and validation reports.
- `Report.md` and `Review.html`: generated summaries and visual review.

Raw TIFFs remain in their input directory. The saved mapping records their source
paths; plotting and validation still need those TIFFs to be available there.
Historical run hashes describe the scripts and parameters at execution time.

Shared-contrast QC figures show up to 12 images per page in a four-column grid.
Every page uses the same display range for that channel across the entire run.
`Review.html` links all pages; the first remains `QC/C2_Shared_Contrast.png`
(for channel 2), followed by `C2_Shared_Contrast_Page_002.png`, and so on.

Input intensity distributions have a separate plot for each combined
`treatment_group` and selected channel. Files are named
`QC/C2_Input_Intensity_Group_001.png` and `.svg` (for channel 2), with group
indices following the treatment order in the image mapping. `Review.html`
includes labeled previews and links. Colors match the treatment comparison
figures. Individual image curves are translucent (alpha 0.25); a thicker solid
curve shows their average distribution, and a dashed vertical line marks their
mean pixel intensity. Both averages give each image equal weight, regardless
of its pixel count, and the legend has only three entries.

Histogram bins and logarithmic axis limits are shared across groups within a
channel. Densities divide each image's bin counts by its total pixel count and
bin widths. Zero pixels contribute to normalization and mean intensity, but
cannot appear on a logarithmic intensity axis; the displayed density integral
can therefore be below one. All-zero images are annotated and contribute zero
to both averages. All scatter plots use the same small marker size, and
treatment-axis labels are rotated 45 degrees and right aligned.

The image-level endpoint panels and focus-positive nuclei figure use one box
per combined `treatment_group`, calculated from saved image summaries. Boxes
show the median and interquartile range (IQR); whiskers reach observations
within 1.5 × IQR of the quartiles. Each valid image appears once per metric as
a faint circular point (`s=8`, alpha 0.25), with reproducible horizontal jitter
of ±0.16 and no text annotation. Separate outlier symbols are hidden because
outliers already appear in the point overlay. Missing and nonfinite values
are omitted per metric while treatment categories remain visible; a
single-image group has a collapsed box and one point. The separate
focus-positive percentage axis remains at 0–110%.

Rerun `python make_outputs.py --output-dir /path/to/existing_analysis` to update
plots and reports from saved measurements without repeating segmentation.
Older combined histogram files remain in place but are not linked from the
updated review report.

If an older version stopped with an "Image size ... is too large" error, rerun
`python make_outputs.py --output-dir /path/to/existing_analysis` with the updated
script. It rebuilds the figures and reports from the saved measurement tables
and masks; there is no need to repeat `analyze_foci.py`.

## Optional workbook

```sh
node Scripts/make_workbook.mjs Analyses/my_experiment
```

The workbook renderer requires `@oai/artifact-tool` in the Node environment.
It still has C2/C3 column-order assumptions, as documented in CHANNELS.md.
It writes the workbook and previews inside the chosen analysis folder.

## Regression checks

```sh
python -m unittest discover -s Scripts/tests -p 'test*.py' -v
```

After copying this folder's contents to your working directory, use `-s tests`.
Tests create their images and outputs in temporary directories.
