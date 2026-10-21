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
- Optional many-to-many colocalization analysis of two foci channels, including
  nuclear Pearson coefficients, overlap, distances, partner counts and signal fractions.
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
├── image_review.py
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
`channel_config.py`, `image_io.py`, `image_review.py`, `progress_reporting.py`,
`colocalization.py`, `colocalization_outputs.py`, and `colocalization_validation.py` together. You can copy this folder's contents into your
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
- Windows setup below targets Windows 10/11 with x86-64 Python 3.12.
- A calibrated multichannel ImageJ/OME TIFF with `CYX` axes, or a calibrated 2D RGB
  TIFF with an explicit color-to-channel mapping, is required.
- If physical pixel size was lost during export, supply the acquisition
  calibration with `--pixel-size-um X Y` in µm per pixel. See
  [CHANNELS.md](CHANNELS.md) for calibration sources and overrides.
- Node.js is optional and is only needed for workbook generation.

### Windows: double-click setup and launch

1. Install **Python 3.12, Windows installer (64-bit)** from
   [python.org](https://www.python.org/downloads/windows/). Include the Python
   launcher, or select **Add Python to PATH** during installation.
2. Download or copy the project folder to your Windows computer. If downloading
   a ZIP, extract it first. Keep the scripts, `requirements.txt`, and
   `parameters.template.json` together. Python environments such as `.venv`
   and `.venv-gui` must be created on Windows rather than copied from a Mac.
3. Double-click **`setup_windows.bat`**. It creates `.venv-gui` and installs
   the pinned analysis and GUI dependencies. Internet access is required for
   setup; no terminal activation or PowerShell policy change is needed.
4. Once setup finishes, double-click **`launch_gui.bat`** whenever you want to
   open scikitORION. The launcher works with spaces and Unicode in folder names.
   Use the GUI's Browse buttons to select your images, parameters, and output.

If setup finds an incompatible `.venv-gui` folder, rename it and rerun setup.
If installation or launch fails, the window stays open with the error message.
Rerun `setup_windows.bat` after updating the project's dependencies. The
environment belongs to its current folder; rerun setup if you move the project.

To transfer the current project from a Mac, double-click
**`package_windows.command`**. It rebuilds the clean Windows ZIP and reveals it
in Finder. You can also run it from Terminal:

```sh
./package_windows.command
```

Or run the underlying packager directly from the project folder:

```sh
python3 package_windows.py
```

Copy `dist/scikitORION-windows.zip` to Windows, extract it, then run setup and
the launcher as above. The bundle includes the current GUI, progress module,
fonts, icons, analysis scripts, configuration template, and tests. Rebuild it
after updating the project. Local Python environments (`.venv`, `.venv-gui`),
Git history, caches, and `Analyses` are excluded. Keep the `assets` folder in
the bundle; it supplies the GUI fonts and icons. If you make a ZIP manually,
exclude those local environments too, since a copied Mac `.venv-gui` prevents
Windows setup from creating its environment. Install Python 3.12 (64-bit) and
allow internet access for the initial setup on the destination machine.

For command-line analysis in Windows PowerShell, use the same environment
directly, with quoted paths:

```powershell
.\.venv-gui\Scripts\python.exe -X utf8 .\analyze_foci.py --input-dir "C:\Data\My TIFFs" --parameters "C:\Data\experiment.json" --output-dir "C:\Data\ORION Results" --qc-output minimal
.\.venv-gui\Scripts\python.exe -X utf8 .\validate_results.py --output-dir "C:\Data\ORION Results"
```

`--qc-output minimal` also generates plots and reports. JSON configuration,
Markdown methods, and HTML reports use UTF-8 on every platform, preserving
scientific symbols and non-English labels. GUI background stages also use UTF-8.

### Manual setup

Install the Python dependencies in a new Python 3.12 virtual environment:

```sh
python3.12 -m venv .venv
. .venv/bin/activate       # Windows PowerShell: .venv\Scripts\Activate.ps1
python --version          # Validated with Python 3.12
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For manual Windows setup, run `py -3.12 -m venv .venv` and then
`.\.venv\Scripts\python.exe -m pip install -r requirements.txt`.
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

`imagecodecs` provides decoding for LZW-compressed TIFFs and is included in
`requirements.txt`, which installs both analysis and GUI dependencies. If an existing environment reports
"COMPRESSION.LZW requires the 'imagecodecs' package", reinstall the updated
requirements in that environment and restart the GUI:

```sh
.venv-gui/bin/python -m pip install -r requirements.txt
.venv-gui/bin/python orion_gui.py
```

The compression format is independent of the filename's letter case. No TIFF
conversion or renaming is needed. New runs record the decoder version in
`Methods/Run_Manifest.json` and the generated `Methods/requirements.txt`.

## Desktop GUI

The Qt desktop application provides a resizable image workspace and a
scrolling setup sidebar. On Windows, use the double-click setup and launcher
above. On macOS/Linux, use Python 3.12 and a separate environment:

```sh
python3.12 -m venv .venv-gui
.venv-gui/bin/python -m pip install -r requirements.txt
.venv-gui/bin/python orion_gui.py
```

On macOS, after this setup, double-click `launch_gui.command` to open the GUI.
On Windows, double-click `launch_gui.bat` after running `setup_windows.bat`.
You can also start with `--input-dir`, `--output-dir`, and `--parameters` options.

The approved **Orbital Diamond** icon appears in the macOS Dock and Windows
taskbar while the GUI is running. Icon resources are bundled in `assets/icons`;
keep the `assets` folder with the application scripts when copying them. Restart
an already running GUI to see an updated icon.

The **Studio** workspace uses the approved Art Deco design: obsidian surfaces,
gold geometric frames, Marcellus headings, and Josefin Sans controls. Fonts and
their licenses are bundled for offline use. Shared presentation tokens and Qt
components live in `gui_theme.py`; the compact panel spacing is 12 px. The image
viewer keeps its own colors and calibrated pixels. The setup sidebar and lower
inspection dock scroll independently so expanded settings and color controls
remain usable in smaller desktop windows.

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
   **Map RGB TIFF colors to channels** is an explicit toggle, off by default.
   Its color selectors appear only while enabled. Loading or running a batch
   enables mapping if any input is an RGB TIFF export; batches containing only
   channel stacks turn it off. Saved color choices are remembered, but no
   color assignment is inferred from the nuclear/foci roles. For RGB inputs,
   choose a color for Channels 1, 2, and 3, using red, green, and
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
   saving output files. Use the **Raw image**, **Nuclei**, **Foci**, and **Overlay**
   tabs above the viewer to switch views while retaining zoom and pan;
   toggle channels and outlines, zoom with the mouse wheel or buttons, and drag
   to pan. Click a nucleus or use the inspector selector to see its area,
   exclusions, foci counts, and applied thresholds. Expand **Channel colors**
   below the viewer to choose independent signal/mask and outline pseudocolors
   for each channel. Color changes redraw immediately, retain computed masks,
   and stay with the channel when its nuclear/foci role changes. **Save JSON…**
   stores the colors under `gui_display`; **Load JSON** restores them. **Reset
   colors** restores the default palette. Display colors and contrast affect
   only the GUI view; exported reports use their existing palette, and detection
   uses the original pixels. Analysis parameter edits invalidate old preview masks.
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

The bottom indicators show **Overall** progress and the **current stage**:
generating masks, making outputs, or validating results. Percentages advance
when work finishes, including nuclei, channel masks, figures, and saved reports;
they are not estimates of elapsed or remaining time. Overall progress weights
analysis/masks at 45%, outputs at 50%, and validation at 5%. With QC output off,
the weights become 90% analysis and 10% validation. Overall reaches 100% only
after all required processes succeed. Stopping or failing retains the last
percentages; a new run resets them. A single-image preview shows **—** for
overall progress and an indeterminate preview indicator while it runs.

For programmatic monitoring, the analysis, output, and validation scripts accept
`--progress-json`. It adds flushed lines starting with `ORION_PROGRESS ` followed
by JSON containing `stage`, `completed`, `total`, and `detail`. Normal command-line
logging is unchanged when this option is absent. Progress is not written into
scientific result tables or parameter files.

This first desktop version launches from Python; it is not yet packaged as a
standalone installer. The original command-line workflow remains available.

## Optional colocalization analysis

Check **Analyze colocalization** beneath the foci-channel selections to reveal
Marker A, Marker B, and Minimum shared pixels. The default pair is Channels 2/3
and the default minimum is one shared pixel. Both distinct channels must already
be selected for foci analysis. Settings are saved by **Save JSON…** and restored
by **Load JSON**. The same settings work through the command-line parameter file:

```json
"colocalization": {
  "enabled": true,
  "channels": [2, 3],
  "minimum_overlap_pixels": 1
}
```

Colocalization uses the existing accepted nuclei and focus masks. All qualifying
relationships are retained: three A foci overlapping one B focus count as three
associated A foci, one associated B focus, and three A–B relationships. No new
segmentation, mask dilation, registration, or statistical testing is performed.

**Preview image** includes colocalization measurements in the results and nucleus
inspector. The optional **Colocalization** view shows associated objects brightly,
unassociated objects dimly, and qualifying shared pixels in yellow. Its colors
encode association status; other views retain the editable channel palettes.

New runs save `Results/Colocalization_Pairs.csv`, `Colocalization_Foci.csv`,
`Colocalization_Nuclei.csv`, and `Colocalization_Images.csv`, even when QC output
is off. With reports enabled, a dedicated section appears after the plots and
files/reproducibility links, immediately before **Inspect every image**.
It includes treatment figures, per-image overlays, definitions and CSV links.
Existing configurations without the colocalization block remain disabled;
existing results remain loadable. To add measurements to an older analysis,
run a new analysis with colocalization enabled in a new output directory.

See [COLOCALIZATION.md](COLOCALIZATION.md) for definitions, aggregation,
missing values, scientific references, and validation scope.

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
- `Experiment_Summary.png` and `.svg`: a single experiment overview for eLab
  entries or sharing, saved beside `Review.html`.
- `Representative_Images.png` and `.svg`: up to two representative full-field
  images per treatment, shown with separate nuclear/foci channel rows.

Representative selection compares each image to its combined treatment's means
for the four summary endpoints across all selected foci channels. Distances are
standardized by sample standard deviation so different measurement units have
equal weight. Metrics with fewer than two finite values or no variation are
ignored. Candidates with fuller metric coverage rank first, followed by smaller
root-mean-square distance; ties use original filename and internal ID. Images
with no usable nuclei or flagged outliers are ineligible. Fewer than two eligible
images produces labeled empty slots, never duplicated images. Partial metric
coverage is labeled, and all-constant measurements use the filename tie-break.
The audit in `Methods/Representative_Image_Selection.csv` records every image's
score, coverage, eligibility, rank, and selection, together with group means/scales.

The representative grid adapts to the number of treatments: 10 groups use five
columns by two rows, and nine groups use three by three. Each treatment contains
two image columns and rows for the unique nuclear/selected foci channels. Source
aspect ratios and calibrated scale bars are retained. A shared dataset-wide
contrast range per channel matches the shared-contrast QC; its limits are saved
in `QC/Representative_Display_Ranges.csv`. Display downsampling leaves measured
pixels unchanged.

`QC/Image_Outlier_Review.csv` assesses the same endpoints within each treatment.
Values outside Q1 − 1.5 × IQR or Q3 + 1.5 × IQR are flagged only when at least five
finite images and positive IQR are available. Missing values, small groups, and
zero IQR are recorded as unassessed. Flags request review and affect representative
selection only: every image remains in all measurements, means, and summary plots.
The HTML shows a flagged-image count, a collapsible reasons table with image
links, and flags on image cards and treatment inspection sections. Existing QC
warnings remain visible.

Original filenames are used for displayed image names and progress messages.
Internal `image_id` values, nucleus/focus IDs, CSV keys, and generated mask paths
remain unchanged, so existing analyses and links remain compatible. Older mappings
without `image_name` use the source-path basename. Both new figures and review
audits are regenerated in minimal and detailed modes for the selected channels;
no new CLI options or parameters are required.

The experiment summary is generated in both minimal and detailed QC modes. Each
selected foci channel has four image-level endpoint panels (mean foci/nucleus,
mean per-focus area, mean per-focus raw intensity, and focus-positive percentage)
and three ECDF panels (nucleus foci counts, individual focus areas, and individual
focus mean raw intensities). Endpoint boxes and faint points use the saved image
summaries, giving each image one observation per metric, grouped by its full
combined `treatment_group`. Colors match the detailed graphs.

Exactly two selected foci channels also produce a nucleus scatter: the first
selected channel is X and the second is Y, with one point per usable nucleus,
including zero counts, colored by treatment. Channel subsets control the summary
contents; one or more than two selected channels omit the scatter. No fitted line
or correlation statistics are added. Missing and nonfinite values are omitted
per panel, and panels without valid observations display "No measurements".

The PNG is exported at 300 dpi; the SVG keeps text and lines editable, with point
layers rasterized. Links appear near the top of `Review.html`. Regenerating outputs
overwrites both summary files without repeating segmentation; `--qc-output off`
continues to skip figure/report generation.

In `Review.html`, summary graphs appear before image inspection. The final
"Inspect every image" section groups images by `treatment_group`, with each
group collapsed by default. Expand a group to see its annotated images and,
for detailed QC, individual nucleus galleries.

Raw TIFFs remain in their input directory. The saved mapping records their source
paths; plotting and validation still need those TIFFs to be available there.
Historical run hashes describe the scripts and parameters at execution time.

Treatment graphs use a 32-color [Glasbey palette](https://colorcet.holoviz.org/user_guide/Categorical.html)
chosen for white backgrounds. Colors follow the treatment order in the saved
image mapping and are consistent across boxes, points, distribution curves,
channels, and input-intensity plots. The first 32 groups each receive a different
color; colors cycle only for analyses with more than 32 groups.

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
python -m unittest discover -s tests -p 'test*.py' -v
```

Run this from the project folder. Tests create their images and outputs in
temporary directories. `.github/workflows/windows.yml` runs installation,
the launcher, and the regression suite on Windows with Python 3.12, including
a checkout path containing spaces and Unicode. Windows checks use Qt's native
Windows backend so the font and layout tests exercise the desktop application.
