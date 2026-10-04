# scikitORION

Reproducible, calibrated 2D nuclear-foci analysis for multichannel ImageJ TIFF
images. scikitORION segments nuclei, detects foci in one or more selected
channels, records measurement provenance, produces quality-control material and
reports, and validates the generated result tables.

The pipeline is intentionally script-based: it can be copied into an analysis
workspace without package installation or a project-specific runtime. Raw
images are read-only inputs and are never copied into an analysis output
directory.

## Features

- Calibration-aware processing of 2D `CYX` ImageJ TIFF files.
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

Keep `analyze_foci.py`, `make_outputs.py`, `validate_results.py`, and
`channel_config.py` together. You can copy this folder's contents into your
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

## Requirements and installation

- Python 3.12 is the validated runtime for the pinned dependency set.
- A calibrated multichannel ImageJ TIFF with `CYX` axes is required.
- Node.js is optional and is only needed for workbook generation.

Install the Python dependencies in a virtual environment:

```sh
python -m venv .venv
. .venv/bin/activate       # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The pinned versions in `requirements.txt` are deliberately recorded with each
analysis output. Review and validate any dependency changes before using them
for a production comparison.

The last two commands reuse the analysis's saved channel selections. See
[CHANNELS.md](CHANNELS.md) for overrides, marker labels, and selecting subsets.

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

| Filename | Assigned groups |
| --- | --- |
| `field_siHELQ_100MMC_01.tif` | siHELQ, MMC |
| `field_siHELQ_20PDS_02.tif` | siHELQ, PDS |

Choose specific names to avoid unintended substring matches: `PDS` also matches
`20PDS`, and configuring both assigns both groups. Names must be nonempty and
unique regardless of case; `AMBIGUOUS` is reserved for unmatched images. Missing
or empty `group_names` retains the legacy parser. Unmatched images remain in the
analysis as `AMBIGUOUS` with a terminal warning.

Measurement and mapping CSVs store all assignments in a `group_names` column as
a JSON list. `treatment_group` shows the combined label (such as `siHELQ + MMC`).
Each image, nucleus, and focus is measured and stored once. Comparison plots
include each observation in every assigned group, so groups may overlap and their
counts must not be summed as independent observations. Image summaries retain
one row per image; `number_images_in_treatment` counts images with the same full
combination of group labels.

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
