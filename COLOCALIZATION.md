# Colocalization measurements

Colocalization is optional and disabled by default. It analyzes one ordered pair
of selected foci channels inside the existing accepted 2D nuclear masks.
Multichannel TIFFs and RGB exports use the same reader, calibration and saved
RGB channel mapping as the rest of the pipeline. Original images and existing
focus measurements are unchanged.

## Object associations

Two accepted foci are associated when their masks share at least
`minimum_overlap_pixels` pixels inside the same nucleus. The default is one.
Objects touching without sharing a pixel are not associated. Every qualifying
pair is retained; there is no forced one-to-one match, distance cutoff, or dilation.

Directional fractions count unique foci: associated A / all A and associated B /
all B. Three A foci overlapping one B focus therefore produce three associated A
foci, one associated B focus and three relationships. Partner counts record zero,
one or multiple partners for each focus. Both all-focus and associated-focus
averages are available.

Every pair records intersection area in pixels and µm², intersection / A area,
intersection / B area, intersection / union (Jaccard), and geometric centroid
distance in µm. X and Y calibration are applied separately. A focus's total
coverage is the union of all its qualifying partners divided by its own area;
it is not an average of pair overlap fractions. Same-channel label masks are
disjoint, so adding qualifying intersection counts does not double-count pixels.

Nearest opposite-channel centroid distance is reported for every focus, including
unassociated foci. Only foci in the same nucleus are considered. A nearest focus
does not automatically qualify as an association.

## Signal measurements

Pearson's coefficient is calculated over **every accepted nuclear pixel**,
independently of focus detection. Raw Pearson uses original unsmoothed intensities.
Corrected Pearson subtracts the existing opening background, using its saved
float32 representation to make previews and independent validation reproducible.
Negative corrected values are retained; display contrast never enters the
calculation. Constant inputs or fewer than two pixels have an undefined
coefficient. No Pearson p-value is calculated.

The directional Manders coefficients measure positive corrected A signal inside
accepted B focus masks / all positive corrected A signal in the nucleus, and
the reverse. Negative corrected values are clipped **only** for these signal
fractions. Nuclear/image Jaccard compares complete accepted masks. These mask and
signal measurements are independent of the minimum-pixel object-association rule.

Focus property comparisons reuse the original area, raw mean intensity,
local-background-corrected mean intensity, and local-corrected integrated
intensity, separately for associated and unassociated A/B foci. Intensities in
RGB exports are the stored color-component values rather than recovered detector
ADU.

## Tables and aggregation

| Table | One row per |
|---|---|
| `Results/Colocalization_Pairs.csv` | Qualifying A–B relationship |
| `Results/Colocalization_Foci.csv` | Selected-channel focus, including unassociated foci |
| `Results/Colocalization_Nuclei.csv` | Accepted nucleus, including zero-focus nuclei |
| `Results/Colocalization_Images.csv` | Input image, including images without accepted nuclei |

Image/nucleus/focus IDs refer to the existing analysis tables and label TIFFs.
Image fractions pool unique-focus counts. Focus means/medians weight each focus
equally, while pair distances weight each relationship equally. Image Pearson
means/medians summarize finite per-nucleus coefficients and record the number
assessed; they are not pooled-pixel correlations. Image Manders coefficients pool
the nuclear signal sums, while image Jaccard pools intersection/union pixel counts.

Treatment boxplots use image observations, preserving equal weight per image.
Coverage/distance ECDFs pool focus or pair observations within each treatment and
describe those distributions. Partner-count bars show the pooled focus fractions
with zero, one, two or three-plus partners; exact counts remain in CSV.

Counts stay zero for empty categories. Empty-denominator fractions, empty-category
averages, missing nearest partners, and invalid Pearson coefficients are blank
in CSV, with explicit status fields. If A exists but B does not, A's associated
fraction is zero, B's fraction is undefined, and A has no nearest B focus.

## Reports and reproducibility

The HTML colocalization section follows the plots and files/reproducibility
section and immediately precedes **Inspect every image**. Six PNG/SVG figures
cover correlations, association fractions, focus properties, coverage/distances,
partner counts, and Manders/Jaccard. Per-image overlays distinguish associated
and unassociated objects; yellow marks qualifying shared pixels. Their legends
are independent of GUI display palettes.

Measurements are calculated during analysis while source pixels/backgrounds are
already available. Reports reuse the saved tables and masks without repeating
segmentation. Effective parameters, algorithm version and source-code hash are
recorded in the run manifest. Independent validation reconciles source pixels,
mask intersections, ID assignments, calibrated distances and table aggregations.
Pipeline consistency is distinct from accuracy against manually annotated
biological ground truth. Measurements describe 2D projected associations, without
inferential testing or conclusions about molecular binding.

The optional spreadsheet workbook remains based on its existing channel-specific
layout; the complete colocalization outputs are the four CSVs and HTML report.

## References

- [MatCol: a tool to measure fluorescence signal colocalisation in biological systems](https://www.nature.com/articles/s41598-017-08786-1)
- [scikit-image: colocalization metrics](https://scikit-image.org/docs/stable/auto_examples/applications/plot_colocalization_metrics.html)
- [scikit-image: Manders coefficient](https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.manders_coloc_coeff)
