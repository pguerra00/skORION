#!/usr/bin/env python3
"""Independently reconcile source pixels, label masks, and exported tables."""
from pathlib import Path
import argparse, json, hashlib, math
import numpy as np
import pandas as pd
import tifffile
from channel_config import add_channel_arguments, resolve_channels, check_channel_inputs


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output-dir',type=Path,required=True,help='Analysis folder containing Results, Masks, and Methods')
    add_channel_arguments(ap)
    args=ap.parse_args();out=args.output_dir
    image=pd.read_csv(out/'Results'/'Image_Summary.csv');cell=pd.read_csv(out/'Results'/'Cell_Summary.csv');focus=pd.read_csv(out/'Results'/'Focus_Data.csv')
    mapping=pd.read_csv(out/'Methods'/'Treatment_Group_Mapping.csv');qc=pd.read_csv(out/'Results'/'QC_Summary.csv')
    try:
        p=resolve_channels(out,args.nuclear_channel,args.foci_channels)
        check_channel_inputs(out,p,mapping,image,cell,qc)
    except ValueError as exc:
        ap.error(str(exc))
    nuclear_channel=p['nuclear_channel'];foci_channels=p['foci_channels']
    focus=focus[focus.channel.isin(foci_channels)]
    checks=[]
    def ck(condition,description):
        checks.append({'check':description,'passed':bool(condition)})
        if not condition:raise AssertionError(description)
    ck(not image.image_id.duplicated().any() and not mapping.image_id.duplicated().any()
       and set(image.image_id)==set(mapping.image_id),'Every mapped image has exactly one image summary')
    ck(not cell.nucleus_id.duplicated().any(),'Nucleus IDs are globally unique')
    ck(not focus.focus_id.duplicated().any(),'Focus IDs are globally unique')
    ck(set(focus.nucleus_id)<=set(cell.nucleus_id),'Every focus maps to an included nucleus')
    ck((focus.replicate_id.isna()).all() and (cell.replicate_id.isna()).all(),'Replicate IDs remain missing; no biological replicates invented')
    numeric=focus.select_dtypes(include='number').drop(columns=['replicate_id'],errors='ignore')
    optional=['aspect_ratio','focus_to_background_ratio','local_signal_to_background_ratio','local_contrast_snr']
    ck(np.isfinite(numeric.drop(columns=optional,errors='ignore')).all().all()
       and not np.isinf(numeric).any().any(),'Numeric focus measurements are finite; undefined ratios may be missing')
    if len(focus):
        ck(((focus.relative_radial_position>=0)&(focus.relative_radial_position<=1)).all(),'Normalized nuclear positions lie in [0,1]')
    for _,m in mapping.iterrows():
        im=m.image_id;raw=tifffile.imread(m.source_path);nu=tifffile.imread(out/'Masks'/f'{im}_nuclei.tif');cs=cell[cell.image_id==im]
        ck(hashlib.sha256(Path(m.source_path).read_bytes()).hexdigest()==m.sha256,f'{im}: raw SHA256 unchanged')
        ck(nu.shape==raw[nuclear_channel-1].shape,f'{im}: nuclear mask dimensions match channel {nuclear_channel}')
        ck(set(np.unique(nu))-{0}==set(cs.nucleus_label),f'{im}: nuclear label IDs match table')
        areas=np.bincount(nu.ravel())
        for _,r in cs.iterrows():
            ck(areas[int(r.nucleus_label)]==r.nuclear_area_px,f'{r.nucleus_id}: nuclear pixel area reconciles')
            values=raw[nuclear_channel-1][nu==int(r.nucleus_label)]
            mean_key='nuclear_stain_mean_adu' if 'nuclear_stain_mean_adu' in cs else 'nuclear_mean_DAPI_adu'
            median_key='nuclear_stain_median_adu' if 'nuclear_stain_median_adu' in cs else 'nuclear_median_DAPI_adu'
            ck(np.isclose(values.mean(),r[mean_key],rtol=1e-9),f'{r.nucleus_id}: nuclear mean reconciles to channel {nuclear_channel}')
            ck(np.isclose(np.median(values),r[median_key],rtol=1e-9),f'{r.nucleus_id}: nuclear median reconciles to channel {nuclear_channel}')
        for c in foci_channels:
            mask=tifffile.imread(out/'Masks'/f'{im}_C{c}_foci.tif');ff=focus[(focus.image_id==im)&(focus.channel==c)]
            ck(mask.shape==raw[c-1].shape,f'{im} C{c}: mask has original image dimensions')
            ck(set(np.unique(mask))-{0}==set(ff.focus_label),f'{im} C{c}: focus labels exactly match exported rows')
            ck(np.all(nu[mask>0]>0),f'{im} C{c}: all counted focus pixels belong to usable nuclei')
            pix=np.bincount(mask.ravel());sums=np.bincount(mask.ravel(),weights=raw[c-1].ravel())
            means=sums[ff.focus_label.to_numpy(int)]/pix[ff.focus_label.to_numpy(int)]
            focus_areas=ff['focus_area_px'].to_numpy(float) if 'focus_area_px' in ff else np.array([],dtype=float)
            ck(np.array_equal(pix[ff.focus_label.to_numpy(int)],focus_areas),f'{im} C{c}: every focus area independently recomputed')
            ck(np.allclose(sums[ff.focus_label.to_numpy(int)],ff.raw_integrated_adu_px.to_numpy(float),rtol=1e-9,atol=1e-6),f'{im} C{c}: every raw integrated intensity independently recomputed')
            ck(np.allclose(means,ff.raw_mean_adu.to_numpy(float),rtol=1e-9),f'{im} C{c}: every raw focus mean independently recomputed')
            ck(np.allclose(ff.focus_area_um2.to_numpy(float),focus_areas*m.pixel_size_x_um*m.pixel_size_y_um,rtol=1e-9),f'{im} C{c}: physical focus areas reconcile to calibration')
            # Check parent assignment for every mask object, not just the centroid.
            for nl,g in ff.groupby('nucleus_label'):
                selected=np.isin(mask,g.focus_label.to_numpy(int))
                ck(np.all(nu[selected]==nl),f'{im} C{c} N{nl:03d}: focus-to-nucleus pixel assignment reconciles')
            for _,r in cs.iterrows():
                f=ff[ff.nucleus_id==r.nucleus_id]
                ck(len(f)==r[f'C{c}_foci_count'],f'{r.nucleus_id} C{c}: count rolls up from focus rows')
                ck(math.isclose(float(f.raw_integrated_adu_px.sum()),r[f'C{c}_total_focus_integrated_adu_px'],rel_tol=1e-9,abs_tol=1e-6),f'{r.nucleus_id} C{c}: intensity total rolls up')
                area=f.focus_area_px.sum() if len(f) else 0
                ck(math.isclose(float(area/r.nuclear_area_px),r[f'C{c}_nuclear_area_fraction_occupied'],rel_tol=1e-9,abs_tol=1e-10),f'{r.nucleus_id} C{c}: nuclear area fraction rolls up')
                if not len(f):
                    ck(r[f'C{c}_total_focus_integrated_adu_px']==0 and pd.isna(r[f'C{c}_mean_focus_intensity_adu']),f'{r.nucleus_id} C{c}: zero-focus missing-value semantics are preserved')
            ir=image[image.image_id==im].iloc[0]
            ck(ir[f'C{c}_total_foci']==len(ff)==cs[f'C{c}_foci_count'].sum(),f'{im} C{c}: image total reconciles to nuclei and foci')
            actual=ir[f'C{c}_mean_foci_per_nucleus'];expected=cs[f'C{c}_foci_count'].mean()
            ck((pd.isna(actual) and pd.isna(expected)) or math.isclose(actual,expected,rel_tol=1e-9),f'{im} C{c}: image mean includes zero-focus nuclei')
    counts={'images':len(image),'nuclei':len(cell),**{f'channel_{c}_foci':int((focus.channel==c).sum()) for c in foci_channels}}
    report={'status':'passed','counts':counts,'checks_passed':len(checks),'checks':checks,
            'nuclear_channel':nuclear_channel,'foci_channels':foci_channels,
            'validation_scope':'Independent pixel/mask/table reconciliation for the selected channels; not validation against human focus ground truth.'}
    (out/'Methods'/'Validation_Report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'status':'passed','counts':counts,'checks_passed':len(checks)},indent=2))


if __name__=='__main__':main()
