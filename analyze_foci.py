#!/usr/bin/env python3
"""Reproducible 2D nuclear foci analysis. Raw TIFFs are read only.

Usage: python analyze_foci.py --input-dir /path/to/tiffs --parameters /path/to/parameters.json
       [--output-dir /path/to/Analyses/my_experiment]
       [--qc-output off|minimal|detailed]
       --nuclear-channel 1 --foci-channels 2 3
Channel numbers are one-based TIFF plane numbers. Each selected foci channel
requires a foci.channel_N threshold entry in the parameters JSON.
make_outputs.py and validate_results.py reuse the saved channel selection.
"""
from pathlib import Path
import argparse, hashlib, json, math, platform, sys, re, tempfile, subprocess
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import scipy
from scipy import ndimage as ndi
import skimage
from skimage import measure, morphology, segmentation
import tifffile
from image_io import read_image, inspect_image, validate_rgb_channel_order, validate_pixel_size


def mad(a):
    a=np.asarray(a)
    return float(1.4826*np.median(np.abs(a-np.median(a)))) if a.size else np.nan


def json_clean(obj):
    if isinstance(obj,dict): return {k:json_clean(v) for k,v in obj.items()}
    if isinstance(obj,(list,tuple)): return [json_clean(v) for v in obj]
    if isinstance(obj,np.ndarray): return obj.tolist()
    if isinstance(obj,np.generic): return obj.item()
    return obj


def validate_group_names(group_names):
    """An empty list preserves the legacy filename parser."""
    if not isinstance(group_names,list) or any(
            not isinstance(g,str) or not g.strip() or g!=g.strip() for g in group_names):
        raise ValueError('group_names must be a list of nonempty strings without surrounding whitespace')
    if len({g.casefold() for g in group_names})!=len(group_names):
        raise ValueError('group_names must not contain duplicates (case-insensitive)')
    if any(g.casefold()=='ambiguous' for g in group_names):
        raise ValueError('group_names cannot include the reserved name AMBIGUOUS')


def metadata_row(path,idx,md,group_names=None):
    name=path.name
    # Preserve the original pilot parser when no group names are configured.
    prefix='63x_TP53KO_488_gH2AX_59453BP1_siCon_'
    tail=path.stem.removeprefix(prefix)
    parts=tail.split('_2026-08-13_')
    group=parts[0] if name.startswith(prefix) and len(parts)==2 else 'AMBIGUOUS'
    if group not in ['20PDS','100MMC','ART558_20PDS']: group='AMBIGUOUS'
    if group_names:
        groups=[g for g in group_names if g.casefold() in path.stem.casefold()]
        group=' + '.join(groups) if groups else 'AMBIGUOUS'
        status='configured filename matches' if groups else 'no configured group matched; manual clarification required'
    else:
        groups=[group] if group!='AMBIGUOUS' else []
        status='unambiguous filename token' if groups else 'manual clarification required'
    legacy=name.startswith(prefix)
    return {'image_id':f'I{idx:02d}','image_name':name,'treatment_group':group,
            'group_names':json.dumps(groups),
            'replicate_id':None,'image_observation_id':f'I{idx:02d}',
            'biological_replicate_status':'none; proof of concept',
            'source_path':str(path.resolve()),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'cell_background':'TP53KO' if legacy else None,'siRNA':'siCon' if legacy else None,
            'objective_from_filename':'63x' if legacy else None,
            'treatment_token':group,'dose_units':'not specified; filename tokens retained verbatim',
            'acquisition_date_from_filename':'2026-08-13' if legacy and len(parts)==2 else None,
            'acquisition_time_from_filename':parts[1].removesuffix('_MAX') if len(parts)==2 else None,
            'projection':'2D RGB export; projection not inferred' if md.get('input_format')=='rgb_export' else '2D MAX projection (filename + CYX TIFF)',
            'width_px':md['shape'][2],'height_px':md['shape'][1],'n_channels':md['shape'][0],
            'z_planes_in_file':1,'timepoints_in_file':1,'bit_depth':md['bit_depth'],'dtype':md['dtype'],
            'input_format':md.get('input_format','channel_stack'),
            'source_axes':md.get('source_axes','CYX'),
            'rgb_channel_order':json.dumps(md.get('rgb_channel_order')),
            'calibration_source':md.get('calibration_source','ImageJ micron unit and TIFF resolution'),
            'saturation_value':md.get('saturation_value',65535),
            'pixel_size_x_um':md['pixel_size_x_um'],'pixel_size_y_um':md['pixel_size_y_um'],
            'stored_z_spacing_um_not_used':md['imagej_metadata'].get('spacing'),
            'group_mapping_status':status}


def nuclear_segmentation(raw,p,sx,sy):
    sm=ndi.gaussian_filter(raw.astype(float),p['gaussian_sigma_px'],mode='reflect')
    baseline=float(np.median(sm));sig=mad(sm)
    threshold=baseline+max(p['threshold_floor_offset_adu'],p['threshold_mad_multiplier']*sig)
    b=morphology.closing(sm>threshold,morphology.disk(p['closing_disk_radius_px']))
    b=ndi.binary_fill_holes(b)
    cc=measure.label(b,connectivity=2)
    areas=np.bincount(cc.ravel()); keep=areas>=p['candidate_min_area_px'];keep[0]=False
    b=keep[cc]
    dist=ndi.gaussian_filter(ndi.distance_transform_edt(b),p['distance_smoothing_sigma_px'])
    peaks=morphology.h_maxima(dist,p['distance_h_maxima_px'])&(dist>p['minimum_seed_distance_to_edge_px'])
    # Every retained connected candidate gets a seed, including very truncated border objects.
    cc=measure.label(b,connectivity=2)
    for r in measure.regionprops(cc):
        if not np.any(peaks[r.slice]&r.image):
            z=np.where(r.image,dist[r.slice],-np.inf)
            yy,xx=np.unravel_index(z.argmax(),z.shape)
            peaks[r.bbox[0]+yy,r.bbox[1]+xx]=True
    seeds=measure.label(peaks,connectivity=2)
    candidates=segmentation.watershed(-dist,seeds,mask=b,connectivity=1)
    usable=np.zeros_like(candidates,dtype=np.uint16)
    audit=[]
    for r in measure.regionprops(candidates,intensity_image=raw):
        area=r.area*sx*sy
        reason=[]
        if area<p['accepted_min_area_um2']:reason.append('area below minimum')
        if area>p['accepted_max_area_um2']:reason.append('area above maximum')
        y0,x0,y1,x1=r.bbox
        edge=p['image_edge_buffer_px']
        if y0<=edge or x0<=edge or y1>=raw.shape[0]-edge or x1>=raw.shape[1]-edge:
            reason.append('touches image border buffer')
        aspect=r.axis_major_length/r.axis_minor_length if r.axis_minor_length else np.nan
        shape=[]
        if r.solidity<p['shape_warning_solidity_below']:shape.append('irregular or potentially merged nucleus')
        if aspect>p['shape_warning_aspect_ratio_above']:shape.append('elongated nucleus')
        rec={'nucleus_label':r.label,'included':not reason,'exclusion_reason':'; '.join(reason),
             'nuclear_area_px':int(r.area),'nuclear_area_um2':area,
             'nuclear_mean_DAPI_adu':r.mean_intensity,'nuclear_median_DAPI_adu':float(np.median(raw[r.slice][r.image])),
             'nuclear_centroid_x_px':r.centroid[1],'nuclear_centroid_y_px':r.centroid[0],
             'nuclear_centroid_x_um':r.centroid[1]*sx,'nuclear_centroid_y_um':r.centroid[0]*sy,
             'nuclear_solidity':r.solidity,'nuclear_eccentricity':r.eccentricity,
             'nuclear_aspect_ratio':aspect,'nuclear_shape_warning':'; '.join(shape),
             'DAPI_threshold_adu':threshold,'DAPI_background_smoothed_adu':baseline,'DAPI_background_smoothed_mad_adu':sig}
        if not reason:usable[r.slice][r.image]=r.label
        audit.append(rec)
    return usable,candidates,audit


def nuclear_noise(raw,nuc):
    v=nuc[:-1,:-1]&nuc[1:,:-1]&nuc[:-1,1:]&nuc[1:,1:]
    w=(raw[:-1,:-1]-raw[1:,:-1]-raw[:-1,1:]+raw[1:,1:])/2
    return max(mad(w[v]),0.5)


def detect_nucleus(raw,bg,corr,nuc,c,p,pixel_area,factor=1.0):
    noise=nuclear_noise(raw,nuc)
    smnoise=noise/(2*np.sqrt(np.pi)*p['gaussian_sigma_px'])
    q=p[f'channel_{c}']
    low=factor*max(q['support_floor_adu'],p['support_noise_multiplier']*smnoise)
    high=factor*max(q['seed_floor_adu'],p['seed_noise_multiplier']*smnoise)
    prominence=factor*max(q['prominence_floor_adu'],p['prominence_noise_multiplier']*smnoise)
    support=(corr>=low)&nuc
    peaks=morphology.h_maxima(corr,prominence)&(corr>=high)&nuc
    seeds=measure.label(peaks,connectivity=p['connectivity'])
    labels=segmentation.watershed(-corr,seeds,mask=support,
                                 connectivity=p['watershed_connectivity'],watershed_line=p['watershed_lines'])
    clipped=np.zeros_like(labels)
    for r in measure.regionprops(labels):
        cr=corr[r.slice]
        cutoff=max(low,p['post_watershed_peak_fraction']*np.max(cr[r.image]))
        b=r.image&(cr>=cutoff)
        cc=measure.label(b,connectivity=p['connectivity'])
        pos=np.unravel_index(np.argmax(np.where(r.image,cr,-np.inf)),cr.shape)
        b=cc==cc[pos]
        clipped[r.slice][b]=r.label
    exclusion=morphology.dilation(clipped>0,morphology.disk(1))
    free=nuc&~exclusion
    out=np.zeros_like(labels)
    audit=[];extra={};j=0
    edge=nuc&~ndi.binary_erosion(nuc)
    for r in measure.regionprops(clipped):
        pad=math.ceil(p['annulus_outer_radius_px'])+1
        y0=max(0,r.bbox[0]-pad);x0=max(0,r.bbox[1]-pad)
        y1=min(raw.shape[0],r.bbox[2]+pad);x1=min(raw.shape[1],r.bbox[3]+pad)
        sl=(slice(y0,y1),slice(x0,x1))
        fm=clipped[sl]==r.label
        dst=ndi.distance_transform_edt(~fm)
        ann=(dst>p['annulus_inner_gap_px'])&(dst<=p['annulus_outer_radius_px'])&free[sl]
        vals=raw[sl][ann];source='annulus'
        if len(vals)<p['annulus_min_pixels']:
            vals=raw[free];source='nucleus nonfocus fallback'
        if len(vals)>=p['annulus_min_pixels']:
            local_bg=float(np.median(vals));local_sig=mad(vals);n_bg=int(len(vals))
        else:
            local_bg=float(np.median(bg[r.slice][r.image]));local_sig=noise;n_bg=0;source='opening model fallback'
        mean=float(raw[r.slice][r.image].mean());net=mean-local_bg
        reasons=[]
        if r.area<p['minimum_area_px']:reasons.append('below minimum area')
        if r.area*pixel_area>p['maximum_area_um2']:reasons.append('above maximum area')
        if net<factor*p['net_mean_noise_multiplier']*noise:reasons.append('insufficient local net mean versus nuclear noise')
        audit.append({'candidate_label':r.label,'area_px':int(r.area),'area_um2':r.area*pixel_area,
                      'mean_raw_adu':mean,'local_background_adu':local_bg,'local_net_mean_adu':net,
                      'included':not reasons,'exclusion_reason':'; '.join(reasons)})
        if reasons:continue
        j+=1;out[r.slice][r.image]=j
        extra[j]={'local_background_median_adu':local_bg,'local_background_mad_adu':local_sig,
                  'local_background_n_pixels':n_bg,'local_background_source':source,
                  'nucleus_noise_raw_adu':noise,'nucleus_noise_smoothed_adu':smnoise,
                  'support_threshold_adu':low,'seed_threshold_adu':high,'prominence_threshold_adu':prominence,
                  'touches_nuclear_boundary':bool(np.any(edge[r.slice]&r.image))}
    return out,extra,audit,{'noise_raw_adu':noise,'noise_smoothed_adu':smnoise,'support_threshold_adu':low,
                           'seed_threshold_adu':high,'prominence_threshold_adu':prominence}


def describe_vals(vals,prefix):
    return {prefix+'mean_adu':float(np.mean(vals)),prefix+'median_adu':float(np.median(vals)),
            prefix+'max_adu':float(np.max(vals)),prefix+'min_adu':float(np.min(vals)),
            prefix+'integrated_adu_px':float(np.sum(vals))}


def focus_measurements(labels,raw,bg,extras,nuc,origin,nuclear_centroid,sx,sy,saturation_value=65535):
    rows=[]
    edge_dist=ndi.distance_transform_edt(nuc,sampling=(sy,sx))
    for r in measure.regionprops(labels):
        rawvals=raw[r.slice][r.image];bgvals=bg[r.slice][r.image]
        x=r.centroid[1]+origin[1];y=r.centroid[0]+origin[0]
        d=math.hypot((x-nuclear_centroid[1])*sx,(y-nuclear_centroid[0])*sy)
        yy=int(round(r.centroid[0]));xx=int(round(r.centroid[1]))
        ed=float(edge_dist[yy,xx])
        per=measure.perimeter_crofton(r.image,directions=4)
        # Axis lengths in physical units via covariance, respecting slight X/Y anisotropy.
        coords=r.coords.astype(float)*np.array([sy,sx])
        ev=np.linalg.eigvalsh(np.cov(coords,rowvar=False,bias=True))
        axminor,axmajor=4*np.sqrt(np.maximum(ev,0))
        e=extras[r.label]
        loc=e['local_background_median_adu'];lsig=e['local_background_mad_adu']
        row={'local_focus_label':r.label,'focus_area_px':int(r.area),'focus_area_um2':r.area*sx*sy,
             'equivalent_diameter_px':r.equivalent_diameter_area,
             'equivalent_diameter_um':math.sqrt(4*r.area*sx*sy/np.pi),
             'major_axis_length_px':r.axis_major_length,'minor_axis_length_px':r.axis_minor_length,
             'major_axis_length_um':axmajor,'minor_axis_length_um':axminor,
             'perimeter_crofton_px':per,'circularity':4*np.pi*r.area/per**2 if per else np.nan,
             'aspect_ratio':r.axis_major_length/r.axis_minor_length if r.axis_minor_length else np.nan,
             'eccentricity':r.eccentricity,'solidity':r.solidity,
             'centroid_x_px':x,'centroid_y_px':y,'centroid_x_um':x*sx,'centroid_y_um':y*sy,
             'distance_from_nuclear_centroid_um':d,'distance_from_nuclear_centroid_px':math.hypot(x-nuclear_centroid[1],y-nuclear_centroid[0]),
             'nearest_nuclear_boundary_distance_um':ed,'relative_radial_position':d/(d+ed) if d+ed else 0,
             **describe_vals(rawvals,'raw_'),**describe_vals(rawvals-bgvals,'opening_corrected_'),
             **describe_vals(rawvals-loc,'local_corrected_'),**e,
             'focus_to_background_ratio':float(rawvals.mean()/loc) if loc>0 else np.nan,
             'local_signal_to_background_ratio':float((rawvals.mean()-loc)/loc) if loc>0 else np.nan,
             'local_contrast_snr':float((rawvals.mean()-loc)/lsig) if lsig>0 else np.nan,
             'saturated_pixels':int(np.count_nonzero(rawvals==saturation_value))}
        rows.append(row)
    return rows


def summarize_cell(f,area_px,area_um2):
    n=len(f)
    result={'foci_count':n,'focus_positive':int(n>0),'foci_density_per_px2':n/area_px,
            'foci_density_per_um2':n/area_um2}
    simple={'focus_area_um2':'focus_area_um2','focus_area_px':'focus_area_px',
            'focus_intensity_adu':'raw_mean_adu','focus_integrated_adu_px':'raw_integrated_adu_px',
            'focus_opening_corrected_mean_adu':'opening_corrected_mean_adu',
            'focus_opening_corrected_integrated_adu_px':'opening_corrected_integrated_adu_px',
            'focus_local_corrected_mean_adu':'local_corrected_mean_adu',
            'focus_local_corrected_integrated_adu_px':'local_corrected_integrated_adu_px'}
    for name,col in simple.items():
        vals=np.asarray([r[col] for r in f])
        for stat in ['mean','median','max']:
            result[f'{stat}_{name}']=float(getattr(np,stat)(vals)) if n else np.nan
    result['maximum_focus_pixel_intensity_adu']=max(r['raw_max_adu'] for r in f) if n else np.nan
    for name,col in [('focus_area_px','focus_area_px'),('focus_area_um2','focus_area_um2'),
                     ('focus_integrated_adu_px','raw_integrated_adu_px'),
                     ('focus_opening_corrected_integrated_adu_px','opening_corrected_integrated_adu_px'),
                     ('focus_local_corrected_integrated_adu_px','local_corrected_integrated_adu_px')]:
        result['total_'+name]=sum(r[col] for r in f)
    result['nuclear_area_fraction_occupied']=result['total_focus_area_px']/area_px
    for col in ['circularity','eccentricity','aspect_ratio','solidity','focus_to_background_ratio','local_contrast_snr','relative_radial_position']:
        result['mean_'+col]=float(np.nanmean([r[col] for r in f])) if n else np.nan
    result['boundary_touching_foci_count']=sum(int(r['touches_nuclear_boundary']) for r in f)
    return result


def save_mask(path,arr,sx,sy):
    tifffile.imwrite(path,arr.astype(np.uint16),imagej=True,resolution=(1/sx,1/sy),
                     metadata={'unit':'micron','axes':'YX'},compression='zlib')


def select_channels(p,nuclear_channel=None,foci_channels=None):
    """Resolve CLI overrides and validate one-based channel configuration."""
    nuclear_channel=p.get('nuclear_channel',1) if nuclear_channel is None else nuclear_channel
    foci_channels=p.get('foci_channels',[2,3]) if foci_channels is None else foci_channels
    if type(nuclear_channel) is not int or nuclear_channel<1:
        raise ValueError('nuclear_channel must be a positive, one-based channel number')
    if not isinstance(foci_channels,list) or not foci_channels:
        raise ValueError('foci_channels must be a nonempty list of one-based channel numbers')
    if any(type(c) is not int or c<1 for c in foci_channels):
        raise ValueError('foci_channels must contain positive, one-based channel numbers')
    if len(set(foci_channels))!=len(foci_channels):
        raise ValueError('foci_channels must not contain duplicate channels')
    for c in foci_channels:
        key=f'channel_{c}'
        if key not in p['foci']:
            raise ValueError(f'Missing foci.{key} in parameters JSON; configure support_floor_adu, '
                             'seed_floor_adu, and prominence_floor_adu for this channel')
        for field in ['support_floor_adu','seed_floor_adu','prominence_floor_adu']:
            value=p['foci'][key].get(field)
            if type(value) not in (int,float) or not math.isfinite(value) or value<0:
                raise ValueError(f'foci.{key}.{field} must be a finite, nonnegative number')
    return nuclear_channel,list(foci_channels)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input-dir',type=Path,required=True,help='Folder containing raw .tif or .tiff images (case-insensitive)')
    ap.add_argument('--output-dir',type=Path,help='Analysis folder; default: a new dated folder under ./Analyses/')
    ap.add_argument('--parameters',type=Path,required=True,help='Parameters for this experiment; start from parameters.template.json')
    ap.add_argument('--nuclear-channel',type=int,help='One-based nuclear channel (default: parameters JSON or 1)')
    ap.add_argument('--foci-channels',type=int,nargs='+',help='One or more one-based foci channels, e.g. 2 3 (default: parameters JSON or 2 3)')
    ap.add_argument('--rgb-channel-order',nargs=3,choices=['red','green','blue'],
                    help='RGB color assigned to channels 1, 2, and 3; e.g. blue green red (required for RGB TIFFs)')
    ap.add_argument('--pixel-size-um',type=float,nargs=2,metavar=('X','Y'),
                    help='Explicit X and Y pixel sizes in µm per pixel; overrides stored calibration')
    ap.add_argument('--qc-output',choices=['off','minimal','detailed'],default='off',
                    help='Automatically generate QC, plots and reports: minimal omits native overlays and nucleus crops; detailed includes them (default: off)')
    args=ap.parse_args();out=args.output_dir
    parameter_bytes=args.parameters.read_bytes()
    p=json.loads(parameter_bytes)
    try:
        nuclear_channel,foci_channels=select_channels(p,args.nuclear_channel,args.foci_channels)
        validate_group_names(p.get('group_names',[]))
        p['rgb_channel_order']=validate_rgb_channel_order(args.rgb_channel_order if args.rgb_channel_order is not None else p.get('rgb_channel_order'))
        p['pixel_size_um']=validate_pixel_size(args.pixel_size_um if args.pixel_size_um is not None else p.get('pixel_size_um'))
    except ValueError as exc:
        ap.error(str(exc))
    p['nuclear_channel']=nuclear_channel;p['foci_channels']=foci_channels;p['qc_output']=args.qc_output
    marker=lambda c:p.get('channels',{}).get(str(c),f'Channel {c}')
    # macOS external drives include ._ AppleDouble companions, not image data.
    files=sorted(path for path in args.input_dir.iterdir()
                 if not path.name.startswith('._') and path.is_file()
                 and path.suffix.lower() in {'.tif','.tiff'})
    if not files:raise ValueError('No TIFF images found')
    # Check nuclear and foci channel availability in every image before writing outputs.
    for path in files:
        try:
            md=inspect_image(path,p)
            for c in [nuclear_channel,*foci_channels]:
                if c>md['shape'][0]:
                    ap.error(f"{path.name}: channel {c} is unavailable; image has {md['shape'][0]} channels")
        except tifffile.TiffFileError as exc:
            ap.error(f'{path.name}: not a readable TIFF image ({exc})')
        except ValueError as exc:
            ap.error(f'{path.name}: {exc}')
    if out is None:
        analyses=Path.cwd()/'Analyses';analyses.mkdir(exist_ok=True)
        label=re.sub(r'[^A-Za-z0-9_-]+','_',args.input_dir.resolve().name).strip('_') or 'analysis'
        prefix=datetime.now().astimezone().strftime('%Y-%m-%d_%H%M%S_')+label+'_'
        out=Path(tempfile.mkdtemp(prefix=prefix,dir=analyses))
    out=out.resolve()
    if out==Path(__file__).resolve().parent:
        ap.error('Choose a separate analysis folder for --output-dir, not the scripts folder')
    for sub in ['Results','QC','Masks','Methods']+[f'Plots/Channel_{c}' for c in foci_channels]:
        (out/sub).mkdir(parents=True,exist_ok=True)
    # Preserve the exact source settings and the settings with CLI overrides.
    (out/'Methods'/'parameters.json').write_bytes(parameter_bytes)
    (out/'Methods'/'Effective_Parameters.json').write_text(json.dumps(p,indent=2))
    print(f'Analysis output: {out}',flush=True)
    # print(f'Analysis: 0/{len(files)} images processed; {len(files)} remaining',flush=True)
    mapping=[];cells=[];foci=[];nucaudit=[];focaudit=[];qc=[];intensity=[];sensitivity=[];metadata={};thresholds=[]
    for idx,path in enumerate(files,1):
        print(f'\nAnalyzing image {idx}/{len(files)}: {path.name}',flush=True)
        arr,md=read_image(path,p);m=metadata_row(path,idx,md,p.get('group_names'));mapping.append(m);metadata[m['image_id']]=md
        saturation_value=md['saturation_value'] if md['saturation_value'] is not None else np.inf
        if m['treatment_group']=='AMBIGUOUS':
            print(f'  Group warning: {m["group_mapping_status"]}',flush=True)
        sx,sy=md['pixel_size_x_um'],md['pixel_size_y_um'];pixel_area=sx*sy
        ids={k:m[k] for k in ['image_id','image_name','treatment_group','group_names','replicate_id']}
        usable,candidates,na=nuclear_segmentation(arr[nuclear_channel-1],p['nuclear'],sx,sy)
        for rec in na:
            # Generic names describe the selected nuclear stain. Retain legacy
            # DAPI names only when the configured marker is actually DAPI.
            aliases={'nuclear_mean_DAPI_adu':'nuclear_stain_mean_adu',
                     'nuclear_median_DAPI_adu':'nuclear_stain_median_adu',
                     'DAPI_threshold_adu':'nuclear_threshold_adu',
                     'DAPI_background_smoothed_adu':'nuclear_background_smoothed_adu',
                     'DAPI_background_smoothed_mad_adu':'nuclear_background_smoothed_mad_adu'}
            for key,generic in aliases.items():
                rec[generic]=rec[key]
                if marker(nuclear_channel).upper()!='DAPI':rec.pop(key)
            rec.update(nuclear_channel=nuclear_channel,nuclear_marker=marker(nuclear_channel))
        accepted=[r for r in na if r['included']]
        excluded=len(na)-len(accepted)
        print(m['image_id'],m['treatment_group'],'---',len(accepted),'usable nuclei;',excluded,'excluded',flush=True)
        save_mask(out/'Masks'/f"{m['image_id']}_nuclei.tif",usable,sx,sy)
        save_mask(out/'Masks'/f"{m['image_id']}_nuclei_all_candidates.tif",candidates,sx,sy)
        for rec in na:nucaudit.append({**ids,'nucleus_id':f"{m['image_id']}_N{rec['nucleus_label']:03d}",**rec})
        cell_index={}
        for rec in accepted:
            cell={**ids,'nucleus_id':f"{m['image_id']}_N{rec['nucleus_label']:03d}",**rec}
            for k in ['included','exclusion_reason']:cell.pop(k)
            cell_index[rec['nucleus_label']]=cell
        outside=~morphology.dilation(candidates>0,morphology.disk(p['qc']['extranuclear_background_exclusion_dilation_px'],decomposition='sequence'))
        for c in range(1,arr.shape[0]+1):
            a=arr[c-1]
            histrow={**ids,'channel':c,'marker':marker(c),'mean_adu':float(a.mean()),'std_adu':float(a.std()),
                     'observed_max_adu':int(a.max()),'observed_max_pixel_pct':float((a==a.max()).mean()*100),
                     'saturated_pixel_pct':float((a==saturation_value).mean()*100)}
            for q in [0,1,10,25,50,75,90,95,99,99.5,99.9,100]:histrow[f'p{q}_adu']=float(np.percentile(a,q))
            intensity.append(histrow)
        for c in foci_channels:
            raw=arr[c-1].astype(float)
            sm=ndi.gaussian_filter(raw,p['foci']['gaussian_sigma_px'],mode='reflect')
            bg=morphology.opening(sm,morphology.disk(p['foci']['background_disk_radius_px'],decomposition=p['foci']['background_disk_decomposition']))
            corr=sm-bg
            output_mask=np.zeros_like(usable)
            tifffile.imwrite(out/'Masks'/f"{m['image_id']}_C{c}_background_model.tif",bg.astype(np.float32),imagej=True,
                             resolution=(1/sx,1/sy),metadata={'unit':'micron','axes':'YX'},compression='zlib')
            global_label=0;imagef=[];audit_start=len(focaudit);sens_counts={x:0 for x in p['foci']['sensitivity_threshold_factors']}
            for nr in measure.regionprops(usable):
                pad=10;y0=max(0,nr.bbox[0]-pad);x0=max(0,nr.bbox[1]-pad)
                y1=min(raw.shape[0],nr.bbox[2]+pad);x1=min(raw.shape[1],nr.bbox[3]+pad)
                sl=(slice(y0,y1),slice(x0,x1));nm=usable[sl]==nr.label
                lab,extra,fa,th=detect_nucleus(raw[sl],bg[sl],corr[sl],nm,c,p['foci'],pixel_area)
                nid=f"{m['image_id']}_N{nr.label:03d}"
                thresholds.append({**ids,'nucleus_id':nid,'channel':c,**th})
                for r in fa:focaudit.append({**ids,'nucleus_id':nid,'channel':c,**r})
                fr=focus_measurements(lab,raw[sl],bg[sl],extra,nm,(y0,x0),nr.centroid,sx,sy,saturation_value)
                for r in fr:
                    loc=r.pop('local_focus_label');global_label+=1
                    output_mask[sl][lab==loc]=global_label
                    r.update({**ids,'nucleus_id':nid,'nucleus_label':nr.label,'channel':c,
                              'marker':marker(c),'focus_label':global_label,
                              'focus_id':f"{m['image_id']}_C{c}_F{global_label:05d}"})
                    foci.append(r);imagef.append(r)
                summary=summarize_cell(fr,nr.area,nr.area*pixel_area)
                summary.update({'nuclear_mean_raw_adu':float(raw[sl][nm].mean()),
                                'nuclear_median_opening_background_adu':float(np.median(bg[sl][nm])),**th})
                for k,v in summary.items():cell_index[nr.label][f'C{c}_{k}']=v
                for factor in p['foci']['sensitivity_threshold_factors']:
                    if factor==1.0:count=len(fr)
                    else:
                        ll,_,_,_=detect_nucleus(raw[sl],bg[sl],corr[sl],nm,c,p['foci'],pixel_area,factor)
                        count=int(ll.max())
                    sens_counts[factor]+=count
                    sensitivity.append({**ids,'nucleus_id':nid,'channel':c,'threshold_factor':factor,'foci_count':count})
            save_mask(out/'Masks'/f"{m['image_id']}_C{c}_foci.tif",output_mask,sx,sy)
            cs=list(cell_index.values());faimage=focaudit[audit_start:]
            counts=np.array([r[f'C{c}_foci_count'] for r in cs])
            ff=pd.DataFrame(imagef)
            median=lambda col:float(ff[col].median()) if len(ff) else np.nan
            qr={**ids,'channel':c,'marker':marker(c),'nuclear_channel':nuclear_channel,'number_usable_nuclei':len(accepted),
                'number_candidate_nuclei':len(na),'number_excluded_nuclei':excluded,'percent_nuclei_excluded':100*excluded/len(na) if na else np.nan,
                'background_median_adu':float(np.median(raw[outside])),'background_mad_adu':mad(raw[outside]),
                'background_std_adu':float(raw[outside].std()),
                'nuclear_background_model_median_adu':float(np.median(bg[usable>0])),
                'signal_to_background_ratio':median('focus_to_background_ratio'),'median_local_contrast_snr':median('local_contrast_snr'),
                'percent_saturated_pixels_image':float((raw==saturation_value).mean()*100),
                'percent_saturated_pixels_usable_nuclei':float((raw[usable>0]==saturation_value).mean()*100),
                'median_focus_intensity_adu':median('raw_mean_adu'),'median_focus_area_um2':median('focus_area_um2'),
                'p95_focus_area_um2':float(ff.focus_area_um2.quantile(.95)) if len(ff) else np.nan,
                'median_foci_per_nucleus':float(np.median(counts)),'number_detected_foci':len(imagef),
                'focus_intensity_p99_to_median':float(ff.raw_mean_adu.quantile(.99)/ff.raw_mean_adu.median()) if len(ff) else np.nan,
                'median_nuclear_area_fraction_occupied':float(np.median([r[f'C{c}_nuclear_area_fraction_occupied'] for r in cs])),
                'nuclei_with_shape_warning':sum(bool(r['nuclear_shape_warning']) for r in accepted),
                'candidate_foci_count':len(faimage),
                'foci_rejected_by_area_count':sum('area' in r['exclusion_reason'] for r in faimage),
                'foci_rejected_by_contrast_count':sum('insufficient' in r['exclusion_reason'] for r in faimage),
                'foci_rejected_too_large_count':sum('above maximum' in r['exclusion_reason'] for r in faimage),
                'threshold_minus20pct_count':sens_counts[.8],'threshold_plus20pct_count':sens_counts[1.2],
                'threshold_sensitivity_max_count_change_pct':max(abs(v-len(imagef)) for v in sens_counts.values())/max(len(imagef),1)*100,
                'manual_inspection_warnings':''}
            qc.append(qr)
            print(' ',c,len(imagef),'foci;',sens_counts,'threshold sensitivity',flush=True)
        cells.extend(cell_index.values())
        if hashlib.sha256(path.read_bytes()).hexdigest()!=m['sha256']:raise RuntimeError('Raw image changed during analysis')
        # print(f'Analysis: {idx}/{len(files)} images processed; {len(files)-idx} remaining',flush=True)
    print('Saving measurement tables and analysis records...',flush=True)
    dfc=pd.DataFrame(cells);dff=pd.DataFrame(foci);dfq=pd.DataFrame(qc)
    # A selected channel can legitimately have no foci or usable nuclei.
    first=['image_id','image_name','treatment_group','group_names','replicate_id','nucleus_id','nucleus_label','channel','marker','focus_id','focus_label']
    if dff.empty:
        dff=pd.DataFrame(columns=first+['focus_area_px','focus_area_um2','relative_radial_position','raw_mean_adu','raw_max_adu','raw_integrated_adu_px',
                                      'focus_to_background_ratio','circularity','eccentricity','aspect_ratio','solidity'])
    if dfc.empty:
        dfc=pd.DataFrame(columns=['image_id','image_name','treatment_group','group_names','replicate_id','nucleus_id','nucleus_label',
                                 'nuclear_area_px','nuclear_area_um2','nuclear_channel','nuclear_marker']+
                         [f'C{c}_{key}' for c in foci_channels for key in summarize_cell([],1,1)])
    for c in foci_channels:
        sub=dfq[dfq.channel==c]
        bm=sub.background_median_adu.median();bs=sub.background_mad_adu.median();mc=sub.median_foci_per_nucleus.median()
        for ix,r in sub.iterrows():
            q=p['qc'];warn=[]
            if r.number_usable_nuclei<q['low_usable_nucleus_count']:warn.append(f'LOW_NUCLEUS_COUNT: {r.number_usable_nuclei} < {q["low_usable_nucleus_count"]}')
            if r.percent_nuclei_excluded>=q['high_excluded_nuclei_percent']:warn.append(f'HIGH_NUCLEAR_EXCLUSION: {r.percent_nuclei_excluded:.1f}% candidates excluded')
            if r.nuclei_with_shape_warning:warn.append(f'IRREGULAR_NUCLEI: {r.nuclei_with_shape_warning} retained nuclei with solidity/aspect warning')
            if r.background_median_adu>bm*q['background_median_fold_above_dataset_median']:warn.append('HIGH_BACKGROUND: extracellular median exceeds dataset rule')
            if r.background_mad_adu>bs*q['background_mad_fold_above_dataset_median']:warn.append('VARIABLE_BACKGROUND: extracellular MAD exceeds dataset rule')
            if r.signal_to_background_ratio<q['low_median_focus_background_ratio']:warn.append(f'LOW_LOCAL_CONTRAST: median focus/background={r.signal_to_background_ratio:.2f}')
            if r.percent_saturated_pixels_image>q['high_saturation_percent']:warn.append('SATURATION: container saturation exceeds limit')
            if r.median_foci_per_nucleus>q['high_median_foci_count']:warn.append(f'HIGH_FOCI_COUNT: median={r.median_foci_per_nucleus:g}')
            if r.median_foci_per_nucleus<q['low_median_foci_count']:warn.append(f'LOW_FOCI_COUNT: median={r.median_foci_per_nucleus:g}')
            if mc and r.median_foci_per_nucleus<mc*q['count_fold_low_vs_dataset_median']:warn.append('LOW_RELATIVE_FOCI_COUNT: <0.25 times dataset median; may be biological')
            if mc and r.median_foci_per_nucleus>mc*q['count_fold_high_vs_dataset_median']:warn.append('HIGH_RELATIVE_FOCI_COUNT: >4 times dataset median; may be biological')
            if r.p95_focus_area_um2>q['large_focus_area_p95_um2']:warn.append(f'LARGE_OBJECTS: area P95={r.p95_focus_area_um2:.2f} um2')
            if r.median_focus_area_um2<q['small_median_focus_area_um2']:warn.append(f'SMALL_OBJECTS: median area={r.median_focus_area_um2:.3f} um2')
            if r.focus_intensity_p99_to_median>q['high_focus_intensity_p99_to_median_ratio']:warn.append(f'INTENSITY_TAIL: focus-mean P99/median={r.focus_intensity_p99_to_median:.2f}')
            if r.median_nuclear_area_fraction_occupied>q['dense_median_occupied_fraction']:warn.append(f'DENSE_FOCI: median occupied nuclear area={100*r.median_nuclear_area_fraction_occupied:.1f}%; inspect splits/merges')
            if r.threshold_sensitivity_max_count_change_pct>q['threshold_sensitivity_percent']:warn.append(f'THRESHOLD_SENSITIVE: counts change up to {r.threshold_sensitivity_max_count_change_pct:.1f}% under uniform +/-20% thresholds')
            if r.foci_rejected_too_large_count/max(r.candidate_foci_count,1)*100>q['large_rejected_fraction_percent']:warn.append('LARGE_REJECTED_OBJECTS: >5% candidate foci exceed maximum area')
            dfq.loc[ix,'manual_inspection_warnings']='; '.join(warn) if warn else 'No numerical QC rule triggered'
    image_rows=[]
    for m in mapping:
        im=m['image_id'];cr=dfc[dfc.image_id==im]
        row={k:m[k] for k in ['image_id','image_name','treatment_group','group_names','replicate_id']}
        row.update({'nuclei_analyzed':len(cr),'pixel_size_x_um':m['pixel_size_x_um'],'pixel_size_y_um':m['pixel_size_y_um'],
                    'nuclear_channel':nuclear_channel,'nuclear_marker':marker(nuclear_channel),
                    'number_images_in_treatment':sum(x['treatment_group']==m['treatment_group'] for x in mapping),
                    'biological_replicates':0,'analysis_unit':'nucleus; nested within single image'})
        for c in foci_channels:
            ff=dff[(dff.image_id==im)&(dff.channel==c)];count=cr[f'C{c}_foci_count'];qrow=dfq[(dfq.image_id==im)&(dfq.channel==c)].iloc[0]
            r={'total_foci':len(ff),'mean_foci_per_nucleus':count.mean(),'median_foci_per_nucleus':count.median(),
               'std_foci_per_nucleus':count.std(ddof=1),'iqr_foci_per_nucleus':count.quantile(.75)-count.quantile(.25),
               'percent_focus_positive_nuclei':100*(count>0).mean(),
               'mean_focus_area_um2':ff.focus_area_um2.mean(),'median_focus_area_um2':ff.focus_area_um2.median(),
               'mean_focus_intensity_adu':ff.raw_mean_adu.mean(),'median_focus_intensity_adu':ff.raw_mean_adu.median(),
               'mean_focus_maximum_intensity_adu':ff.raw_max_adu.mean(),'maximum_focus_intensity_adu':ff.raw_max_adu.max(),
               'mean_integrated_intensity_per_focus_adu_px':ff.raw_integrated_adu_px.mean(),
               'mean_total_focus_intensity_per_nucleus_adu_px':cr[f'C{c}_total_focus_integrated_adu_px'].mean(),
               'mean_total_local_corrected_focus_intensity_per_nucleus_adu_px':cr[f'C{c}_total_focus_local_corrected_integrated_adu_px'].mean(),
               'mean_total_opening_corrected_focus_intensity_per_nucleus_adu_px':cr[f'C{c}_total_focus_opening_corrected_integrated_adu_px'].mean(),
               'mean_foci_density_per_um2':cr[f'C{c}_foci_density_per_um2'].mean(),
               'mean_nuclear_area_fraction_occupied':cr[f'C{c}_nuclear_area_fraction_occupied'].mean(),
               'mean_total_focus_area_per_nucleus_um2':cr[f'C{c}_total_focus_area_um2'].mean(),
               'median_cell_median_focus_area_um2':cr[f'C{c}_median_focus_area_um2'].median(),
               'mean_focus_to_background_ratio':ff.focus_to_background_ratio.mean(),
               'median_focus_to_background_ratio':ff.focus_to_background_ratio.median(),
               'mean_circularity':ff.circularity.mean(),'mean_eccentricity':ff.eccentricity.mean(),
               'mean_aspect_ratio':ff.aspect_ratio.mean(),'mean_solidity':ff.solidity.mean(),
               'qc_flags':qrow.manual_inspection_warnings}
            row.update({f'C{c}_{k}':v for k,v in r.items()})
        row['qc_flags']=' | '.join(f'C{c}: '+str(row[f'C{c}_qc_flags']) for c in foci_channels)
        image_rows.append(row)
    # Put identity columns first in the per-focus table.
    dff=dff[first+[c for c in dff if c not in first]]
    tables={'Image_Summary':pd.DataFrame(image_rows),'Cell_Summary':dfc,'Focus_Data':dff,'QC_Summary':dfq}
    for name,df in tables.items():df.to_csv(out/'Results'/f'{name}.csv',index=False,float_format='%.10g')
    dfq.to_csv(out/'QC'/'QC_Summary.csv',index=False,float_format='%.10g')
    pd.DataFrame(mapping).to_csv(out/'Methods'/'Treatment_Group_Mapping.csv',index=False)
    pd.DataFrame(intensity).to_csv(out/'QC'/'Input_Intensity_Distributions.csv',index=False,float_format='%.10g')
    pd.DataFrame(nucaudit).to_csv(out/'QC'/'Nucleus_Inclusion_Audit.csv',index=False,float_format='%.10g')
    pd.DataFrame(focaudit).to_csv(out/'QC'/'Focus_Candidate_Audit.csv',index=False,float_format='%.10g')
    pd.DataFrame(sensitivity).to_csv(out/'QC'/'Threshold_Sensitivity.csv',index=False)
    pd.DataFrame(thresholds).to_csv(out/'Methods'/'Applied_Thresholds.csv',index=False,float_format='%.10g')
    (out/'Methods'/'Image_Metadata.json').write_text(json.dumps(json_clean(metadata),indent=2))
    run={'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'scipy':scipy.__version__,
         'skimage':skimage.__version__,'pandas':pd.__version__,'tifffile':tifffile.__version__,
         'run_utc':datetime.now(timezone.utc).isoformat(),'input_images':len(files),'nuclei_analyzed':len(dfc),
         'nuclear_channel':nuclear_channel,'foci_channels':foci_channels,'qc_output':args.qc_output,
         'rgb_channel_order':p.get('rgb_channel_order'),
         'pixel_size_um':p.get('pixel_size_um'),
         'image_reader_sha256':hashlib.sha256(Path(__file__).with_name('image_io.py').read_bytes()).hexdigest(),
         **{f'channel_{c}_foci':int((dff.channel==c).sum()) for c in foci_channels},
         'raw_sha256_rechecked_unchanged':True,'parameters_sha256':hashlib.sha256(parameter_bytes).hexdigest(),
         'output_directory':str(out),'input_directory':str(args.input_dir.resolve()),
         'parameters_source':str(args.parameters.resolve()),'pipeline_source':str(Path(__file__).resolve()),
         'pipeline_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out/'Methods'/'Run_Manifest.json').write_text(json.dumps(run,indent=2))
    (out/'Methods'/'Effective_Parameters.json').write_text(json.dumps(p,indent=2))
    print(json.dumps(run,indent=2),flush=True)
    print(f'Analysis complete: {len(files)}/{len(files)} images processed; 0 remaining',flush=True)
    if args.qc_output!='off':
        print(f'Generating {args.qc_output} QC outputs...',flush=True)
        try:
            subprocess.run([sys.executable,str(Path(__file__).resolve().with_name('make_outputs.py')),
                            '--output-dir',str(out),'--qc-output',args.qc_output],check=True)
        except subprocess.CalledProcessError as exc:
            print(f'Output generation failed; completed analysis is saved in {out}. '
                  'Rerun make_outputs.py for this folder to retry.',file=sys.stderr,flush=True)
            raise SystemExit(exc.returncode)


if __name__=='__main__':main()
