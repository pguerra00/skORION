#!/usr/bin/env python3
"""Build QC, descriptive figures, data dictionary and an HTML review report."""
import os, tempfile
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'foci-matplotlib-cache'))
import argparse, json, html, math, textwrap, re
import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi
from skimage import segmentation, measure
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from channel_config import add_channel_arguments, resolve_channels, check_channel_inputs, marker_name, selected_columns
from image_io import read_channels

GROUPS=['20PDS','100MMC','ART558_20PDS']
LABELS=['20PDS','100MMC','ART558 + 20PDS']
COLORS=['#0072B2','#D55E00','#009E73']
MARKERS={'I01':'s','I02':'o','I03':'^'}
STYLE={'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,'axes.labelsize':10,
       'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','savefig.facecolor':'white'}
plt.rcParams.update(STYLE)
SHARED_CONTRAST_PAGE_SIZE=12
SHARED_CONTRAST_COLUMNS=4


def norm(raw,limits):
    lo,hi=limits
    return np.clip((raw.astype(float)-lo)/max(hi-lo,1),0,1)


def label_rgb(labels):
    ids=np.arange(int(labels.max())+1)
    hue=(ids*.61803398875)%1
    import matplotlib.colors as mc
    rgb=mc.hsv_to_rgb(np.stack([hue,np.full_like(hue,.70),np.full_like(hue,.95)],axis=1))
    rgb[0]=0
    return rgb[labels]


def overlay(raw,limits,labels,nuc=None,excluded=None):
    rgb=np.repeat(norm(raw,limits)[...,None],3,axis=2)
    if nuc is not None:rgb[segmentation.find_boundaries(nuc,mode='inner')]=[.25,.85,1]
    if excluded is not None:rgb[segmentation.find_boundaries(excluded,mode='inner')]=[1,.3,.4]
    rgb[segmentation.find_boundaries(labels,mode='inner')]=[1,.85,.15]
    return rgb


def save_rgb(path,rgb):
    Image.fromarray(np.uint8(np.clip(rgb,0,1)*255)).save(path)


def scale_bar(ax,sx,width=20):
    lim=ax.get_xlim();ylim=ax.get_ylim()
    length=width/sx;x=lim[0]+.055*(lim[1]-lim[0]);y=ylim[0]-.05*(ylim[0]-ylim[1])
    ax.plot([x,x+length],[y,y],color='white',lw=3)
    ax.text(x,y-.02*abs(ylim[0]-ylim[1]),f'{width:g} μm',color='white',fontsize=8)


def figsave(fig,folder,name):
    fig.savefig(folder/f'{name}.png',dpi=300,bbox_inches='tight')
    fig.savefig(folder/f'{name}.svg',bbox_inches='tight')
    plt.close(fig)


def shared_contrast_pages(mapping,c):
    """Keep the original filename for page one and number additional pages."""
    for start in range(0,len(mapping),SHARED_CONTRAST_PAGE_SIZE):
        page=start//SHARED_CONTRAST_PAGE_SIZE+1
        suffix='' if page==1 else f'_Page_{page:03d}'
        yield f'C{c}_Shared_Contrast{suffix}.png',mapping.iloc[start:start+SHARED_CONTRAST_PAGE_SIZE]


def make_shared_contrast(qdir,mapping,p):
    """Render bounded grids with one dataset-wide display range per channel."""
    if mapping.empty:return
    for c in p['foci_channels']:
        # Copy only sampled pixels so the views cannot retain full TIFF stacks.
        samples=[]
        for path in mapping.source_path:
            raw=read_channels(path,p)
            samples.append(raw[c-1,::4,::4].ravel().copy())
            del raw
        sample=np.concatenate(samples)
        del samples
        lim=np.percentile(sample,[1,99.9])
        del sample
        total=math.ceil(len(mapping)/SHARED_CONTRAST_PAGE_SIZE)
        for page,(filename,fields) in enumerate(shared_contrast_pages(mapping,c),1):
            ncols=min(SHARED_CONTRAST_COLUMNS,len(fields))
            nrows=math.ceil(len(fields)/ncols)
            fig,axs=plt.subplots(nrows,ncols,figsize=(5*ncols,5*nrows+.4),squeeze=False,layout='constrained')
            try:
                for ax in axs.flat:ax.axis('off')
                for ax,(_,m) in zip(axs.flat,fields.iterrows()):
                    raw=read_channels(m.source_path,p)
                    ax.imshow(raw[c-1],cmap='gray',vmin=lim[0],vmax=lim[1])
                    del raw
                    ax.set_title(f'{m.treatment_group} · {m.image_id}')
                    scale_bar(ax,m.pixel_size_x_um)
                fig.suptitle(f'Channel {c} · shared display range {lim[0]:.0f}–{lim[1]:.0f} ADU for all images · page {page}/{total}\nDisplay only. Bright pixels above the display range appear white.',fontsize=11)
                fig.savefig(qdir/filename,dpi=200)
            finally:
                plt.close(fig)


def make_qc(out,mapping,cell,focus,p):
    qdir=out/'QC'
    detailed=p.get('qc_output','detailed')=='detailed'
    if detailed:
        (qdir/'Nucleus_Crops').mkdir(exist_ok=True);(qdir/'Native_Overlays').mkdir(exist_ok=True)
    displays=[]
    # Separate rows for nuclear segmentation and foci, even if a channel has both roles.
    rows=[(p['nuclear_channel'],'nuclear')]+[(c,'foci') for c in p['foci_channels']]
    print(f'QC: 0/{len(mapping)} images processed',flush=True)
    for index,(_,m) in enumerate(mapping.iterrows(),1):
        im=m.image_id;raw=read_channels(m.source_path,p);sx=m.pixel_size_x_um
        nuc=tifffile.imread(out/'Masks'/f'{im}_nuclei.tif');alln=tifffile.imread(out/'Masks'/f'{im}_nuclei_all_candidates.tif')
        exc=np.where(nuc==0,alln,0)
        masks={c:tifffile.imread(out/'Masks'/f'{im}_C{c}_foci.tif') for c in p['foci_channels']}
        limits=[np.percentile(a,[1,99.9]) for a in raw]
        for c in dict.fromkeys(c for c,_ in rows):displays.append({'image_id':im,'channel':c,'display_low_adu':limits[c-1][0],'display_high_adu':limits[c-1][1],'rule':'per-image percentile 1 to 99.9; display only'})
        fig,axs=plt.subplots(len(rows),3,figsize=(17,6*len(rows)),squeeze=False,layout='constrained')
        for row,(c,role) in enumerate(rows):
            a=raw[c-1];lim=limits[c-1];mask=nuc if role=='nuclear' else masks[c]
            axs[row,0].imshow(a,cmap='gray',vmin=lim[0],vmax=lim[1],interpolation='nearest')
            axs[row,0].set_title(f'C{c} {marker_name(p,c)} · original\ndisplay {lim[0]:.0f}–{lim[1]:.0f} ADU')
            if role=='nuclear':
                rgb=overlay(a,lim,np.zeros_like(nuc),nuc,exc)
                axs[row,1].imshow(rgb)
                for r in measure.regionprops(alln):
                    included=bool(np.any(nuc==r.label))
                    axs[row,1].text(r.centroid[1],r.centroid[0],('N' if included else 'X')+str(r.label),
                                  color='#00ffff' if included else '#ff8494',fontsize=6,ha='center',va='center')
                axs[row,1].set_title('Nuclear outlines and IDs\ncyan N = analyzed; red X = excluded')
                axs[row,2].imshow(label_rgb(nuc));axs[row,2].set_title(f'Nuclear label mask · {cell[cell.image_id==im].shape[0]} analyzed')
                if detailed:save_rgb(qdir/'Native_Overlays'/f'{im}_C{c}_nuclear_outlines.png',rgb)
            else:
                rgb=overlay(a,lim,mask,nuc,exc)
                axs[row,1].imshow(rgb);axs[row,1].set_title(f'C{c} counted foci outlined in yellow\n{len(focus[(focus.image_id==im)&(focus.channel==c)]):,} objects in usable nuclei')
                axs[row,2].imshow(label_rgb(mask));axs[row,2].set_title(f'C{c} object label mask\ncolors distinguish adjacent objects')
                if detailed:
                    save_rgb(qdir/'Native_Overlays'/f'{im}_C{c}_foci_outlines.png',rgb)
                    save_rgb(qdir/'Native_Overlays'/f'{im}_C{c}_raw_display.png',np.repeat(norm(a,lim)[...,None],3,axis=2))
                    save_rgb(qdir/'Native_Overlays'/f'{im}_C{c}_labels.png',label_rgb(mask))
            for ax in axs[row]:ax.set_xticks([]);ax.set_yticks([])
            scale_bar(axs[row,0],sx)
        fig.suptitle(f'{im} · {m.treatment_group} · {m.image_name}\n2D image. Display scaling is for QC only; measurements use stored pixel values.',fontsize=11)
        fig.savefig(qdir/f'{im}_Annotated_QC.png',dpi=260);plt.close(fig)
        # Per-nucleus paired raw/outline views include every usable nucleus.
        crops=cell[cell.image_id==im] if detailed else cell.iloc[:0]
        for _,cr in crops.iterrows():
            nl=int(cr.nucleus_label);coords=np.argwhere(nuc==nl)
            y0,x0=np.maximum(coords.min(axis=0)-12,0);y1,x1=np.minimum(coords.max(axis=0)+13,nuc.shape)
            sl=(slice(y0,y1),slice(x0,x1));nr=nuc[sl];roi=nr==nl
            fig,axs=plt.subplots(len(rows),2,figsize=(8,3.7*len(rows)),squeeze=False,layout='constrained')
            for row,(c,role) in enumerate(rows):
                a=raw[c-1][sl];lim=np.percentile(a[roi],[1,99.8]);lim[1]=max(lim[1],lim[0]+10)
                fm=np.where(roi,(nuc if role=='nuclear' else masks[c])[sl],0)
                axs[row,0].imshow(a,cmap='gray',vmin=lim[0],vmax=lim[1],interpolation='nearest')
                axs[row,0].set_title(f'C{c} {marker_name(p,c)} original · {lim[0]:.0f}–{lim[1]:.0f} ADU',fontsize=10)
                if role=='nuclear':
                    rgb=overlay(a,lim,np.zeros_like(fm),np.where(roi,nr,0))
                    title=f'Nucleus N{nl} · {cr.nuclear_area_um2:.1f} μm²'
                else:
                    rgb=overlay(a,lim,fm,np.where(roi,nr,0))
                    title=f'{int(cr[f"C{c}_foci_count"])} counted objects · C{c}'
                axs[row,1].imshow(rgb,interpolation='nearest');axs[row,1].set_title(title,fontsize=10)
                for ax in axs[row]:ax.set_xticks([]);ax.set_yticks([])
                scale_bar(axs[row,0],sx,5)
            fig.suptitle(f'{cr.nucleus_id} · {m.treatment_group}\nYellow = counted focus boundary.',fontsize=11)
            fig.savefig(qdir/'Nucleus_Crops'/f'{cr.nucleus_id}.png',dpi=150);plt.close(fig)
        print(f'QC: {index}/{len(mapping)} images processed ({im})',flush=True)
    pd.DataFrame(displays).to_csv(qdir/'QC_Display_Ranges.csv',index=False)
    print('Rendering shared-contrast QC figures...',flush=True)
    make_shared_contrast(qdir,mapping,p)


def expand_groups(df):
    """Expand only plotting copies; saved measurements keep their unique IDs."""
    if 'group_names' not in df or df.empty:return df.copy()
    result=df.copy()
    result['treatment_group']=[json.loads(value) or ['AMBIGUOUS'] for value in result.group_names]
    return result.explode('treatment_group',ignore_index=True)


def stripbox(ax,df,col,ylabel,rng,log=False):
    for j,g in enumerate(GROUPS):
        vals=df.loc[df.treatment_group==g,col].dropna().to_numpy(float)
        if not len(vals):continue
        bp=ax.boxplot([vals],positions=[j],widths=.40,showfliers=False,patch_artist=True,
                      boxprops={'facecolor':COLORS[j]+'25','edgecolor':COLORS[j]},medianprops={'color':'#222222','linewidth':1.5},
                      whiskerprops={'color':COLORS[j]},capprops={'color':COLORS[j]})
        ax.scatter(j+rng.uniform(-.16,.16,len(vals)),vals,s=16 if len(vals)<100 else 3,
                   color=COLORS[j],alpha=.65 if len(vals)<100 else .18,edgecolors='none',rasterized=True)
    ax.set_xticks(range(len(GROUPS)),LABELS,fontsize=8)
    ax.set_ylabel(ylabel);ax.grid(axis='y',alpha=.16)
    if log and (pd.to_numeric(df[col],errors='coerce')>0).any():ax.set_yscale('log')
    elif ax.get_ylim()[0]>0:ax.set_ylim(bottom=0)


def make_plots(out,cell,focus,image,p):
    cell,focus,image=(expand_groups(df) for df in (cell,focus,image))
    rng=np.random.default_rng(p['random_seed_for_plot_jitter'])
    for c in p['foci_channels']:
        print(f'Rendering comparison plots for channel {c}...',flush=True)
        folder=out/'Plots'/f'Channel_{c}';folder.mkdir(parents=True,exist_ok=True);name=marker_name(p,c)
        specs=[('Cellular_Abundance_Size',[
            ('foci_count','Foci per nucleus'),('foci_density_per_um2','Foci density (μm⁻²)'),
            ('mean_focus_area_um2','Mean focus area / nucleus (μm²)'),('median_focus_area_um2','Median focus area / nucleus (μm²)'),
            ('total_focus_area_um2','Total focus area / nucleus (μm²)'),('nuclear_area_fraction_occupied','Nuclear area fraction occupied')]),
            ('Cellular_Intensity_Contrast',[
            ('mean_focus_intensity_adu','Mean focus intensity / nucleus (ADU)'),('maximum_focus_pixel_intensity_adu','Maximum focus pixel / nucleus (ADU)'),
            ('mean_focus_integrated_adu_px','Mean integrated focus intensity (ADU·px)'),('total_focus_integrated_adu_px','Total focus intensity / nucleus (ADU·px)'),
            ('total_focus_local_corrected_integrated_adu_px','Total local-corrected intensity (ADU·px)'),('mean_focus_to_background_ratio','Mean focus / local background')]),
            ('Cellular_Morphology',[
            ('mean_circularity','Mean circularity'),('mean_eccentricity','Mean eccentricity'),
            ('mean_aspect_ratio','Mean aspect ratio'),('mean_solidity','Mean solidity'),
            ('mean_local_contrast_snr','Mean local contrast / background MAD'),('mean_relative_radial_position','Mean normalized radial position')])]
        for fname,metrics in specs:
            fig,axs=plt.subplots(2,3,figsize=(15,8.7),layout='constrained')
            for ax,(key,label) in zip(axs.flat,metrics):stripbox(ax,cell,f'C{c}_{key}',label,rng)
            fig.suptitle(f'Channel {c} · {name} · cellular distributions\nEach dot is one nucleus. Nuclei are nested within images; descriptive comparisons only.',fontsize=13)
            figsave(fig,folder,fname)
        ff=focus[focus.channel==c]
        metrics=[('focus_area_um2','Focus area (μm²)'),('raw_mean_adu','Mean focus intensity (ADU)'),
                 ('raw_max_adu','Maximum focus intensity (ADU)'),('raw_integrated_adu_px','Integrated focus intensity (ADU·px)'),
                 ('focus_to_background_ratio','Focus / local background'),('circularity','Focus circularity')]
        fig,axs=plt.subplots(2,3,figsize=(15,8.7),layout='constrained')
        for ax,(key,label) in zip(axs.flat,metrics):stripbox(ax,ff,key,label,rng,key in ['focus_area_um2','raw_integrated_adu_px'])
        fig.suptitle(f'Channel {c} · {name} · individual focus measurements\nDots are foci nested within nuclei and images. Boxes show median and IQR; no inference from pooled foci.',fontsize=13)
        figsave(fig,folder,'Focus_Properties')
        metrics=[('mean_foci_per_nucleus','Mean foci / nucleus'),('median_foci_per_nucleus','Median foci / nucleus'),
                 ('mean_foci_density_per_um2','Mean density (μm⁻²)'),('percent_focus_positive_nuclei','Focus-positive nuclei (%)'),
                 ('mean_focus_area_um2','Mean focus area (μm²)'),('median_focus_area_um2','Median focus area (μm²)'),
                 ('mean_total_focus_area_per_nucleus_um2','Mean total focus area / nucleus (μm²)'),('mean_nuclear_area_fraction_occupied','Mean nuclear area fraction occupied'),
                 ('mean_focus_intensity_adu','Mean focus intensity (ADU)'),('mean_focus_maximum_intensity_adu','Mean focus maximum intensity (ADU)'),
                 ('mean_integrated_intensity_per_focus_adu_px','Mean integrated focus intensity (ADU·px)'),
                 ('mean_total_focus_intensity_per_nucleus_adu_px','Mean total intensity / nucleus (ADU·px)'),
                 ('mean_total_local_corrected_focus_intensity_per_nucleus_adu_px','Mean local-corrected total / nucleus (ADU·px)'),
                 ('mean_focus_to_background_ratio','Mean focus / local background'),('mean_eccentricity','Mean focus eccentricity'),('mean_aspect_ratio','Mean focus aspect ratio')]
        fig,axs=plt.subplots(4,4,figsize=(18,16),layout='constrained')
        for ax,(key,label) in zip(axs.flat,metrics):
            for j,g in enumerate(GROUPS):
                for _,r in image[image.treatment_group==g].iterrows():
                    val=r[f'C{c}_{key}'];ax.scatter(j,val,c=COLORS[j],s=75,marker=MARKERS[r.image_id],zorder=4)
                    ax.annotate(r.image_id,(j,val),xytext=(0,8),textcoords='offset points',ha='center',fontsize=8)
            ax.set_xticks(range(len(GROUPS)),LABELS,fontsize=8);ax.set_xlim(-.5,len(GROUPS)-.5);ax.set_ylabel(label);ax.grid(axis='y',alpha=.16)
            maximum=float(image[f'C{c}_{key}'].max())
            ax.set_ylim(bottom=0,top=max(maximum*1.22,.01) if np.isfinite(maximum) else 1)
        fig.suptitle(f'Channel {c} · {name} · image-level comparisons\nEach point is one image. Descriptive comparisons; nuclei are nested observations.',fontsize=14)
        figsave(fig,folder,'Image_Level_Endpoints')
        fig,axs=plt.subplots(1,3,figsize=(15,4.8),layout='constrained')
        for ax,df,key,label,lg in [(axs[0],cell,f'C{c}_foci_count','Foci per nucleus',False),
                                  (axs[1],ff,'focus_area_um2','Focus area (μm²)',True),
                                  (axs[2],ff,'raw_mean_adu','Mean focus intensity (ADU)',True)]:
            for j,g in enumerate(GROUPS):
                vals=np.sort(df.loc[df.treatment_group==g,key].dropna().to_numpy(float))
                ax.step(vals,np.arange(1,len(vals)+1)/max(len(vals),1),where='post',color=COLORS[j],label=f'{LABELS[j]} (n={len(vals):,})',lw=1.8)
            ax.set_xlabel(label);ax.set_ylabel('Cumulative proportion');ax.set_ylim(0,1.02);ax.grid(alpha=.15);ax.legend(fontsize=8,loc='lower right')
            if lg and (pd.to_numeric(df[key],errors='coerce')>0).any():ax.set_xscale('log')
            if not df[key].notna().any():
                ax.text(.5,.5,'No measurements',transform=ax.transAxes,ha='center',va='center')
        fig.suptitle(f'Channel {c} · {name} · distribution curves\nNucleus/focus counts label the observed distributions, not biological replicate counts.',fontsize=12)
        figsave(fig,folder,'Distribution_ECDF')
        fig,ax=plt.subplots(figsize=(7.5,5),layout='constrained')
        for j,g in enumerate(GROUPS):
            for _,r in image[image.treatment_group==g].iterrows():
                v=r[f'C{c}_percent_focus_positive_nuclei']
                n=int(r.nuclei_analyzed);pos=int(cell.loc[(cell.image_id==r.image_id)&(cell.treatment_group==g),f'C{c}_focus_positive'].sum())
                ax.scatter(j,v,s=110,marker=MARKERS[r.image_id],color=COLORS[j]);ax.annotate(f'{r.image_id}: {pos}/{n} nuclei\n{v:.1f}%',(j,v),xytext=(0,-35),textcoords='offset points',ha='center',fontsize=10)
        ax.set_xticks(range(len(GROUPS)),LABELS);ax.set_xlim(-.5,len(GROUPS)-.5);ax.set_ylim(0,110);ax.set_ylabel('Focus-positive nuclei (%)');ax.grid(axis='y',alpha=.2)
        ax.set_title(f'Channel {c} · {name} · focus-positive nuclei\nOne point per image; no biological replicates')
        figsave(fig,folder,'Focus_Positive_Nuclei')
    # Input distributions before any image correction, on a log-count histogram scale.
    mapping=expand_groups(pd.read_csv(out/'Methods'/'Treatment_Group_Mapping.csv'))
    for c in p['foci_channels']:
        fig,ax=plt.subplots(figsize=(9,5),layout='constrained')
        for j,g in enumerate(GROUPS):
            for _,m in mapping[mapping.treatment_group==g].iterrows():
                a=read_channels(m.source_path,p)[c-1]
                upper=256 if a.dtype==np.uint8 else max(7500,float(a.max())+1)
                bins=np.geomspace(max(1,a.min()),upper,130)
                ax.hist(a.ravel(),bins=bins,histtype='step',density=True,color=COLORS[j],label=f'{LABELS[j]} · {m.image_id}',lw=1.5)
        ax.set_xscale('log');ax.set_yscale('log');ax.set_xlabel('Raw pixel intensity (ADU)');ax.set_ylabel('Probability density');ax.legend();ax.set_title(f'Channel {c} · input intensity distributions\nAll image pixels before correction; same axes for treatments')
        figsave(fig,out/'QC',f'C{c}_Input_Intensity_Histogram')


def flatten(p,prefix=''):
    rows=[]
    for k,v in p.items():
        key=f'{prefix}.{k}' if prefix else k
        if isinstance(v,dict):rows.extend(flatten(v,key))
        else:rows.append({'parameter':key,'value':json.dumps(v) if isinstance(v,list) else v})
    return rows


def definition(col):
    exact={
      'image_id':'Stable image observation ID assigned by sorted filename. Not a biological replicate ID.',
      'image_name':'Unmodified raw TIFF filename.', 'treatment_group':'Combined matched group labels, or legacy filename treatment token. Dose units are not inferred.',
      'group_names':'JSON list of all matched groups. Measurements appear in each group comparison, but are stored once. Empty list means unmatched.',
      'number_images_in_treatment':'Number of images with the same complete combination of group labels.',
      'replicate_id':'Missing for this proof of concept, as confirmed by user.',
      'nucleus_id':'Globally unique image ID plus candidate nucleus label; matches nuclear TIFF label.',
      'nucleus_label':'Integer object value in the nuclear label TIFF; IDs can have gaps after exclusions.',
      'focus_id':'Globally unique image, channel and focus label ID.',
      'focus_label':'Integer object value in that image/channel focus TIFF.',
      'raw_integrated_adu_px':'Sum of stored source pixel intensities over this focus. For RGB exports, values are color-component intensities.',
      'focus_to_background_ratio':'Mean original focus intensity / local background median.',
      'local_signal_to_background_ratio':'(Mean original focus intensity - local background median) / local background median.',
      'local_contrast_snr':'(Mean original focus intensity - local background median) / (1.4826 * MAD of local background pixels).',
      'relative_radial_position':'Centroid distance / (centroid distance + distance to nearest nuclear boundary), in physical units. 0 center, 1 edge.',
      'circularity':'4*pi*area_px/perimeter_Crofton_px^2. Raster discretization may yield values above 1; not clipped.',
      'aspect_ratio':'Major-axis length / minor-axis length.',
      'eccentricity':'Region second-moment ellipse eccentricity, 0 circular to 1 elongated.',
      'solidity':'Object area / convex-hull area.',
      'manual_inspection_warnings':'Rule-based inspection flags. These never exclude images.',
      'signal_to_background_ratio':'Median per-focus raw mean / local background median in this image/channel.',
      'background_median_adu':'Median raw intensity outside all candidate nuclei dilated by 15 px.',
      'background_mad_adu':'1.4826 * median absolute deviation in that extracellular region.',
      'biological_replicates':'0: no biological replication supplied; does not denote absence of measured images.'}
    if col in exact:return exact[col]
    s=col
    channel=''
    match=re.match(r'^C(\d+)_(.*)$',s)
    if match:channel=f'Channel {match[1]}: ';s=match[2]
    if s=='foci_count':return channel+'Number of accepted objects assigned to the nucleus, including zero.'
    if s=='focus_positive':return channel+'1 when count >=1; otherwise 0.'
    if s=='nuclear_area_fraction_occupied':return channel+'Sum of accepted focus areas / nuclear area; fraction from 0 to 1.'
    if 'foci_density' in s:return channel+'Accepted focus count divided by nuclear area in the stated units. Image summaries average nucleus-specific densities.'
    if 'mean_focus_intensity' in s:return channel+'Arithmetic mean of individual focus mean raw intensities (equal weight per focus).'
    if 'median_focus_intensity' in s:return channel+'Median of individual focus mean raw intensities.'
    if 'maximum_focus_pixel' in s:return channel+'Maximum raw pixel value across accepted focus objects in the nucleus.'
    if s.startswith('max_focus_intensity'):return channel+'Largest focus mean intensity in the nucleus; differs from maximum pixel intensity.'
    if s.startswith('opening_corrected_'):return 'Statistic of original raw pixels minus the opening background at each pixel. Signed values preserved.'
    if s.startswith('local_corrected_'):return 'Statistic of original raw pixels minus the scalar local background median. Signed values preserved.'
    if 'local_background' in s:return channel+'Local annulus 2–8 px from the focus, same nucleus only, excluding provisional foci plus 1 px. Fallback and sample size recorded.'
    if 'threshold' in s or 'noise' in s:return channel+'Applied noise estimate/threshold; see Output_Parameters.json and Applied_Thresholds.csv for exact common rule.'
    if 'centroid' in s or 'distance' in s:return channel+'Unweighted geometric coordinate/distance; zero-based pixel centers, X column and Y row. Physical values use calibrated X/Y pixels.'
    if 'area' in s:return channel+s.replace('_',' ')+'. Area in px counts segmented pixels; um2 uses pixel_size_x_um * pixel_size_y_um. Zero-focus totals=0; per-focus summaries missing.'
    if 'integrated' in s or 'total_focus_intensity' in s:return channel+s.replace('_',' ')+'. Integrated intensity sums pixel values (ADU*pixel); per-nucleus totals include zero-focus nuclei as 0.'
    if s.startswith('raw_'):return 'Statistic of stored unsmoothed source intensities inside the focus; ADU columns contain color-component values for RGB exports.'
    return channel+s.replace('_',' ')+'. See Methods.md for definitions, units, nesting and missing-data conventions.'


def build_methods(out,p,mapping,images,cell,focus,qc):
    par=flatten(p)
    for _,m in mapping.iterrows():
        for k in ['pixel_size_x_um','pixel_size_y_um']:par.append({'parameter':f'{m.image_id}.{k}','value':m[k]})
    par.extend([{'parameter':'method_reference.morphology','value':'https://scikit-image.org/docs/0.26.x/api/skimage.morphology.html'},
                {'parameter':'method_reference.watershed','value':'https://scikit-image.org/docs/0.26.x/api/skimage.segmentation.html#skimage.segmentation.watershed'}])
    pd.DataFrame(par).to_csv(out/'Methods'/'Analysis_Parameters.csv',index=False)
    tables={'Image_Summary':images,'Cell_Summary':cell,'Focus_Data':focus,'QC_Summary':qc}
    dictionary=[]
    for sheet,df in tables.items():
        for col in df:
            dictionary.append({'table':sheet,'column':col,'data_type':str(df[col].dtype),'definition':definition(col)})
    pd.DataFrame(dictionary).to_csv(out/'Methods'/'Data_Dictionary.csv',index=False)
    # JSON matrix for the required spreadsheet authoring library, preserving numeric types and missing values.
    wb={name:{'columns':list(df.columns),'rows':json.loads(df.to_json(orient='values'))} for name,df in tables.items()}
    for name,df in [('Analysis_Parameters',pd.DataFrame(par)),('Treatment_Mapping',mapping),('Data_Dictionary',pd.DataFrame(dictionary))]:
        wb[name]={'columns':list(df.columns),'rows':json.loads(df.to_json(orient='values'))}
    (out/'Methods'/'workbook_data.json').write_text(json.dumps(wb,allow_nan=False))
    versions=json.loads((out/'Methods'/'Run_Manifest.json').read_text())
    requirements=[f'numpy=={versions["numpy"]}',f'scipy=={versions["scipy"]}',f'scikit-image=={versions["skimage"]}',
                  f'pandas=={versions["pandas"]}',f'tifffile=={versions["tifffile"]}',f'matplotlib=={matplotlib.__version__}']
    (out/'Methods'/'requirements.txt').write_text('\n'.join(requirements)+'\n')
    channels=', '.join(f'C{c} ({marker_name(p,c)})' for c in p['foci_channels'])
    nuclear=f"C{p['nuclear_channel']} ({marker_name(p,p['nuclear_channel'])})"
    floors='; '.join(f"C{c}: support {p['foci'][f'channel_{c}']['support_floor_adu']:g}, seed {p['foci'][f'channel_{c}']['seed_floor_adu']:g}, prominence {p['foci'][f'channel_{c}']['prominence_floor_adu']:g} ADU" for c in p['foci_channels'])
    detailed_qc=('Nucleus_Crops contains every accepted nucleus. Native_Overlays retains displays at the original image dimensions.'
                 if p.get('qc_output','detailed')=='detailed' else
                 'Minimal QC was requested; separate native overlays and nucleus crops were not generated.')
    rgb_inputs='input_format' in mapping and mapping.input_format.eq('rgb_export').any()
    rgb_note=('RGB exports were explicitly mapped to channels in the order '+', '.join(p['rgb_channel_order'])+
              '. Values are measured directly from the stored color components without rescaling. '
              'These rendered intensities may reflect acquisition display settings and clipping; they do not recover original detector ADU. '
              'Threshold floors use the stored pixel scale (0–255 for uint8), and RGB results should be treated as exploratory.'
              if rgb_inputs else '')
    rgb_option=' --rgb-channel-order '+' '.join(p['rgb_channel_order']) if rgb_inputs else ''
    calibration_option=' --pixel-size-um '+' '.join(map(str,p['pixel_size_um'])) if p.get('pixel_size_um') else ''
    methods=f'''# Reproducible 2D nuclear foci quantification

## Scope and experimental units
This report covers {len(mapping)} TIFF images, with foci measurements for {channels}. Nuclear segmentation uses {nuclear}. Channel identities come from the saved analysis parameters. Image IDs identify image observations, not biological replicates. Treatment tokens and replicate metadata are retained in Treatment_Group_Mapping.csv.

Inputs are read as calibrated 2D channel planes in CYX order. Source layout, channel mapping, bit depth, pixel sizes, calibration source, and source paths are recorded per image in Treatment_Group_Mapping.csv and Image_Metadata.json. RGB exports are reordered only when an explicit rgb_channel_order is supplied. Calibration priority is an explicit pixel_size_um override, then OME PhysicalSizeX/PhysicalSizeY, then ImageJ micron metadata, then TIFF inch/centimeter resolution. Areas, axes and distances use both X and Y calibration. These scripts do not modify the source TIFFs. Run_Manifest.json records the analysis selections; Output_Parameters.json records the selections included in this report.

{rgb_note}

Configured group_names use case-insensitive literal substring matching against the filename without its extension. Every matching name is retained in the group_names JSON-list column; treatment_group displays their combination. Without configured names, the legacy filename parser is used. Unmatched images remain AMBIGUOUS. Measurements and totals count each object once; comparison plots include it in every assigned group. Overlapping groups are not independent observations. number_images_in_treatment counts images with the same full combination of labels.

## Nuclear segmentation
Gaussian smoothing (sigma 2 px) is used only for detection. The foreground threshold is the full-image smoothed nuclear-channel median plus max(5 ADU, 4 x robust MAD sigma); nuclei occupy a minority of these fields. Closing uses a 3 px disk and internal holes are filled. Components below 1,000 px are discarded as debris before the candidate list. A distance transform smoothed at sigma 3 px supplies h-maxima (height 10 px, edge distance >20 px) for watershed splitting. A connected candidate without such a seed receives its greatest-distance pixel as a seed, allowing border fragments to be audited.

Candidates from 25 to 500 square microns are retained unless their bounding box intersects a 2 px image-edge buffer. This deliberately excludes all border-touching nuclei, including potentially mildly truncated ones. Exclusion percentage uses the post-debris, post-watershed candidate count as denominator. Nuclear masks retain candidate label IDs. Objects below the debris prefilter are not included in this denominator. Nuclear solidity <0.85 or aspect ratio >3 generates an inspection warning only; such nuclei remain in the analysis. Nucleus_Inclusion_Audit.csv records every candidate and exclusion reason. No manual mask edits were made.

## Foci segmentation
The same numerical parameters and automatic rule apply to every image within a channel. Thresholds and segmentation settings come from the saved analysis parameters. This is exploratory segmentation, not a blinded or ground-truth-validated detector.

Original intensities are lightly smoothed (Gaussian sigma 1 px). Background is a grayscale opening with a radius 12 px approximate disk (scikit-image sequence decomposition; physical size depends on pixel calibration). The difference between smooth image and opening supplies the detection signal. Structures larger than this background scale can be attenuated; the result measures discrete puncta rather than diffuse or pan-nuclear staining.

Per-nucleus noise sigma is 1.4826 x MAD of (I00 - I10 - I01 + I11)/2 on 2x2 windows fully inside the nucleus. Approximate noise after Gaussian smoothing is sigma_raw / (2 sqrt(pi) sigma_gaussian). This estimate is a reproducible noise proxy, not a calibrated photon model or statistical error rate.

Support threshold = max(channel floor, 3 x sigma_smooth). Seed threshold = max(channel floor, 6 x sigma_smooth). Peak prominence = max(channel floor, 3 x sigma_smooth). Selected channel floors are {floors}. Applied per-nucleus values are saved in Applied_Thresholds.csv. H-maxima establish seeds; watershed on the negative corrected image segments actual pixel objects inside support masks. Local maxima alone are never counted. Watershed boundaries are left unassigned so adjacent objects remain separable. Each basin is trimmed at max(support threshold, 25% of its corrected peak), and only the connected component containing that peak is retained.

Objects require at least 9 pixels (physical area depends on pixel calibration), at most 25 square microns, and raw mean minus local background >=2 x per-nucleus raw noise sigma. No shape-based focus exclusion is applied. Dense neighboring foci with insufficient prominence remain merged. The detector may miss faint objects below the common noise/contrast limits. Focus_Candidate_Audit.csv records size and contrast exclusions; subthreshold unseeded signal does not enter this audit. Counted foci are constrained to usable nuclei. Boundary-touching focus objects are retained with a flag. Excluded nuclei and extranuclear puncta are not part of the per-nucleus primary counts.

## Local background and measurements
The local background region is an annulus at Euclidean pixel distances >2 and <=8 from the segmented object, limited to its own nucleus and excluding all provisional foci dilated by 1 px. Provisional includes objects later rejected by size/contrast, avoiding their reuse as background. At least 20 annular pixels are required. Otherwise all eligible nonfocus pixels in the same nucleus are used; if still insufficient, the opening background over the object is used. Background source and sample size are preserved.

Measurements use the original unsmoothed pixels. Raw mean/median/min/max/sum, opening-corrected versions, and local-background-corrected versions are saved. Corrected pixels are not clipped to zero. Integrated intensity is ADU x pixel, not calibrated molecular abundance. Focus/background = raw mean / local median. Local signal/background = (raw mean - local median)/local median. Local contrast SNR uses the annulus robust MAD sigma in the denominator, distinct from the noise estimate used in filtering. Dataset background is measured outside all candidate nuclei after 15 px dilation.

Equivalent diameter is the diameter of a circle of equal segmented area. Major/minor axes derive from region second moments; calibrated axes are recomputed after scaling X and Y. Circularity uses 4*pi*area/perimeter_Crofton^2 (four directions); pixelation may yield values >1, and these are retained. Small-focus morphology is resolution limited. Solidity is area/convex area. Unweighted centroids use zero-based pixel centers, X=column and Y=row. Relative radial position is d_centroid / (d_centroid + nearest_edge_distance), in physical units. It is 0 at the centroid and approaches 1 at the boundary; it is not a ray-based or equal-area nuclear coordinate.

For each nucleus, per-focus intensity means and medians summarize the focus means with equal weight per focus. Maximum focus mean and maximum focus pixel are separate columns. Foci counts, density, total areas, occupied fraction and summed intensity are zero for focus-negative nuclei. Per-focus size/intensity/morphology averages are missing when there are no foci. Image summaries include every usable nucleus, including zero-focus nuclei. Image mean density and area fraction average nucleus-specific values; they are not ratios of pooled totals. SD is a sample SD (ddof=1) across nuclei and does not represent biological-replicate uncertainty.

## QC and sensitivity
Annotated_QC figures show paired raw and overlay views plus label masks for each channel. {detailed_qc} Yellow marks counted foci, cyan accepted nuclear boundaries and red excluded candidates. TIFF mask values map exactly to the result label columns. Raw/overlay pairs share display limits; per-image QC contrast is for inspection only. Shared_Contrast figures use one range per channel across all images. Display clipping is distinct from detector saturation and does not change measurements.

Saturation uses each input dtype's maximum stored value: 255 for uint8, 65535 for uint16. For RGB exports this measures clipping of exported color components, not detector saturation. The source files do not necessarily specify detector effective bit depth; absence of container saturation cannot prove absence of detector clipping. Per-image saturation values are in Treatment_Group_Mapping.csv. Observed maxima and saturation fractions should be reviewed in Input_Intensity_Distributions.csv and the saved QC tables.

QC_Summary.csv reports fixed numerical and dataset-relative flags, with exact cutoffs in Output_Parameters.json. These include nuclear counts/exclusions, shape concerns, background, saturation, count extremes, size extremes, intensity tails, contrast, density and threshold sensitivity. Flags request inspection and never exclude images. Dataset-relative heuristics have limited reliability with few images. Genuine treatment biology can trigger a flag.

Threshold sensitivity reruns each nucleus at common factors 0.8 and 1.2 applied to support, seed, prominence and net-mean contrast thresholds, with all other rules fixed. It is reported separately from the primary factor-1 masks and tables. Because splitting and contrast checks interact, counts need not be strictly monotonic with threshold factor. Sensitivity quantifies technical dependence; it does not estimate detection accuracy or a confidence interval.

## Treatment comparisons and limits
The experimental structure is focus -> nucleus -> image -> treatment. The reports are descriptive: no hypothesis tests, p-values, biological confidence intervals or multiple-comparison correction are reported. Replicate structure must be assessed from the experiment metadata. Cellular/focus distribution plots visualize heterogeneity only. Image-level plots preserve the experimental hierarchy. The observed treatment ranking cannot establish a reproducible biological treatment effect.

This analysis measures objects in 2D projections, not 3D foci: overlap in Z can merge distinct foci and alter areas. Signal background subtraction, thresholds, resolution and the field selected affect counts. No cross-channel colocalization or biological comparison is performed. Nuclear annotation and focus ground truth would be needed to estimate precision/recall and validate absolute counts.

## Reproduce
Use Python 3.12 and the pinned packages in requirements.txt. Run the commands below from the centralized scripts folder (or wherever you copied the scripts). The analysis folder contains parameters and outputs, not executable scripts:

```sh
python -m pip install -r requirements.txt
python analyze_foci.py --input-dir /path/to/raw_tiffs --parameters /path/to/analysis/Methods/parameters.json --output-dir /path/to/new_analysis{rgb_option}{calibration_option}
python make_outputs.py --output-dir /path/to/new_analysis --nuclear-channel {p['nuclear_channel']} --foci-channels {' '.join(map(str,p['foci_channels']))} --qc-output {p.get('qc_output','detailed')}
python validate_results.py --output-dir /path/to/new_analysis --nuclear-channel {p['nuclear_channel']} --foci-channels {' '.join(map(str,p['foci_channels']))}
```

The optional workbook renderer still has C2/C3 column-order assumptions. It uses @oai/artifact-tool (available in the Codex bundled runtime; Codex bundle version 26.921.10847). CSV tables and masks are fully reproducible using the Python steps independently of the workbook engine. make_workbook.mjs imports the saved numeric measurement matrix; the workbook is a fixed analysis result, not a live image-segmentation tool. Changing a workbook value does not rerun segmentation. Run_Manifest.json captures software versions, raw checksums and parameter/code hashes. Source file paths can be changed with --input-dir without changing the raw files.

## Algorithm references
scikit-image official documentation: [morphology and h-maxima](https://scikit-image.org/docs/0.26.x/api/skimage.morphology.html), [marker-controlled watershed](https://scikit-image.org/docs/0.26.x/api/skimage.segmentation.html#skimage.segmentation.watershed), and [region measurements](https://scikit-image.org/docs/0.26.x/api/skimage.measure.html). These document implementations; they do not validate the selected parameters for this assay.
'''
    (out/'Methods'/'Methods.md').write_text(methods)


def report(out,mapping,images,cell,focus,qc,p):
    def esc(x):return html.escape(str(x))
    detailed=p.get('qc_output','detailed')=='detailed'
    channels=p['foci_channels']
    nuclear=f"C{p['nuclear_channel']} ({marker_name(p,p['nuclear_channel'])})"
    cols=['image_id','treatment_group','nuclei_analyzed']
    labels=['Image','Treatment','Nuclei']
    for c in channels:
        cols.extend([f'C{c}_total_foci',f'C{c}_mean_foci_per_nucleus'])
        labels.extend([f'C{c} {marker_name(p,c)} foci',f'C{c} mean / nucleus'])
    summary=images[cols].copy();summary.columns=labels
    totals=' · '.join(f'C{c} {marker_name(p,c)}: {int((focus.channel==c).sum()):,} foci' for c in channels)
    body='''<!doctype html><html><head><meta charset="utf-8"><title>Foci analysis review</title><style>
body{font:16px/1.55 system-ui,Arial;color:#203043;background:#f7f9fb;max-width:1250px;margin:auto;padding:35px}h1{font-size:32px}h2{margin-top:2.2em}a{color:#006b9c}table{border-collapse:collapse;background:white;width:100%;font-size:14px}th,td{padding:10px;border-bottom:1px solid #dbe2e9;text-align:left}th{background:#243c50;color:white}img{max-width:100%;height:auto}article{background:white;padding:22px;margin:18px 0;border:1px solid #dbe2e9;border-radius:8px}.note{border-left:4px solid #d69132;padding:12px 18px;background:#fff6e8}.gallery{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:10px}.gallery img{width:100%}summary{cursor:pointer;font-weight:600}small{color:#526779}.links a{margin-right:18px}li{margin-bottom:7px}</style></head><body>'''
    body+=f'<h1>Fluorescent foci quantification</h1><p>{len(mapping)} images · {expand_groups(mapping).treatment_group.nunique()} groups · {len(cell)} usable nuclei · {esc(totals)}</p>'
    body+='<p class="note">All comparisons are descriptive. Nuclei and foci are nested within images. Detected objects and 2D areas require QC review; dense and faint signal remain sources of uncertainty.</p>'
    if 'input_format' in mapping and mapping.input_format.eq('rgb_export').any():
        body+='<p class="note">RGB export analysis: channels use the saved color mapping. Intensities are stored color-component values, not recovered detector ADU. Export display settings and clipping can affect measurements; treat these results as exploratory.</p>'
    if p.get('group_names'):
        body+='<p>Images appear in every matching group comparison. Groups can overlap; group counts must not be added as independent observations. The summary below shows each image once with its combined group labels.</p>'
    body+='<p class="links"><a href="Methods/Methods.md">Detailed methods</a><a href="Methods/Output_Parameters.json">Report parameters</a><a href="Methods/Treatment_Group_Mapping.csv">Image mapping</a><a href="QC/QC_Summary.csv">QC summary CSV</a></p>'
    body+=summary.to_html(index=False,float_format=lambda x:f'{x:,.2f}',border=0)
    body+=f'<p>Nuclear segmentation: {esc(nuclear)}. Marker labels come from the analysis configuration. Border nuclei are excluded; focus-negative nuclei remain in summaries. Per-image calibration is recorded in the image mapping.</p>'
    body+='<h2>QC findings</h2><p>Flags identify inspection targets and never exclude images. Full reasons, thresholds and sensitivity counts are in the QC table.</p>'
    body+=qc[['image_id','channel','number_usable_nuclei','number_excluded_nuclei','manual_inspection_warnings']].to_html(index=False,border=0)
    body+='<h2>Inspect every image</h2><p>Open the full annotated image to zoom. Yellow outlines mark counted objects; cyan marks analyzed nuclei and red marks excluded nuclei. Colors in label masks distinguish objects, not intensity.</p>'
    for _,m in mapping.iterrows():
        im=m.image_id;body+=f'<article><h3>{esc(im)} · {esc(m.treatment_group)}</h3><small>{esc(m.image_name)}</small>'
        body+=f'<a href="QC/{im}_Annotated_QC.png"><img loading="lazy" src="QC/{im}_Annotated_QC.png"></a>'
        body+='<p class="links">'
        for c in p['foci_channels']:
            if detailed:body+=f'<a href="QC/Native_Overlays/{im}_C{c}_foci_outlines.png">C{c} native overlay</a>'
            body+=f'<a href="Masks/{im}_C{c}_foci.tif">C{c} label TIFF</a>'
        if detailed:body+=f'<a href="QC/Native_Overlays/{im}_C{p["nuclear_channel"]}_nuclear_outlines.png">Nuclear overlay</a>'
        body+=f'<a href="Masks/{im}_nuclei.tif">Nuclear label TIFF</a></p>'
        if detailed:
            body+=f'<details><summary>All {len(cell[cell.image_id==im])} analyzed nuclei, individually</summary><div class="gallery">'
            for _,cr in cell[cell.image_id==im].iterrows():
                nid=cr.nucleus_id;body+=f'<a href="QC/Nucleus_Crops/{nid}.png"><img loading="lazy" src="QC/Nucleus_Crops/{nid}.png"><small>{nid}</small></a>'
            body+='</div></details>'
        body+='</article>'
    body+='<h2>Treatment comparisons within each channel</h2>'
    for c in p['foci_channels']:
        body+=f'<article><h3>Channel {c} · '+esc(marker_name(p,c))+'</h3>'
        body+=f'<a href="Plots/Channel_{c}/Image_Level_Endpoints.png"><img loading="lazy" src="Plots/Channel_{c}/Image_Level_Endpoints.png"></a>'
        for f in ['Cellular_Abundance_Size','Cellular_Intensity_Contrast','Cellular_Morphology','Focus_Properties','Distribution_ECDF','Focus_Positive_Nuclei']:
            body+=f'<p><a href="Plots/Channel_{c}/{f}.png">{f.replace("_"," ")}</a> · <a href="Plots/Channel_{c}/{f}.svg">editable SVG</a></p>'
        body+='<details><summary>Shared-contrast raw fields</summary>'
        for page,(filename,_) in enumerate(shared_contrast_pages(mapping,c),1):
            body+=f'<p>Page {page}</p><a href="QC/{filename}"><img loading="lazy" src="QC/{filename}" alt="Channel {c} shared contrast, page {page}"></a>'
        body+='</details></article>'
    body+='<h2>Files and reproducibility</h2><ul>'
    for f in ['Results/Image_Summary.csv','Results/Cell_Summary.csv','Results/Focus_Data.csv','QC/Nucleus_Inclusion_Audit.csv','QC/Focus_Candidate_Audit.csv','QC/Threshold_Sensitivity.csv','QC/Input_Intensity_Distributions.csv','Methods/Applied_Thresholds.csv','Methods/Data_Dictionary.csv','Methods/Run_Manifest.json','Methods/Validation_Report.json','Methods/Methods.md']:
        body+=f'<li><a href="{f}">{f}</a></li>'
    body+='</ul><p>No cross-channel colocalization analysis was performed. Thresholds and object definitions were shared across treatments. Run parameters and provenance are in Methods/; reusable tools are kept separately in the scripts folder.</p></body></html>'
    (out/'Review.html').write_text(body)
    lines=['# Foci analysis report','',f'Processed {len(mapping)} images; {len(cell)} nuclei analyzed.',
           f'Nuclear channel: {nuclear}.',totals,'',
           'Descriptive measurements of segmented 2D nuclear objects; no inferential tests.','',
           '| '+' | '.join(labels)+' |','|'+'|'.join(['---']*len(labels))+'|']
    for row in summary.itertuples(index=False,name=None):
        lines.append('| '+' | '.join(f'{v:.2f}' if isinstance(v,float) else str(v) for v in row)+' |')
    lines+=['',f'Methods: foreground segmentation and distance watershed on {nuclear}; channel-specific background opening, noise-adaptive thresholds, object watershed, size and local-contrast filters.',
            '', 'QC inspection findings:','']
    for _,r in qc.iterrows():lines.append(f'- {r.image_id}, C{r.channel}: {r.manual_inspection_warnings}.')
    if p.get('group_names'):
        lines+=['','Group comparisons include each image in every matching group; groups overlap. Measurement totals count each object once.']
    lines+=['','Major outputs:','', '- Review.html: linked review report'+(' and nucleus galleries.' if detailed else '.'),
            '- Results/: image, cell, focus, and QC measurement tables.',
            '- QC/: annotated figures, '+('native overlays, nucleus crops, ' if detailed else '')+'shared-contrast figures, and QC metrics.',
            '- Plots/: '+', '.join(f'Channel_{c}' for c in channels)+'; comparison figures in PNG and SVG.',
            '- Methods/Output_Parameters.json: parameters and channel selection for this report.',
            '', 'Limitations: 2D projections; no manual ground truth. Dense or dim foci are parameter sensitive. See Methods/Methods.md for definitions.']
    (out/'Report.md').write_text('\n'.join(lines)+'\n')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output-dir',type=Path,required=True,help='Analysis folder containing Results, Masks, and Methods')
    ap.add_argument('--qc-output',choices=['minimal','detailed'],default='detailed',
                    help='minimal omits separate native overlays and nucleus crops (default: detailed)')
    add_channel_arguments(ap)
    args=ap.parse_args();out=args.output_dir
    mapping=pd.read_csv(out/'Methods'/'Treatment_Group_Mapping.csv');images=pd.read_csv(out/'Results'/'Image_Summary.csv')
    cell=pd.read_csv(out/'Results'/'Cell_Summary.csv');focus=pd.read_csv(out/'Results'/'Focus_Data.csv')
    qc=pd.read_csv(out/'Results'/'QC_Summary.csv')
    try:
        p=resolve_channels(out,args.nuclear_channel,args.foci_channels)
        check_channel_inputs(out,p,mapping,images,cell,qc)
    except ValueError as exc:
        ap.error(str(exc))
    channels=p['foci_channels']
    p['qc_output']=args.qc_output
    images=selected_columns(images,channels);cell=selected_columns(cell,channels)
    if 'qc_flags' in images:
        images['qc_flags']=images.apply(lambda row:' | '.join(f'C{c}: {row[f"C{c}_qc_flags"]}' for c in channels),axis=1)
    focus=focus[focus.channel.isin(channels)];qc=qc[qc.channel.isin(channels)]
    # Derive groups and image symbols from this run rather than the pilot dataset.
    global GROUPS,LABELS,COLORS,MARKERS
    GROUPS=list(expand_groups(mapping).treatment_group.unique());LABELS=[g.replace('ART558_20PDS','ART558 + 20PDS') for g in GROUPS]
    palette=['#0072B2','#D55E00','#009E73','#CC79A7','#E69F00','#56B4E9','#000000']
    COLORS=[palette[i%len(palette)] for i in range(len(GROUPS))]
    MARKERS={im:['s','o','^','D','v','P','X'][i%7] for i,im in enumerate(mapping.image_id)}
    (out/'Methods'/'Output_Parameters.json').write_text(json.dumps(p,indent=2))
    make_qc(out,mapping,cell,focus,p)
    make_plots(out,cell,focus,images,p)
    print('Writing methods and review reports...',flush=True)
    build_methods(out,p,mapping,images,cell,focus,qc)
    report(out,mapping,images,cell,focus,qc,p)
    print('QC, plots, methods and report complete',flush=True)


if __name__=='__main__':main()
