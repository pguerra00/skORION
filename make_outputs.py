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
from progress_reporting import add_progress_argument, ProgressReporter, advance
from image_review import image_display_name, representative_grid, review_images
from colocalization import settings as colocalization_settings, read_tables as read_colocalization_tables
import colocalization_outputs

GROUPS=['20PDS','100MMC','ART558_20PDS']
LABELS=['20PDS','100MMC','ART558 + 20PDS']
# First 32 colors of Colorcet's glasbey_dark, for plots on white backgrounds.
# https://colorcet.holoviz.org/user_guide/Categorical.html
TREATMENT_PALETTE=(
    '#d60000','#8c3bff','#018700','#00acc6','#e6a500','#ff7ed1','#6b004f','#573b00',
    '#005659','#15e18c','#0000dd','#a17569','#bcb6ff','#bf03b8','#645472','#790000',
    '#0774d8','#729a7c','#ff7752','#004b00','#8e7b01','#f2007b','#8eba00','#a57bb8',
    '#5901a3','#e2afaf','#a03a52','#a1c8c8','#9e4b00','#546744','#bac389','#5e7b87',
)
COLORS=list(TREATMENT_PALETTE)
STYLE={'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,'axes.labelsize':10,
       'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','savefig.facecolor':'white'}
plt.rcParams.update(STYLE)
SHARED_CONTRAST_PAGE_SIZE=12
SHARED_CONTRAST_COLUMNS=4
SCATTER_SIZE=8


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


def shared_display_ranges(mapping,channels,p,progress=None):
    ranges={}
    for c in channels:
        # Copy only sampled pixels so the views cannot retain full TIFF stacks.
        samples=[]
        for path in mapping.source_path:
            raw=read_channels(path,p)
            samples.append(raw[c-1,::4,::4].ravel().copy())
            del raw
            advance(progress,f'Channel {c} · shared-contrast sample: {Path(path).name}')
        sample=np.concatenate(samples)
        del samples
        lim=np.percentile(sample,[1,99.9])
        del sample
        advance(progress,f'Channel {c} · shared display range calculated')
        ranges[c]=lim
    return ranges


def make_shared_contrast(qdir,mapping,p,progress=None,display_ranges=None):
    """Render bounded grids with one dataset-wide display range per channel."""
    if mapping.empty:return
    ranges=shared_display_ranges(mapping,p['foci_channels'],p,progress) if display_ranges is None else display_ranges
    for c in p['foci_channels']:
        lim=ranges[c]
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
                    ax.set_title(textwrap.fill(f'{m.treatment_group} · {image_display_name(m)}',60),fontsize=10)
                    scale_bar(ax,m.pixel_size_x_um)
                fig.suptitle(f'Channel {c} · shared display range {lim[0]:.0f}–{lim[1]:.0f} ADU for all images · page {page}/{total}\nDisplay only. Bright pixels above the display range appear white.',fontsize=11)
                fig.savefig(qdir/filename,dpi=200)
                advance(progress,f'Channel {c} · shared-contrast page {page}/{total} saved')
            finally:
                plt.close(fig)


def make_qc(out,mapping,cell,focus,p,progress=None,display_ranges=None):
    qdir=out/'QC'
    detailed=p.get('qc_output','detailed')=='detailed'
    if detailed:
        (qdir/'Nucleus_Crops').mkdir(exist_ok=True);(qdir/'Native_Overlays').mkdir(exist_ok=True)
    displays=[]
    # Separate rows for nuclear segmentation and foci, even if a channel has both roles.
    rows=[(p['nuclear_channel'],'nuclear')]+[(c,'foci') for c in p['foci_channels']]
    print(f'QC: 0/{len(mapping)} images processed',flush=True)
    for index,(_,m) in enumerate(mapping.iterrows(),1):
        im=m.image_id;name=image_display_name(m);raw=read_channels(m.source_path,p);sx=m.pixel_size_x_um
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
                if detailed:
                    save_rgb(qdir/'Native_Overlays'/f'{im}_C{c}_nuclear_outlines.png',rgb)
                    advance(progress,f'{name} · Channel {c} · nuclear overlay saved')
            else:
                rgb=overlay(a,lim,mask,nuc,exc)
                axs[row,1].imshow(rgb);axs[row,1].set_title(f'C{c} counted foci outlined in yellow\n{len(focus[(focus.image_id==im)&(focus.channel==c)]):,} objects in usable nuclei')
                axs[row,2].imshow(label_rgb(mask));axs[row,2].set_title(f'C{c} object label mask\ncolors distinguish adjacent objects')
                if detailed:
                    save_rgb(qdir/'Native_Overlays'/f'{im}_C{c}_foci_outlines.png',rgb)
                    advance(progress,f'{name} · Channel {c} · foci overlay saved')
                    save_rgb(qdir/'Native_Overlays'/f'{im}_C{c}_raw_display.png',np.repeat(norm(a,lim)[...,None],3,axis=2))
                    advance(progress,f'{name} · Channel {c} · raw display saved')
                    save_rgb(qdir/'Native_Overlays'/f'{im}_C{c}_labels.png',label_rgb(mask))
                    advance(progress,f'{name} · Channel {c} · label display saved')
            for ax in axs[row]:ax.set_xticks([]);ax.set_yticks([])
            scale_bar(axs[row,0],sx)
        fig.suptitle(textwrap.fill(f'{name} · {m.treatment_group}',110)+'\n2D image. Display scaling is for QC only; measurements use stored pixel values.',fontsize=11)
        fig.savefig(qdir/f'{im}_Annotated_QC.png',dpi=260);plt.close(fig)
        advance(progress,f'Image {index}/{len(mapping)} · {name} · annotated QC saved')
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
            fig.suptitle(textwrap.fill(f'{name} · N{nl:03d} · {m.treatment_group}',85)+'\nYellow = counted focus boundary.',fontsize=11)
            fig.savefig(qdir/'Nucleus_Crops'/f'{cr.nucleus_id}.png',dpi=150);plt.close(fig)
            advance(progress,f'{name} · N{nl:03d} · nucleus crop saved')
        print(f'QC: {index}/{len(mapping)} images processed ({name})',flush=True)
    pd.DataFrame(displays).to_csv(qdir/'QC_Display_Ranges.csv',index=False)
    advance(progress,'QC display ranges saved',force=True)
    print('Rendering shared-contrast QC figures...',flush=True)
    make_shared_contrast(qdir,mapping,p,progress=progress,display_ranges=display_ranges)


def make_representative_images(out,mapping,selection,p,display_ranges,progress=None):
    """Render two full-field representatives per group with shared contrast."""
    groups=list(mapping.treatment_group.unique());nrows,ncols=representative_grid(len(groups))
    channels=list(dict.fromkeys([p['nuclear_channel'],*p['foci_channels']]))
    pd.DataFrame([dict(channel=c,display_low_adu=display_ranges[c][0],display_high_adu=display_ranges[c][1],
                       rule='Dataset-wide sampled percentile 1 to 99.9; shared with shared-contrast QC')
                  for c in channels]).to_csv(out/'QC'/'Representative_Display_Ranges.csv',index=False)
    advance(progress,'Representative display ranges saved')
    height=1.4+1.8*len(channels)
    fig=plt.figure(figsize=(5.8*ncols,height*nrows+1),layout='constrained')
    try:
        panels=np.asarray(fig.subfigures(nrows,ncols,squeeze=False)).ravel()
        fig.suptitle(textwrap.fill(out.name,100)+' · representative images',fontsize=15,weight='bold')
        for j,(panel,g) in enumerate(zip(panels,groups)):
            color=COLORS[j%len(COLORS)]
            panel.suptitle(textwrap.fill(str(g),48),color=color,weight='bold',fontsize=11)
            axes=panel.subplots(len(channels),2,squeeze=False)
            candidates=selection[selection.treatment_group==g]
            chosen=candidates[candidates.selected].sort_values('rank')
            for slot in range(2):
                if slot>=len(chosen):
                    reasons=candidates.loc[~candidates.eligible,'ineligibility_reason'].unique()
                    explanation='Fewer than two eligible images'
                    if len(reasons):explanation+='\n'+', '.join(reasons)
                    for ax in axes[:,slot]:ax.axis('off')
                    axes[0,slot].text(.5,.5,textwrap.fill(explanation,30),ha='center',va='center',transform=axes[0,slot].transAxes,fontsize=9)
                    continue
                record=chosen.iloc[slot];m=mapping[mapping.image_id==record.image_id].iloc[0]
                raw=read_channels(m.source_path,p)
                stride=max(1,math.ceil(max(raw.shape[-2:])/1024))
                for row,c in enumerate(channels):
                    ax=axes[row,slot];lo,hi=display_ranges[c]
                    # Copy a bounded display plane so imshow cannot retain the raw stack.
                    display=raw[c-1,::stride,::stride].copy()
                    width=raw.shape[-1]*float(m.pixel_size_x_um)
                    height_um=raw.shape[-2]*float(m.pixel_size_y_um)
                    ax.imshow(display,cmap='gray',vmin=lo,vmax=max(hi,lo+1),
                              extent=(0,width,height_um,0),interpolation='nearest',aspect='equal')
                    label=f'C{c} {marker_name(p,c)}'
                    if row==0:
                        label=textwrap.fill(image_display_name(m),32)+'\n'+label
                        if record.partial_metrics:label+=' · partial metrics'
                    ax.set_title(label,fontsize=8)
                    ax.set_xticks([]);ax.set_yticks([])
                    scale_bar(ax,1,width=min(20,width/4))
                del raw
            advance(progress,f'{g} · representative image panels prepared')
        for panel in panels[len(groups):]:panel.subplots().axis('off')
        fig.supxlabel('Closest unflagged images to treatment means across selected-channel endpoints.\n'
                      'Shared display range per channel; full fields, calibrated scale bars. Selection audit: Methods/Representative_Image_Selection.csv.',fontsize=8)
        figsave(fig,out,'Representative_Images')
        advance(progress,'Representative image figure saved',force=True)
    finally:
        plt.close(fig)


def treatment_ticks(ax,fontsize=8):
    ax.set_xticks(range(len(GROUPS)),LABELS,fontsize=fontsize,
                  rotation=45,ha='right',rotation_mode='anchor')


def stripbox(ax,df,col,ylabel,rng,log=False,point_alpha=None):
    for j,g in enumerate(GROUPS):
        vals=pd.to_numeric(df.loc[df.treatment_group==g,col],errors='coerce').to_numpy(float)
        vals=vals[np.isfinite(vals)]
        if not len(vals):continue
        ax.boxplot([vals],positions=[j],widths=.40,whis=1.5,showfliers=False,patch_artist=True,
                      boxprops={'facecolor':COLORS[j]+'25','edgecolor':COLORS[j]},medianprops={'color':'#222222','linewidth':1.5},
                      whiskerprops={'color':COLORS[j]},capprops={'color':COLORS[j]})
        ax.scatter(j+rng.uniform(-.16,.16,len(vals)),vals,s=SCATTER_SIZE,
                   color=COLORS[j],marker='o',
                   alpha=point_alpha if point_alpha is not None else (.65 if len(vals)<100 else .18),
                   edgecolors='none',rasterized=True,zorder=3 if point_alpha is not None else 1)
    treatment_ticks(ax)
    ax.set_ylabel(ylabel);ax.grid(axis='y',alpha=.16)
    numeric=pd.to_numeric(df[col],errors='coerce')
    if log and (np.isfinite(numeric)&(numeric>0)).any():ax.set_yscale('log')
    elif ax.get_ylim()[0]>0:ax.set_ylim(bottom=0)


def make_experiment_summary(out,mapping,image,cell,focus,p,progress=None):
    """One overview from saved measurements; each image/nucleus is used once."""
    print('Rendering experiment summary...',flush=True)
    channels=p['foci_channels'];paired=len(channels)==2
    rng=np.random.default_rng(p['random_seed_for_plot_jitter'])
    legend_labels=[]
    for group,label in zip(GROUPS,LABELS):
        images=image[image.treatment_group==group];nuclei=cell[cell.treatment_group==group]
        image_count=images.image_id.nunique()
        nucleus_count=nuclei.nucleus_id.nunique() if 'nucleus_id' in nuclei else len(nuclei)
        legend_labels.append(textwrap.fill(str(label),40)+f'\n{image_count:,} images · {nucleus_count:,} nuclei')
    legend_lines=max((label.count('\n')+1 for label in legend_labels),default=1)
    header_height=1.2+.3*math.ceil(len(GROUPS)/4)*legend_lines
    heights=[header_height]+[8]*len(channels)+([5] if paired else [])
    fig=plt.figure(figsize=(18,sum(heights)+.8),layout='constrained')
    try:
        blocks=np.atleast_1d(fig.subfigures(len(heights),1,height_ratios=heights))
        header=blocks[0].subplots();header.axis('off')
        header.text(.5,1,textwrap.fill(out.name,90),ha='center',va='top',fontsize=18,weight='bold',transform=header.transAxes)
        header.text(.5,1-.55/header_height,
                    f'{len(image)} images · {len(cell):,} usable nuclei · {len(GROUPS)} treatment groups',
                    ha='center',va='top',fontsize=11,transform=header.transAxes)
        handles=[Line2D([],[],color=COLORS[j],marker='o',linestyle='none',
                        label=legend_labels[j]) for j in range(len(GROUPS))]
        if handles:header.legend(handles=handles,loc='lower center',ncols=min(4,len(handles)),frameon=False,fontsize=9)
        rgb='input_format' in mapping and mapping.input_format.eq('rgb_export').any()
        intensity_unit='stored units' if rgb else 'ADU'
        for block,c in zip(blocks[1:],channels):
            block.suptitle(f'Channel {c} · {marker_name(p,c)}',fontsize=15,weight='bold')
            grid=block.add_gridspec(2,1)
            endpoint_grid=grid[0].subgridspec(1,4)
            specs=[('mean_foci_per_nucleus','Mean foci/nucleus','Foci per nucleus'),
                   ('mean_focus_area_um2','Mean focus area','Focus area (μm²)'),
                   ('mean_focus_intensity_adu','Mean focus intensity',f'Raw intensity ({intensity_unit})'),
                   ('percent_focus_positive_nuclei','Focus-positive nuclei','Focus-positive nuclei (%)')]
            for j,(key,title,label) in enumerate(specs):
                ax=block.add_subplot(endpoint_grid[0,j]);col=f'C{c}_{key}'
                stripbox(ax,image,col,label,rng,point_alpha=.25)
                ax.set_title(title);ax.set_xlim(-.5,len(GROUPS)-.5)
                # Wrapping keeps long combined treatment names within the panels.
                ax.set_xticklabels([textwrap.fill(str(label),24) for label in LABELS],
                                   rotation=45,ha='right',rotation_mode='anchor',fontsize=8)
                values=pd.to_numeric(image[col],errors='coerce').to_numpy(float)
                values=values[np.isfinite(values)]
                if key=='percent_focus_positive_nuclei':ax.set_ylim(0,110)
                else:ax.set_ylim(bottom=min(0,float(values.min())*1.1) if len(values) else 0,
                                 top=max(float(values.max())*1.22,.01) if len(values) else 1)
                if not len(values):ax.text(.5,.5,'No measurements',transform=ax.transAxes,ha='center',va='center')
            ecdf_grid=grid[1].subgridspec(1,3)
            ff=focus[focus.channel==c]
            for j,(df,key,label,log) in enumerate([
                    (cell,f'C{c}_foci_count','Foci per nucleus',False),
                    (ff,'focus_area_um2','Focus area (μm²)',True),
                    (ff,'raw_mean_adu',f'Mean focus intensity ({intensity_unit})',True)]):
                ax=block.add_subplot(ecdf_grid[0,j]);any_values=False;any_positive=False
                for k,g in enumerate(GROUPS):
                    values=pd.to_numeric(df.loc[df.treatment_group==g,key],errors='coerce').to_numpy(float)
                    values=np.sort(values[np.isfinite(values)])
                    if not len(values):continue
                    any_values=True;any_positive|=bool((values>0).any())
                    ax.step(values,np.arange(1,len(values)+1)/len(values),where='post',color=COLORS[k],
                            label=f'{textwrap.fill(str(LABELS[k]),32)} (n={len(values):,})',lw=1.8)
                ax.set_title('Distribution ECDF');ax.set_xlabel(label)
                ax.set_ylabel('Cumulative proportion');ax.set_ylim(0,1.02);ax.grid(alpha=.15)
                if any_values:ax.legend(fontsize=8,loc='lower right')
                else:ax.text(.5,.5,'No measurements',transform=ax.transAxes,ha='center',va='center')
                if log and any_positive:ax.set_xscale('log')
        if paired:
            ax=blocks[-1].subplots();first,second=channels;valid_count=0;undefined_fits=[]
            for j,g in enumerate(GROUPS):
                rows=cell[cell.treatment_group==g]
                x=pd.to_numeric(rows[f'C{first}_foci_count'],errors='coerce').to_numpy(float)
                y=pd.to_numeric(rows[f'C{second}_foci_count'],errors='coerce').to_numpy(float)
                valid=np.isfinite(x)&np.isfinite(y);valid_count+=int(valid.sum())
                if valid.any():ax.scatter(x[valid],y[valid],s=SCATTER_SIZE,alpha=.3,
                                          color=COLORS[j],edgecolors='none',rasterized=True)
                xv,yv=x[valid],y[valid]
                if len(xv)>=2 and np.ptp(xv)>0:
                    dx=xv-xv.mean()
                    slope=np.dot(dx,yv-yv.mean())/np.dot(dx,dx)
                    intercept=yv.mean()-slope*xv.mean()
                    fit_x=np.array([xv.min(),xv.max()])
                    ax.plot(fit_x,intercept+slope*fit_x,color=COLORS[j],lw=1.8,zorder=4)
                else:undefined_fits.append(str(LABELS[j]))
            ax.set_title('Foci counts across channels · one point per nucleus',fontsize=14)
            ax.set_xlabel(f'C{first} {marker_name(p,first)} · foci per nucleus')
            ax.set_ylabel(f'C{second} {marker_name(p,second)} · foci per nucleus')
            ax.set_xlim(left=0);ax.set_ylim(bottom=0);ax.grid(alpha=.15)
            if not valid_count:ax.text(.5,.5,'No measurements',transform=ax.transAxes,ha='center',va='center')
            if undefined_fits:
                note='No trend fit (fewer than two finite pairs or constant X): '+', '.join(undefined_fits)
                ax.text(.01,.98,textwrap.fill(note,110),transform=ax.transAxes,va='top',fontsize=8,
                        bbox={'facecolor':'white','edgecolor':'none','alpha':.85})
        caption=('Endpoint boxes: image-level median and IQR; whiskers: within 1.5 × IQR. Faint endpoint points: individual images.\n'
                 'ECDF counts use individual nuclei; area and intensity ECDFs use individual foci. Nuclei and foci are nested within images; descriptive comparisons only.')
        if paired:caption+='\nSolid treatment lines: ordinary least-squares trends over observed counts; no confidence bands or significance tests.'
        if rgb:caption+='\nRGB intensities are stored color-component values, not recovered detector ADU.'
        fig.supxlabel(caption,fontsize=9)
        figsave(fig,out,'Experiment_Summary')
        advance(progress,'Experiment summary saved',force=True)
    finally:
        plt.close(fig)


def make_plots(out,cell,focus,image,p,progress=None):
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
            advance(progress,f'Channel {c} · {fname} saved')
        ff=focus[focus.channel==c]
        metrics=[('focus_area_um2','Focus area (μm²)'),('raw_mean_adu','Mean focus intensity (ADU)'),
                 ('raw_max_adu','Maximum focus intensity (ADU)'),('raw_integrated_adu_px','Integrated focus intensity (ADU·px)'),
                 ('focus_to_background_ratio','Focus / local background'),('circularity','Focus circularity')]
        fig,axs=plt.subplots(2,3,figsize=(15,8.7),layout='constrained')
        for ax,(key,label) in zip(axs.flat,metrics):stripbox(ax,ff,key,label,rng,key in ['focus_area_um2','raw_integrated_adu_px'])
        fig.suptitle(f'Channel {c} · {name} · individual focus measurements\nDots are foci nested within nuclei and images. Boxes show median and IQR; no inference from pooled foci.',fontsize=13)
        figsave(fig,folder,'Focus_Properties')
        advance(progress,f'Channel {c} · focus properties saved')
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
            col=f'C{c}_{key}'
            stripbox(ax,image,col,label,rng,point_alpha=.25)
            ax.set_xlim(-.5,len(GROUPS)-.5)
            vals=pd.to_numeric(image[col],errors='coerce').to_numpy(float)
            vals=vals[np.isfinite(vals)]
            ax.set_ylim(bottom=min(0,float(vals.min())*1.1) if len(vals) else 0,
                        top=max(float(vals.max())*1.22,.01) if len(vals) else 1)
        fig.suptitle(f'Channel {c} · {name} · image-level comparisons\nBoxes: median and IQR; whiskers: observations within 1.5 × IQR. Faint points: individual images.\nDescriptive comparisons; nuclei are nested observations.',fontsize=14)
        figsave(fig,folder,'Image_Level_Endpoints')
        advance(progress,f'Channel {c} · image-level endpoints saved')
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
        advance(progress,f'Channel {c} · distribution curves saved')
        fig,ax=plt.subplots(figsize=(7.5,5),layout='constrained')
        stripbox(ax,image,f'C{c}_percent_focus_positive_nuclei','Focus-positive nuclei (%)',rng,point_alpha=.25)
        treatment_ticks(ax,fontsize=10);ax.set_xlim(-.5,len(GROUPS)-.5);ax.set_ylim(0,110);ax.set_ylabel('Focus-positive nuclei (%)');ax.grid(axis='y',alpha=.2)
        ax.set_title(f'Channel {c} · {name} · focus-positive nuclei\nBoxes: median and IQR; whiskers: within 1.5 × IQR.\nFaint points: individual images; no biological replicates')
        figsave(fig,folder,'Focus_Positive_Nuclei')
        advance(progress,f'Channel {c} · focus-positive nuclei saved',force=True)
    mapping=pd.read_csv(out/'Methods'/'Treatment_Group_Mapping.csv')
    make_input_intensity_plots(out/'QC',mapping,p,progress=progress)


def input_intensity_groups(mapping,c):
    """Use stable indices rather than treatment names in exported filenames."""
    for j,g in enumerate(mapping.treatment_group.unique()):
        yield f'C{c}_Input_Intensity_Group_{j+1:03d}',g,mapping[mapping.treatment_group==g]


def input_intensity_distributions(mapping,c,p,progress=None):
    """Retain only small histogram arrays; read one source image at a time."""
    lower,upper=1.,1.
    records=[]
    for _,m in mapping.iterrows():
        a=read_channels(m.source_path,p)[c-1]
        positive=a[a>0]
        if positive.size:lower=min(lower,float(positive.min()))
        upper=max(upper,256 if a.dtype==np.uint8 else 7500,float(a.max())+1)
        records.append({'image_id':m.image_id,'treatment_group':m.treatment_group,
                        'mean_intensity':float(np.mean(a,dtype=np.float64)),
                        'all_zero':bool(np.all(a==0))})
        del a,positive
        advance(progress,f'{image_display_name(m)} · Channel {c} · intensity range inspected')
    bins=np.geomspace(lower,upper,130)
    for record,(_,m) in zip(records,mapping.iterrows()):
        a=read_channels(m.source_path,p)[c-1]
        counts=np.histogram(a.ravel(),bins=bins)[0]
        # Normalize by ALL pixels, including the zero mass omitted on a log axis.
        record['density']=counts/(a.size*np.diff(bins))
        del a
        advance(progress,f'{image_display_name(m)} · Channel {c} · intensity distribution calculated')
    return bins,records


def make_input_intensity_plots(qdir,mapping,p,progress=None):
    if mapping.empty:return
    for c in p['foci_channels']:
        print(f'Rendering per-treatment input intensity plots for channel {c}...',flush=True)
        bins,records=input_intensity_distributions(mapping,c,p,progress=progress)
        groups=[]
        for stem,g,fields in input_intensity_groups(mapping,c):
            images=[r for r in records if r['treatment_group']==g]
            groups.append((stem,g,images,np.mean([r['density'] for r in images],axis=0),
                           float(np.mean([r['mean_intensity'] for r in images]))))
        # Include the average curves when finding shared limits: averaging can
        # reduce a nonzero density below every individual image's nonzero bins.
        densities=[r['density'] for r in records]+[group[3] for group in groups]
        positive=np.concatenate([d[d>0] for d in densities])
        ylim=(float(positive.min())/2,float(positive.max())*2) if positive.size else (1e-6,1)
        xmin=min([bins[0]]+[mean for _,_,_,_,mean in groups if mean>0])*.9
        for j,(stem,g,images,average,mean) in enumerate(groups):
            color=COLORS[j%len(COLORS)]
            fig,ax=plt.subplots(figsize=(9,5),layout='constrained')
            try:
                ax.set_xscale('log');ax.set_yscale('log')
                ax.set_xlim(xmin,bins[-1]);ax.set_ylim(*ylim)
                for image in images:
                    ax.stairs(image['density'],bins,baseline=None,color=color,alpha=.25,lw=1)
                ax.stairs(average,bins,baseline=None,color=color,lw=2.5)
                if mean>0:ax.axvline(mean,color=color,linestyle='--',lw=1.5)
                handles=[Line2D([],[],color=color,alpha=.25,lw=1,label=f'Individual images (n={len(images)})'),
                         Line2D([],[],color=color,lw=2.5,label='Average distribution'),
                         Line2D([],[],color=color,linestyle='--',lw=1.5,label=f'Mean intensity: {mean:.4g} ADU')]
                ax.legend(handles=handles,fontsize=9)
                zeros=sum(r['all_zero'] for r in images)
                if zeros==len(images):
                    ax.text(.5,.5,'All images contain only zero-intensity pixels.\nMean intensity: 0 ADU (outside logarithmic axes).',
                            transform=ax.transAxes,ha='center',va='center')
                elif zeros:
                    ax.text(.02,.96,f'{zeros} all-zero image(s); zero intensity is outside logarithmic axes.',
                            transform=ax.transAxes,va='top',fontsize=8)
                label=textwrap.fill(str(g).replace('ART558_20PDS','ART558 + 20PDS'),width=70)
                ax.set_title(f'Channel {c} · {marker_name(p,c)} · input intensity distribution\n{label} · {len(images)} image(s)')
                ax.set_xlabel('Raw pixel intensity (ADU)');ax.set_ylabel('Probability density')
                fig.supxlabel('Each image has equal weight. Zero pixels contribute to normalization and mean,\nbut are omitted from the logarithmic intensity axis.',fontsize=8)
                figsave(fig,qdir,stem)
                advance(progress,f'Channel {c} · treatment intensity figure {j+1}/{len(groups)} saved')
            finally:
                plt.close(fig)


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
      'group_names':'JSON list of matched filename labels retained as metadata. Comparisons use the combined treatment_group. Empty list means unmatched.',
      'number_images_in_treatment':'Number of images with the same complete combination of group labels.',
      'replicate_id':'Missing for this proof of concept, as confirmed by user.',
      'nucleus_id':'Globally unique image ID plus candidate nucleus label; matches nuclear TIFF label.',
      'nucleus_label':'Integer object value in the nuclear label TIFF; IDs can have gaps after exclusions.',
      'nuclear_threshold_adu':'Lower nuclear foreground threshold in stored intensity units; adaptive uses smoothed intensity > threshold, fixed_raw uses original intensity >= threshold.',
      'DAPI_threshold_adu':'Legacy alias of nuclear_threshold_adu for nuclear channels labeled DAPI.',
      'nuclear_threshold_upper_adu':'Inclusive upper bound on original nuclear-channel intensities in fixed_raw mode; missing in adaptive mode. Closing and hole filling follow foreground selection.',
      'nuclear_threshold_method':'adaptive estimates a smoothed-intensity cutoff per image; fixed_raw uses the configured inclusive range on original intensities.',
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
    if s.startswith('adaptive_split_'):
        return channel+'Adaptive secondary-seed pruning setting/statistic for this nucleus. Applied_Thresholds.csv records brightness/area gates, relative prominence fraction, seeds tested/suppressed and maximum tested prominence. The baseline absolute prominence is recorded separately.'
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
    tables.update({f'Colocalization_{name}':table for name,table in read_colocalization_tables(out,p).items()})
    dictionary=[]
    for sheet,df in tables.items():
        for col in df:
            desc=(colocalization_outputs.definition(col) if sheet.startswith('Colocalization_') else None)
            dictionary.append({'table':sheet,'column':col,'data_type':str(df[col].dtype),'definition':desc or definition(col)})
    pd.DataFrame(dictionary).to_csv(out/'Methods'/'Data_Dictionary.csv',index=False)
    # JSON matrix for the required spreadsheet authoring library, preserving numeric types and missing values.
    wb={name:{'columns':list(df.columns),'rows':json.loads(df.to_json(orient='values'))} for name,df in tables.items()}
    for name,df in [('Analysis_Parameters',pd.DataFrame(par)),('Treatment_Mapping',mapping),('Data_Dictionary',pd.DataFrame(dictionary))]:
        wb[name]={'columns':list(df.columns),'rows':json.loads(df.to_json(orient='values'))}
    (out/'Methods'/'workbook_data.json').write_text(json.dumps(wb,allow_nan=False), encoding='utf-8')
    versions=json.loads((out/'Methods'/'Run_Manifest.json').read_text(encoding='utf-8'))
    requirements=[f'numpy=={versions["numpy"]}',f'scipy=={versions["scipy"]}',f'scikit-image=={versions["skimage"]}',
                  f'pandas=={versions["pandas"]}',f'tifffile=={versions["tifffile"]}',f'matplotlib=={matplotlib.__version__}']
    if versions.get('imagecodecs'):
        requirements.append(f'imagecodecs=={versions["imagecodecs"]}')
    (out/'Methods'/'requirements.txt').write_text('\n'.join(requirements)+'\n', encoding='utf-8')
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
    np_cfg=p['nuclear'];fp=p['foci'];fixed=np_cfg.get('fixed_threshold_adu')
    hole_description='internal holes are filled' if np_cfg.get('fill_holes',True) else 'internal holes are retained'
    disk_description=(f'approximate disk (scikit-image {fp["background_disk_decomposition"]} decomposition'
                      if fp['background_disk_decomposition'] else 'exact disk (no decomposition')
    watershed_description=('Watershed boundaries are left unassigned so adjacent objects remain separable.'
                           if fp['watershed_lines'] else 'Watershed boundaries are assigned to adjacent basins.')
    sensitivity_factors=' and '.join(f'{v:g}' for v in fp['sensitivity_threshold_factors'] if v!=1)
    adaptive_channels=[]
    for c in p['foci_channels']:
        q=fp[f'channel_{c}'];fraction=q.get('adaptive_split_prominence_fraction',0)
        if fraction:
            adaptive_channels.append(f"C{c}: fraction {fraction:g}, minimum corrected peak {q.get('adaptive_split_minimum_peak_adu',100):g} ADU, minimum connected support area {q.get('adaptive_split_minimum_area_um2',2):g} µm², merged-basin trimming at {100*q.get('adaptive_split_post_watershed_peak_fraction',fp['post_watershed_peak_fraction']):g}% of its corrected peak")
    adaptive_description=('Adaptive splitting is enabled for '+('; '.join(adaptive_channels))+'. '
                          'After the usual thresholds establish candidate peaks, eligible secondary peaks in sufficiently large connected support regions must survive h-maxima at max(absolute prominence, fraction times their own corrected peak intensity). '
                          'Each support region retains its strongest original seed; peaks below the brightness cutoff and smaller regions retain the original rule. '
                          'An optional adaptive_split_post_watershed_peak_fraction overrides the normal basin trimming only for basins that receive suppressed secondary seeds; the support floor and connected-component rule still apply. '
                          'Applied_Thresholds.csv and cell summaries record enabled settings, secondary peaks tested/suppressed and the maximum tested prominence. '
                          'prominence_threshold_adu remains the absolute baseline; the adaptive criterion only suppresses secondary seeds before watershed and does not raise the common detection floor. '
                          'Threshold sensitivity changes the original absolute thresholds while these adaptive settings stay fixed.'
                          if adaptive_channels else
                          'Adaptive focus splitting is disabled for all selected channels (missing or zero adaptive_split_prominence_fraction); the original absolute-prominence rule is used.')
    if fixed is not None:
        nuclear_threshold_description=(f'Initial nuclear foreground uses the fixed inclusive range {fixed[0]:g}–{fixed[1]:g} ADU '
                                       'on the original nuclear-channel pixels, identically for every image. '
                                       'Gaussian intensity smoothing and background subtraction are not used for foreground selection. '
                                       'The recorded smoothed background median and MAD are diagnostics only. '
                                       'The lower and upper bounds and fixed_raw mode are recorded in the nucleus audit and cell table.')
    else:
        nuclear_threshold_description=(f'Gaussian smoothing (sigma {np_cfg["gaussian_sigma_px"]:g} px) is used only for detection. '
                                       'The foreground threshold is the full-image smoothed nuclear-channel median plus '
                                       f'max({np_cfg["threshold_floor_offset_adu"]:g} ADU, {np_cfg["threshold_mad_multiplier"]:g} x robust MAD sigma); '
                                       'nuclei occupy a minority of these fields. There is no upper intensity cutoff.')
    methods=f'''# Reproducible 2D nuclear foci quantification

## Scope and experimental units
This report covers {len(mapping)} TIFF images, with foci measurements for {channels}. Nuclear segmentation uses {nuclear}. Channel identities come from the saved analysis parameters. Image IDs identify image observations, not biological replicates. Treatment tokens and replicate metadata are retained in Treatment_Group_Mapping.csv.

Inputs are read as calibrated 2D channel planes in CYX order. Source layout, channel mapping, bit depth, pixel sizes, calibration source, and source paths are recorded per image in Treatment_Group_Mapping.csv and Image_Metadata.json. RGB exports are reordered only when an explicit rgb_channel_order is supplied. Calibration priority is an explicit pixel_size_um override, then OME PhysicalSizeX/PhysicalSizeY, then ImageJ micron metadata, then TIFF inch/centimeter resolution. Areas, axes and distances use both X and Y calibration. These scripts do not modify the source TIFFs. Run_Manifest.json records the analysis selections; Output_Parameters.json records the selections included in this report.

{rgb_note}

Configured group_names use case-insensitive literal substring matching against the filename without its extension. Every matching name is retained in the group_names JSON-list column; treatment_group displays their combination in configuration order. Without configured names, the legacy filename parser is used. Unmatched images remain AMBIGUOUS. Every comparison uses treatment_group as its category: each image, nucleus, and focus contributes once to its combined treatment group. number_images_in_treatment counts images with the same full combination of labels.

## Nuclear segmentation
{nuclear_threshold_description} Closing uses a {np_cfg['closing_disk_radius_px']:g} px disk and {hole_description}. Components below {np_cfg['candidate_min_area_px']:,} px are discarded as debris before the candidate list. A distance transform smoothed at sigma {np_cfg['distance_smoothing_sigma_px']:g} px supplies h-maxima (height {np_cfg['distance_h_maxima_px']:g} px, edge distance >{np_cfg['minimum_seed_distance_to_edge_px']:g} px) for watershed splitting. A connected candidate without such a seed receives its greatest-distance pixel as a seed, allowing border fragments to be audited.

Candidates from {np_cfg['accepted_min_area_um2']:g} to {np_cfg['accepted_max_area_um2']:g} square microns are retained unless their bounding box intersects a {np_cfg['image_edge_buffer_px']:g} px image-edge buffer. This deliberately excludes all border-touching nuclei, including potentially mildly truncated ones. Exclusion percentage uses the post-debris, post-watershed candidate count as denominator. Nuclear masks retain candidate label IDs. Objects below the debris prefilter are not included in this denominator. Nuclear solidity <{np_cfg['shape_warning_solidity_below']:g} or aspect ratio >{np_cfg['shape_warning_aspect_ratio_above']:g} generates an inspection warning only; such nuclei remain in the analysis. Nucleus_Inclusion_Audit.csv records every candidate and exclusion reason. No manual mask edits were made.

## Foci segmentation
The same numerical parameters and automatic rule apply to every image within a channel. Thresholds and segmentation settings come from the saved analysis parameters. This is exploratory segmentation, not a blinded or ground-truth-validated detector.

Original intensities are lightly smoothed (Gaussian sigma {fp['gaussian_sigma_px']:g} px). Background is a grayscale opening with a radius {fp['background_disk_radius_px']:g} px {disk_description}; physical size depends on pixel calibration). The difference between smooth image and opening supplies the detection signal. Structures larger than this background scale can be attenuated; the result measures discrete puncta rather than diffuse or pan-nuclear staining.

Per-nucleus noise sigma is 1.4826 x MAD of (I00 - I10 - I01 + I11)/2 on 2x2 windows fully inside the nucleus. Approximate noise after Gaussian smoothing is sigma_raw / (2 sqrt(pi) sigma_gaussian). This estimate is a reproducible noise proxy, not a calibrated photon model or statistical error rate.

Support threshold = max(channel floor, {fp['support_noise_multiplier']:g} x sigma_smooth). Seed threshold = max(channel floor, {fp['seed_noise_multiplier']:g} x sigma_smooth). Peak prominence = max(channel floor, {fp['prominence_noise_multiplier']:g} x sigma_smooth). Selected channel floors are {floors}. Applied per-nucleus values are saved in Applied_Thresholds.csv. H-maxima establish seeds; watershed on the negative corrected image segments actual pixel objects inside support masks. Local maxima alone are never counted. {watershed_description} Each basin is trimmed at max(support threshold, {100*fp['post_watershed_peak_fraction']:g}% of its corrected peak), and only the connected component containing that peak is retained.

{adaptive_description}

Objects require at least {fp['minimum_area_px']:g} pixels (physical area depends on pixel calibration), at most {fp['maximum_area_um2']:g} square microns, and raw mean minus local background >={fp['net_mean_noise_multiplier']:g} x per-nucleus raw noise sigma. No shape-based focus exclusion is applied. Dense neighboring foci with insufficient prominence remain merged. The detector may miss faint objects below the common noise/contrast limits. Focus_Candidate_Audit.csv records size and contrast exclusions; subthreshold unseeded signal does not enter this audit. Counted foci are constrained to usable nuclei. Boundary-touching focus objects are retained with a flag. Excluded nuclei and extranuclear puncta are not part of the per-nucleus primary counts.

## Local background and measurements
The local background region is an annulus at Euclidean pixel distances >{fp['annulus_inner_gap_px']:g} and <={fp['annulus_outer_radius_px']:g} from the segmented object, limited to its own nucleus and excluding all provisional foci dilated by 1 px. Provisional includes objects later rejected by size/contrast, avoiding their reuse as background. At least {fp['annulus_min_pixels']:g} annular pixels are required. Otherwise all eligible nonfocus pixels in the same nucleus are used; if still insufficient, the opening background over the object is used. Background source and sample size are preserved.

Measurements use the original unsmoothed pixels. Raw mean/median/min/max/sum, opening-corrected versions, and local-background-corrected versions are saved. Corrected pixels are not clipped to zero. Integrated intensity is ADU x pixel, not calibrated molecular abundance. Focus/background = raw mean / local median. Local signal/background = (raw mean - local median)/local median. Local contrast SNR uses the annulus robust MAD sigma in the denominator, distinct from the noise estimate used in filtering. Dataset background is measured outside all candidate nuclei after {p['qc']['extranuclear_background_exclusion_dilation_px']:g} px dilation.

Equivalent diameter is the diameter of a circle of equal segmented area. Major/minor axes derive from region second moments; calibrated axes are recomputed after scaling X and Y. Circularity uses 4*pi*area/perimeter_Crofton^2 (four directions); pixelation may yield values >1, and these are retained. Small-focus morphology is resolution limited. Solidity is area/convex area. Unweighted centroids use zero-based pixel centers, X=column and Y=row. Relative radial position is d_centroid / (d_centroid + nearest_edge_distance), in physical units. It is 0 at the centroid and approaches 1 at the boundary; it is not a ray-based or equal-area nuclear coordinate.

For each nucleus, per-focus intensity means and medians summarize the focus means with equal weight per focus. Maximum focus mean and maximum focus pixel are separate columns. Foci counts, density, total areas, occupied fraction and summed intensity are zero for focus-negative nuclei. Per-focus size/intensity/morphology averages are missing when there are no foci. Image summaries include every usable nucleus, including zero-focus nuclei. Image mean density and area fraction average nucleus-specific values; they are not ratios of pooled totals. SD is a sample SD (ddof=1) across nuclei and does not represent biological-replicate uncertainty.

## QC and sensitivity
Annotated_QC figures show paired raw and overlay views plus label masks for each channel. {detailed_qc} Yellow marks counted foci, cyan accepted nuclear boundaries and red excluded candidates. TIFF mask values map exactly to the result label columns. Raw/overlay pairs share display limits; per-image QC contrast is for inspection only. Shared_Contrast figures use one range per channel across all images. Display clipping is distinct from detector saturation and does not change measurements.

Input intensity figures are exported separately for each combined treatment_group and selected channel as QC/C{{channel}}_Input_Intensity_Group_{{index:03d}}.png and .svg. Group indices follow the image mapping's treatment order, and colors match the comparison figures. Faint curves (alpha 0.25) represent individual images; the thicker solid curve is the arithmetic mean of their probability densities. The dashed vertical line is the arithmetic mean of their raw mean pixel intensities. Each image receives equal weight in both averages, regardless of its dimensions. Bins and logarithmic axis limits are shared across groups within each channel. Each image's histogram counts are divided by its total pixel count and bin widths, including zero pixels in the normalization; zero intensity is omitted from the logarithmic axis but included in the raw mean. Thus the density integral can be below one when zero pixels are present. All-zero images contribute zero density and zero mean, with an annotation instead of a vertical line at zero. Legends contain only three entries. Scatter markers use one fixed small size (8 points squared), and treatment-axis labels are rotated 45 degrees and right aligned.

Treatment colors use the first 32 colors of Colorcet's Glasbey dark palette, designed for white backgrounds, assigned in the saved image mapping's treatment order. Boxes, points, distribution curves and input-intensity figures share those assignments across channels. Colors repeat only after 32 treatment groups.

Saturation uses each input dtype's maximum stored value: 255 for uint8, 65535 for uint16. For RGB exports this measures clipping of exported color components, not detector saturation. The source files do not necessarily specify detector effective bit depth; absence of container saturation cannot prove absence of detector clipping. Per-image saturation values are in Treatment_Group_Mapping.csv. Observed maxima and saturation fractions should be reviewed in Input_Intensity_Distributions.csv and the saved QC tables.

QC_Summary.csv reports fixed numerical and dataset-relative flags, with exact cutoffs in Output_Parameters.json. These include nuclear counts/exclusions, shape concerns, background, saturation, count extremes, size extremes, intensity tails, contrast, density and threshold sensitivity. Flags request inspection and never exclude images. Dataset-relative heuristics have limited reliability with few images. Genuine treatment biology can trigger a flag.

Threshold sensitivity reruns each nucleus at common factors {sensitivity_factors} applied to support, seed, prominence and net-mean contrast thresholds, with all other rules fixed. It is reported separately from the primary factor-1 masks and tables. Because splitting and contrast checks interact, counts need not be strictly monotonic with threshold factor. Sensitivity quantifies technical dependence; it does not estimate detection accuracy or a confidence interval.

## Treatment comparisons and limits
The experimental structure is focus -> nucleus -> image -> treatment. The reports are descriptive: no hypothesis tests, p-values, biological confidence intervals or multiple-comparison correction are reported. Replicate structure must be assessed from the experiment metadata. Cellular/focus distribution plots visualize heterogeneity only. Image-level plots preserve the experimental hierarchy. The observed treatment ranking cannot establish a reproducible biological treatment effect.

Image_Level_Endpoints and Focus_Positive_Nuclei use saved image-level measurements, grouped by combined treatment_group. Boxes show the median and interquartile range (IQR); whiskers extend to the most extreme observations within 1.5 times the IQR of the quartiles. Separate outlier symbols are hidden because every valid image appears once per metric as a circular point (size 8 points squared, alpha 0.25), with reproducible horizontal jitter within ±0.16 of its treatment position. Points have no image ID, count or percentage annotations. Missing and nonfinite values are omitted per metric while their treatment categories remain visible. A single-image group has a collapsed box and one point. Treatment labels are rotated 45 degrees and right aligned; the separate focus-positive percentage axis spans 0–110%.

Experiment_Summary.png and Experiment_Summary.svg combine four saved image-level endpoints and three ECDFs per selected foci channel. ECDFs use all finite nucleus counts (including zero-focus nuclei), individual focus areas, and individual focus mean raw intensities. Area and intensity ECDF axes are logarithmic when positive observations exist. Exactly two selected foci channels add a scatter of their paired counts from each usable nucleus row, colored by combined treatment_group; zero counts are retained and pairs with nonfinite counts are omitted. Ordinary least-squares lines summarize each treatment over its observed X range, requiring at least two finite pairs with distinct X values. Undefined fits are noted; no confidence bands, correlation statistics, or significance tests are added. The top legend counts all image observations and accepted nuclei per treatment, independently of finite scatter pairs. Summary generation uses saved tables without reading raw images or repeating segmentation.

Representative_Images.png and Representative_Images.svg show up to two eligible full-field images per combined treatment, using the same images for all selected channels. Selection uses the four image-level endpoints in the experiment summary for every selected foci channel. Each metric's arithmetic mean and sample standard deviation include all finite image values, including outliers; metrics with fewer than two values or zero standard deviation are ignored. Eligible images are ordered by the number of available usable metrics (descending), then root-mean-square standardized distance from the group means (ascending), then original filename and internal ID. Images without usable nuclei, images flagged as within-treatment outliers, and images with no usable endpoint values are ineligible. All-constant metrics give equal distances and deterministic filename ties. Partial-metric selections and missing representative slots are identified explicitly. Methods/Representative_Image_Selection.csv records scores, coverage, ranks, eligibility, selections, and group means/scales.

Within-treatment image outlier review applies Q1 - 1.5 x IQR and Q3 + 1.5 x IQR independently to those endpoints, with NumPy linear quantiles. A metric requires at least five finite image values and positive IQR. Boundary-equal values are not flagged; missing values, insufficient sample sizes, and zero IQR are recorded as unassessed. QC/Image_Outlier_Review.csv records every assessment and its fences. An image flagged on any selected-channel endpoint is ineligible as a representative but remains in every measurement table, group mean, and statistical plot. Existing numerical QC warnings remain separate. These are descriptive manual-review flags, not automatic exclusions from analysis.

Representative panels and shared-contrast QC use the same dataset-wide sampled 1st–99.9th percentile display ranges per channel. Representatives include the nuclear channel, retain full fields and physical aspect ratio, and use bounded display copies plus calibrated scale bars. Display values are not measurement transforms. QC/Representative_Display_Ranges.csv records their limits. Original filenames identify images in figures, reports, and progress; I-style IDs and mask/nucleus/focus keys remain stable internally for compatibility.

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
    if colocalization_settings(p)['enabled']:
        methods=methods.replace('No cross-channel colocalization or biological comparison is performed.',
                                'Cross-channel colocalization is described below; no inferential biological comparisons are performed.')
        methods+=colocalization_outputs.methods_text(p)
    (out/'Methods'/'Methods.md').write_text(methods, encoding='utf-8')


def report(out,mapping,images,cell,focus,qc,p,outliers=None):
    def esc(x):return html.escape(str(x))
    names={m.image_id:image_display_name(m) for _,m in mapping.iterrows()}
    if outliers is None:outliers,_=review_images(mapping,images,p)
    flagged=outliers[outliers.outlier] if len(outliers) else outliers
    flagged_ids=set(flagged.image_id) if len(flagged) else set()
    detailed=p.get('qc_output','detailed')=='detailed'
    channels=p['foci_channels']
    nuclear=f"C{p['nuclear_channel']} ({marker_name(p,p['nuclear_channel'])})"
    cols=['image_id','treatment_group','nuclei_analyzed']
    labels=['Image','Treatment','Nuclei']
    for c in channels:
        cols.extend([f'C{c}_total_foci',f'C{c}_mean_foci_per_nucleus'])
        labels.extend([f'C{c} {marker_name(p,c)} foci',f'C{c} mean / nucleus'])
    summary=images[cols].copy();summary.columns=labels
    summary['Image']=images.image_id.map(names).fillna(images.image_id)
    totals=' · '.join(f'C{c} {marker_name(p,c)}: {int((focus.channel==c).sum()):,} foci' for c in channels)
    body='''<!doctype html><html><head><meta charset="utf-8"><title>Foci analysis review</title><style>
body{font:16px/1.55 system-ui,Arial;color:#203043;background:#f7f9fb;max-width:1250px;margin:auto;padding:35px}h1{font-size:32px}h2{margin-top:2.2em}a{color:#006b9c}table{border-collapse:collapse;background:white;width:100%;font-size:14px}th,td{padding:10px;border-bottom:1px solid #dbe2e9;text-align:left}th{background:#243c50;color:white}img{max-width:100%;height:auto}article{background:white;padding:22px;margin:18px 0;border:1px solid #dbe2e9;border-radius:8px}.treatment-images{background:white;padding:18px 22px;margin:14px 0;border:1px solid #dbe2e9;border-radius:8px}.note{border-left:4px solid #d69132;padding:12px 18px;background:#fff6e8}.gallery{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:10px}.gallery img{width:100%}summary{cursor:pointer;font-weight:600}small{color:#526779}.links a{margin-right:18px}li{margin-bottom:7px}</style></head><body>'''
    body+=f'<h1>Fluorescent foci quantification</h1><p>{len(mapping)} images · {mapping.treatment_group.nunique()} groups · {len(cell)} usable nuclei · {esc(totals)}</p>'
    body+='<p class="links"><a href="Experiment_Summary.png">Experiment summary (PNG)</a><a href="Experiment_Summary.svg">Experiment summary (editable SVG)</a></p>'
    body+='<p class="links"><a href="Representative_Images.png">Representative images (PNG)</a><a href="Representative_Images.svg">Representative images (SVG)</a><a href="Methods/Representative_Image_Selection.csv">Selection audit</a></p>'
    body+='<p class="note">All comparisons are descriptive. Nuclei and foci are nested within images. Detected objects and 2D areas require QC review; dense and faint signal remain sources of uncertainty.</p>'
    if 'input_format' in mapping and mapping.input_format.eq('rgb_export').any():
        body+='<p class="note">RGB export analysis: channels use the saved color mapping. Intensities are stored color-component values, not recovered detector ADU. Export display settings and clipping can affect measurements; treat these results as exploratory.</p>'
    if p.get('group_names'):
        body+='<p>Comparisons use combined treatment groups. Each image, nucleus, and focus contributes once to its treatment_group; group_names retains the matched filename labels as metadata.</p>'
    body+='<p class="links"><a href="Methods/Methods.md">Detailed methods</a><a href="Methods/Output_Parameters.json">Report parameters</a><a href="Methods/Treatment_Group_Mapping.csv">Image mapping</a><a href="QC/QC_Summary.csv">QC summary CSV</a></p>'
    body+=summary.to_html(index=False,float_format=lambda x:f'{x:,.2f}',border=0)
    body+=f'<p>Nuclear segmentation: {esc(nuclear)}. Marker labels come from the analysis configuration. Border nuclei are excluded; focus-negative nuclei remain in summaries. Per-image calibration is recorded in the image mapping.</p>'
    qc_footer='<details id="qc-findings" class="treatment-images"><summary>QC Findings</summary><p>Flags identify inspection targets and never exclude images. Full reasons, thresholds and sensitivity counts are in the QC table.</p>'
    qc_display=qc[['image_id','channel','number_usable_nuclei','number_excluded_nuclei','manual_inspection_warnings']].copy()
    qc_display['image_id']=qc_display.image_id.map(names).fillna(qc_display.image_id)
    qc_footer+=qc_display.rename(columns={'image_id':'Image'}).to_html(index=False,border=0)+'</details>'
    review_footer=f'<details id="flagged-image-review" class="treatment-images"><summary>Flagged image review · {len(flagged_ids)} image(s)</summary>'
    review_footer+=f'<p class="note"><strong>{len(flagged_ids)} image(s) flagged for within-treatment outlier review.</strong> Flags use 1.5 × IQR fences with at least five finite images and positive IQR per endpoint. Flagged images remain in all measurement summaries and graphs; they are ineligible as representatives.</p>'
    assessed=int((outliers.assessment_status=='assessed').sum()) if len(outliers) else 0
    if not assessed:review_footer+='<p>No endpoints could be assessed for outliers: groups had insufficient finite values or zero IQR. This does not establish that the images are free of outliers.</p>'
    review_footer+='<p><a href="QC/Image_Outlier_Review.csv">Full outlier assessment, including unassessed endpoints (CSV)</a></p>'
    if len(flagged):
        rows=[]
        for _,r in flagged.iterrows():
            rows.append({'Image':f'<a href="QC/{esc(r.image_id)}_Annotated_QC.png">{esc(names.get(r.image_id,r.image_name))}</a>',
                         'Treatment':esc(r.treatment_group),'Channel':int(r.channel),'Endpoint':esc(r.metric),
                         'Observed':f'{r.value:.5g}','Lower fence':f'{r.lower_fence:.5g}','Upper fence':f'{r.upper_fence:.5g}'})
        review_footer+='<details><summary>Review flagged images and reasons</summary><div style="overflow-x:auto">'+pd.DataFrame(rows).to_html(index=False,border=0,escape=False)+'</div></details>'
    assessment_columns=['image_name','treatment_group','channel','metric','value','finite_images',
                        'q1','q3','lower_fence','upper_fence','assessment_status','outlier']
    full_assessment=outliers.reindex(columns=assessment_columns).copy()
    if len(outliers):full_assessment['image_name']=outliers.image_id.map(names).fillna(full_assessment.image_name)
    full_assessment=full_assessment.rename(columns={'image_name':'Image','treatment_group':'Treatment',
        'channel':'Channel','metric':'Endpoint','value':'Observed','finite_images':'Finite images',
        'q1':'Q1','q3':'Q3','lower_fence':'Lower fence','upper_fence':'Upper fence',
        'assessment_status':'Status','outlier':'Flagged'})
    review_footer+='<details id="full-outlier-assessment"><summary>Full outlier assessment</summary>'
    review_footer+='<div style="overflow-x:auto">'+full_assessment.to_html(index=False,border=0,na_rep='—',float_format=lambda v:f'{v:.5g}')+'</div></details></details>'
    body+='<h2>Treatment comparisons within each channel</h2>'
    for c in p['foci_channels']:
        body+=f'<article><h3>Channel {c} · '+esc(marker_name(p,c))+'</h3>'
        body+=f'<a href="Plots/Channel_{c}/Image_Level_Endpoints.png"><img loading="lazy" src="Plots/Channel_{c}/Image_Level_Endpoints.png"></a>'
        body+='<p>Image-level comparisons and Focus Positive Nuclei: boxes show the median and interquartile range; whiskers reach observations within 1.5 × IQR of the quartiles. Faint circular points show individual images, including outliers, once per valid metric. Missing or nonfinite values are omitted.</p>'
        for f in ['Cellular_Abundance_Size','Cellular_Intensity_Contrast','Cellular_Morphology','Focus_Properties','Distribution_ECDF','Focus_Positive_Nuclei']:
            body+=f'<p><a href="Plots/Channel_{c}/{f}.png">{f.replace("_"," ")}</a> · <a href="Plots/Channel_{c}/{f}.svg">editable SVG</a></p>'
        body+='<details><summary>Shared-contrast raw fields</summary>'
        for page,(filename,_) in enumerate(shared_contrast_pages(mapping,c),1):
            body+=f'<p>Page {page}</p><a href="QC/{filename}"><img loading="lazy" src="QC/{filename}" alt="Channel {c} shared contrast, page {page}"></a>'
        body+='</details>'
        histograms=[(stem,g) for stem,g,_ in input_intensity_groups(mapping,c) if (out/'QC'/f'{stem}.png').exists()]
        if histograms:
            body+='<details><summary>Input intensity distributions by treatment</summary><p>Faint curves show individual images; the solid curve is their average distribution and the dashed line marks their mean intensity. Each image has equal weight. Groups share the same axes within this channel.</p>'
            for stem,g in histograms:
                body+=f'<h4>{esc(g)}</h4><a href="QC/{stem}.png"><img loading="lazy" src="QC/{stem}.png" alt="Channel {c} input intensities: {esc(g)}"></a>'
                body+=f'<p><a href="QC/{stem}.png">PNG</a> · <a href="QC/{stem}.svg">editable SVG</a></p>'
            body+='</details>'
        body+='</article>'
    body+='<h2>Files and reproducibility</h2><ul>'
    for f in ['Results/Image_Summary.csv','Results/Cell_Summary.csv','Results/Focus_Data.csv','QC/Nucleus_Inclusion_Audit.csv','QC/Focus_Candidate_Audit.csv','QC/Threshold_Sensitivity.csv','QC/Input_Intensity_Distributions.csv','Methods/Applied_Thresholds.csv','Methods/Data_Dictionary.csv','Methods/Run_Manifest.json','Methods/Validation_Report.json','Methods/Methods.md']:
        body+=f'<li><a href="{f}">{f}</a></li>'
    body+='</ul><p>'+('' if colocalization_settings(p)['enabled'] else 'No cross-channel colocalization analysis was performed. ')
    body+='Thresholds and object definitions were shared across treatments. Run parameters and provenance are in Methods/; reusable tools are kept separately in the scripts folder.</p>'
    body+=colocalization_outputs.report_section(out,p,read_colocalization_tables(out,p))
    body+='<h2>Inspect every image</h2><p>Expand a treatment group to inspect its images. Open the full annotated image to zoom. Yellow outlines mark counted objects; cyan marks analyzed nuclei and red marks excluded nuclei. Colors in label masks distinguish objects, not intensity.</p>'
    for group,fields in mapping.groupby('treatment_group',sort=False,dropna=False):
        count=len(fields)
        flagged_count=sum(im in flagged_ids for im in fields.image_id)
        flag_label=f' · {flagged_count} flagged for review' if flagged_count else ''
        body+=f'<details class="treatment-images"><summary>{esc(group)} · {count} image{"s" if count!=1 else ""}{flag_label}</summary>'
        for _,m in fields.iterrows():
            im=m.image_id;body+=f'<article><h3>{esc(image_display_name(m))} · {esc(m.treatment_group)}</h3>'
            if im in flagged_ids:body+='<p class="note"><strong>Outlier review requested</strong> — one or more image endpoints lie outside the within-treatment fences.</p>'
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
                    nid=cr.nucleus_id;display=f'{image_display_name(m)} · {str(nid).rsplit("_",1)[-1]}'
                    body+=f'<a href="QC/Nucleus_Crops/{nid}.png"><img loading="lazy" src="QC/Nucleus_Crops/{nid}.png"><small>{esc(display)}</small></a>'
                body+='</div></details>'
            body+='</article>'
        body+='</details>'
    body+=review_footer+qc_footer+'</body></html>'
    (out/'Review.html').write_text(body, encoding='utf-8')
    lines=['# Foci analysis report','',f'Processed {len(mapping)} images; {len(cell)} nuclei analyzed.',
           f'Nuclear channel: {nuclear}.',totals,'',
           'Descriptive measurements of segmented 2D nuclear objects; no inferential tests.','',
           '| '+' | '.join(labels)+' |','|'+'|'.join(['---']*len(labels))+'|']
    for row in summary.itertuples(index=False,name=None):
        lines.append('| '+' | '.join(f'{v:.2f}' if isinstance(v,float) else str(v) for v in row)+' |')
    lines+=['',f'Methods: foreground segmentation and distance watershed on {nuclear}; channel-specific background opening, noise-adaptive thresholds, object watershed, size and local-contrast filters.',
            '', 'QC inspection findings:','']
    for _,r in qc.iterrows():lines.append(f'- {names.get(r.image_id,r.image_id)}, C{r.channel}: {r.manual_inspection_warnings}.')
    lines+=['',f'Within-treatment outlier review: {len(flagged_ids)} image(s) flagged. See QC/Image_Outlier_Review.csv for endpoint values, fences, and unassessed cases.']
    if p.get('group_names'):
        lines+=['','Comparisons use combined treatment_group labels. Each image, nucleus, and focus contributes once to its treatment group.']
    lines+=['','Major outputs:','', '- Review.html: linked review report'+(' and nucleus galleries.' if detailed else '.'),
            '- [Experiment_Summary.png](Experiment_Summary.png) and [Experiment_Summary.svg](Experiment_Summary.svg): experiment overview with image-level endpoints, ECDFs, and a nucleus scatter when exactly two foci channels are selected.',
            '- [Representative_Images.png](Representative_Images.png) and [Representative_Images.svg](Representative_Images.svg): up to two unflagged images per treatment, with separate channel views and shared contrast.',
            '- Methods/Representative_Image_Selection.csv: scores, metric coverage, selection ranks, and eligibility reasons.',
            '- QC/Image_Outlier_Review.csv: within-treatment endpoint assessments and review flags.',
            '- Results/: image, cell, focus, and QC measurement tables.',
            '- QC/: annotated figures, '+('native overlays, nucleus crops, ' if detailed else '')+'shared-contrast figures, per-treatment input intensity plots (PNG and SVG), and QC metrics.',
            '- Plots/: '+', '.join(f'Channel_{c}' for c in channels)+'; comparison figures in PNG and SVG.',
            '- Methods/Output_Parameters.json: parameters and channel selection for this report.',
            '', 'Limitations: 2D projections; no manual ground truth. Dense or dim foci are parameter sensitive. See Methods/Methods.md for definitions.']
    if colocalization_settings(p)['enabled']:
        lines+=['', '## Colocalization analysis', '',
                'See the colocalization section in [Review.html](Review.html#colocalization), after the plots and before image inspection.',
                'Tables: '+', '.join(f'[Colocalization_{name}.csv](Results/Colocalization_{name}.csv)' for name in ('Pairs','Foci','Nuclei','Images')),
                colocalization_outputs.methods_text(p)]
    (out/'Report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def output_work_total(mapping,cell,p):
    """Count the work executed by QC, plotting, summary, and report helpers.

    A PNG/SVG pair is one figure task. Sampling and histogram passes are also
    counted so source-image preparation reports real progress before rendering.
    """
    n=len(mapping);channels=len(p['foci_channels'])
    display_channels=len(set([p['nuclear_channel'],*p['foci_channels']]))
    pages=math.ceil(n/SHARED_CONTRAST_PAGE_SIZE)
    groups=len(mapping.treatment_group.unique())
    work={
        'preparation_and_records':5,  # setup, QC ranges, summary, methods, reports
        'annotated_qc':n,
        'shared_contrast_sampling':n*display_channels,
        'shared_contrast_ranges':display_channels,
        'shared_contrast_pages':pages*channels,
        'comparison_figures':7*channels,
        'intensity_scan_and_histogram':2*n*channels,
        'treatment_intensity_figures':groups*channels,
        'representatives_and_outliers':groups+3,  # review audits, display ranges, group panels, figure
    }
    if p.get('qc_output','detailed')=='detailed':
        work['native_overlays']=n*(1+3*channels)
        work['nucleus_crops']=len(cell)
    if colocalization_settings(p)['enabled']:
        work['colocalization']=len(colocalization_outputs.FIGURES)+n
    return sum(work.values())


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output-dir',type=Path,required=True,help='Analysis folder containing Results, Masks, and Methods')
    ap.add_argument('--qc-output',choices=['minimal','detailed'],default='detailed',
                    help='minimal omits separate native overlays and nucleus crops (default: detailed)')
    add_channel_arguments(ap)
    add_progress_argument(ap)
    args=ap.parse_args();out=args.output_dir
    mapping=pd.read_csv(out/'Methods'/'Treatment_Group_Mapping.csv');images=pd.read_csv(out/'Results'/'Image_Summary.csv')
    cell=pd.read_csv(out/'Results'/'Cell_Summary.csv');focus=pd.read_csv(out/'Results'/'Focus_Data.csv')
    qc=pd.read_csv(out/'Results'/'QC_Summary.csv')
    try:
        p=resolve_channels(out,args.nuclear_channel,args.foci_channels)
        check_channel_inputs(out,p,mapping,images,cell,qc)
        coloc_tables=read_colocalization_tables(out,p)
    except ValueError as exc:
        ap.error(str(exc))
    channels=p['foci_channels']
    p['qc_output']=args.qc_output
    images=selected_columns(images,channels);cell=selected_columns(cell,channels)
    if 'qc_flags' in images:
        images['qc_flags']=images.apply(lambda row:' | '.join(f'C{c}: {row[f"C{c}_qc_flags"]}' for c in channels),axis=1)
    focus=focus[focus.channel.isin(channels)];qc=qc[qc.channel.isin(channels)]
    # Derive treatment groups and colors from this run rather than the pilot dataset.
    global GROUPS,LABELS,COLORS
    GROUPS=list(mapping.treatment_group.unique());LABELS=[g.replace('ART558_20PDS','ART558 + 20PDS') for g in GROUPS]
    COLORS=[TREATMENT_PALETTE[i%len(TREATMENT_PALETTE)] for i in range(len(GROUPS))]
    progress=ProgressReporter('outputs',output_work_total(mapping,cell,p),args.progress_json)
    progress.update(0,'Preparing output generation',force=True)
    (out/'Methods'/'Output_Parameters.json').write_text(json.dumps(p,indent=2), encoding='utf-8')
    progress.advance('Output preparation and saved-input checks complete',force=True)
    outliers,selection=review_images(mapping,images,p)
    outliers.to_csv(out/'QC'/'Image_Outlier_Review.csv',index=False)
    selection.to_csv(out/'Methods'/'Representative_Image_Selection.csv',index=False)
    progress.advance('Outlier assessments and representative selection saved',force=True)
    display_channels=list(dict.fromkeys([p['nuclear_channel'],*channels]))
    display_ranges=shared_display_ranges(mapping,display_channels,p,progress=progress)
    make_qc(out,mapping,cell,focus,p,progress=progress,display_ranges=display_ranges)
    make_plots(out,cell,focus,images,p,progress=progress)
    make_experiment_summary(out,mapping,images,cell,focus,p,progress=progress)
    make_representative_images(out,mapping,selection,p,display_ranges,progress=progress)
    if coloc_tables:
        colocalization_outputs.make_figures(out,mapping,p,coloc_tables,GROUPS,COLORS,progress)
    print('Writing methods and review reports...',flush=True)
    build_methods(out,p,mapping,images,cell,focus,qc)
    progress.advance('Methods and data dictionary saved',force=True)
    report(out,mapping,images,cell,focus,qc,p,outliers=outliers)
    progress.advance('Review reports saved',force=True)
    print('QC, plots, methods and report complete',flush=True)


if __name__=='__main__':main()
