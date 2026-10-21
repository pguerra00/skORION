"""Display names and reproducible image-level review from saved measurements."""
from pathlib import Path
import json
import math

import numpy as np
import pandas as pd


ENDPOINTS=('mean_foci_per_nucleus','mean_focus_area_um2',
           'mean_focus_intensity_adu','percent_focus_positive_nuclei')


def image_display_name(record):
    for key in ('image_name','source_path','image_id'):
        value=record.get(key)
        if value is not None and pd.notna(value) and str(value).strip():
            return Path(str(value)).name if key=='source_path' else str(value)
    return 'Unknown image'


def representative_grid(count):
    """Return rows, columns; favor exact grids without excessively wide strips."""
    if count<1:return 1,1
    rows=max(i for i in range(1,math.isqrt(count)+1) if count%i==0)
    columns=count//rows
    if columns/rows<=2.5:return rows,columns
    columns=math.ceil(math.sqrt(count))
    return math.ceil(count/columns),columns


def review_images(mapping,images,parameters):
    """Return per-metric outlier assessments and a ranked representative audit.

    Means and scales include all finite image values, including flagged images.
    Flags only restrict representative eligibility, never measurement summaries.
    """
    names={row.image_id:image_display_name(row) for _,row in mapping.iterrows()}
    metrics=[(c,f'C{c}_{key}') for c in parameters['foci_channels'] for key in ENDPOINTS]
    assessments=[];candidates=[]
    for group,rows in images.groupby('treatment_group',sort=False,dropna=False):
        numeric={key:pd.to_numeric(rows.get(key,pd.Series(np.nan,index=rows.index)),errors='coerce')
                 .to_numpy(dtype=float) for _,key in metrics}
        means={};scales={};active=[]
        for channel,key in metrics:
            values=numeric[key];finite=values[np.isfinite(values)];n=len(finite)
            mean=float(finite.mean()) if n else None
            std=(0. if finite.min()==finite.max() else float(finite.std(ddof=1))) if n>=2 else None
            means[key]=mean;scales[key]=std
            if std is not None and np.isfinite(std) and std>0:active.append(key)
            q1,q3=np.quantile(finite,[.25,.75]) if n else (np.nan,np.nan)
            iqr=q3-q1;lower=q1-1.5*iqr;upper=q3+1.5*iqr
            for value,(_,row) in zip(values,rows.iterrows()):
                status=('missing value' if not np.isfinite(value) else
                        'insufficient images' if n<5 else 'zero IQR' if iqr<=0 else 'assessed')
                flag=bool(status=='assessed' and (value<lower or value>upper))
                assessments.append(dict(image_id=row.image_id,image_name=names.get(row.image_id,image_display_name(row)),
                    treatment_group=group,channel=channel,metric=key,value=value,finite_images=n,
                    q1=q1,q3=q3,lower_fence=lower,upper_fence=upper,assessment_status=status,outlier=flag))
        flagged={r['image_id'] for r in assessments if r['treatment_group']==group and r['outlier']}
        eligible=[]
        for index,(_,row) in enumerate(rows.iterrows()):
            used=[key for key in active if np.isfinite(numeric[key][index])]
            score=float(np.sqrt(np.mean([((numeric[key][index]-means[key])/scales[key])**2 for key in used]))) if used else 0.
            reason=('no usable nuclei' if not np.isfinite(row.nuclei_analyzed) or row.nuclei_analyzed<=0 else
                    'within-treatment outlier' if row.image_id in flagged else
                    'no usable endpoint values' if (active and not used) or not any(np.isfinite(v[index]) for v in numeric.values()) else '')
            record=dict(image_id=row.image_id,image_name=names.get(row.image_id,image_display_name(row)),
                treatment_group=group,score=score if used or not active else np.nan,
                metrics_used=len(used),metrics_available=len(active),partial_metrics=len(used)<len(active),
                outlier=row.image_id in flagged,eligible=not bool(reason),ineligibility_reason=reason,
                rank=np.nan,selected=False,group_means=json.dumps(means,allow_nan=False),
                group_standard_deviations=json.dumps(scales,allow_nan=False))
            candidates.append(record)
            if not reason:eligible.append(record)
        eligible.sort(key=lambda r:(-r['metrics_used'],r['score'],r['image_name'],r['image_id']))
        for rank,record in enumerate(eligible,1):
            record['rank']=rank;record['selected']=rank<=2
    return pd.DataFrame(assessments),pd.DataFrame(candidates)
