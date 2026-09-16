"""Read small windows around three samples; produce evidence panels, not model outputs."""
from pathlib import Path
import json
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from rasterio.features import geometry_mask
import fiona
from shapely.geometry import shape, mapping
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'diagnostics/results/pihe_local_review'
OUT.mkdir(parents=True, exist_ok=True)
DATES = ['2024-06-01', '2025-05-21']
report = []
for target_date, pid in [('2025-05-21',46608),('2025-05-21',45219),('2024-06-01',14660)]:
    sample = ROOT / f'diagnostics/results/pihe_gis_samples/pihe_stage_samples_{target_date.replace("-", "")}.gpkg'
    with fiona.open(sample) as src:
        feature = next(f for f in src if f['properties']['pond_id'] == pid)
        geom = shape(feature['geometry'])
        crs = src.crs
    bounds = geom.buffer(150).bounds
    fig, axes = plt.subplots(2,4,figsize=(16,9), constrained_layout=True)
    item = {'target_date':target_date,'pond_id':pid,'area_m2':geom.area,'dates':{}}
    for row,date in enumerate(DATES):
        with rasterio.open(ROOT / f'data/s2/s2_stack_{date}.tif') as src:
            assert src.crs == crs
            win = from_bounds(*bounds,src.transform).round_offsets().round_lengths()
            arr = src.read(window=win, masked=True).astype('float32').filled(np.nan)
            trans = src.window_transform(win)
            inside = geometry_mask([mapping(geom)],out_shape=arr.shape[1:],transform=trans,invert=True)
            extent = rasterio.transform.array_bounds(arr.shape[1],arr.shape[2],trans)
            extent = (extent[0],extent[2],extent[1],extent[3])
        def stretch(indices):
            rgb = np.moveaxis(arr[indices],0,-1)
            finite=rgb[np.isfinite(rgb)]
            lo,hi=np.percentile(finite,[2,98]) if finite.size else (0,1)
            return np.nan_to_num(np.clip((rgb-lo)/max(hi-lo,1e-6),0,1))**0.7
        with np.errstate(invalid='ignore',divide='ignore'):
            mndwi=(arr[1]-arr[4])/(arr[1]+arr[4]+1e-8)
        axes[row,0].imshow(stretch([2,1,0]),extent=extent)
        axes[row,1].imshow(stretch([4,3,2]),extent=extent)
        axes[row,2].imshow(mndwi,extent=extent,vmin=-1,vmax=1,cmap='BrBG')
        stats={}
        for name in ['source_flag','water_mask','water_prob']:
            with rasterio.open(ROOT / f'output/{date.replace("-", "")}/pihe/{name}_{date}.tif') as src:
                w=from_bounds(*bounds,src.transform).round_offsets().round_lengths()
                a=src.read(1,window=w)
                mask=geometry_mask([mapping(geom)],out_shape=a.shape,transform=src.window_transform(w),invert=True)
                vals=a[mask]
                if name=='source_flag':
                    stats['source_fractions']={str(i):float(np.mean(vals==i)) for i in range(5)}
                    axes[row,3].imshow(a,extent=extent,vmin=0,vmax=4,cmap='tab10',interpolation='nearest')
                else:
                    stats[name+'_mean']=float(np.mean(vals))
        finite=inside & np.isfinite(mndwi)
        stats['mndwi_positive_fraction']=float(np.mean(mndwi[finite]>0)) if finite.any() else None
        stats['mndwi_median']=float(np.median(mndwi[finite])) if finite.any() else None
        scl=arr[6][inside]; scl=scl[np.isfinite(scl)]
        stats['scl_fractions']={str(int(i)):float(np.mean(scl==i)) for i in np.unique(scl)}
        item['dates'][date]=stats
        for col,title in enumerate(['RGB (3,2,1)','SWIR/NIR/Red (5,4,3)','MNDWI [-1,1]','Source: 1 SAR; 2 optical; 3 both; 4 prior']):
            ax=axes[row,col]
            for poly in ([geom] if geom.geom_type=='Polygon' else geom.geoms):
                x,y=poly.exterior.xy; ax.plot(x,y,color='yellow',linewidth=1)
            ax.set_title(date+' | '+title,fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f'Target {target_date} pond_id={pid}; yellow = model object boundary',fontsize=14)
    fig.savefig(OUT / f'review_{target_date}_{pid}.png',dpi=130)
    plt.close(fig)
    report.append(item)
(OUT/'evidence.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
