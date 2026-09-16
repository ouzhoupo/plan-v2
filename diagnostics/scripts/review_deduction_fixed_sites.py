"""Three deterministic sites, two dates; read-only production inputs, local outputs."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
from datetime import datetime
import json
import numpy as np
import fiona, rasterio
from rasterio.windows import from_bounds
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from shapely.geometry import shape, box
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code'))
import pond_storage_monitor as m
OUT=ROOT/'diagnostics/results/deduction_fixed_sites'/datetime.now().strftime('%Y%m%d_%H%M%S')
OUT.mkdir(parents=True)
base=ROOT/'output/20240601/pihe/buffered_river'
def load(name):
    with fiona.open(base/name) as src:
        return [(str(f.id),shape(f['geometry'])) for f in src]
res=load('psh_res.shp')
small=sorted([(i,g) for i,g in res if 500<=g.area<=500000],key=lambda x:(x[1].area,x[0]))
large=max(res,key=lambda x:x[1].area)
canal=max(load('__1.shp'),key=lambda x:x[1].area)
sites=[('large_reservoir',*large),('small_reservoir',*small[len(small)//2]),('main_canal',*canal)]
report={'selection':'Largest reservoir polygon; median area reservoir polygon among 500..500000m2; largest main-canal buffered feature. Representative point, fixed 1200m square. No selection by observed error. Nonrepresentative mechanism samples.', 'sites':[]}
for name,fid,geom in sites:
    p=geom.representative_point()
    bounds=(p.x-600,p.y-600,p.x+600,p.y+600)
    entry={'site':name,'source_feature_id':fid,'source_area_m2':geom.area,'center_utm50n':[p.x,p.y],'dates':{}}
    fig,axes=plt.subplots(2,4,figsize=(16,8),constrained_layout=True)
    for row,date in enumerate(['2024-06-01','2025-05-21']):
        day=ROOT/'output'/date.replace('-','')
        with rasterio.open(day/f'pihe/water_mask_{date}.tif') as src:
            win=from_bounds(*bounds,src.transform).round_offsets().round_lengths()
            water=src.read(1,window=win)>0
            trans=src.window_transform(win); crs=src.crs
            actual=src.window_bounds(win)
        h,w=water.shape
        aoi=m.read_union_geometry(str(ROOT/'data/aoi/pihe.shp'),crs)
        inside=m.rasterize_geoms_for_window([aoi],(h,w),trans,all_touched=True)>0
        water &= inside
        layers=m.build_deduct_layers(river_dir=str(day/'pihe/buffered_river'),reservoir_shp=str(ROOT/'data/reservoir/psh_res.shp'),extra_vectors=[],target_crs=crs,bbox_filter=actual,line_buffer_m=0)
        union=np.zeros((h,w),bool); overlaps={}
        for layer in layers:
            mask=m.rasterize_geoms_for_window(layer.geoms,(h,w),trans,all_touched=True)>0
            union |= mask
            overlaps[layer.name]=int((water & mask).sum())
        deducted=water & union
        with rasterio.open(day/f'final/pihe/pond_water_mask_{date}.tif') as src:
            cw=from_bounds(*actual,src.transform).round_offsets().round_lengths()
            assert src.window_transform(cw).almost_equals(trans)
            candidate=src.read(1,window=cw,boundless=True,fill_value=0)>0
            assert candidate.shape==water.shape
        with rasterio.open(ROOT/f'data/s2/s2_stack_{date}.tif') as src:
            with WarpedVRT(src,crs=crs,transform=trans,width=w,height=h,resampling=Resampling.nearest) as vrt:
                arr=vrt.read(masked=True).astype('float32').filled(np.nan)
        def stretch(indices):
            rgb=np.moveaxis(arr[indices],0,-1)
            finite=rgb[np.isfinite(rgb)]
            lo,hi=np.percentile(finite,[2,98]) if finite.size else (0,1)
            return np.nan_to_num(np.clip((rgb-lo)/max(float(hi-lo),1e-6),0,1))**.7
        extent=(actual[0],actual[2],actual[1],actual[3])
        rgb=stretch([2,1,0])
        overlay=np.zeros((h,w,4)); overlay[water & ~union]=[0,.7,1,.6]; overlay[deducted]=[1,0,0,.65]
        axes[row,0].imshow(rgb,extent=extent)
        axes[row,1].imshow(stretch([4,3,2]),extent=extent)
        axes[row,2].imshow(rgb,extent=extent); axes[row,2].imshow(overlay,extent=extent)
        axes[row,3].imshow(candidate,extent=extent,vmin=0,vmax=1,cmap='Blues')
        for ax,title in zip(axes[row],['S2 RGB','S2 SWIR/NIR/red','Red: direct deduction; cyan: remaining input','Saved candidate (includes morphology)']):
            if union.any() and not union.all():
                ax.contour(union,levels=[.5],colors='yellow',linewidths=.7,extent=extent,origin='upper')
            ax.set_title(date+' | '+title,fontsize=8); ax.set_xticks([]); ax.set_yticks([])
        vals=arr[6][deducted]; vals=vals[np.isfinite(vals)]
        entry['dates'][date]={'input_water_pixels':int(water.sum()),'direct_deducted_pixels':int(deducted.sum()),'remaining_before_morph_pixels':int((water & ~union).sum()),'saved_candidate_pixels':int(candidate.sum()),'deducted_then_candidate_pixels':int((deducted & candidate).sum()),'raw_layer_overlap_pixels_nonadditive':overlaps,'deducted_SCL_counts':{str(int(v)):int((vals==v).sum()) for v in np.unique(vals)},'pixel_area_m2':abs(trans.a*trans.e),'bounds':actual}
    fig.suptitle(name+' | fixed 1.2 km window | yellow: deduction union boundary | visual evidence, not truth labels')
    fig.savefig(OUT/f'{name}.png',dpi=130); plt.close(fig)
    report['sites'].append(entry)
    print(name,'complete',flush=True)
(OUT/'review.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print('OUTPUT:',OUT,flush=True)
