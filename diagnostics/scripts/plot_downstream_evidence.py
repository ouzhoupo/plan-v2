"""Local evidence from saved baseline outputs; no training or production changes."""
from pathlib import Path
from datetime import datetime
import json
import numpy as np
import fiona
import rasterio
from rasterio.features import geometry_window, geometry_mask, shapes
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from rasterio.windows import from_bounds
from shapely.geometry import shape, Polygon
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from build_pihe_gis_samples import transformed_geom, same_grid

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'diagnostics/results/downstream_visual_review' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
OUT.mkdir(parents=True, exist_ok=False)
lu_path = ROOT / 'data/landuse_water/PSH_土地利用水体.tif'
if not lu_path.is_file():
    raise FileNotFoundError(lu_path)
records = []

def draw(ax, geom, color, width=1):
    for poly in ([geom] if geom.geom_type == 'Polygon' else geom.geoms):
        for ring in [poly.exterior, *poly.interiors]:
            x,y = ring.xy
            ax.plot(x,y,color=color,linewidth=width)

for date,pid in [('2025-05-21',46608),('2025-05-21',45219),('2024-06-01',14660)]:
    with fiona.open(ROOT / f'diagnostics/results/pihe_gis_samples/pihe_stage_samples_{date.replace("-", "")}.gpkg') as src:
        matches = []
        for f in src:
            value = f['properties'].get('pond_id')
            if value is not None and str(value) == str(pid):
                if not f['geometry']:
                    raise RuntimeError(f'Missing geometry: {date} {pid}')
                matches.append(shape(f['geometry']))
        if not matches or any(not matches[0].equals(g) for g in matches[1:]):
            raise RuntimeError(f'Missing or ambiguous object: {date} {pid}')
        geom = matches[0]
        crs = src.crs_wkt or src.crs
    day = ROOT / 'output' / date.replace('-','')
    with rasterio.open(day / 'pihe' / f'water_mask_{date}.tif') as water, rasterio.open(day / 'final/pihe' / f'pond_water_mask_{date}.tif') as candidate:
        if water.crs != candidate.crs:
            raise RuntimeError('Water/candidate CRS mismatch')
        geom = transformed_geom(geom,crs,water.crs)
        with rasterio.open(lu_path) as lu:
            lg = transformed_geom(geom,water.crs,lu.crs)
            lw = geometry_window(lu,[lg.__geo_interface__],pad_x=1,pad_y=1)
            data = lu.read(1,window=lw)
            lt = lu.window_transform(lw)
            footprint = Polygon([lt*(0,0),lt*(data.shape[1],0),lt*(data.shape[1],data.shape[0]),lt*(0,data.shape[0])])
            rectangle = transformed_geom(footprint,lu.crs,water.crs)
            hit = data == 1
            exact = geometry_mask([lg.__geo_interface__],out_shape=data.shape,transform=lt,invert=True,all_touched=True)
            lu_polygons = [transformed_geom(shape(g),lu.crs,water.crs) for g,v in shapes(hit.astype('uint8'),mask=hit,transform=lt) if v == 1]
        win = geometry_window(water,[geom.union(rectangle).buffer(120).__geo_interface__])
        before = water.read(1,window=win)>0
        cw = from_bounds(*water.window_bounds(win),transform=candidate.transform).round_offsets().round_lengths()
        if not candidate.window_transform(cw).almost_equals(water.window_transform(win)):
            raise RuntimeError('Water/candidate pixel alignment mismatch')
        after = candidate.read(1,window=cw,boundless=True,fill_value=0)>0
        if after.shape != before.shape:
            raise RuntimeError('Water/candidate window size mismatch')
        trans = water.window_transform(win)
        h,w = before.shape
        inside = geometry_mask([geom.__geo_interface__],out_shape=(h,w),transform=trans,invert=True)
        with rasterio.open(ROOT / f'data/s2/s2_stack_{date}.tif') as s2:
            with WarpedVRT(s2,crs=water.crs,transform=trans,width=w,height=h,resampling=Resampling.nearest) as vrt:
                a = vrt.read([3,2,1],masked=True).astype('float32').filled(np.nan)
        rgb = np.moveaxis(a,0,-1)
        finite = rgb[np.isfinite(rgb)]
        lo,hi = np.percentile(finite,[2,98]) if finite.size else (0,1)
        rgb = np.nan_to_num(np.clip((rgb-lo)/max(float(hi-lo),1e-6),0,1))**.7
        left,bottom,right,top = rasterio.transform.array_bounds(h,w,trans)
        extent = (left,right,bottom,top)
        added = after & ~before
        removed = before & ~after
        overlay = np.zeros((h,w,4))
        overlay[before & after] = [0,.6,1,.4]
        overlay[added] = [1,0,1,.65]
        overlay[removed] = [1,0,0,.65]
        fig,axes = plt.subplots(1,3,figsize=(18,8),constrained_layout=True)
        for ax,title in zip(axes,['S2 RGB','Land-use water cells in actual read window','Saved water mask -> saved candidate mask']):
            ax.imshow(rgb,extent=extent,interpolation='nearest')
            draw(ax,geom,'yellow')
            ax.set_title(title,fontsize=10); ax.set_xticks([]); ax.set_yticks([])
            ax.set_xlim(left,right); ax.set_ylim(bottom,top)
        draw(axes[1],rectangle,'red',1.5)
        for poly in lu_polygons:
            draw(axes[1],poly,'cyan',1.5)
        axes[2].imshow(overlay,extent=extent,interpolation='nearest')
        draw(axes[2],geom,'yellow')
        axes[1].legend(handles=[Patch(color='red',label='Actual read window'),Patch(color='cyan',label='Land-use value 1 outlines')],fontsize=8,loc='lower left')
        axes[2].legend(handles=[Patch(color='deepskyblue',label='Retained'),Patch(color='magenta',label='Added'),Patch(color='red',label='Removed')],fontsize=8,loc='lower left')
        fig.suptitle(f'{date} object {pid} | yellow: original candidate | baseline outputs, not label experiment')
        image_path = OUT / f'{date}_{pid}.png'
        fig.savefig(image_path,dpi=150); plt.close(fig)
        records.append({'date':date,'pond_id':pid,'image':str(image_path),'landuse_file':str(lu_path),'bbox_water_pixels':int(hit.sum()),'polygon_water_pixels_all_touched':int((hit & exact).sum()),'landuse_rule_keeps':bool(hit.any()),'inside_original_object':{'water_pixels':int((before & inside).sum()),'candidate_pixels':int((after & inside).sum()),'added_pixels':int((added & inside).sum()),'removed_pixels':int((removed & inside).sum())},'pixel_area_m2':abs(trans.a*trans.e-trans.b*trans.d),'caveat':'Saved stage differences combine all processing; not a morphology-only causal experiment. S2 display is not production clear composite.'})
        print(date,pid,'complete',flush=True)
path = OUT / 'review.json'
path.write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
print('RESULT:',path)
