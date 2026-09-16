"""User-run: locate value-1 land-use cells for six already-kept objects."""
import sys
sys.dont_write_bytecode = True
import csv, json
from pathlib import Path
from datetime import datetime
import numpy as np
import rasterio
from rasterio.features import shapes, geometry_window
from rasterio.windows import from_bounds
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from scipy import ndimage
from shapely.geometry import shape, mapping, Polygon
from shapely.ops import transform, unary_union
from pyproj import Transformer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'diagnostics/results/fixed_site_closing/20260914_170521'
PREVIOUS = ROOT / 'diagnostics/results/closing_landuse_gate/20260914_181033_105443'
DATE = '2025-05-21'


def polygons(g):
    if g.is_empty:
        return []
    if g.geom_type == 'Polygon':
        return [g]
    return [p for part in getattr(g, 'geoms', []) for p in polygons(part)]


def draw(ax, geom, color, fill=False):
    for poly in polygons(geom):
        if fill:
            ax.fill(*poly.exterior.xy, color=color, alpha=.25)
        for ring in [poly.exterior, *poly.interiors]:
            ax.plot(*ring.xy, color=color, linewidth=1)


def main():
    if json.loads((PREVIOUS/'summary.json').read_text())['status'] != 'complete':
        raise RuntimeError('Previous check incomplete')
    with (PREVIOUS/'objects.csv').open(encoding='utf-8-sig', newline='') as f:
        targets = [r for r in csv.DictReader(f) if r['closing']=='1' and r['landuse_kept']=='True']
    if len(targets) != 6:
        raise RuntimeError('Expected the six previously retained objects')
    sites = {s['site']:s for s in json.loads((ROOT/'diagnostics/results/deduction_fixed_sites/20260914_164745/review.json').read_text())['sites']}
    out = ROOT/'diagnostics/results/kept_landuse_location'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True, exist_ok=False)
    records=[]
    with rasterio.open(ROOT/'data/landuse_water/PSH_土地利用水体.tif') as lu:
        for site in sorted({r['site'] for r in targets}):
            with rasterio.open(SOURCE/f'{site}_{DATE}_close1.tif') as src:
                after=src.read(1)>0; trans=src.transform; crs=src.crs; bounds=src.bounds
            with rasterio.open(SOURCE/f'{site}_{DATE}_close0.tif') as src:
                before=src.read(1)>0
                if src.crs != crs or not src.transform.almost_equals(trans) or before.shape != after.shape:
                    raise RuntimeError('Variant grids differ')
            with rasterio.open(ROOT/f'output/20250521/final/pihe/pond_water_mask_{DATE}.tif') as src:
                w=from_bounds(*bounds,src.transform).round_offsets().round_lengths()
                if not src.window_transform(w).almost_equals(trans) or not np.array_equal(src.read(1,window=w)>0,after):
                    raise RuntimeError('Historical baseline mismatch')
            core=np.zeros(after.shape,bool)
            w=from_bounds(*sites[site]['dates'][DATE]['bounds'],trans).round_offsets().round_lengths()
            y,x,h,width=map(int,[w.row_off,w.col_off,w.height,w.width])
            core[y:y+h,x:x+width]=True
            added=after & ~before & core
            labels,_=ndimage.label(after)
            selected=[r for r in targets if r['site']==site]
            ids=[int(r['local_component_id']) for r in selected]
            edge_ids=set(np.concatenate([labels[0],labels[-1],labels[:,0],labels[:,-1]]))
            if edge_ids.intersection(ids):
                raise RuntimeError('Target intersects patch edge')
            geoms={int(v):shape(g) for g,v in shapes(labels.astype('int32'),mask=np.isin(labels,ids),transform=trans,connectivity=4)}
            to_lu=Transformer.from_crs(crs,lu.crs,always_xy=True)
            to_water=Transformer.from_crs(lu.crs,crs,always_xy=True)
            for previous in selected:
                lid=int(previous['local_component_id']); geom=geoms[lid]
                if abs(geom.area-float(previous['patch_component_area_m2']))>.01:
                    raise RuntimeError('Object area changed')
                target=added & (labels==lid)
                if int(target.sum()) != int(previous['changed_core_pixels_in_component']):
                    raise RuntimeError('Tracked added pixels changed')
                additions=unary_union([shape(g) for g,v in shapes(target.astype('uint8'),mask=target,transform=trans) if v==1])
                lg=transform(to_lu.transform,geom)
                lw=geometry_window(lu,[mapping(lg)],pad_x=1,pad_y=1)
                data=lu.read(1,window=lw); lt=lu.window_transform(lw)
                hit=data==1
                if int(hit.sum()) != int(previous['window_water_value1_pixels']):
                    raise RuntimeError('Land-use window differs from previous check')
                footprint=transform(to_water.transform,Polygon([lt*(0,0),lt*(data.shape[1],0),lt*(data.shape[1],data.shape[0]),lt*(0,data.shape[0])]))
                # Native raster cell footprints transformed to metre CRS; no nearest-neighbour
                # resampling for the overlap measurement, and no all-touched hit counted as area.
                water_cells=unary_union([transform(to_water.transform,shape(g)) for g,v in shapes(hit.astype('uint8'),mask=hit,transform=lt) if v==1])
                overlap=geom.intersection(water_cells).area
                added_overlap=additions.intersection(water_cells).area
                record={'site':site,'local_component_id':lid,'object_area_m2':geom.area,'tracked_added_area_m2':additions.area,
                        'window_value1_cells':int(hit.sum()),'value1_footprint_overlap_object_m2':overlap,
                        'value1_footprint_overlap_tracked_additions_m2':added_overlap,
                        'value1_footprint_outside_object_m2':water_cells.difference(geom).area,
                        'support_location':'outside_only' if overlap<=1e-6 else 'object_overlap',
                        'limits':'30m land-use cell footprint overlap is spatial support, not independent truth; tracked additions limited to original window.'}
                key=f'{site}_{lid}'
                view=geom.union(footprint).buffer(80).bounds
                with rasterio.open(ROOT/f'data/s2/s2_stack_{DATE}.tif') as s2:
                    wt=from_bounds(*view,s2.transform).round_offsets().round_lengths()
                    # Existing S2 stacks and candidates are in the same CRS.
                    if s2.crs != crs:
                        raise RuntimeError('Unexpected S2 CRS; stop rather than misalign display')
                    rgb=s2.read([3,2,1],window=wt,masked=True).astype('float32').filled(np.nan)
                    vb=s2.window_bounds(wt)
                rgb=np.moveaxis(rgb,0,-1); finite=rgb[np.isfinite(rgb)]
                lo,hi=np.percentile(finite,[2,98]) if finite.size else (0,1)
                rgb=np.nan_to_num(np.clip((rgb-lo)/max(float(hi-lo),1e-6),0,1))**.7
                fig,axes=plt.subplots(1,2,figsize=(12,6),constrained_layout=True)
                for ax in axes:
                    ax.imshow(rgb,extent=(vb[0],vb[2],vb[1],vb[3]))
                    draw(ax,geom,'yellow');draw(ax,additions,'magenta')
                    ax.set_xlim(view[0],view[2]);ax.set_ylim(view[1],view[3]);ax.set_aspect('equal')
                    ax.set_xticks([]);ax.set_yticks([])
                draw(axes[1],water_cells,'cyan',True);draw(axes[1],footprint,'red')
                axes[0].set_title('Object and tracked closing additions')
                axes[1].set_title('Native land-use value-1 footprints and read window')
                axes[1].legend(handles=[Patch(color=c,label=l) for c,l in [('yellow','Object'),('magenta','Tracked additions'),('cyan','Land-use value 1'),('red','Read window')]],fontsize=8)
                fig.suptitle(key+' | 2025-05-21 | RGB is visual context, not clear-sky truth')
                fig.savefig(out/f'{key}.png',dpi=140);plt.close(fig)
                records.append(record)
                print(key,'complete',flush=True)
    with (out/'objects.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    (out/'summary.json').write_text(json.dumps({'status':'complete','objects':records},ensure_ascii=False,indent=2),encoding='utf-8')
    print('RESULT_FOLDER:',out,flush=True)


if __name__=='__main__':
    main()
