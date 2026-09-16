"""Plot existing diagnostic masks against aligned S2; no training."""
from pathlib import Path
from datetime import datetime
import json
import numpy as np
import fiona
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from rasterio.features import geometry_mask, geometry_window
from shapely.geometry import shape, box
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / 'diagnostics/results/positive_label_experiment/20260912_074812_069687'
OUT = ROOT / 'diagnostics/results/label_mask_visual_review' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
OUT.mkdir(parents=True, exist_ok=False)
records = []
for date, pid in [('2025-05-21', 46608), ('2025-05-21', 45219), ('2024-06-01', 14660)]:
    with fiona.open(ROOT / f'diagnostics/results/pihe_gis_samples/pihe_stage_samples_{date.replace("-", "")}.gpkg') as src:
        matches = []
        for feature in src:
            value = feature['properties'].get('pond_id')
            if value is None:
                continue
            try:
                object_id = int(value)
            except (TypeError, ValueError):
                continue
            if object_id == pid:
                matches.append(feature)
        if not matches:
            raise RuntimeError(f'Object not found: {date} pond_id={pid}')
        if any(not f['geometry'] for f in matches):
            raise RuntimeError(f'Object has no geometry: {date} pond_id={pid}')
        geom = shape(matches[0]['geometry'])
        # One object may be sampled under both final_kept_large and final_bbox_only.
        # Accept repeated records only when their spatial geometries are equal.
        if any(not geom.equals(shape(f['geometry'])) for f in matches[1:]):
            raise RuntimeError(f'Conflicting geometries for {date} pond_id={pid}; inspect sample categories')
        if len(matches) > 1:
            print(f'{date} {pid}: {len(matches)} records with identical geometry; using one', flush=True)
        vector_crs = rasterio.crs.CRS.from_user_input(src.crs_wkt or src.crs)
    matched = 0
    for path in sorted(INPUT.glob(f'{date}_*_masks.tif')):
        with rasterio.open(path) as ds:
            if ds.crs != vector_crs:
                raise RuntimeError('Object and mask CRS mismatch')
            if not box(*ds.bounds).intersects(geom):
                continue
            matched += 1
            win = geometry_window(ds, [geom.buffer(120).__geo_interface__])
            masks = ds.read(window=win) > 0
            trans = ds.window_transform(win)
            h, w = masks.shape[1:]
            inside = geometry_mask([geom.__geo_interface__], out_shape=(h,w), transform=trans, invert=True)
            with rasterio.open(ROOT / f'data/s2/s2_stack_{date}.tif') as s2:
                # Nearest preserves categorical SCL; display is on exact mask grid.
                with WarpedVRT(s2, crs=ds.crs, transform=trans, width=w, height=h, resampling=Resampling.nearest) as vrt:
                    a = vrt.read(masked=True).astype('float32').filled(np.nan)
            def stretch(indices):
                rgb = np.moveaxis(a[indices], 0, -1)
                finite = rgb[np.isfinite(rgb)]
                lo, hi = np.percentile(finite, [2,98]) if finite.size else (0,1)
                return np.nan_to_num(np.clip((rgb-lo)/max(float(hi-lo), 1e-6),0,1)) ** .7
            rgb, false = stretch([2,1,0]), stretch([4,3,2])
            removed = inside & masks[1] & ~masks[2]
            added = inside & ~masks[1] & masks[2]
            retained = inside & masks[1] & masks[2]
            overlay = np.zeros((h,w,4))
            overlay[retained] = [0,.6,1,.4]
            overlay[removed] = [1,0,0,.5]
            overlay[added] = [0,1,0,.7]
            left,bottom,right,top = rasterio.transform.array_bounds(h,w,trans)
            extent = (left,right,bottom,top)
            fig, axes = plt.subplots(2,2,figsize=(13,11),constrained_layout=True)
            for ax, image, title in zip(axes.flat,[rgb,false,rgb,false],['S2 RGB','S2 SWIR / NIR / Red','RGB + mask changes','SWIR / NIR / Red + mask changes']):
                ax.imshow(image,extent=extent,interpolation='nearest')
                if 'changes' in title:
                    ax.imshow(overlay,extent=extent,interpolation='nearest')
                for poly in ([geom] if geom.geom_type == 'Polygon' else geom.geoms):
                    for ring in [poly.exterior, *poly.interiors]:
                        x,y = ring.xy
                        ax.plot(x,y,color='yellow',linewidth=.7)
                ax.set_xlim(left,right); ax.set_ylim(bottom,top)
                ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])
            axes[1,1].legend(handles=[Patch(color='red',label='Removed'),Patch(color='deepskyblue',label='Retained'),Patch(color='lime',label='Added')],loc='lower right',fontsize=8)
            fig.suptitle(f'{date} object {pid} | yellow: original object | changes inside object only')
            output = OUT / f'{date}_{pid}_{path.stem}.png'
            fig.savefig(output,dpi=160); plt.close(fig)
            records.append({'date':date,'pond_id':pid,'mask_file':str(path),'image':str(output),'removed_pixels':int(removed.sum()),'added_pixels':int(added.sum()),'retained_pixels':int(retained.sum()),'baseline_saved_mismatch':int((inside & (masks[0]!=masks[1])).sum()),'note':'Display S2 stack, not cloud-filtered model composite. Visual evidence is provisional; no truth labels assigned.'})
    if not matched:
        raise RuntimeError(f'No mask tile found: {date} {pid}')
(OUT / 'review.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
print('RESULT:', OUT / 'review.json')
