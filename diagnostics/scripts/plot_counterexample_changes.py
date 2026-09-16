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
from rasterio.windows import from_bounds
from shapely.geometry import shape, box
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / 'diagnostics/results/positive_label_experiment/20260913_125712_794605'
OUT = ROOT / 'diagnostics/results/counterexample_review' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
OUT.mkdir(parents=True, exist_ok=False)
records = []
experiment = json.loads((INPUT / 'experiment.json').read_text(encoding='utf-8'))
selected = [('2024-06-01',14660,576),('2025-05-21',45219,2850),('2024-06-01',14660,1051)]
for date, window_id, pid in selected:
    key = f'released_{date}_{window_id}_{pid}'
    geom = shape(experiment['released_samples'][key]['geometry'])
    with rasterio.open(ROOT / f'output/{date.replace("-", "")}/pihe/water_mask_{date}.tif') as reference:
        vector_crs = reference.crs
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
            with rasterio.open(ROOT / f'output/{date.replace("-", "")}/pihe/source_flag_{date}.tif') as source:
                sw = from_bounds(*ds.window_bounds(win), transform=source.transform).round_offsets().round_lengths()
                if not source.window_transform(sw).almost_equals(trans):
                    raise RuntimeError('Source grid mismatch')
                flags = source.read(1, window=sw)
            details = {}
            for label, support in [('removed',removed),('added',added),('retained',retained)]:
                values, counts = np.unique(flags[support], return_counts=True)
                scl = a[6][support]; finite = scl[np.isfinite(scl)]
                classes, numbers = np.unique(finite, return_counts=True)
                details[label] = {'pixels':int(support.sum()),'source_counts':{str(int(v)):int(n) for v,n in zip(values,counts)},'display_scl_counts':{str(int(v)):int(n) for v,n in zip(classes,numbers)},'display_scl_missing':int(scl.size-finite.size)}
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
            records.append({'date':date,'window_id':window_id,'local_id':pid,'mask_file':str(path),'image':str(output),'removed_pixels':int(removed.sum()),'added_pixels':int(added.sum()),'retained_pixels':int(retained.sum()),'baseline_saved_mismatch':int((inside & (masks[0]!=masks[1])).sum()),'change_details':details,'note':'SCL is from display stack, not production aggregate clear mask. Source flag alone does not prove SAR probability finite. Visual labels provisional.'})
    if not matched:
        raise RuntimeError(f'No mask tile found: {date} {pid}')
(OUT / 'review.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
print('RESULT:', OUT / 'review.json')

