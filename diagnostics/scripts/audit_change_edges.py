"""Distances to baseline mask edge are diagnostics, not true shorelines."""
from pathlib import Path
from datetime import datetime
import json
import numpy as np
import rasterio
from rasterio.features import geometry_mask
from scipy.ndimage import distance_transform_edt, binary_erosion
from shapely.geometry import shape
ROOT=Path(__file__).resolve().parents[2]
source=ROOT/'diagnostics/results/positive_label_experiment/20260913_125712_794605'
exp=json.loads((source/'experiment.json').read_text(encoding='utf-8'))
records=[]
for date,window,pid in [('2024-06-01',14660,576),('2025-05-21',45219,2850),('2024-06-01',14660,1051)]:
    key=f'released_{date}_{window}_{pid}'
    geom=shape(exp['released_samples'][key]['geometry'])
    rows=[]
    for path in sorted(source.glob(f'{date}_*_masks.tif')):
        with rasterio.open(path) as ds:
            inside=geometry_mask([geom.__geo_interface__],out_shape=(ds.height,ds.width),transform=ds.transform,invert=True)
            if not inside.any(): continue
            masks=ds.read()>0; base,other=masks[1],masks[2]
            res=(abs(ds.transform.e),abs(ds.transform.a))
            # Pad with nonwater so tile limits cannot create arbitrarily deep water.
            depth=distance_transform_edt(np.pad(base,1),sampling=res)[1:-1,1:-1]
            removed=inside & base & ~other; added=inside & ~base & other
            valid_interior=np.ones(base.shape,bool); valid_interior[:3,:]=False;valid_interior[-3:,:]=False;valid_interior[:,:3]=False;valid_interior[:,-3:]=False
            item={'tile':str(path),'removed':int(removed.sum()),'added':int(added.sum()),'removed_near_tile_edge_30m':int((removed & ~valid_interior).sum())}
            nearest=distance_transform_edt(~base,sampling=res)[added]
            item['added_distance_to_baseline_water']={'pixels':int(nearest.size),'le_10m':int((nearest<=10.001).sum()),'le_20m':int((nearest<=20.001).sum()),'max_m':float(nearest.max()) if nearest.size else None}
            for label,support in [('removed',removed),('retained',inside & base & other)]:
                d=depth[support]
                item[label+'_baseline_edge_distance']={'pixels':int(d.size),'le_10m':int((d<=10.001).sum()),'le_20m':int((d<=20.001).sum()),'gt_20m':int((d>20.001).sum()),'max_m':float(d.max()) if d.size else None}
            rows.append(item)
    records.append({'date':date,'window_id':window,'local_id':pid,'tiles':rows})
out=ROOT/'diagnostics/results/change_edge_audit'/datetime.now().strftime('%Y%m%d_%H%M%S_%f');out.mkdir(parents=True,exist_ok=False)
path=out/'edges.json';path.write_text(json.dumps({'caveat':'Distance to baseline classified mask, not surveyed bank; edge concentration cannot prove correct removal.','objects':records},ensure_ascii=False,indent=2),encoding='utf-8')
print(path)
print(json.dumps(records,ensure_ascii=False))
