"""Read metadata and small child windows; preserve raw NoData values."""
from pathlib import Path
from datetime import datetime
import json
import numpy as np
import rasterio
from rasterio.features import geometry_window, geometry_mask
from shapely.geometry import shape
from build_pihe_gis_samples import transformed_geom

ROOT=Path(__file__).resolve().parents[2]
path=ROOT/'data/landuse_water/PSH_土地利用水体.tif'
trace=json.loads((ROOT/'diagnostics/results/local_downstream/20260913_094837_488085/trace.json').read_text(encoding='utf-8'))
def counts(a):
    v,n=np.unique(a,return_counts=True)
    return {str(x.item()):int(c) for x,c in zip(v,n)}
with rasterio.open(path) as ds:
    report={'file':str(path),'crs':str(ds.crs),'transform':list(ds.transform),'resolution':list(ds.res),'bounds':list(ds.bounds),'shape':[ds.height,ds.width],'dtype':ds.dtypes,'nodata':ds.nodata,'tags':ds.tags(),'band_tags':ds.tags(1),'descriptions':ds.descriptions,'children':[]}
    for entry in trace['objects']:
        date,pid=entry['date'],entry['pond_id']
        with rasterio.open(ROOT/f'output/{date.replace("-", "")}/pihe/water_mask_{date}.tif') as ref:
            crs=ref.crs
        for child in entry['variants']['experiment']:
            geom=transformed_geom(shape(child['geometry']),crs,ds.crs)
            win=geometry_window(ds,[geom.__geo_interface__],pad_x=1,pad_y=1)
            raw=ds.read(1,window=win,masked=False)
            valid=ds.read_masks(1,window=win)>0
            inside=geometry_mask([geom.__geo_interface__],out_shape=raw.shape,transform=ds.window_transform(win),invert=True,all_touched=True)
            report['children'].append({'date':date,'original_id':pid,'local_id':child['local_id'],'landuse_kept':child['landuse_kept'],'raw_window_values':counts(raw),'raw_object_values_all_touched':counts(raw[inside]),'window_valid_pixels':int(valid.sum()),'object_pixels':int(inside.sum()),'object_valid_pixels':int((valid & inside).sum())})
out=ROOT/'diagnostics/results/landuse_values'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
out.mkdir(parents=True,exist_ok=False)
result=out/'audit.json'; result.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print('RESULT:',result)
