"""Four local diagnostic arms from saved candidates; no training or DEM."""
from pathlib import Path
from datetime import datetime
import sys,json
import fiona,rasterio,yaml
from shapely.geometry import shape,box,mapping
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code'))
import pond_storage_monitor as m
SOURCE=ROOT/'diagnostics/results/local_downstream/20260913_094837_488085'
OUT=ROOT/'diagnostics/results/local_screening_comparison'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
OUT.mkdir(parents=True,exist_ok=False)
trace=json.loads((SOURCE/'trace.json').read_text(encoding='utf-8'))
lu=ROOT/'data/landuse_water/PSH_土地利用水体.tif'
if not lu.exists(): raise FileNotFoundError(lu)
report={'scope':'Local candidate windows only; bypassing landuse is a diagnostic, not a proposed production rule. No truth accuracy or volume.', 'targets':[]}
for item in trace['objects']:
    date,pid=item['date'],item['pond_id']
    if item['baseline_candidate_mismatch_pixels']!=0: raise RuntimeError('Unvalidated baseline')
    baseline_targets=item['variants']['baseline']
    if len(baseline_targets)!=1: raise RuntimeError('Ambiguous original target')
    original=shape(baseline_targets[0]['geometry'])
    cfg=yaml.safe_load((ROOT/f'output/{date.replace("-", "")}/final/pihe/resolved_monitor_cfg_{date}.yaml').read_text(encoding='utf-8'))
    result={'date':date,'original_id':pid,'arms':{}}
    for variant in ['baseline','experiment']:
        path=SOURCE/f'{date}_{pid}/{variant}_candidate.tif'
        with rasterio.open(path) as ds: crs=ds.crs; boundary=box(*ds.bounds).boundary
        objects=m.polygonize_water_mask(str(path),min_area_m2=float(cfg['min_water_area_m2']),max_area_m2=float(cfg['max_water_area_m2']),simplify_tolerance=float(cfg.get('simplify_tolerance_m',0)),watershed_split_enable=False)
        kept_ids={o.obj_id for o in m.filter_by_landuse_water(objects,str(lu),crs)}
        gpkg=OUT/f'{date}_{pid}_{variant}.gpkg'
        schema={'geometry':'Unknown','properties':{'local_id':'int','area_m2':'float','land_kept':'int','edge':'int','target_m2':'float'}}
        with fiona.open(gpkg,'w',driver='GPKG',layer='area_qualified',crs_wkt=crs.to_wkt(),schema=schema) as dst:
            for o in objects:
                dst.write({'geometry':mapping(o.geom),'properties':{'local_id':o.obj_id,'area_m2':o.area_m2,'land_kept':int(o.obj_id in kept_ids),'edge':int(o.geom.intersects(boundary)),'target_m2':o.geom.intersection(original).area}})
        for rule in ['original_landuse','bypass_landuse']:
            selected=[o for o in objects if rule=='bypass_landuse' or o.obj_id in kept_ids]
            # Edge-truncated components cannot support object-count/area claims.
            safe=[o for o in selected if not o.geom.intersects(boundary)]
            target=[o for o in safe if o.geom.intersection(original).area>0]
            result['arms'][variant+'__'+rule]={'interior_local_objects':len(safe),'interior_local_area_m2':sum(o.area_m2 for o in safe),'excluded_edge_objects':len(selected)-len(safe),'target_objects':len(target),'target_full_object_area_m2':sum(o.area_m2 for o in target),'target_overlap_area_m2':sum(o.geom.intersection(original).area for o in target),'gpkg':str(gpkg)}
    report['targets'].append(result)
    print(date,pid,'complete',flush=True)
path=OUT/'comparison.json'; path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print('RESULT:',path)
