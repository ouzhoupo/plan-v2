"""User-run: select one nonidentical reliable pair and compare saved levels only."""
import sys
sys.dont_write_bytecode=True
import csv,json
from pathlib import Path
from datetime import datetime
import fiona
from shapely.geometry import shape
from pyproj import CRS

ROOT=Path(__file__).resolve().parents[2]
PREVIOUS=ROOT/'diagnostics/results/one_two_date_pair/20260915_143901_705294'


def main():
    if json.loads((PREVIOUS/'summary.json').read_text())['status']!='complete':
        raise RuntimeError('Previous matching incomplete')
    with (PREVIOUS/'eligible_pairs.csv').open(encoding='utf-8-sig',newline='') as f:
        candidates=[r for r in csv.DictReader(f) if r['eligible']=='True' and r['isolated_pair']=='True' and float(r['iou'])<1-1e-9]
    if not candidates: raise RuntimeError('No eligible nonidentical pair; do not relax rules')
    candidates.sort(key=lambda r:(-float(r['iou']),int(r['pond_id_2024']),int(r['pond_id_2025'])))
    chosen=candidates[0]
    dates=['2024-06-01','2025-05-21']
    ids=[int(chosen['pond_id_2024']),int(chosen['pond_id_2025'])]
    objects=[];crs=None
    for date,pid in zip(dates,ids):
        found=[]
        with fiona.open(ROOT/f'output/{date.replace("-", "")}/final/pihe/pond_storage_{date}.gpkg',layer='ponds') as src:
            current=CRS.from_user_input(src.crs_wkt or src.crs)
            if crs is not None and current!=crs: raise RuntimeError('CRS differs')
            crs=current
            for feat in src:
                if int(feat['properties']['pond_id'])==pid:
                    found.append((shape(feat['geometry']),dict(feat['properties'])))
        if len(found)!=1: raise RuntimeError('Official object missing or ambiguous')
        if not found[0][0].is_valid: raise RuntimeError('Invalid official geometry')
        objects.append(found[0])
    g0,p0=objects[0];g1,p1=objects[1]
    overlap=g0.intersection(g1).area
    iou=overlap/g0.union(g1).area
    for name,value in [('area_2024_m2',g0.area),('area_2025_m2',g1.area),('overlap_m2',overlap)]:
        if abs(value-float(chosen[name]))>.01: raise RuntimeError('Geometry differs from previous matching')
    if abs(iou-float(chosen['iou']))>1e-9 or g0.equals(g1):
        raise RuntimeError('Pair does not match selection criteria')
    levels={}
    for key in ['bnd_n','h05','h50','h75','h95','hmean']:
        a,b=p0.get(key),p1.get(key)
        levels[key]={'2024':a,'2025':b,'difference_2025_minus_2024':None if a is None or b is None else float(b)-float(a)}
    report={'status':'complete','dates':dates,'official_ids':ids,'eligible_nonidentical_pair_count':len(candidates),
            'selection_rule':'From previously eligible isolated pairs, exclude IoU=1, select highest IoU then lowest official IDs; no selection on level differences.',
            'geometry':{'area_2024_m2':g0.area,'area_2025_m2':g1.area,'overlap_m2':overlap,'iou':iou,
                        'only_2024_m2':g0.difference(g1).area,'only_2025_m2':g1.difference(g0).area,
                        'net_area_change_m2':g1.area-g0.area},
            'saved_levels':levels,'qa':{'2024':p0.get('qa'),'2025':p1.get('qa')},
            'limits':'Saved-output comparison only; not new DEM sampling or measured water-level change. Geometry-only pairing has not been visually confirmed. Boundary difference combines all upstream processing, not a closing-only intervention. No volumes computed.'}
    out=ROOT/'diagnostics/results/changed_pair_levels'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True,exist_ok=False)
    (out/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'geometries.json').write_text(json.dumps({'crs_wkt':crs.to_wkt(),'features':[{'date':d,'pond_id':pid,'geometry':g.__geo_interface__} for d,pid,(g,p) in zip(dates,ids,objects)]},ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
    print('RESULT_FOLDER:',out,flush=True)


if __name__=='__main__': main()
