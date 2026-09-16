"""User-run: deterministic geometry-only screening for one two-date pair."""
import sys
sys.dont_write_bytecode=True
import json,csv
from pathlib import Path
from datetime import datetime
import fiona
import numpy as np
from shapely.geometry import shape
from shapely.strtree import STRtree
from pyproj import CRS

ROOT=Path(__file__).resolve().parents[2]
DATES=['2024-06-01','2025-05-21']


def main():
    sets=[]; expected_crs=None
    for date in DATES:
        rows=[]
        with fiona.open(ROOT/f'output/{date.replace("-", "")}/final/pihe/pond_storage_{date}.gpkg',layer='ponds') as src:
            crs=CRS.from_user_input(src.crs_wkt or src.crs)
            if expected_crs is not None and crs!=expected_crs: raise RuntimeError('Official CRS mismatch')
            expected_crs=crs
            for f in src:
                if not f['geometry']: raise RuntimeError('Missing official geometry')
                g=shape(f['geometry']);p=dict(f['properties'])
                if g.is_empty or not g.is_valid: raise RuntimeError('Invalid official geometry: stop without silent repair')
                rows.append({'geom':g,'props':p})
        sets.append(rows)
    first,second=sets
    geoms=[r['geom'] for r in second];tree=STRtree(geoms)
    lookup={id(g):i for i,g in enumerate(geoms)}
    links=[]; first_links=[[] for _ in first];second_links=[[] for _ in second]
    for i,a in enumerate(first):
        for hit in tree.query(a['geom']):
            j=int(hit) if isinstance(hit,(int,np.integer)) else lookup[id(hit)]
            b=second[j]; overlap=a['geom'].intersection(b['geom']).area
            if overlap<=.01: continue
            aa,bb=a['geom'].area,b['geom'].area
            item={'i':i,'j':j,'pond_id_2024':int(a['props']['pond_id']),'pond_id_2025':int(b['props']['pond_id']),
                  'area_2024_m2':aa,'area_2025_m2':bb,'overlap_m2':overlap,
                  'fraction_2024':overlap/aa,'fraction_2025':overlap/bb,'iou':overlap/(aa+bb-overlap)}
            links.append(item);first_links[i].append(item);second_links[j].append(item)
    eligible=[]
    for item in links:
        i,j=item['i'],item['j']
        # Strict isolation: any other positive overlap above .01m2 disqualifies the pair.
        isolated=len(first_links[i])==1 and len(second_links[j])==1
        has_levels=all(sets[k][idx]['props'].get('h75') is not None for k,idx in [(0,i),(1,j)])
        item['isolated_pair']=isolated
        item['eligible']=isolated and has_levels and item['iou']>=.5 and min(item['fraction_2024'],item['fraction_2025'])>=.7
        if item['eligible']: eligible.append(item)
    eligible.sort(key=lambda r:(-r['iou'],r['pond_id_2024'],r['pond_id_2025']))
    report={'status':'complete','dates':DATES,'objects_each':[len(s) for s in sets],
            'positive_overlap_pairs':len(links),'eligible_pair_count':len(eligible),
            'selection_rule':'Exactly one positive-area overlap partner each way (>0.01m2); both saved h75 available; IoU>=0.5 and each overlap fraction>=0.7. Rank by descending IoU then ascending official IDs; select one. No selection by water-level differences or desired conclusion.',
            'selected':None,'limits':'Geometry-only screening, not confirmation of physical identity. High-overlap selection favours stable outlines and is not a representative temporal accuracy sample. No new DEM or volume computation. No eligible pair means stop, not relax thresholds silently.'}
    if eligible:
        best=eligible[0]
        report['selected']={k:v for k,v in best.items() if k not in ('i','j')}
        report['selected']['saved_levels']={}
        for date,row in [(DATES[0],first[best['i']]),(DATES[1],second[best['j']])]:
            p=row['props']
            report['selected']['saved_levels'][date]={k:p.get(k) for k in ['pond_id','bnd_n','h05','h50','h75','h95','hmean','qa']}
    out=ROOT/'diagnostics/results/one_two_date_pair'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True,exist_ok=False)
    (out/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    if eligible:
        with (out/'eligible_pairs.csv').open('w',encoding='utf-8-sig',newline='') as f:
            fields=[k for k in eligible[0] if k not in ('i','j')]
            writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(eligible)
        features=[]
        for date,row in [(DATES[0],first[best['i']]),(DATES[1],second[best['j']])]:
            features.append({'type':'Feature','properties':{'date':date,'pond_id':int(row['props']['pond_id'])},'geometry':row['geom'].__geo_interface__})
        # Explicit projected CRS in JSON: not advertised as RFC7946 longitude/latitude GeoJSON.
        (out/'selected_geometries.json').write_text(json.dumps({'crs_wkt':expected_crs.to_wkt(),'features':features},ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
    print('RESULT_FOLDER:',out,flush=True)


if __name__=='__main__': main()
