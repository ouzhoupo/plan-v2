"""Compare frozen C0-C4 coverage within reviewed parent reservoir polygons."""
import csv
import json
from datetime import datetime
from pathlib import Path
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import geometry_window, rasterize

ROOT=Path(__file__).resolve().parents[2]
MATCH=ROOT/'diagnostics/results/second_round_reservoir_vector_overlap/20260928_153258_290974'
UP=ROOT/'diagnostics/results/combined_upstream_c0_c4/20260922_180153_112712/2024-06-01'

def main():
    with (MATCH/'object_vector_matches.csv').open(encoding='utf-8-sig',newline='') as f:
        matches=[r for r in csv.DictReader(f) if '/data/reservoir/' in r['source'].replace('\\','/')]
    parents={}
    for r in matches:
        parents.setdefault(int(r['feature_position']),[]).append(r['component_id'])
    frame=gpd.read_file(ROOT/'data/reservoir/psh_res.shp')
    records=[]
    from contextlib import ExitStack
    with ExitStack() as stack:
        sources={v:stack.enter_context(rasterio.open(UP/f'{v}_water_mask.tif')) for v in ('c0','c1','c2','c3','c4')}
        ref=sources['c0']
        frame=frame.to_crs(ref.crs)
        for s in sources.values():
            if s.shape!=ref.shape or s.crs!=ref.crs or not s.transform.almost_equals(ref.transform):
                raise RuntimeError('Upstream grids differ')
        pixel_area=abs(ref.transform.a*ref.transform.e-ref.transform.b*ref.transform.d)
        for pos,ids in parents.items():
            geom=frame.geometry.iloc[pos]
            if not geom.is_valid: geom=geom.buffer(0)
            win=geometry_window(ref,[geom])
            inside=rasterize([geom],out_shape=(int(win.height),int(win.width)),transform=ref.window_transform(win),all_touched=False)>0
            n=int(inside.sum())
            if not n: raise RuntimeError(f'Empty parent raster: {pos}')
            row=dict(parent_position=pos,reviewed_component_ids=';'.join(ids),vector_area_m2=float(geom.area),raster_scope_m2=n*pixel_area)
            masks={}
            valid={}
            for v,s in sources.items():
                data=s.read(1,window=win,masked=True)
                valid[v]=~np.ma.getmaskarray(data)
                masks[v]=(data.filled(0)>0)&inside
                row[f'{v}_water_m2']=float(masks[v].sum()*pixel_area)
                row[f'{v}_water_fraction_of_scope']=float(masks[v].sum()/n)
                row[f'{v}_mask_storage_valid_fraction_of_scope']=float((valid[v]&inside).sum()/n)
            for old,new in zip(('c0','c1','c2','c3'),('c1','c2','c3','c4')):
                row[f'{new}_vs_{old}_added_m2']=float((masks[new]&~masks[old]).sum()*pixel_area)
                row[f'{new}_vs_{old}_removed_m2']=float((masks[old]&~masks[new]).sum()*pixel_area)
            records.append(row)
    out=ROOT/'diagnostics/results/second_round_parent_upstream_coverage'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True)
    with (out/'parent_stage_coverage.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0])); w.writeheader(); w.writerows(records)
    (out/'summary.json').write_text(json.dumps(dict(status='complete',parent_features=len(records),reviewed_components=len(matches),date='2024-06-01',limits=['Reservoir vector polygons are comparison scopes, not contemporaneous water truth.','Mask storage validity is not sensor observation validity; binary-mask nodata may equal zero.','Low coverage can reflect drawdown, invalid observations, vector mismatch or recognition error; it is not a measured omission rate.','C0-C4 stages are frozen first-round upstream variants before vector deduction.','Parent polygons may overlap; do not sum their areas as an AOI total.','No thresholds changed, no water restored and no volume calculated.']),ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'RESULT_FOLDER: {out}')
    print('STATUS: complete')

if __name__=='__main__': main()
