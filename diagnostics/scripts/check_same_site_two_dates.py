"""User-run: fixed-footprint two-date presence and saved-level comparison only."""
import sys
sys.dont_write_bytecode=True
import json, csv
from pathlib import Path
from datetime import datetime
import numpy as np
import fiona, rasterio
from rasterio.features import geometry_window, geometry_mask
from rasterio.windows import from_bounds
from shapely.geometry import shape, mapping
from shapely.ops import transform
from pyproj import CRS, Transformer

ROOT=Path(__file__).resolve().parents[2]
DATES=['2024-06-01','2025-05-21']


def main():
    reference_path=ROOT/'output/20250521/final/pihe/pond_storage_2025-05-21.gpkg'
    matches=[]
    with fiona.open(reference_path,layer='ponds') as src:
        ref_crs=CRS.from_user_input(src.crs_wkt or src.crs)
        for f in src:
            if int(f['properties']['pond_id'])==31854:
                matches.append((shape(f['geometry']),dict(f['properties'])))
    if len(matches)!=1: raise RuntimeError('Reference object missing or ambiguous')
    reference,props=matches[0]
    if abs(reference.area-44400)>.01 or props.get('h75')!=34.0:
        raise RuntimeError('Reference differs from prior verified object')
    records=[]; object_rows=[]
    for date in DATES:
        day=ROOT/'output'/date.replace('-','')
        with rasterio.open(day/f'pihe/water_mask_{date}.tif') as src:
            if CRS.from_user_input(src.crs)!=ref_crs: raise RuntimeError('Unexpected source CRS')
            win=geometry_window(src,[mapping(reference)])
            trans=src.window_transform(win); bounds=src.window_bounds(win)
            water=src.read(1,window=win)>0
            inside=geometry_mask([mapping(reference)],out_shape=water.shape,transform=trans,invert=True,all_touched=False)
            source_valid=src.read_masks(1,window=win)>0
            pixel_area=abs(trans.a*trans.e-trans.b*trans.d)
        with rasterio.open(day/f'final/pihe/pond_water_mask_{date}.tif') as src:
            cw=from_bounds(*bounds,src.transform).round_offsets().round_lengths()
            if CRS.from_user_input(src.crs)!=ref_crs or not src.window_transform(cw).almost_equals(trans):
                raise RuntimeError('Candidate grid mismatch')
            candidate=src.read(1,window=cw,boundless=True,fill_value=0)>0
            if candidate.shape!=inside.shape: raise RuntimeError('Candidate shape mismatch')
        source_counts={}
        with rasterio.open(day/f'pihe/source_flag_{date}.tif') as src:
            sw=from_bounds(*bounds,src.transform).round_offsets().round_lengths()
            if CRS.from_user_input(src.crs)!=ref_crs or not src.window_transform(sw).almost_equals(trans):
                raise RuntimeError('Source flag grid mismatch')
            flags=src.read(1,window=sw)
            for value,count in zip(*np.unique(flags[inside],return_counts=True)):
                source_counts[str(int(value))]=int(count)
        overlaps=[]
        with fiona.open(day/f'final/pihe/pond_storage_{date}.gpkg',layer='ponds') as src:
            crs=CRS.from_user_input(src.crs_wkt or src.crs)
            tf=Transformer.from_crs(crs,ref_crs,always_xy=True)
            for f in src:
                if not f['geometry']: continue
                geom=shape(f['geometry'])
                if crs!=ref_crs: geom=transform(tf.transform,geom)
                if not geom.intersects(reference): continue
                intersection=geom.intersection(reference).area
                if intersection<=.01: continue
                p=dict(f['properties'])
                row={'date':date,'pond_id':int(p['pond_id']),'object_area_m2':geom.area,
                     'overlap_reference_m2':intersection,'reference_covered_fraction':intersection/reference.area,
                     'object_overlap_fraction':intersection/geom.area,
                     'intersection_over_union':intersection/geom.union(reference).area,
                     'h50':p.get('h50'),'h75':p.get('h75'),'h95':p.get('h95'),'bnd_n':p.get('bnd_n'),'qa':p.get('qa')}
                overlaps.append(row);object_rows.append(row)
        overlaps.sort(key=lambda r:r['overlap_reference_m2'],reverse=True)
        if date=='2025-05-21' and not any(r['pond_id']==31854 and abs(r['intersection_over_union']-1)<1e-9 for r in overlaps):
            raise RuntimeError('Reference date identity check failed')
        records.append({'date':date,'reference_pixel_count':int(inside.sum()),'pixel_area_m2':pixel_area,
                        'input_water_pixels_inside_reference':int((inside & water).sum()),
                        'candidate_pixels_inside_reference':int((inside & candidate).sum()),
                        'source_mask_valid_pixels_inside_reference':int((inside & source_valid).sum()),
                        'source_flag_counts_inside_reference':source_counts,
                        'overlapping_final_objects':overlaps,
                        'comparison_availability':'no_final_object_no_h75_comparison' if not overlaps else 'overlap_found_identity_requires_review'})
        print(date,'complete; overlapping final objects:',len(overlaps),flush=True)
    out=ROOT/'diagnostics/results/same_site_two_dates'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True,exist_ok=False)
    report={'status':'complete','reference_date':'2025-05-21','reference_pond_id':31854,'reference_area_m2':reference.area,
            'dates':records,'limits':'Fixed 2025 footprint is not 2024 truth. Overlap does not establish same physical pond; retain all overlaps for review. No object means no comparable h75, not zero water level. Source raster mask may encode nonwater as nodata, so it is not sensor availability. No new DEM sampling, volume calculation or parameter change.'}
    (out/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    if object_rows:
        with (out/'overlapping_objects.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(object_rows[0]));writer.writeheader();writer.writerows(object_rows)
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
    print('RESULT_FOLDER:',out,flush=True)


if __name__=='__main__': main()
