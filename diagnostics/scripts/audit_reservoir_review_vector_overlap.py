"""Trace reviewed components to reservoir features; no rescue or volume change."""
import csv
import json
from datetime import datetime
from pathlib import Path
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import shapes
from rasterio.windows import Window
from scipy import ndimage
from shapely.geometry import shape
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parents[2]
REVIEW = ROOT / 'diagnostics/results/second_round_reservoir_screening_holdout_review/20260928_152854_549492'

def main():
    with (REVIEW/'review_manifest.csv').open(encoding='utf-8-sig', newline='') as f:
        targets = list(csv.DictReader(f))
    with rasterio.open(ROOT/'diagnostics/results/second_round_deduction_morphology_split/20260926_203928_115516/c4_pre_deduction_no_morph.tif') as src:
        labels, _ = ndimage.label(src.read(1)>0, structure=np.array([[0,1,0],[1,1,1],[0,1,0]]))
        slices = ndimage.find_objects(labels)
        crs = src.crs
        objects = []
        for row in targets:
            cid = int(row['component_id'])
            sl = slices[cid-1]
            mask = (labels[sl] == cid).astype('uint8')
            transform = src.window_transform(Window(sl[1].start, sl[0].start, mask.shape[1], mask.shape[0]))
            geom = unary_union([shape(g) for g,v in shapes(mask, mask=mask>0, transform=transform)])
            if abs(geom.area-float(row['area_m2'])) > 0.01:
                raise RuntimeError(f'Component area mismatch: {cid}')
            objects.append((row, geom))
    group = json.loads((ROOT/'diagnostics/results/second_round_deduction_groups/20260926_204740_877320/summary.json').read_text(encoding='utf-8'))
    out = ROOT/'diagnostics/results/second_round_reservoir_vector_overlap'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True)
    records = []
    metadata = []
    for path in group['groups']['reservoir']['sources']:
        frame = gpd.read_file(path)
        if frame.crs is None:
            raise RuntimeError(f'Missing CRS: {path}')
        frame = frame.to_crs(crs)
        metadata.append(dict(path=path, features=len(frame), columns=list(frame.columns)))
        for row, geom in objects:
            hits = list(frame.sindex.query(geom, predicate='intersects'))
            if not hits:
                records.append(dict(component_id=row['component_id'],source=path,feature_position='',feature_area_m2='',intersection_m2=0,component_overlap_fraction=0,component_fraction_of_feature='',attributes='{}'))
            for pos in hits:
                feature = frame.iloc[int(pos)]
                polygon = feature.geometry
                if not polygon.is_valid:
                    polygon = polygon.buffer(0)
                intersection = geom.intersection(polygon).area
                attrs = {k:str(v) for k,v in feature.items() if k != frame.geometry.name}
                records.append(dict(component_id=row['component_id'],source=path,feature_position=int(pos),feature_area_m2=polygon.area,intersection_m2=intersection,component_overlap_fraction=intersection/geom.area,component_fraction_of_feature=intersection/polygon.area if polygon.area else None,attributes=json.dumps(attrs,ensure_ascii=False)))
    with (out/'object_vector_matches.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    (out/'summary.json').write_text(json.dumps(dict(status='complete',reviewed_objects=len(objects),sources=metadata,match_rows=len(records),limits=['Vector attributes do not independently establish survey inclusion.','Geometric intersections differ from raster cell inclusion at boundaries.','Duplicate source layers must not be summed.','Small component within large reservoir may indicate upstream fragmentation; overlap alone does not prove target class.']),ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'RESULT_FOLDER: {out}')
    print('STATUS: complete')

if __name__ == '__main__':
    main()
