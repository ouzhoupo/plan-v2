"""Read-only input audit; reconstruct buffers in memory, never overwrite outputs."""
import sys
sys.dont_write_bytecode = True
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
import fiona
import yaml
import shapely
from shapely.geometry import shape
from shapely.ops import transform, unary_union
from pyproj import CRS, Transformer

ROOT = Path(__file__).resolve().parents[2]
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p, target):
    with fiona.open(p) as src:
        crs = CRS.from_user_input(src.crs_wkt or src.crs)
        tf = Transformer.from_crs(crs, target, always_xy=True)
        return [transform(tf.transform, shape(f['geometry'])) if crs != target else shape(f['geometry'])
                for f in src if f.get('geometry')]

dates = ['20240601', '20250521']
dirs = [ROOT / 'output' / d / 'pihe' / 'buffered_river' for d in dates]
hashes = [{p.name: sha(p) for p in sorted(d.iterdir()) if p.is_file()} for d in dirs]
cfg = yaml.safe_load((ROOT/'output/20240601/pihe/monitor_cfg_2024-06-01.yaml').read_text(encoding='utf-8'))
pipeline = yaml.safe_load((ROOT/'config/pipeline_auto/pihe_pipeline_auto.yaml').read_text(encoding='utf-8'))
default_buffer = float(pipeline.get('downstream', {}).get('line_buffer_m', 15.0))
with fiona.open(dirs[0]/'psh_res.shp') as src:
    target = CRS.from_user_input(src.crs_wkt or src.crs)
aoi = unary_union(read(ROOT/'data/aoi/pihe.shp', target))
rows = []
used = set()
for p in sorted((ROOT/'data/river').rglob('*.shp')):
    rule = next((k for k in cfg['buffer_rules'] if k.lower() in p.stem.lower()), None)
    # Default has no effect on retained unmatched polygon layers; record explicitly.
    buf = float(cfg['buffer_rules'][rule]) if rule else default_buffer
    result = []
    types = set()
    for g in read(p, target):
        types.add(g.geom_type)
        if g.is_empty:
            continue
        if g.geom_type in ('LineString', 'MultiLineString', 'LinearRing'):
            g = g.buffer(buf, cap_style=2, join_style=2)
        elif g.geom_type in ('Point', 'MultiPoint'):
            g = g.buffer(buf)
        if not g.intersects(aoi):
            continue
        g = shapely.make_valid(g).intersection(shapely.make_valid(aoi).buffer(max(buf, 1.0)))
        if not g.is_empty:
            result.append(g)
    if not result:
        continue
    base = re.sub(r'[^0-9A-Za-z_]+', '_', p.stem)[:40] or 'layer'
    name = base
    i = 1
    while name in used:
        name = f'{base}_{i}'
        i += 1
    used.add(name)
    saved = read(dirs[0]/f'{name}.shp', target)
    exact = len(saved) == len(result) and all(a.equals(b) for a,b in zip(saved,result))
    rows.append(dict(source=p.name, output=name+'.shp', types=sorted(types), matched_rule=rule,
                     buffer_or_aoi_margin_m=buf, features_saved=len(saved), features_rebuilt=len(result),
                     all_features_topologically_equal=exact))
report = dict(dates=dates, crs=target.to_string(), files_each=[len(h) for h in hashes],
              all_buffer_files_byte_identical=hashes[0]==hashes[1], hashes=dict(zip(dates,hashes)), mapping=rows,
              reservoir_source_hashes={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'data/reservoir').glob('psh_res.*')},
              limits=['Current retained files only; original historical reservoir snapshot unavailable.',
                      'Unmatched polygon default read from current pipeline config; AOI clipping margin is max(default,1m).',
                      'Identical input does not establish spatial correctness or quantify storage error.'])
out = ROOT/'diagnostics/results/deduction_input_audit'/datetime.now().strftime('%Y%m%d_%H%M%S')
out.mkdir(parents=True)
(out/'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k not in ('hashes','reservoir_source_hashes')},ensure_ascii=False,indent=2))
print('OUTPUT:',out/'audit.json')
