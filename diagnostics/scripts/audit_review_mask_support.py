"""Read-only point audit; no model training or production writes."""
import json
import math
from datetime import datetime
from pathlib import Path
import sys
import rasterio
from rasterio.windows import Window, transform

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code'))
import s1s2_pond_production as m

source = ROOT / 'diagnostics/results/positive_label_experiment/20260912_074812_069687/experiment.json'
experiment = json.loads(source.read_text(encoding='utf-8'))
configs = m.resolve_multi_date_configs(m.load_yaml(ROOT / 'config/pipeline_auto/pihe_pipeline_auto.yaml'))
report = {'experiment': str(source), 'dates': {}}
for date, entry in experiment['dates'].items():
    cfg = m.resolve_runtime_auto_inputs(next(c for c in configs if c['date'] == date))
    grid = m.infer_grid(cfg)
    folder = ROOT / 'output' / date.replace('-', '') / 'pihe'
    readers = {name: rasterio.open(folder / f'{name}_{date}.tif') for name in ['water_prob', 'source_flag', 'confidence', 'water_mask']}
    try:
        rows = []
        for point in entry['reviewed_validation_points']:
            row, col = readers['water_mask'].index(point['x'], point['y'])
            win = Window(col, row, 1, 1)
            inside = bool(m.rasterize_geom_mask(grid.aoi_geom, (1, 1), transform(win, readers['water_mask'].transform))[0, 0])
            values = {}
            for name, ds in readers.items():
                array = ds.read(1, window=win, masked=True)
                values[name] = None if array.size != 1 or bool(array.mask.all()) else float(array[0, 0])
                if values[name] is not None and not math.isfinite(values[name]):
                    values[name] = None
            rows.append({'sample_index': point['sample_index'], 'group': point['group'], 'inside_aoi': inside, 'saved': values, 'experiment_mask_counts': point['mask_counts']})
        report['dates'][date] = rows
        print(date, 'checked', len(rows), 'points', flush=True)
    finally:
        for ds in readers.values():
            ds.close()
out = ROOT / 'diagnostics/results/review_mask_support' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
out.mkdir(parents=True, exist_ok=False)
path = out / 'support.json'
path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
print('RESULT:', path)
