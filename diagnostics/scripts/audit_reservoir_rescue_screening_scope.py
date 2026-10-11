"""Exploratory screening sensitivity; no masks or volumes are modified."""
import csv
import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'diagnostics/results/second_round_reservoir_component_features/20260928_152002_125810'


def write_csv(path, rows, fields):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    with (SOURCE / 'reservoir_affected_component_features.csv').open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    assert len({r['component_id'] for r in rows}) == len(rows)
    out = ROOT / 'diagnostics/results/second_round_reservoir_screening_scope' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True)
    summaries = []
    for maximum in (20000, 200000):
        for fill in (0.4, 0.5):
            name = f'area2000_{maximum}_fill{int(fill * 100)}'
            selected = [r for r in rows if
                        2000 <= float(r['area_m2']) <= maximum
                        and float(r['bbox_fill_fraction']) >= fill
                        and float(r['bbox_elongation']) <= 2
                        and float(r['lake_involved_pixels']) == 0
                        and float(r['probability_p50'] or '-1') >= 0.7]
            write_csv(out / f'{name}_candidates.csv', selected, list(rows[0]))
            for state in ('True', 'False'):
                subset = [r for r in selected if r['fully_removed'] == state]
                labeled = [r for r in subset if r['proxy_label'] != 'unreviewed']
                summaries.append(dict(
                    screening=name, fully_removed=state, objects=len(subset),
                    original_area_km2=sum(float(r['area_m2']) for r in subset)/1e6,
                    removed_area_km2=sum(float(r['removed_area_m2']) for r in subset)/1e6,
                    unseen_objects=sum(r['proxy_label']=='unreviewed' for r in subset),
                    known_labels=json.dumps({r['component_id']:r['proxy_label'] for r in labeled}, ensure_ascii=False)))
    write_csv(out / 'screening_comparison.csv', summaries, list(summaries[0]))
    report = dict(status='complete', source=str(SOURCE), source_objects=len(rows),
                  comparisons=summaries,
                  interpretation=[
                      'Exploratory sensitivity screens, not validated rescue rules.',
                      'Area and shape cutoffs are diagnostic choices, not literature or administrative pond definitions.',
                      'All 18 previously reviewed reservoir objects are design evidence; future validation must exclude them.',
                      'Removed area is screening capacity, not confirmed error or storage gain.',
                      'Small reservoirs may remain excluded targets even when visually isolated; inventory scope must be checked.',
                      'C2 probability measures upstream water evidence, not pond membership; C4 may include later temporal fills.'
                  ])
    (out / 'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'RESULT_FOLDER: {out}')
    print('STATUS: complete')


if __name__ == '__main__':
    main()
