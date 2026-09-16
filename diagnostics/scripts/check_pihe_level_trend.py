"""User-run: saved water-level scenarios versus two-date surveyed trend; CSV only."""
import csv,json,math
from pathlib import Path
from datetime import datetime

ROOT=Path(__file__).resolve().parents[2]
DATES=['2024-06-01','2025-05-21']
# Previously verified survey totals, recorded in the root research state/logs.
HUMAN_WAN_M3={'2024-06-01':17058.0,'2025-05-21':9174.0}
FIELDS=['v05_m3','v50_m3','v75_m3','v95_m3','vmean_m3']


def main():
    inputs={}
    for date in DATES:
        path=ROOT/f'output/{date.replace("-", "")}/final/pihe/pond_storage_{date}.csv'
        with path.open(encoding='utf-8-sig',newline='') as f:
            reader=csv.DictReader(f)
            if not set(['pond_id','area_m2',*FIELDS]).issubset(reader.fieldnames or []):
                raise RuntimeError('Required columns missing')
            rows=[];seen=set();excluded=0
            for r in reader:
                pid=r.get('pond_id','').strip()
                if not pid: excluded+=1;continue  # Exclude aggregate footer explicitly.
                try: number=int(pid)
                except ValueError: raise RuntimeError(f'Unexpected nonempty pond_id: {pid}')
                if number in seen: raise RuntimeError('Duplicate object ID')
                seen.add(number);rows.append(r)
        if not rows: raise RuntimeError('No objects')
        sums={k:0.0 for k in FIELDS};missing={k:0 for k in FIELDS}
        for r in rows:
            for k in FIELDS:
                value=r[k].strip()
                if not value: missing[k]+=1;continue
                v=float(value)
                if not math.isfinite(v) or v<0: raise RuntimeError('Invalid saved volume')
                sums[k]+=v
        inputs[date]={'file':str(path),'objects':len(rows),'excluded_blank_id_rows':excluded,
                      'area_m2':sum(float(r['area_m2']) for r in rows),'volumes_m3':sums,'missing_counts':missing}
    scenarios=[]
    human_delta=HUMAN_WAN_M3[DATES[1]]-HUMAN_WAN_M3[DATES[0]]
    for field in FIELDS:
        values=[inputs[d]['volumes_m3'][field]/10000 for d in DATES]
        complete=all(inputs[d]['missing_counts'][field]==0 for d in DATES)
        delta=values[1]-values[0]
        scenarios.append({'scenario':field,'total_2024_wan_m3':values[0],'total_2025_wan_m3':values[1],
                          'delta_wan_m3':delta,'ratio_2025_to_2024':values[1]/values[0] if values[0]>0 else None,
                          'all_object_volumes_present':complete,
                          'same_direction_as_survey':bool(delta*human_delta>0) if complete else None,
                          'bias_2024_wan_m3':values[0]-HUMAN_WAN_M3[DATES[0]],
                          'bias_2025_wan_m3':values[1]-HUMAN_WAN_M3[DATES[1]]})
    report={'status':'complete','scope':'Pihe, two previously selected dates, existing water-level scenarios only. No new GIS calculation or parameter fitting.',
            'survey_wan_m3':HUMAN_WAN_M3,'survey_ratio':HUMAN_WAN_M3[DATES[1]]/HUMAN_WAN_M3[DATES[0]],
            'survey_provenance':'Previously verified totals recorded in project research records; workbook not re-read in this script.',
            'inputs':inputs,'scenarios':scenarios,
            'limits':'Two-date direction screen, not multi-date accuracy validation or stage error contribution. Missing volumes remain flagged; partial totals are not confirmed full estimates. Scenario differences measure model sensitivity, not truth. Do not select a quantile solely to fit these dates.'}
    out=ROOT/'diagnostics/results/pihe_level_trend'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True,exist_ok=False)
    (out/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    with (out/'scenarios.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(scenarios[0]));writer.writeheader();writer.writerows(scenarios)
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
    print('RESULT_FOLDER:',out,flush=True)


if __name__=='__main__': main()
