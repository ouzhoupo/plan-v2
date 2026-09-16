"""User-run: two-date bookkeeping from saved summaries; not causal attribution."""
import csv,json,math
from pathlib import Path
from datetime import datetime

ROOT=Path(__file__).resolve().parents[2]
DATES=['2024-06-01','2025-05-21']


def main():
    prior=json.loads((ROOT/'diagnostics/results/pihe_level_trend/20260915_152859_225857/summary.json').read_text(encoding='utf-8'))
    if prior['status']!='complete': raise RuntimeError('Previous volume summary incomplete')
    records=[]
    for date in DATES:
        folder=ROOT/f'output/{date.replace("-", "")}/final/pihe'
        summary=json.loads((folder/f'deduct_summary_{date}.json').read_text(encoding='utf-8'))
        with (folder/f'pond_storage_{date}.csv').open(encoding='utf-8-sig',newline='') as f:
            rows=[r for r in csv.DictReader(f) if r.get('pond_id','').strip()]
        ids=[int(r['pond_id']) for r in rows]
        if not rows or len(set(ids))!=len(ids): raise RuntimeError('Missing objects or duplicate IDs')
        if any(not r['v75_m3'].strip() for r in rows): raise RuntimeError('Missing v75; do not treat as zero')
        area=sum(float(r['area_m2']) for r in rows)
        volume=sum(float(r['v75_m3']) for r in rows)
        if not math.isclose(volume,prior['inputs'][date]['volumes_m3']['v75_m3'],abs_tol=.01):
            raise RuntimeError('Volume differs from previous check')
        if not math.isclose(area,summary['pond_area_after_min_area_m2'],abs_tol=max(.1,len(rows)*.001)) or len(rows)!=summary['pond_count_after_min_area']:
            raise RuntimeError('Final object count/area differs from saved summary')
        # The historical summary names its FINAL object totals 'after_min_area'.
        # They do not isolate the intermediate area gate.
        source=float(summary['source_water_area_m2']);aoi=float(summary['aoi_water_area_m2'])
        pre=float(summary['landuse_intersected_area_m2'])
        deducted=float(summary['deducted_union_area_m2'])
        remaining=pre-deducted;candidate=float(summary['pond_candidate_area_m2'])
        if min(source,aoi,pre,remaining,candidate,area,volume)<=0:
            raise RuntimeError('Nonpositive stage value; ratio chain not applicable')
        if not math.isclose(aoi,pre,abs_tol=.01):
            raise RuntimeError('Unexpected pre-deduction mask change; inspect configuration')
        records.append({'date':date,'source_area_m2':source,'aoi_area_m2':aoi,'pre_deduction_area_m2':pre,
                        'deducted_union_area_m2':deducted,'post_deduction_area_m2':remaining,
                        'morphology_net_area_m2':candidate-remaining,'candidate_area_m2':candidate,
                        'final_area_m2':area,'final_objects':len(rows),'v75_m3':volume,
                        'equivalent_depth_v75_div_final_area':volume/area,
                        'aoi_retention':aoi/source,'pre_deduction_retention':pre/aoi,
                        'deduction_retention':remaining/pre,'morphology_multiplier':candidate/remaining,
                        'combined_object_screen_retention':area/candidate})
    a,b=records
    stage_keys=['source_area_m2','aoi_area_m2','pre_deduction_area_m2','post_deduction_area_m2','candidate_area_m2','final_area_m2','equivalent_depth_v75_div_final_area','v75_m3']
    ratios={k:b[k]/a[k] for k in stage_keys}
    factor_keys=['aoi_retention','pre_deduction_retention','deduction_retention','morphology_multiplier','combined_object_screen_retention','equivalent_depth_v75_div_final_area']
    factors={k:b[k]/a[k] for k in factor_keys}
    reconstructed=ratios['source_area_m2']*math.prod(factors.values())
    if not math.isclose(reconstructed,ratios['v75_m3'],rel_tol=1e-10):
        raise RuntimeError('Multiplicative bookkeeping does not close')
    report={'status':'complete','dates':records,'ratios_2025_to_2024':ratios,
            'stage_multiplier_ratios':factors,'reconstructed_volume_ratio':reconstructed,
            'survey_ratio':prior['survey_ratio'],'survey_wan_m3':prior['survey_wan_m3'],
            'missing_stage_detail':'Candidate-to-final retention combines area and land-use screening (and any other enabled object operations); saved summary does not separate them.',
            'limits':'Accounting identities, not error contribution percentages. Area is not surveyed volume truth. Equivalent depth V/A is not measured depth. Ratio amplification and absolute differences are distinct. Source-to-candidate net morphology is not gross added/error area. No GIS computation or model modification.'}
    out=ROOT/'diagnostics/results/pihe_stage_trend'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True,exist_ok=False)
    (out/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    with (out/'stages.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
    print('RESULT_FOLDER:',out,flush=True)


if __name__=='__main__': main()
