"""Paired local closing ablation on original output block boundaries."""
import sys
sys.dont_write_bytecode=True
import json, math
from pathlib import Path
from datetime import datetime
import numpy as np
import rasterio, yaml
from rasterio.windows import Window, from_bounds
from scipy import ndimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code'))
import pond_storage_monitor as m
OUT=ROOT/'diagnostics/results/fixed_site_closing'/datetime.now().strftime('%Y%m%d_%H%M%S')
OUT.mkdir(parents=True)
sites=json.loads((ROOT/'diagnostics/results/deduction_fixed_sites/20260914_164745/review.json').read_text())['sites']
records=[]
for site in sites:
    fig,axs=plt.subplots(2,3,figsize=(12,8),constrained_layout=True)
    for row,(date,old) in enumerate(site['dates'].items()):
        day=ROOT/'output'/date.replace('-','')
        cfg=yaml.safe_load((day/f'final/pihe/resolved_monitor_cfg_{date}.yaml').read_text(encoding='utf-8'))
        assert not any(cfg.get(k) for k in ['landuse_water_hard_mask','cropland_mask_tif','extra_vectors','open_radius_px','fill_holes'])
        assert cfg['close_radius_px']==1
        bs=int(cfg['block_size'])
        with rasterio.open(day/f'final/pihe/pond_water_mask_{date}.tif') as saved:
            core=from_bounds(*old['bounds'],saved.transform).round_offsets().round_lengths()
            c=int(core.col_off)//bs*bs; r=int(core.row_off)//bs*bs
            ec=min(saved.width,math.ceil((core.col_off+core.width)/bs)*bs)
            er=min(saved.height,math.ceil((core.row_off+core.height)/bs)*bs)
            win=Window(c,r,ec-c,er-r)
            bounds=saved.window_bounds(win); trans=saved.window_transform(win); crs=saved.crs
            historical=saved.read(1,window=win)>0
        layers=m.build_deduct_layers(river_dir=str(day/'pihe/buffered_river'),reservoir_shp=str(ROOT/'data/reservoir/psh_res.shp'),extra_vectors=[],target_crs=crs,bbox_filter=bounds,line_buffer_m=0)
        aoi=m.read_union_geometry(str(ROOT/'data/aoi/pihe.shp'),crs)
        arrays=[]
        for radius in [0,1]:
            dest=OUT/f'{site["site"]}_{date}_close{radius}.tif'
            m.prepare_water_mask(str(day/f'pihe/water_mask_{date}.tif'),str(dest),process_bounds=bounds,aoi_geom=aoi,deduct_layers=layers,water_value=cfg.get('water_value'),respect_source_mask=bool(cfg.get('respect_source_mask',False)),block_size=bs,close_radius_px=radius,open_radius_px=0,fill_holes=False,all_touched=bool(cfg['all_touched']))
            with rasterio.open(dest) as ds:
                assert ds.transform.almost_equals(trans)
                arrays.append(ds.read(1)>0)
        before,after=arrays
        mismatch=int((after!=historical).sum())
        assert mismatch==0, (site['site'],date,mismatch)
        union=np.zeros(before.shape,bool)
        for layer in layers:
            union |= m.rasterize_geoms_for_window(layer.geoms,before.shape,trans,all_touched=bool(cfg['all_touched']))>0
        y,x=int(core.row_off)-r,int(core.col_off)-c
        sl=np.s_[y:y+int(core.height),x:x+int(core.width)]
        b,a=before[sl],after[sl]; added=a & ~b; removed=b & ~a
        lb,nb=ndimage.label(before); la,na=ndimage.label(after)
        # Local patch component IDs; no claim about global objects or final area filtering.
        merges=[]
        for aid in np.unique(la[sl][added]):
            if aid==0: continue
            ids=np.unique(lb[la==aid]); ids=ids[ids>0]
            if len(ids)>1: merges.append({'after_patch_component':int(aid),'before_patch_components':ids.tolist()})
        yy,xx=np.indices(before.shape)
        edge=(yy%bs==0)|(yy%bs==bs-1)|(xx%bs==0)|(xx%bs==bs-1)
        rec={'site':site['site'],'date':date,'baseline_mismatch_pixels_entire_patch':mismatch,'patch_window':[c,r,ec-c,er-r], 'core_before_pixels':int(b.sum()),'core_after_pixels':int(a.sum()),'added_pixels':int(added.sum()),'removed_pixels':int(removed.sum()),'added_inside_deduction_union_pixels':int((added & union[sl]).sum()),'removed_on_processing_block_edge_pixels':int((removed & edge[sl]).sum()),'local_merge_groups_touching_core_additions':merges,'pixel_area_m2':100,'limits':'Patch connectivity only, four-neighbour; no final object counts or volume estimates.'}
        records.append(rec)
        diff=np.zeros((*a.shape,3)); diff[b&a]=[.5,.5,.5]; diff[added]=[1,0,1]; diff[removed]=[1,.3,0]
        for ax,img,title in zip(axs[row],[b,a,diff],['Closing OFF','Closing ON = saved candidate','Magenta added / orange removed / grey retained']):
            ax.imshow(img,interpolation='nearest',cmap='Blues' if img.ndim==2 else None)
            ax.set_title(date+' | '+title,fontsize=8); ax.set_xticks([]);ax.set_yticks([])
        print(json.dumps(rec),flush=True)
    fig.suptitle(site['site']+' | original fixed window; computation on complete production blocks')
    fig.savefig(OUT/f'{site["site"]}.png',dpi=140);plt.close(fig)
(OUT/'audit.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
print('OUTPUT:',OUT,flush=True)
