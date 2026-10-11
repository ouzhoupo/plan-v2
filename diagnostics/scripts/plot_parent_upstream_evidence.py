"""Four parent-scale diagnostic maps; fixed selection, no model changes."""
import csv
import json
from pathlib import Path
from datetime import datetime
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import geometry_window,rasterize
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,BoundaryNorm
from plot_2024_c4_pre_deduction_component_review import stretch

ROOT=Path(__file__).resolve().parents[2]
UP=ROOT/'diagnostics/results/combined_upstream_c0_c4/20260922_180153_112712/2024-06-01'

def main():
    table=ROOT/'diagnostics/results/second_round_parent_upstream_evidence/20260928_155622_869718/parent_evidence.csv'
    with table.open(encoding='utf-8-sig',newline='') as f:
        rows=[r for r in csv.DictReader(f) if r['reviewed_component_ids'] in ('12808','13210','53549','10612')]
    if len(rows)!=4: raise RuntimeError('Expected four fixed parent scopes')
    frame=gpd.read_file(ROOT/'data/reservoir/psh_res.shp')
    out=ROOT/'diagnostics/results/second_round_parent_evidence_review'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True)
    with rasterio.open(UP/'c4_water_mask.tif') as ref:
        frame=frame.to_crs(ref.crs)
        for r in rows:
            cid=r['reviewed_component_ids']; geom=frame.geometry.iloc[int(r['parent_position'])]
            if not geom.is_valid: geom=geom.buffer(0)
            win=geometry_window(ref,[geom],pad_x=10,pad_y=10)
            transform=ref.window_transform(win); h,w=int(win.height),int(win.width)
            inside=rasterize([geom],out_shape=(h,w),transform=transform)>0
            c4=ref.read(1,window=win)>0
            arrays={}
            for key,name in [('prob','c2_fused_probability.tif'),('source','c2_source_flag.tif')]:
                with rasterio.open(UP/name) as src:
                    if src.shape!=ref.shape or src.crs!=ref.crs or not src.transform.almost_equals(ref.transform): raise RuntimeError('Grid mismatch')
                    arrays[key]=src.read(1,window=win)
            with rasterio.open(ROOT/'data/s2/s2_stack_2024-06-01.tif') as src:
                with WarpedVRT(src,crs=ref.crs,transform=transform,width=w,height=h,resampling=Resampling.nearest) as vrt:
                    bands=vrt.read([3,2,1],masked=True).astype('float32').filled(np.nan)
            rgb=np.dstack([stretch(b) for b in bands])
            fig,ax=plt.subplots(2,2,figsize=(13,12),constrained_layout=True)
            ax[0,0].imshow(rgb); ax[0,0].set_title('S2 RGB | yellow = reservoir comparison boundary')
            ax[0,1].imshow(rgb)
            overlay=np.zeros((h,w,4),dtype='float32');overlay[inside&c4]=(0,1,1,.65)
            overlay[inside&~c4&(arrays['prob']>=.7)]=(1,0,0,.8)
            ax[0,1].imshow(overlay);ax[0,1].set_title('Cyan: C4 water | red: C4 nonwater with C2 P>=0.7')
            im=ax[1,0].imshow(arrays['prob'],vmin=0,vmax=1,cmap='viridis');fig.colorbar(im,ax=ax[1,0],shrink=.7);ax[1,0].set_title('C2 probability (not reference truth)')
            cmap=ListedColormap(['black','orange','dodgerblue','limegreen','purple'])
            im=ax[1,1].imshow(arrays['source'],cmap=cmap,norm=BoundaryNorm(np.arange(-.5,5),5),interpolation='nearest')
            cb=fig.colorbar(im,ax=ax[1,1],ticks=range(5),shrink=.7);cb.ax.set_yticklabels(['0 outside/none','1 SAR only','2 optical only','3 both','4 prior only'])
            ax[1,1].set_title('C2 source pathway')
            for a in ax.flat:
                a.contour(inside.astype('uint8'),levels=[.5],colors='yellow',linewidths=.8);a.set_xticks([]);a.set_yticks([])
            fig.suptitle(f'Parent {r["parent_position"]} | reviewed component {cid} | 2024-06-01')
            fig.savefig(out/f'parent_{cid}.png',dpi=150);plt.close(fig)
            print(f'Wrote parent_{cid}.png',flush=True)
    (out/'summary.json').write_text(json.dumps(dict(status='complete',selected_components=[r['reviewed_component_ids'] for r in rows],limits=['Selection is diagnostic, not independent validation.','Reservoir boundary is not contemporaneous true water extent.','Red pixels are evidence conflicts, not confirmed omission.','C2 source is not C4 temporal-fill provenance.']),indent=2),encoding='utf-8')
    print(f'RESULT_FOLDER: {out}');print('STATUS: complete')

if __name__=='__main__': main()
