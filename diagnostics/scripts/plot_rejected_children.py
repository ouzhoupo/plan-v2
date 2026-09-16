"""View all experiment children, including kept controls. No model execution."""
from pathlib import Path
from datetime import datetime
import json
import numpy as np
import rasterio
from rasterio.features import geometry_window, shapes
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from shapely.geometry import shape, Polygon
from shapely.ops import unary_union
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from build_pihe_gis_samples import transformed_geom

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'diagnostics/results/local_downstream/20260913_094837_488085'
OUT=ROOT/'diagnostics/results/child_visual_review'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
OUT.mkdir(parents=True,exist_ok=False)
report=json.loads((SOURCE/'trace.json').read_text(encoding='utf-8'))
records=[]

def outline(ax,g,color,width=1):
    for p in ([g] if g.geom_type=='Polygon' else g.geoms):
        for ring in [p.exterior,*p.interiors]:
            x,y=ring.xy; ax.plot(x,y,color=color,linewidth=width)

for item in report['objects']:
    date,pid=item['date'],item['pond_id']
    children=item['variants']['experiment']
    geoms=[shape(o['geometry']) for o in children]
    if not geoms: continue
    with rasterio.open(SOURCE/f'{date}_{pid}/experiment_candidate.tif') as ds:
        crs=ds.crs
        win=geometry_window(ds,[unary_union(geoms).buffer(180).__geo_interface__])
        trans=ds.window_transform(win); h,w=int(win.height),int(win.width)
        extent=ds.window_bounds(win); extent=(extent[0],extent[2],extent[1],extent[3])
        with rasterio.open(ROOT/f'data/s2/s2_stack_{date}.tif') as s2:
            with WarpedVRT(s2,crs=crs,transform=trans,width=w,height=h,resampling=Resampling.nearest) as vrt:
                a=vrt.read([3,2,1,5,4],masked=True).astype('float32').filled(np.nan)
    def stretch(ids):
        rgb=np.moveaxis(a[ids],0,-1); f=rgb[np.isfinite(rgb)]
        lo,hi=np.percentile(f,[2,98]) if f.size else (0,1)
        return np.nan_to_num(np.clip((rgb-lo)/max(float(hi-lo),1e-6),0,1))**.7
    rgb,false=stretch([0,1,2]),stretch([3,4,0])
    fig,axes=plt.subplots(2,2,figsize=(14,12),constrained_layout=True)
    for ax,img,title in zip(axes.flat,[rgb,false,rgb,false],['S2 RGB: no overlays','S2 SWIR / NIR / Red: no overlays','Children: red rejected / lime kept','Land-use read windows and water cells']):
        ax.imshow(img,extent=extent,interpolation='nearest'); ax.set_title(title,fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    evidence=[]
    with rasterio.open(ROOT/'data/landuse_water/PSH_土地利用水体.tif') as lu:
        for child,g in zip(children,geoms):
            color='lime' if child['landuse_kept'] else 'red'
            for ax in axes[1]: outline(ax,g,color,1.4)
            p=g.representative_point()
            axes[1,0].text(p.x,p.y,f"{child['local_id']}\n{child['area_m2']/10000:.2f} ha",fontsize=8,color='white',bbox=dict(facecolor='black',alpha=.65,pad=1))
            lg=transformed_geom(g,crs,lu.crs)
            lw=geometry_window(lu,[lg.__geo_interface__],pad_x=1,pad_y=1)
            data=lu.read(1,window=lw); lt=lu.window_transform(lw)
            rectangle=Polygon([lt*(0,0),lt*(data.shape[1],0),lt*(data.shape[1],data.shape[0]),lt*(0,data.shape[0])])
            outline(axes[1,1],transformed_geom(rectangle,lu.crs,crs),'orange',.7)
            hit=data==1
            if bool(hit.any())!=child['landuse_kept']: raise RuntimeError('Land-use decision not reproduced')
            for mapping,value in shapes(hit.astype('uint8'),mask=hit,transform=lt):
                if value==1: outline(axes[1,1],transformed_geom(shape(mapping),lu.crs,crs),'cyan',1.5)
            evidence.append({k:v for k,v in child.items() if k!='geometry'} | {'landuse_window_water_pixels':int(hit.sum())})
    for ax in axes.flat:
        ax.set_xlim(extent[0],extent[1]); ax.set_ylim(extent[2],extent[3])
    axes[1,1].legend(handles=[Patch(color='orange',label='Read window'),Patch(color='cyan',label='Land-use water outlines')],loc='lower left',fontsize=8)
    fig.suptitle(f'{date} original object {pid}: ALL area-qualified experiment children')
    path=OUT/f'{date}_{pid}_children.png'; fig.savefig(path,dpi=160); plt.close(fig)
    records.append({'date':date,'original_id':pid,'image':str(path),'children':evidence,'caveat':'Display stack, not production clear composite; visual labels pending. Local IDs are not historical pond IDs.'})
    print(date,pid,'complete',flush=True)
path=OUT/'review.json'; path.write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
print('RESULT:',path)
