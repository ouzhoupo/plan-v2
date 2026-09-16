"""User-run: small-window imagery for the already selected unchanged pair."""
import sys
sys.dont_write_bytecode=True
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import fiona, rasterio
from rasterio.windows import from_bounds
from rasterio.features import geometry_mask, geometry_window
from shapely.geometry import shape
from shapely.ops import transform as transform_geometry
from pyproj import CRS, Transformer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

ROOT=Path(__file__).resolve().parents[2]
PREVIOUS=ROOT/'diagnostics/results/one_two_date_pair/20260915_143901_705294'


def draw(ax,g):
    for p in ([g] if g.geom_type=='Polygon' else g.geoms):
        for ring in [p.exterior,*p.interiors]:
            ax.plot(*ring.xy,color='yellow',linewidth=1)


def main():
    previous=json.loads((PREVIOUS/'summary.json').read_text(encoding='utf-8'))
    pair=previous.get('selected')
    if previous['status']!='complete' or not pair or (pair['pond_id_2024'],pair['pond_id_2025'])!=(9823,62203):
        raise RuntimeError('Expected the previously selected pair; no automatic reselection')
    dates=['2024-06-01','2025-05-21']; geoms=[]; crs=None
    for date,pid in zip(dates,[9823,62203]):
        found=[]
        with fiona.open(ROOT/f'output/{date.replace("-", "")}/final/pihe/pond_storage_{date}.gpkg',layer='ponds') as src:
            current=CRS.from_user_input(src.crs_wkt or src.crs)
            if crs is not None and current!=crs: raise RuntimeError('Official CRS differs')
            crs=current
            for f in src:
                if int(f['properties']['pond_id'])==pid: found.append(shape(f['geometry']))
        if len(found)!=1: raise RuntimeError('Missing or ambiguous official object')
        geoms.append(found[0])
    if not geoms[0].equals(geoms[1]) or abs(geoms[0].area-1300)>.01:
        raise RuntimeError('Previously matched geometry changed')
    bounds=geoms[0].buffer(150).bounds
    images=[]; records=[]
    for date,g in zip(dates,geoms):
        with rasterio.open(ROOT/f'data/s2/s2_stack_{date}.tif') as src:
            if CRS.from_user_input(src.crs)!=crs or src.count!=7:
                raise RuntimeError('Unexpected S2 CRS or band layout')
            w=from_bounds(*bounds,src.transform).round_offsets().round_lengths()
            a=src.read(window=w,masked=True).astype('float32').filled(np.nan)
            trans=src.window_transform(w); actual=src.window_bounds(w)
            inside=geometry_mask([g.__geo_interface__],out_shape=a.shape[1:],transform=trans,invert=True,all_touched=False)
        # Each date is displayed at its own geographic extent; no pixelwise subtraction.
        scl=a[6][inside]; scl=scl[np.isfinite(scl)]
        flags_path=ROOT/f'output/{date.replace("-", "")}/pihe/source_flag_{date}.tif'
        with rasterio.open(flags_path) as src:
            if src.crs is None: raise RuntimeError('Source flag has no CRS')
            flag_crs=CRS.from_user_input(src.crs)
            fg=g if flag_crs==crs else transform_geometry(Transformer.from_crs(crs,flag_crs,always_xy=True).transform,g)
            fw=geometry_window(src,[fg.__geo_interface__])
            flags=src.read(1,window=fw)
            flag_trans=src.window_transform(fw)
            flag_inside=geometry_mask([fg.__geo_interface__],out_shape=flags.shape,transform=flag_trans,invert=True,all_touched=False)
        record={'date':date,'official_pond_id':9823 if date==dates[0] else 62203,
                'inside_pixel_count':int(inside.sum()),'scl_valid_pixels':int(scl.size),
                'scl_counts':{str(int(v)):int(n) for v,n in zip(*np.unique(scl,return_counts=True))},
                'source_flag_inside_pixel_count':int(flag_inside.sum()),
                'source_flag_counts':{str(int(v)):int(n) for v,n in zip(*np.unique(flags[flag_inside],return_counts=True))},
                's2_window_transform':list(trans),'source_flag_window_transform':list(flag_trans),
                'source_flag_crs':flag_crs.to_string(),
                'counting_note':'SCL and source flags counted independently on native grids; totals need not agree. No resampling or shared pixel mask.',
                'bounds':list(actual),'band_finite_pixels_inside':{str(k+1):int(np.count_nonzero(inside & np.isfinite(a[k]))) for k in range(7)}}
        records.append(record);images.append({'array':a,'transform':trans,'extent':(actual[0],actual[2],actual[1],actual[3])})
    # The same display stretch across dates; display only, no classification changes.
    stretches=[]
    for indices in [[2,1,0],[4,3,2]]:
        vals=np.concatenate([im['array'][indices].ravel() for im in images]); vals=vals[np.isfinite(vals)]
        lo,hi=np.percentile(vals,[2,98]) if vals.size else (0,1)
        stretches.append((float(lo),float(hi)))
    fig,axes=plt.subplots(2,3,figsize=(12,8),constrained_layout=True)
    scl_cmap=ListedColormap(['black','dimgray','darkslategray','saddlebrown','forestgreen','tan','royalblue','gray','silver','white','cyan','pink'])
    scl_norm=BoundaryNorm(np.arange(-.5,12.5,1),scl_cmap.N)
    for row,(date,g,im) in enumerate(zip(dates,geoms,images)):
        a=im['array']
        for col,indices in enumerate([[2,1,0],[4,3,2]]):
            lo,hi=stretches[col]
            rgb=np.moveaxis(a[indices],0,-1)
            missing=~np.isfinite(rgb).all(axis=2)
            rgb=np.nan_to_num(np.clip((rgb-lo)/max(hi-lo,1e-6),0,1))**.7
            rgb[missing]=[.45,.45,.45]
            axes[row,col].imshow(rgb,extent=im['extent'],interpolation='nearest')
        plot=axes[row,2].imshow(np.ma.masked_invalid(a[6]),extent=im['extent'],cmap=scl_cmap,norm=scl_norm,interpolation='nearest')
        for ax,title in zip(axes[row],['RGB B4/B3/B2','SWIR/NIR/red B11/B8/B4','SCL raw class codes']):
            draw(ax,g);ax.set_title(date+' | '+title,fontsize=9);ax.set_xticks([]);ax.set_yticks([])
            ax.set_xlim(bounds[0],bounds[2]);ax.set_ylim(bounds[1],bounds[3])
    fig.colorbar(plot,ax=list(axes[:,2]),ticks=range(12),label='SCL code')
    fig.suptitle('Fixed pair 9823 / 62203 | yellow: official outline | same stretch across dates')
    out=ROOT/'diagnostics/results/selected_pair_images'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True,exist_ok=False)
    fig.savefig(out/'pair_images.png',dpi=160);plt.close(fig)
    report={'status':'complete','geometry_equal':True,'object_area_m2':1300,'dates':records,
            'display_stretches':stretches,'limits':'Small 13-pixel object; original S2 imagery and SCL are visual context, not ground truth or production clear-sky composite. Same geometry/h75 does not establish same real water level. No DEM or volume computation.'}
    (out/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('RESULT_FOLDER:',out,flush=True)


if __name__=='__main__': main()
