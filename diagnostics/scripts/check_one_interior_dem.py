"""User-run: one official object's interior DEM distribution; no volume calculation."""
import sys
sys.dont_write_bytecode = True
import csv, json
from pathlib import Path
from datetime import datetime
import numpy as np
import fiona, rasterio
from rasterio.features import geometry_window, rasterize
from shapely.geometry import shape, mapping
from shapely.ops import transform
from pyproj import CRS, Transformer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

ROOT=Path(__file__).resolve().parents[2]
DATE='2025-05-21'


def main():
    previous=json.loads((ROOT/'diagnostics/results/one_boundary_h75/20260915_085911_220264/summary.json').read_text())
    if not previous.get('saved_levels_all_match') or previous['official_pond_id']!=31854:
        raise RuntimeError('Previous boundary baseline not verified')
    matches=[]
    with fiona.open(ROOT/f'output/20250521/final/pihe/pond_storage_{DATE}.gpkg',layer='ponds') as src:
        water_crs=CRS.from_user_input(src.crs_wkt or src.crs)
        for f in src:
            if int(f['properties']['pond_id'])==31854:
                matches.append((shape(f['geometry']),dict(f['properties'])))
    if len(matches)!=1: raise RuntimeError('Official object missing or ambiguous')
    geom,props=matches[0]
    h75=float(props['h75'])
    if abs(geom.area-44400)>.01 or h75!=34.0:
        raise RuntimeError('Official geometry area or h75 changed')
    with rasterio.open(ROOT/'data/dem/dem.tif') as dem:
        dem_crs=CRS.from_user_input(dem.crs)
        tf=Transformer.from_crs(water_crs,dem_crs,always_xy=True)
        dg=transform(tf.transform,geom) if water_crs!=dem_crs else geom
        # Boundless window records outside-DEM cells explicitly rather than dropping them.
        win=geometry_window(dem,[mapping(dg)],pad_x=1,pad_y=1,boundless=True)
        trans=dem.window_transform(win)
        raw=dem.read(1,window=win,boundless=True,fill_value=dem.nodata if dem.nodata is not None else 0)
        mask=rasterize([(mapping(dg),1)],out_shape=raw.shape,transform=trans,all_touched=True).astype(bool)
        yy,xx=np.indices(raw.shape)
        in_extent=(yy+int(win.row_off)>=0)&(yy+int(win.row_off)<dem.height)&(xx+int(win.col_off)>=0)&(xx+int(win.col_off)<dem.width)
        finite=np.isfinite(raw)
        is_nodata=np.isclose(raw,dem.nodata) if dem.nodata is not None else np.zeros(raw.shape,bool)
        valid=mask & in_extent & finite & ~is_nodata
        pixel_area=abs(trans.a*trans.e-trans.b*trans.d)
        extent=rasterio.transform.array_bounds(raw.shape[0],raw.shape[1],trans)
        nodata=dem.nodata
    if not valid.any(): raise RuntimeError('No valid interior DEM cells')
    vals=raw[valid].astype(float)
    equal=valid & np.isclose(raw,h75,rtol=0,atol=1e-6)
    below=valid & (raw<h75) & ~equal
    above=valid & (raw>h75) & ~equal
    if int((below|equal|above).sum())!=int(valid.sum()): raise RuntimeError('Classification does not close')
    groups={name:{'cells':int(m.sum()),'fraction_of_valid_cells':float(m.sum()/valid.sum()),'grid_footprint_area_m2':float(m.sum()*pixel_area)} for name,m in [('below_h75',below),('equal_h75',equal),('above_h75',above)]}
    report={'status':'complete','date':DATE,'official_pond_id':31854,'h75_m':h75,'object_polygon_area_m2':geom.area,
            'all_touched':True,'dem_nodata':float(nodata) if nodata is not None else None,'dem_pixel_area_m2':pixel_area,
            'selected_cells':int(mask.sum()),'valid_cells':int(valid.sum()),'invalid_cells':int((mask & ~valid).sum()),
            'outside_dem_cells':int((mask & ~in_extent).sum()),'nodata_cells_inside_dem':int((mask & in_extent & is_nodata).sum()),
            'nonfinite_non_nodata_cells_inside_dem':int((mask & in_extent & ~finite & ~is_nodata).sum()),
            'valid_fraction_of_selected_cells':float(valid.sum()/mask.sum()),'groups':groups,
            'elevation_min_m':float(vals.min()),'elevation_median_m':float(np.median(vals)),'elevation_max_m':float(vals.max()),
            'limits':'Single-object all-touched raster footprint, including touched boundary cells, not strict polygon area. Other object ID-raster competition not reproduced. DEM below h75 indicates positive depth under the model, not measured bathymetry. No volume computed.'}
    out=ROOT/'diagnostics/results/one_interior_dem'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True,exist_ok=False)
    with (out/'elevation_counts.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f);writer.writerow(['elevation_m','valid_cell_count'])
        for value,count in zip(*np.unique(vals,return_counts=True)): writer.writerow([float(value),int(count)])
    (out/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    classes=np.zeros(raw.shape,dtype='uint8');classes[mask & ~valid]=1;classes[below]=2;classes[equal]=3;classes[above]=4
    fig,axes=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
    axes[0].imshow(classes,extent=(extent[0],extent[2],extent[1],extent[3]),cmap=ListedColormap(['white','black','royalblue','gold','tomato']),vmin=0,vmax=4,interpolation='nearest')
    for poly in ([dg] if dg.geom_type=='Polygon' else dg.geoms):
        for ring in [poly.exterior,*poly.interiors]: axes[0].plot(*ring.xy,color='grey',linewidth=.6)
    axes[0].legend(handles=[Patch(color=c,label=l) for c,l in [('royalblue','Below h75'),('gold','Equal h75'),('tomato','Above h75'),('black','Invalid')]],fontsize=8)
    axes[0].set_title('DEM cells touched by object');axes[0].set_aspect('equal')
    axes[1].hist(vals,bins=min(40,max(5,len(np.unique(vals)))),color='steelblue')
    axes[1].axvline(h75,color='red',label='h75 = 34 m');axes[1].legend()
    axes[1].set_xlabel('DEM elevation (m)');axes[1].set_ylabel('Valid cells')
    fig.suptitle('2025-05-21 pond 31854 | interior DEM only; no volume calculation')
    fig.savefig(out/'interior_dem.png',dpi=140);plt.close(fig)
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
    print('RESULT_FOLDER:',out,flush=True)


if __name__=='__main__': main()
