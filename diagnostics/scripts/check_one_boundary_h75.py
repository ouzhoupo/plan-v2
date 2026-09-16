"""User-run: identify one official object and audit boundary DEM sampling only."""
import sys
sys.dont_write_bytecode = True
import csv, json
from pathlib import Path
from datetime import datetime
import numpy as np
import fiona, rasterio, yaml
from rasterio.features import shapes, geometry_window
from scipy import ndimage
from shapely.geometry import shape, mapping
from shapely.ops import transform
from pyproj import CRS, Transformer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'code'))
import pond_storage_monitor as model
DATE = '2025-05-21'


def main():
    source = ROOT/'diagnostics/results/fixed_site_closing/20260914_170521/main_canal_2025-05-21_close1.tif'
    with rasterio.open(source) as src:
        labels, _ = ndimage.label(src.read(1)>0)
        mask = labels == 634
        if not mask.any() or mask[0].any() or mask[-1].any() or mask[:,0].any() or mask[:,-1].any():
            raise RuntimeError('Missing or clipped local component 634')
        parts = [shape(g) for g,v in shapes(mask.astype('uint8'),mask=mask,transform=src.transform) if v==1]
        water_crs = CRS.from_user_input(src.crs)
    if len(parts)!=1 or abs(parts[0].area-44400)>.01:
        raise RuntimeError('Local object differs from previous record')
    local = parts[0]
    official = ROOT/f'output/20250521/final/pihe/pond_storage_{DATE}.gpkg'
    matches = []
    with fiona.open(official,layer='ponds') as src:
        src_crs = CRS.from_user_input(src.crs_wkt or src.crs)
        tf = Transformer.from_crs(src_crs,water_crs,always_xy=True)
        for f in src:
            if not f['geometry']: continue
            g = shape(f['geometry'])
            if src_crs != water_crs: g=transform(tf.transform,g)
            if g.equals(local): matches.append((g,dict(f['properties'])))
    if len(matches)!=1:
        raise RuntimeError(f'Expected exactly one geometrically identical official object; found {len(matches)}. Stop without choosing a near match.')
    geom, props = matches[0]
    cfg = yaml.safe_load((ROOT/f'output/20250521/final/pihe/resolved_monitor_cfg_{DATE}.yaml').read_text(encoding='utf-8'))
    step = float(cfg.get('boundary_sample_step_m',5.0))
    cap = int(cfg.get('max_boundary_samples',5000))
    dem_path = ROOT/'data/dem/dem.tif'
    obj = model.WaterObject(obj_id=int(props['pond_id']),geom=geom,area_m2=geom.area)
    model.estimate_water_levels([obj],str(dem_path),water_crs=water_crs,sample_step_m=step,max_boundary_samples=cap)
    with rasterio.open(dem_path) as dem:
        dem_crs = CRS.from_user_input(dem.crs)
        tf = Transformer.from_crs(water_crs,dem_crs,always_xy=True)
        dg = transform(tf.transform,geom) if water_crs!=dem_crs else geom
        pts = model.boundary_sample_points(dg,step,cap)
        elevations = np.array([float(s[0]) for s in dem.sample(pts)])
        valid = np.isfinite(elevations)
        if dem.nodata is not None: valid &= ~np.isclose(elevations,dem.nodata)
        if not valid.any(): raise RuntimeError('No valid boundary DEM samples')
        q = np.quantile(elevations[valid],[.05,.5,.75,.95])
        if int(obj.props['bnd_n'])!=int(valid.sum()) or abs(float(obj.props['h75'])-round(float(q[2]),3))>1e-6:
            raise RuntimeError('Independent sampling differs from current production function')
        pixel_rows_cols = [dem.index(x,y) for x,y in pts]
        win = geometry_window(dem,[mapping(dg.buffer(50))])
        background = dem.read(1,window=win,masked=True)
        bounds = dem.window_bounds(win)
    fields = ['bnd_n','h05','h50','h75','h95','hmean']
    checks = {k:{'saved':props.get(k),'recomputed':obj.props.get(k),
                 'matches':props.get(k) is not None and obj.props.get(k) is not None and abs(float(props[k])-float(obj.props[k]))<=.0011} for k in fields}
    out = ROOT/'diagnostics/results/one_boundary_h75'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True,exist_ok=False)
    rows = [{'sample_index':i,'x_dem':p[0],'y_dem':p[1],'dem_row':rc[0],'dem_col':rc[1],
             'valid':bool(v),'elevation_m':float(z) if v else None} for i,(p,rc,v,z) in enumerate(zip(pts,pixel_rows_cols,valid,elevations))]
    with (out/'boundary_samples.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    report={'status':'complete','date':DATE,'local_component_id':634,'official_pond_id':int(props['pond_id']),
            'official_geometry_exact_match':True,'saved_comparison':checks,'saved_levels_all_match':all(c['matches'] for c in checks.values()),
            'sample_step_m':step,'sample_cap_parameter':cap,'total_samples':len(pts),'valid_samples':int(valid.sum()),
            'unique_valid_dem_cells':len({rc for rc,v in zip(pixel_rows_cols,valid) if v}),
            'boundary_min_m':float(elevations[valid].min()),'boundary_max_m':float(elevations[valid].max()),
            'boundary_h95_minus_h05_m':float(q[3]-q[0]),'dem_crs':dem_crs.to_string(),
            'limits':'Boundary sampling only, includes hole boundaries and repeated cell samples. Matching saved levels proves reproduction, not actual water level or bathymetric accuracy. No volumes calculated.'}
    (out/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    fig,axes=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
    axes[0].imshow(background,extent=(bounds[0],bounds[2],bounds[1],bounds[3]),cmap='terrain')
    xy=np.asarray(pts)
    scat=axes[0].scatter(xy[valid,0],xy[valid,1],c=elevations[valid],cmap='viridis',s=12)
    if (~valid).any(): axes[0].scatter(xy[~valid,0],xy[~valid,1],marker='x',c='red',s=20)
    fig.colorbar(scat,ax=axes[0],label='Boundary DEM elevation (m)')
    axes[0].set_title('DEM and boundary samples (including holes)');axes[0].set_aspect('equal')
    axes[1].plot(np.linspace(0,100,int(valid.sum())),np.sort(elevations[valid]))
    axes[1].axhline(float(obj.props['h75']),color='red',label=f'h75 = {obj.props["h75"]} m')
    axes[1].set_xlabel('Sorted valid samples (%)');axes[1].set_ylabel('DEM elevation (m)');axes[1].legend()
    fig.suptitle(f'{DATE} official pond {props["pond_id"]} | boundary-level audit only')
    fig.savefig(out/'boundary_h75.png',dpi=140);plt.close(fig)
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
    print('RESULT_FOLDER:',out,flush=True)


if __name__=='__main__':
    main()
