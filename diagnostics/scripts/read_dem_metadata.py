"""User-run: metadata only; no raster pixel reads or statistics computation."""
import sys
sys.dont_write_bytecode = True
import json
from pathlib import Path
from datetime import datetime
import rasterio

ROOT = Path(__file__).resolve().parents[2]


def main():
    path = ROOT / 'data/dem/dem.tif'
    with rasterio.open(path) as src:
        dataset_namespaces = src.tag_namespaces()
        report = {
            'status': 'complete', 'file': str(path),
            'driver': src.driver, 'width': src.width, 'height': src.height,
            'bands': src.count, 'dtypes': list(src.dtypes), 'nodata': src.nodata,
            'crs_wkt': src.crs.to_wkt() if src.crs else None,
            'transform': list(src.transform), 'resolution': list(src.res),
            'descriptions': list(src.descriptions), 'units': list(src.units),
            'scales': list(src.scales), 'offsets': list(src.offsets),
            'dataset_tags': src.tags(),
            'dataset_namespace_tags': {ns: src.tags(ns=ns) for ns in dataset_namespaces},
            'band_metadata': [],
            'limits': 'Metadata only. File dates and processing software are not survey dates or source provenance. Grid spacing is not accuracy. Missing source tags do not establish a source.'
        }
        for band in range(1, src.count + 1):
            report['band_metadata'].append({'band': band, 'tags': src.tags(band),
                'namespace_tags': {ns: src.tags(band, ns=ns) for ns in src.tag_namespaces(band)}})
    out = ROOT / 'diagnostics/results/dem_metadata' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True, exist_ok=False)
    (out / 'metadata.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Metadata read complete; no raster pixel arrays read.', flush=True)
    print('RESULT_FOLDER:', out, flush=True)


if __name__ == '__main__':
    main()
