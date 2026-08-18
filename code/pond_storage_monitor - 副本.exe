#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
塘坝水面扣除与蓄水量监测算法（生产版，仅接受真实 DEM）

功能：
1. 从二值/分类水体栅格提取水体面积，并进行形态学净化。
2. 扣除渠道/河流线、湖泊、大中小型水库等已知水面。
3. 在扣除后的水面上，结合真实 DEM 估算塘坝蓄水量。

依赖：
    rasterio, fiona, shapely, pyproj, numpy, scipy, pyyaml(可选)
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import logging
import math
import os
import re
import sys
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def configure_proj_runtime() -> None:
    rasterio_spec = importlib.util.find_spec("rasterio")
    if rasterio_spec is None or not rasterio_spec.submodule_search_locations:
        os.environ.setdefault("GTIFF_SRS_SOURCE", "EPSG")
        return
    rasterio_dir = Path(next(iter(rasterio_spec.submodule_search_locations)))
    candidate_dirs = [
        rasterio_dir / "proj_data",
        Path(__file__).resolve().parent / "proj_data",
    ]
    proj_dir = next((p for p in candidate_dirs if (p / "proj.db").exists()), None)
    if proj_dir is not None:
        os.environ["PROJ_DATA"] = str(proj_dir)
        os.environ["PROJ_LIB"] = str(proj_dir)
    gdal_dir = rasterio_dir / "gdal_data"
    if gdal_dir.exists():
        os.environ["GDAL_DATA"] = str(gdal_dir)
    os.environ.setdefault("GTIFF_SRS_SOURCE", "EPSG")
    os.environ.setdefault("CPL_LOG_ERRORS", "OFF")


configure_proj_runtime()
warnings.filterwarnings(
    "ignore",
    message=".*'Memory' driver is deprecated since GDAL 3.11.*",
    category=DeprecationWarning,
)

import fiona
import numpy as np
import rasterio
from affine import Affine
from fiona.crs import CRS as FionaCRS
from pyproj import CRS, Transformer
from rasterio.enums import Resampling
from rasterio.features import rasterize, shapes
from rasterio.transform import array_bounds
from rasterio.windows import Window, bounds as window_bounds, from_bounds as window_from_bounds
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds
from scipy import ndimage
try:
    from shapely import STRtree
except Exception:  # shapely 1.x
    from shapely.strtree import STRtree
from shapely.geometry import MultiPolygon, Polygon, box, mapping, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shapely_transform, unary_union

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


LOGGER = logging.getLogger("pond_storage_monitor")


# ----------------------------
# 数据结构
# ----------------------------

@dataclass
class DeductLayer:
    name: str
    source: str
    geoms: List[BaseGeometry]
    tree: Optional[STRtree] = None
    raw_overlap_m2: float = 0.0

    def build_index(self) -> None:
        if self.geoms:
            self.tree = STRtree(self.geoms)

    def query(self, query_geom: BaseGeometry) -> List[BaseGeometry]:
        if not self.geoms:
            return []
        if self.tree is None:
            return [g for g in self.geoms if g.intersects(query_geom)]
        idxs = self.tree.query(query_geom)
        out: List[BaseGeometry] = []
        for item in idxs:
            if isinstance(item, (int, np.integer)):
                g = self.geoms[int(item)]
            else:
                # 兼容 shapely 1.x：query 可能直接返回几何对象
                g = item
            if g is not None and not g.is_empty and g.intersects(query_geom):
                out.append(g)
        return out


@dataclass
class WaterObject:
    obj_id: int
    geom: BaseGeometry
    area_m2: float
    props: Dict[str, Any] = field(default_factory=dict)


# ----------------------------
# 通用工具函数
# ----------------------------

def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def parse_crs(crs_like: Any) -> Optional[CRS]:
    if not crs_like:
        return None
    try:
        return CRS.from_user_input(crs_like)
    except Exception:
        try:
            return CRS.from_wkt(str(crs_like))
        except Exception:
            return None


def flatten_2d(geom: BaseGeometry) -> BaseGeometry:
    """去除 Z 值，避免 3D LineString/Polygon 影响缓冲和栅格化。"""
    if geom is None or geom.is_empty:
        return geom
    return shapely_transform(lambda x, y, z=None: (x, y), geom)


def fix_geometry(geom: BaseGeometry) -> BaseGeometry:
    if geom is None or geom.is_empty:
        return geom
    geom = flatten_2d(geom)
    try:
        if not geom.is_valid:
            # shapely 2.x make_valid 如果可用则更稳；buffer(0) 作为兜底。
            try:
                from shapely.validation import make_valid
                geom = make_valid(geom)
            except Exception:
                geom = geom.buffer(0)
    except Exception:
        geom = geom.buffer(0)
    return geom


def reproject_geom(geom: BaseGeometry, src_crs: Optional[CRS], dst_crs: Optional[CRS]) -> BaseGeometry:
    geom = fix_geometry(geom)
    if geom is None or geom.is_empty:
        return geom
    if src_crs is None or dst_crs is None or src_crs == dst_crs:
        return geom
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    return shapely_transform(transformer.transform, geom)


def list_shapefiles(directory: Optional[str]) -> List[str]:
    if not directory:
        return []
    p = Path(directory)
    if not p.exists():
        return []
    return sorted(str(x) for x in p.rglob("*.shp"))


def read_geometries(
    vector_path: str,
    target_crs: Optional[CRS],
    *,
    buffer_m: float = 0.0,
    bbox_filter: Optional[Tuple[float, float, float, float]] = None,
    name: Optional[str] = None,
) -> List[BaseGeometry]:
    """读取矢量几何并转换到目标 CRS。线状对象会在投影坐标系下缓冲。"""
    geoms: List[BaseGeometry] = []
    query_box = box(*bbox_filter) if bbox_filter else None

    with fiona.open(vector_path) as src:
        src_crs = parse_crs(src.crs_wkt or src.crs)
        for feat in src:
            if not feat.get("geometry"):
                continue
            try:
                geom = shape(feat["geometry"])
                geom = reproject_geom(geom, src_crs, target_crs)
                if geom is None or geom.is_empty:
                    continue

                # 仅在转换到目标 CRS 后进行缓冲，确保 buffer_m 是米。
                if geom.geom_type in ("LineString", "MultiLineString", "LinearRing"):
                    if buffer_m > 0:
                        geom = geom.buffer(buffer_m, cap_style=2, join_style=2)
                    else:
                        # 0 缓冲的线无法栅格化出稳定面积，至少给 1 像素级宽度更合理。
                        geom = geom.buffer(0.5, cap_style=2, join_style=2)
                elif buffer_m > 0 and geom.geom_type in ("Point", "MultiPoint"):
                    geom = geom.buffer(buffer_m)

                geom = fix_geometry(geom)
                if geom is None or geom.is_empty:
                    continue
                if query_box is not None and not geom.intersects(query_box):
                    continue
                if query_box is not None:
                    # 限定到处理范围附近，减少后续栅格化成本；保留稍微外扩后的相交部分。
                    geom = geom.intersection(query_box)
                    if geom is None or geom.is_empty:
                        continue
                geoms.append(geom)
            except Exception as exc:
                LOGGER.warning("跳过无效要素：%s -> %s", vector_path, exc)

    LOGGER.info("读取扣除图层 %-20s 要素数：%d", name or Path(vector_path).stem, len(geoms))
    return geoms


def read_union_geometry(vector_path: Optional[str], target_crs: Optional[CRS]) -> Optional[BaseGeometry]:
    if not vector_path:
        return None
    p = Path(vector_path)
    if not p.exists():
        LOGGER.warning("AOI 文件不存在：%s", vector_path)
        return None
    with fiona.open(str(p)) as src:
        src_crs = parse_crs(src.crs_wkt or src.crs)
        geoms = []
        for feat in src:
            if feat.get("geometry"):
                g = reproject_geom(shape(feat["geometry"]), src_crs, target_crs)
                if g is not None and not g.is_empty:
                    geoms.append(g)
    if not geoms:
        return None
    return fix_geometry(unary_union(geoms))


def intersect_bounds(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    left = max(a[0], b[0])
    bottom = max(a[1], b[1])
    right = min(a[2], b[2])
    top = min(a[3], b[3])
    if left >= right or bottom >= top:
        raise ValueError(f"处理范围无交集：{a} 与 {b}")
    return (left, bottom, right, top)


def snap_bounds_to_res(bounds: Tuple[float, float, float, float], res: float) -> Tuple[float, float, float, float]:
    left, bottom, right, top = bounds
    return (
        math.floor(left / res) * res,
        math.floor(bottom / res) * res,
        math.ceil(right / res) * res,
        math.ceil(top / res) * res,
    )


def window_from_intersection(src: rasterio.io.DatasetReader, bounds_: Tuple[float, float, float, float]) -> Window:
    bounds_ = intersect_bounds(tuple(src.bounds), bounds_)
    win = window_from_bounds(*bounds_, transform=src.transform)
    # 包含完整像元，避免边界漏取。
    row_off = max(0, int(math.floor(win.row_off)))
    col_off = max(0, int(math.floor(win.col_off)))
    row_end = min(src.height, int(math.ceil(win.row_off + win.height)))
    col_end = min(src.width, int(math.ceil(win.col_off + win.width)))
    return Window(col_off, row_off, col_end - col_off, row_end - row_off)


def iter_windows(width: int, height: int, block_size: int) -> Iterable[Window]:
    for row in range(0, height, block_size):
        h = min(block_size, height - row)
        for col in range(0, width, block_size):
            w = min(block_size, width - col)
            yield Window(col, row, w, h)


def transform_for_window(base_transform: Affine, win: Window) -> Affine:
    return rasterio.windows.transform(win, base_transform)


def remove_existing_vector(path: str | Path) -> None:
    p = Path(path)
    if p.suffix.lower() == ".gpkg" and p.exists():
        p.unlink()
    elif p.suffix.lower() == ".shp":
        for ext in [".shp", ".shx", ".dbf", ".prj", ".cpg", ".qpj"]:
            q = p.with_suffix(ext)
            if q.exists():
                q.unlink()


# ----------------------------
# 水体净化与扣除
# ----------------------------

def build_deduct_layers(
    *,
    river_dir: Optional[str],
    reservoir_shp: Optional[str],
    extra_vectors: Sequence[str],
    target_crs: CRS,
    bbox_filter: Tuple[float, float, float, float],
    line_buffer_m: float,
) -> List[DeductLayer]:
    paths: List[Tuple[str, str, float]] = []

    for shp in list_shapefiles(river_dir):
        stem = Path(shp).stem
        paths.append((stem, shp, line_buffer_m))

    if reservoir_shp:
        paths.append(("known_reservoir", reservoir_shp, 0.0))

    for v in extra_vectors:
        paths.append((Path(v).stem, v, line_buffer_m))

    layers: List[DeductLayer] = []
    for name, p, buf in paths:
        if not Path(p).exists():
            LOGGER.warning("扣除矢量不存在：%s", p)
            continue
        geoms = read_geometries(p, target_crs, buffer_m=buf, bbox_filter=bbox_filter, name=name)
        if geoms:
            layer = DeductLayer(name=name, source=p, geoms=geoms)
            layer.build_index()
            layers.append(layer)
    LOGGER.info("有效扣除图层数量：%d", len(layers))
    return layers


def rasterize_geoms_for_window(
    geoms: Sequence[BaseGeometry],
    out_shape: Tuple[int, int],
    transform: Affine,
    *,
    all_touched: bool = True,
    dtype: str = "uint8",
    fill: int = 0,
    value: int = 1,
) -> np.ndarray:
    if not geoms:
        return np.zeros(out_shape, dtype=dtype)
    return rasterize(
        [(mapping(g), value) for g in geoms if g is not None and not g.is_empty],
        out_shape=out_shape,
        transform=transform,
        fill=fill,
        dtype=dtype,
        all_touched=all_touched,
    )


def refine_binary_water(
    water: np.ndarray,
    *,
    close_radius_px: int = 1,
    open_radius_px: int = 0,
    fill_holes: bool = False,
) -> np.ndarray:
    """水体二值图像形态学净化。小斑块过滤放在矢量化后按面积处理。"""
    arr = water.astype(bool, copy=False)
    if open_radius_px and open_radius_px > 0:
        structure = np.ones((2 * open_radius_px + 1, 2 * open_radius_px + 1), dtype=bool)
        arr = ndimage.binary_opening(arr, structure=structure)
    if close_radius_px and close_radius_px > 0:
        structure = np.ones((2 * close_radius_px + 1, 2 * close_radius_px + 1), dtype=bool)
        arr = ndimage.binary_closing(arr, structure=structure)
    if fill_holes:
        arr = ndimage.binary_fill_holes(arr)
    return arr.astype("uint8")


def prepare_water_mask(
    water_tif: str,
    out_mask_tif: str,
    *,
    process_bounds: Tuple[float, float, float, float],
    aoi_geom: Optional[BaseGeometry],
    deduct_layers: Sequence[DeductLayer],
    water_value: Optional[int],
    respect_source_mask: bool,
    block_size: int,
    close_radius_px: int,
    open_radius_px: int,
    fill_holes: bool,
    all_touched: bool,
    cropland_mask_tif: Optional[str] = None,
    landuse_water_hard_mask_tif: Optional[str] = None,
    landuse_water_value: int = 1,
) -> Tuple[str, Dict[str, Any]]:
    """生成扣除后的水体掩膜 GeoTIFF，值 1 表示保留塘坝候选水面。"""
    ensure_dir(Path(out_mask_tif).parent)

    with rasterio.open(water_tif) as src:
        src_window = window_from_intersection(src, process_bounds)
        out_transform = src.window_transform(src_window)
        out_width = int(src_window.width)
        out_height = int(src_window.height)
        pixel_area = abs(src.transform.a * src.transform.e)
        profile = {
            "driver": "GTiff",
            "height": out_height,
            "width": out_width,
            "count": 1,
            "dtype": "uint8",
            "crs": src.crs,
            "transform": out_transform,
            "nodata": 0,
            "compress": "deflate",
            "tiled": True,
            "blockxsize": min(block_size, 512),
            "blockysize": min(block_size, 512),
            "BIGTIFF": "IF_SAFER",
        }

        if Path(out_mask_tif).exists():
            Path(out_mask_tif).unlink()

        LOGGER.info("创建扣除后水体掩膜：%s，尺寸：%d x %d", out_mask_tif, out_width, out_height)

        summary = {
            "source_water_pixels": 0,
            "source_water_area_m2": 0.0,
            "aoi_water_pixels": 0,
            "aoi_water_area_m2": 0.0,
            "landuse_intersected_pixels": 0,
            "landuse_intersected_area_m2": 0.0,
            "deducted_union_pixels": 0,
            "deducted_union_area_m2": 0.0,
            "pond_candidate_pixels": 0,
            "pond_candidate_area_m2": 0.0,
            "pixel_area_m2": pixel_area,
            "layers": {},
        }
        for layer in deduct_layers:
            summary["layers"][layer.name] = {
                "source": layer.source,
                "raw_overlap_m2": 0.0,
                "raw_overlap_pixels": 0,
            }

        cropland_vrt = None
        cropland_src = None
        if cropland_mask_tif and Path(cropland_mask_tif).exists():
            cropland_src = rasterio.open(cropland_mask_tif)
            cropland_vrt = WarpedVRT(
                cropland_src,
                crs=src.crs,
                transform=out_transform,
                width=out_width,
                height=out_height,
                resampling=Resampling.nearest,
            )

        landuse_vrt = None
        landuse_src = None
        if landuse_water_hard_mask_tif and Path(landuse_water_hard_mask_tif).exists():
            landuse_src = rasterio.open(landuse_water_hard_mask_tif)
            landuse_vrt = WarpedVRT(
                landuse_src,
                crs=src.crs,
                transform=out_transform,
                width=out_width,
                height=out_height,
                resampling=Resampling.nearest,
            )
            LOGGER.info("启用 landuse_water 硬掩膜（取交集）：%s value=%d", landuse_water_hard_mask_tif, landuse_water_value)

        try:
            with rasterio.open(out_mask_tif, "w", **profile) as dst:
                for out_win in iter_windows(out_width, out_height, block_size):
                    src_win = Window(
                        src_window.col_off + out_win.col_off,
                        src_window.row_off + out_win.row_off,
                        out_win.width,
                        out_win.height,
                    )
                    data = src.read(1, window=src_win, boundless=False)
                    if water_value is None:
                        water = data > 0
                    else:
                        water = data == water_value

                    summary["source_water_pixels"] += int(water.sum())

                    if respect_source_mask:
                        src_mask = src.read_masks(1, window=src_win) > 0
                        water &= src_mask

                    win_transform = transform_for_window(out_transform, out_win)
                    out_shape = (int(out_win.height), int(out_win.width))
                    win_bounds = window_bounds(out_win, out_transform)
                    win_box = box(*win_bounds)

                    if aoi_geom is not None:
                        if not aoi_geom.intersects(win_box):
                            water &= False
                        else:
                            aoi_mask = rasterize_geoms_for_window(
                                [aoi_geom.intersection(win_box)],
                                out_shape,
                                win_transform,
                                all_touched=all_touched,
                            ).astype(bool)
                            water &= aoi_mask

                    summary["aoi_water_pixels"] += int(water.sum())

                    if landuse_vrt is not None and water.any():
                        lu = landuse_vrt.read(1, window=out_win) == landuse_water_value
                        water &= lu

                    summary["landuse_intersected_pixels"] += int(water.sum())

                    if cropland_vrt is not None and water.any():
                        crop = cropland_vrt.read(1, window=out_win) > 0
                        water &= ~crop

                    deduct_union = np.zeros(out_shape, dtype=bool)
                    if water.any() and deduct_layers:
                        for layer in deduct_layers:
                            q_geoms = layer.query(win_box)
                            if not q_geoms:
                                continue
                            layer_mask = rasterize_geoms_for_window(
                                q_geoms,
                                out_shape,
                                win_transform,
                                all_touched=all_touched,
                            ).astype(bool)
                            overlap_px = int((water & layer_mask).sum())
                            if overlap_px:
                                summary["layers"][layer.name]["raw_overlap_pixels"] += overlap_px
                                summary["layers"][layer.name]["raw_overlap_m2"] += overlap_px * pixel_area
                            deduct_union |= layer_mask

                    deducted_px = int((water & deduct_union).sum())
                    summary["deducted_union_pixels"] += deducted_px

                    pond = water & (~deduct_union)
                    pond = refine_binary_water(
                        pond,
                        close_radius_px=close_radius_px,
                        open_radius_px=open_radius_px,
                        fill_holes=fill_holes,
                    ).astype("uint8")

                    summary["pond_candidate_pixels"] += int(pond.sum())
                    dst.write(pond, 1, window=out_win)
        finally:
            if cropland_vrt is not None:
                cropland_vrt.close()
            if cropland_src is not None:
                cropland_src.close()
            if landuse_vrt is not None:
                landuse_vrt.close()
            if landuse_src is not None:
                landuse_src.close()

        summary["source_water_area_m2"] = summary["source_water_pixels"] * pixel_area
        summary["aoi_water_area_m2"] = summary["aoi_water_pixels"] * pixel_area
        summary["landuse_intersected_area_m2"] = summary["landuse_intersected_pixels"] * pixel_area
        summary["deducted_union_area_m2"] = summary["deducted_union_pixels"] * pixel_area
        summary["pond_candidate_area_m2"] = summary["pond_candidate_pixels"] * pixel_area
        return out_mask_tif, summary


def shape_metrics(geom: BaseGeometry) -> Dict[str, float]:
    area = float(getattr(geom, "area", 0.0) or 0.0)
    peri = float(getattr(geom, "length", 0.0) or 0.0)
    compactness = float(4.0 * math.pi * area / max(peri * peri, 1e-6)) if area > 0 else 0.0
    elongation = 1.0
    width_m = 0.0
    length_m = 0.0
    try:
        mrr = geom.minimum_rotated_rectangle
        coords = list(mrr.exterior.coords)
        edges = []
        for i in range(min(4, len(coords) - 1)):
            x1, y1 = coords[i]
            x2, y2 = coords[i + 1]
            edges.append(math.hypot(x2 - x1, y2 - y1))
        edges = sorted([e for e in edges if e > 0])
        if len(edges) >= 2:
            width_m = float(edges[0])
            length_m = float(edges[-1])
            elongation = float(length_m / max(width_m, 1e-6))
    except Exception:
        pass
    return {
        "compactness": round(compactness, 6),
        "elongation": round(elongation, 6),
        "width_m": round(width_m, 3),
        "length_m": round(length_m, 3),
    }


def filter_linear_artifacts(
    objects: List[WaterObject],
    *,
    apply_min_area_m2: float = 3000.0,
    max_elongation: float = 12.0,
    min_compactness: float = 0.06,
    min_width_m: float = 8.0,
) -> List[WaterObject]:
    """剔除剩余的细长线状水体残片，避免渠道残留进入塘坝库容。"""
    kept: List[WaterObject] = []
    removed = 0
    for obj in objects:
        area = float(obj.props.get("area_m2", obj.area_m2) or obj.area_m2)
        metrics = shape_metrics(obj.geom)
        obj.props.update(metrics)
        is_linear = (
            area >= apply_min_area_m2
            and metrics["elongation"] >= max_elongation
            and metrics["compactness"] <= min_compactness
            and metrics["width_m"] <= min_width_m
        )
        if is_linear:
            removed += 1
            continue
        kept.append(obj)
    LOGGER.info("线状残片过滤：%d → %d（移除 %d）", len(objects), len(kept), removed)
    return kept


def watershed_split_labels(
    binary: np.ndarray,
    *,
    res_m: float,
    split_min_area_m2: float,
    peak_min_distance_m: float,
    peak_min_distance_ratio: float = 0.45,
) -> np.ndarray:
    """对粘连水体做距离变换+分水岭分割，把每个独立塘坝标记为不同 label。

    binary: 0/1 二值数组
    返回: int32 标签数组（0=背景，>=1=各塘坝 ID）
    仅对面积 >= split_min_area_m2 的连通块做分水岭，小块直接保留为单个 label。
    """
    from scipy import ndimage as _nd
    binary = binary.astype(bool, copy=False)
    if not binary.any():
        return np.zeros(binary.shape, dtype="int32")

    res2 = res_m * res_m
    cross = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    cc, n_cc = _nd.label(binary, structure=cross)
    if n_cc == 0:
        return np.zeros(binary.shape, dtype="int32")

    sizes = _nd.sum_labels(np.ones_like(cc, dtype=np.int32), cc, index=np.arange(1, n_cc + 1))
    out = np.zeros(binary.shape, dtype="int32")
    next_id = 1
    needs_split = sizes * res2 >= split_min_area_m2
    bboxes = _nd.find_objects(cc)

    big_mask = np.zeros_like(binary)
    for idx, slc in enumerate(bboxes):
        if slc is None or not needs_split[idx]:
            continue
        big_mask[slc] |= (cc[slc] == idx + 1)

    small_mask = binary & ~big_mask
    if small_mask.any():
        small_lab, n_small = _nd.label(small_mask, structure=cross)
        out[small_lab > 0] = small_lab[small_lab > 0] + (next_id - 1)
        next_id += n_small

    if not big_mask.any():
        return out

    dist = _nd.distance_transform_edt(big_mask).astype("float32")
    peak_min_px = max(1, int(round(peak_min_distance_m / max(res_m, 1e-6))))
    fp_size = 2 * peak_min_px + 1
    local_max = _nd.maximum_filter(dist, size=fp_size, mode="constant", cval=0.0)
    seed_core_thr = max(peak_min_px * peak_min_distance_ratio, 1.25)
    seeds = (dist == local_max) & (dist >= seed_core_thr) & big_mask
    seed_lab, n_seeds = _nd.label(seeds, structure=cross)
    if n_seeds == 0:
        big_lab, n_big = _nd.label(big_mask, structure=cross)
        out[big_lab > 0] = big_lab[big_lab > 0] + (next_id - 1)
        return out

    try:
        from skimage.segmentation import watershed as _watershed  # type: ignore
        ws = _watershed(-dist, seed_lab, mask=big_mask)
    except Exception:
        # 无 skimage 时用最近种子近似分水岭
        idx_to_seed = _nd.distance_transform_edt(seed_lab == 0, return_distances=False, return_indices=True)
        nearest = seed_lab[tuple(idx_to_seed)]
        ws = np.where(big_mask, nearest, 0).astype("int32")

    ws = ws.astype("int32")
    if ws.max() > 0:
        ws[ws > 0] += next_id - 1
        next_id += int(ws.max() - (next_id - 1))
        out = np.where(ws > 0, ws, out)

    return out


def polygonize_water_mask(
    mask_tif: str,
    *,
    min_area_m2: float,
    max_area_m2: Optional[float] = None,
    simplify_tolerance: float = 0.0,
    watershed_split_enable: bool = False,
    watershed_split_min_area_m2: float = 30000.0,
    watershed_peak_min_distance_m: float = 15.0,
    labels_out_tif: Optional[str] = None,
) -> List[WaterObject]:
    """将扣除后的水体掩膜矢量化，可选分水岭分割粘连塘坝。"""
    objs: List[WaterObject] = []
    n_too_small = 0
    n_too_large = 0
    n_split_in = 0
    n_split_out = 0
    with rasterio.open(mask_tif) as src:
        if watershed_split_enable:
            arr = src.read(1) > 0
            res_m = abs(src.transform.a)
            n_split_in = int(arr.sum())
            labels = watershed_split_labels(
                arr,
                res_m=res_m,
                split_min_area_m2=watershed_split_min_area_m2,
                peak_min_distance_m=watershed_peak_min_distance_m,
            )
            n_split_out = int(labels.max())
            if labels_out_tif:
                profile = src.profile.copy()
                profile.update(dtype="int32", nodata=0, count=1, compress="deflate", BIGTIFF="IF_SAFER")
                ensure_dir(Path(labels_out_tif).parent)
                with rasterio.open(labels_out_tif, "w", **profile) as dst_lab:
                    dst_lab.write(labels.astype("int32"), 1)
            shape_iter = shapes(labels.astype("int32"), transform=src.transform, connectivity=4)
        else:
            shape_iter = shapes(rasterio.band(src, 1), transform=src.transform)

        for geom_mapping, value in shape_iter:
            if int(value) <= 0:
                continue
            geom = fix_geometry(shape(geom_mapping))
            if geom is None or geom.is_empty:
                continue

            area = float(geom.area)
            if area < min_area_m2:
                n_too_small += 1
                continue
            if max_area_m2 is not None and area > max_area_m2:
                n_too_large += 1
                continue

            if simplify_tolerance and simplify_tolerance > 0:
                geom = geom.simplify(simplify_tolerance, preserve_topology=True)
                geom = fix_geometry(geom)
                if geom is None or geom.is_empty:
                    continue
                area = float(geom.area)

            obj = WaterObject(obj_id=len(objs) + 1, geom=geom, area_m2=area)
            obj.props["area_m2"] = area
            obj.props["area_mu"] = area / 666.6666667
            obj.props.update(shape_metrics(geom))
            objs.append(obj)

    LOGGER.info(
        "扣除后候选塘坝数量：%d（min=%.1f m2，max=%s m2，过小剔除=%d，过大剔除=%d，watershed=%s 种子数=%d）",
        len(objs), min_area_m2,
        f"{max_area_m2:.1f}" if max_area_m2 is not None else "None",
        n_too_small, n_too_large,
        "ON" if watershed_split_enable else "OFF",
        n_split_out,
    )
    return objs


def filter_by_landuse_water(
    objects: List[WaterObject],
    landuse_water_tif: str,
    water_crs: CRS,
) -> List[WaterObject]:
    """保留与土地利用水体（value=1）有任意交集的塘坝多边形。"""
    if not landuse_water_tif or not Path(landuse_water_tif).exists():
        return objects
    from rasterio.features import geometry_window as _gwin
    kept = []
    with rasterio.open(landuse_water_tif) as lu:
        lu_crs = parse_crs(lu.crs)
        transformer = Transformer.from_crs(water_crs, lu_crs, always_xy=True) if lu_crs and lu_crs != water_crs else None
        for obj in objects:
            geom = shapely_transform(transformer.transform, obj.geom) if transformer else obj.geom
            geom = fix_geometry(geom)
            if geom is None or geom.is_empty:
                continue
            try:
                win = _gwin(lu, [mapping(geom)], pad_x=1, pad_y=1)
                data = lu.read(1, window=win)
                if (data == 1).any():
                    kept.append(obj)
            except Exception:
                pass
    LOGGER.info("土地利用水体交集过滤：%d → %d 个塘坝", len(objects), len(kept))
    return kept


# ----------------------------
# DEM 采样、水位估计与体积计算
# ----------------------------

def boundary_sample_points(geom: BaseGeometry, sample_step_m: float, max_samples: int) -> List[Tuple[float, float]]:
    boundary = geom.boundary
    lines: List[BaseGeometry]
    if boundary.geom_type == "LineString":
        lines = [boundary]
    elif boundary.geom_type == "MultiLineString":
        lines = list(boundary.geoms)
    else:
        lines = [g for g in getattr(boundary, "geoms", []) if g.geom_type == "LineString"]

    points: List[Tuple[float, float]] = []
    total_len = sum(float(line.length) for line in lines)
    if total_len <= 0:
        return points
    step = max(sample_step_m, total_len / max(max_samples, 1))

    for line in lines:
        length = float(line.length)
        if length <= 0:
            continue
        n = max(2, int(math.ceil(length / step)) + 1)
        for d in np.linspace(0, length, n):
            p = line.interpolate(float(d))
            points.append((float(p.x), float(p.y)))
    return points


def sample_dem_values(dem: rasterio.io.DatasetReader, coords: Sequence[Tuple[float, float]]) -> np.ndarray:
    vals: List[float] = []
    nodata = dem.nodata
    for sample in dem.sample(coords):
        if sample is None or len(sample) == 0:
            continue
        v = float(sample[0])
        if nodata is not None and np.isclose(v, nodata):
            continue
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return np.array([], dtype="float64")
    return np.asarray(vals, dtype="float64")


def estimate_water_levels(
    objects: List[WaterObject],
    dem_tif: str,
    *,
    water_crs: CRS,
    sample_step_m: float,
    max_boundary_samples: int,
) -> None:
    with rasterio.open(dem_tif) as dem:
        dem_crs = parse_crs(dem.crs)
        water_to_dem = None
        if dem_crs is not None and water_crs is not None and dem_crs != water_crs:
            transformer = Transformer.from_crs(water_crs, dem_crs, always_xy=True)
            water_to_dem = transformer.transform

        # Batch all boundary points → single dem.sample() call instead of one per object
        obj_ranges: List[Tuple[Any, int, int]] = []
        all_pts: List[Tuple[float, float]] = []
        for obj in objects:
            if obj.props.get("skip_volume"):
                continue
            geom_dem = shapely_transform(water_to_dem, obj.geom) if water_to_dem else obj.geom
            pts = boundary_sample_points(geom_dem, sample_step_m, max_boundary_samples)
            start = len(all_pts)
            all_pts.extend(pts)
            obj_ranges.append((obj, start, len(all_pts)))

        nodata = dem.nodata
        raw_samples = list(dem.sample(all_pts)) if all_pts else []

        _no_dem = {"bnd_n": 0, "h05": None, "h50": None, "h75": None, "h95": None, "hmean": None, "qa": "NO_DEM_ON_BOUNDARY"}
        for obj, start, end in obj_ranges:
            if start == end:
                obj.props.update(_no_dem)
                continue
            vals_list = []
            for s in raw_samples[start:end]:
                v = float(s[0])
                if nodata is not None and np.isclose(v, nodata):
                    continue
                if np.isfinite(v):
                    vals_list.append(v)
            if not vals_list:
                obj.props.update(_no_dem)
                continue
            vals = np.asarray(vals_list, dtype="float64")
            h05 = float(np.quantile(vals, 0.05))
            h50 = float(np.quantile(vals, 0.50))
            h75 = float(np.quantile(vals, 0.75))
            h95 = float(np.quantile(vals, 0.95))
            hmean = float(np.mean(vals[(vals >= h05) & (vals <= h95)])) if vals.size >= 5 else float(vals.mean())
            obj.props.update({
                "bnd_n": int(vals.size),
                "h05": round(h05, 3),
                "h50": round(h50, 3),
                "h75": round(h75, 3),
                "h95": round(h95, 3),
                "hmean": round(hmean, 3),
                "qa": "OK",
            })


def compute_volumes(
    objects: List[WaterObject],
    dem_tif: str,
    *,
    water_crs: CRS,
    block_size: int,
    all_touched: bool,
) -> None:
    """按 DEM 网格分块计算体积，支持 DEM 与水体栅格分辨率不同。"""
    for obj in objects:
        if obj.props.get("skip_volume"):
            for k in ["v05_m3", "v50_m3", "v75_m3", "v95_m3", "vmean_m3"]:
                obj.props[k] = 0.0
    valid_objects = [obj for obj in objects if obj.props.get("h50") is not None and not obj.props.get("skip_volume")]
    if not valid_objects:
        LOGGER.warning("没有可计算体积的塘坝对象。")
        return

    with rasterio.open(dem_tif) as dem:
        dem_crs = parse_crs(dem.crs)
        transformer = None
        if dem_crs is not None and water_crs is not None and dem_crs != water_crs:
            transformer = Transformer.from_crs(water_crs, dem_crs, always_xy=True).transform

        id_to_obj: Dict[int, WaterObject] = {}
        dem_geoms: List[BaseGeometry] = []
        geom_ids: List[int] = []
        for obj in valid_objects:
            geom = shapely_transform(transformer, obj.geom) if transformer else obj.geom
            geom = fix_geometry(geom)
            if geom is None or geom.is_empty:
                continue
            id_to_obj[obj.obj_id] = obj
            dem_geoms.append(geom)
            geom_ids.append(obj.obj_id)
            obj.props.update({
                "dem_count": 0,
                "dem_min": None,
                "dem_max": None,
                "dem_mean": None,
                "dem_sum": 0.0,
                "v05_m3": 0.0,
                "v50_m3": 0.0,
                "v75_m3": 0.0,
                "v95_m3": 0.0,
                "vmean_m3": 0.0,
            })

        if not dem_geoms:
            return
        tree = STRtree(dem_geoms)
        geom_id_to_idx = {id(g): i for i, g in enumerate(dem_geoms)}
        pixel_area = abs(dem.transform.a * dem.transform.e)
        nodata = dem.nodata

        union_bounds = unary_union(dem_geoms).bounds
        dem_bounds = tuple(dem.bounds)
        try:
            proc_bounds = intersect_bounds(dem_bounds, union_bounds)
        except ValueError:
            for obj in valid_objects:
                obj.props["qa"] = "OUTSIDE_DEM"
            return

        proc_win = window_from_intersection(dem, proc_bounds)
        LOGGER.info("按 DEM 网格分块计算体积，DEM 窗口：%s", proc_win)

        for rel_win in iter_windows(int(proc_win.width), int(proc_win.height), block_size):
            win = Window(
                proc_win.col_off + rel_win.col_off,
                proc_win.row_off + rel_win.row_off,
                rel_win.width,
                rel_win.height,
            )
            win_bounds = window_bounds(win, dem.transform)
            win_box = box(*win_bounds)
            hits = tree.query(win_box)
            if len(hits) == 0:
                continue

            shapes_for_raster = []
            for item in hits:
                idx = int(item) if isinstance(item, (int, np.integer)) else geom_id_to_idx[id(item)]
                g = dem_geoms[idx]
                if g.intersects(win_box):
                    shapes_for_raster.append((mapping(g), int(geom_ids[idx])))
            if not shapes_for_raster:
                continue

            out_shape = (int(win.height), int(win.width))
            win_transform = transform_for_window(dem.transform, win)
            id_raster = rasterize(
                shapes_for_raster,
                out_shape=out_shape,
                transform=win_transform,
                fill=0,
                dtype="int32",
                all_touched=all_touched,
            )
            if not np.any(id_raster > 0):
                continue

            dem_arr = dem.read(1, window=win)
            valid = id_raster > 0
            if nodata is not None:
                valid &= ~np.isclose(dem_arr, nodata)
            valid &= np.isfinite(dem_arr)
            if not np.any(valid):
                continue

            ids = id_raster[valid]
            elevs = dem_arr[valid].astype("float64")
            for oid in np.unique(ids):
                obj = id_to_obj.get(int(oid))
                if obj is None:
                    continue
                mask = ids == oid
                vals = elevs[mask]
                if vals.size == 0:
                    continue

                cnt = int(vals.size)
                obj.props["dem_count"] += cnt
                obj.props["dem_sum"] += float(vals.sum())
                obj.props["dem_min"] = float(vals.min()) if obj.props["dem_min"] is None else min(obj.props["dem_min"], float(vals.min()))
                obj.props["dem_max"] = float(vals.max()) if obj.props["dem_max"] is None else max(obj.props["dem_max"], float(vals.max()))

                _h = ("h05", "h50", "h75", "h95", "hmean")
                _v = ("v05_m3", "v50_m3", "v75_m3", "v95_m3", "vmean_m3")
                levels = np.array([obj.props.get(h) if obj.props.get(h) is not None else np.nan for h in _h], dtype="float64")
                finite = np.isfinite(levels)
                if np.any(finite):
                    depths = np.maximum(levels[finite, np.newaxis] - vals[np.newaxis, :], 0.0)
                    vols = depths.sum(axis=1) * pixel_area
                    for vi, vf in zip(np.where(finite)[0], vols):
                        obj.props[_v[vi]] += float(vf)

        for obj in valid_objects:
            cnt = int(obj.props.get("dem_count", 0))
            if cnt > 0:
                obj.props["dem_mean"] = obj.props["dem_sum"] / cnt
                for k in ["dem_min", "dem_max", "dem_mean", "v05_m3", "v50_m3", "v75_m3", "v95_m3", "vmean_m3"]:
                    if obj.props.get(k) is not None:
                        obj.props[k] = round(float(obj.props[k]), 3)
            else:
                obj.props["qa"] = "NO_DEM_PIXELS"


# ----------------------------
# 输出
# ----------------------------

def write_objects_gpkg(objects: Sequence[WaterObject], out_gpkg: str, *, crs: Any, layer: str = "ponds") -> str:
    remove_existing_vector(out_gpkg)
    ensure_dir(Path(out_gpkg).parent)
    schema = {
        "geometry": "Unknown",
        "properties": {
            "pond_id": "int",
            "area_m2": "float:20.3",
            "area_mu": "float:20.3",
            "bnd_n": "int",
            "h05": "float:20.3",
            "h50": "float:20.3",
            "h75": "float:20.3",
            "h95": "float:20.3",
            "hmean": "float:20.3",
            "v05_m3": "float:20.3",
            "v50_m3": "float:20.3",
            "v75_m3": "float:20.3",
            "v95_m3": "float:20.3",
            "vmean_m3": "float:20.3",
            "dem_min": "float:20.3",
            "dem_mean": "float:20.3",
            "dem_max": "float:20.3",
            "qa": "str:80",
        },
    }
    with fiona.open(out_gpkg, "w", driver="GPKG", layer=layer, crs=crs, schema=schema) as dst:
        for obj in objects:
            props = {
                "pond_id": obj.obj_id,
                "area_m2": round(float(obj.props.get("area_m2", obj.area_m2)), 3),
                "area_mu": round(float(obj.props.get("area_mu", obj.area_m2 / 666.6666667)), 3),
                "bnd_n": int(obj.props.get("bnd_n") or 0),
                "h05": obj.props.get("h05"),
                "h50": obj.props.get("h50"),
                "h75": obj.props.get("h75"),
                "h95": obj.props.get("h95"),
                "hmean": obj.props.get("hmean"),
                "v05_m3": obj.props.get("v05_m3"),
                "v50_m3": obj.props.get("v50_m3"),
                "v75_m3": obj.props.get("v75_m3"),
                "v95_m3": obj.props.get("v95_m3"),
                "vmean_m3": obj.props.get("vmean_m3"),
                "dem_min": obj.props.get("dem_min"),
                "dem_mean": obj.props.get("dem_mean"),
                "dem_max": obj.props.get("dem_max"),
                "qa": obj.props.get("qa", "OK"),
            }
            dst.write({"geometry": mapping(obj.geom), "properties": props})
    return out_gpkg


def write_objects_csv(objects: Sequence[WaterObject], out_csv: str) -> str:
    ensure_dir(Path(out_csv).parent)
    fields = [
        "pond_id", "area_m2", "area_mu", "bnd_n", "h05", "h50", "h75", "h95", "hmean",
        "v05_m3", "v50_m3", "v75_m3", "v95_m3", "vmean_m3", "dem_min", "dem_mean", "dem_max", "qa",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for obj in objects:
            row = {"pond_id": obj.obj_id}
            row.update({k: obj.props.get(k) for k in fields if k != "pond_id"})
            row["area_m2"] = round(float(row.get("area_m2") or obj.area_m2), 3)
            row["area_mu"] = round(float(row.get("area_mu") or obj.area_m2 / 666.6666667), 3)
            writer.writerow(row)
    return out_csv


def write_summary(summary: Dict[str, Any], out_json: str, out_csv: str) -> None:
    ensure_dir(Path(out_json).parent)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    rows = []
    for name, data in summary.get("layers", {}).items():
        rows.append({
            "layer": name,
            "source": data.get("source"),
            "raw_overlap_pixels": data.get("raw_overlap_pixels", 0),
            "raw_overlap_m2": data.get("raw_overlap_m2", 0.0),
            "raw_overlap_mu": data.get("raw_overlap_m2", 0.0) / 666.6666667,
        })
    rows.append({
        "layer": "UNION_DEDUCTED",
        "source": "all_deduct_layers_union",
        "raw_overlap_pixels": summary.get("deducted_union_pixels", 0),
        "raw_overlap_m2": summary.get("deducted_union_area_m2", 0.0),
        "raw_overlap_mu": summary.get("deducted_union_area_m2", 0.0) / 666.6666667,
    })
    rows.append({
        "layer": "POND_CANDIDATE_AFTER_DEDUCT",
        "source": "water_after_deduction",
        "raw_overlap_pixels": summary.get("pond_candidate_pixels", 0),
        "raw_overlap_m2": summary.get("pond_candidate_area_m2", 0.0),
        "raw_overlap_mu": summary.get("pond_candidate_area_m2", 0.0) / 666.6666667,
    })

    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["layer", "source", "raw_overlap_pixels", "raw_overlap_m2", "raw_overlap_mu"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# ----------------------------
# 主流程
# ----------------------------

def load_config(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    if yaml is None:
        raise RuntimeError("读取 YAML 配置需要安装 pyyaml")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def dump_yaml(obj: Dict[str, Any], path: str | Path) -> None:
    if yaml is None:
        raise RuntimeError("写入 YAML 配置需要安装 pyyaml")
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, allow_unicode=True, sort_keys=False)


AUTO_CONFIG_VALUES = {"auto", "latest", "new"}
WATER_MASK_RE = re.compile(
    r"^water_mask_(?P<date>\d{4}-\d{2}-\d{2}|\d{8})(?:_(?P<region>.+))?$",
    re.IGNORECASE,
)


def is_auto_config_value(value: Any) -> bool:
    if value is None:
        return True
    return str(value).strip().lower() in AUTO_CONFIG_VALUES


def normalize_path_value(value: Any) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip()
    return s or None


def parse_water_mask_name(path: str | Path) -> Tuple[Optional[str], Optional[str]]:
    match = WATER_MASK_RE.match(Path(path).stem)
    if not match:
        return None, None
    date_token = match.group("date")
    if date_token and len(date_token) == 8:
        date_token = f"{date_token[:4]}-{date_token[4:6]}-{date_token[6:8]}"
    region = match.group("region")
    return date_token, region


def infer_region_from_config(cfg: Dict[str, Any], config_path: Optional[str] = None) -> Optional[str]:
    for key in ("region", "region_name", "region_tag"):
        value = normalize_path_value(cfg.get(key))
        if value:
            return value

    for key in ("out_dir", "aoi_shp", "river_dir"):
        value = normalize_path_value(cfg.get(key))
        if not value:
            continue
        stem = Path(value).stem if key == "aoi_shp" else Path(value).name
        if stem and stem.lower() not in {"output", "final", "pond_storage", "buffered_river"}:
            return stem

    if config_path:
        stem = Path(config_path).stem
        stem = re.sub(r"_total_tuned(?:_formal)?$", "", stem)
        stem = re.sub(r"_formal$", "", stem)
        if stem and not stem.startswith("monitor_cfg"):
            return stem
    return None


def water_mask_sort_key(path: Path) -> Tuple[datetime, float, str]:
    date_tag, _ = parse_water_mask_name(path)
    if date_tag:
        try:
            dt = datetime.strptime(date_tag, "%Y-%m-%d")
        except ValueError:
            dt = datetime.fromtimestamp(path.stat().st_mtime)
    else:
        dt = datetime.fromtimestamp(path.stat().st_mtime)
    return dt, path.stat().st_mtime, path.name.lower()


def find_latest_source_water_mask(mask_dir: str | Path, region: Optional[str]) -> Path:
    root = Path(mask_dir)
    if not root.exists():
        raise FileNotFoundError(f"源水面掩膜目录不存在：{root}")

    all_masks = [p for p in root.glob("water_mask_*.tif") if p.is_file()]
    if not all_masks:
        raise FileNotFoundError(f"源水面掩膜目录内没有 water_mask_*.tif：{root}")

    region_masks = all_masks
    if region:
        region_lower = region.lower()
        region_masks = []
        for path in all_masks:
            _, mask_region = parse_water_mask_name(path)
            if mask_region and mask_region.lower() == region_lower:
                region_masks.append(path)
            elif path.stem.lower().endswith(f"_{region_lower}"):
                region_masks.append(path)
        if not region_masks:
            raise FileNotFoundError(f"未找到区域 {region} 对应的源水面掩膜：{root}")

    return sorted(region_masks, key=water_mask_sort_key, reverse=True)[0]


def infer_date_from_water_tif(path: str | Path) -> str:
    date_tag, _ = parse_water_mask_name(path)
    if date_tag:
        return date_tag
    p = Path(path)
    if p.exists():
        return datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d")
    return p.stem


def resolve_runtime_water_tif_and_date(
    *,
    cli_water_tif: Optional[str],
    cli_date: Optional[str],
    cfg: Dict[str, Any],
    config_path: Optional[str],
    project_root: Optional[str | Path] = None,
) -> Tuple[str, str]:
    """Resolve the actual water mask and date used for this run.

    `water_tif: auto` selects the latest `water_mask_*.tif` in
    `source_water_mask_dir`, filtered by region when available.
    `date: auto` derives the output date tag from the selected mask filename.
    """
    root = Path(project_root) if project_root is not None else Path(__file__).resolve().parent.parent
    region = infer_region_from_config(cfg, config_path)
    source_dir = (
        normalize_path_value(cfg.get("source_water_mask_dir"))
        or normalize_path_value(cfg.get("source_water_masks_dir"))
        or str(root / "output" / "source_water_masks")
    )

    water_value = cli_water_tif if cli_water_tif is not None else cfg.get("water_tif")
    if is_auto_config_value(water_value):
        water_path = find_latest_source_water_mask(source_dir, region)
        LOGGER.info("自动选择源水面掩膜：%s", water_path)
    else:
        water_path = Path(str(water_value))
        if not water_path.exists():
            fallback = find_latest_source_water_mask(source_dir, region)
            LOGGER.warning("配置中的 water_tif 不存在：%s；自动改用：%s", water_path, fallback)
            water_path = fallback

    date_value = cli_date if cli_date is not None else cfg.get("date")
    if is_auto_config_value(date_value):
        date_tag = infer_date_from_water_tif(water_path)
    else:
        date_tag = str(date_value)

    return str(water_path), date_tag


def write_resolved_runtime_config(
    *,
    cfg: Dict[str, Any],
    out_dir: str | Path,
    water_tif: str,
    date_tag: str,
) -> Path:
    resolved = dict(cfg)
    resolved["water_tif"] = water_tif
    resolved["date"] = date_tag
    resolved["resolved_water_tif"] = water_tif
    resolved["resolved_date"] = date_tag
    path = ensure_dir(out_dir) / f"resolved_monitor_cfg_{date_tag}.yaml"
    dump_yaml(resolved, path)
    return path


def arg_or_cfg(args: argparse.Namespace, cfg: Dict[str, Any], key: str, default: Any = None) -> Any:
    v = getattr(args, key, None)
    if v is None:
        return cfg.get(key, default)
    # argparse 中 bool 默认 False 不能直接覆盖配置，因此这里只对 None 做 fallback。
    return v


def run(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = load_config(args.config)

    out_dir = ensure_dir(arg_or_cfg(args, cfg, "out_dir", "output"))
    water_tif, date_tag = resolve_runtime_water_tif_and_date(
        cli_water_tif=getattr(args, "water_tif", None),
        cli_date=getattr(args, "date", None),
        cfg=cfg,
        config_path=getattr(args, "config", None),
        project_root=Path(__file__).resolve().parent.parent,
    )
    resolved_cfg_path = write_resolved_runtime_config(
        cfg=cfg,
        out_dir=out_dir,
        water_tif=water_tif,
        date_tag=date_tag,
    )

    aoi_shp = arg_or_cfg(args, cfg, "aoi_shp")
    river_dir = arg_or_cfg(args, cfg, "river_dir")
    reservoir_shp = arg_or_cfg(args, cfg, "reservoir_shp")
    extra_vectors = arg_or_cfg(args, cfg, "extra_vectors", []) or []
    if isinstance(extra_vectors, str):
        extra_vectors = [extra_vectors]

    with rasterio.open(water_tif) as water_src:
        water_crs = parse_crs(water_src.crs)
        if water_crs is None:
            raise ValueError("水体栅格缺少 CRS，无法做矢量扣除和 DEM 配准。")
        water_bounds = tuple(water_src.bounds)

    aoi_geom = read_union_geometry(aoi_shp, water_crs) if aoi_shp else None
    if aoi_geom is not None:
        process_bounds = intersect_bounds(water_bounds, aoi_geom.bounds)
    else:
        process_bounds = water_bounds

    dem_tif = arg_or_cfg(args, cfg, "dem_tif")
    if not dem_tif:
        raise ValueError("必须提供真实 DEM 路径（dem_tif 或 --dem-tif），不再支持合成 DEM。")
    if not Path(str(dem_tif)).exists():
        raise FileNotFoundError(f"DEM 文件不存在：{dem_tif}")

    # 用 DEM 范围再次收紧处理范围，避免水体超出 DEM。
    with rasterio.open(str(dem_tif)) as dem_src:
        dem_crs = parse_crs(dem_src.crs)
        dem_bounds_in_water = transform_bounds(dem_crs, water_crs, *dem_src.bounds, densify_pts=21) if dem_crs != water_crs else tuple(dem_src.bounds)
        process_bounds = intersect_bounds(process_bounds, dem_bounds_in_water)

    line_buffer_m = float(arg_or_cfg(args, cfg, "line_buffer_m", 15.0))
    bbox_for_deduct = (
        process_bounds[0] - line_buffer_m,
        process_bounds[1] - line_buffer_m,
        process_bounds[2] + line_buffer_m,
        process_bounds[3] + line_buffer_m,
    )
    deduct_layers = build_deduct_layers(
        river_dir=river_dir,
        reservoir_shp=reservoir_shp,
        extra_vectors=extra_vectors,
        target_crs=water_crs,
        bbox_filter=bbox_for_deduct,
        line_buffer_m=line_buffer_m,
    )

    landuse_water_tif = arg_or_cfg(args, cfg, "landuse_water_tif", None) or None
    _landuse_hard_cfg = arg_or_cfg(args, cfg, "landuse_water_hard_mask", None)
    landuse_hard_mask = bool(_landuse_hard_cfg) if _landuse_hard_cfg is not None else bool(landuse_water_tif)
    landuse_water_value = int(arg_or_cfg(args, cfg, "landuse_water_value", 1))

    mask_tif = str(out_dir / f"pond_water_mask_{date_tag}.tif")
    mask_tif, summary = prepare_water_mask(
        water_tif,
        mask_tif,
        process_bounds=process_bounds,
        aoi_geom=aoi_geom,
        deduct_layers=deduct_layers,
        water_value=arg_or_cfg(args, cfg, "water_value", None),
        respect_source_mask=bool(arg_or_cfg(args, cfg, "respect_source_mask", False)),
        block_size=int(arg_or_cfg(args, cfg, "block_size", 512)),
        close_radius_px=int(arg_or_cfg(args, cfg, "close_radius_px", 0)),
        open_radius_px=int(arg_or_cfg(args, cfg, "open_radius_px", 0)),
        fill_holes=bool(arg_or_cfg(args, cfg, "fill_holes", False)),
        all_touched=bool(arg_or_cfg(args, cfg, "all_touched", False)),
        cropland_mask_tif=arg_or_cfg(args, cfg, "cropland_mask_tif", None) or None,
        landuse_water_hard_mask_tif=str(landuse_water_tif) if (landuse_water_tif and landuse_hard_mask) else None,
        landuse_water_value=landuse_water_value,
    )

    _max_area_cfg = arg_or_cfg(args, cfg, "max_water_area_m2", None)
    ws_enable = bool(arg_or_cfg(args, cfg, "watershed_split_enable", True))
    ws_min_area = float(arg_or_cfg(args, cfg, "watershed_split_min_area_m2", 6000.0))
    ws_peak_dist = float(arg_or_cfg(args, cfg, "watershed_peak_min_distance_m", 10.0))
    labels_out_tif = str(out_dir / f"pond_labels_{date_tag}.tif") if ws_enable else None
    objects = polygonize_water_mask(
        mask_tif,
        min_area_m2=float(arg_or_cfg(args, cfg, "min_water_area_m2", 500.0)),
        max_area_m2=float(_max_area_cfg) if _max_area_cfg is not None else None,
        simplify_tolerance=float(arg_or_cfg(args, cfg, "simplify_tolerance_m", 0.0)),
        watershed_split_enable=ws_enable,
        watershed_split_min_area_m2=ws_min_area,
        watershed_peak_min_distance_m=ws_peak_dist,
        labels_out_tif=labels_out_tif,
    )

    # 当 landuse_water 已作为硬掩膜使用，就不再做"任意相交保留"的弱过滤。
    if landuse_water_tif and not landuse_hard_mask:
        objects = filter_by_landuse_water(objects, str(landuse_water_tif), water_crs)

    if bool(arg_or_cfg(args, cfg, "linear_artifact_filter_enable", True)):
        objects = filter_linear_artifacts(
            objects,
            apply_min_area_m2=float(arg_or_cfg(args, cfg, "linear_artifact_apply_min_area_m2", 3000.0)),
            max_elongation=float(arg_or_cfg(args, cfg, "linear_artifact_max_elongation", 12.0)),
            min_compactness=float(arg_or_cfg(args, cfg, "linear_artifact_min_compactness", 0.06)),
            min_width_m=float(arg_or_cfg(args, cfg, "linear_artifact_max_width_m", 8.0)),
        )

    skip_volume_above_area_m2 = arg_or_cfg(args, cfg, "skip_volume_above_area_m2", 500000.0)
    if skip_volume_above_area_m2 is not None:
        skip_thr = float(skip_volume_above_area_m2)
        for obj in objects:
            if float(obj.props.get("area_m2", obj.area_m2)) > skip_thr:
                obj.props["skip_volume"] = True
                obj.props["qa"] = "TOO_LARGE_SKIP_VOLUME"

    estimate_water_levels(
        objects,
        str(dem_tif),
        water_crs=water_crs,
        sample_step_m=float(arg_or_cfg(args, cfg, "boundary_sample_step_m", 5.0)),
        max_boundary_samples=int(arg_or_cfg(args, cfg, "max_boundary_samples", 5000)),
    )

    compute_volumes(
        objects,
        str(dem_tif),
        water_crs=water_crs,
        block_size=int(arg_or_cfg(args, cfg, "block_size", 512)),
        all_touched=bool(arg_or_cfg(args, cfg, "all_touched", False)),
    )

    out_gpkg = str(out_dir / f"pond_storage_{date_tag}.gpkg")
    out_csv = str(out_dir / f"pond_storage_{date_tag}.csv")
    write_objects_gpkg(objects, out_gpkg, crs=FionaCRS.from_user_input(water_crs.to_string()))
    write_objects_csv(objects, out_csv)

    summary.update({
        "water_tif": water_tif,
        "dem_tif": str(dem_tif),
        "aoi_shp": aoi_shp,
        "river_dir": river_dir,
        "reservoir_shp": reservoir_shp,
        "process_bounds": process_bounds,
        "pond_count_after_min_area": len(objects),
        "pond_area_after_min_area_m2": float(sum(o.area_m2 for o in objects)),
        "outputs": {
            "mask_tif": mask_tif,
            "pond_gpkg": out_gpkg,
            "pond_csv": out_csv,
            "resolved_config": str(resolved_cfg_path),
            "summary_json": str(out_dir / f"deduct_summary_{date_tag}.json"),
            "summary_csv": str(out_dir / f"deduct_summary_{date_tag}.csv"),
        },
    })
    write_summary(summary, summary["outputs"]["summary_json"], summary["outputs"]["summary_csv"])

    LOGGER.info("完成。输出：%s", out_dir)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="水体扣除、塘坝水面面积与蓄水量分块计算工具",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", help="YAML 配置文件")
    parser.add_argument("--water-tif", dest="water_tif", help="水体二值/分类 GeoTIFF，水体默认为像元值 > 0")
    parser.add_argument("--dem-tif", dest="dem_tif", required=False, help="真实 DEM GeoTIFF（必填，可写入配置或命令行）")
    parser.add_argument("--aoi-shp", dest="aoi_shp", help="处理区域 AOI，可选；不填则使用水体栅格全图")
    parser.add_argument("--river-dir", dest="river_dir", help="河流/渠道/湖泊等扣除矢量文件夹，递归读取 .shp")
    parser.add_argument("--reservoir-shp", dest="reservoir_shp", help="大中小型水库等已知水库扣除矢量")
    parser.add_argument("--extra-vector", dest="extra_vectors", action="append", default=None, help="其他扣除矢量，可重复")
    parser.add_argument("--out-dir", dest="out_dir", default=None, help="输出目录")
    parser.add_argument("--date", dest="date", help="日期/批次标签，例如 20260321")
    parser.add_argument("--water-value", dest="water_value", type=int, help="水体类别值；不填则使用 >0")
    parser.add_argument("--respect-source-mask", action="store_true", default=None, help="是否同时使用源 GeoTIFF 的内部/外部 mask")
    parser.add_argument("--line-buffer-m", type=float, default=None, help="线状渠道/河流扣除缓冲宽度，单位米")
    parser.add_argument("--min-water-area-m2", type=float, default=None, help="塘坝最小保留面积，单位 m2")
    parser.add_argument("--max-water-area-m2", type=float, default=None, help="塘坝最大保留面积，单位 m2；超过此面积视为湖泊/河段并剔除")
    parser.add_argument("--block-size", type=int, default=None, help="分块大小，像元")
    parser.add_argument("--close-radius-px", type=int, default=None, help="水体闭运算半径，像元；0 表示关闭")
    parser.add_argument("--open-radius-px", type=int, default=None, help="水体开运算半径，像元；0 表示关闭")
    parser.add_argument("--fill-holes", action="store_true", default=None, help="是否填洞，复杂河网区慎用")
    parser.add_argument("--all-touched", action="store_true", default=None, help="栅格化时碰到即算入")
    parser.add_argument("--boundary-sample-step-m", type=float, default=None, help="水体边界 DEM 采样间距，单位米")
    parser.add_argument("--max-boundary-samples", type=int, default=None, help="单个水体边界最大采样点数")
    parser.add_argument("--cropland-mask-tif", dest="cropland_mask_tif", default=None, help="耕地掩膜 GeoTIFF，非零像元视为耕地并从塘坝候选中排除")
    parser.add_argument("--landuse-water-tif", dest="landuse_water_tif", default=None, help="土地利用水体 GeoTIFF（value=1=水体）；默认按硬掩膜取交集")
    parser.add_argument("--skip-volume-above-area-m2", type=float, default=None, help="面积超过该阈值的对象不参与体积计算，仅保留几何和 QA 标记")
    parser.add_argument("--simplify-tolerance-m", type=float, default=None, help="输出多边形简化容差，单位米")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def normalize_none_args(args: argparse.Namespace) -> argparse.Namespace:
    # argparse 的 bool action 默认 False 会覆盖配置；这里保留显式命令行选项的行为，
    # 常用配置仍建议写入 YAML。all_touched 若为 None，则在 arg_or_cfg 中走配置/默认。
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = normalize_none_args(parser.parse_args(argv))

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # argparse 对短横线参数转下划线；配置文件也使用下划线。
    try:
        summary = run(args)
        print(json.dumps({
            "pond_count": summary["pond_count_after_min_area"],
            "pond_area_m2": round(summary["pond_area_after_min_area_m2"], 3),
            "outputs": summary["outputs"],
        }, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        LOGGER.exception("运行失败：%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
