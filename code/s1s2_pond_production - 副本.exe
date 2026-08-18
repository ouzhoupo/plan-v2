#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生产级塘坝监测前端：基于 Sentinel-1 / Sentinel-2 原始多波段影像（分析就绪输入）
生成融合水体概率图 / 二值水体图，并可继续调用现有 pond_storage_monitor.py 完成
"非塘坝扣除 + DEM 体积计算"。

设计目标
--------
1. 不再要求上游直接提供二值水体，改为从 Sentinel-1 VV/VH 与 Sentinel-2 多光谱直接生成水体概率；
2. 在 S1/S2 重叠且 S2 晴空区域，用 SAR 结果对光学水体分数做本地统计校准；
3. 将校准后的光学模型迁移到只有光学更新、但没有 SAR 更新的区域；
4. 云遮挡区域优先使用近时相 SAR，若无 SAR 再回退到 prior / 上一期水体先验，并显式输出 source_flag；
5. 输出 water_prob.tif / water_mask.tif / source_flag.tif / confidence.tif，后端继续复用 V1 的扣除与体积代码。

重要说明
--------
- 本脚本默认输入为“分析就绪”的 Sentinel-1/2 波段 GeoTIFF 或 Sentinel-2 L2A SAFE 波段文件。
- 对 Sentinel-1，生产环境建议先做标准 ARD 预处理（辐射定标、RTC/GTC、边界噪声处理等），
  然后将 VV/VH 以 GeoTIFF 方式输入本脚本。脚本不内置 SNAP 全流程替代。
- 对 Sentinel-2，推荐使用 L2A（含 SCL）作为输入。

子命令
------
1. build-prior  : 由历史 water_mask 或参考水体图生成 soft prior
2. fuse-water   : 仅输出融合水体产品
3. pipeline     : 融合水体后，继续调用 pond_storage_monitor.py 计算塘坝蓄水量

依赖
----
numpy, scipy, rasterio, fiona, shapely, pyproj, scikit-learn, pyyaml
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import logging
import math
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def configure_proj_runtime() -> None:
    rasterio_spec = importlib.util.find_spec("rasterio")
    if rasterio_spec is None or not rasterio_spec.submodule_search_locations:
        os.environ.setdefault("GTIFF_SRS_SOURCE", "EPSG")
        return
    rasterio_dir = Path(next(iter(rasterio_spec.submodule_search_locations)))
    pkg_dir = Path(__file__).resolve().parent
    candidate_dirs = [
        rasterio_dir / "proj_data",
        pkg_dir / "proj_data",
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
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window, bounds as window_bounds, from_bounds as window_from_bounds
from scipy import ndimage
from shapely.geometry import box, mapping, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shapely_transform, unary_union
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

LOGGER = logging.getLogger("s1s2_pond_production")
EPS = 1e-6
AUTO_CONFIG_VALUES = {"auto", "latest", "new"}
MODEL_FEATURES = [
    "green", "nir", "swir1", "swir2", "red", "blue",
    "ndwi", "mndwi", "awei_sh", "ndvi", "prior", "slope"
]


# -----------------------------------------------------------------------------
# 基础工具
# -----------------------------------------------------------------------------

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


def load_yaml(path: str | Path) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError("读取 YAML 需要安装 pyyaml")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    # 解析 project_root：将相对路径转为绝对路径
    project_root = data.pop("project_root", None)
    if project_root:
        root = Path(project_root)
        if not root.is_absolute():
            root = Path(path).resolve().parent / project_root
        root = root.resolve()
        _resolve_paths(data, str(root).replace("\\", "/"))
    return data


def _resolve_paths(obj: Any, root: str) -> None:
    """递归遍历 dict，将值为相对路径的字段拼上 root 前缀。"""
    if not isinstance(obj, dict):
        return
    # 已知路径字段后缀
    _PATH_KEYS = {
        "scene_dir", "input_dir", "dem_tif", "aoi_shp", "river_dir",
        "reservoir_shp", "out_dir", "landuse_water_tif", "water_tif",
        "reference_water_tif", "prior_tif", "source_water_mask_dir",
    }
    for k, v in obj.items():
        if isinstance(v, dict):
            _resolve_paths(v, root)
        elif isinstance(v, str) and k in _PATH_KEYS and v and not Path(v).is_absolute():
            obj[k] = f"{root}/{v}"


def dump_yaml(obj: Dict[str, Any], path: str | Path) -> None:
    if yaml is None:
        raise RuntimeError("写 YAML 需要安装 pyyaml")
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, allow_unicode=True, sort_keys=False)


def dump_json(obj: Dict[str, Any], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def parse_date(s: str | None) -> Optional[datetime]:
    if not s:
        return None
    s = str(s)
    for fmt in ["%Y-%m-%d", "%Y%m%d", "%Y/%m/%d", "%Y-%m-%dT%H:%M:%S"]:
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            continue
    raise ValueError(f"无法解析日期: {s}")


def date_diff_days(a: Optional[datetime], b: Optional[datetime]) -> float:
    if a is None or b is None:
        return 0.0
    return abs((a - b).total_seconds()) / 86400.0


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    x = np.clip(x, -40, 40)
    return 1.0 / (1.0 + np.exp(-x))


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def safe_div(a: np.ndarray, b: np.ndarray, fill: float = 0.0) -> np.ndarray:
    out = np.full_like(a, fill, dtype="float32")
    valid = np.abs(b) > EPS
    out[valid] = (a[valid] / b[valid]).astype("float32")
    return out


def norm_index(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return safe_div(a - b, a + b, fill=0.0)


def local_std(arr: np.ndarray, size: int = 5) -> np.ndarray:
    mean = ndimage.uniform_filter(arr, size=size, mode="nearest")
    mean2 = ndimage.uniform_filter(arr * arr, size=size, mode="nearest")
    var = np.maximum(mean2 - mean * mean, 0.0)
    return np.sqrt(var, dtype="float32")


def read_union_geometry(vector_path: Optional[str], target_crs: Optional[CRS]) -> Optional[BaseGeometry]:
    if not vector_path:
        return None
    p = Path(vector_path)
    if not p.exists():
        return None
    geoms: List[BaseGeometry] = []
    with fiona.open(str(p)) as src:
        src_crs = parse_crs(src.crs_wkt or src.crs)
        for feat in src:
            if not feat.get("geometry"):
                continue
            geom = shape(feat["geometry"])
            if geom.is_empty:
                continue
            if src_crs and target_crs and src_crs != target_crs:
                tfm = Transformer.from_crs(src_crs, target_crs, always_xy=True)
                geom = shapely_transform(tfm.transform, geom)
            geoms.append(geom)
    if not geoms:
        return None
    return unary_union(geoms)


def snap_bounds(bounds: Tuple[float, float, float, float], res: float) -> Tuple[float, float, float, float]:
    left, bottom, right, top = bounds
    return (
        math.floor(left / res) * res,
        math.floor(bottom / res) * res,
        math.ceil(right / res) * res,
        math.ceil(top / res) * res,
    )


def intersect_bounds(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    left = max(a[0], b[0])
    bottom = max(a[1], b[1])
    right = min(a[2], b[2])
    top = min(a[3], b[3])
    if left >= right or bottom >= top:
        raise ValueError(f"无有效范围交集: {a} vs {b}")
    return (left, bottom, right, top)


def iter_windows(width: int, height: int, block_size: int) -> Iterable[Window]:
    for row in range(0, height, block_size):
        h = min(block_size, height - row)
        for col in range(0, width, block_size):
            w = min(block_size, width - col)
            yield Window(col, row, w, h)


def window_transform(base_transform: Affine, win: Window) -> Affine:
    return rasterio.windows.transform(win, base_transform)


def rasterize_geom_mask(geom: Optional[BaseGeometry], out_shape: Tuple[int, int], transform: Affine) -> np.ndarray:
    if geom is None or geom.is_empty:
        return np.ones(out_shape, dtype=bool)
    arr = rasterize([(mapping(geom), 1)], out_shape=out_shape, transform=transform, fill=0, dtype="uint8", all_touched=True)
    return arr.astype(bool)


def otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    vals = values[np.isfinite(values)]
    if vals.size < 32:
        return float(np.nanmedian(vals)) if vals.size else 0.0
    vmin, vmax = float(np.nanpercentile(vals, 0.5)), float(np.nanpercentile(vals, 99.5))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
        return float(np.nanmedian(vals))
    hist, edges = np.histogram(vals, bins=bins, range=(vmin, vmax))
    hist = hist.astype("float64")
    total = hist.sum()
    if total <= 0:
        return float(np.nanmedian(vals))
    prob = hist / total
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * np.arange(bins))
    mu_t = mu[-1]
    sigma_b2 = (mu_t * omega - mu) ** 2 / np.maximum(omega * (1 - omega), EPS)
    idx = int(np.nanargmax(sigma_b2))
    return float((edges[idx] + edges[idx + 1]) / 2.0)


def find_first(patterns: Sequence[str]) -> Optional[str]:
    for pat in patterns:
        hits = sorted(Path().glob(pat))
        if hits:
            return str(hits[0])
    return None


def is_auto_config_value(value: Any) -> bool:
    if value is None:
        return True
    return str(value).strip().lower() in AUTO_CONFIG_VALUES


def normalize_path_value(value: Any) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip()
    return s or None


def parse_date_from_filename(path: str | Path) -> Optional[str]:
    match = re.search(r"(\d{4}-\d{2}-\d{2}|\d{8})", Path(path).stem)
    if not match:
        return None
    token = match.group(1)
    if len(token) == 8:
        return f"{token[:4]}-{token[4:6]}-{token[6:8]}"
    return token


def file_recency_key(path: Path) -> Tuple[datetime, float, str]:
    date_tag = parse_date_from_filename(path)
    if date_tag:
        try:
            dt = datetime.strptime(date_tag, "%Y-%m-%d")
        except ValueError:
            dt = datetime.fromtimestamp(path.stat().st_mtime)
    else:
        dt = datetime.fromtimestamp(path.stat().st_mtime)
    return dt, path.stat().st_mtime, path.name.lower()


def normalize_globs(value: Any, default: Sequence[str]) -> List[str]:
    if value is None:
        return list(default)
    if isinstance(value, str):
        return [value]
    return [str(v) for v in value]


def find_all_dated_scenes(scene_dir: str | Path, patterns: Sequence[str]) -> Dict[str, Path]:
    """扫描目录中所有含日期的影像，返回 {日期字符串: 文件路径}。"""
    root = Path(scene_dir)
    result: Dict[str, Path] = {}
    if not root.exists():
        return result
    hits: List[Path] = []
    for pattern in patterns:
        hits.extend(p for p in root.glob(pattern) if p.is_file())
    for p in sorted(set(hits)):
        date_str = parse_date_from_filename(p)
        if date_str:
            result[date_str] = p
    return result


def discover_multi_date_pairs(
    s1_dir: str | Path,
    s2_dir: str | Path,
    s1_patterns: Sequence[str] = ("*.tif", "*.tiff"),
    s2_patterns: Sequence[str] = ("*.tif", "*.tiff"),
) -> List[Tuple[str, Path, Path]]:
    """发现 S1/S2 目录中日期匹配的影像对，按日期排序返回。"""
    s1_map = find_all_dated_scenes(s1_dir, s1_patterns)
    s2_map = find_all_dated_scenes(s2_dir, s2_patterns)
    common = sorted(set(s1_map.keys()) & set(s2_map.keys()))
    return [(d, s1_map[d], s2_map[d]) for d in common]


def find_latest_scene_file(scene_dir: str | Path, patterns: Sequence[str]) -> Path:
    root = Path(scene_dir)
    if not root.exists():
        raise FileNotFoundError(f"自动输入目录不存在：{root}")
    hits: List[Path] = []
    for pattern in patterns:
        hits.extend(p for p in root.glob(pattern) if p.is_file())
    if not hits:
        raise FileNotFoundError(f"自动输入目录内没有匹配影像：{root} patterns={list(patterns)}")
    return sorted(set(hits), key=file_recency_key, reverse=True)[0]


def resolve_auto_scene_configs(
    sensor_cfg: Dict[str, Any],
    *,
    sensor: str,
    project_root: Optional[str | Path] = None,
) -> List[Dict[str, Any]]:
    scenes = sensor_cfg.get("scenes", [])
    if isinstance(scenes, list) and scenes:
        return [dict(scene) for scene in scenes]
    if isinstance(scenes, dict):
        return [dict(scenes)]

    has_auto_input = (
        is_auto_config_value(scenes)
        or any(k in sensor_cfg for k in ("scene_dir", "input_dir", "scene_glob"))
    )
    if not has_auto_input:
        return []

    root = Path(project_root) if project_root is not None else Path(__file__).resolve().parent.parent
    default_subdir = "s1" if sensor == "sentinel1" else "s2"
    scene_dir = (
        normalize_path_value(sensor_cfg.get("scene_dir"))
        or normalize_path_value(sensor_cfg.get("input_dir"))
        or str(root / "data" / default_subdir)
    )
    patterns = normalize_globs(sensor_cfg.get("scene_glob"), ["*.tif", "*.tiff"])
    stack_path = find_latest_scene_file(scene_dir, patterns)

    inferred_date = parse_date_from_filename(stack_path)
    explicit_date = sensor_cfg.get("date")
    date_value = inferred_date if is_auto_config_value(explicit_date) else explicit_date

    scene: Dict[str, Any] = {
        "name": str(sensor_cfg.get("name") or stack_path.stem),
        "stack_path": str(stack_path),
    }
    if date_value:
        scene["date"] = str(date_value)

    if sensor == "sentinel1":
        scene.update({
            "vv_band": int(sensor_cfg.get("vv_band", 1)),
            "vh_band": int(sensor_cfg.get("vh_band", 2)),
            "in_db": bool(sensor_cfg.get("in_db", True)),
        })
    elif sensor == "sentinel2":
        defaults = {"b02": 1, "b03": 2, "b04": 3, "b08": 4, "b11": 5, "b12": 6, "scl": 7}
        for key, band in defaults.items():
            scene[f"{key}_band"] = int(sensor_cfg.get(f"{key}_band", band))
    else:
        raise ValueError(f"未知传感器类型：{sensor}")

    LOGGER.info("自动选择 %s 输入影像：%s", sensor, stack_path)
    return [scene]


def _build_scene_dict(sensor_cfg: Dict[str, Any], sensor: str, stack_path: Path, date_str: str) -> Dict[str, Any]:
    """根据传感器配置和指定文件路径构建单景配置字典。"""
    scene: Dict[str, Any] = {
        "name": str(sensor_cfg.get("name") or stack_path.stem),
        "stack_path": str(stack_path),
        "date": date_str,
    }
    if sensor == "sentinel1":
        scene.update({
            "vv_band": int(sensor_cfg.get("vv_band", 1)),
            "vh_band": int(sensor_cfg.get("vh_band", 2)),
            "in_db": bool(sensor_cfg.get("in_db", True)),
        })
    elif sensor == "sentinel2":
        defaults = {"b02": 1, "b03": 2, "b04": 3, "b08": 4, "b11": 5, "b12": 6, "scl": 7}
        for key, band in defaults.items():
            scene[f"{key}_band"] = int(sensor_cfg.get(f"{key}_band", band))
    return scene


def resolve_multi_date_configs(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """扫描 S1/S2 目录中所有带日期的影像对，为每期生成一份独立配置。"""
    import copy
    s1_cfg = config.get("sentinel1", {}) or {}
    s2_cfg = config.get("sentinel2", {}) or {}

    s1_dir = normalize_path_value(s1_cfg.get("scene_dir")) or normalize_path_value(s1_cfg.get("input_dir"))
    s2_dir = normalize_path_value(s2_cfg.get("scene_dir")) or normalize_path_value(s2_cfg.get("input_dir"))
    if not s1_dir or not s2_dir:
        raise ValueError("multi-pipeline 需要配置 sentinel1.scene_dir 和 sentinel2.scene_dir")

    s1_patterns = normalize_globs(s1_cfg.get("scene_glob"), ["*.tif", "*.tiff"])
    s2_patterns = normalize_globs(s2_cfg.get("scene_glob"), ["*.tif", "*.tiff"])

    pairs = discover_multi_date_pairs(s1_dir, s2_dir, s1_patterns, s2_patterns)
    if not pairs:
        raise FileNotFoundError(f"未发现日期配对的 S1/S2 影像：s1={s1_dir}, s2={s2_dir}")

    LOGGER.info("多期影像发现 %d 期：%s", len(pairs), [d for d, _, _ in pairs])

    original_out_dir = Path(config.get("out_dir", "output"))
    downstream_cfg = config.get("downstream", {}) or {}
    original_ds_out_dir = downstream_cfg.get("out_dir")
    region = downstream_cfg.get("region", "")

    configs: List[Dict[str, Any]] = []
    for date_str, s1_path, s2_path in pairs:
        cfg = copy.deepcopy(config)
        cfg["date"] = date_str
        cfg["sentinel1"] = dict(s1_cfg)
        cfg["sentinel2"] = dict(s2_cfg)
        cfg["sentinel1"]["scenes"] = [_build_scene_dict(s1_cfg, "sentinel1", s1_path, date_str)]
        cfg["sentinel2"]["scenes"] = [_build_scene_dict(s2_cfg, "sentinel2", s2_path, date_str)]
        # 目录名用 YYYYMMDD 格式，日期在前流域在后
        dir_date = date_str.replace("-", "")
        # 融合输出 → output/<YYYYMMDD>/<region>/
        date_out_dir = str(original_out_dir.parent / dir_date / original_out_dir.name)
        cfg["out_dir"] = date_out_dir
        # 下游蓄量输出 → output/<YYYYMMDD>/final/<region>/
        if original_ds_out_dir:
            ds_out = Path(original_ds_out_dir)
            cfg.setdefault("downstream", {})
            cfg["downstream"]["out_dir"] = str(original_out_dir.parent / dir_date / "final" / ds_out.name)
            cfg["downstream"]["source_water_mask_dir"] = date_out_dir
        configs.append(cfg)
    return configs


def infer_date_from_scene_configs(scene_cfgs: Sequence[Dict[str, Any]]) -> Optional[str]:
    for scene in scene_cfgs:
        value = scene.get("date")
        if value:
            parsed = parse_date(value)
            return parsed.strftime("%Y-%m-%d") if parsed else str(value)
        for key in ("stack_path", "vv_path", "vh_path", "b02_path", "safe_dir"):
            path_value = scene.get(key)
            if path_value:
                path = Path(str(path_value))
                date_tag = parse_date_from_filename(path)
                if date_tag:
                    return date_tag
                if path.exists():
                    return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d")
    return None


def resolve_runtime_auto_inputs(
    config: Dict[str, Any],
    *,
    project_root: Optional[str | Path] = None,
) -> Dict[str, Any]:
    resolved = dict(config)
    s1_cfg = dict(config.get("sentinel1", {}) or {})
    s2_cfg = dict(config.get("sentinel2", {}) or {})
    s1_scenes = resolve_auto_scene_configs(s1_cfg, sensor="sentinel1", project_root=project_root)
    s2_scenes = resolve_auto_scene_configs(s2_cfg, sensor="sentinel2", project_root=project_root)

    if s1_scenes:
        s1_cfg["scenes"] = s1_scenes
        resolved["sentinel1"] = s1_cfg
    if s2_scenes:
        s2_cfg["scenes"] = s2_scenes
        resolved["sentinel2"] = s2_cfg

    if is_auto_config_value(config.get("date")):
        date_tag = infer_date_from_scene_configs(s1_scenes) or infer_date_from_scene_configs(s2_scenes)
        if date_tag:
            resolved["date"] = date_tag
    return resolved


def jsonable(obj: Any) -> Any:
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    return obj


# -----------------------------------------------------------------------------
# RasterReader：统一重投影到目标格网
# -----------------------------------------------------------------------------

@dataclass
class RasterBandRef:
    path: str
    band: int = 1
    resampling: str = "bilinear"
    nodata: Optional[float] = None


class RasterReader:
    def __init__(
        self,
        band_ref: RasterBandRef,
        *,
        target_crs: CRS,
        target_transform: Affine,
        width: int,
        height: int,
    ) -> None:
        self.band_ref = band_ref
        self.src = rasterio.open(band_ref.path)
        resampling = getattr(Resampling, band_ref.resampling)
        self.vrt = WarpedVRT(
            self.src,
            crs=target_crs,
            transform=target_transform,
            width=width,
            height=height,
            resampling=resampling,
            nodata=band_ref.nodata,
        )
        self.band = band_ref.band

    def read(self, window: Window, out_dtype: str = "float32") -> np.ndarray:
        return self.vrt.read(self.band, window=window, out_dtype=out_dtype, masked=False)

    def mask(self, window: Window) -> np.ndarray:
        return self.vrt.read_masks(self.band, window=window) > 0

    def read_coarse(self, out_shape: Tuple[int, int], out_dtype: str = "float32") -> np.ndarray:
        return self.vrt.read(self.band, out_shape=out_shape, out_dtype=out_dtype, masked=False)

    def close(self) -> None:
        try:
            self.vrt.close()
        finally:
            self.src.close()


# -----------------------------------------------------------------------------
# Sentinel-1 / Sentinel-2 场景解析
# -----------------------------------------------------------------------------

@dataclass
class S1Scene:
    date: Optional[datetime]
    vv: RasterReader
    vh: RasterReader
    in_db: bool = True
    name: str = "s1"
    params: Dict[str, Any] = None


@dataclass
class S2Scene:
    date: Optional[datetime]
    b02: RasterReader
    b03: RasterReader
    b04: RasterReader
    b08: RasterReader
    b11: RasterReader
    b12: RasterReader
    scl: RasterReader
    name: str = "s2"


@dataclass
class GridSpec:
    crs: CRS
    transform: Affine
    width: int
    height: int
    bounds: Tuple[float, float, float, float]
    res: float
    aoi_geom: Optional[BaseGeometry]


def _resolve_stack_or_single(scene: Dict[str, Any], key: str, default_band_key: str) -> RasterBandRef:
    path_key = f"{key}_path"
    band_key = f"{key}_band"
    if scene.get(path_key):
        return RasterBandRef(path=str(scene[path_key]), band=int(scene.get(band_key, 1)))
    if scene.get("stack_path"):
        if band_key not in scene:
            raise ValueError(f"场景 {scene.get('name', '')} 缺少 {band_key}")
        return RasterBandRef(path=str(scene["stack_path"]), band=int(scene[band_key]))
    raise ValueError(f"场景 {scene.get('name', '')} 缺少 {path_key} / stack_path")


def resolve_s2_safe(scene: Dict[str, Any]) -> Dict[str, str]:
    safe_dir = Path(scene["safe_dir"])
    if not safe_dir.exists():
        raise FileNotFoundError(f"SAFE 目录不存在: {safe_dir}")

    def one(globs: Sequence[str]) -> str:
        hits: List[Path] = []
        for g in globs:
            hits.extend(sorted(safe_dir.glob(g)))
        if not hits:
            raise FileNotFoundError(f"SAFE 中未找到波段: {globs}")
        return str(hits[0])

    return {
        "b02_path": one(["GRANULE/*/IMG_DATA/R10m/*B02_10m.jp2", "GRANULE/*/IMG_DATA/*B02*.jp2"]),
        "b03_path": one(["GRANULE/*/IMG_DATA/R10m/*B03_10m.jp2", "GRANULE/*/IMG_DATA/*B03*.jp2"]),
        "b04_path": one(["GRANULE/*/IMG_DATA/R10m/*B04_10m.jp2", "GRANULE/*/IMG_DATA/*B04*.jp2"]),
        "b08_path": one(["GRANULE/*/IMG_DATA/R10m/*B08_10m.jp2", "GRANULE/*/IMG_DATA/*B08*.jp2"]),
        "b11_path": one(["GRANULE/*/IMG_DATA/R20m/*B11_20m.jp2", "GRANULE/*/IMG_DATA/*B11*.jp2"]),
        "b12_path": one(["GRANULE/*/IMG_DATA/R20m/*B12_20m.jp2", "GRANULE/*/IMG_DATA/*B12*.jp2"]),
        "scl_path": one(["GRANULE/*/IMG_DATA/R20m/*SCL_20m.jp2", "GRANULE/*/IMG_DATA/*SCL*.jp2"]),
    }


def infer_grid(config: Dict[str, Any]) -> GridSpec:
    target_res = float(config.get("target_resolution", 10.0))
    target_crs = parse_crs(config.get("target_crs"))
    ref_path: Optional[str] = None

    s1_scenes_cfg = resolve_auto_scene_configs(config.get("sentinel1", {}), sensor="sentinel1")
    s2_scenes_cfg = resolve_auto_scene_configs(config.get("sentinel2", {}), sensor="sentinel2")

    for item in s1_scenes_cfg or config.get("sentinel1", {}).get("scenes", []):
        if item.get("vv_path"):
            ref_path = str(item["vv_path"])
            break
        if item.get("stack_path"):
            ref_path = str(item["stack_path"])
            break
    if ref_path is None:
        for item in s2_scenes_cfg or config.get("sentinel2", {}).get("scenes", []):
            if item.get("safe_dir"):
                ref_path = resolve_s2_safe(item)["b02_path"]
                break
            if item.get("b02_path"):
                ref_path = str(item["b02_path"])
                break
            if item.get("stack_path"):
                ref_path = str(item["stack_path"])
                break
    if ref_path is None:
        raise ValueError("无法确定 target grid：没有可用的 S1/S2 参考影像")

    with rasterio.open(ref_path) as src:
        base_crs = parse_crs(src.crs)
        if base_crs is None:
            raise ValueError(f"参考影像缺少 CRS: {ref_path}")
        if target_crs is None:
            target_crs = base_crs
        if base_crs != target_crs:
            tfm_bounds = rasterio.warp.transform_bounds(base_crs, target_crs, *src.bounds, densify_pts=21)
        else:
            tfm_bounds = tuple(src.bounds)

    process_bounds = tfm_bounds
    aoi_shp = config.get("aoi_shp")
    aoi_geom = read_union_geometry(aoi_shp, target_crs) if aoi_shp else None
    if aoi_geom is not None:
        process_bounds = intersect_bounds(process_bounds, aoi_geom.bounds)
    process_bounds = snap_bounds(process_bounds, target_res)
    left, bottom, right, top = process_bounds
    width = int(round((right - left) / target_res))
    height = int(round((top - bottom) / target_res))
    transform = Affine(target_res, 0.0, left, 0.0, -target_res, top)
    return GridSpec(
        crs=target_crs,
        transform=transform,
        width=width,
        height=height,
        bounds=process_bounds,
        res=target_res,
        aoi_geom=aoi_geom,
    )


def open_s1_scenes(config: Dict[str, Any], grid: GridSpec) -> List[S1Scene]:
    scenes: List[S1Scene] = []
    scene_cfgs = resolve_auto_scene_configs(config.get("sentinel1", {}), sensor="sentinel1")
    for idx, item in enumerate(scene_cfgs or config.get("sentinel1", {}).get("scenes", []), start=1):
        vv_ref = _resolve_stack_or_single(item, "vv", "vv_band")
        vh_ref = _resolve_stack_or_single(item, "vh", "vh_band")
        vv = RasterReader(vv_ref, target_crs=grid.crs, target_transform=grid.transform, width=grid.width, height=grid.height)
        vh = RasterReader(vh_ref, target_crs=grid.crs, target_transform=grid.transform, width=grid.width, height=grid.height)
        scenes.append(
            S1Scene(
                date=parse_date(item.get("date")),
                vv=vv,
                vh=vh,
                in_db=bool(item.get("in_db", True)),
                name=str(item.get("name") or f"s1_{idx}"),
                params={},
            )
        )
    return scenes


def open_s2_scenes(config: Dict[str, Any], grid: GridSpec) -> List[S2Scene]:
    scenes: List[S2Scene] = []
    scene_cfgs = resolve_auto_scene_configs(config.get("sentinel2", {}), sensor="sentinel2")
    for idx, item in enumerate(scene_cfgs or config.get("sentinel2", {}).get("scenes", []), start=1):
        scene = dict(item)
        if scene.get("safe_dir"):
            scene.update(resolve_s2_safe(scene))

        def mk(key: str, resampling: str) -> RasterReader:
            if scene.get(f"{key}_path"):
                ref = RasterBandRef(path=str(scene[f"{key}_path"]), band=int(scene.get(f"{key}_band", 1)), resampling=resampling)
            elif scene.get("stack_path"):
                ref = RasterBandRef(path=str(scene["stack_path"]), band=int(scene.get(f"{key}_band", 1)), resampling=resampling)
            else:
                raise KeyError(f"场景 {scene.get('name', '')} 缺少 {key}_path 或 stack_path")
            return RasterReader(ref, target_crs=grid.crs, target_transform=grid.transform, width=grid.width, height=grid.height)

        scenes.append(
            S2Scene(
                date=parse_date(scene.get("date")),
                b02=mk("b02", "bilinear"),
                b03=mk("b03", "bilinear"),
                b04=mk("b04", "bilinear"),
                b08=mk("b08", "bilinear"),
                b11=mk("b11", "bilinear"),
                b12=mk("b12", "bilinear"),
                scl=mk("scl", "nearest"),
                name=str(scene.get("name") or f"s2_{idx}"),
            )
        )
    return scenes


def close_scenes(s1_scenes: Sequence[S1Scene], s2_scenes: Sequence[S2Scene], extras: Sequence[RasterReader] = ()) -> None:
    for sc in s1_scenes:
        sc.vv.close(); sc.vh.close()
    for sc in s2_scenes:
        sc.b02.close(); sc.b03.close(); sc.b04.close(); sc.b08.close(); sc.b11.close(); sc.b12.close(); sc.scl.close()
    for r in extras:
        r.close()


# -----------------------------------------------------------------------------
# DEM slope / prior
# -----------------------------------------------------------------------------


def compute_slope_deg(dem_arr: np.ndarray, res: float) -> np.ndarray:
    dem = dem_arr.astype("float32")
    dzdy, dzdx = np.gradient(dem, res, res)
    slope = np.degrees(np.arctan(np.sqrt(dzdx * dzdx + dzdy * dzdy))).astype("float32")
    slope[~np.isfinite(dem_arr)] = np.nan
    return slope


def open_optional_dem_reader(config: Dict[str, Any], grid: GridSpec) -> Optional[RasterReader]:
    dem_path = config.get("terrain_dem_tif") or config.get("dem_tif")
    if not dem_path:
        return None
    if not Path(str(dem_path)).exists():
        LOGGER.warning("DEM 不存在，水体识别阶段不使用地形约束: %s", dem_path)
        return None
    ref = RasterBandRef(path=str(dem_path), band=1, resampling="bilinear")
    return RasterReader(ref, target_crs=grid.crs, target_transform=grid.transform, width=grid.width, height=grid.height)


def build_prior_from_masks(
    mask_paths: Sequence[str],
    *,
    grid: GridSpec,
    out_prior_tif: Optional[str] = None,
    threshold: float = 0.1,
    block_size: int = 512,
) -> Tuple[Optional[str], Optional[np.ndarray], Dict[str, Any]]:
    if not mask_paths:
        return None, None, {"prior_source_count": 0}

    valid_paths = [str(p) for p in mask_paths if Path(str(p)).exists()]
    if not valid_paths:
        return None, None, {"prior_source_count": 0}

    readers = [RasterReader(RasterBandRef(path=p, band=1, resampling="nearest"), target_crs=grid.crs, target_transform=grid.transform, width=grid.width, height=grid.height) for p in valid_paths]
    try:
        if out_prior_tif is None:
            out_prior_tif = str(Path(tempfile.mkdtemp(prefix="prior_")) / "water_prior.tif")
        meta = {
            "driver": "GTiff",
            "width": grid.width,
            "height": grid.height,
            "count": 1,
            "dtype": "float32",
            "crs": grid.crs,
            "transform": grid.transform,
            "compress": "deflate",
            "tiled": True,
            "nodata": None,
        }
        ensure_dir(Path(out_prior_tif).parent)
        prior_sum = 0.0
        prior_count = 0
        with rasterio.open(out_prior_tif, "w", **meta) as dst:
            for win in iter_windows(grid.width, grid.height, block_size):
                acc = np.zeros((int(win.height), int(win.width)), dtype="float32")
                for rr in readers:
                    arr = rr.read(win, out_dtype="float32")
                    m = rr.mask(win)
                    acc += ((arr > 0) & m).astype("float32")
                prior = np.clip(acc / float(len(readers)), 0.02, 0.98).astype("float32")
                dst.write(prior, 1, window=win)
                prior_sum += float(prior.sum())
                prior_count += int(prior.size)
        return out_prior_tif, None, {
            "prior_source_count": len(readers),
            "prior_threshold_for_extent": threshold,
            "prior_mean": float(prior_sum / max(prior_count, 1)),
        }
    finally:
        for rr in readers:
            rr.close()


def build_prior_extent_gpkg(prior_tif: str, out_gpkg: str, threshold: float = 0.1) -> Optional[str]:
    ensure_dir(Path(out_gpkg).parent)
    if Path(out_gpkg).exists():
        Path(out_gpkg).unlink()
    schema = {"geometry": "Polygon", "properties": {"id": "int", "occ": "float:20.3"}}
    features: List[Tuple[BaseGeometry, float]] = []
    with rasterio.open(prior_tif) as src:
        arr = src.read(1)
        mask = (arr >= threshold).astype("uint8")
        for gm, value in shapes(mask, transform=src.transform):
            if int(value) != 1:
                continue
            geom = shape(gm)
            if geom.is_empty or geom.area < (src.res[0] * src.res[1] * 4):
                continue
            sub = rasterize([(mapping(geom), 1)], out_shape=(src.height, src.width), transform=src.transform, fill=0, dtype="uint8")
            occ = float(arr[sub == 1].mean()) if np.any(sub == 1) else float(threshold)
            features.append((geom, occ))
    if not features:
        return None
    with fiona.open(out_gpkg, "w", driver="GPKG", layer="prior_extent", crs=FionaCRS.from_user_input(grid.crs.to_string()) if False else None, schema=schema) as dst:
        pass
    # Fiona 对 crs 在这里需要重新打开；为避免复杂性，改写为 ESRI Shapefile / GPKG 时直接使用 raster CRS。
    with rasterio.open(prior_tif) as src:
        crs = FionaCRS.from_user_input(src.crs.to_string())
    with fiona.open(out_gpkg, "w", driver="GPKG", layer="prior_extent", crs=crs, schema=schema) as dst:
        for idx, (geom, occ) in enumerate(features, start=1):
            dst.write({"geometry": mapping(geom), "properties": {"id": idx, "occ": occ}})
    return out_gpkg


# -----------------------------------------------------------------------------
# SAR / Optical 特征与模型
# -----------------------------------------------------------------------------


def to_reflectance(arr: np.ndarray) -> np.ndarray:
    out = arr.astype("float32")
    finite = np.isfinite(out)
    if not np.any(finite):
        return out
    p99 = float(np.nanpercentile(out[finite], 99))
    if p99 > 2.0:
        out = out / 10000.0
    return out.astype("float32")


def compute_optical_features(b02: np.ndarray, b03: np.ndarray, b04: np.ndarray, b08: np.ndarray, b11: np.ndarray, b12: np.ndarray) -> Dict[str, np.ndarray]:
    blue = to_reflectance(b02)
    green = to_reflectance(b03)
    red = to_reflectance(b04)
    nir = to_reflectance(b08)
    swir1 = to_reflectance(b11)
    swir2 = to_reflectance(b12)
    ndwi = norm_index(green, nir)
    mndwi = norm_index(green, swir1)
    ndvi = norm_index(nir, red)
    awei_sh = blue + 2.5 * green - 1.5 * (nir + swir1) - 0.25 * swir2
    return {
        "blue": blue,
        "green": green,
        "red": red,
        "nir": nir,
        "swir1": swir1,
        "swir2": swir2,
        "ndwi": ndwi,
        "mndwi": mndwi,
        "ndvi": ndvi,
        "awei_sh": awei_sh.astype("float32"),
    }


def optical_base_probability(feats: Dict[str, np.ndarray]) -> np.ndarray:
    score = (
        2.3 * feats["mndwi"]
        + 0.8 * feats["ndwi"]
        + 0.30 * np.tanh(3.0 * feats["awei_sh"])
        - 1.0 * feats["ndvi"]
        - 0.6 * feats["nir"]
        - 0.4 * feats["swir1"]
    )
    return sigmoid(3.5 * score).astype("float32")


def build_clear_mask_from_scl(scl: np.ndarray, cloud_dilate_px: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    scl = scl.astype("int16")
    clear = np.isin(scl, [4, 5, 6])
    invalid = np.isin(scl, [0, 1, 2, 3, 7, 8, 9, 10, 11])
    if cloud_dilate_px > 0:
        invalid = ndimage.binary_dilation(invalid, structure=np.ones((2 * cloud_dilate_px + 1, 2 * cloud_dilate_px + 1), dtype=bool))
    clear &= ~invalid
    cloud_dist = ndimage.distance_transform_edt(clear).astype("float32")
    cloud_quality = np.clip(cloud_dist / max(cloud_dilate_px * 2, 1), 0.0, 1.0)
    return clear, cloud_quality


def estimate_s1_scene_params(scene: S1Scene, grid: GridSpec, dem_reader: Optional[RasterReader], sample_size: int = 1024) -> Dict[str, Any]:
    out_shape = (min(sample_size, grid.height), min(sample_size, grid.width))
    vv = scene.vv.read_coarse(out_shape=out_shape)
    vh = scene.vh.read_coarse(out_shape=out_shape)
    vv_mask = scene.vv.vrt.read_masks(scene.vv.band, out_shape=out_shape) > 0
    vh_mask = scene.vh.vrt.read_masks(scene.vh.band, out_shape=out_shape) > 0
    valid = vv_mask & vh_mask & np.isfinite(vv) & np.isfinite(vh)

    if not scene.in_db:
        vv = 10.0 * np.log10(np.maximum(vv, 1e-6)).astype("float32")
        vh = 10.0 * np.log10(np.maximum(vh, 1e-6)).astype("float32")

    if dem_reader is not None:
        dem = dem_reader.read_coarse(out_shape=out_shape)
        slope = compute_slope_deg(dem, grid.res * grid.width / out_shape[1])
        valid &= np.isfinite(slope)
        valid &= slope < 12.0

    vals_vv = vv[valid]
    vals_vh = vh[valid]
    if vals_vv.size == 0 or vals_vh.size == 0:
        raise RuntimeError(f"场景 {scene.name} 无有效像元，无法估计 SAR 参数")

    th_vv = otsu_threshold(vals_vv)
    th_vh = otsu_threshold(vals_vh)
    p05_vv, p40_vv = np.nanpercentile(vals_vv, [5, 40])
    p05_vh, p40_vh = np.nanpercentile(vals_vh, [5, 40])
    th_vv = float(np.clip(th_vv, p05_vv, p40_vv))
    th_vh = float(np.clip(th_vh, p05_vh, p40_vh))
    iqr_vv = float(np.nanpercentile(vals_vv, 75) - np.nanpercentile(vals_vv, 25))
    iqr_vh = float(np.nanpercentile(vals_vh, 75) - np.nanpercentile(vals_vh, 25))
    sigma_vv = float(np.clip(iqr_vv / 4.0, 1.2, 2.5))
    sigma_vh = float(np.clip(iqr_vh / 4.0, 1.2, 2.5))
    return {
        "th_vv": round(th_vv, 3),
        "th_vh": round(th_vh, 3),
        "sigma_vv": round(sigma_vv, 3),
        "sigma_vh": round(sigma_vh, 3),
    }


def sar_scene_probability(
    vv: np.ndarray,
    vh: np.ndarray,
    *,
    in_db: bool,
    th_vv: float,
    th_vh: float,
    sigma_vv: float,
    sigma_vh: float,
    slope: Optional[np.ndarray],
    texture_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    vv = vv.astype("float32")
    vh = vh.astype("float32")
    if not in_db:
        vv = 10.0 * np.log10(np.maximum(vv, 1e-6)).astype("float32")
        vh = 10.0 * np.log10(np.maximum(vh, 1e-6)).astype("float32")
    valid = np.isfinite(vv) & np.isfinite(vh)

    p_vv = sigmoid((th_vv - vv) / max(sigma_vv, 1.0)).astype("float32")
    p_vh = sigmoid((th_vh - vh) / max(sigma_vh, 1.0)).astype("float32")
    vv_std = local_std(vv, size=texture_size)
    p_tex = sigmoid((1.8 - vv_std) / 0.6).astype("float32")

    p = (0.50 * p_vv + 0.30 * p_vh + 0.20 * p_tex).astype("float32")
    if slope is not None:
        slope_penalty = np.clip(1.0 - np.maximum(slope - 5.0, 0.0) / 15.0, 0.15, 1.0).astype("float32")
        p *= slope_penalty
        valid &= np.isfinite(slope)
    p[~valid] = np.nan
    return p, valid


def aggregate_sar(
    scenes: Sequence[S1Scene],
    window: Window,
    *,
    target_date: Optional[datetime],
    tau_days: float,
    dem_reader: Optional[RasterReader],
    grid: GridSpec,
    texture_size: int,
) -> Dict[str, np.ndarray]:
    h = int(window.height)
    w = int(window.width)
    sum_p = np.zeros((h, w), dtype="float32")
    sum_w = np.zeros((h, w), dtype="float32")
    scene_count = np.zeros((h, w), dtype="uint8")

    slope = None
    if dem_reader is not None:
        dem = dem_reader.read(window)
        slope = compute_slope_deg(dem, grid.res)

    for sc in scenes:
        vv = sc.vv.read(window)
        vh = sc.vh.read(window)
        vv_m = sc.vv.mask(window)
        vh_m = sc.vh.mask(window)
        p, valid = sar_scene_probability(
            vv, vh,
            in_db=sc.in_db,
            th_vv=sc.params["th_vv"],
            th_vh=sc.params["th_vh"],
            sigma_vv=sc.params["sigma_vv"],
            sigma_vh=sc.params["sigma_vh"],
            slope=slope,
            texture_size=texture_size,
        )
        valid &= vv_m & vh_m
        age = date_diff_days(target_date, sc.date)
        w_time = math.exp(-(age / max(tau_days, EPS)))
        weight = np.where(valid, w_time, 0.0).astype("float32")
        p = np.where(valid, p, 0.0).astype("float32")
        sum_p += p * weight
        sum_w += weight
        scene_count += valid.astype("uint8")

    sar_valid = sum_w > 0
    sar_prob = np.full((h, w), np.nan, dtype="float32")
    sar_prob[sar_valid] = (sum_p[sar_valid] / np.maximum(sum_w[sar_valid], EPS)).astype("float32")
    sar_conf = (1.0 - np.exp(-sum_w)).astype("float32")
    return {
        "sar_prob": sar_prob,
        "sar_valid": sar_valid,
        "sar_conf": sar_conf,
        "scene_count": scene_count,
        "slope": slope,
    }


def aggregate_optical(
    scenes: Sequence[S2Scene],
    window: Window,
    *,
    target_date: Optional[datetime],
    tau_days: float,
    cloud_dilate_px: int,
) -> Dict[str, np.ndarray]:
    h = int(window.height)
    w = int(window.width)
    feat_sum: Dict[str, np.ndarray] = {k: np.zeros((h, w), dtype="float32") for k in ["blue", "green", "red", "nir", "swir1", "swir2", "ndwi", "mndwi", "ndvi", "awei_sh"]}
    w_sum = np.zeros((h, w), dtype="float32")
    clear_count = np.zeros((h, w), dtype="uint8")
    quality_best = np.zeros((h, w), dtype="float32")

    for sc in scenes:
        b02 = sc.b02.read(window)
        b03 = sc.b03.read(window)
        b04 = sc.b04.read(window)
        b08 = sc.b08.read(window)
        b11 = sc.b11.read(window)
        b12 = sc.b12.read(window)
        scl = sc.scl.read(window)
        valid = sc.b02.mask(window) & sc.b03.mask(window) & sc.b04.mask(window) & sc.b08.mask(window) & sc.b11.mask(window) & sc.b12.mask(window)
        clear, cloud_quality = build_clear_mask_from_scl(scl, cloud_dilate_px=cloud_dilate_px)
        clear &= valid
        feats = compute_optical_features(b02, b03, b04, b08, b11, b12)
        age = date_diff_days(target_date, sc.date)
        w_time = math.exp(-(age / max(tau_days, EPS)))
        q = np.where(clear, w_time * cloud_quality, 0.0).astype("float32")
        for k in feat_sum:
            feat_sum[k] += feats[k].astype("float32") * q
        w_sum += q
        clear_count += clear.astype("uint8")
        quality_best = np.maximum(quality_best, q)

    clear_any = w_sum > 0
    feat_mean = {k: np.full((h, w), np.nan, dtype="float32") for k in feat_sum}
    for k in feat_sum:
        feat_mean[k][clear_any] = feat_sum[k][clear_any] / np.maximum(w_sum[clear_any], EPS)
    opt_prob_base = np.full((h, w), np.nan, dtype="float32")
    if np.any(clear_any):
        opt_prob_base[clear_any] = optical_base_probability(feat_mean)[clear_any]
    return {
        "features": feat_mean,
        "opt_prob_base": opt_prob_base,
        "opt_clear": clear_any,
        "opt_conf": np.clip(quality_best, 0.0, 1.0).astype("float32"),
        "clear_count": clear_count,
    }


# -----------------------------------------------------------------------------
# overlap 训练：用重叠区的 SAR 伪标签校准 optical
# -----------------------------------------------------------------------------


def sample_training_from_overlap(
    config: Dict[str, Any],
    grid: GridSpec,
    s1_scenes: Sequence[S1Scene],
    s2_scenes: Sequence[S2Scene],
    dem_reader: Optional[RasterReader],
    prior_reader: Optional[RasterReader],
) -> Tuple[Optional[Pipeline], Dict[str, Any]]:
    if not s2_scenes:
        return None, {"status": "NO_S2_SCENES"}
    if not s1_scenes:
        return None, {"status": "NO_S1_SCENES"}

    fusion_cfg = config.get("fusion", {})
    target_date = parse_date(config.get("date"))
    block_size = int(fusion_cfg.get("block_size", 512))
    max_samples = int(fusion_cfg.get("max_training_samples", 200000))
    min_samples = int(fusion_cfg.get("min_training_samples", 5000))
    sar_tau = float(fusion_cfg.get("sar_tau_days", 8.0))
    opt_tau = float(fusion_cfg.get("optical_tau_days", 6.0))
    cloud_dilate_px = int(fusion_cfg.get("cloud_dilate_px", 2))
    texture_size = int(fusion_cfg.get("sar_texture_size", 5))
    high_thr = float(fusion_cfg.get("sar_high_conf_threshold", 0.80))
    low_thr = float(fusion_cfg.get("sar_low_conf_threshold", 0.20))

    X_chunks: List[np.ndarray] = []
    y_chunks: List[np.ndarray] = []
    n_pos = 0
    n_neg = 0
    rng = np.random.default_rng(int(fusion_cfg.get("random_seed", 42)))

    for win in iter_windows(grid.width, grid.height, block_size):
        sar = aggregate_sar(s1_scenes, win, target_date=target_date, tau_days=sar_tau, dem_reader=dem_reader, grid=grid, texture_size=texture_size)
        opt = aggregate_optical(s2_scenes, win, target_date=target_date, tau_days=opt_tau, cloud_dilate_px=cloud_dilate_px)
        prior = prior_reader.read(win) if prior_reader is not None else np.zeros((int(win.height), int(win.width)), dtype="float32")
        prior = np.where(np.isfinite(prior), prior, 0.05).astype("float32")
        slope = sar["slope"] if sar["slope"] is not None else np.zeros_like(prior, dtype="float32")
        candidate = sar["sar_valid"] & opt["opt_clear"] & np.isfinite(opt["opt_prob_base"])
        pos = candidate & (sar["sar_prob"] >= high_thr)
        neg = candidate & (sar["sar_prob"] <= low_thr)
        if not np.any(pos) and not np.any(neg):
            continue

        pos_idx = np.flatnonzero(pos.ravel())
        neg_idx = np.flatnonzero(neg.ravel())
        if pos_idx.size == 0 or neg_idx.size == 0:
            continue

        n_take = min(pos_idx.size, neg_idx.size, max(256, max_samples // 100))
        pos_pick = rng.choice(pos_idx, size=n_take, replace=False)
        neg_pick = rng.choice(neg_idx, size=n_take, replace=False)
        take = np.concatenate([pos_pick, neg_pick])
        rng.shuffle(take)

        block_feats = {
            **opt["features"],
            "prior": prior,
            "slope": np.where(np.isfinite(slope), slope, 0.0).astype("float32"),
        }
        X_block = np.column_stack([block_feats[k].ravel()[take] for k in MODEL_FEATURES]).astype("float32")
        y_block = np.concatenate([np.ones(n_take, dtype="uint8"), np.zeros(n_take, dtype="uint8")])
        y_block = y_block[np.argsort(np.concatenate([np.arange(n_take), np.arange(n_take)]))]  # placeholder, overwritten below
        # 重新生成对应标签，确保与 take 顺序一致
        y_all = np.zeros(pos.size, dtype="uint8")
        y_all[pos.ravel()] = 1
        y_block = y_all[take]

        X_chunks.append(X_block)
        y_chunks.append(y_block)
        n_pos += int(y_block.sum())
        n_neg += int((y_block == 0).sum())

        if (n_pos + n_neg) >= max_samples:
            break

    if not X_chunks:
        return None, {"status": "NO_OVERLAP_SAMPLES", "n_pos": 0, "n_neg": 0}

    X = np.vstack(X_chunks)
    y = np.concatenate(y_chunks)
    if X.shape[0] < min_samples or np.unique(y).size < 2:
        return None, {"status": "INSUFFICIENT_SAMPLES", "n_samples": int(X.shape[0]), "n_pos": int(y.sum()), "n_neg": int((y == 0).sum())}

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=400, class_weight="balanced", solver="lbfgs")),
    ])
    model.fit(X_train, y_train)
    p_val = model.predict_proba(X_val)[:, 1]
    y_hat = (p_val >= 0.5).astype("uint8")
    metrics = {
        "status": "OK",
        "n_samples": int(X.shape[0]),
        "n_train": int(X_train.shape[0]),
        "n_val": int(X_val.shape[0]),
        "n_pos": int(y.sum()),
        "n_neg": int((y == 0).sum()),
        "roc_auc": float(roc_auc_score(y_val, p_val)),
        "brier": float(brier_score_loss(y_val, p_val)),
        "balanced_accuracy": float(balanced_accuracy_score(y_val, y_hat)),
        "feature_order": list(MODEL_FEATURES),
    }
    return model, metrics


# -----------------------------------------------------------------------------
# 水体 prior / 融合输出
# -----------------------------------------------------------------------------


def fuse_probability(
    sar_prob: np.ndarray,
    sar_valid: np.ndarray,
    sar_conf: np.ndarray,
    opt_prob: np.ndarray,
    opt_clear: np.ndarray,
    opt_conf: np.ndarray,
    prior: np.ndarray,
    model_quality: float,
    cfg: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """按来源加权融合水体概率。

    关键改动：
    1) 降低 prior 的默认权重，避免无观测区域被先验“顶成水体”；
    2) 对 optical-only 区域下调权重，减少湿田/浅水田埂扩张；
    3) 输出 source_flag 与 conf，供后续按来源做不同阈值与桥接抑制。
    """
    wp = float(cfg.get("prior_weight", 0.04))
    opt_quality = float(np.clip(model_quality, 0.35, 0.95))

    source_flag = np.zeros(prior.shape, dtype="uint8")
    both = sar_valid & opt_clear
    sar_only = sar_valid & ~opt_clear
    opt_only = ~sar_valid & opt_clear
    prior_only = ~sar_valid & ~opt_clear
    source_flag[sar_only] = 1
    source_flag[opt_only] = 2
    source_flag[both] = 3
    source_flag[prior_only] = 4

    w_sar = np.where(sar_valid, np.clip(0.45 + 0.55 * sar_conf, 0.0, 1.0), 0.0).astype("float32")
    # optical-only 场景更易把湿农田拉成水面，基础权重显著低于 SAR。
    w_opt = np.where(opt_clear, np.clip(0.12 + 0.55 * opt_conf * opt_quality, 0.0, 0.75), 0.0).astype("float32")
    # prior 只做弱先验，不允许在无观测场景主导结果。
    w_prior = np.where(np.isfinite(prior), wp, 0.0).astype("float32")

    total = w_sar + w_opt + w_prior
    fused = np.full_like(prior, 0.05, dtype="float32")
    good = total > 0
    if np.any(good):
        lp = np.zeros_like(prior, dtype="float32")
        lp += np.where(sar_valid, w_sar * logit(np.where(np.isfinite(sar_prob), sar_prob, 0.5)), 0.0)
        lp += np.where(opt_clear, w_opt * logit(np.where(np.isfinite(opt_prob), opt_prob, 0.5)), 0.0)
        lp += np.where(np.isfinite(prior), w_prior * logit(np.clip(prior, 0.02, 0.98)), 0.0)
        fused[good] = sigmoid(lp[good] / np.maximum(total[good], EPS)).astype("float32")

    conf = np.zeros_like(prior, dtype="float32")
    conf[sar_only] = np.clip(0.55 + 0.35 * sar_conf[sar_only], 0.0, 1.0)
    conf[opt_only] = np.clip(0.25 + 0.35 * opt_conf[opt_only] * opt_quality, 0.0, 1.0)
    conf[both] = np.clip(0.68 + 0.24 * np.minimum(1.0, sar_conf[both] + opt_conf[both] * opt_quality), 0.0, 1.0)
    conf[prior_only] = np.clip(0.05 + 0.10 * np.where(np.isfinite(prior[prior_only]), prior[prior_only], 0.0), 0.0, 1.0)
    return fused.astype("float32"), source_flag, conf.astype("float32")


def _cross_connectivity() -> np.ndarray:
    return np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)


def break_narrow_bridges(
    cand: np.ndarray,
    seeds: np.ndarray,
    prob: np.ndarray,
    source_flag: np.ndarray,
    conf: np.ndarray,
    *,
    res_m: float,
    cfg: Dict[str, Any],
) -> np.ndarray:
    """删除低置信度的细窄桥接像元，避免多个塘坝被连成一片。"""
    if not np.any(cand):
        return cand
    bridge_width_m = float(cfg.get("bridge_break_width_m", 12.0))
    if bridge_width_m <= 0:
        return cand
    half_width_px = max(1.0, bridge_width_m / max(res_m * 2.0, 1e-6))
    dist = ndimage.distance_transform_edt(cand).astype("float32")
    narrow = cand & (~seeds) & (dist <= half_width_px)
    # 仅在低置信度或 optical-only 像元上切桥，保留真正高置信小塘坝核心。
    low_prob_thr = float(cfg.get("bridge_break_prob_threshold", 0.78))
    low_conf_thr = float(cfg.get("bridge_break_conf_threshold", 0.65))
    removable = narrow & (
        (prob < low_prob_thr)
        | (conf < low_conf_thr)
        | (source_flag == 2)
        | (source_flag == 4)
    )
    cand2 = cand & (~removable)
    if not np.any(cand2):
        return cand
    return cand2


def source_aware_hysteresis(
    prob: np.ndarray,
    source_flag: np.ndarray,
    conf: np.ndarray,
    prior: np.ndarray,
    *,
    res_m: float,
    cfg: Dict[str, Any],
) -> np.ndarray:
    """按来源分别阈值化，再做连通保留。

    目的：
    - SAR+Optical 重叠区可适度放宽；
    - optical-only 区域必须更严格，并优先要求先验支持；
    - prior-only 默认不直接生成观测水面。
    """
    high_default = float(cfg.get("hysteresis_high", 0.68))
    low_default = float(cfg.get("hysteresis_low", 0.50))
    high_both = float(cfg.get("hysteresis_high_both", high_default))
    low_both = float(cfg.get("hysteresis_low_both", low_default))
    high_sar = float(cfg.get("hysteresis_high_sar", max(high_both, 0.66)))
    low_sar = float(cfg.get("hysteresis_low_sar", max(low_both - 0.02, 0.35)))
    high_opt = float(cfg.get("hysteresis_high_opt", max(high_default + 0.10, 0.78)))
    low_opt = float(cfg.get("hysteresis_low_opt", max(low_default + 0.10, 0.62)))
    allow_prior_only = bool(cfg.get("allow_prior_only_water_mask", False))
    high_prior = float(cfg.get("hysteresis_high_prior", 0.90))
    low_prior = float(cfg.get("hysteresis_low_prior", 0.80))
    opt_prior_thr = float(cfg.get("opt_only_prior_threshold", 0.12))
    opt_force_prob = float(cfg.get("opt_only_force_probability", 0.93))
    opt_min_conf = float(cfg.get("opt_only_min_confidence", 0.55))

    prior_support = np.isfinite(prior) & (prior >= opt_prior_thr)

    seed_both = (source_flag == 3) & (prob >= high_both)
    seed_sar = (source_flag == 1) & (prob >= high_sar)
    seed_opt = (source_flag == 2) & (prob >= high_opt) & (conf >= opt_min_conf) & (prior_support | (prob >= opt_force_prob))
    seed_prior = allow_prior_only & (source_flag == 4) & (prob >= high_prior)
    seeds = seed_both | seed_sar | seed_opt | seed_prior

    cand_both = (source_flag == 3) & (prob >= low_both)
    cand_sar = (source_flag == 1) & (prob >= low_sar)
    cand_opt = (source_flag == 2) & (prob >= low_opt) & (conf >= opt_min_conf) & (prior_support | (prob >= opt_force_prob))
    cand_prior = allow_prior_only & (source_flag == 4) & (prob >= low_prior)
    cand = cand_both | cand_sar | cand_opt | cand_prior
    if not np.any(cand):
        return np.zeros_like(prob, dtype="uint8")

    cand = break_narrow_bridges(cand, seeds, prob, source_flag, conf, res_m=res_m, cfg=cfg)
    if not np.any(cand):
        return np.zeros_like(prob, dtype="uint8")

    labels, _ = ndimage.label(cand, structure=_cross_connectivity())
    keep = np.unique(labels[seeds & cand])
    keep = keep[keep > 0]
    if keep.size == 0:
        # 若没有种子，则仅保留极高概率连通块，避免整窗被低阈值拉满。
        strong = cand & (prob >= max(high_both, high_sar, high_opt))
        labels, _ = ndimage.label(strong, structure=_cross_connectivity())
        keep = np.unique(labels[strong])
        keep = keep[keep > 0]
        if keep.size == 0:
            return np.zeros_like(prob, dtype="uint8")
        out = np.isin(labels, keep)
        return out.astype("uint8")
    out = np.isin(labels, keep)
    return out.astype("uint8")


def refine_mask(mask: np.ndarray, close_px: int, open_px: int, fill_holes: bool) -> np.ndarray:
    arr = mask.astype(bool)
    if open_px > 0:
        arr = ndimage.binary_opening(arr, structure=np.ones((2 * open_px + 1, 2 * open_px + 1), dtype=bool))
    if close_px > 0:
        arr = ndimage.binary_closing(arr, structure=np.ones((2 * close_px + 1, 2 * close_px + 1), dtype=bool))
    if fill_holes:
        arr = ndimage.binary_fill_holes(arr)
    return arr.astype("uint8")


def write_raster(path: str, grid: GridSpec, dtype: str, nodata: Optional[float], count: int = 1) -> rasterio.io.DatasetWriter:
    ensure_dir(Path(path).parent)
    def _tile_size(n: int) -> int:
        if n >= 512:
            return 512
        # GTiff tile size 需要是 16 的倍数；小图用尽量接近边长的合法值。
        return max(16, int(max(1, n // 16) * 16))
    profile = {
        "driver": "GTiff",
        "width": grid.width,
        "height": grid.height,
        "count": count,
        "dtype": dtype,
        "crs": grid.crs,
        "transform": grid.transform,
        "compress": "deflate",
        "tiled": True,
        "blockxsize": _tile_size(grid.width),
        "blockysize": _tile_size(grid.height),
        "BIGTIFF": "IF_SAFER",
        "nodata": nodata,
    }
    return rasterio.open(path, "w", **profile)


def run_fuse_water(config: Dict[str, Any]) -> Dict[str, Any]:
    config = resolve_runtime_auto_inputs(config)
    grid = infer_grid(config)
    fusion_cfg = config.get("fusion", {})
    out_dir = ensure_dir(config.get("out_dir", "output"))
    date_tag = str(config.get("date") or "nodate")
    target_date = parse_date(config.get("date"))

    s1_scenes = open_s1_scenes(config, grid)
    s2_scenes = open_s2_scenes(config, grid)
    dem_reader = open_optional_dem_reader(config, grid)

    if not s1_scenes and not s2_scenes:
        raise ValueError("sentinel1.scenes / sentinel2.scenes 至少要有一类")

    prior_reader: Optional[RasterReader] = None
    prior_tif = config.get("prior", {}).get("prior_tif")
    prior_summary: Dict[str, Any] = {}
    if prior_tif and Path(str(prior_tif)).exists():
        prior_reader = RasterReader(RasterBandRef(path=str(prior_tif), band=1, resampling="bilinear"), target_crs=grid.crs, target_transform=grid.transform, width=grid.width, height=grid.height)
        prior_summary = {"prior_tif": str(prior_tif), "prior_source_count": 1}
    else:
        mask_paths = config.get("prior", {}).get("mask_paths", []) or []
        ref_mask = config.get("prior", {}).get("reference_water_tif")
        if ref_mask:
            mask_paths = list(mask_paths) + [str(ref_mask)]
        if mask_paths:
            built_prior_tif = str(out_dir / f"water_prior_{date_tag}.tif")
            prior_tif, _, prior_summary = build_prior_from_masks(mask_paths, grid=grid, out_prior_tif=built_prior_tif, threshold=float(config.get("prior", {}).get("extent_threshold", 0.1)))
            if prior_tif:
                prior_reader = RasterReader(RasterBandRef(path=str(prior_tif), band=1, resampling="bilinear"), target_crs=grid.crs, target_transform=grid.transform, width=grid.width, height=grid.height)

    for sc in s1_scenes:
        sc.params = estimate_s1_scene_params(sc, grid, dem_reader, sample_size=int(fusion_cfg.get("coarse_sample_size", 1024)))

    model, model_metrics = sample_training_from_overlap(config, grid, s1_scenes, s2_scenes, dem_reader, prior_reader)
    model_quality = float(model_metrics.get("roc_auc", 0.70)) if model is not None else float(fusion_cfg.get("fallback_model_quality", 0.55))

    prob_tif = str(out_dir / f"water_prob_{date_tag}.tif")
    mask_tif = str(out_dir / f"water_mask_{date_tag}.tif")
    source_tif = str(out_dir / f"source_flag_{date_tag}.tif")
    conf_tif = str(out_dir / f"confidence_{date_tag}.tif")

    high = float(fusion_cfg.get("hysteresis_high", 0.68))
    low = float(fusion_cfg.get("hysteresis_low", 0.50))
    # 10 m 小塘坝场景默认不做 closing，避免把 1 像元田埂直接跨过去。
    close_px = int(fusion_cfg.get("close_radius_px", 0))
    open_px = int(fusion_cfg.get("open_radius_px", 0))
    fill_holes = bool(fusion_cfg.get("fill_holes", False))
    block_size = int(fusion_cfg.get("block_size", 512))
    sar_tau = float(fusion_cfg.get("sar_tau_days", 8.0))
    opt_tau = float(fusion_cfg.get("optical_tau_days", 6.0))
    cloud_dilate_px = int(fusion_cfg.get("cloud_dilate_px", 2))
    texture_size = int(fusion_cfg.get("sar_texture_size", 5))

    summary = {
        "date": date_tag,
        "grid": {"width": grid.width, "height": grid.height, "crs": grid.crs.to_string(), "bounds": grid.bounds, "res": grid.res},
        "n_s1_scenes": len(s1_scenes),
        "n_s2_scenes": len(s2_scenes),
        "s1_scene_params": {sc.name: sc.params for sc in s1_scenes},
        "optical_calibration": model_metrics,
        "outputs": {"water_prob_tif": prob_tif, "water_mask_tif": mask_tif, "source_flag_tif": source_tif, "confidence_tif": conf_tif},
        **prior_summary,
    }

    area_pixel = grid.res * grid.res
    total_prob_area = 0.0
    total_mask_area = 0.0
    src_counts = {1: 0, 2: 0, 3: 0, 4: 0}

    with write_raster(prob_tif, grid, dtype="float32", nodata=np.nan) as dst_prob, \
         write_raster(mask_tif, grid, dtype="uint8", nodata=0) as dst_mask, \
         write_raster(source_tif, grid, dtype="uint8", nodata=0) as dst_source, \
         write_raster(conf_tif, grid, dtype="uint8", nodata=0) as dst_conf:
        for win in iter_windows(grid.width, grid.height, block_size):
            win_transform = window_transform(grid.transform, win)
            aoi_mask = rasterize_geom_mask(grid.aoi_geom, (int(win.height), int(win.width)), win_transform)
            sar = aggregate_sar(s1_scenes, win, target_date=target_date, tau_days=sar_tau, dem_reader=dem_reader, grid=grid, texture_size=texture_size) if s1_scenes else {
                "sar_prob": np.full((int(win.height), int(win.width)), np.nan, dtype="float32"),
                "sar_valid": np.zeros((int(win.height), int(win.width)), dtype=bool),
                "sar_conf": np.zeros((int(win.height), int(win.width)), dtype="float32"),
                "scene_count": np.zeros((int(win.height), int(win.width)), dtype="uint8"),
                "slope": compute_slope_deg(dem_reader.read(win), grid.res) if dem_reader is not None else None,
            }
            opt = aggregate_optical(s2_scenes, win, target_date=target_date, tau_days=opt_tau, cloud_dilate_px=cloud_dilate_px) if s2_scenes else {
                "features": {k: np.full((int(win.height), int(win.width)), np.nan, dtype="float32") for k in ["blue", "green", "red", "nir", "swir1", "swir2", "ndwi", "mndwi", "ndvi", "awei_sh"]},
                "opt_prob_base": np.full((int(win.height), int(win.width)), np.nan, dtype="float32"),
                "opt_clear": np.zeros((int(win.height), int(win.width)), dtype=bool),
                "opt_conf": np.zeros((int(win.height), int(win.width)), dtype="float32"),
                "clear_count": np.zeros((int(win.height), int(win.width)), dtype="uint8"),
            }
            prior = prior_reader.read(win) if prior_reader is not None else np.full((int(win.height), int(win.width)), 0.05, dtype="float32")
            prior = np.where(np.isfinite(prior), np.clip(prior, 0.02, 0.98), 0.05).astype("float32")

            if model is not None and np.any(opt["opt_clear"]):
                slope = sar["slope"] if sar["slope"] is not None else np.zeros_like(prior, dtype="float32")
                feats = {
                    **opt["features"],
                    "prior": prior,
                    "slope": np.where(np.isfinite(slope), slope, 0.0).astype("float32"),
                }
                good = opt["opt_clear"]
                X = np.column_stack([feats[k][good].ravel() for k in MODEL_FEATURES]).astype("float32")
                np.nan_to_num(X, copy=False)
                opt_prob = np.full_like(prior, np.nan, dtype="float32")
                model_p = model.predict_proba(X)[:, 1].astype("float32")
                base_p = np.clip(opt["opt_prob_base"][good], 0.0, 1.0).astype("float32")
                blend = float(fusion_cfg.get("optical_model_blend", np.clip(0.35 + 0.45 * max(model_quality - 0.60, 0.0) / 0.30, 0.35, 0.80)))
                opt_prob[good] = np.clip(blend * model_p + (1.0 - blend) * base_p, 0.0, 1.0).astype("float32")
            else:
                opt_prob = opt["opt_prob_base"]

            fused, source_flag, conf = fuse_probability(
                sar_prob=sar["sar_prob"],
                sar_valid=sar["sar_valid"],
                sar_conf=sar["sar_conf"],
                opt_prob=opt_prob,
                opt_clear=opt["opt_clear"],
                opt_conf=opt["opt_conf"],
                prior=prior,
                model_quality=model_quality,
                cfg=fusion_cfg,
            )
            fused = np.where(aoi_mask, fused, np.nan).astype("float32")
            source_flag = np.where(aoi_mask, source_flag, 0).astype("uint8")
            conf = np.where(aoi_mask, np.clip(conf * 100.0, 0.0, 100.0), 0).astype("uint8")

            mask_prob = np.nan_to_num(fused, nan=0.0)
            mask = source_aware_hysteresis(
                mask_prob,
                source_flag=source_flag,
                conf=np.clip(conf.astype("float32") / 100.0, 0.0, 1.0),
                prior=prior,
                res_m=grid.res,
                cfg=fusion_cfg,
            )
            mask = refine_mask(mask, close_px=close_px, open_px=open_px, fill_holes=fill_holes)
            mask = np.where(aoi_mask, mask, 0).astype("uint8")

            total_prob_area += float(np.nansum(fused)) * area_pixel
            total_mask_area += float(mask.sum()) * area_pixel
            for k in src_counts:
                src_counts[k] += int(np.sum(source_flag == k))

            dst_prob.write(fused, 1, window=win)
            dst_mask.write(mask, 1, window=win)
            dst_source.write(source_flag, 1, window=win)
            dst_conf.write(conf, 1, window=win)

    summary.update({
        "expected_water_area_m2": round(total_prob_area, 3),
        "binary_water_area_m2": round(total_mask_area, 3),
        "source_flag_pixels": src_counts,
    })

    summary_json = str(out_dir / f"fusion_summary_{date_tag}.json")
    dump_json(jsonable(summary), summary_json)
    summary["outputs"]["summary_json"] = summary_json

    close_scenes(s1_scenes, s2_scenes, extras=([prior_reader] if prior_reader is not None else []) + ([dem_reader] if dem_reader is not None else []))
    return summary


# -----------------------------------------------------------------------------
# build-prior 子命令
# -----------------------------------------------------------------------------


def cmd_build_prior(config: Dict[str, Any]) -> Dict[str, Any]:
    grid = infer_grid(config)
    out_dir = ensure_dir(config.get("out_dir", "output"))
    date_tag = str(config.get("date") or "nodate")
    mask_paths = config.get("prior", {}).get("mask_paths", []) or []
    ref_mask = config.get("prior", {}).get("reference_water_tif")
    if ref_mask:
        mask_paths = list(mask_paths) + [str(ref_mask)]
    if not mask_paths:
        raise ValueError("build-prior 需要 prior.mask_paths 或 prior.reference_water_tif")
    prior_tif = str(out_dir / f"water_prior_{date_tag}.tif")
    extent_tif, _, summary = build_prior_from_masks(mask_paths, grid=grid, out_prior_tif=prior_tif, threshold=float(config.get("prior", {}).get("extent_threshold", 0.1)))
    summary["outputs"] = {"prior_tif": extent_tif}
    dump_json(jsonable(summary), out_dir / f"prior_summary_{date_tag}.json")
    return summary


# -----------------------------------------------------------------------------
# 下游调用：继续使用 V1 pond_storage_monitor.py
# -----------------------------------------------------------------------------


def list_shapefiles(directory: Optional[str]) -> List[str]:
    if not directory:
        return []
    p = Path(directory)
    if not p.exists():
        return []
    return sorted(str(x) for x in p.rglob("*.shp"))


def choose_buffer(layer_name: str, rules: Dict[str, float], default_value: float) -> float:
    lname = layer_name.lower()
    for k, v in rules.items():
        if str(k).lower() in lname:
            return float(v)
    return float(default_value)


def write_vector_geoms(out_shp: str, geoms: Sequence[BaseGeometry], crs: FionaCRS) -> Optional[str]:
    if not geoms:
        return None
    out = Path(out_shp)
    ensure_dir(out.parent)
    for ext in [".shp", ".shx", ".dbf", ".prj", ".cpg"]:
        p = out.with_suffix(ext)
        if p.exists():
            p.unlink()
    schema = {"geometry": "Polygon", "properties": {"id": "int"}}
    with fiona.open(str(out), "w", driver="ESRI Shapefile", crs=crs, schema=schema, encoding="utf-8") as dst:
        for idx, geom in enumerate(geoms, start=1):
            dst.write({"geometry": mapping(geom), "properties": {"id": idx}})
    return str(out)


def prepare_buffered_river_dir(
    river_dir: str,
    *,
    out_dir: str,
    target_crs: CRS,
    buffer_rules: Dict[str, float],
    default_buffer_m: float,
    aoi_geom: Optional[BaseGeometry],
) -> str:
    out_dir = str(ensure_dir(out_dir))
    used_names: set = set()
    for shp in list_shapefiles(river_dir):
        name = Path(shp).stem
        buf = choose_buffer(name, buffer_rules, default_buffer_m)
        geoms: List[BaseGeometry] = []
        with fiona.open(shp) as src:
            src_crs = parse_crs(src.crs_wkt or src.crs)
            tfm = None
            if src_crs and src_crs != target_crs:
                tfm = Transformer.from_crs(src_crs, target_crs, always_xy=True)
            for feat in src:
                if not feat.get("geometry"):
                    continue
                g = shape(feat["geometry"])
                if tfm is not None:
                    g = shapely_transform(tfm.transform, g)
                if g.is_empty:
                    continue
                if g.geom_type in ("LineString", "MultiLineString", "LinearRing"):
                    g = g.buffer(buf, cap_style=2, join_style=2)
                elif g.geom_type in ("Point", "MultiPoint"):
                    g = g.buffer(buf)
                if aoi_geom is not None and not g.intersects(aoi_geom):
                    continue
                if aoi_geom is not None:
                    import shapely
                    aoi_buf = shapely.make_valid(aoi_geom).buffer(max(buf, 1.0))
                    g = shapely.make_valid(g).intersection(aoi_buf)
                if g.is_empty:
                    continue
                geoms.append(g)
        if geoms:
            safe_name = re.sub(r"[^0-9A-Za-z_]+", "_", name)[:40] or "layer"
            if safe_name in used_names:
                i = 1
                while f"{safe_name}_{i}" in used_names:
                    i += 1
                safe_name = f"{safe_name}_{i}"
            used_names.add(safe_name)
            write_vector_geoms(str(Path(out_dir) / f"{safe_name}.shp"), geoms, FionaCRS.from_user_input(target_crs.to_string()))
    return out_dir


def find_monitor_script() -> Path:
    here = Path(__file__).resolve().parent
    local = here / "pond_storage_monitor.py"
    if local.exists():
        return local
    raise FileNotFoundError("未找到 pond_storage_monitor.py")


def merge_pond_storage(final_dir: str | Path, date_tag: str) -> Optional[str]:
    """聚合 final_dir 下各流域子文件夹中的 pond_storage CSV 为长表。

    在最前面增加 river_basin 列标识流域名称，合并后保存为
    psh_pond_storage_{date_tag}.csv。
    """
    final_dir = Path(final_dir)
    if not final_dir.is_dir():
        return None

    basin_dirs = sorted(d for d in final_dir.iterdir() if d.is_dir())
    all_rows: List[Dict[str, str]] = []
    fieldnames: Optional[List[str]] = None

    for basin_dir in basin_dirs:
        basin = basin_dir.name
        csv_candidates = [
            basin_dir / f"pond_storage_{date_tag}.csv",
            basin_dir / "pond_storage.csv",
        ]
        csv_path = next((c for c in csv_candidates if c.exists()), None)
        if csv_path is None:
            continue
        with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if fieldnames is None:
                fieldnames = list(reader.fieldnames or [])
            for row in reader:
                merged_row = {"river_basin": basin}
                merged_row.update(row)
                all_rows.append(merged_row)

    if not all_rows or fieldnames is None:
        return None

    out_columns = ["river_basin"] + fieldnames
    out_path = final_dir / f"psh_pond_storage_{date_tag}.csv"
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=out_columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    LOGGER.info("合并塘坝蓄量 CSV：%d 行 → %s", len(all_rows), out_path)
    return str(out_path)


def run_pipeline(config: Dict[str, Any]) -> Dict[str, Any]:
    config = resolve_runtime_auto_inputs(config)
    summary = run_fuse_water(config)
    out_dir = ensure_dir(config.get("out_dir", "output"))
    date_tag = str(summary.get("date") or config.get("date") or "nodate")
    grid = infer_grid(config)
    aoi_geom = grid.aoi_geom

    downstream_cfg = dict(config.get("downstream", {}))
    downstream_cfg.setdefault("date", date_tag)
    downstream_cfg["water_tif"] = summary["outputs"]["water_mask_tif"]
    downstream_cfg["dem_tif"] = config.get("dem_tif")
    downstream_cfg["aoi_shp"] = config.get("aoi_shp")
    downstream_cfg.setdefault("out_dir", str(out_dir / "pond_storage"))

    river_dir = downstream_cfg.get("river_dir") or config.get("river_dir")
    reservoir_shp = downstream_cfg.get("reservoir_shp") or config.get("reservoir_shp")
    downstream_cfg["reservoir_shp"] = reservoir_shp

    buffer_rules = downstream_cfg.get("buffer_rules") or {}
    if river_dir and buffer_rules:
        buffered_dir = prepare_buffered_river_dir(
            str(river_dir),
            out_dir=str(out_dir / "buffered_river"),
            target_crs=grid.crs,
            buffer_rules={str(k): float(v) for k, v in buffer_rules.items()},
            default_buffer_m=float(downstream_cfg.get("line_buffer_m", 15.0)),
            aoi_geom=aoi_geom,
        )
        downstream_cfg["river_dir"] = buffered_dir
        downstream_cfg["line_buffer_m"] = 0.0
    else:
        downstream_cfg["river_dir"] = river_dir

    monitor_cfg_path = out_dir / f"monitor_cfg_{date_tag}.yaml"
    dump_yaml(downstream_cfg, monitor_cfg_path)

    monitor_script = find_monitor_script()
    cmd = [sys.executable, str(monitor_script), "--config", str(monitor_cfg_path)]
    LOGGER.info("调用下游 pond_storage_monitor.py: %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    summary["downstream"] = {
        "config": str(monitor_cfg_path),
        "returncode": int(proc.returncode),
        "stdout": proc.stdout[-4000:],
        "stderr": proc.stderr[-4000:],
    }
    dump_json(jsonable(summary), out_dir / f"pipeline_summary_{date_tag}.json")
    if proc.returncode != 0:
        raise RuntimeError(f"下游 pond_storage_monitor.py 运行失败\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    return summary


def merge_all_dates(output_root: str | Path) -> List[str]:
    """扫描 output_root 下所有 <YYYYMMDD>/final/ 子目录，逐期合并 pond_storage CSV。

    output_root 通常为项目根目录下的 output/。
    返回所有成功合并的文件路径列表。
    """
    output_root = Path(output_root)
    if not output_root.is_dir():
        return []

    merged_paths: List[str] = []
    # 扫描形如 20260311 / 20260421 的日期目录
    date_pattern = re.compile(r"^\d{8}$")
    for child in sorted(output_root.iterdir()):
        if not child.is_dir() or not date_pattern.match(child.name):
            continue
        final_dir = child / "final"
        if not final_dir.is_dir():
            continue
        date_tag = f"{child.name[:4]}-{child.name[4:6]}-{child.name[6:8]}"
        result = merge_pond_storage(final_dir, date_tag)
        if result:
            merged_paths.append(result)

    if merged_paths:
        LOGGER.info("全流域逐期合并完成，共 %d 期：%s", len(merged_paths), merged_paths)
    else:
        LOGGER.warning("未找到可合并的 pond_storage CSV")
    return merged_paths


def run_multi_pipeline(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """多期自动化 pipeline：扫描 S1/S2 目录中所有带日期影像对，逐期运行。"""
    configs = resolve_multi_date_configs(config)
    results: List[Dict[str, Any]] = []
    for i, cfg in enumerate(configs, 1):
        date_tag = cfg.get("date", "unknown")
        LOGGER.info("===== 多期运行 [%d/%d] 日期=%s =====", i, len(configs), date_tag)
        try:
            result = run_pipeline(cfg)
            results.append(result)
            LOGGER.info("===== 日期=%s 完成 =====", date_tag)
        except Exception as exc:
            LOGGER.error("日期=%s 运行失败: %s", date_tag, exc)
            results.append({"date": date_tag, "error": str(exc)})
    return results


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sentinel-1/2 融合生产级塘坝监测前端")
    sub = parser.add_subparsers(dest="command", required=True)

    for name in ["build-prior", "fuse-water", "pipeline"]:
        p = sub.add_parser(name)
        p.add_argument("--config", required=True, help="YAML 配置文件")
        p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    mp = sub.add_parser("multi-pipeline")
    mp.add_argument("--config", required=True, help="YAML 配置文件")
    mp.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    ms = sub.add_parser("merge-storage")
    ms.add_argument("--final-dir", required=True, help="包含流域子文件夹的 final 目录")
    ms.add_argument("--date", required=True, help="日期标签")
    ms.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    ma = sub.add_parser("merge-all-dates")
    ma.add_argument("--output-root", required=True, help="项目 output 根目录")
    ma.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    if args.command == "merge-storage":
        result = merge_pond_storage(args.final_dir, args.date)
        if result is None:
            LOGGER.error("合并失败：未找到有效的 pond_storage CSV")
            return 1
        print(json.dumps({"merged_csv": result}, ensure_ascii=False, indent=2))
        return 0

    if args.command == "merge-all-dates":
        results = merge_all_dates(args.output_root)
        if not results:
            LOGGER.error("未找到可合并的 pond_storage CSV")
            return 1
        print(json.dumps({"merged_csvs": results}, ensure_ascii=False, indent=2))
        return 0

    cfg = load_yaml(args.config)
    try:
        if args.command == "build-prior":
            result = cmd_build_prior(cfg)
        elif args.command == "fuse-water":
            result = run_fuse_water(cfg)
        elif args.command == "multi-pipeline":
            result = run_multi_pipeline(cfg)
        else:
            result = run_pipeline(cfg)
        print(json.dumps(jsonable(result), ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        LOGGER.exception("运行失败: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
