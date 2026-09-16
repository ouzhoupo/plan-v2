import json
import sys
import warnings
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
import s1s2_pond_production as m

warnings.filterwarnings("ignore", message="invalid value encountered in cast")

DATES = ("2024-06-01", "2025-05-21")
THRESHOLDS = (0.20, 0.50, 0.62, 0.68, 0.78, 0.80, 0.93)


def add_stats(acc, prefix, arr, valid, pixel_area):
    vals = arr[valid & np.isfinite(arr)]
    key = acc.setdefault(prefix, {"n": 0, "sum": 0.0, "sumsq": 0.0, "ge": {str(t): 0 for t in THRESHOLDS}})
    key["n"] += int(vals.size)
    if vals.size:
        key["sum"] += float(vals.sum(dtype=np.float64))
        key["sumsq"] += float(np.square(vals.astype(np.float64)).sum())
        for t in THRESHOLDS:
            key["ge"][str(t)] += int(np.count_nonzero(vals >= t))


def finish_stats(acc, pixel_area):
    out = {}
    for key, x in acc.items():
        n = x["n"]
        mean = x["sum"] / n if n else None
        var = max(x["sumsq"] / n - mean * mean, 0.0) if n else None
        out[key] = {
            "area_valid_km2": n * pixel_area / 1e6,
            "mean": mean,
            "sd": var ** 0.5 if var is not None else None,
            "area_ge_km2": {t: c * pixel_area / 1e6 for t, c in x["ge"].items()},
        }
    return out


def setup(date):
    cfg0 = m.load_yaml(ROOT / "config/pipeline_auto/pihe_pipeline_auto.yaml")
    configs = m.resolve_multi_date_configs(cfg0)
    cfg = next(x for x in configs if x["date"] == date)
    cfg = m.resolve_runtime_auto_inputs(cfg)
    grid = m.infer_grid(cfg)
    fcfg = cfg.get("fusion", {})
    s1 = m.open_s1_scenes(cfg, grid)
    s2 = m.open_s2_scenes(cfg, grid)
    dem = m.open_optional_dem_reader(cfg, grid)
    for sc in s1:
        sc.params = m.estimate_s1_scene_params(sc, grid, dem, sample_size=int(fcfg.get("coarse_sample_size", 1024)))
    model, metrics = m.sample_training_from_overlap(cfg, grid, s1, s2, dem, None)
    quality = float(metrics.get("roc_auc", 0.70)) if model is not None else float(fcfg.get("fallback_model_quality", 0.55))
    blend = float(fcfg.get("optical_model_blend", np.clip(0.35 + 0.45 * max(quality - 0.60, 0.0) / 0.30, 0.35, 0.80)))
    tag = date.replace("-", "")
    saved_dir = ROOT / "output" / tag / "pihe"
    readers = {
        "prob": rasterio.open(saved_dir / f"water_prob_{date}.tif"),
        "source": rasterio.open(saved_dir / f"source_flag_{date}.tif"),
        "conf": rasterio.open(saved_dir / f"confidence_{date}.tif"),
        "mask": rasterio.open(saved_dir / f"water_mask_{date}.tif"),
    }
    return {"date": date, "cfg": cfg, "fcfg": fcfg, "grid": grid, "s1": s1, "s2": s2, "dem": dem,
            "model": model, "metrics": metrics, "quality": quality, "blend": blend, "readers": readers}


def calc(ctx, win):
    g, f = ctx["grid"], ctx["fcfg"]
    target = m.parse_date(ctx["date"])
    sar = m.aggregate_sar(ctx["s1"], win, target_date=target, tau_days=float(f.get("sar_tau_days", 8.0)),
                          dem_reader=ctx["dem"], grid=g, texture_size=int(f.get("sar_texture_size", 5)))
    opt = m.aggregate_optical(ctx["s2"], win, target_date=target, tau_days=float(f.get("optical_tau_days", 6.0)),
                              cloud_dilate_px=int(f.get("cloud_dilate_px", 2)))
    prior = np.full((int(win.height), int(win.width)), 0.05, dtype="float32")
    model_p = np.full_like(prior, np.nan)
    opt_p = opt["opt_prob_base"].copy()
    if ctx["model"] is not None and np.any(opt["opt_clear"]):
        slope = sar["slope"] if sar["slope"] is not None else np.zeros_like(prior)
        feats = {**opt["features"], "prior": prior, "slope": np.where(np.isfinite(slope), slope, 0).astype("float32")}
        good = opt["opt_clear"]
        X = np.column_stack([feats[k][good].ravel() for k in m.MODEL_FEATURES]).astype("float32")
        np.nan_to_num(X, copy=False)
        model_p[good] = ctx["model"].predict_proba(X)[:, 1].astype("float32")
        opt_p[good] = np.clip(ctx["blend"] * model_p[good] + (1 - ctx["blend"]) * opt["opt_prob_base"][good], 0, 1)
    fused, source, conf = m.fuse_probability(
        sar["sar_prob"], sar["sar_valid"], sar["sar_conf"],
        opt_p, opt["opt_clear"], opt["opt_conf"],
        prior, ctx["quality"], f,
    )
    return sar, opt, model_p, opt_p, fused, source, conf, prior



