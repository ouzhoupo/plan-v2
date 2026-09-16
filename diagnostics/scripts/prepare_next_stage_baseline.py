from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATES = ("2024-05-21", "2024-06-01", "2025-05-21")
DATE_DIR = {date: date.replace("-", "") for date in DATES}
FULL_HASH_LIMIT = 64 * 1024 * 1024
SAMPLE_BYTES = 1024 * 1024


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def sha256_full(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_sampled(path: Path) -> str:
    size = path.stat().st_size
    offsets = sorted({0, max(0, size // 2 - SAMPLE_BYTES // 2), max(0, size - SAMPLE_BYTES)})
    digest = hashlib.sha256()
    digest.update(f"size={size};sample={SAMPLE_BYTES};offsets={offsets}".encode("ascii"))
    with path.open("rb") as stream:
        for offset in offsets:
            stream.seek(offset)
            digest.update(stream.read(SAMPLE_BYTES))
    return digest.hexdigest()


def fingerprint(path: Path) -> dict:
    record = {"path": rel(path), "exists": path.is_file()}
    if not path.is_file():
        return record
    stat = path.stat()
    record.update(
        {
            "size_bytes": stat.st_size,
            "mtime_local": datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(),
        }
    )
    if stat.st_size <= FULL_HASH_LIMIT:
        record["fingerprint_method"] = "sha256_full"
        record["sha256"] = sha256_full(path)
    else:
        record["fingerprint_method"] = "sha256_first_middle_last_1MiB"
        record["sha256_sampled"] = sha256_sampled(path)
    return record


def run_git(*args: str) -> dict:
    try:
        completed = subprocess.run(
            ["git", *args], cwd=ROOT, text=True, encoding="utf-8", errors="replace",
            capture_output=True, check=False,
        )
        return {
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
    except OSError as exc:
        return {"returncode": None, "stdout": "", "stderr": repr(exc)}


def package_versions() -> dict:
    names = ("numpy", "scipy", "scikit-learn", "rasterio", "geopandas", "shapely", "pandas", "pyyaml")
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sum_v75(path: Path) -> tuple[int, int, float]:
    object_rows = 0
    missing_v75 = 0
    total_v75 = 0.0
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            pond_id = (row.get("pond_id") or "").strip()
            if not pond_id:
                continue
            object_rows += 1
            raw = (row.get("v75_m3") or "").strip()
            if not raw:
                missing_v75 += 1
            else:
                total_v75 += float(raw)
    return object_rows, missing_v75, total_v75


def baseline_row(date: str) -> dict:
    folder = ROOT / "output" / DATE_DIR[date]
    fusion_path = folder / "pihe" / f"fusion_summary_{date}.json"
    pipeline_path = folder / "pihe" / f"pipeline_summary_{date}.json"
    final = folder / "final" / "pihe"
    deduct_path = final / f"deduct_summary_{date}.json"
    csv_path = final / f"pond_storage_{date}.csv"
    fusion = load_json(fusion_path)
    pipeline = load_json(pipeline_path)
    deduct = load_json(deduct_path)
    object_rows, missing_v75, v75_m3 = sum_v75(csv_path)
    calibration = pipeline.get("optical_calibration") or {}
    flags = fusion.get("source_flag_pixels") or {}
    return {
        "date": date,
        "optical_auc": calibration.get("roc_auc"),
        "optical_brier": calibration.get("brier"),
        "prior_source_count": fusion.get("prior_source_count"),
        "source_flag_1_pixels": flags.get("1"),
        "source_flag_2_pixels": flags.get("2"),
        "source_flag_3_pixels": flags.get("3"),
        "source_flag_4_pixels": flags.get("4"),
        "source_water_area_m2": deduct.get("source_water_area_m2"),
        "deducted_union_area_m2": deduct.get("deducted_union_area_m2"),
        "candidate_area_m2": deduct.get("pond_candidate_area_m2"),
        "final_area_m2": deduct.get("pond_area_after_min_area_m2"),
        "summary_final_objects": deduct.get("pond_count_after_min_area"),
        "csv_object_rows": object_rows,
        "csv_missing_v75": missing_v75,
        "v75_m3": v75_m3,
    }


def required_files() -> list[Path]:
    files = [
        ROOT / "code" / "s1s2_pond_production.py",
        ROOT / "code" / "pond_storage_monitor.py",
        ROOT / "code" / "requirements.txt",
        ROOT / "config" / "pipeline_auto" / "pihe_pipeline_auto.yaml",
        ROOT / "config" / "formal_total_tuned" / "pihe_total_tuned_formal.yaml",
        ROOT / "data" / "dem" / "dem.tif",
        ROOT / "data" / "landuse_water" / "PSH_土地利用水体.tif",
    ]
    for stem in (ROOT / "data" / "aoi" / "pihe", ROOT / "data" / "reservoir" / "psh_res"):
        files.extend(sorted(stem.parent.glob(stem.name + ".*")))
    for date in DATES:
        folder = ROOT / "output" / DATE_DIR[date]
        files.extend(
            [
                ROOT / "data" / "s1" / f"s1_stack_{date}.tif",
                ROOT / "data" / "s2" / f"s2_stack_{date}.tif",
                folder / "pihe" / f"water_prob_{date}.tif",
                folder / "pihe" / f"water_mask_{date}.tif",
                folder / "pihe" / f"source_flag_{date}.tif",
                folder / "pihe" / f"confidence_{date}.tif",
                folder / "pihe" / f"fusion_summary_{date}.json",
                folder / "pihe" / f"pipeline_summary_{date}.json",
                folder / "final" / "pihe" / f"pond_water_mask_{date}.tif",
                folder / "final" / "pihe" / f"pond_storage_{date}.csv",
                folder / "final" / "pihe" / f"pond_storage_{date}.gpkg",
                folder / "final" / "pihe" / f"deduct_summary_{date}.json",
                folder / "final" / "pihe" / f"resolved_monitor_cfg_{date}.yaml",
            ]
        )
    unique = {}
    for path in files:
        unique[str(path.resolve()).lower()] = path
    return sorted(unique.values(), key=lambda item: rel(item))


def main() -> None:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out = ROOT / "diagnostics" / "results" / "next_stage_baseline" / stamp
    out.mkdir(parents=True, exist_ok=False)

    git_head = run_git("rev-parse", "HEAD")
    git_branch = run_git("branch", "--show-current")
    git_status = run_git("status", "--porcelain=v1", "--untracked-files=normal")
    git_upstream = run_git("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}")
    git_remote = run_git("remote", "-v")
    manifest = {
        "status": "complete",
        "purpose": "Freeze the pre-A1 diagnostic baseline; this is not a backup and does not modify production outputs.",
        "created_local": datetime.now().astimezone().isoformat(),
        "root": str(ROOT),
        "fingerprint_policy": {
            "files_up_to_64MiB": "full SHA-256",
            "larger_files": "SHA-256 over size descriptor and first/middle/last 1 MiB",
            "limitation": "Sampled hashes detect ordinary changes but are not full-file integrity hashes.",
        },
        "environment": {
            "python_executable": sys.executable,
            "python_version": sys.version,
            "platform": platform.platform(),
            "packages": package_versions(),
        },
        "git": {
            "head": git_head,
            "branch": git_branch,
            "upstream": git_upstream,
            "remote": git_remote,
            "status": git_status,
            "clean": git_status.get("returncode") == 0 and not git_status.get("stdout"),
            "backup_verified": False,
            "backup_note": "A local manifest cannot prove that the current working tree is committed and pushed to GitHub.",
        },
        "dates": list(DATES),
        "files": [fingerprint(path) for path in required_files()],
    }
    missing = [item["path"] for item in manifest["files"] if not item["exists"]]
    manifest["missing_required_files"] = missing
    if missing:
        manifest["status"] = "incomplete"

    rows = [baseline_row(date) for date in DATES]
    manifest["baseline_metrics"] = rows
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out / "baseline_metrics.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    readme = (
        "NEXT-STAGE BASELINE MANIFEST\n"
        "This folder records the state before the A1 optical-only confidence-gate diagnostic.\n"
        "It does not change production code or outputs and is not a GitHub/data backup.\n"
        f"status={manifest['status']}\n"
        f"git_clean={manifest['git']['clean']}\n"
        f"missing_required_files={len(missing)}\n"
        "Review manifest.json and baseline_metrics.csv before preparing the A1 script.\n"
    )
    (out / "README.txt").write_text(readme, encoding="utf-8")
    print("RESULT_FOLDER:", out, flush=True)


if __name__ == "__main__":
    main()
