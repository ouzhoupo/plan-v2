@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

set REGIONS=hangbuhe pihe shihe
set FAILED=

for %%R in (%REGIONS%) do (
    echo.
    echo ===== [multi-pipeline] %%R =====
    python code\s1s2_pond_production.py multi-pipeline --config config\pipeline_auto\%%R_pipeline_auto.yaml
    if errorlevel 1 (
        echo [WARN] %%R pipeline failed
        set FAILED=!FAILED! %%R
    ) else (
        echo [OK] %%R pipeline finished
    )
)

echo.
echo ===== Merging pond_storage CSV per date =====
python code\s1s2_pond_production.py merge-all-dates --output-root output

echo.
if defined FAILED (
    echo ===== Done. Failed regions:!FAILED! =====
    exit /b 1
) else (
    echo ===== Done. All regions succeeded =====
    exit /b 0
)
