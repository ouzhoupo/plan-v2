# 淠河分阶段 GIS 抽样检查

该脚本读取现有输出，不重跑模型，也不改写正式结果。默认比较：

- 2024-06-01：人工值较高、模型异常偏低；
- 2025-05-21：人工值较低、模型偏高。

## 运行方法

在项目根目录的 PowerShell 中，用已经安装 `numpy`、`rasterio`、`fiona`、`shapely`、`pyproj` 的 Python 环境运行：

```powershell
python diagnostics\scripts\build_pihe_gis_samples.py
```

默认每类抽取 15 个面积较大的或接近面积门槛的对象。需要减少输出对象数量时可以改成 5 个；连通斑块仍会完整统计，所以总运行时间不会按比例下降：

```powershell
python diagnostics\scripts\build_pihe_gis_samples.py --sample-count 5
```

结果写入 `diagnostics/results/pihe_gis_samples`。其中：

- `water_mask_change_20240601_20250521.tif`：1 为仅 2024-06-01 有水，2 为两期均有水，3 为仅 2025-05-21 有水；
- `water_removed_before_candidate_*.tif`：上游 `water_mask` 中存在、候选 `pond_water_mask` 中消失的像元；
- `pihe_stage_samples_*.gpkg`：供 GIS 定位核查的代表对象；
- `pihe_stage_samples_*.csv`：同一批对象的概率、置信度、来源、跨期持续性和土地利用证据；
- `summary.json`：各阶段对象计数和输出位置。

## 抽样类别

- `removed_before_candidate`：在水体掩膜到候选掩膜之间消失的较大片段；
- `too_small_near_gate`：低于 500 平方米且最接近下限的对象；
- `too_large_near_gate`：高于 500000 平方米且最接近上限的对象；
- `landuse_rejected_large`：通过面积门槛、随后被土地利用弱筛选删除的较大对象；
- `final_kept_large`：最终保留的较大对象；
- `final_bbox_only`：对象本身没有与土地利用水体像元重叠，但因外接矩形附近存在水体而被保留。

## GIS 检查顺序

1. 先加载两期 `water_mask`、候选 `pond_water_mask` 和变化栅格，判断增减是否集中在影像覆盖边缘、河渠或农田。
2. 加载两期样本 GPKG，按 `category` 分类显示。
3. 叠加 Sentinel-2 真彩色、土地利用水体、河渠缓冲区和水库图层。
4. 对每类至少检查 5 个对象，记录“真实塘坝、河渠、农田积水、阴影/建筑、无法判断”。
5. 优先核查 `landuse_rejected_large` 与 `final_bbox_only`，它们直接检验当前土地利用弱筛选是否删错或留错。

字段 `other_wtr` 表示该位置在另一个日期被识别为水的像元比例。反复出现可作为稳定水体证据，但不能单独作为人工真值。`prob_mean` 与 `conf_mean` 分别是对象范围内的融合概率和置信度均值，均按0—1记录。
