# Porto 数据集制备与使用文档

> 状态：✅ 已完成（2026-07-09 重构为"主要车行道"方案 + 黑边对齐）
> 适用：sam_road / P2CNet / DelvMap 三个框架

Porto 数据集基于 [Porto Taxi Trajectory](https://www.kaggle.com/c/pkdd-15-predict-taxi-service-trajectory-i) 的 2013-07-01~2014-06-30 出租车轨迹 + 2014 年 Esri Wayback 卫星影像 + OSM 路网，按 sam_road 的 didi/xian 范式制备为 400×400 tile 数据集。

---

## 一、数据集组成

### 1.1 地理范围与网格

| 项 | 值 |
|---|---|
| bbox | lat [41.1125, 41.2385]，lon [-8.6875, -8.5495]（轨迹 1%-99% 分位，宽范围） |
| 物理范围 | 14.0 km (N-S) × 11.5 km (E-W)，约 161 km² |
| tile 网格 | 35 × 29 = **1015 块**，size=400m，NW-first 编号（左上→右下，行优先） |
| 数据划分 | train 812 / val 101 / test 102（seed=42，8:1:1） |

### 1.2 大图（rawdata，Web Mercator EPSG:3857）

所有大图同尺寸同投影（15603×12862，~1.194 m/px），逐像素对齐：

| 文件 | 内容 | 来源 |
|---|---|---|
| `porto_2014_r3026_z17.png` | 卫星影像 sat | Esri Wayback release 3026（2014-07-02 vintage），z17 瓦片拼接切边 |
| `porto_2014_r3026_z18.png` | 卫星影像高清版（备用） | 同上 z18 |
| `road_heat.png` | 路网大图（basemap + map_label 同源） | OSM PBF，只画主要车行道 |
| `building_heat.png` | 建筑大图 | OSM PBF，building way fillPoly |
| `traj_point_heat.png` | 轨迹点图（原始单像素） | train.csv GPS 点栅格化 |
| `traj_point_heat_closed.png` | 轨迹点图（3×3 闭运算，消横纹） | 同上闭运算 |
| `traj_line_heat.png` | 轨迹段图（相邻点连线） | train.csv GPS 连线 |
| `porto_1km.osm.pbf` | OSM PBF（裁剪外扩 1km） | portugal-150101.osm.pbf 裁剪 |
| `portugal-150101.osm.pbf` | 葡萄牙全域 PBF（归档） | Geofabrik/OSM |
| `train.csv` | 原始出租车轨迹 | Kaggle Porto Taxi |

### 1.3 tile 产物（datasets/porto/2014_400/，每块 400×400）

| 文件 | 内容 | 生成脚本 |
|---|---|---|
| `region_{c}_sat.png` | 卫星影像 | download_use_osm.py |
| `region_{c}_gt.png` | GT 路网 mask（二值 {0,255}，线宽2） | download_use_osm.py + regenerate_labels_after_clip.py |
| `region_{c}_graph_gt.pickle` | GT 图（未精修，评测用） | download_use_osm.py |
| `region_{c}_refine_gt_graph.p` | GT 图（精修后，训练用） | download_use_osm.py |
| `region_{c}_refine_gt_graph_samplepoints.json` | 采样点 | download_use_osm.py |
| `region_{c}_traj.png` | 轨迹点（closed 版，completion 输入） | generate_traj.py |
| `region_{c}_traj_line.png` | 轨迹段（连线版） | generate_traj.py --suffix _line |
| `region_{c}_refine_gt_graph_partial.p/.png` | edge_random partial（keep_ratio=0.5） | generate_partial_prior.py |
| `partial_component/` | component partial keep_ratio=0.50 | generate_partial_prior.py |
| `partial_component_25/` | component partial keep_ratio=0.25 | 同上 |
| `partial_component_75/` | component partial keep_ratio=0.75 | 同上 |

`processed/`（与 2014_400 同级）：
- `keypoint_mask_{c}.png` / `road_mask_{c}.png` —— 从 refine_gt_graph 生成，训练标签

`data_split.json`（与 2014_400 同级）：train/validation/test 划分

---

## 二、数据获取与处理方案

### 2.1 坐标系约定

- **大图（sat/traj/road/building）**：Web Mercator EPSG:3857，与 DelvMap/ESRI 瓦片同投影。像素映射：
  ```
  px = (x_m - x_min) / (x_max - x_min) * img_w
  py = (y_max - y_m) / (y_max - y_min) * img_h   # 北在顶，y 向下
  ```
- **tile 内 pickle**：左上原点 y-down，(row, col) 图像坐标系，`coord_transform = v[:,::-1]`（swap，同 cityscale/didi_xian）。

### 2.2 路网过滤方案（关键，2026-07-09 重构，依据 docs/chatgpt-osm类别分析.md）

**问题**：原始 OSM PBF 含大量非路 way（building/landuse/leisure/amenity/barrier 闭合多边形）+ 纯行人路（pedestrian/footway/steps/path）+ services 服务区闭合 + construction/proposed 等。若不过滤，GT pickle/gt.png 会出现奇怪闭合矩形 + 密集区堆叠。

**方案**：`download_use_osm.py --keep-highway-only`，保留**所有汽车可通行**的 highway（依据 OSM 类别语义，见 chatgpt-osm类别分析.md 的"汽车可通行"判定）：
```python
_KEEP_HIGHWAY = {
    # 主干道 + 匝道
    'motorway', 'motorway_link', 'trunk', 'trunk_link',
    'primary', 'primary_link', 'secondary', 'secondary_link',
    'tertiary', 'tertiary_link',
    # 居民区/低等级公共道路
    'residential', 'unclassified', 'road', 'living_street',
    # 内部/服务道路 (停车场/小区/加油站内部, 汽车可通行)
    'service',
    # 农用/林区土路 (越野车)
    'track',
}
```
排除：纯行人路（pedestrian/footway/steps/path）+ services（服务区闭合多边形）+ construction/proposed/bus_stop/platform/elevator/raceway + 非路 way（building/landuse/leisure/amenity/barrier/natural 等）。

白名单收 15672 条 highway（占 PBF 总 19129 的 81.9%）。闭合环路（residential/service 偶有）按 line 画，不填充。

**DelvMap 同源**：DelvMap road 大图（`generate_porto_delvmap_bigimgs.py`）用**同一白名单**，保证 DelvMap 大图与 samroad 数据集 GT 完全同源（仅渲染方式不同：DelvMap 直接画 OSM way 线宽2，samroad 走 graph 精修后画）。

**注意**：xian 数据集**不用**此过滤（保持兼容，xian 的 OSMHandler 默认 `keep_highway_only=False`）。Porto 显式开启。

### 2.3 黑边对齐方案（关键）

**问题**：边缘 tile 的 sat 在超出 bbox 的方向有黑边（外溢像素置 0），但 GT 路网若画到黑边区会导致 sat/traj 与 gt 不对齐，污染 mask loss/评测。

**方案**（与 xian 一致，事后裁剪）：
1. `download_use_osm.py` 的 `get_local_sat` 把边缘 tile 外溢像素置黑；GT 路网用 `clip_bbox` 裁到 bbox 内。
2. 但 `graphVis2048Segmentation` 画 gt.png 时用原始 bbox，bbox 外的路仍画进黑边区 → 需事后裁剪。
3. `tools/clip_rn_to_sat.py`：检测 sat 有效区（非黑边），用 Cohen-Sutherland 把 refine_gt_graph.p + graph_gt.pickle 裁到有效区，裁断处补 node（无悬空边）。
4. `tools/regenerate_labels_after_clip.py`：根据裁剪后的 graph 重生成 gt.png。
5. 重跑 generate_labels.py（mask）+ generate_partial_prior.py（partial）。

Porto 有 **34 个**边缘 tile 需黑边裁剪（右边缘 47px 黑边）。裁剪后黑边区路像素 ≈ 0（仅线宽溢出 3-8px，可接受）。

### 2.4 轨迹栅格化方案

- **投影**：Web Mercator（与 sat 同），逐像素对齐。
- **画法**：点图 = GPS 点单像素 255；线图 = 相邻点连线（距离 5-300m 过滤飞点/静止）255。纯二值 {0,255}。
- **GPS 量化横纹**：Porto GPS 纬度被量化到 0.0001°（~11m），在 1.194 m/px 下每 ~9px 出现空行。`traj_point_heat_closed.png` 用 3×3 闭运算填补横纹；`traj_point_heat.png` 保留原始。

### 2.5 完整制备命令序列

```bash
cd /Users/highee/research/sam_road
PY=/Users/highee/miniconda3/envs/samroad/bin/python

# 0. PBF 裁剪（外扩 1km，84M -> 2.3M）
$PY tools/prepare_dataset/clip_pbf_bbox.py \
  --in rawdata/porto/portugal-150101.osm.pbf \
  --out rawdata/porto/porto_1km.osm.pbf \
  --lat-min 41.1035 --lat-max 41.2475 \
  --lon-min -8.699457 --lon-max -8.537543

# 1. sat 大图（Wayback release 3026, z17，切边不 resize -> 15603x12862）
#    见 tools/prepare_dataset/porto_rsimg_test.py --full-bbox --release 3026 --zoom 17

# 2. building 大图
$PY tools/prepare_dataset/generate_porto_building.py \
  --pbf rawdata/porto/porto_1km.osm.pbf \
  --sat rawdata/porto/porto_2014_r3026_z17.png \
  --lat-min 41.1125 --lat-max 41.2385 \
  --lon-min -8.6875 --lon-max -8.5495 \
  --out rawdata/porto/building_heat.png

# 3. road 大图（DelvMap basemap/map_label，主要车行道）
$PY tools/prepare_dataset/generate_porto_delvmap_bigimgs.py \
  --pbf rawdata/porto/porto_1km.osm.pbf \
  --sat rawdata/porto/porto_2014_r3026_z17.png \
  --out rawdata/porto/road_heat.png

# 4. 轨迹大图（点/线/closed）
$PY tools/prepare_dataset/generate_porto_traj.py \
  --csv rawdata/porto/train.csv \
  --sat rawdata/porto/porto_2014_r3026_z17.png \
  --lat-min 41.1125 --lat-max 41.2385 \
  --lon-min -8.6875 --lon-max -8.5495 \
  --out-dir rawdata/porto

# 5. tile 全套（sat/gt/pickle，从 2014_400/ 运行，--keep-highway-only）
cd datasets/porto/2014_400
PYTHONPATH=/Users/highee/research/sam_road/tools/prepare_dataset $PY \
  /Users/highee/research/sam_road/tools/prepare_dataset/download_use_osm.py \
  --dataset_type original --keep-highway-only \
  --osm_pbf /Users/highee/research/sam_road/rawdata/porto/porto_1km.osm.pbf \
  --sat_source local:/Users/highee/research/sam_road/rawdata/porto/porto_2014_r3026_z17.png \
  --sat_local_extent 41.1125,41.2385,-8.6875,-8.5495,12862,15603 \
  /Users/highee/research/sam_road/tools/prepare_dataset/config/porto.json
cd /Users/highee/research/sam_road

# 6. 黑边裁剪 + 重生成 gt.png
$PY tools/clip_rn_to_sat.py --dataset didi --input_dir datasets/porto/2014_400
$PY tools/regenerate_labels_after_clip.py --input_dir datasets/porto/2014_400 --dataset didi

# 7. road/keypoint mask
cd datasets/porto/2014_400
$PY /Users/highee/research/sam_road/datasets/didi/xian/generate_labels.py \
  --root . --split /Users/highee/research/sam_road/datasets/porto/data_split.json
cd /Users/highee/research/sam_road

# 8. 轨迹 tile（点 closed 版 -> region_{c}_traj.png；线版 -> region_{c}_traj_line.png）
$PY tools/prepare_dataset/generate_traj.py \
  --config tools/prepare_dataset/config/porto.json \
  --traj-png rawdata/porto/traj_point_heat_closed.png \
  --out-dir datasets/porto/2014_400 --mode point \
  --delvmap-lat-min 41.1125 --delvmap-lat-max 41.2385 \
  --delvmap-lon-min -8.6875 --delvmap-lon-max -8.5495 \
  --delvmap-img-w 12862 --delvmap-img-h 15603
$PY tools/prepare_dataset/generate_traj.py \
  --config tools/prepare_dataset/config/porto.json \
  --traj-png rawdata/porto/traj_line_heat.png \
  --out-dir datasets/porto/2014_400 --mode point --suffix _line \
  --delvmap-lat-min 41.1125 --delvmap-lat-max 41.2385 \
  --delvmap-lon-min -8.6875 --delvmap-lon-max -8.5495 \
  --delvmap-img-w 12862 --delvmap-img-h 15603

# 9. partial（edge_random completion用 + component 50/25/75 p2cnet用）
$PY data/generate_partial_prior.py --dataset didi --input_dir datasets/porto/2014_400 \
  --output_dir datasets/porto/2014_400 --keep_ratio 0.5 --seed 42 --strategy edge_random
$PY data/generate_partial_prior.py --dataset didi --input_dir datasets/porto/2014_400 \
  --output_dir datasets/porto/2014_400/partial_component --keep_ratio 0.5 --seed 42 --strategy component
$PY data/generate_partial_prior.py --dataset didi --input_dir datasets/porto/2014_400 \
  --output_dir datasets/porto/2014_400/partial_component_25 --keep_ratio 0.25 --seed 42 --strategy component
$PY data/generate_partial_prior.py --dataset didi --input_dir datasets/porto/2014_400 \
  --output_dir datasets/porto/2014_400/partial_component_75 --keep_ratio 0.75 --seed 42 --strategy component

# 10. data_split.json
$PY -c "
import sys; sys.path.insert(0,'data')
from img_folder_to_json_list import generate_data_split_from_cases
generate_data_split_from_cases(data_root='datasets/porto/2014_400/',
  save_path='datasets/porto/data_split.json',
  train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, case_regex=r'region_(\d+)', seed=42)
"
```

---

## 三、存储路径

```
sam_road/
├── rawdata/porto/                    ← 原始素材 + 大图（.gitignore 整个 rawdata/）
│   ├── portugal-150101.osm.pbf       84M  全域 PBF（归档）
│   ├── porto_1km.osm.pbf             2.3M 裁剪外扩1km（制备用）
│   ├── porto_2014_r3026_z17.png      362M sat 大图（Wayback 2014）
│   ├── porto_2014_r3026_z18.png      1.4G sat 高清（备用）
│   ├── road_heat.png                 3.5M 路网大图（DelvMap basemap/label）
│   ├── building_heat.png             1.2M 建筑大图
│   ├── traj_point_heat.png           6.0M 轨迹点（原始）
│   ├── traj_point_heat_closed.png    5.0M 轨迹点（闭运算消横纹）
│   ├── traj_line_heat.png            9.3M 轨迹段
│   ├── train.csv                     1.8G 原始轨迹
│   └── porto_bbox.json               bbox/尺寸/投影元数据
│
├── datasets/porto/                   ← 制备产物（.gitignore，保留 .gitkeep）
│   ├── 2014_400/                     region_{c}_* tile（1015 块）
│   │   ├── region_{c}_sat.png
│   │   ├── region_{c}_gt.png
│   │   ├── region_{c}_graph_gt.pickle        评测 GT
│   │   ├── region_{c}_refine_gt_graph.p      训练 GT
│   │   ├── region_{c}_refine_gt_graph_samplepoints.json
│   │   ├── region_{c}_traj.png               轨迹点（completion 输入）
│   │   ├── region_{c}_traj_line.png          轨迹段
│   │   ├── region_{c}_refine_gt_graph_partial.p/.png  edge_random 0.5
│   │   ├── partial_component/                component 0.50
│   │   ├── partial_component_25/             component 0.25
│   │   └── partial_component_75/             component 0.75
│   ├── processed/                    keypoint_mask_{c}.png + road_mask_{c}.png
│   └── data_split.json               train 812 / val 101 / test 102
│
└── tools/prepare_dataset/config/porto.json   sam_road 网格配置
```

P2CNet 软链：`P2CNet/data/didi/porto` → `sam_road/datasets/porto`

---

## 四、各模型需要的输入

### 4.1 sam_road（extraction / 4ch / completion）

配置：`config/toponet_vitb_256_porto{,_4ch,_completion}.yaml`，`DATASET: 'porto'`。dataloader 在 `data/dataset.py` / `dataset_4ch.py` / `dataset_completion.py` 的 `porto` 分支 + `data/dataset_registry.py`。

| 模型 | 输入文件 | 说明 |
|---|---|---|
| **extraction**（dataset.py） | `region_{c}_sat.png` + `processed/keypoint_mask_{c}.png` + `road_mask_{c}.png` + `region_{c}_refine_gt_graph.p` | 纯视觉路网提取，3 通道 sat |
| **4ch**（dataset_4ch.py） | 同上 + 第4通道先验：训练用 `region_{c}_traj.png`，推理用 `region_{c}_refine_gt_graph_partial.png`（edge_random 0.5） | RGB+先验 4 通道。Porto 无 active.png（xian 的 active 是错的，已忽略），用 traj 作先验 |
| **completion**（dataset_completion.py） | `region_{c}_sat.png` + masks + `refine_gt_graph.p` + `region_{c}_traj.png`（USE_TRAJ=True 时） | 路网补全，动态 keep_ratio U[0.2,0.8] 删边 |

坐标系：`coord_transform = v[:,::-1]`（swap，左上原点），IMAGE_SIZE=400，SAMPLE_MARGIN=0。

### 4.2 P2CNet

配置：`P2CNet/configs/deeplabv3plus_mix_mp_sat_gsam_porto_config.json`，`type: "PortoDataLoader"`。独立 dataloader `P2CNet/data_loader/porto_data_loader.py`（`PortoDataset`/`PortoDataLoader`，不与 didi 混用）。训练/测试脚本 `train_porto.py` / `test_porto.py`。

| 输入 | 文件 |
|---|---|
| sat | `2014_400/region_{id}_sat.png` |
| GT complete road | `2014_400/region_{id}_gt.png` |
| partial（训练 mix） | `partial_component{,_25,_75}/region_{id}_refine_gt_graph_partial.png`（25/50/75 随机） |
| partial（验证/测试） | `partial_component/...`（固定 0.5） |

数据布局：`data/didi/porto/`（软链到 sam_road datasets/porto）+ `train/valid/test.txt` + `mix_info.json` + `data_split.json`。坐标系同 sam_road（didi，swap）。patch 400×400。

### 4.3 DelvMap

DelvMap 是 **256×256 patch 范式**（不同于 samroad/P2CNet 的 400 tile），用 `dataset/create_dataset.py` 从大图切 patch。dataloader `DelvMap/Dual_Signal_Fusion_based_Map_Completion/data_loader.py`（`MultistageDataset`）。

| 输入（patch） | 来源大图 | 说明 |
|---|---|---|
| `src_split/{idx}.png` | `porto_2014_r3026_z17.png` | sat RGB |
| `label/{idx}.png` | `road_heat.png` | GT 路网（map_label） |
| `traj_and_point_split/{idx}.npy` | `traj_point_heat.png` + `traj_line_heat.png` | 2 通道轨迹（traj+point） |
| `building_label/{idx}.png` | `building_heat.png` | 建筑 mask |
| `basemap_split/{idx}.png` | `road_heat.png` | 现有路网（basemap，与 label 同源） |

大图均与 sat 同尺寸同投影（15603×12862，Mercator），create_dataset.py 滑动窗口切 256 patch + `split_indices.json` 划分。DelvMap road 大图用"主要车行道"白名单（同 samroad GT），保证无闭合矩形/密集碎路。

---

## 五、关键注意事项

1. **路网过滤**：Porto 必须用 `--keep-highway-only`（保留所有汽车可通行 highway：主干道+residential/unclassified/road/living_street/service/track），否则 GT 混入 building/pedestrian 闭合。xian 不用（兼容）。依据 docs/chatgpt-osm类别分析.md。
2. **黑边对齐**：每次重建 GT 后必须跑 `clip_rn_to_sat.py` + `regenerate_labels_after_clip.py` + 重生成 mask/partial。黑边 tile 共 34 个（右边缘）。
3. **traj 语义**：`region_{c}_traj.png` 是**轨迹点**（closed 版），非连线。completion/4ch 用它作先验。如需连线版用 `traj_line.png`。
4. **GPS 量化横纹**：原始点图每 ~9px 有空行（源数据特性），closed 版已填补。
5. **release 3026**：2014-07-02 vintage，与 taxi traj 2013-07~2014-06 同年。Wayback CDN 偶发不可达时挂代理。
6. **4 个密集区精修**：重构后 0 个退化（过滤后路网稀疏，精修不再崩溃）。若未过滤会有 4 个 region 退化为未精修。
7. **坐标系一致性**：samroad/P2CNet/DelvMap 三者 Porto 均用同一套 Mercator 大图 + 同一坐标系（swap，左上原点），tile 与 patch 逐像素对齐。
