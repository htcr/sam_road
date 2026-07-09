# didi/xian 数据集 OSM 使用与过滤分析

> 状态：📋 仅供分析（2026-07-09），未修改 xian 实际数据
> 关联：[porto数据集制备与使用.md](porto数据集制备与使用.md)、[chatgpt-osm类别分析.md](chatgpt-osm类别分析.md)

本文档记录 didi/xian 数据集制备时 OSM 数据的来源、过滤情况，并与 Porto 重构后方案对比，供未来决策"是否重构 xian"参考。

---

## 一、OSM 数据清单（`datasets/didi/xian/osm/`）

### 1.1 PBF 文件（4 个）

| 文件 | node | way | 用途 |
|---|---|---|---|
| `xian-plus-190101.osm.pbf` | 151094 | 24625 | **制备 GT 用这个**（外扩版，2019-01-01 vintage） |
| `xian-190101.osm.pbf` | 14390 | 3724 | 精确 bbox 版（未用于制备） |
| `xian-plus-140101.osm.pbf` | — | — | 2014 版（备用） |
| `xian-140101.osm.pbf` | — | — | 2014 版（未用） |

命名规则：`xian[-plus]-YYMMDD.osm.pbf`，`plus` = bbox 外扩版（边界路网不被截断），`YYMMDD` = OSM 数据快照日期。

### 1.2 路网 shapefile（4 套，active 用）

| 目录 | 内容 | 用途 |
|---|---|---|
| `rn-comp-xa-190101-seg5/` | edges.shp + nodes.shp | **制备 active 用这个**（已构图路网边，seg5 = 5米分段） |
| `rn-comp-xa-190101-didi/` | edges.shp + nodes.shp | didi 变体（备用） |
| `rn-comp-xa-190101-filter/` | edges.shp（无 nodes） | filter 变体（备用） |
| `rn-comp-cd-190101-didi/` | edges.shp + nodes.shp | 成都（chengdu）数据（备用） |

---

## 二、制备实际用了哪些

依据 `tools/prepare_dataset/RUN.md:536-541`：

```bash
python download_use_osm.py \
  --dataset_type trajectory \
  --osm_pbf ../osm/xian-plus-190101.osm.pbf \       # GT 来源
  --active_shp ../osm/rn-comp-xa-190101-seg5/edges.shp \  # active 来源
  --sat_source local:... \
  config/xian.json
```

- **GT 路网**（`graph_gt.pickle` / `refine_gt_graph.p` / `gt.png`）：从 `xian-plus-190101.osm.pbf` 构建。
- **active 路网**（`active.png` / `active_graph.pickle`）：从 `rn-comp-xa-190101-seg5/edges.shp` 构建（shapefile，已预构图）。

> ⚠️ **active 已废弃**：xian 的 active 被判定为"错的"（见前序对话），Porto 制备时已忽略 active，走 `--dataset_type original`。xian 历史数据仍保留 active 文件但不再使用。

---

## 三、是否经过过滤

### 3.1 PBF 层面：已预过滤（只含 highway）

`xian-plus-190101.osm.pbf` 统计：
- 总 way 24625，**非 highway way = 0**
- 即 PBF 在导出时已经过滤掉 building / landuse / leisure / amenity / barrier / natural 等非路 way

**结论**：xian GT **不会**出现 Porto 那种 building 闭合矩形问题（因为 PBF 源头就没 building）。

### 3.2 OSMHandler 层面：不过滤

`download_use_osm.py` 的 `OSMHandler` 默认 `keep_highway_only=False`，xian 制备时不传 `--keep-highway-only`，所以：
- 收所有 `len(w.nodes) >= 2` 的 way
- 由于 PBF 已预过滤，实际收的都是 highway
- **但包含纯行人路**（footway / pedestrian / steps / path / cycleway）

### 3.3 xian PBF 的 highway 类型分布

| 类型 | 数量 | 占比 | 车行? |
|---|---|---|---|
| residential | 8078 | 32.8% | ✅ |
| service | 2290 | 9.3% | ✅ |
| unclassified | 1919 | 7.8% | ✅ |
| secondary | 1884 | 7.6% | ✅ |
| motorway | 1873 | 7.6% | ✅ |
| tertiary | 1806 | 7.3% | ✅ |
| primary | 1700 | 6.9% | ✅ |
| **footway** | **1186** | **4.8%** | ❌ 行人 |
| motorway_link | 1012 | 4.1% | ✅ |
| trunk | 786 | 3.2% | ✅ |
| trunk_link | 463 | 1.9% | ✅ |
| primary_link | 388 | 1.6% | ✅ |
| living_street | 266 | 1.1% | ✅ |
| **path** | **229** | **0.9%** | ❌ 行人 |
| **pedestrian** | **229** | **0.9%** | ❌ 行人 |
| track | 128 | 0.5% | ⚠️ 越野 |
| **steps** | **103** | **0.4%** | ❌ 行人 |
| secondary_link | 95 | 0.4% | ✅ |
| construction | 57 | 0.2% | 🚧 |
| **cycleway** | **48** | **0.2%** | ❌ 自行车 |

**纯行人路合计 1795 条（7.3%）**：footway 1186 + path 229 + pedestrian 229 + steps 103 + cycleway 48。这些被 OSMHandler 无过滤地收进了 xian GT。

---

## 四、xian vs Porto 对比

| 项 | xian（现状） | Porto（2026-07-09 重构后） |
|---|---|---|
| PBF 来源 | `xian-plus-190101.osm.pbf`（已预过滤只含 highway） | `porto_1km.osm.pbf`（全量，含 building/landuse 等） |
| PBF 预过滤 | ✅ 只含 highway（0 非路 way） | ❌ 全量 |
| OSMHandler 过滤 | ❌ 不过滤（`keep_highway_only=False`） | ✅ `--keep-highway-only` |
| GT 含非路 way（building 等） | ❌ 不含（PBF 已过滤） | ❌ 不含（白名单过滤） |
| GT 含纯行人路（footway 等） | ✅ **含 1795 条（7.3%）** | ❌ 不含（白名单排除） |
| GT 含 construction | ✅ 含 57 条 | ❌ 不含 |
| building 闭合矩形 | ❌ 无 | ❌ 无 |
| 过滤白名单 | 无（全收 highway） | `_KEEP_HIGHWAY`（16 类车行道） |

**核心差异**：
- xian 靠 **PBF 预过滤**去掉了非路 way，但**保留了行人路**（7.3%）。
- Porto 靠 **代码层白名单**过滤，既去非路 way，又去行人路，只留车行道。
- 两者 GT 都无 building 闭合，但 **xian 多了行人路、Porto 不含**。

---

## 五、xian GT 的潜在问题

1. **行人路噪声（7.3%）**：footway/pedestrian/steps/path 被当路网 GT。路网重建任务若只关心车行道，这些是噪声；若关心完整交通网络，则可接受。
2. **construction（57 条）**：建设中的道路（未通车）也被收进 GT，严格说是错误的（现实不可通行）。
3. **active 已废弃**：xian 仍保留 active 文件但不再使用，占用空间且易混淆。

**但**：xian 已基于现状训练了大量模型（extraction/completion/4ch 等），重构 GT 会导致：
- 所有 xian 模型需重训
- 历史实验结果（metrics）不可比
- 投入产出需权衡

---

## 六、未来决策：是否重构 xian

**待决策项**（记录于 TODO）：

- [ ] **是否对 xian 加 `--keep-highway-only` 重构 GT**，排除行人路（footway/pedestrian/steps/path/cycleway）+ construction，与 Porto 对齐标准。
  - **利**：xian/Porto GT 标准统一（只车行道），消除行人路噪声，跨数据集对比更公平。
  - **弊**：xian 所有已训模型需重训；历史 metrics 失效；投入大。
  - **触发条件**：若未来 xian/Porto 跨数据集对比实验发现行人路噪声影响结论，或需要统一基准时，再重构。
  - **若重构**：对 xian 跑 `download_use_osm.py --keep-highway-only`（xian PBF 已无非路 way，只需排除行人路），然后黑边裁剪 + mask + partial 重建。命令同 Porto 文档 2.5 节，config 换 `xian.json`，bbox 换 xian 范围。

---

## 七、附：xian GT 实测数据

- `region_0_graph_gt.pickle`：节点 193，边 187
- 对比 Porto `region_0`：节点 94，边 94（Porto 过滤后更稀疏，且 region_0 在边缘）

xian 数据集规模：378 块（21×18），bbox lat[34.206385, 34.279658] lon[108.917423, 108.99286]，sat 5625×6610（Wayback 2019 release 645）。
