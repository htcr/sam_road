# 实验执行清单 (TODO)

> **当前状态(2026-06-26)**:partial 采样回退到**按边随机**(component 采样导致训练震荡+退化,详见 `docs/component采样退化分析.md`)。
> 服务器需先 `git pull` 拿到最新代码,重新生成 partial(按边随机),重训 completion。

---

## ✅ 完成情况 (2026-06-26)

| 步骤 | 状态 | 产物 / 结果 |
|---|---|---|
| 第0步: 服务器准备 / partial 重生成 | ✅ 已完成 | partial 已按**按边随机**策略重新生成(本地完成,服务器需重跑) |
| 实验1: SpaceNet completion 重跑重测 | 🔄 需重跑 | 之前 component 版 APLS=0.5293 已作废,需用按边随机重训 |
| 实验2: DiDi Xian completion 重跑重测 | 🔄 需重跑 | 之前 component 版 APLS=0.4571 已作废,需用按边随机重训 |
| 实验3: P2CNet 对比复现 | ⏳ 未开始 | 仍阻塞于 P2CNet 数据格式转换脚本 |

### 历史结果(已作废,component 采样)

| 实验 | component 版(已废) | 按边随机 0625 版(基线) |
|---|:---:|:---:|
| spacenet completion | APLS 0.5293 | APLS **0.7000**(待复现) |
| didi_xian completion | APLS 0.4571 | APLS **0.5878**(待复现) |

> component 采样让 APLS 退化 0.17/0.13,且 val_loss 震荡。回退按边随机应恢复 0.70/0.59 水平。

---

## 第0步:服务器准备(必做,一次性)

```bash
cd ~/sam_road
git pull origin main
conda activate samroad

# 重新生成 partial (按边随机, seed 42, 覆盖 component 版)
python data/generate_partial_prior.py --dataset didi \
    --input_dir datasets/didi/xian/2019_400 --output_dir datasets/didi/xian/2019_400 \
    --keep_ratio 0.5 --seed 42 --strategy edge_random

python data/generate_partial_prior.py --dataset spacenet \
    --input_dir datasets/spacenet/RGB_1.0_meter --output_dir datasets/spacenet/RGB_1.0_meter \
    --keep_ratio 0.5 --seed 42 --strategy edge_random
```

---

## 实验1:spacenet completion 重跑重测

partial 回退按边随机,需重训 completion + infer + eval。目标恢复 APLS 0.70。

```bash
python run.py --task completion --dataset spacenet --gpus 0
# 默认 steps=train,infer,eval, ckpt=auto(best)
# 产物: runs/completion_spacenet_<新时间戳>/
# 预期: APLS ~0.70 (恢复0625水平, 之前component版0.53已废)
```

---

## 实验2:didi_xian completion 重跑重测

```bash
python run.py --task completion --dataset didi_xian --gpus 0
# 产物: runs/completion_didi_xian_<新时间戳>/
# 预期: APLS ~0.59 (恢复0625水平, 之前component版0.46已废)
```

> 训练侧 `_create_known_graph` 目前仍是旧的按边随机采样。
> 若要训练也用按连通块保(与推理一致),先改 `data/dataset_completion.py` 再训(见备注)。

---

## 实验3:P2CNet 对比复现(spacenet + didi_xian)

用**你的数据集**(不是 P2CNet 自带的 spacenet)复现 P2CNet,做对比基线。
**问题**:P2CNet 数据格式与 sam_road 不同,需先转换。

### 3a. 数据格式转换(需先实现转换脚本)

P2CNet 期望目录结构:
```
data/<dataset>/
  Vegas/         # 仅 spacenet 按城市分; didi_xian 可单目录
    sats/        # 卫星图 PNG (你的 *_sat.png)
    maps/        # 完整GT路网 mask PNG (你的 road_mask_*.png)
    maps_50/     # 50% partial mask PNG (你的 *_partial.png, 已生成)
```
你的数据是 graph pickle + sat PNG,需转成像素级 mask PNG。
**待实现**: `tools/convert_to_p2cnet_format.py`
- sat: 直接复制/重命名 `*_sat.png` → `sats/`
- maps: 用 `road_mask_*.png`(已有)或从 GT graph 渲染 → `maps/`
- maps_50: 用已生成的 `*_partial.png` → `maps_50/`
- 注意坐标系:P2CNet 是像素 mask,需用 `transform_node` 渲染(didi/spacenet 不同)

### 3b. P2CNet 训练 + 测试

```bash
cd ~/research/P2CNet
conda activate <p2cnet环境>

# spacenet
python train_deeplabv3plus_mix_mp_sat_gsam_spacenet.py
python test_deeplabv3plus_mix_mp_sat_gsam_spacenet.py

# didi_xian (用 osm 入口, 改 config data_dir 指向你的 didi 数据)
python train_deeplabv3plus_mix_mp_sat_gsam_osm.py
python test_deeplabv3plus_mix_mp_sat_gsam_osm.py
```

> P2CNet config 需改 `data_dir` 指向转换后的数据;ratio 用 `mix` 或固定 `0.50`。
> P2CNet 的指标是像素级 IoU/F1(非 APLS/TOPO),对比时注意口径。

---

## 备注

1. **训练侧采样统一**:实验1/2 若要训练也用按连通块保,需先改 `data/dataset_completion.py` 的 `_create_known_graph`。开销已验证可忽略(每 epoch +0.3-2s)。不改则训练用旧策略、推理用新策略,有 train/infer gap。
2. **extraction 不用重跑**:extraction 不碰 partial,现有结果(extraction spacenet 0.7012 / didi_xian 0.4288)仍有效。
3. **P2CNet 转换脚本是阻塞项**:实验3 必须先完成 3a 的转换脚本,才能跑 3b。

---

## 待决策:是否重构 didi/xian GT(对齐 Porto 过滤标准)

> 背景:Porto 数据集(2026-07-09)用 `--keep-highway-only` 重构,GT 只保留车行道(排除行人路/非路 way)。
> xian 现状是 PBF 预过滤(无非路 way)但 OSMHandler 不过滤,**GT 含 7.3% 纯行人路**(footway/pedestrian/steps/path/cycleway)+ construction。
> 详见 `docs/xian数据集OSM使用与过滤分析.md`。

- [ ] **决策:是否对 xian 加 `--keep-highway-only` 重构 GT**,排除行人路 + construction,与 Porto 标准统一。
  - **利**:xian/Porto GT 标准统一(只车行道),消除行人路噪声,跨数据集对比更公平。
  - **弊**:xian 所有已训模型(extraction/completion/4ch)需重训;历史 metrics(extraction didi_xian 0.4288、completion APLS 0.5878 等)失效不可比;投入大。
  - **触发条件**:若未来 xian/Porto 跨数据集对比实验发现行人路噪声影响结论,或需统一基准时再重构。
  - **若重构**:对 xian 跑 `download_use_osm.py --keep-highway-only`(xian PBF 已无非路 way,只需排除行人路)→ 黑边裁剪 → mask → partial 重建。命令同 `docs/porto数据集制备与使用.md` 2.5 节,config 换 `xian.json`,bbox 换 xian 范围 lat[34.206385,34.279658] lon[108.917423,108.99286]。
  - **当前结论**:暂不重构(xian 已训大量模型,现状可接受;Porto 走新标准即可)。
