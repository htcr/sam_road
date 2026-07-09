# SAM-Road: 基于 SAM 的路网图提取

"Segment Anything Model for Road Network Graph Extraction" 官方代码库，CVPRW 2024。

📄 [论文](https://arxiv.org/pdf/2403.16051.pdf) | 🏆 CVPR 2024 第二届场景图与图表征学习研讨会 [最佳论文](https://sites.google.com/corp/view/sg2rl/)

> 原始英文 README 见 [docs/README_original_en.md](docs/README_original_en.md)

---

## 模型变体

### 1. SAM-Road（原始模型）

输入 RGB 卫星影像，经 SAM 编码器后分两路：MapDecoder 生成分割 mask，TopoNet 预测图拓扑。

```
RGB [B,H,W,3]
  → SAM ImageEncoderViT (ViT-B/L/H)
  → image_embeddings [B,256,h,w]
      ├→ MapDecoder → mask [B,H,W,2]  (ch0=keypoint, ch1=road)
      └→ TopoNet → topo_scores [B,N_s,N_p,1]
```

- 损失：`mask_loss (BCE/Focal) + topo_loss (BCE, masked)`
- 支持 LoRA 微调 qkv

### 2. SAM-Road 4ch（4通道模型）

在原始模型基础上扩展为 4 通道输入（RGB + Active 先验 mask），用于迭代式路网更新：

| 改动 | 原始 | 4ch |
|------|------|-----|
| 输入通道 | 3 (RGB) | 4 (RGB + Active) |
| `pixel_mean/std` | `[3]` 维 | `[4]` 维 (第4通道 mean=0, std=1) |
| `patch_embed` | `Conv2d(3,...)` | `Conv2d(4,...)`，第4通道零初始化 |
| LoRA 解冻 | 仅 qkv | qkv + `patch_embed` |

**先验增强策略**（仅训练时）：
- 20%：全黑先验（保持纯视觉生成能力）
- 60%：腐蚀先验（随机擦除块，模拟轨迹中断）
- 20%：完美 GT（学习信任准确先验）
- 验证集：始终全黑先验

### 3. SAM-Road Completion（路网补全模型）

专为路网补全设计，输入 RGB + 已知路网特征图：

```
RGB + road_feature_map [B,4,H,W]
  → SAM Encoder ─┬─ image_embeddings
  → RoadGraphEncoder (CNN) ─┬─ road_embeddings
                             └→ FeatureFusion (1×1 Conv) → fused_features
      ├→ MapDecoder (仅用 image_emb，不混入图信息避免过拟合)
      └→ TopoNetCompletion
           + RoadGraphGNN (编码已知图拓扑)
           pair_proj 输入 +2*graph_dim → topo_scores
```

**路网特征图 4 通道**：道路 mask / 距离场 / 方向场 / 关键节点位置

**RoadGraphGNN**：在已知路网边上做 MultiheadAttention 消息传递，为每个节点生成图拓扑嵌入。

---

## 数据集与坐标系

四个数据集的 **pickle 坐标原点不同**，这是最容易踩坑的地方：

|  | CityScale | SpaceNet | Xian (DiDi) | Porto |
|---|---|---|---|---|
| **图片尺寸** | 2048×2048 | 400×400 | 400×400 | 400×400 |
| **pickle 原点** | 左上（图像坐标） | 左下（数学坐标） | 左上（图像坐标） | 左上（图像坐标） |
| **coord_transform** | `v[:, ::-1]`（swap） | `np.stack([v[:,1], SIZE-v[:,0]])`（swap+flip-y） | `v[:, ::-1]`（swap） | `v[:, ::-1]`（swap） |
| **样本数** | 180 | 2549 | 378 | 1015 |
| **train/val/test** | 180（手动） | 2040/127/382 | 302/37/39 | 812/101/102 |
| **真实轨迹 traj** | ✗ | ✗ | ✓ (`region_{c}_traj.png`) | ✓ (`region_{c}_traj.png`) |
| **partial (completion)** | ✗ | ✗ | ✓ | ✓ (edge_random + component 50/25/75) |

> ⚠️ SpaceNet 的 pickle 使用数学坐标系（y 轴从下向上），需要翻转 y 轴。CityScale/Xian/Porto 使用图像坐标系（row 从上向下），只需交换。Porto 与 Xian 坐标系完全一致，可直接复用 didi 范式。

**IoU 定量验证**（变换后的 road_mask vs GT.png）：

| 变换 | CityScale | SpaceNet | Xian |
|---|:---:|:---:|:---:|
| swap (cityscale) | **~0.60 ✅** | ~0.03 | **~0.60 ✅** |
| flip-y (spacenet) | ~0.03 | **~0.62 ✅** | ~0.10 |
| raw | ~0.03 | ~0.03 | ~0.01 |

详细分析见 [数据集与坐标系分析](docs/数据集与坐标系分析.md)。Porto 制备详见 [porto数据集制备与使用](docs/porto数据集制备与使用.md)。

---

## 项目结构

```
sam_road/
├── config/            # 训练配置 (YAML)
├── data/              # 数据集类 & DataLoader
│   ├── dataset.py             # 基础: CityScale / SpaceNet / Xian / Porto
│   ├── dataset_4ch.py         # 4ch: + active mask 与先验增强
│   ├── dataset_completion.py  # Completion: + 已知图 & 特征图
│   └── dataset_registry.py    # 数据集注册表 (路径/坐标变换/划分)
├── datasets/          # 原始数据 & generate_labels.py
│   ├── cityscale/
│   ├── spacenet/
│   ├── didi/xian/             # 2019_400/ (region 文件, 含 traj.png), processed/, data_split.json
│   └── porto/                 # 2014_400/ (1015 块, keep-highway-only 车行道), processed/, data_split.json
├── rawdata/           # 原始素材 + 大图 (gitignore, 与 datasets/ 平级)
│   └── porto/                # sat/road/building/traj 大图 + PBF + train.csv
├── tools/prepare_dataset/      # 数据制备脚本
│   ├── download_use_osm.py     # sat+rn+active (支持 --sat_source local + NW 编号 + clip_bbox + --keep-highway-only)
│   ├── generate_traj.py        # 生成 region_{c}_traj.png (DelvMap 真实轨迹, 已对齐, 支持 --suffix)
│   ├── generate_porto_*.py     # Porto 大图制备 (traj/building/road) + clip_pbf_bbox
│   ├── esri_wayback.py         # Esri Wayback 历史影像下载
│   ├── graph_ops.py / esri.py
│   └── config/{xian,porto}.json  # size=400, 各城市 bbox
├── docs/              # 文档
│   ├── 数据集与坐标系分析.md   # 坐标系详细分析
│   ├── 方案B_路网补全设计.md   # 补全模型设计文档
│   ├── Train.md                # 训练流程详解
│   ├── nan_fix_completion.md   # 补全模型 fp16 NaN 修复总结
│   ├── experiment_metrics_20260617.md # 已完成推理/metric 实验汇总
│   ├── README_original_en.md  # 原始英文 README
│   └── imgs/                  # 验证图片（已入库）
├── engine/            # 训练 & 推理
│   ├── train.py / train_4ch.py / train_completion.py
│   ├── inferencer.py / inferencer_4ch.py / inferencer_completion.py
│   └── test.py
├── models/            # 模型定义
│   ├── sam_road.py            # 原始模型
│   ├── sam_road_4ch.py        # 4通道模型
│   └── sam_road_completion.py # 补全模型
├── outputs/           # 早期生成物（gitignore）
├── postprocess/       # 图提取、triage
├── scripts/           # 验证 & 可视化脚本
├── sam/               # SAM 子模块
├── tools/             # 工具集
│   ├── config_utils.py        # load_config / run_root 路径 / profile / best ckpt
│   ├── registry.py            # TASKS/DATASETS 注册表 (run.py 用)
│   ├── run_info.py            # run_info.yaml dumper (复现用)
│   └── migrate_ckpts_to_runs.py # 旧 checkpoints/ → runs/ 迁移
├── metrics/           # APLS & TOPO 评估
├── run.py             # 🆕 实验编排入口 (train+infer+eval 一条命令)
├── batch.yaml         # 🆕 四任务批量配置
├── runs/              # 🆕 实验产物根目录 (每个 run_id 一个子目录, gitignore 大产物)
├── conda_only.yml
├── myenv.yml
└── pip_requirements.txt
```

---

## 快速开始

### 安装

```bash
# 方式一: conda + pip 分开安装（推荐，避免 pip 卡死）
conda env create -f conda_only.yml
conda activate samroad
pip install -r pip_requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 方式二: 全量 conda
conda env create -f myenv.yml
```

#### ⚠️ 已知环境坑：`setuptools` 与 `libstdc++` 版本冲突

在某些机器上（尤其是系统 GCC ≤ 11，例如 Debian 12），新建 SAM 环境后第一次 `python -m engine.train ...`
会接连撞两个错误，必须按下面顺序处理。

**坑 1：`ModuleNotFoundError: No module named 'pkg_resources'`**

`setuptools >= 81` 移除了 `pkg_resources`，但 `lightning.fabric` 仍在
`__import__('pkg_resources').declare_namespace(...)`，导致 `import lightning.pytorch` 直接报错。

```bash
conda activate SAM
pip install "setuptools<81"      # 80.x 自带 pkg_resources，长期可用
```

**坑 2：`ImportError: libstdc++.so.6: version 'CXXABI_1.3.15' not found`**

只在导入 `torch` 之后再导入 `scipy` / `lightning` 时触发。原因不是 SAM 环境坏了：
`torch/lib/libnvToolsExt-*.so` 没有正确的 RPATH 指向 `$CONDA_PREFIX/lib`，
`import torch` 时 loader 顺着默认路径加载了系统的旧版 `libstdc++.so.6`
（GCC 11，最高 `CXXABI_1.3.13`），并把这个 SONAME pin 在进程里；
之后 `scipy._highspy` / `pypocketfft` 再 NEEDED `libstdc++.so.6` 时就拿到了旧的，
即使 conda 自己的 `libstdc++.so.6.0.34`（含 `CXXABI_1.3.15`）就在 RPATH 上也用不上。

最干净的修法是给 SAM 环境加一对 conda activate/deactivate 钩子，
让它在 `conda activate SAM` 时自动 `LD_PRELOAD` conda 自己的新版 libstdc++，
deactivate 时自动还原。**只影响 SAM 环境**，新开 tmux 窗口无需任何 export。

```bash
SAM_PREFIX=$(conda env list | awk '$1=="SAM"{print $NF}')
mkdir -p "$SAM_PREFIX/etc/conda/activate.d" "$SAM_PREFIX/etc/conda/deactivate.d"

cat > "$SAM_PREFIX/etc/conda/activate.d/zz-libstdcxx-preload.sh" <<'EOF'
# Force conda's libstdc++ to load before torch pulls in the system one.
if [ -f "$CONDA_PREFIX/lib/libstdc++.so.6" ] && [ "${_SAM_LD_PRELOAD_SET:-0}" != "1" ]; then
    export _SAM_OLD_LD_PRELOAD="${LD_PRELOAD-__unset__}"
    export _SAM_LD_PRELOAD_SET=1
    if [ -z "${LD_PRELOAD:-}" ]; then
        export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6"
    else
        export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6:$LD_PRELOAD"
    fi
fi
EOF

cat > "$SAM_PREFIX/etc/conda/deactivate.d/zz-libstdcxx-preload.sh" <<'EOF'
if [ "${_SAM_LD_PRELOAD_SET:-0}" = "1" ]; then
    if [ "${_SAM_OLD_LD_PRELOAD-}" = "__unset__" ]; then
        unset LD_PRELOAD
    else
        export LD_PRELOAD="$_SAM_OLD_LD_PRELOAD"
    fi
    unset _SAM_OLD_LD_PRELOAD _SAM_LD_PRELOAD_SET
fi
EOF
```

写完重新 `conda deactivate && conda activate SAM`，`echo $LD_PRELOAD` 应自动指向
`$CONDA_PREFIX/lib/libstdc++.so.6`，从此 `python -m engine.train ...` 直接可用。

> 若以后系统 GCC 升到 12+ 自带 `CXXABI_1.3.15`，可删除这两个钩子；
> 它们只是把 torch 错误 RPATH 的副作用屏蔽掉，治标不治本。

### SAM 权重

下载 [ViT-B 权重](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)，放到：
```
sam_road/sam_ckpts/sam_vit_b_01ec64.pth
```

### 数据准备

按 [RNGDet++](https://github.com/TonyXuQAQ/RNGDetPlusPlus) 说明下载数据集：

| 数据集 | 下载链接 |
|--------|---------|
| SpaceNet | https://drive.google.com/uc?id=1FiZVkEEEVir_iUJpEH5NQunrtlG0Ff1W |
| CityScale | https://drive.google.com/uc?id=1R8sI1RmFe3rUfWMQaOfsYlBDHpQxFH-H |

放到 `datasets/` 下，然后分别运行 `generate_labels.py`：
```bash
cd datasets/spacenet && python generate_labels.py
cd datasets/cityscale && python generate_labels.py
```

**Xian (DiDi)** 数据集本地制备（sat + rn + active + traj，378 块，对齐 DelvMap 范围）：
```bash
# 1. sat + GT路网图 + active (从 2019_400/ 目录运行; ESRI 不可达时用 DelvMap sat 大图)
cd datasets/didi/xian/2019_400
PYTHONPATH=../../../tools/prepare_dataset python ../../../tools/prepare_dataset/download_use_osm.py \
    --dataset_type trajectory \
    --osm_pbf ../osm/xian-plus-190101.osm.pbf \
    --active_shp ../osm/rn-comp-xa-190101-seg5/edges.shp \
    --sat_source local:/path/to/DelvMap/rawdata/sat_img.png \
    ../../../tools/prepare_dataset/config/xian.json

# 2. road_mask / keypoint_mask (从 2019_400/ 运行, 输出到上一层 processed/)
python ../../../datasets/didi/xian/generate_labels.py --root .

# 3. 真实轨迹 traj (从 repo 根运行)
python tools/prepare_dataset/generate_traj.py \
    --config tools/prepare_dataset/config/xian.json \
    --traj-png /path/to/DelvMap/rawdata/traj_heat.png \
    --out-dir datasets/didi/xian/2019_400 --mode traj --qc

# 4. data_split.json (302/37/39)
python data/img_folder_to_json_list.py
```
详见 [docs/数据集与坐标系分析.md](docs/数据集与坐标系分析.md) §2.2.1 与 [tools/prepare_dataset/RUN.md](tools/prepare_dataset/RUN.md)。

### 训练

> **推荐：用 `run.py` 编排**（统一 train+infer+eval，路径自动隔离，详见 [docs/实验编排方案设计.md](docs/实验编排方案设计.md)）
> ```bash
> # 单个全流程 (extraction / completion / 4ch)
> python run.py --task completion --dataset spacenet --gpus 0
> python run.py --task completion --dataset porto --gpus 0      # Porto (1015 块)
> # 四任务批量 (spacenet→GPU0, didi_xian→GPU1)
> python run.py --batch batch.yaml --parallel
> ```
> 下方为底层脚本直调（`--run-root` 可选，不给则走老路径）。

```bash
# 原始模型
python engine/train.py --config=config/toponet_vitb_512_cityscale.yaml
python engine/train.py --config=config/toponet_vitb_256_spacenet.yaml

# 4通道模型
python engine/train_4ch.py --config=config/toponet_vitb_256_spacenet.yaml

# 补全模型
python engine/train_completion.py --config=config/toponet_vitb_256_spacenet_completion.yaml
```

#### 训练参数一览

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--config` | `None` | YAML 配置文件路径，定义 BATCH_SIZE / LR / DATASET 等超参，见 `config/` |
| `--resume` | `None` | 从断点续训，传入 ckpt 文件路径 |
| `--gpus` | `"0"` | GPU 编号，单卡 `"0"` / 多卡 `"0,1"`（train_4ch.py 无此参数，用 `CUDA_VISIBLE_DEVICES`） |
| `--precision` | `16` | 精度档位：`16` (fp16 mixed) / `bf16-mixed` / `32` (fp32)，详见下方 `--precision` 专节 |
| `--patience` | `0` | Early stopping 等待轮数；`0` = 不启用；`>0` 时 val_loss 连续 patience 轮不降则停 |
| `--fast_dev_run` | `False` | 冒烟测试：仅跑 1 个 train batch + 1 个 val batch，不写 ckpt |
| `--dev_run` | `False` | 开发模式：数据集取小子集，但走完整 trainer（含真实 epoch 和验证） |

> `train.py` / `train_4ch.py` / `train_completion.py` 三个脚本参数完全相同（除 `train_4ch.py` 无 `--gpus`）。

#### 多卡机：指定单卡训练

`engine/train.py` 和 `engine/train_completion.py` 内置 `--gpus` 参数（默认 `"0"`，即第 0 块卡），多卡机要训单卡时显式传：

```bash
# 用第 1 块卡训练补全模型
python engine/train_completion.py \
    --config config/toponet_vitb_256_spacenet_completion.yaml \
    --gpus 1

# 用 0,1 两块卡 DDP（如需）
python engine/train_completion.py \
    --config config/toponet_vitb_256_spacenet_completion.yaml \
    --gpus 0,1
```

`engine/train_4ch.py` 暂未实现 `--gpus`，需要用 `CUDA_VISIBLE_DEVICES` 环境变量遮罩：

```bash
CUDA_VISIBLE_DEVICES=1 python engine/train_4ch.py \
    --config config/toponet_vitb_256_spacenet.yaml
```

> 提示：`CUDA_VISIBLE_DEVICES` 也可以叠在 `--gpus` 上，先遮罩再选卡。常用于把整张卡从其他用户那"独占"出来：`CUDA_VISIBLE_DEVICES=2,3 ... --gpus 0` = 物理 GPU 2 被映射成逻辑 0。

#### `--precision`：选混合精度档位

三个 train 脚本都接受 `--precision`（默认 `16`，即 fp16 mixed precision）：

| 取值 | 显存（B=16, ViT-B 256）| 速度 | 硬件门槛 |
|---|---|---|---|
| `16` / `16-mixed` | ~6–8 GB | 1.0× | Pascal 起 |
| `bf16-mixed` | ~6–8 GB | 0.95–1.0× | **Ampere (sm_80) 起** |
| `32` / `32-true` | ~13–16 GB | 0.5–0.6× | 任意 |

按硬件选择经验：

```bash
# 4090 / 3090 / A100：bf16 最稳，零代价
python engine/train_completion.py --config <cfg> --gpus 0 --precision bf16-mixed

# 2080 Ti：bf16 不被原生支持，留 fp16
python engine/train_completion.py --config <cfg> --gpus 0 --precision 16

# 论文最终对照：fp32 (4090 24G 跑 B=16 fp32 约 ~13 GB)
python engine/train_completion.py --config <cfg> --gpus 0 --precision 32
```

> ⚠️ 2080 Ti 强行用 `bf16-mixed` 不会报错但会走慢速 emulation（约 0.4× fp16 速度），不建议。

> 💡 补全模型 (`engine/train_completion.py`) 在 fp16 下已加六层 NaN 防御
> （梯度裁剪 + attention mask 改 -1e4 + RoadGraphGNN/mask/topo 输出 nan_to_num
> + topo_loss 布尔索引 + `on_after_backward` 梯度清零），fast_dev_run 与长训均稳定。
> 完整原因分析、修复细节和测试记录见 [docs/nan_fix_completion.md](docs/nan_fix_completion.md)。

#### 快速冒烟测试（fast dev run）

正式训练前先用 `--fast_dev_run` 跑 1 个 batch（含 1 个 val batch），
用来验证数据管线、模型 forward、loss、checkpoint 目录权限是否都 OK。
这个模式**不会写 ckpt、不会发 logger，只在几秒内跑完一个完整循环**：

```bash
# 原始模型（spacenet/cityscale 任一 config 都行）
python engine/train.py            --config=config/toponet_vitb_256_spacenet.yaml --fast_dev_run --gpus 0

# 4通道模型 (无 --gpus，用 CUDA_VISIBLE_DEVICES)
CUDA_VISIBLE_DEVICES=0 python engine/train_4ch.py --config=config/toponet_vitb_256_spacenet.yaml --fast_dev_run

# 补全模型
python engine/train_completion.py --config=config/toponet_vitb_256_spacenet_completion.yaml --fast_dev_run --gpus 0
```

如果想跑得久一点观察 loss 走势但仍不写 ckpt，可以改用 `--dev_run`
（数据集会在 dev 模式下取小子集，但走的是完整 trainer，会有真实 epoch / 验证）。

### 推理

> **推荐：用 `run.py` 编排**（ckpt 自动选 best，输出走 `runs/{id}/infer/`）
> ```bash
> # 复用已有训练, 只跑 infer+eval (ckpt 自动选 best)
> python run.py --task completion --dataset spacenet \
>     --run-id completion_spacenet_20260616_202527 --steps infer,eval --resume-run
> ```

三个 inferencer 都支持 `--run-root`（走 `runs/{id}/infer/`）和 `--checkpoint auto`（自动选 best）。`run.py --checkpoint` 还支持 `last`、`epoch:N` / `epN` / 纯数字 `N`（按 Lightning 0-based epoch 编号找 ckpt；例如 `9` = `epoch=09` = 第 10 个 epoch 结束），以及完整路径。不给 `--run-root` 时走老路径 `./save/<前缀>_<timestamp>/`，目录结构一致：

```
save/<前缀>_<timestamp>/
├── config.yaml         # 推理时使用的模型 config 副本
├── run_info.yaml       # 运行元信息 (ckpt 路径 / 命令行 / git commit / 环境 / 耗时)
├── inference_time.txt  # 总推理时长
├── graph/{name}.p      # 预测路网 (sat2graph 邻接表 pickle), 评估脚本读这个
├── mask/               # {name}_road.png + {name}_itsc.png
└── viz/{name}.png      # 节点+边叠加在原图上的可视化
```

> 💡 **`config.yaml` vs `run_info.yaml`**：前者是模型超参（机器读，能 `load_config` 直接重跑），后者是这次"怎么跑的"（人看，回答"哪份 ckpt？哪条命令？哪个 git commit？"）。两者关注点分离。
> 训练时类似的 `run_info_<timestamp>.yaml` 会同时写到 `checkpoints/<exp>/` 和 `train_logs/`，让权重文件 / 文本日志都能反查到来源。

#### 推理参数一览

**通用参数**（三个推理脚本均有）：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--config` | `None` | 模型配置 YAML，需与训练时一致 |
| `--checkpoint` | `None` | 模型 ckpt 文件路径 |
| `--output_dir` | `None` | 自定义输出子目录名（默认用时间戳） |
| `--device` | `"cuda"` | 推理设备：`cuda` / `cpu` |

**补全模型专属参数**（仅 `inferencer_completion.py`）：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--input_graph` | `None` | 单个已知路网 pickle 文件路径（邻接表格式） |
| `--input_graph_dir` | `None` | 已知路网 pickle 目录（每个 region 一个 .p 文件） |
| `--traj_dir` | `None` | 轨迹热力图目录（仅 Xian 数据集，默认 `region_{c}_active.png`；也可指向 `generate_traj.py` 生成的 `region_{c}_traj.png` 真实 GPS 轨迹） |

**4 通道模型专属参数**（仅 `inferencer_4ch.py`）：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--exp_id` | 必填 | 实验标识，输出目录 `save/<exp_id>/` |

> ⚠️ **必须用 `python -m engine.xxx` 启动，不能用 `python engine/xxx.py`。**
> `inferencer.py` 和 `inferencer_4ch.py` 没有在脚本顶部插入 `sys.path`，
> 直接按文件路径跑会报 `ModuleNotFoundError: No module named 'tools'`。
> 用 `-m` 形式 Python 把 cwd（项目根）当作模块搜索根，`tools/`、`data/`、`models/` 都能正确解析。

```bash
cd /home/hanhaoyu/sam_road
conda activate SAM

# 原始模型 (timestamp 自动附加 → save/infer_<timestamp>/)
CUDA_VISIBLE_DEVICES=1 python -m engine.inferencer \
    --config config/toponet_vitb_256_spacenet_local.yaml \
    --checkpoint "checkpoints/samroad_spacenet/<best.ckpt>"

# 4 通道模型 (必须显式给 --exp_id, 否则报错)
CUDA_VISIBLE_DEVICES=1 python -m engine.inferencer_4ch \
    --config config/toponet_vitb_256_spacenet.yaml \
    --checkpoint "checkpoints/samroad_4ch/<best.ckpt>" \
    --exp_id my_4ch_run

# 补全模型 (用已知路网作为先验; 此脚本内部已插 sys.path, -m 不强制但保持风格统一)
CUDA_VISIBLE_DEVICES=1 python -m engine.inferencer_completion \
    --config config/toponet_vitb_256_spacenet_completion.yaml \
    --checkpoint "checkpoints/samroad_completion/<best.ckpt>" \
    --input_graph_dir datasets/spacenet/RGB_1.0_meter

# 补全模型 - 无先验回退模式 (省略 --input_graph_dir, 模型退化为纯 SAM-Road extraction)
# 主要用于: 评估补全模型在"无先验"输入上的纯生成能力, 与 inferencer.py 做对照
CUDA_VISIBLE_DEVICES=1 python -m engine.inferencer_completion \
    --config config/toponet_vitb_256_spacenet_completion.yaml \
    --checkpoint "checkpoints/samroad_completion/<best.ckpt>"
```

> 💡 **几个常见坑**：
> - 多卡机用 `CUDA_VISIBLE_DEVICES=N` 选卡，**不是** `CUDA_VISIBLE_DIVICES`（拼写易错），
>   推理脚本内部用 `args.device="cuda"`，不接受 `--gpus` 参数
> - ckpt 文件名带 `=` 字符（PL 默认模板）：建议用双引号包起来 `"checkpoints/.../epoch-epoch=00-val_loss=0.1148.ckpt"`
> - 想自定义输出名而不是用时间戳：`--output_dir my_run` → 实际目录 `save/my_run/`（`output_dir_prefix` 不再生效）
> - 找最佳 ckpt：`ls -lh checkpoints/samroad_spacenet/`，挑文件名里 `val_loss=` 最小的那个

### 评估

[metrics/eval.py](metrics/eval.py) 一站式跑 APLS 和 TOPO 两个指标，
直接读 `<inferencer 输出>/graph/{name}.p` 与 `datasets/<dataset>/...` 下的 GT 比较。

#### 评估参数一览

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--dataset` | 必填 | 数据集类型：`cityscale` / `spacenet` / `didi` |
| `--dir` | 必填 | 推理输出目录（相对项目根，如 `save/infer_20260614`） |
| `--metric` | `all` | 评估指标：`apls` / `topo` / `all` |
| `--workers` | `None`（自动） | 并行 worker 数，16~32 适合多核日常跑 |

**前置要求**：
- `go` 编译器（APLS 用 Go 实现的二进制评估，本机已装 1.25.4）
- 推理已跑完，`graph/` 目录里包含**测试集全部样本**的 `.p` 文件
  （spacenet 测试集 382 张；漏样本会被 `SKIP: Missing` 跳过、最终指标偏乐观）

**用法**（必须 `cd metrics/` 再跑，eval.py 内部用大量 `../datasets/...` 相对路径）：

支持的数据集名：`cityscale` / `spacenet` / `didi_xian`。
历史别名 `didi` 仍可用，但已不推荐；统一使用 `didi_xian`，避免未来加入 `didi_chengdu` 时混淆。

```bash
cd /home/hanhaoyu/sam_road/metrics
conda activate SAM

# 一次跑两个指标 (默认), --workers 视机器核数, 16~32 比较合适
python eval.py --dataset spacenet \
               --dir save/infer_<timestamp> \
               --workers 16

# Xian / DiDi Xian
python eval.py --dataset didi_xian \
               --dir save/infer_completion_didi_xian_<timestamp> \
               --workers 16

# 只跑其中一个
python eval.py --dataset spacenet --dir save/<...> --metric topo --workers 16
python eval.py --dataset spacenet --dir save/<...> --metric apls --workers 16
```

> 💡 **`--dir` 路径相对于项目根**（不是相对 `metrics/`），eval.py 内部统一加 `../` 前缀。
> 例如推理输出在 `/home/hanhaoyu/sam_road/save/infer_20260614_010203/`，参数就写
> `--dir save/infer_20260614_010203`，**不要**写绝对路径或 `../save/...`。

**输出** 写到 `save/<...>/results/`：
- `apls.json` — `{"apls": [[name, val], ...], "final_APLS": <mean>}`
- `topo.json` — `{"mean topo": [F1, P, R], "f1": ...}`
- `apls/{name}.txt` — 每张图一行 `precision recall apls`
- `topo/{name}.txt` — 每张图最后一行 `precision recall`

**预估时长**（spacenet 382 张，本机 384 核）：

| `--workers` | 总时长 | 适用场景 |
|---|---|---|
| 1 | ~17 分钟 | 单步调试，看具体哪张图卡住 |
| 16 | ~5–10 分钟 | 日常跑，CPU 占用约 16 核（多用户友好）|
| 32 | ~3–5 分钟 | 全机空闲时最快 |

> ⚠️ `--dataset` 必须与推理时 config 的 `DATASET` 一致。eval 不会自动检查，跑错了
> 会报 `Missing pred` 或者用错的坐标系算指标，最终数值看上去合理但实际无意义。

#### 一条命令把推理 + 评估串起来

跑完推理后立刻评估的常用模板（注意推理输出的目录要从终端日志里拿，或用 `--output_dir` 自己起名）：

```bash
# 1. 推理 (会打印输出目录, 例如 save/infer_my_run/)
cd /home/hanhaoyu/sam_road
CUDA_VISIBLE_DEVICES=1 python -m engine.inferencer \
    --config config/toponet_vitb_256_spacenet_local.yaml \
    --checkpoint "checkpoints/samroad_spacenet/<best.ckpt>" \
    --output_dir my_run

# 2. 评估 (用上一步固定的目录名, 不必抄 timestamp)
cd metrics
python eval.py --dataset spacenet --dir save/my_run --workers 16

# 3. 看结果
cat ../save/my_run/results/apls.json | python -m json.tool | head
cat ../save/my_run/results/topo.json | python -m json.tool | head
```

### 坐标系验证

```bash
# 定量验证 (IoU, ~30秒)
python scripts/quantitative_verify.py

# 可视化验证 (对齐图, ~2分钟)
python scripts/visualize_dataset.py --dataset cityscale
python scripts/visualize_dataset.py --dataset spacenet
```

### 阈值标定（PR 曲线自动选取）

推理时有三个后处理阈值决定最终图的质量，它们都作用在**模型输出的概率**上：

| 阈值 | 作用对象 | 影响 |
|------|---------|------|
| `ITSC_THRESHOLD` | 关键点（交叉口）mask | 节点生成数量，阈值高→节点少→Recall 上限低 |
| `ROAD_THRESHOLD` | 路网 mask | 节点生成数量，同上 |
| `TOPO_THRESHOLD` | TopoNet 边分数 | 边保留数量，阈值高→边少→Recall 低 |

**这三个阈值是数据集相关的**，不能跨数据集共用。SpaceNet 上调出来的 `0.195 / 0.341 / 0.705` 直接用到 DiDi Xian 上会导致 Recall 偏低（Precision 高、Recall 低，典型的"预测过保守"）。

#### 原理：一次 PR 曲线 + argmax F1

阈值标定**不需要**逐个阈值跑全量推理，也**不是**二分查找。机制藏在模型的 `on_test_end`（[models/sam_road.py](models/sam_road.py) / [models/sam_road_completion.py](models/sam_road_completion.py)）：

1. 用 torchmetrics 的 `BinaryPrecisionRecallCurve` 在**验证集 patch 级别**收集所有预测分数 + GT 标签
2. `.compute()` **一次性**返回完整 PR 曲线（所有阈值下的 precision/recall）——内部是对预测分数排序后累积 TP/FP，O(N log N) 一次，不是 O(N×K)
3. 取 `F1 = 2PR/(P+R)` 最大的那个阈值

因为评估在 **256×256 patch 级**（像素/边二分类）而非全图 APLS/TOPO，所以一次验证几分钟就出全部三个阈值。patch 级最优阈值 ≈ 全图最优，原始项目在 SpaceNet 上验证过这套够用（patch 级 0.705 对应全图 TOPO≈0.8）。

#### 用法

**SAM-Road（原始模型）** —— `engine/test.py`：

```bash
cd /home/hanhaoyu/sam_road
CUDA_VISIBLE_DEVICES=0 python -m engine.test \
    --config config/toponet_vitb_256_didi_xian.yaml \
    --checkpoint "checkpoints/samroad_didi_xian/<best.ckpt>"
```

**SAM-Road Completion（补全模型）** —— `engine/test_completion.py`：

```bash
CUDA_VISIBLE_DEVICES=0 python -m engine.test_completion \
    --config config/toponet_vitb_256_didi_xian_completion.yaml \
    --checkpoint "checkpoints/samroad_completion_didi_xian/<best.ckpt>"
```

两者运行结束都会打印三行：

```
======= Finding best thresholds =======
======= keypoint ======
Best threshold 0.1949462890625, P=0.3438 R=0.3268 F1=0.3351   → ITSC_THRESHOLD
======= road ======
Best threshold 0.3408203125, P=0.6585 R=0.7146 F1=0.6854      → ROAD_THRESHOLD
======= topo ======
Best threshold 0.705078125, P=0.9747 R=0.9701 F1=0.9724       → TOPO_THRESHOLD
```

把这三个值抄进对应 config 的 `ITSC_THRESHOLD / ROAD_THRESHOLD / TOPO_THRESHOLD`，重新推理评估即可。config 文件里 `# Best threshold ...` 形式的注释就是这么来的（每个数据集标定一次）。

> 注：patch 级 PR 曲线选出的阈值是像素/边二分类意义下的最优，与全图 APLS/TOPO 正相关但不完全等价。若标定后全图指标仍不理想，可在该阈值附近手动微调（通常 ±0.1 内）观察 APLS/TOPO 变化。

---

## 预训练权重

官方权重: [congrui/sam_road](https://huggingface.co/congrui/sam_road)

---

## 引用

```bibtex
@article{hetang2024segment,
  title={Segment Anything Model for Road Network Graph Extraction},
  author={Hetang, Congrui and Xue, Haoru and Le, Cindy and Yue, Tianwei and Wang, Wenping and He, Yihui},
  journal={arXiv preprint arXiv:2403.16051},
  year={2024}
}
```

## 致谢

- [Segment Anything Model](https://github.com/facebookresearch/segment-anything)
- [RNGDet++](https://github.com/TonyXuQAQ/RNGDetPlusPlus)
- SAMed, Detectron2
