"""
generate_traj.py — 把 DelvMap 的真实轨迹二值图 (rawdata/traj_heat.png) 对齐进每个
SamRoad tile，产出 region_{c}_traj.png。

为什么需要这个脚本
-------------------
SamRoad tile 内部是 WGS84 线性像素坐标系 (~1.0 m/px, 400px=400m)；DelvMap 的 traj_heat.png
是一张 Web Mercator (EPSG:3857) 大图 (5625x6610, ~1.493 m/px)。两者投影、m/px、尺寸都不同，
不能简单 cv2.resize / 仿射拉伸 —— 纬度方向 Mercator 比 WGS84 拉长约 1/cos(34.24°)≈1.21×，
直接 resize 会在 tile 边缘产生 ~30px 的系统性 N-S 错位。

解法：按经纬度逐像素重映射。对每个 tile 的每个像素：
  tile 像素 (px,py) -> 反推 lat/lon (线性, 与 SamRoad graph2RegionCoordinate 同坐标系)
                    -> 转 Web Mercator (复用 DelvMap wgs84_to_mercator, 向量化)
                    -> 落到 DelvMap 大图像素 (geo_to_pixel 同公式)
                    -> 最近邻采样 (保二值)
这样投影差异被精确吸收，与 DelvMap 自身 label 逐像素同公式对齐。

用法
----
    python tools/prepare_dataset/generate_traj.py \
        --config tools/prepare_dataset/config/xian.json \
        --traj-png /Users/highee/research/DelvMap/rawdata/traj_heat.png \
        --out-dir datasets/didi/xian/2019_400 \
        --mode traj

前置：必须先用同一份 xian.json (size=400, DelvMap bbox) 跑过 prepare_dataset 生成 region_*_sat.png。
脚本会用“region 数量 == lat_n*lon_n”做硬断言，防止跑在旧数据集上导致 region 号错配。

依赖：numpy, opencv (cv2)。仅这两者 + 标准库。
"""

import argparse
import json
import math
import os
from pathlib import Path

import cv2
import numpy as np


# ============================================================
# DelvMap 投影函数 (照搬 DelvMap/dataset/prepare_dataset.py:62-84, 向量化)
# 原样复用是为了和 DelvMap 自身 label/traj 逐像素同公式, 保证对齐
# ============================================================
# Web Mercator (EPSG:3857) 的"伪墨卡托"参数, 与 DelvMap 完全一致
MERC_R = 20037508.34


def wgs84_to_mercator(lon, lat):
    """WGS84 经纬度 -> Web Mercator 平面坐标。

    支持标量或 numpy 数组。输入单位: 度。
    """
    x = lon * MERC_R / 180.0
    y_deg = np.log(np.tan((90.0 + lat) * np.pi / 360.0)) / (np.pi / 180.0)
    y = y_deg * MERC_R / 180.0
    return x, y


# ============================================================
# SamRoad 切图公式 (照搬 download_use_osm.py:122-161, 保证 region 号一致)
# ============================================================
def compute_region_bboxes(cfg):
    """从 xian.json 单条配置复算每个 region 的 bbox。

    返回 list[(c, [lat_st, lon_st, lat_ed, lon_ed])]，c = i*lon_n + j, 行优先, 西南角起。
    必须与 download_use_osm.py 逐字一致 (含 norm(.,7) 舍入、cos(lat_min) 常量)。
    """
    size = cfg["size"]
    lat_origin, lon_origin = cfg["lat_min"], cfg["lon_min"]   # 与 download_use_osm.py:148 一致
    lat_min, lon_min = cfg["lat_min"], cfg["lon_min"]
    lat_max, lon_max = cfg["lat_max"], cfg["lon_max"]

    dlat = size / 111111.0
    dlon = size / (111111.0 * math.cos(math.radians(lat_min)))
    lat_n = math.ceil((lat_max - lat_min) / dlat)
    lon_n = math.ceil((lon_max - lon_min) / dlon)

    def norm(x, nd=7):
        return round(x, nd)

    # 与 download_use_osm.py 切片循环逐字一致: 编号从左上角(NW)开始, 行优先 TL->BR.
    #   i=0 = 最北行 (lat_ed 接近 lat_max), i 增大向南
    #   j=0 = 最西列, j 增大向东
    # bbox 仍是 [lat_st(南), lon_st(西), lat_ed(北), lon_ed(东)].
    regions = []
    c = 0
    for i in range(lat_n):
        for j in range(lon_n):
            lat_ed = norm(lat_max - size / 111111.0 * i)
            lat_st = norm(lat_max - size / 111111.0 * (i + 1))
            lon_st = norm(lon_origin + size / 111111.0 * j / math.cos(math.radians(lat_origin)))
            lon_ed = norm(lon_origin + size / 111111.0 * (j + 1) / math.cos(math.radians(lat_origin)))
            regions.append((c, [lat_st, lon_st, lat_ed, lon_ed]))
            c += 1
    return regions, lat_n, lon_n


# ============================================================
# 核心: 把 traj 大图按经纬度重映射进单个 tile
# ============================================================
def render_tile_traj(big_traj, bbox, size, merc_bounds, mode):
    """对单个 region 生成对齐的 traj 二值图。

    big_traj: (H, W) uint8, DelvMap traj_heat.png 灰度图
    bbox: [lat_st, lon_st, lat_ed, lon_ed]
    size: tile 边长像素 (400)
    merc_bounds: (x_min, y_min, x_max, y_max) 大图四角 Mercator 坐标
    mode: 'traj' (3x3 闭运算, 默认) | 'point' (仅二值化)
    返回: (size, size) uint8, {0,255}
    """
    lat_st, lon_st, lat_ed, lon_ed = bbox
    x_min, y_min, x_max, y_max = merc_bounds
    img_w = big_traj.shape[1]
    img_h = big_traj.shape[0]

    # tile 像素 (px, py) -> lat/lon (线性, 与 graph2RegionCoordinate 逆映射一致)
    # graph2RegionCoordinate: raw_x=(lon-lon_st)/(lon_ed-lon_st)*size, raw_y=(lat_ed-lat)/(lat_ed-lat_st)*size
    # 故 lat=lat_ed-(py/size)*(lat_ed-lat_st), lon=lon_st+(px/size)*(lon_ed-lon_st), 北在 py=0, y 向下
    coords = np.arange(size, dtype=np.float64)
    px, py = np.meshgrid(coords, coords)  # (size, size); py[r,c]=r, px[r,c]=c
    lat = lat_ed - (py / size) * (lat_ed - lat_st)
    lon = lon_st + (px / size) * (lon_ed - lon_st)

    # WGS84 -> Web Mercator (向量化)
    x_m, y_m = wgs84_to_mercator(lon, lat)

    # Mercator -> DelvMap 大图像素 (与 geo_to_pixel 同公式)
    bx = (x_m - x_min) / (x_max - x_min) * img_w
    by = (y_max - y_m) / (y_max - y_min) * img_h  # y 向下, 北在顶

    # 最近邻采样 (保二值, 禁双线性)
    ix = np.rint(bx).astype(np.int64)
    iy = np.rint(by).astype(np.int64)
    # 边缘 tile 外溢像素: 补黑(置0), 不夹到边界行/列 (避免出现奇怪横竖道道)
    in_range = (ix >= 0) & (ix < img_w) & (iy >= 0) & (iy < img_h)
    ix_c = np.clip(ix, 0, img_w - 1)
    iy_c = np.clip(iy, 0, img_h - 1)
    tile = big_traj[iy_c, ix_c]  # (size, size) uint8
    tile = np.where(in_range, tile, 0)  # 外溢像素置 0
    n_clip = int((~in_range).sum())  # 外溢像素数, 返回给调用方做警告

    # 二值化 (>0->255, 含 128->255), 等同 DelvMap binaryzation
    tile = np.where(tile > 0, 255, 0).astype(np.uint8)

    # 可选: 3x3 闭运算连接点 (等同 DelvMap traj 变体)
    if mode == "traj":
        kernel = np.ones((3, 3), np.uint8)
        tile = cv2.morphologyEx(tile, cv2.MORPH_CLOSE, kernel)

    return tile, n_clip


# ============================================================
# QC
# ============================================================
def run_qc(regions, big_traj, merc_bounds, out_dir, size, cfg, suffix=""):
    """生成后 QC: 全局 extent 闸 + 抽样叠加图 + IoU/质心统计。"""
    qc_dir = out_dir / f"qc_traj{suffix}"
    qc_dir.mkdir(parents=True, exist_ok=True)
    img_w = big_traj.shape[1]
    img_h = big_traj.shape[0]
    x_min, y_min, x_max, y_max = merc_bounds

    # ---- 1. 全局 extent 闸: 城市四角应精确落到大图四角 ----
    # region 编号 NW 角起, 行优先 TL->BR (region 0 = NW-corner tile, region 末 = SE-corner tile).
    # 直接用城市四角 (lat_min/lat_max/lon_min/lon_max) 投影校验, 与编号无关.
    mx, my = wgs84_to_mercator(cfg["lon_min"], cfg["lat_min"])  # SW
    px_sw = (mx - x_min) / (x_max - x_min) * img_w
    py_sw = (y_max - my) / (y_max - y_min) * img_h
    print(f"[QC] SW 城市角 (lat_min,lon_min) -> ({px_sw:.1f}, {py_sw:.1f}), 期望 ~(0, {img_h-1}) 左下")
    mx, my = wgs84_to_mercator(cfg["lon_min"], cfg["lat_max"])  # NW
    px_nw = (mx - x_min) / (x_max - x_min) * img_w
    py_nw = (y_max - my) / (y_max - y_min) * img_h
    print(f"[QC] NW 城市角 (lat_max,lon_min) -> ({px_nw:.1f}, {py_nw:.1f}), 期望 ~(0, 0) 左上")
    mx, my = wgs84_to_mercator(cfg["lon_max"], cfg["lat_max"])  # NE
    px_ne = (mx - x_min) / (x_max - x_min) * img_w
    py_ne = (y_max - my) / (y_max - y_min) * img_h
    print(f"[QC] NE 城市角 (lat_max,lon_max) -> ({px_ne:.1f}, {py_ne:.1f}), 期望 ~({img_w-1}, 0) 右上")
    ok = abs(px_sw) < 5 and abs(py_sw - (img_h - 1)) < 5 and abs(px_nw) < 5 and abs(py_nw) < 5 and abs(px_ne - (img_w - 1)) < 5
    if not ok:
        print("  [!!] 城市四角未精确落到大图四角 -> Mercator 或 y 朝向可能错, 请先检查再信任结果!")
    else:
        print("  [OK] 城市四角精确对齐大图四角, 投影/朝向正确")

    # ---- 2/3/4. 抽样叠加 + IoU + 质心 ----
    sample_cs = sorted({0, len(regions) // 2, len(regions) - 1})
    ious = []
    print("[QC] 抽样统计 (IoU(traj, road_mask), 质心偏移px):")
    for c in sample_cs:
        traj_path = out_dir / f"region_{c}_traj{suffix}.png"
        gt_path = out_dir / f"region_{c}_gt.png"
        sat_path = out_dir / f"region_{c}_sat.png"
        if not traj_path.exists():
            continue
        traj = cv2.imread(str(traj_path), cv2.IMREAD_GRAYSCALE)
        traj_b = (traj > 0).astype(np.uint8)

        # 叠加图: sat 暗底 + gt 路网绿 + traj 红
        if sat_path.exists():
            sat = cv2.imread(str(sat_path))
            sat = cv2.resize(sat, (size, size)) if sat.shape[:2] != (size, size) else sat
            overlay = (sat.astype(np.float32) * 0.5).astype(np.uint8)
        else:
            overlay = np.zeros((size, size, 3), np.uint8)
        if gt_path.exists():
            gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
            gt = cv2.resize(gt, (size, size)) if gt.shape[:2] != (size, size) else gt
            gt_b = (gt > 0).astype(np.uint8)
            overlay[gt_b > 0] = [0, 200, 0]  # 绿
            inter = (traj_b > 0) & (gt_b > 0)
            union = (traj_b > 0) | (gt_b > 0)
            iou = inter.sum() / max(union.sum(), 1)
            ious.append(iou)
            # 质心偏移
            def centroid(m):
                ys, xs = np.nonzero(m)
                return (xs.mean(), ys.mean()) if len(xs) else (None, None)
            tcx, tcy = centroid(traj_b)
            gcx, gcy = centroid(gt_b)
            cdist = math.hypot(tcx - gcx, tcy - gcy) if (tcx is not None and gcx is not None) else float('nan')
            print(f"  region {c}: IoU={iou:.3f}, 质心偏移={cdist:.1f}px")
        overlay[traj_b > 0, 2] = 255  # 红通道 (BGR)
        cv2.imwrite(str(qc_dir / f"overlay_region_{c}.png"), overlay)
    if ious:
        print(f"[QC] 抽样 IoU 中位数={np.median(ious):.3f} (期望 ~0.2-0.6; 0=投影失败信号)")

    # ---- 5. 全零 tile 与取值检查 ----
    n_zero, n_badval = 0, 0
    for c, _ in regions:
        p = out_dir / f"region_{c}_traj{suffix}.png"
        if not p.exists():
            continue
        a = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if not np.any(a > 0):
            n_zero += 1
        if not set(np.unique(a).tolist()).issubset({0, 255}):
            n_badval += 1
    print(f"[QC] 全零 tile={n_zero}/{len(regions)} (期望≈0); 取值非{{0,255}}的 tile={n_badval} (期望0)")
    print(f"[QC] 叠加图已写到 {qc_dir}/")


# ============================================================
# main
# ============================================================
def main():
    ap = argparse.ArgumentParser(description="生成对齐 DelvMap traj 的 region_{c}_traj.png")
    ap.add_argument("--config", required=True, help="xian.json (与 prepare_dataset 同一份)")
    ap.add_argument("--traj-png", required=True, help="DelvMap rawdata/traj_heat.png")
    ap.add_argument("--out-dir", required=True, help="region_*_sat.png 所在目录")
    ap.add_argument("--mode", choices=["traj", "point"], default="traj",
                    help="traj=3x3闭运算(默认,更像路网); point=仅二值化(=DelvMap trajpoint)")
    ap.add_argument("--suffix", default="",
                    help="输出文件名后缀: region_{c}_traj{suffix}.png (默认空=region_{c}_traj.png). "
                         "例如 --suffix _line 生成 region_{c}_traj_line.png, 不覆盖 point 版")
    ap.add_argument("--delvmap-lat-min", type=float, default=34.206385)
    ap.add_argument("--delvmap-lat-max", type=float, default=34.279658)
    ap.add_argument("--delvmap-lon-min", type=float, default=108.917423)
    ap.add_argument("--delvmap-lon-max", type=float, default=108.99286)
    ap.add_argument("--delvmap-img-w", type=int, default=5625)
    ap.add_argument("--delvmap-img-h", type=int, default=6610)
    ap.add_argument("--qc", action="store_true", help="生成后跑 QC (叠加图/IoU/extent闸)")
    args = ap.parse_args()

    # ---- 读 xian.json ----
    cfg_list = json.load(open(args.config, "r"))
    assert len(cfg_list) == 1, "目前只支持单条配置 (xian)"
    cfg = cfg_list[0]
    size = cfg["size"]
    regions, lat_n, lon_n = compute_region_bboxes(cfg)
    print(f"[INFO] xian.json: size={size}, 网格 {lat_n}x{lon_n}={lat_n*lon_n} 块, bbox=lat[{cfg['lat_min']},{cfg['lat_max']}] lon[{cfg['lon_min']},{cfg['lon_max']}]")

    # ---- 硬性 sanity 闸: region_*_sat.png 数量必须 == lat_n*lon_n ----
    out_dir = Path(args.out_dir)
    sat_files = sorted(out_dir.glob("region_*_sat.png"))
    expected = lat_n * lon_n
    if len(sat_files) != expected:
        raise SystemExit(
            f"[FATAL] out_dir 下 region_*_sat.png 数量={len(sat_files)} != 期望 {expected} (lat_n*lon_n).\n"
            f"        这说明 out_dir 是用另一份 xian.json (旧 bbox/size) 跑的 prepare_dataset 产物.\n"
            f"        请先用本脚本 --config 指向的 xian.json 重新跑 download_use_osm.py, 再运行本脚本.\n"
            f"        (R1: 跑在旧数据上会 region 号错配 -> 静默错位)"
        )
    print(f"[INFO] sanity 闸通过: {len(sat_files)} sat.png == {expected}")

    # ---- 一次性加载 traj 大图 + 预算 Mercator 边界 ----
    big_traj = cv2.imread(args.traj_png, cv2.IMREAD_GRAYSCALE)
    if big_traj is None:
        raise SystemExit(f"[FATAL] 读不到 traj 大图: {args.traj_png}")
    # 校验大图尺寸与传入常量一致
    if big_traj.shape != (args.delvmap_img_h, args.delvmap_img_w):
        print(f"[WARN] traj 大图 shape={big_traj.shape} != 期望 ({args.delvmap_img_h},{args.delvmap_img_w}), 按实际尺寸处理")
    print(f"[INFO] traj 大图: {big_traj.shape[1]}x{big_traj.shape[0]}, unique={np.unique(big_traj)[:5]}")

    x_min, y_min = wgs84_to_mercator(args.delvmap_lon_min, args.delvmap_lat_min)
    x_max, y_max = wgs84_to_mercator(args.delvmap_lon_max, args.delvmap_lat_max)
    merc_bounds = (x_min, y_min, x_max, y_max)

    # ---- 逐 tile 生成 ----
    n_warn_clip = 0
    for idx, (c, bbox) in enumerate(regions):
        tile, n_clip = render_tile_traj(big_traj, bbox, size, merc_bounds, args.mode)
        # 边缘 tile 外溢警告 (>1% 像素补黑)
        if n_clip > 0.01 * size * size:
            n_warn_clip += 1
        cv2.imwrite(str(out_dir / f"region_{c}_traj{args.suffix}.png"), tile)
        if (idx + 1) % 50 == 0 or idx == 0:
            print(f"[INFO] {idx+1}/{len(regions)} region_{c}_traj{args.suffix}.png done (clip_px={n_clip})")

    print(f"[INFO] 完成: {len(regions)} 个 region_{c}_traj{args.suffix}.png 写入 {out_dir}/ (mode={args.mode})")
    print(f"[INFO] 边缘外溢警告 tile 数: {n_warn_clip}")

    if args.qc:
        run_qc(regions, big_traj, merc_bounds, out_dir, size, cfg, suffix=args.suffix)


if __name__ == "__main__":
    main()
