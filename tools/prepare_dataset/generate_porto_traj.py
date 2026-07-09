"""
generate_porto_traj.py — 把 sam_road/datasets/porto/train.csv 的全部 GPS 轨迹
栅格化为两张与 Porto sat 影像同尺寸同投影(Web Mercator)的纯二值 PNG:
  traj_point_heat.png — 轨迹点图 (每个 GPS 点 = 单像素 255)
  traj_line_heat.png  — 轨迹段图 (相邻点连线 = 255)
两张图都纯二值 {0, 255}, 不混用 128/255。

设计 (与 sat 严格对齐)
----------------------
sat 影像由 esri_wayback.GetMapInRect 下载: ESRI z17 瓦片(球面 Web Mercator,
EPSG:3857)拼接后裁到精确 bbox, 输出 (H, W, 3), 不 resize, 保持原始 m/px。
所以 traj PNG 必须用同一套 Web Mercator 投影 + 同一 bbox + 同一画布尺寸,
才能与 sat 逐像素对齐 (无投影残差、无重采样)。

投影链 (与 DelvMap prepare_dataset.py / generate_traj.py 完全一致):
  WGS84 (lon,lat) -> Web Mercator (x_m, y_m)  [wgs84_to_mercator, MERC_R=20037508.34]
                  -> 大图像素 (px, py)         [线性映射, 北在顶 y 向下]
      px = (x_m - x_min) / (x_max - x_min) * img_w
      py = (y_max - y_m) / (y_max - y_min) * img_h
其中 (x_min,y_min,x_max,y_max) 是 bbox 四角的 Mercator 坐标。
ESRI 瓦片用的也是 EPSG:3857 球面墨卡托 (R=6378137, 与 MERC_R=6378137*pi/2 等价),
故 DelvMap 公式与 ESRI 瓦片同投影, traj 与 sat 逐像素对齐。

画法
----
- 点图: 每个 bbox 内 GPS 点 -> cv2.circle 半径0 -> 值 255
- 线图: 相邻点距离在 [min_seg_m, max_seg_m] 内才连线 (过滤飞点/静止) ->
        cv2.line thickness=1 LINE_8 (无 AA) -> 值 255
两张图分别画到独立 uint8 数组, 纯二值 {0,255}。

输出
----
1. traj_point_heat.png         — (H, W) uint8 {0,255}, 轨迹点 (原始单像素, 保真)
2. traj_point_heat_closed.png  — 同上经 3x3 闭运算, 填补 GPS 量化导致的横向空行条纹
3. traj_line_heat.png          — (H, W) uint8 {0,255}, 轨迹段
4. porto_bbox.json             — 经纬度范围 + 像素尺寸 + m/px + 投影信息
5. config/porto.json           — sam_road 数据集配置 (size=400, NW-first 网格)
6. QC: 像素分布/空行数/覆盖比/四角校验/与 sat 尺寸比对 + point/point_closed/line/overlay 缩略图

注: Porto 出租车 GPS 纬度被量化到 0.0001°(~11m), 在 1.194m/px 下原始点图每 ~9px
出现一条空行 (横向条纹, 非查看器混叠, 是源数据特性)。traj_point_heat 保留原始;
traj_point_heat_closed 用 3x3 闭运算填补, 供可视化/训练。traj_line_heat 因连线跨过
空行, 无此条纹。

用法 (sat 已由 porto_rsimg_test.py --full-bbox 下载好之后)
----
    python tools/prepare_dataset/generate_porto_traj.py \\
        --csv /Users/highee/research/sam_road/datasets/porto/train.csv \\
        --sat /Users/highee/research/sam_road/tools/prepare_dataset/porto_2014_r3026_z17.png \\
        --lat-min 41.1125 --lat-max 41.2385 \\
        --lon-min -8.6875 --lon-max -8.5495 \\
        --out-dir /Users/highee/research/sam_road/datasets/porto

若不给 --sat, 则用 bbox + mpp 自算尺寸 (Mercator, m/px 两轴一致)。

依赖: numpy, opencv (cv2), tqdm。仅这三者 + 标准库。
"""
import argparse
import csv
import json
import math
import os
import ast

import cv2
import numpy as np
from tqdm import tqdm


# ============================================================
# Web Mercator (EPSG:3857), 与 DelvMap prepare_dataset.py 完全一致
# ESRI 瓦片也是 EPSG:3857 球面墨卡托, 故同投影
# ============================================================
MERC_R = 20037508.34


def wgs84_to_mercator(lon, lat):
    """WGS84 经纬度 -> Web Mercator 平面坐标。支持标量或 numpy 数组。"""
    x = lon * MERC_R / 180.0
    y_deg = np.log(np.tan((90.0 + lat) * np.pi / 360.0)) / (np.pi / 180.0)
    y = y_deg * MERC_R / 180.0
    return x, y


def merc_bounds(lat_min, lat_max, lon_min, lon_max):
    """bbox 四角的 Mercator 坐标 (x_min, y_min, x_max, y_max)。"""
    x_min, y_min = wgs84_to_mercator(lon_min, lat_min)
    x_max, y_max = wgs84_to_mercator(lon_max, lat_max)
    return x_min, y_min, x_max, y_max


def geo_to_pixel(lat, lon, x_min, y_min, x_max, y_max, img_w, img_h):
    """Mercator -> 大图像素 (北在顶, y 向下)。与 DelvMap geo_to_pixel 同公式。"""
    x_m, y_m = wgs84_to_mercator(lon, lat)
    px = (x_m - x_min) / (x_max - x_min) * img_w
    py = (y_max - y_m) / (y_max - y_min) * img_h
    return px, py  # 浮点, 调用方按需 round/clip


# ============================================================
# 主栅格化: 输出两张纯二值 (0/255) 大图 — 点图 + 线图
# ============================================================
def render_traj_heat(csv_path, lat_min, lat_max, lon_min, lon_max,
                     img_w, img_h, max_seg_m=300, min_seg_m=5):
    """流式读 train.csv, 产出两张独立 uint8 大图 (Mercator, 与 sat 同尺寸同投影):

      img_point — 轨迹点图: 每个落在 bbox 内的 GPS 点画成单像素 255
      img_line  — 轨迹段图: 相邻点连线 (距离在 [min_seg_m, max_seg_m] 内) 画成 255

    两张图都纯二值 {0, 255}, 不混用 128/255。越界点丢弃; 跨边界线段用
    cv2.clipLine 裁掉图外部分。LINE_8 (无 AA) 避免抗锯齿产生中间灰度。

    返回 (img_point, img_line, n_traj, n_pts, n_pts_in, n_traj_in)。
    """
    img_point = np.zeros((img_h, img_w), dtype=np.uint8)
    img_line = np.zeros((img_h, img_w), dtype=np.uint8)
    x_min, y_min, x_max, y_max = merc_bounds(lat_min, lat_max, lon_min, lon_max)

    LAT_PER_M = 111111.0
    cos_lat = math.cos(math.radians((lat_min + lat_max) / 2.0))

    n_traj = 0
    n_pts = 0
    n_pts_in = 0
    n_traj_in = 0

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, desc="rasterizing", unit="traj"):
            poly = row.get('POLYLINE', '')
            if not poly or poly == '[]':
                continue
            try:
                pts = ast.literal_eval(poly)
            except Exception:
                continue
            if not pts:
                continue
            n_traj += 1
            traj_has_in = False
            pxs, pys, ins = [], [], []
            for lonlat in pts:
                lon, lat = lonlat[0], lonlat[1]
                n_pts += 1
                if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
                    pxf, pyf = geo_to_pixel(lat, lon, x_min, y_min, x_max, y_max, img_w, img_h)
                    x = int(round(pxf)); y = int(round(pyf))
                    x = 0 if x < 0 else (img_w - 1 if x >= img_w else x)
                    y = 0 if y < 0 else (img_h - 1 if y >= img_h else y)
                    pxs.append(x); pys.append(y); ins.append(True)
                    n_pts_in += 1
                    traj_has_in = True
                else:
                    pxs.append(None); pys.append(None); ins.append(False)
            if traj_has_in:
                n_traj_in += 1
            # 点图: 每个在图内的 GPS 点画单像素 255
            for i in range(len(pxs)):
                if ins[i]:
                    cv2.circle(img_point, (pxs[i], pys[i]), 0, 255, -1)
            # 线图: 相邻点距离过滤 + 跨边界裁剪。LINE_8 (无 AA) 保纯二值。
            for i in range(len(pts) - 1):
                lon1, lat1 = pts[i][0], pts[i][1]
                lon2, lat2 = pts[i + 1][0], pts[i + 1][1]
                dlat_m = (lat2 - lat1) * LAT_PER_M
                dlon_m = (lon2 - lon1) * LAT_PER_M * cos_lat
                dist_m = math.hypot(dlat_m, dlon_m)
                if not (min_seg_m < dist_m < max_seg_m):
                    continue
                if not (ins[i] or ins[i + 1]):
                    continue
                p1f = geo_to_pixel(lat1, lon1, x_min, y_min, x_max, y_max, img_w, img_h)
                p2f = geo_to_pixel(lat2, lon2, x_min, y_min, x_max, y_max, img_w, img_h)
                p1 = (int(round(p1f[0])), int(round(p1f[1])))
                p2 = (int(round(p2f[0])), int(round(p2f[1])))
                # cv2.clipLine rect = (x, y, width, height) = (0, 0, img_w, img_h)
                ok, p1c, p2c = cv2.clipLine((0, 0, img_w, img_h), p1, p2)
                if ok:
                    cv2.line(img_line, p1c, p2c, 255, 1, lineType=cv2.LINE_8)
    return img_point, img_line, n_traj, n_pts, n_pts_in, n_traj_in


# ============================================================
# QC (点图原始/闭运算 + 线图分别统计)
# ============================================================
def qc(img_point, img_point_closed, img_line, lat_min, lat_max, lon_min, lon_max,
       img_w, img_h, out_dir, n_traj_in, n_traj, sat_shape):
    print("\n==== QC ====")
    for name, img in [("point(原始)", img_point),
                      ("point(闭运算)", img_point_closed),
                      ("line", img_line)]:
        u, c = np.unique(img, return_counts=True)
        nz = int((img > 0).sum())
        print(f"[{name}] unique={u.tolist()}  非零像素={nz} / {img.size} "
              f"= {100.0 * nz / img.size:.2f}%")
    # 横纹量化校验: 原始点图空行数 vs 闭运算后空行数
    rownz_raw = (img_point > 0).sum(axis=1)
    rownz_cls = (img_point_closed > 0).sum(axis=1)
    empty_raw = int((rownz_raw == 0).sum())
    empty_cls = int((rownz_cls == 0).sum())
    print(f"空行数: 原始={empty_raw} 闭运算后={empty_cls} "
          f"(闭运算应大幅减少; 残余空行多为轨迹本就稀疏的区域)")
    print(f"覆盖轨迹: {n_traj_in} / {n_traj} ({100.0 * n_traj_in / max(n_traj, 1):.1f}%)")
    if sat_shape is not None:
        match = (img_point.shape[0] == sat_shape[0] and img_point.shape[1] == sat_shape[1])
        print(f"与 sat 尺寸比对: traj={img_point.shape[1]}x{img_point.shape[0]} "
              f"sat={sat_shape[1]}x{sat_shape[0]} "
              f"-> {'✓ 一致 (逐像素对齐)' if match else '✗ 不一致! 检查 --sat / bbox'}")
    # 城市四角像素校验 (Mercator)
    x_min, y_min, x_max, y_max = merc_bounds(lat_min, lat_max, lon_min, lon_max)
    def corner(lat, lon):
        pxf, pyf = geo_to_pixel(lat, lon, x_min, y_min, x_max, y_max, img_w, img_h)
        return (int(round(pxf)), int(round(pyf)))
    sw = corner(lat_min, lon_min); nw = corner(lat_max, lon_min)
    ne = corner(lat_max, lon_max); se = corner(lat_min, lon_max)
    print(f"四角像素: SW={sw}(期望~(0,{img_h-1})) NW={nw}(期望~(0,0)) "
          f"NE={ne}(期望~({img_w-1},0)) SE={se}(期望~({img_w-1},{img_h-1}))")
    # 各象限非零分布 (用 line 图, 路网结构更直观)
    h2, w2 = img_h // 2, img_w // 2
    quads = {"NW": img_line[:h2, :w2], "NE": img_line[:h2, w2:],
             "SW": img_line[h2:, :w2], "SE": img_line[h2:, w2:]}
    print("各象限非零像素比 (line 图):")
    for k, q in quads.items():
        print(f"  {k}: {100.0 * (q > 0).sum() / q.size:.2f}%")
    # 缩略图 (point 原始/闭运算 + line + overlay)。用 AREA 插值避免查看器近邻混叠假横纹
    tw, th = min(1024, img_w), min(1024, img_h)
    tp = cv2.resize(img_point, (tw, th), interpolation=cv2.INTER_AREA)
    tpc = cv2.resize(img_point_closed, (tw, th), interpolation=cv2.INTER_AREA)
    tl = cv2.resize(img_line, (tw, th), interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(out_dir, "porto_traj_point_thumb.png"), tp)
    cv2.imwrite(os.path.join(out_dir, "porto_traj_point_closed_thumb.png"), tpc)
    cv2.imwrite(os.path.join(out_dir, "porto_traj_line_thumb.png"), tl)
    # 叠加: line=绿, point(闭运算)=红, 便于肉眼校验路网与点是否吻合
    overlay = np.zeros((th, tw, 3), np.uint8)
    overlay[tl > 0, 1] = 255   # 绿 = line
    overlay[tpc > 0, 2] = 255  # 红 = point
    cv2.imwrite(os.path.join(out_dir, "porto_traj_overlay_thumb.png"), overlay)
    print(f"缩略图: {out_dir}/porto_traj_{{point,point_closed,line,overlay}}_thumb.png")


# ============================================================
# main
# ============================================================
def main():
    ap = argparse.ArgumentParser(description="Porto train.csv -> Mercator 二值 traj point/line PNG (对齐 sat)")
    ap.add_argument("--csv", required=True, help="train.csv 路径")
    ap.add_argument("--sat", default=None, help="已下载的 Porto sat PNG (读其尺寸, 让 traj 同尺寸对齐)")
    ap.add_argument("--lat-min", type=float, default=41.1125)
    ap.add_argument("--lat-max", type=float, default=41.2385)
    ap.add_argument("--lon-min", type=float, default=-8.6875)
    ap.add_argument("--lon-max", type=float, default=-8.5495)
    ap.add_argument("--mpp", type=float, default=1.194, help="无 --sat 时按此 m/px 自算尺寸 (Mercator z17 两轴一致, 与 GetMapInRect 对齐)")
    ap.add_argument("--out-dir", default="/Users/highee/research/sam_road/datasets/porto")
    ap.add_argument("--max-seg-m", type=float, default=300.0)
    ap.add_argument("--min-seg-m", type=float, default=5.0)
    ap.add_argument("--size", type=int, default=400, help="sam_road tile 边长(像素/米), 生成 porto.json 用")
    args = ap.parse_args()

    lat_min, lat_max = args.lat_min, args.lat_max
    lon_min, lon_max = args.lon_min, args.lon_max
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # 尺寸: 优先读 sat, 否则按 mpp 自算 (Mercator, 两轴 m/px 一致)
    sat_shape = None
    if args.sat and os.path.isfile(args.sat):
        sat = cv2.imread(args.sat, cv2.IMREAD_UNCHANGED)
        if sat is None:
            raise SystemExit(f"[FATAL] 读不到 sat: {args.sat}")
        img_h, img_w = sat.shape[0], sat.shape[1]
        sat_shape = sat.shape
        print(f"[INFO] 读 sat 尺寸: {img_w} x {img_h} (来自 {args.sat})")
    else:
        x_min, y_min, x_max, y_max = merc_bounds(lat_min, lat_max, lon_min, lon_max)
        Wm = x_max - x_min
        Hm = y_max - y_min  # Mercator 米 (y 向北增)
        img_w = int(round(Wm / args.mpp))
        img_h = int(round(Hm / args.mpp))
        print(f"[INFO] 无 --sat, 按 mpp={args.mpp} 自算尺寸: {img_w} x {img_h}")

    print(f"[INFO] bbox: lat[{lat_min},{lat_max}] lon[{lon_min},{lon_max}]")
    print(f"[INFO] 投影: Web Mercator (EPSG:3857), 画布 {img_w} x {img_h}")

    img_point, img_line, n_traj, n_pts, n_pts_in, n_traj_in = render_traj_heat(
        args.csv, lat_min, lat_max, lon_min, lon_max,
        img_w, img_h, args.max_seg_m, args.min_seg_m,
    )

    # 点图闭运算: Porto GPS 纬度被量化到 0.0001° (~11m), 在 1.194m/px 下每 ~9px
    # 出现一条空行, 单像素点图呈横向条纹。3x3 闭运算 (MORPH_CLOSE) 填补空行,
    # 消除条纹同时保持纯二值 {0,255}, 供可视化/训练用。原始点图单独保留 (保真)。
    kernel = np.ones((3, 3), np.uint8)
    img_point_closed = cv2.morphologyEx(img_point, cv2.MORPH_CLOSE, kernel)

    out_point = os.path.join(out_dir, "traj_point_heat.png")
    out_point_closed = os.path.join(out_dir, "traj_point_heat_closed.png")
    out_line = os.path.join(out_dir, "traj_line_heat.png")
    cv2.imwrite(out_point, img_point)
    cv2.imwrite(out_point_closed, img_point_closed)
    cv2.imwrite(out_line, img_line)
    print(f"[INFO] 已写轨迹点图(原始): {out_point} ({img_w}x{img_h}) unique={np.unique(img_point).tolist()}")
    print(f"[INFO] 已写轨迹点图(闭运算): {out_point_closed} unique={np.unique(img_point_closed).tolist()} "
          f"非零% {100.0*(img_point_closed>0).mean():.2f} (原始 {100.0*(img_point>0).mean():.2f})")
    print(f"[INFO] 已写轨迹段图: {out_line} ({img_w}x{img_h}) unique={np.unique(img_line).tolist()}")

    qc(img_point, img_point_closed, img_line, lat_min, lat_max, lon_min, lon_max, img_w, img_h,
       out_dir, n_traj_in, n_traj, sat_shape)

    # bbox json
    cos_lat = math.cos(math.radians((lat_min + lat_max) / 2.0))
    Hm_phys = (lat_max - lat_min) * 111111.0
    bbox_path = os.path.join(out_dir, "porto_bbox.json")
    bbox_info = {
        "dataset": "porto",
        "projection": "Web_Mercator_EPSG3857",
        "lat_min": lat_min, "lat_max": lat_max,
        "lon_min": lon_min, "lon_max": lon_max,
        "img_w": img_w, "img_h": img_h,
        "mpp_lat": Hm_phys / img_h,
        "mpp_lon": Hm_phys / img_w,  # Mercator 下与 lat 接近 (此处用物理米近似)
        "cos_lat_mid": cos_lat,
        "lat_mid": (lat_min + lat_max) / 2.0,
        "traj_total": n_traj, "traj_in_bbox": n_traj_in,
        "pts_total": n_pts, "pts_in_bbox": n_pts_in,
        "png_point": out_point,
        "png_point_closed": out_point_closed,
        "png_line": out_line,
        "gps_quantization": "Porto GPS lat quantized to 0.0001deg (~11m); at 1.194m/px yields ~9px empty-row stripes in raw point map; closed variant fills them via 3x3 MORPH_CLOSE",
        "sat": args.sat,
        "pixel_mapping": {
            "mercator": "wgs84_to_mercator (MERC_R=20037508.34, 与 DelvMap/ESRI 同 EPSG:3857)",
            "x": "(x_m - x_min) / (x_max - x_min) * img_w",
            "y": "(y_max - y_m) / (y_max - y_min) * img_h",
            "note": "北在顶, y 向下; 与 DelvMap geo_to_pixel / generate_traj.render_tile_traj 同公式"
        },
    }
    with open(bbox_path, 'w') as f:
        json.dump(bbox_info, f, indent=2)
    print(f"[INFO] bbox 信息: {bbox_path}")

    # porto.json (sam_road 配置, size=400, 与 xian.json 同结构)
    dlat = args.size / 111111.0
    dlon = args.size / (111111.0 * math.cos(math.radians(lat_min)))
    lat_n = math.ceil((lat_max - lat_min) / dlat)
    lon_n = math.ceil((lon_max - lon_min) / dlon)
    porto_cfg = [{
        "region": "porto",
        "year": "2014",
        "size": args.size,
        "lat_min": lat_min, "lon_min": lon_min,
        "lat_max": lat_max, "lon_max": lon_max,
    }]
    cfg_path = os.path.join(os.path.dirname(__file__), "config", "porto.json")
    os.makedirs(os.path.dirname(cfg_path), exist_ok=True)
    with open(cfg_path, 'w') as f:
        json.dump(porto_cfg, f, indent=2)
    print(f"[INFO] sam_road 配置: {cfg_path}")
    print(f"       网格 {lat_n} x {lon_n} = {lat_n * lon_n} tiles (size={args.size}m, NW-first)")


if __name__ == "__main__":
    main()
