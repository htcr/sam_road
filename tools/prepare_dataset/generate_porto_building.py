"""
generate_porto_building.py — 从裁剪后的 PBF 渲染 Porto building 二值大图,
与 sat 同尺寸同投影 (Web Mercator EPSG:3857), 不裁切 tile (只出大图)。

why: download_use_osm.py 不生成 building, Porto 需要 building 大图作为附加模态。
     只出大图 (与 sat/traj 同 bbox 同尺寸), 后续如需 tile 可再用 generate_traj.py
     同款逐像素重映射裁切, 但本任务只要大图。

how: 从 porto_1km.osm.pbf 读 building way (tag 含 'building'), 用 DelvMap 同款
     Web Mercator 像素映射 (wgs84_to_mercator + 线性插值) fillPoly 到大图,
     二值化 {0,255}。投影与 sat/traj 完全一致, 逐像素对齐。

用法:
    python tools/prepare_dataset/generate_porto_building.py \
        --pbf rawdata/porto/porto_1km.osm.pbf \
        --sat rawdata/porto/porto_2014_r3026_z17.png \
        --lat-min 41.1125 --lat-max 41.2385 \
        --lon-min -8.6875 --lon-max -8.5495 \
        --out rawdata/porto/building_heat.png

依赖: pyosmium, numpy, opencv (samroad env)
"""
import argparse
import math
import os

import cv2
import numpy as np
import osmium


# Web Mercator (与 DelvMap / generate_porto_traj.py 完全一致)
MERC_R = 20037508.34


def wgs84_to_mercator(lon, lat):
    x = lon * MERC_R / 180.0
    y_deg = math.log(math.tan((90.0 + lat) * math.pi / 360.0)) / (math.pi / 180.0)
    y = y_deg * MERC_R / 180.0
    return x, y


class BuildingExtractor(osmium.SimpleHandler):
    """从 PBF 提取 building way (node_ref -> (lat,lon))。"""
    def __init__(self):
        super().__init__()
        self.nodes = {}        # node_id -> (lat, lon)
        self.buildings = []    # list of [node_id, ...]

    def node(self, n):
        if n.location.valid():
            self.nodes[n.id] = (n.location.lat, n.location.lon)

    def way(self, w):
        tags = dict(w.tags)
        if 'building' in tags and len(w.nodes) >= 3:
            self.buildings.append([nr.ref for nr in w.nodes])


def render_buildings(buildings, node_map, lat_min, lat_max, lon_min, lon_max, img_w, img_h):
    """渲染 building 为二值大图 (Mercator, 与 sat 同尺寸)。"""
    x_min, y_min = wgs84_to_mercator(lon_min, lat_min)
    x_max, y_max = wgs84_to_mercator(lon_max, lat_max)
    img = np.zeros((img_h, img_w), dtype=np.uint8)

    valid = 0
    for node_refs in buildings:
        poly = []
        ok = True
        for ref in node_refs:
            loc = node_map.get(ref)
            if loc is None:
                ok = False
                break
            poly.append(loc)
        if not ok or len(poly) < 3:
            continue
        valid += 1
        # 经纬度 -> Mercator -> 像素 (北在顶, y 向下)
        px_pts = []
        for (lat, lon) in poly:
            xm, ym = wgs84_to_mercator(lon, lat)
            px = (xm - x_min) / (x_max - x_min) * img_w
            py = (y_max - ym) / (y_max - y_min) * img_h
            px_pts.append([int(round(px)), int(round(py))])
        pts = np.array(px_pts, np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(img, [pts], 255)
    print(f"[INFO] 有效建筑: {valid} / {len(buildings)}")
    return img


def main():
    ap = argparse.ArgumentParser(description="Porto building 二值大图 (Mercator, 与 sat 同尺寸)")
    ap.add_argument("--pbf", required=True, help="裁剪后的 PBF (porto_1km.osm.pbf)")
    ap.add_argument("--sat", default=None, help="sat 大图 (读尺寸让 building 同尺寸对齐)")
    ap.add_argument("--lat-min", type=float, default=41.1125)
    ap.add_argument("--lat-max", type=float, default=41.2385)
    ap.add_argument("--lon-min", type=float, default=-8.6875)
    ap.add_argument("--lon-max", type=float, default=-8.5495)
    ap.add_argument("--mpp", type=float, default=1.194, help="无 --sat 时按此 m/px 自算 (Mercator z17)")
    ap.add_argument("--out", default="rawdata/porto/building_heat.png")
    args = ap.parse_args()

    lat_min, lat_max = args.lat_min, args.lat_max
    lon_min, lon_max = args.lon_min, args.lon_max

    # 尺寸: 优先读 sat, 否则按 mpp 自算
    if args.sat and os.path.isfile(args.sat):
        sat = cv2.imread(args.sat, cv2.IMREAD_UNCHANGED)
        if sat is None:
            raise SystemExit(f"[FATAL] 读不到 sat: {args.sat}")
        img_h, img_w = sat.shape[0], sat.shape[1]
        print(f"[INFO] 读 sat 尺寸: {img_w} x {img_h}")
    else:
        x_min, y_min = wgs84_to_mercator(lon_min, lat_min)
        x_max, y_max = wgs84_to_mercator(lon_max, lat_max)
        img_w = int(round((x_max - x_min) / args.mpp))
        img_h = int(round((y_max - y_min) / args.mpp))
        print(f"[INFO] 无 --sat, 按 mpp={args.mpp} 自算: {img_w} x {img_h}")

    print(f"[INFO] bbox: lat[{lat_min},{lat_max}] lon[{lon_min},{lon_max}]")
    print(f"[INFO] 解析 PBF: {args.pbf}")
    h = BuildingExtractor()
    h.apply_file(args.pbf, locations=True)
    print(f"[INFO] PBF 节点: {len(h.nodes)}, building way: {len(h.buildings)}")

    img = render_buildings(h.buildings, h.nodes, lat_min, lat_max, lon_min, lon_max, img_w, img_h)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    cv2.imwrite(args.out, img)
    print(f"[INFO] 已写 building 大图: {args.out} ({img_w}x{img_h}) unique={np.unique(img).tolist()}")
    print(f"[INFO] 非零像素: {100.0*(img>0).mean():.2f}%")

    # 缩略图 (AREA 插值, 避免查看器混叠)
    tw, th = min(1024, img_w), min(1024, img_h)
    thumb = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
    thumb_path = args.out.replace('.png', '_thumb.png')
    cv2.imwrite(thumb_path, thumb)
    print(f"[INFO] 缩略图: {thumb_path}")


if __name__ == "__main__":
    main()
