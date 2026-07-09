"""
generate_porto_delvmap_bigimgs.py — 从裁剪后的 PBF 渲染 Porto 路网大图 (road_heat.png),
供 DelvMap create_dataset.py 作 basemap + map_label (DelvMap 里两者同源 OSM highway).

why: DelvMap 的 create_dataset.py 需要 basemap(现有路网) + map_label(GT路网) 大图,
     与 sat/traj/building 同尺寸同投影 (Web Mercator)。Porto 已有 building 大图,
     缺 road 大图。本脚本补这个: 从 porto_1km.osm.pbf 渲染 highway 为二值大图。

how: 复用 generate_porto_building.py 的 Mercator 像素映射, 画 highway way 的每条边
     (cv2.line thickness=2, 与 DelvMap road_width=2 一致), 二值化 {0,255}。
     过滤 footway/path/service 等非车行道 (与 DelvMap prepare_dataset.py OSMExtractor 一致)。

用法:
    python tools/prepare_dataset/generate_porto_delvmap_bigimgs.py \
        --pbf rawdata/porto/porto_1km.osm.pbf \
        --sat rawdata/porto/porto_2014_r3026_z17.png \
        --lat-min 41.1125 --lat-max 41.2385 \
        --lon-min -8.6875 --lon-max -8.5495 \
        --out rawdata/porto/road_heat.png

依赖: pyosmium, numpy, opencv (samroad env)
"""
import argparse
import math
import os

import cv2
import numpy as np
import osmium


MERC_R = 20037508.34


def wgs84_to_mercator(lon, lat):
    x = lon * MERC_R / 180.0
    y_deg = math.log(math.tan((90.0 + lat) * math.pi / 360.0)) / (math.pi / 180.0)
    y = y_deg * MERC_R / 180.0
    return x, y


# DelvMap road 大图: 保留所有汽车可通行 highway (与 download_use_osm.py --keep-highway-only 同源,
# 保证 DelvMap 大图与 samroad 数据集 GT 完全一致)。依据 docs/chatgpt-osm类别分析.md。
# 排除: 纯行人路 (pedestrian/footway/steps/path) + services(服务区闭合) +
# construction/proposed/bus_stop/platform/elevator/raceway + 非路 way (building/landuse/...)。
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


class RoadExtractor(osmium.SimpleHandler):
    """从 PBF 提取 highway way (node_ref -> (lat,lon)), 排除纯行人路 + 非路 way。"""
    def __init__(self):
        super().__init__()
        self.nodes = {}
        self.roads = []

    def node(self, n):
        if n.location.valid():
            self.nodes[n.id] = (n.location.lat, n.location.lon)

    def way(self, w):
        tags = dict(w.tags)
        if 'highway' in tags:
            hw = tags['highway']
            # 白名单内车行道, 按 line 画 (闭合环路去首尾重复点, 不填充)
            if hw in _KEEP_HIGHWAY and len(w.nodes) >= 2:
                nodes = list(w.nodes)
                is_closed = len(nodes) >= 4 and nodes[0].ref == nodes[-1].ref
                if is_closed:
                    nodes = nodes[:-1]
                self.roads.append([nr.ref for nr in nodes])


def render_roads(roads, node_map, lat_min, lat_max, lon_min, lon_max, img_w, img_h, thickness=2):
    """渲染 road 为二值大图 (Mercator, 与 sat/building 同尺寸)。"""
    x_min, y_min = wgs84_to_mercator(lon_min, lat_min)
    x_max, y_max = wgs84_to_mercator(lon_max, lat_max)
    img = np.zeros((img_h, img_w), dtype=np.uint8)

    valid = 0
    for node_refs in roads:
        coords = []
        ok = True
        for ref in node_refs:
            loc = node_map.get(ref)
            if loc is None:
                ok = False
                break
            coords.append(loc)
        if not ok or len(coords) < 2:
            continue
        valid += 1
        # 逐段画线
        for i in range(len(coords) - 1):
            lat1, lon1 = coords[i]
            lat2, lon2 = coords[i + 1]
            xm1, ym1 = wgs84_to_mercator(lon1, lat1)
            xm2, ym2 = wgs84_to_mercator(lon2, lat2)
            x1 = int((xm1 - x_min) / (x_max - x_min) * img_w)
            y1 = int((y_max - ym1) / (y_max - y_min) * img_h)
            x2 = int((xm2 - x_min) / (x_max - x_min) * img_w)
            y2 = int((y_max - ym2) / (y_max - y_min) * img_h)
            if (0 <= x1 < img_w and 0 <= y1 < img_h) or (0 <= x2 < img_w and 0 <= y2 < img_h):
                cv2.line(img, (x1, y1), (x2, y2), 255, thickness=thickness, lineType=cv2.LINE_8)
    # 二值化 (LINE_8 无 AA, 但 thickness>1 多线叠加可能有中间值, 统一二值化)
    img = np.where(img > 0, 255, 0).astype(np.uint8)
    print(f"[INFO] 有效道路 way: {valid} / {len(roads)}")
    return img


def main():
    ap = argparse.ArgumentParser(description="Porto road 二值大图 (Mercator, 与 sat 同尺寸, DelvMap basemap/map_label)")
    ap.add_argument("--pbf", required=True)
    ap.add_argument("--sat", default=None, help="sat 大图 (读尺寸对齐)")
    ap.add_argument("--lat-min", type=float, default=41.1125)
    ap.add_argument("--lat-max", type=float, default=41.2385)
    ap.add_argument("--lon-min", type=float, default=-8.6875)
    ap.add_argument("--lon-max", type=float, default=-8.5495)
    ap.add_argument("--mpp", type=float, default=1.194)
    ap.add_argument("--thickness", type=int, default=2, help="线宽, 与 DelvMap road_width=2 一致")
    ap.add_argument("--out", default="rawdata/porto/road_heat.png")
    args = ap.parse_args()

    lat_min, lat_max = args.lat_min, args.lat_max
    lon_min, lon_max = args.lon_min, args.lon_max

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
    h = RoadExtractor()
    h.apply_file(args.pbf, locations=True)
    print(f"[INFO] PBF 节点: {len(h.nodes)}, highway way: {len(h.roads)}")

    img = render_roads(h.roads, h.nodes, lat_min, lat_max, lon_min, lon_max, img_w, img_h, args.thickness)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    cv2.imwrite(args.out, img)
    print(f"[INFO] 已写 road 大图: {args.out} ({img_w}x{img_h}) unique={np.unique(img).tolist()}")
    print(f"[INFO] 非零像素: {100.0*(img>0).mean():.2f}%")

    tw, th = min(1024, img_w), min(1024, img_h)
    thumb = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
    cv2.imwrite(args.out.replace('.png', '_thumb.png'), thumb)
    print(f"[INFO] 缩略图: {args.out.replace('.png', '_thumb.png')}")


if __name__ == "__main__":
    main()
