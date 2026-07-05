#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Xi'an rsimg (卫星影像) 获取 — Esri Wayback 2019 影像, 对齐 DelvMap sat_img.png。

下载逻辑复用 esri_wayback.fetch_wayback_canvas (下载→拼接→缩放→存图)。
本脚本只做: 设定 xian 参数 (bbox/release/zoom/画布) → 调库 → 出 PNG。
默认产物 xian_2019_wayback_645.png (5625x6610), 可直接喂 download_use_osm.py
--sat_source local:<out> (与 DelvMap/rawdata/sat_img.png 同 bbox 同尺寸, drop-in)。

为何要这张图: didi/xian 数据集的 rn/traj 标签是 2019 (xian-plus-190101.osm.pbf),
但 sat_img.png 是 Google 2024, 时间不匹配。本脚本拉 2019 (release 645, 2019-06-26)
的 Esri Wayback 影像, 缩放到同一 5625x6610 画布, 让 sat 与 rn/traj 同年代。

用法:
  # 1. 默认: 西安 2019-06-26 (release 645) -> 5625x6610 PNG
  python xian_rsimg_test.py

  # 2. 存到数据集目录 (drop-in for download_use_osm.py)
  python xian_rsimg_test.py --out ../../datasets/didi/xian/xian_2019_wayback_645.png

  # 3. 指定 release (Wayback app 的 active= 号)
  python xian_rsimg_test.py --release 645

  # 4. 最新影像作对照 (release=None)
  python xian_rsimg_test.py --latest

  # 5. 列出可用 Wayback release (复用 porto_rsimg_test.list_releases)
  python xian_rsimg_test.py --list-releases --year 2019
"""

import argparse

import esri_wayback as ew

# Xi'an 范围 (与 config/xian.json / DelvMap sat_img.png / download_use_osm.py
# --sat_local_extent 默认值 完全一致)
XIAN_LAT_MIN, XIAN_LAT_MAX = 34.206385, 34.279658
XIAN_LON_MIN, XIAN_LON_MAX = 108.917423, 108.99286
# 目标画布 (与 sat_img.png 同尺寸: W=5625 H=6610, ~1.493 m/px Web Mercator)
XIAN_TARGET_W, XIAN_TARGET_H = 5625, 6610
# 默认 release: 645 = 2019-06-26 vintage (Wayback app active=645 == MapServer release
# == tile URL {release}, 由 Porto active=31144=release 31144 验证此映射)
XIAN_DEFAULT_RELEASE = 645
# zoom: lat~34 用 17 (与 download_use_osm.py: zoom=18 if abs(lat)<30 else 17 一致)
XIAN_DEFAULT_ZOOM = 17


def fetch_xian(release, zoom, out_path):
    """下载整个 Xi'an bbox 的 Wayback 影像, 缩放到 5625x6610, 存 PNG。

    复用 esri_wayback.fetch_wayback_canvas: GetMapInRect 拼接 z17 瓦片
    (原生 ~7031x8262) -> cv2.resize LANCZOS -> 5625x6610。因 GetMapInRect 裁剪
    是纯 Web Mercator 且恰好覆盖 bbox, 与 sat_img.png 同投影同范围, 故纯 resize
    即地理对齐, 无需 warp 重投影。

    release: Wayback release 号 (int)。None -> 最新 World_Imagery。
    zoom: 瓦片 zoom。
    out_path: 输出 PNG 路径。
    """
    rel_tag = "latest" if release is None else f"r{release}"
    folder = f"cache_xian/{rel_tag}"  # 与 cache_porto/ 同构, 按城市隔离
    print(f"\n下载 Xi'an 影像: release={release} zoom={zoom} -> {out_path}")
    info = ew.fetch_wayback_canvas(
        XIAN_LAT_MIN, XIAN_LON_MIN, XIAN_LAT_MAX, XIAN_LON_MAX, out_path,
        release=release, zoom=zoom,
        target_w=XIAN_TARGET_W, target_h=XIAN_TARGET_H,
        folder=folder, save_rgb=True)
    if not info["ok"]:
        print("[WARN] 部分瓦片下载失败, 影像对应位置为黑边; "
              "重跑会用 cache_xian/ 缓存补洞, 或换 --release。")
    print(f"✓ 存图: {out_path}  "
          f"native {info['native_w']}x{info['native_h']} -> "
          f"{XIAN_TARGET_W}x{XIAN_TARGET_H} (resized={info['resized']})")


def main():
    p = argparse.ArgumentParser(
        description="Xi'an rsimg 获取 (Esri Wayback 2019, 对齐 sat_img.png 5625x6610)。"
                    "下载→拼接→缩放→存图, 复用 esri_wayback 库。")
    p.add_argument("--list-releases", action="store_true",
                   help="列出 Wayback 所有可用 release 及日期 (复用 porto_rsimg_test.list_releases)")
    p.add_argument("--year", type=int, default=None,
                   help="目标年份 (如 2019), 仅 --list-releases 时用于高亮")
    p.add_argument("--release", type=int, default=XIAN_DEFAULT_RELEASE,
                   help=f"Wayback release 号 (默认 {XIAN_DEFAULT_RELEASE} = 2019-06-26)。"
                        "用 --latest 取 release=None")
    p.add_argument("--latest", action="store_true",
                   help="拉最新影像 (release=None, 作对照), 覆盖 --release")
    p.add_argument("--zoom", type=int, default=XIAN_DEFAULT_ZOOM,
                   help=f"瓦片 zoom (默认 {XIAN_DEFAULT_ZOOM}, lat~34)")
    p.add_argument("--out", default="xian_2019_wayback_645.png",
                   help="输出 PNG 路径 (默认当前目录)。drop-in 用: "
                        "--out ../../datasets/didi/xian/xian_2019_wayback_645.png")
    args = p.parse_args()

    if args.list_releases:
        # 复用 porto_rsimg_test.list_releases; 惰性导入 (porto 顶层 import esri_wayback,
        # 但不 import xian_rsimg_test, 故无循环)。两脚本同目录, 从本目录运行即可导入。
        from porto_rsimg_test import list_releases
        list_releases(year=args.year)
        return

    release = None if args.latest else args.release
    fetch_xian(release, args.zoom, args.out)


if __name__ == "__main__":
    main()
