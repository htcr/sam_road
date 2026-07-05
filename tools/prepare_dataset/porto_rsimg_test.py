#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Porto rsimg (卫星影像) 获取测试 — 仅验证 Esri Wayback 能否拉到 Porto 2014 影像。

瓦片下载复用 esri_wayback.py (复制自 esri.py, URL 换成 Wayback + release 维度)。
本脚本只做两件事:
  1. 自动探测 Wayback 可用 release 列表 (号 + 日期), 筛出 2014 年的
  2. 对候选 release 调 esri_wayback.GetMapInRect 拉 Porto 一小块影像, 存 PNG 肉眼确认

用法 (在服务器上, 需联网):
  # 1. 列出所有可用 release 及日期, 高亮 2014 年的
  python porto_rsimg_test.py --list-releases --year 2014

  # 2. 用自动找到的 2014 release 拉 Porto 影像 (存 porto_2014.png)
  python porto_rsimg_test.py --fetch-year 2014 --zoom 17 --out porto_2014.png

  # 3. 用已知 release 号直接拉 (跳过探测)
  python porto_rsimg_test.py --release 3318 --zoom 17 --out porto_test.png

  # 4. 拉最新影像作对照
  python porto_rsimg_test.py --latest --zoom 17 --out porto_latest.png
"""

import argparse
import json
import sys
import urllib.request
from urllib.error import URLError, HTTPError

import esri_wayback as ew

# Porto 市中心范围 (与 config/porto.json 保持一致)
PORTO_LAT_MIN, PORTO_LAT_MAX = 41.14, 41.20
PORTO_LON_MIN, PORTO_LON_MAX = -8.69, -8.55

WAYBACK_BASE = "https://wayback.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/MapServer"


def fetch_url(url, timeout=40):
    """GET URL, 返回 (bytes, status). status=-1 表示失败. 仅用于元数据查询."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "porto-rsimg-test/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read(), r.status
    except (URLError, HTTPError, TimeoutError) as e:
        print(f"  [fetch failed] {url} -> {e}")
        return None, -1


def list_releases(year=None):
    """列出 Wayback 可用 release.

    Wayback MapServer 的 release 列表有几种可能入口, 逐个尝试:
      - MapServer?f=json : 可能含 layers[], 每个 layer id 即 release 号
      - 专用 wayback-api : 返回 release 元数据 (含日期)
    返回 [{release, date, name}].
    """
    print("=" * 60)
    print("探测 Wayback 可用 release")
    print("=" * 60)

    releases = []

    # 策略1: MapServer?f=json
    url = f"{WAYBACK_BASE}?f=json"
    print(f"\n[1] {url}")
    data, status = fetch_url(url, timeout=40)
    if status != -1 and data:
        print(f"  HTTP {status}, {len(data)} bytes")
        try:
            j = json.loads(data.decode("utf-8", errors="replace"))
        except json.JSONDecodeError:
            j = None
        if j and isinstance(j.get("layers"), list):
            print(f"  找到 {len(j['layers'])} 个 layers")
            for lyr in j["layers"]:
                releases.append({
                    "release": lyr.get("id"),
                    "name": str(lyr.get("name", "")),
                    "date": _extract_date(lyr),
                })
        else:
            print(f"  JSON keys: {list(j.keys()) if j else 'N/A'} (无 layers 字段)")

    # 策略2: Wayback 专用 release 列表 API (常见路径)
    api_candidates = [
        "https://wayback.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/MapServer/0?f=json",
    ]
    for u in api_candidates:
        if releases:
            break
        print(f"\n[2] {u}")
        data, status = fetch_url(u, timeout=30)
        if status != -1 and data:
            print(f"  HTTP {status}, {len(data)} bytes")
            try:
                j = json.loads(data.decode("utf-8", errors="replace"))
                print(f"  layer 0 keys: {list(j.keys())[:15]}")
            except json.JSONDecodeError:
                pass

    # 策略3: 保存 WMTS Capabilities 供人工查看 (含所有 release 的 TileMatrixSet 描述)
    if not releases:
        url = "https://wayback.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/WMTS/1.0.0/WMTSCapabilities.xml"
        print(f"\n[3] {url}")
        data, status = fetch_url(url, timeout=40)
        if status != -1 and data:
            cap_path = "wayback_wmts_capabilities.xml"
            with open(cap_path, "w", encoding="utf-8") as f:
                f.write(data.decode("utf-8", errors="replace"))
            print(f"  WMTS Capabilities 已保存到 {cap_path} (人工查看 release 列表/日期)")
            print("  提示: 用 grep 提取 release 号, 如  grep -oE 'World_Imagery_[0-9]+' {cap_path}")

    # 去重 + 排序
    seen = set()
    uniq = []
    for r in releases:
        if r["release"] is not None and r["release"] not in seen:
            seen.add(r["release"])
            uniq.append(r)
    uniq.sort(key=lambda r: (r["release"] is None, r["release"]))

    print(f"\n共找到 {len(uniq)} 个 release")
    if uniq:
        print("\n全部 release:")
        for r in uniq:
            mark = ""
            if year and r["date"] and str(year) in r["date"]:
                mark = "  <<< 目标年份"
            elif year and r["name"] and str(year) in r["name"]:
                mark = "  <<< 目标年份(name)"
            print(f"  release={r['release']:>6}  date={r['date']:30s}  name={r['name'][:50]}{mark}")

    if year:
        year_rels = [r for r in uniq
                     if (r["date"] and str(year) in r["date"]) or (r["name"] and str(year) in r["name"])]
        if year_rels:
            print(f"\n>>> 匹配年份 {year} 的 release: {[r['release'] for r in year_rels]}")
        else:
            print(f"\n[WARN] 未在 release 列表元数据中找到 {year} 年标记。")
            print("       可能 release 的 name/date 字段不含年份, 需人工查 Wayback app 确认 release 号。")
            print("       Wayback app: https://livingatlas.arcgis.com/wayback/")
    return uniq


def _extract_date(lyr):
    for k in ("date", "vintage", "copyrightText"):
        v = lyr.get(k)
        if v:
            return str(v)
    return ""


def fetch_porto(release, zoom, out_path, box_meters=500):
    """用 esri_wayback.GetMapInRect 拉 Porto 中心一小块影像, 存 PNG.

    box_meters: 中心点周围边长 (米), 默认 500m (zoom17 下约 2x2 瓦片, ~9 个 tile).
    测试用, 避免下整个 Porto 市区 (1560 个瓦片). 完整数据集请用 download_use_osm.py.
    """
    import math
    print(f"\n下载 Porto 影像: release={release} zoom={zoom} box={box_meters}m -> {out_path}")
    lat_c = (PORTO_LAT_MIN + PORTO_LAT_MAX) / 2
    lon_c = (PORTO_LON_MIN + PORTO_LON_MAX) / 2
    half_lat = box_meters / 2 / 111111.0
    half_lon = box_meters / 2 / (111111.0 * math.cos(math.radians(lat_c)))
    folder = f"cache_porto/{'latest' if release is None else 'r'+str(release)}"
    img, ok = ew.GetMapInRect(lat_c - half_lat, lon_c - half_lon,
                              lat_c + half_lat, lon_c + half_lon,
                              folder=folder, zoom=zoom, release=release)
    if not ok:
        print(f"[WARN] 部分瓦片下载失败, 影像可能不完整")
    from PIL import Image
    Image.fromarray(img.astype("uint8")).save(out_path)
    print(f"✓ 存图: {out_path}  shape={img.shape}")


def main():
    p = argparse.ArgumentParser(description="Porto rsimg 获取测试 (Esri Wayback 2014 影像验证)")
    p.add_argument("--list-releases", action="store_true", help="列出 Wayback 所有可用 release 及日期")
    p.add_argument("--year", type=int, default=None, help="目标年份 (如 2014), list-releases 高亮 / fetch-year 筛选")
    p.add_argument("--fetch-year", type=int, default=None, help="自动找该年份的 release 拉 Porto 影像")
    p.add_argument("--release", type=int, default=None, help="用指定 release 号直接拉")
    p.add_argument("--latest", action="store_true", help="拉最新影像 (release=None, 作对照)")
    p.add_argument("--zoom", type=int, default=17, help="瓦片 zoom (Porto 纬度~41, 默认17)")
    p.add_argument("--box-meters", type=int, default=500,
                   help="中心点周围边长(米), 默认500m (测试用, 避免下整个Porto 1560个瓦片). 完整数据集用 download_use_osm.py")
    p.add_argument("--out", default="porto_test.png", help="输出 PNG 路径")
    args = p.parse_args()

    if not (args.list_releases or args.fetch_year or args.release is not None or args.latest):
        p.print_help()
        print("\n至少指定一个: --list-releases / --fetch-year <year> / --release <num> / --latest")
        sys.exit(1)

    if args.list_releases:
        list_releases(year=args.year)

    if args.fetch_year:
        rels = list_releases(year=args.fetch_year)
        cand = [r for r in rels
                if (r["date"] and str(args.fetch_year) in r["date"])
                or (r["name"] and str(args.fetch_year) in r["name"])]
        if not cand:
            print(f"[ERROR] 未找到 {args.fetch_year} 年的 release, 无法 fetch。请用 --release <num> 手动指定。")
            sys.exit(1)
        # 取第一个匹配的
        rel = cand[0]["release"]
        print(f"\n使用 release={rel} 拉取 {args.fetch_year} 年影像")
        fetch_porto(rel, args.zoom, args.out, box_meters=args.box_meters)

    if args.release is not None:
        fetch_porto(args.release, args.zoom, args.out, box_meters=args.box_meters)

    if args.latest:
        fetch_porto(None, args.zoom, args.out, box_meters=args.box_meters)


if __name__ == "__main__":
    main()
