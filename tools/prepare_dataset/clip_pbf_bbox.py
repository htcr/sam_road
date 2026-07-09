"""
clip_pbf_bbox.py — 用 pyosmium 把全域 PBF 裁到一个 bbox 子集 PBF。

why: 葡萄牙全域 PBF (84M) 每次扫描太慢, 先裁到 Porto bbox 外扩 1km 的小圈,
后续 download_use_osm.py / building 渲染都读这个小 PBF。

how: osmium 的几何裁剪。用 osmium.geom.GeoJSONFactory + osmium.filter 比较繁,
这里用最稳的方式: osmium.SimpleHandler 遍历, 用 osmium.io.Writer 写出 bbox 内的
node/way/relation。way 需要其引用的 node 都在 (osmium area filter 会自动处理
引用, 但实现复杂); 这里采用 "保留所有 node + 保留与 bbox 相交的 way/relation",
下载_use_osm.py 的 OSMHandler 也只看 way.nodes 是否 in_bbox, 不依赖严格裁剪。

用法:
    python clip_pbf_bbox.py \
        --in rawdata/porto/portugal-150101.osm.pbf \
        --out rawdata/porto/porto_1km.osm.pbf \
        --lat-min 41.1035 --lat-max 41.2475 \
        --lon-min -8.699457 --lon-max -8.537543

依赖: pyosmium (samroad env 已装 4.0.2)
"""
import argparse
import osmium
from osmium.io import Writer
from osmium.osm import Node, Way, Relation


class BBoxClipper(osmium.SimpleHandler):
    """遍历 PBF, 把 bbox 内的 node + 相交 way/relation 写到输出。

    策略 (够 download_use_osm.py 用):
    - node: 落在 bbox 内则写 (且缓存其 id->location, 供 way 判断)
    - way: 若任一引用 node 在 bbox 内则写整条 way (含 bbox 外的 node 引用,
      这样路网边界不被截断; download_use_osm.py 的 build_osmmap_from_pbf
      会用 in_bbox 过滤, 不影响)
    - relation: 跳过 (download_use_osm.py 不用 relation)
    为支持 way 引用 bbox 外 node, 先全扫一遍 node 缓存所有 node location,
    再扫 way。两遍扫描。
    """
    def __init__(self, bbox, writer):
        super().__init__()
        self.lat_min, self.lat_max, self.lon_min, self.lon_max = bbox
        self.writer = writer
        self.node_loc = {}  # id -> (lat, lon)
        self.n_node = 0
        self.n_way = 0

    def _in_bbox(self, lat, lon):
        return (self.lat_min <= lat <= self.lat_max and
                self.lon_min <= lon <= self.lon_max)

    def node(self, n):
        # 缓存所有 node location (way 可能引用 bbox 外 node)
        if n.location.valid():
            self.node_loc[n.id] = (n.location.lat, n.location.lon)
            if self._in_bbox(n.location.lat, n.location.lon):
                self.writer.add_node(n)
                self.n_node += 1

    def way(self, w):
        # 任一引用 node 在 bbox 内则写
        keep = False
        for nr in w.nodes:
            loc = self.node_loc.get(nr.ref)
            if loc and self._in_bbox(loc[0], loc[1]):
                keep = True
                break
        if keep:
            self.writer.add_way(w)
            self.n_way += 1


def main():
    ap = argparse.ArgumentParser(description="裁剪 PBF 到 bbox 子集 (pyosmium)")
    ap.add_argument("--in", dest="inpbf", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lat-min", type=float, required=True)
    ap.add_argument("--lat-max", type=float, required=True)
    ap.add_argument("--lon-min", type=float, required=True)
    ap.add_argument("--lon-max", type=float, required=True)
    args = ap.parse_args()

    bbox = (args.lat_min, args.lat_max, args.lon_min, args.lon_max)
    print(f"[INFO] 裁剪 bbox: lat[{args.lat_min},{args.lat_max}] lon[{args.lon_min},{args.lon_max}]")
    print(f"[INFO] 输入: {args.inpbf}")
    print(f"[INFO] 输出: {args.out}")

    # SimpleWriter (osmium.SimpleWriter) 有 add_node/add_way, osmium.io.Writer 没有
    print("[INFO] 扫描 PBF (node 先 way 后, 单次 apply)...")
    writer = osmium.SimpleWriter(args.out)
    clipper = BBoxClipper(bbox, writer)
    clipper.apply_file(args.inpbf, locations=True)
    writer.close()
    print(f"[INFO] 完成: 写出 node={clipper.n_node} way={clipper.n_way}")
    print(f"[INFO] 输出: {args.out}")


if __name__ == "__main__":
    main()
