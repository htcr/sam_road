# prepare_dataset/download_use_osm.py
import sys  
import json 
import mapdriver as md 
# import mapbox as md2
import esri as md2
import graph_ops as graphlib 
import math 
import numpy as np 
import argparse
from PIL import Image 
import pickle 
import os
import osmium
from osgeo import ogr

# ==========================================
# 模式 1：PBF 解析器 (保留你的原始逻辑)
# ==========================================
# 车行道白名单 (与 generate_porto_delvmap_bigimgs.py 一致)。
# 默认不启用 (保持 xian 等旧数据集兼容), Porto 用 --keep-highway-only 开启,
# 过滤掉 building/landuse/leisure/amenity/barrier 等非路 way + 纯行人路
# (pedestrian/footway/steps/path) + services(服务区闭合) + construction/proposed 等,
# 保留所有汽车可通行的 highway: 主干道 + 低等级公共路 + service 内部道路 + track 农用路。
# 依据 docs/chatgpt-osm类别分析.md 的"汽车可通行"判定。
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
# 闭合多边形处理: pedestrian 广场/services 服务区等若是闭合 way 会画成填充,
# 但它们已在白名单外被排除。白名单内的 highway 偶有闭合环路 (residential 等),
# 按 line 画即可 (build_osmmap_from_pbf 取相邻节点建边, 闭合环自然成环, 不填充)。


class OSMHandler(osmium.SimpleHandler):
    def __init__(self, keep_highway_only=False):
        super().__init__()
        self.nodes = {}      # node_id -> (lat, lon)
        self.ways = []       # list of node_id sequences
        self.keep_highway_only = keep_highway_only

    def node(self, n):
        self.nodes[n.id] = (n.location.lat, n.location.lon)

    def way(self, w):
        if len(w.nodes) < 2:
            return
        if self.keep_highway_only:
            tags = dict(w.tags)
            hw = tags.get('highway')
            if hw not in _KEEP_HIGHWAY:
                return
        self.ways.append([n.ref for n in w.nodes])


def build_osmmap_from_pbf(handler, bbox):
    nodedict = {}
    nodes = handler.nodes 
    for way in handler.ways:
        keep_way = False
        for nid in way:
            if nid in nodes:
                lat, lon = nodes[nid]
                if in_bbox(lat, lon, bbox):
                    keep_way = True
                    break

        if not keep_way:
            continue 

        for u, v in zip(way[:-1], way[1:]):
            if u not in nodes or v not in nodes:
                continue
            for a, b in [(u, v), (v, u)]:
                if a not in nodedict:
                    lat, lon = nodes[a]
                    nodedict[a] = {"lat": lat, "lon": lon, "to": {}, "from": {}}
                nodedict[a]["to"][b] = 1
    return nodedict


# ==========================================
# 模式 2：SHP 解析器 
# ==========================================
def load_global_ways_from_shp(shp_path):
    print(f"[INFO] Loading geometry from SHP: {shp_path}")
    driver = ogr.GetDriverByName("ESRI Shapefile")
    ds = driver.Open(shp_path, 0)
    layer = ds.GetLayer()
    
    global_ways = []
    for feature in layer:
        geom = feature.GetGeometryRef()
        if geom is not None:
            pts = geom.GetPoints()
            # OGR is (lon, lat), model expects (lat, lon)
            way = [(p[1], p[0]) for p in pts]
            global_ways.append(way)
    print(f"[INFO] Loaded {len(global_ways)} paths.")
    return global_ways

# ==========================================
# 通用工具
# ==========================================
def in_bbox(lat, lon, bbox):
    lat_st, lon_st, lat_ed, lon_ed = bbox
    return lat_st <= lat <= lat_ed and lon_st <= lon <= lon_ed


# ==========================================
# 主程序入口
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sat2Graph Dataset Generator with Trajectory Augmentation")
    parser.add_argument("--dataset_type", choices=['original', 'trajectory'], default='original', 
                        help="'original': Only Sat + OSM_GT. 'trajectory': Sat + Active_SHP (Input) -> OSM_GT.")
    parser.add_argument("--osm_pbf", required=True, help="Path to .osm.pbf file (Supervisory Signal/GT)")
    parser.add_argument("--active_shp", help="Path to active edges .shp file (Required if type='trajectory')")
    parser.add_argument("--sat_source", default="esri",
                        help="卫星图来源: 'esri' (默认, 联网下载) 或 'local:<png路径>' "
                             "(从本地大图按经纬度 Mercator 重投影采样, 需 --sat_local_extent). "
                             "用于 ESRI 不可达时复用 DelvMap 的 sat_img.png.")
    parser.add_argument("--sat_local_extent", default="34.206385,34.279658,108.917423,108.99286,5625,6610",
                        help="本地大图的范围: lat_min,lat_max,lon_min,lon_max,img_w,img_h "
                             "(默认 DelvMap 西安 sat_img.png 范围). 仅 --sat_source=local 时生效.")
    parser.add_argument("--resume", action="store_true",
                        help="断点续跑: 跳过已存在 region_{c}_refine_gt_graph.p 的 region")
    parser.add_argument("--keep-highway-only", action="store_true",
                        help="只保留主要车行道 (residential/primary/secondary/tertiary/motorway/trunk"
                             "及_link), 过滤 building/pedestrian 等非路 way。Porto 用, xian 不用")
    parser.add_argument("configs", nargs="+", help="Dataset JSON config files")
    args = parser.parse_args()

    # --- 0. 本地卫星图源初始化 (可选, 替代 ESRI) ---
    _local_sat = None  # (big_img ndarray, merc_bounds)
    _clip_extent = None  # (lat_min, lat_max, lon_min, lon_max): rn/active 渲染时裁到此范围, 与 sat/traj 黑边对齐
    if args.sat_source.startswith("local:"):
        sat_path = args.sat_source.split("local:", 1)[1]
        parts = [float(x) for x in args.sat_local_extent.split(",")]
        _lat_min, _lat_max, _lon_min, _lon_max, _img_w, _img_h = parts
        _clip_extent = (_lat_min, _lat_max, _lon_min, _lon_max)  # rn/active 只画此范围内的路
        import cv2
        _big = cv2.imread(sat_path, cv2.IMREAD_COLOR)  # BGR, (H, W, 3)
        if _big is None:
            raise SystemExit(f"[FATAL] 读不到本地卫星大图: {sat_path}")
        _big = np.ascontiguousarray(_big)  # 保连续, 兼容 cv2 非连续视图
        # 若是 RGBA (4ch), cv2 IMREAD_COLOR 已自动丢 alpha -> 3ch
        if _big.shape[:2] != (int(_img_h), int(_img_w)):
            print(f"[WARN] 本地大图 shape={_big.shape} != 期望 ({int(_img_h)},{int(_img_w)}), 按实际处理")
        _R = 20037508.34
        _x_min = _lon_min * _R / 180.0
        _x_max = _lon_max * _R / 180.0
        _y_min = math.log(math.tan((90.0 + _lat_min) * math.pi / 360.0)) / (math.pi / 180.0) * _R / 180.0
        _y_max = math.log(math.tan((90.0 + _lat_max) * math.pi / 360.0)) / (math.pi / 180.0) * _R / 180.0
        _local_sat = (_big, (_x_min, _y_min, _x_max, _y_max))
        print(f"[INFO] 使用本地卫星大图: {sat_path} shape={_big.shape}")
        print(f"[INFO] rn/active 裁剪范围 clip_extent=lat[{_lat_min},{_lat_max}] lon[{_lon_min},{_lon_max}]")

    def clip_bbox(bbox):
        """把 tile bbox 裁到 DelvMap extent 内, 用于 rn/active 渲染时只画范围内的路.
        像素映射仍用原始 bbox (graph2RegionCoordinate/graphVis 传原始 bbox, 不拉伸)."""
        if _clip_extent is None:
            return bbox
        clat_min, clat_max, clon_min, clon_max = _clip_extent
        lat_st = max(bbox[0], clat_min); lon_st = max(bbox[1], clon_min)
        lat_ed = min(bbox[2], clat_max); lon_ed = min(bbox[3], clon_max)
        return [lat_st, lon_st, lat_ed, lon_ed]


    def get_local_sat(lat_st, lon_st, lat_ed, lon_ed, size):
        """从本地 Mercator 大图按经纬度重投影采样一个 tile (与 generate_traj.py 同公式)."""
        big, (x_min, y_min, x_max, y_max) = _local_sat
        img_h, img_w = big.shape[:2]
        coords = np.arange(size, dtype=np.float64)
        px, py = np.meshgrid(coords, coords)
        # tile 像素 (px,py) -> lat/lon (线性, 与 graph2RegionCoordinate 一致: 北在 py=0, y 向下)
        lat = lat_ed - (py / size) * (lat_ed - lat_st)
        lon = lon_st + (px / size) * (lon_ed - lon_st)
        R = 20037508.34
        x_m = lon * R / 180.0
        y_m = np.log(np.tan((90.0 + lat) * np.pi / 360.0)) / (np.pi / 180.0) * R / 180.0
        bx = (x_m - x_min) / (x_max - x_min) * img_w
        by = (y_max - y_m) / (y_max - y_min) * img_h
        # 边缘 tile 外溢像素: 补黑(0), 不夹边界行/列 (避免拉伸出奇怪道道)
        in_range = (bx >= 0) & (bx < img_w) & (by >= 0) & (by < img_h)
        ix = np.clip(np.rint(bx).astype(np.int64), 0, img_w - 1)
        iy = np.clip(np.rint(by).astype(np.int64), 0, img_h - 1)
        tile = big[iy, ix]  # (size, size, 3) BGR uint8
        tile = np.where(in_range[..., None], tile, 0)  # 外溢像素置黑
        return tile

    if args.dataset_type == 'trajectory' and not args.active_shp:
        raise ValueError("[ERROR] --active_shp must be provided when --dataset_type is 'trajectory'")

    # --- 1. 全局数据预加载 ---
    print(f"[INFO] Parsing Full OSM PBF file for Ground Truth: {args.osm_pbf}")
    handler = OSMHandler(keep_highway_only=args.keep_highway_only)
    if args.keep_highway_only:
        print(f"[INFO] keep_highway_only=True: 只保留主要车行道 {_KEEP_HIGHWAY}")
    handler.apply_file(args.osm_pbf, locations=True)
    
    global_shp_ways = None
    if args.dataset_type == 'trajectory':
        global_shp_ways = load_global_ways_from_shp(args.active_shp)

    # --- 2. 加载 Configs ---
    dataset_cfg = []
    total_regions = 0 
    tid = 0 
    tn = 1 

    for name_cfg in args.configs:
        dataset_cfg_ = json.load(open(name_cfg, "r"))
        for item in dataset_cfg_:
            region_name = item["region"]
            year = item["year"]
            size = item["size"]
            
            lat_min, lon_min = item["lat_min"], item["lon_min"]        
            lat_max, lon_max = item["lat_max"], item["lon_max"]        
            
            dlat = size / 111111.0           
            dlon = size / (111111.0 * math.cos(math.radians(lat_min)))  
            lat_n = math.ceil((lat_max - lat_min) / dlat)  
            lon_n = math.ceil((lon_max - lon_min) / dlon)  
            
            dataset_cfg.append({
                "lat": lat_min, "lon": lon_min, "lat_max": lat_max, "lat_n": lat_n, "lon_n": lon_n
            })
            total_regions += lat_n * lon_n

    print(f"[INFO] Total regions to process: {total_regions}")
    # region 文件直接写到 cwd (扁平结构, 与 cityscale/spacenet 对齐:
    # datasets/didi/xian/2019_400/region_*.png, processed/ 与 data_split.json 在上一层).
    # 从 2019_400/ 目录运行即可.
    dataset_folder = "."
    folder_mapbox_cache = "cache_esri/{}/".format(region_name)
    os.makedirs(folder_mapbox_cache, exist_ok=True)

    # --- 3. 核心切片循环 ---
    c = 0
    for item in dataset_cfg:
        ilat, ilon = item["lat_n"], item["lon_n"]
        lat_origin, lon_origin = item["lat"], item["lon"]
        lat_max_city = item["lat_max"]

        for i in range(ilat):
            for j in range(ilon):
                print(f"Processing region {c}/{total_regions}")
                if c % tn != tid:
                    c += 1; continue

                def norm(x, nd=7): return round(x, nd)
                # 编号从左上角(NW)开始, 行优先 TL->BR:
                #   i=0 = 最北行 (lat_ed 接近 lat_max), i 增大向南
                #   j=0 = 最西列 (lon_st 接近 lon_min), j 增大向东
                # bbox 仍是 [lat_st(南), lon_st(西), lat_ed(北), lon_ed(东)] (lat_ed>lat_st),
                # 与 graph2RegionCoordinate / graphVis / render_tile_traj 约定一致.
                lat_ed = norm(lat_max_city - size/111111.0 * i)
                lat_st = norm(lat_max_city - size/111111.0 * (i+1))
                lon_st = norm(lon_origin + size/111111.0 * j / math.cos(math.radians(lat_origin)))
                lon_ed = norm(lon_origin + size/111111.0 * (j+1) / math.cos(math.radians(lat_origin)))
                bbox = [lat_st, lon_st, lat_ed, lon_ed]
                zoom = 18 if abs(lat_st) < 30 else 17

                # 断点续跑: 若 --resume 且该 region 的 refine_gt_graph.p 已存在, 跳过
                # (refine 是最后一步, 存在即代表该 region 全套文件已完整生成)
                if args.resume and os.path.isfile(f"{dataset_folder}/region_{c}_refine_gt_graph.p"):
                    c += 1
                    continue

                # --- 3.1 截取卫星图 (Input) ---
                if _local_sat is not None:
                    # 本地大图按经纬度 Mercator 重投影 (无需联网), BGR->RGB 后存
                    tile_bgr = get_local_sat(lat_st, lon_st, lat_ed, lon_ed, size)
                    img_rgb = tile_bgr[:, :, ::-1].copy()  # BGR -> RGB
                    Image.fromarray(img_rgb.astype(np.uint8)).save(f"{dataset_folder}/region_{c}_sat.png")
                else:
                    img, _ = md2.GetMapInRect(lat_st, lon_st, lat_ed, lon_ed, zoom=zoom, folder=folder_mapbox_cache)
                    img = Image.fromarray(img.astype(np.uint8)).resize((size, size), Image.BILINEAR)
                    Image.fromarray(np.array(img)).save(f"{dataset_folder}/region_{c}_sat.png")

                # --- 3.2 构建完整 OSM 拓扑图 (Ground Truth / 监督信号) ---
                # 用 clip_bbox 取节点 (只画 DelvMap 范围内的路, 与 sat/traj 黑边对齐);
                # 像素映射用原始 bbox [lat_st,lon_st,lat_ed,lon_ed] (不拉伸).
                rn_filter_bbox = clip_bbox(bbox)
                node_neighbor_gt = {}
                node_dict = build_osmmap_from_pbf(handler, rn_filter_bbox)
                for node_id, node_info in node_dict.items():
                    n1key = (node_info["lat"], node_info["lon"])
                    neighbors = list(node_info["to"].keys()) + list(node_info["from"].keys())
                    for nid in set(neighbors):
                        n2key = (node_dict[nid]["lat"], node_dict[nid]["lon"])
                        node_neighbor_gt = graphlib.graphInsert(node_neighbor_gt, n1key, n2key)

                # GT 后处理及保存
                node_neighbor_gt = graphlib.graphDensify(node_neighbor_gt)
                nn_gt_region = graphlib.graph2RegionCoordinate(node_neighbor_gt, [lat_st,lon_st,lat_ed,lon_ed], size)

                with open(f"{dataset_folder}/region_{c}_graph_gt.pickle", "wb") as f:
                    pickle.dump(nn_gt_region, f)
                graphlib.graphVis2048Segmentation(node_neighbor_gt, [lat_st,lon_st,lat_ed,lon_ed], f"{dataset_folder}/region_{c}_gt.png", size)

                # graphGroundTruthPreProcess 在路网密集区 (如 Porto) 偶发 KeyError
                # (图迭代中节点被删后仍被引用)。崩溃时退化为用未精修的 nn_gt_region 作 refine,
                # 损失精修质量但不中断整体流程。记录到 log 供事后排查。
                try:
                    nn_refine_gt, sample_points = graphlib.graphGroundTruthPreProcess(nn_gt_region)
                except Exception as _e:
                    print(f"[WARN] region {c} graphGroundTruthPreProcess 崩溃 ({type(_e).__name__}: {_e}), "
                          f"退化为未精修 graph_gt")
                    nn_refine_gt = nn_gt_region
                    sample_points = {"parallel_road": [], "complicated_intersections": [], "overpass": []}
                with open(f"{dataset_folder}/region_{c}_refine_gt_graph.p", "wb") as f:
                    pickle.dump(nn_refine_gt, f)
                with open(f"{dataset_folder}/region_{c}_refine_gt_graph_samplepoints.json", "w") as f:
                    json.dump(sample_points, f, indent=2)

                # --- 3.3 构建活跃轨迹图 (Model Input / 你的创新点) ---
                if args.dataset_type == 'trajectory':
                    node_neighbor_active = {}
                    for way in global_shp_ways:
                        if any(in_bbox(p[0], p[1], rn_filter_bbox) for p in way):
                            for k in range(len(way) - 1):
                                node_neighbor_active = graphlib.graphInsert(node_neighbor_active, way[k], way[k+1])
                                node_neighbor_active = graphlib.graphInsert(node_neighbor_active, way[k+1], way[k])
                    
                    # 对活跃轨迹进行相同的插值和坐标转换
                    node_neighbor_active = graphlib.graphDensify(node_neighbor_active)
                    nn_active_region = graphlib.graph2RegionCoordinate(node_neighbor_active, [lat_st,lon_st,lat_ed,lon_ed], size)
                    
                    # 保存为 mask 图片 (可与卫星图 concat) 以及 pickle 图结构
                    graphlib.graphVis2048Segmentation(node_neighbor_active, [lat_st,lon_st,lat_ed,lon_ed], f"{dataset_folder}/region_{c}_active.png", size)
                    with open(f"{dataset_folder}/region_{c}_active_graph.pickle", "wb") as f:
                        pickle.dump(nn_active_region, f)

                c += 1
                