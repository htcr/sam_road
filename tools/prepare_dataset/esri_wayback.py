import math
import numpy as np
import os
from PIL import Image
from time import sleep
import urllib.request

# 优先用 requests (TLS/重试更稳), 不可用时退回 urllib。Wayback CDN 对默认
# Python-urllib 客户端常返回 SSL UNEXPECTED_EOF, requests + 浏览器 UA 能显著改善。
try:
    import requests
    _HAVE_REQUESTS = True
except ImportError:
    _HAVE_REQUESTS = False

# 浏览器风格的请求头 (Wayback CDN 对裸 urllib 偶发 SSL EOF / 截断)
_HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Referer": "https://livingatlas.arcgis.com/wayback/",
    "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
}


def lonlat2TileIndex(lonlat, zoom):
    """经纬度转瓦片索引（Web Mercator 投影）"""
    n = np.exp2(zoom)
    x = int((lonlat[0] + 180) / 360 * n)
    y = int((1 - math.log(math.tan(lonlat[1] * math.pi / 180) + 1 / math.cos(lonlat[1] * math.pi / 180)) / math.pi) / 2 * n)
    return [x, y]


def lonlat2TilePos(lonlat, zoom, tile_size=256):
    """经纬度转瓦片内偏移像素位置"""
    n = np.exp2(zoom)
    fx = (lonlat[0] + 180) / 360 * n
    fy = (1 - math.log(math.tan(lonlat[1] * math.pi / 180) + 1 / math.cos(lonlat[1] * math.pi / 180)) / math.pi) / 2 * n

    ix = int(fx)
    iy = int(fy)

    dx = int((fx - ix) * tile_size)
    dy = int((fy - iy) * tile_size)
    return dx, dy


def _tile_url(zoom, tile_xy, release):
    """构造 Esri / Wayback 瓦片 URL。release=None -> 最新 World_Imagery。"""
    if release is None:
        return f"https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{zoom}/{tile_xy[1]}/{tile_xy[0]}"
    return f"https://wayback.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/MapServer/tile/{release}/{zoom}/{tile_xy[1]}/{tile_xy[0]}"


def _fetch_tile_bytes(url, timeout=30, retries=5):
    """单瓦片下载, 返回 bytes。带浏览器 UA + 超时 + 指数退避重试。

    Wayback CDN 常对裸 urllib 返回 SSL UNEXPECTED_EOF; requests 的 TLS 实现更稳,
    且能处理 chunked 截断。失败重试到 retries 次后抛异常。
    """
    last_err = None
    backoff = 2
    for attempt in range(retries):
        try:
            if _HAVE_REQUESTS:
                # timeout=(连接, 读); r.content 一次性读 (瓦片仅 ~10-30KB)
                r = requests.get(url, headers=_HTTP_HEADERS, timeout=(10, timeout))
                r.raise_for_status()
                data = r.content
                if not data:
                    raise IOError("empty response body")
                return data
            else:
                import ssl
                req = urllib.request.Request(url, headers=_HTTP_HEADERS)
                ctx = ssl.create_default_context()
                with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
                    data = resp.read()
                if not data:
                    raise IOError("empty response body")
                return data
        except Exception as e:
            last_err = e
            msg = str(e)
            # 404 是真没数据, 不重试; 429 (限流) 和 SSL EOF / 连接重置 才重试
            if "404" in msg or "Not Found" in msg:
                raise
            sleep(backoff)
            backoff = min(backoff * 2, 30)
    raise IOError(f"failed after {retries} retries: {url} -> {last_err}")


def downloadTileImage(zoom, tile_xy, outputname, release=None, timeout=30, retries=5):
    """下载 Esri 瓦片影像。

    release=None: 最新 World_Imagery (与 esri.py 一致)
    release=<int>: Esri Wayback 历史影像, URL 为
      https://wayback.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/MapServer/tile/{release}/{zoom}/{y}/{x}

    用 requests (回退 urllib) + 浏览器 UA + 超时 + 有限重试, 替代原 urlretrieve,
    解决 Wayback CDN 对裸 urllib 偶发 SSL UNEXPECTED_EOF 的问题。
    返回 True/False (不再无限重试阻塞整个任务)。
    """
    url = _tile_url(zoom, tile_xy, release)
    tmp_file = outputname + ".tmp.jpg"
    try:
        data = _fetch_tile_bytes(url, timeout=timeout, retries=retries)
        with open(tmp_file, "wb") as f:
            f.write(data)
        os.replace(tmp_file, outputname)
        return True
    except Exception as e:
        if os.path.exists(tmp_file):
            try:
                os.remove(tmp_file)
            except OSError:
                pass
        print(f"  [tile fail] {url} -> {e}")
        return False


def _download_one(args):
    """线程池 worker: 下载单个瓦片 (若缓存存在则跳过)。返回 (i, j, filename, succ)。"""
    zoom, x, y, i, j, filename, release = args
    if os.path.isfile(filename):
        return i, j, filename, True
    succ = downloadTileImage(zoom, [x, y], filename, release=release)
    return i, j, filename, succ


def GetMapInRect(min_lat, min_lon, max_lat, max_lon, folder="tile_cache/", zoom=19, tile_size=256, release=None, workers=16):
    os.makedirs(folder, exist_ok=True)

    tile1 = lonlat2TileIndex([min_lon, min_lat], zoom)
    tile2 = lonlat2TileIndex([max_lon, max_lat], zoom)
    print(f"tiles: {tile1}, {tile2}")
    print(f"delta of tiles: {(tile2[0]-tile1[0])*(tile2[1]-tile1[1])}")

    x_start, y_start = tile1
    x_end, y_end = tile2

    x_range = range(min(x_start, x_end), max(x_start, x_end) + 1)
    y_range = range(min(y_start, y_end), max(y_start, y_end) + 1)

    dimx = len(x_range) * tile_size
    dimy = len(y_range) * tile_size

    img = np.zeros((dimy, dimx, 3), dtype=np.uint8)

    rel_tag = "latest" if release is None else f"r{release}"

    # 构造全部瓦片任务, 并行下载 (默认 16 线程; 957 瓦片串行太慢)。
    # 单瓦片失败不再中断整体: 失败处保持黑 (np.zeros), ok=False, 上层决定是否重跑。
    tasks = []
    for i, x in enumerate(x_range):
        for j, y in enumerate(y_range):
            filename = os.path.join(folder, f"{rel_tag}_{zoom}_{x}_{y}.jpg")
            tasks.append((zoom, x, y, i, j, filename, release))

    ok = True
    n_done = 0
    n_total = len(tasks)
    try:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(workers, n_total)) as ex:
            for i, j, filename, succ in ex.map(_download_one, tasks):
                n_done += 1
                if n_done % 50 == 0 or n_done == n_total:
                    print(f"  [tiles] {n_done}/{n_total}")
                if not succ:
                    ok = False
                    continue  # 失败处留黑, 继续拼其余瓦片
                try:
                    subimg = Image.open(filename).convert("RGB")
                    subimg = np.array(subimg).astype(np.uint8)
                    img[j * tile_size:(j + 1) * tile_size, i * tile_size:(i + 1) * tile_size, :] = subimg
                except Exception as e:
                    print(f"  [tile read fail] {filename} -> {e}")
                    ok = False
    except ImportError:
        # 无 concurrent.futures (极旧 Python), 退回串行
        for t in tasks:
            i, j, filename, succ = _download_one(t)
            if not succ:
                ok = False
                continue
            subimg = Image.open(filename).convert("RGB")
            subimg = np.array(subimg).astype(np.uint8)
            img[j * tile_size:(j + 1) * tile_size, i * tile_size:(i + 1) * tile_size, :] = subimg

    # 计算裁剪偏移
    x1, y1 = lonlat2TilePos([min_lon, max_lat], zoom, tile_size)
    x2, y2 = lonlat2TilePos([max_lon, min_lat], zoom, tile_size)

    x2 += dimx - tile_size
    y2 += dimy - tile_size

    img = img[y1:y2, x1:x2]

    return img, ok


def fetch_wayback_canvas(min_lat, min_lon, max_lat, max_lon, out_path,
                         release=645, zoom=17, target_w=5625, target_h=6610,
                         folder="tile_cache/", tile_size=256, save_rgb=True):
    """下载任意区域/任意年份的 Wayback 影像, 写成与指定画布尺寸对齐的 PNG,
    可直接作为 download_use_osm.py --sat_source local:<out_path> 的输入。

    流程 (复用已有代码):
      1. GetMapInRect(...) -> 原生 Mercator 裁剪 (RGB uint8, shape (Hn,Wn,3))。
         其裁剪是纯 Web Mercator (lonlat2TilePos 与 wgs84_to_mercator 同公式),
         且恰好覆盖 bbox -> 与目标画布同投影同范围, 故只需重采样, 无需 warp 重投影。
      2. 打印原生尺寸 vs 目标尺寸 + 各轴缩放系数。
      3. 若 (Wn,Hn) != (target_w,target_h): cv2.resize INTER_LANCZOS4,
         dsize=(target_w, target_h)。  # cv2 dsize 是 (W,H) -> 输出 shape (target_h, target_w, 3)
      4. 存 PNG (默认 RGB)。download_use_osm.py 用 cv2.imread(IMREAD_COLOR) 读,
         会丢 alpha (见 download_use_osm.py:116,120), 故 RGB 即为真正的 drop-in。

    Args:
      min_lat,min_lon,max_lat,max_lon: bbox (度), 顺序与 GetMapInRect 一致。
        西安: 34.206385,108.917423,34.279658,108.99286。
      out_path: 输出 PNG 路径。
      release: Wayback release layer id (int)。645 = 西安 2019-06-26 vintage
        (app active=645 == MapServer release == URL {release}, 由 Porto 31144 验证)。
        None -> 最新 World_Imagery。
      zoom: 瓦片 zoom。默认 17 (lat~34, 与 download_use_osm.py 一致)。
      target_w,target_h: 输出画布 (像素)。cv2 dsize=(target_w, target_h) ->
        numpy shape (target_h, target_w, 3)。默认 5625,6610 = sat_img.png (W,H)。
        务必保持 W,H 顺序, 与 --sat_local_extent img_w,img_h 对齐。
      folder: 瓦片缓存目录 (建议按区域隔离, 如 "cache_xian_2019_r645/")。
        GetMapInRect 已用 f"r{release}_{zoom}_{x}_{y}.jpg" 命名, 不会冲突。
      tile_size: 256 (Esri 标准)。
      save_rgb: True -> 3 通道 RGB PNG (默认, drop-in)。False -> 加不透明 alpha,
        形式上对齐 sat_img.png 的 RGBA (仅美观, 不影响功能)。

    Returns:
      dict: {native_w, native_h, target_w, target_h, out_path, ok, resized}
      ok 镜像 GetMapInRect 的 ok (任一瓦片失败即 False; 影像仍会保存,
      失败瓦片处为黑, 因 GetMapInRect 用 np.zeros 初始化)。

    Note:
      - release 是 Wayback 全局快照层号, 某地显示的影像 vintage 可能比 release
        日期旧 (该地未被刷新)。脚本存 release 实际返回的内容, vintage 由肉眼确认。
      - 跨换日线/极区未处理 (西安 108.9°E 不涉及), 未来全球用需自行扩展。
    """
    img, ok = GetMapInRect(min_lat, min_lon, max_lat, max_lon,
                           folder=folder, zoom=zoom,
                           tile_size=tile_size, release=release)

    Hn, Wn = img.shape[:2]
    resized = (Wn != target_w) or (Hn != target_h)
    print(f"[wayback] native: {Wn}x{Hn}  target: {target_w}x{target_h}  "
          f"scale_x={target_w / Wn:.4f} scale_y={target_h / Hn:.4f}"
          f"  -> {'resize (LANCZOS)' if resized else 'no resize needed'}")

    if resized:
        try:
            import cv2  # 惰性导入, 与 download_use_osm.py:115 一致, 保持本模块无 cv2 也可导入
        except ImportError as e:
            raise ImportError(
                "resize 需要 opencv-python, 但导入 cv2 失败。"
                "请 `pip install opencv-python` 后重试。"
            ) from e
        # cv2 dsize=(W,H); 通道序无关 (RGB-in -> RGB-out)
        img = cv2.resize(img, dsize=(target_w, target_h),
                         interpolation=cv2.INTER_LANCZOS4)

    if not save_rgb:
        # 加不透明 alpha, 形式对齐 sat_img.png 的 RGBA
        img = np.dstack([img, np.full(img.shape[:2], 255, np.uint8)])

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    Image.fromarray(img.astype(np.uint8)).save(out_path)
    print(f"[wayback] saved: {out_path}  shape={img.shape}  ok={ok}")
    if not ok:
        print("[WARN] 部分瓦片下载失败, 影像对应位置为黑边; 请肉眼检查并考虑重试或换 release。")

    return {
        "native_w": Wn, "native_h": Hn,
        "target_w": target_w, "target_h": target_h,
        "out_path": out_path, "ok": ok, "resized": resized,
    }
