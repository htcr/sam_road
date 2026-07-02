#!/usr/bin/env python
"""
SAM-Road vs P2CNet per-tile visualization panels.

每个 test tile 输出一张 3x4 大图：
  Row 1: Inputs / GT
    Satellite | Partial prior (graph) | Trajectory (灰阶) | GT graph
  Row 2: Masks & overlay
    Partial mask (二值) | P2CNet mask (二值) | P2CNet mask overlay | P2CNet graph overlay
  Row 3: SAM-Road family (末列为重点对比项)
    SAMRoad extraction | Completion no prior | Completion RN+Traj | Completion RN-only

默认路径针对当前 didi_xian 0628 实验，可通过 CLI 覆盖。

示例：
  python tools/viz_compare_samroad_p2cnet.py --ids 100,109,111 --out outputs/viz_compare/didi_xian_0628_preview
  python tools/viz_compare_samroad_p2cnet.py --ids all --out outputs/viz_compare/didi_xian_0628_all
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

RGB = Tuple[int, int, int]

YELLOW: RGB = (255, 220, 0)
CYAN: RGB = (0, 255, 255)
# triage-style colors, matching postprocess/triage.py infer visualization:
# edge BGR (15,160,253) -> RGB (253,160,15); node BGR (0,255,255) -> RGB (255,255,0)
TRIAGE_EDGE: RGB = (253, 160, 15)
TRIAGE_NODE: RGB = (255, 255, 0)
BLACK: RGB = (0, 0, 0)
RED: RGB = (255, 60, 40)
GREEN: RGB = (30, 220, 80)
BLUE: RGB = (50, 120, 255)
WHITE: RGB = (255, 255, 255)
GRAY: RGB = (220, 220, 220)


def read_rgb(path: str | Path) -> Optional[np.ndarray]:
    path = str(path)
    if not os.path.exists(path):
        return None
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2RGB)
    return img


def read_mask(path: str | Path) -> Optional[np.ndarray]:
    path = str(path)
    if not os.path.exists(path):
        return None
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    return m


def load_graph(path: str | Path) -> Dict:
    path = str(path)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        return obj if isinstance(obj, dict) else {}
    except Exception as e:
        print(f'[WARN] failed to read graph {path}: {e}')
        return {}


def resize_img(img: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def ensure_rgb(img: Optional[np.ndarray], size: int, text: str = 'Missing') -> np.ndarray:
    if img is None:
        canvas = np.full((size, size, 3), 235, dtype=np.uint8)
        put_center_text(canvas, text, color=(80, 80, 80), scale=0.8)
        return canvas
    return resize_img(img, size)


def put_center_text(img: np.ndarray, text: str, color: RGB = BLACK, scale: float = 0.6) -> None:
    lines = text.split('\n')
    y0 = img.shape[0] // 2 - (len(lines) - 1) * 14
    for i, line in enumerate(lines):
        (w, h), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        x = max(5, (img.shape[1] - w) // 2)
        y = y0 + i * 28
        cv2.putText(img, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def add_title(panel: np.ndarray, title: str, h: int = 32) -> np.ndarray:
    title_bar = np.full((h, panel.shape[1], 3), 18, dtype=np.uint8)
    cv2.putText(title_bar, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1, cv2.LINE_AA)
    return np.vstack([title_bar, panel])


def mask_as_binary(mask: Optional[np.ndarray], size: int) -> np.ndarray:
    """Triage-style binary mask: road=255, background=0 (single channel replicated to RGB)."""
    if mask is None:
        return ensure_rgb(None, size, 'Missing mask')
    m = resize_img(mask, size)
    out = np.zeros((size, size, 3), dtype=np.uint8)
    out[m > 0] = WHITE
    return out


def overlay_mask(rgb: Optional[np.ndarray], mask: Optional[np.ndarray], size: int, color: RGB = WHITE, alpha: float = 1.0) -> np.ndarray:
    base = ensure_rgb(rgb, size, 'Missing sat')
    if mask is None:
        return base
    m = resize_img(mask, size)
    overlay = base.copy()
    overlay[m > 0] = color
    return cv2.addWeighted(overlay, alpha, base, 1 - alpha, 0)


def traj_overlay(rgb: Optional[np.ndarray], traj: Optional[np.ndarray], size: int) -> np.ndarray:
    """Trajectory as grayscale intensity overlay (brighter = denser trajectory), no colormap."""
    base = ensure_rgb(rgb, size, 'Missing sat')
    if traj is None:
        return base
    t = resize_img(traj, size)
    if t.max() > 0:
        norm = cv2.normalize(t, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        norm = t.astype(np.uint8)
    mask = norm > 0
    out = base.copy()
    # blend white by trajectory intensity: denser traj -> brighter overlay on satellite
    gray3 = cv2.cvtColor(norm, cv2.COLOR_GRAY2RGB)
    out[mask] = cv2.addWeighted(gray3, 0.60, base, 0.40, 0)[mask]
    return out


def node_to_xy(node) -> Tuple[int, int]:
    """Graph dict convention in this project/P2CNet export is mostly (row, col). Draw as (x=col, y=row)."""
    return int(round(float(node[1]))), int(round(float(node[0])))


def draw_graph(rgb: Optional[np.ndarray], adj: Dict, size: int,
               edge_color: RGB = TRIAGE_EDGE, node_color: RGB = TRIAGE_NODE,
               edge_thick: int = 4, node_radius: int = 4) -> np.ndarray:
    img = ensure_rgb(rgb, size, 'Missing sat')
    scale_x = size / 400.0
    scale_y = size / 400.0

    def scale_pt(p):
        x, y = node_to_xy(p)
        return int(round(x * scale_x)), int(round(y * scale_y))

    drawn = set()
    for node, nbs in adj.items():
        for nb in nbs:
            edge = tuple(sorted((tuple(node), tuple(nb))))
            if edge in drawn:
                continue
            drawn.add(edge)
            p0, p1 = scale_pt(node), scale_pt(nb)
            cv2.line(img, p0, p1, edge_color, edge_thick, cv2.LINE_AA)
    # nodes
    for node in adj.keys():
        p = scale_pt(node)
        cv2.circle(img, p, node_radius, node_color, -1, cv2.LINE_AA)
    return img


def make_grid(panels: List[Tuple[str, np.ndarray]], cols: int, gap: int = 8) -> np.ndarray:
    titled = [add_title(img, title) for title, img in panels]
    h, w = titled[0].shape[:2]
    rows = (len(titled) + cols - 1) // cols
    canvas = np.full((rows * h + (rows - 1) * gap, cols * w + (cols - 1) * gap, 3), 255, dtype=np.uint8)
    for i, img in enumerate(titled):
        r, c = divmod(i, cols)
        y = r * (h + gap)
        x = c * (w + gap)
        canvas[y:y+h, x:x+w] = img
    return canvas


def parse_ids(ids_arg: str, dataset_root: Path, limit: Optional[int] = None) -> List[str]:
    if ids_arg != 'all':
        ids = [x.strip() for x in ids_arg.split(',') if x.strip()]
    else:
        split_path = dataset_root.parent / 'data_split.json' if dataset_root.name == '2019_400' else dataset_root / 'data_split.json'
        if not split_path.exists():
            split_path = Path('datasets/didi/xian/data_split.json')
        data = json.load(open(split_path))
        ids = [str(x) for x in data['test']]
    return ids[:limit] if limit else ids


def p2c_index_map(dataset_root: Path) -> Dict[str, int]:
    split_path = dataset_root.parent / 'data_split.json' if dataset_root.name == '2019_400' else dataset_root / 'data_split.json'
    if not split_path.exists():
        split_path = Path('datasets/didi/xian/data_split.json')
    data = json.load(open(split_path))
    return {str(tile_id): i for i, tile_id in enumerate(data['test'])}


def panel_for_id(tile_id: str, args, idx_map: Dict[str, int]) -> Optional[np.ndarray]:
    size = args.tile_size
    dataset_root = Path(args.dataset_root)
    p2c_root = Path(args.p2cnet_root)
    p2c_idx = idx_map.get(str(tile_id))

    sat = read_rgb(dataset_root / f'region_{tile_id}_sat.png')
    traj = read_mask(dataset_root / f'region_{tile_id}_traj.png')
    if traj is None:
        traj = read_mask(dataset_root / f'region_{tile_id}_active.png')
    # partial as graph pickle (same adj-dict format as GT) for Row1 graph panel
    partial_graph = load_graph(dataset_root / 'partial_component' / f'region_{tile_id}_refine_gt_graph_partial.p')
    # partial as binary mask png for Row2 mask panel
    partial_mask = read_mask(dataset_root / 'partial_component' / f'region_{tile_id}_refine_gt_graph_partial.png')

    gt_graph = load_graph(dataset_root / f'region_{tile_id}_graph_gt.pickle')

    p2c_dir = p2c_root / str(p2c_idx) if p2c_idx is not None else None
    p2c_mask = read_mask(p2c_dir / 'maps.png') if p2c_dir else None
    p2c_graph = load_graph(p2c_root / 'graph' / f'{tile_id}.p')

    sam_ext = load_graph(Path(args.sam_extraction) / 'infer' / 'graph' / f'{tile_id}.p')
    sam_no = load_graph(Path(args.sam_noprior) / 'infer' / 'graph' / f'{tile_id}.p')
    sam_rn = load_graph(Path(args.sam_rn) / 'infer' / 'graph' / f'{tile_id}.p')
    sam_full = load_graph(Path(args.sam_full) / 'infer' / 'graph' / f'{tile_id}.p')

    panels = [
        ('Satellite', ensure_rgb(sat, size)),
        ('Partial prior', draw_graph(sat, partial_graph, size, GREEN, CYAN)),
        ('Trajectory', traj_overlay(sat, traj, size)),
        ('GT graph', draw_graph(sat, gt_graph, size, GREEN, CYAN)),
        ('Partial mask', mask_as_binary(partial_mask, size)),
        ('P2CNet mask', mask_as_binary(p2c_mask, size)),
        ('P2CNet mask overlay', overlay_mask(sat, p2c_mask, size, WHITE, 1.0)),
        ('P2CNet graph overlay', draw_graph(sat, p2c_graph, size)),
        ('SAMRoad extraction', draw_graph(sat, sam_ext, size)),
        ('Completion no prior', draw_graph(sat, sam_no, size)),
        ('Completion RN+Traj', draw_graph(sat, sam_full, size)),
        ('Completion RN-only', draw_graph(sat, sam_rn, size)),
    ]
    grid = make_grid(panels, cols=4)
    header_h = 44
    header = np.full((header_h, grid.shape[1], 3), 255, dtype=np.uint8)
    cv2.putText(header, f'DiDi Xian region_{tile_id}', (10, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.85, BLACK, 2, cv2.LINE_AA)
    return np.vstack([header, grid])


def write_index(out_dir: Path, ids: Iterable[str]) -> None:
    rows = ['<html><body><h1>SAM-Road vs P2CNet Visualization</h1>']
    for tile_id in ids:
        rows.append(f'<h2>region_{tile_id}</h2><img src="tile_{tile_id}.png" style="max-width:100%;border:1px solid #ddd">')
    rows.append('</body></html>')
    (out_dir / 'index.html').write_text('\n'.join(rows), encoding='utf-8')


def build_argparser():
    p = argparse.ArgumentParser(description='Visualize SAM-Road vs P2CNet results per DiDi Xian tile')
    p.add_argument('--ids', default='100,109,111,138,370', help="comma ids or 'all'")
    p.add_argument('--limit', type=int, default=None, help='limit number of ids when --ids all')
    p.add_argument('--dataset-root', default='datasets/didi/xian/2019_400')
    p.add_argument('--p2cnet-root', default='/home/hanhaoyu/P2CNet/saved/didi_xian/0628_172032/test/50')
    p.add_argument('--sam-extraction', default='runs/extraction_didi_xian_clipped_ep9')
    p.add_argument('--sam-noprior', default='runs/completion_didi_xian_clipped_notraj_norn_ep9')
    p.add_argument('--sam-rn', default='runs/completion_didi_xian_clipped_component_rnonly_ep9')
    p.add_argument('--sam-full', default='runs/completion_didi_xian_clipped_component_full_ep9')
    p.add_argument('--out', default='outputs/viz_compare/didi_xian_0628_preview')
    p.add_argument('--tile-size', type=int, default=384)
    return p


def main():
    args = build_argparser().parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_root = Path(args.dataset_root)
    ids = parse_ids(args.ids, dataset_root, args.limit)
    idx_map = p2c_index_map(dataset_root)
    print(f'Generating {len(ids)} panels → {out_dir}')
    done = []
    for tile_id in ids:
        panel = panel_for_id(str(tile_id), args, idx_map)
        out_path = out_dir / f'tile_{tile_id}.png'
        cv2.imwrite(str(out_path), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
        print(f'  ✓ {out_path}')
        done.append(str(tile_id))
    write_index(out_dir, done)
    print(f'✓ index: {out_dir / "index.html"}')


if __name__ == '__main__':
    main()
