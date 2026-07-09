"""
SAM-Road Completion v2 Dataset
===============================
路网补全数据集: 从完整 GT 图随机删除边, 模拟不完整路网输入

v2 变更 (对齐方案B):
  1. road_feature_map: 2通道 (ch0=已知路mask, ch1=已知节点位置), 去掉距离场和方向场
  2. 动态 keep_ratio: 每次采样随机 U[0.2, 0.8], 不固定0.5
  3. 支持 traj_heatmap: Xian 数据集加载 traj.png (真实GPS轨迹), 其他数据集返回全零
  4. 修复 known_edge_index: 正确映射到 NMS 后节点索引
  5. 修复特征图一致性: 渲染和标签使用同一组删边
  6. 模态 Dropout: 20%概率清空所有先验 (road_feature_map + traj_heatmap + known_edge_index)
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import math
from postprocess import graph_utils
import rtree
import scipy
import pickle
import os
import addict
import json
import random


def read_rgb_img(path):
    bgr = cv2.imread(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb


def cityscale_data_partition():
    indrange_train, indrange_test, indrange_validation = [], [], []
    for x in range(180):
        if x % 10 < 8:
            indrange_train.append(x)
        if x % 10 == 9:
            indrange_test.append(x)
        if x % 20 == 18:
            indrange_validation.append(x)
        if x % 20 == 8:
            indrange_test.append(x)
    return indrange_train, indrange_validation, indrange_test


def spacenet_data_partition():
    with open('./datasets/spacenet/data_split.json', 'r') as jf:
        data_list = json.load(jf)
    return data_list['train'], data_list['validation'], data_list['test']


def didi_data_partition():
    with open('datasets/didi/xian/data_split.json', 'r') as jf:
        data_list = json.load(jf)
    return data_list['train'], data_list['validation'], data_list['test']


def porto_data_partition():
    """Porto (2014_400) 数据划分. 与 didi_xian 同构."""
    with open('datasets/porto/data_split.json', 'r') as jf:
        data_list = json.load(jf)
    return data_list['train'], data_list['validation'], data_list['test']


def get_patch_info_one_img(image_index, image_size, sample_margin, patch_size, patches_per_edge):
    patch_info = []
    sample_min = sample_margin
    sample_max = image_size - (patch_size + sample_margin)
    eval_samples = np.linspace(start=sample_min, stop=sample_max, num=patches_per_edge)
    eval_samples = [round(x) for x in eval_samples]
    for x in eval_samples:
        for y in eval_samples:
            patch_info.append(
                (image_index, (x, y), (x + patch_size, y + patch_size))
            )
    return patch_info


def transform_known_graph_coords(known_graph_adj, dataset):
    """
    把 known_graph_adj 的节点坐标从 pickle 原始格式统一转成 (x, y) 图像坐标空间.

    各数据集 pickle 坐标系 (与 dataset_completion.py 的 coord_transform 对齐):
      - didi_xian / cityscale: pickle 是 (row, col), 转 (x, y) = (col, row)
      - spacenet: pickle 是 (raw_y, raw_x), 原点左下, 转 (x, y) = (raw_x, 400 - raw_y)

    修复坐标系 Bug 1/2 的统一入口:
      - 训练侧: get_known_adj_for_render 调用 → render_graph_feature_map 拿到 (x,y) (修 Bug 2)
      - 推理侧: load_known_graph 调用 → render/匹配/注入 都拿到 (x,y) (修 Bug 1+2 推理部分)

    转完后所有下游 (render_graph_feature_map, _match_known_edges_to_graph_points,
    _inject_known_nodes) 拿到的都是 (x, y), 与 graph_points / image_embeddings 对齐.

    Args:
        known_graph_adj: dict, 原始 pickle 邻接表 {(row,col): [(row,col), ...]}
        dataset: config.DATASET 取值 ('spacenet' / 'didi_xian' / 'didi' / 'cityscale')

    Returns:
        新邻接表 dict, key/val 均为 (x, y) 坐标 (float)
    """
    if known_graph_adj is None or len(known_graph_adj) == 0:
        return known_graph_adj

    # 按数据集选择坐标变换: (row, col) → (x, y)
    if dataset in ('spacenet',):
        # spacenet: (raw_y, raw_x) → (x, y) = (raw_x, 400 - raw_y)
        # IMAGE_SIZE 固定 400, 与 dataset.py spacenet 分支一致
        transform_node = lambda node: (node[1], 400.0 - node[0])
    else:
        # didi_xian / cityscale / didi: (row, col) → (x, y) = (col, row)
        transform_node = lambda node: (node[1], node[0])

    new_adj = {}
    for node, neighbors in known_graph_adj.items():
        new_node = transform_node(node)
        new_adj[new_node] = [transform_node(n) for n in neighbors]
    return new_adj


def render_graph_feature_map(known_graph_adj, patch_x0, patch_y0, patch_size):
    """
    从已知路网的邻接表渲染2通道几何特征图 (v2精简版)

    通道说明:
      - ch0: 已知道路 mask (哪里有路) — CNN无法从RGB 100%确定, 是强先验
      - ch1: 已知节点位置 (确定的路网节点) — 区分已知/未知节点

    Args:
        known_graph_adj: dict, {(x,y): [(x1,y1), (x2,y2), ...]}
        patch_x0, patch_y0: 当前 patch 左上角坐标
        patch_size: patch 大小

    Returns:
        feature_map: [patch_size, patch_size, 2]
    """
    H = W = patch_size
    feature_map = np.zeros((H, W, 2), dtype=np.float32)

    if len(known_graph_adj) == 0:
        return feature_map

    # 提取所有边和节点
    all_nodes = set()
    edges = []
    for node, neighbors in known_graph_adj.items():
        all_nodes.add(node)
        for neighbor in neighbors:
            edges.append((node, neighbor))
            all_nodes.add(neighbor)

    # ---- 通道 0: 已知道路 Mask ----
    mask_ch = np.ascontiguousarray(feature_map[:, :, 0])
    for (src, tgt) in edges:
        p0 = (int(src[0] - patch_x0), int(src[1] - patch_y0))
        p1 = (int(tgt[0] - patch_x0), int(tgt[1] - patch_y0))
        cv2.line(mask_ch, p0, p1, 1.0, thickness=2)
    feature_map[:, :, 0] = mask_ch

    # ---- 通道 1: 已知节点位置 ----
    node_ch = np.ascontiguousarray(feature_map[:, :, 1])
    for node in all_nodes:
        px = int(node[0] - patch_x0)
        py = int(node[1] - patch_y0)
        if 0 <= px < W and 0 <= py < H:
            cv2.circle(node_ch, (px, py), 3, 1.0, -1)
    feature_map[:, :, 1] = node_ch

    return feature_map


def map_known_edges_to_nms(known_edge_set_subdivide, nmsed_indices, patch_x0, patch_y0,
                           subdivide_points, nms_keypoints, distance_threshold=8.0):
    """
    将已知路网在 subdivided 图上的边映射到 NMS 后的节点索引

    这是训练时的 known_edge_index 构造核心逻辑:
    1. 从 known_edge_set_subdivide 获取已知边的 (src_sub_idx, tgt_sub_idx)
    2. 优先精确匹配 (端点恰好在 nmsed_indices 中)
    3. 精确匹配失败时, 用 KDTree 最近邻匹配 (与推理时对齐)
    4. 构造 known_edge_index [2, E]

    P1-3: 增加 KDTree 最近邻回退, 消除训练(精确匹配)/推理(KDTree)gap.
    此处 nms_keypoints 与 subdivide_points 处于同一全局坐标系
    (sample_patch 调用本函数时尚未做 patch 偏移), 可直接 KDTree 匹配.

    Args:
        known_edge_set_subdivide: set of (src, tgt) 在 subdivided 图上的边
        nmsed_indices: np.array, 当前 patch NMS 后节点在 subdivided 图中的索引
        patch_x0, patch_y0: patch 左上角坐标 (保留接口, KDTree 路径不需要)
        subdivide_points: subdivided 图所有节点坐标 (全局)
        nms_keypoints: NMS 后节点的坐标 (全局, 未减 patch 偏移)
        distance_threshold: 最近邻匹配阈值

    Returns:
        known_edge_index: [2, E] tensor
    """
    if len(nmsed_indices) == 0 or len(known_edge_set_subdivide) == 0:
        return torch.zeros(2, 0, dtype=torch.long)

    # 精确匹配: subdivide_idx → nms_idx
    sub_to_nms = {}
    for nms_idx, sub_idx in enumerate(nmsed_indices):
        sub_to_nms[sub_idx] = nms_idx

    # KDTree 最近邻 (用于精确匹配失败的端点), 与推理时 _match_known_edges_to_graph_points 对齐
    nms_kdtree = scipy.spatial.KDTree(nms_keypoints)

    def _map_endpoint(sub_idx):
        """精确匹配优先, 失败则 KDTree 最近邻 (阈值内才算命中)"""
        if sub_idx in sub_to_nms:
            return sub_to_nms[sub_idx]
        coord = subdivide_points[sub_idx]
        dist, idx = nms_kdtree.query(coord, k=1)
        if dist < distance_threshold:
            return int(idx)
        return None

    edges_src = []
    edges_tgt = []
    for (s, t) in known_edge_set_subdivide:
        s_mapped = _map_endpoint(s)
        t_mapped = _map_endpoint(t)
        if s_mapped is not None and t_mapped is not None and s_mapped != t_mapped:
            edges_src.append(s_mapped)
            edges_tgt.append(t_mapped)

    if len(edges_src) == 0:
        return torch.zeros(2, 0, dtype=torch.long)

    return torch.tensor([edges_src, edges_tgt], dtype=torch.long)


class CompletionGraphLabelGenerator:
    """
    路网补全的图标签生成器

    v2 变更:
      - 动态 keep_ratio: 每次 refresh 时随机 U[0.2, 0.8]
      - 保存原始图级别的已知边 (known_edges_original), 确保特征图渲染一致性
      - sample_patch 返回 known_edge_index (映射到 NMS 节点索引)
    """

    def __init__(self, config, full_graph, coord_transform, keep_ratio_range=(0.2, 0.8)):
        self.config = config
        self.keep_ratio_range = keep_ratio_range
        self.current_keep_ratio = random.uniform(*keep_ratio_range)

        # 完整图 (igraph)
        self.full_graph_origin = graph_utils.igraph_from_adj_dict(full_graph, coord_transform)
        self.crossover_points = graph_utils.find_crossover_points(self.full_graph_origin)

        # 原始图的邻接表 (用于渲染 road_feature_map)
        self.full_graph_adj_original = full_graph

        # 子图 (与原版相同)
        self.subdivide_resolution = 4
        self.full_graph_subdivide = graph_utils.subdivide_graph(
            self.full_graph_origin, self.subdivide_resolution
        )
        self.subdivide_points = np.array(self.full_graph_subdivide.vs['point'])

        # 空间索引 (与原版相同)
        self.graph_rtee = rtree.index.Index()
        for i, v in enumerate(self.subdivide_points):
            x, y = v
            self.graph_rtee.insert(i, (x, y, x, y))
        self.graph_kdtree = scipy.spatial.KDTree(self.subdivide_points)

        # 排除交叉点附近的点 (与原版相同)
        crossover_exclude_radius = 4
        exclude_indices = set()
        for p in self.crossover_points:
            nearby_indices = self.graph_kdtree.query_ball_point(p, crossover_exclude_radius)
            exclude_indices.update(nearby_indices)
        self.exclude_indices = exclude_indices

        # 交叉点永远保留 (与原版相同)
        itsc_indices = set()
        point_num = len(self.full_graph_subdivide.vs)
        for i in range(point_num):
            if self.full_graph_subdivide.degree(i) != 2:
                itsc_indices.add(i)
        self.nms_score_override = np.zeros((point_num,), dtype=np.float32)
        self.nms_score_override[np.array(list(itsc_indices))] = 2.0

        # 采样权重 (与原版相同)
        interesting_indices = set()
        interesting_radius = 32
        for i in itsc_indices:
            p = self.subdivide_points[i]
            nearby_indices = self.graph_kdtree.query_ball_point(p, interesting_radius)
            interesting_indices.update(nearby_indices)
        for p in self.crossover_points:
            nearby_indices = self.graph_kdtree.query_ball_point(
                np.array(p), interesting_radius
            )
            interesting_indices.update(nearby_indices)
        self.sample_weights = np.full((point_num,), 0.1, dtype=np.float32)
        self.sample_weights[list(interesting_indices)] = 0.9

        # ---- 补全任务专用: 构造已知图 ----
        self.known_edge_set_subdivide = set()
        self.known_edges_original = set()  # 原始图级别的已知边, 用于渲染
        self._create_known_graph()

    def _create_known_graph(self):
        """
        随机删除部分边, 构造已知图 (按边随机采样)

        同时保存 subdivided 图和原始图两个级别的已知边集合, 确保渲染和标签一致.
        训练时不固定 seed (每 epoch refresh 随机, 数据增强);
        推理 generate_partial_prior.py 用固定 seed 42 (可复现).

        注: 曾试 component(按连通块保)采样, 但每epoch整块留/删导致训练目标
        剧烈波动, val_loss 震荡, APLS 退化(0.588→0.457). 回退按边随机.
        详见 docs/component采样退化分析.md.
        """
        self.current_keep_ratio = random.uniform(*self.keep_ratio_range)

        # ---- Subdivided 图级别的删边 ----
        all_edges_sub = [e.tuple for e in self.full_graph_subdivide.es]
        keep_num_sub = int(len(all_edges_sub) * self.current_keep_ratio)
        kept_edges_sub = random.sample(all_edges_sub, min(keep_num_sub, len(all_edges_sub)))

        self.known_edge_set_subdivide = set()
        for src, tgt in kept_edges_sub:
            self.known_edge_set_subdivide.add((min(src, tgt), max(src, tgt)))

        # ---- 原始图级别的删边 (用于渲染 road_feature_map) ----
        # 收集原始图的所有边
        all_edges_orig = []
        for node, neighbors in self.full_graph_adj_original.items():
            for neighbor in neighbors:
                edge = (min(node, neighbor), max(node, neighbor))
                all_edges_orig.append(edge)
        all_edges_orig = list(set(all_edges_orig))

        keep_num_orig = int(len(all_edges_orig) * self.current_keep_ratio)
        kept_edges_orig = random.sample(all_edges_orig, min(keep_num_orig, len(all_edges_orig)))

        self.known_edges_original = set(kept_edges_orig)

    def refresh_known_graph(self):
        """每个 epoch 调用, 重新随机删边"""
        self._create_known_graph()

    def get_known_adj_for_render(self):
        """
        从已知边集合构造邻接表 (用于渲染 road_feature_map)

        Returns:
            known_adj: dict, {(x,y): [(x1,y1), ...]}  ← 已转成 (x,y) 图像坐标空间
        """
        full_adj = self.full_graph_adj_original
        known_adj_rc = {}
        for node, neighbors in full_adj.items():
            for neighbor in neighbors:
                edge = (min(node, neighbor), max(node, neighbor))
                if edge in self.known_edges_original:
                    if node not in known_adj_rc:
                        known_adj_rc[node] = []
                    known_adj_rc[node].append(neighbor)
        # Bug 2 修复: 把 pickle (row,col) 统一转成 (x,y), 与 image_embeddings 对齐
        return transform_known_graph_coords(known_adj_rc, self.config.DATASET)

    def sample_patch(self, patch, rot_index=0):
        """
        采样 patch 的图标签

        v2 变更:
          - 已知已有边的候选对标记为 valid=False (不参与 topo loss)
          - 标签仍基于完整图的 BFS (目标: 预测完整路网)
          - 额外返回 known_edge_index (映射到 NMS 节点索引)

        Returns:
            nmsed_points: [N_nms, 2]
            samples: list of (pairs, shall_connect, valid)
            known_edge_index: [2, E] tensor, 已知路网在 NMS 节点中的边
        """
        (x0, y0), (x1, y1) = patch
        query_box = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
        patch_indices_all = set(self.graph_rtee.intersection(query_box))
        patch_indices = patch_indices_all - self.exclude_indices

        patch_indices = np.array(list(patch_indices))
        if len(patch_indices) == 0:
            sample_num = self.config.TOPO_SAMPLE_NUM
            max_nbr_queries = self.config.MAX_NEIGHBOR_QUERIES
            fake_points = np.array([[0.0, 0.0]], dtype=np.float32)
            fake_sample = (
                ([[0, 0]] * max_nbr_queries,
                 [False] * max_nbr_queries,
                 [False] * max_nbr_queries)
            )
            return fake_points, [fake_sample] * sample_num, torch.zeros(2, 0, dtype=torch.long)

        patch_points = self.subdivide_points[patch_indices, :]

        nms_scores = np.random.uniform(low=0.9, high=1.0, size=patch_indices.shape[0])
        nms_score_override = self.nms_score_override[patch_indices]
        nms_scores = np.maximum(nms_scores, nms_score_override)
        nms_radius = self.config.ROAD_NMS_RADIUS

        nmsed_points, kept_indices = graph_utils.nms_points(
            patch_points, nms_scores, radius=nms_radius, return_indices=True
        )
        nmsed_indices = patch_indices[kept_indices]
        nmsed_point_num = nmsed_points.shape[0]

        # ---- 构造 known_edge_index (映射到 NMS 节点索引) ----
        known_edge_index = map_known_edges_to_nms(
            self.known_edge_set_subdivide, nmsed_indices,
            x0, y0, self.subdivide_points, nmsed_points
        )

        sample_num = self.config.TOPO_SAMPLE_NUM
        sample_weights = self.sample_weights[nmsed_indices]
        sample_indices_in_nmsed = np.random.choice(
            np.arange(start=0, stop=nmsed_points.shape[0], dtype=np.int32),
            size=sample_num, replace=True, p=sample_weights / np.sum(sample_weights)
        )
        sample_indices = nmsed_indices[sample_indices_in_nmsed]

        radius = self.config.NEIGHBOR_RADIUS
        max_nbr_queries = self.config.MAX_NEIGHBOR_QUERIES
        nmsed_kdtree = scipy.spatial.KDTree(nmsed_points)
        sampled_points = self.subdivide_points[sample_indices, :]
        knn_d, knn_idx = nmsed_kdtree.query(
            sampled_points, k=max_nbr_queries + 1, distance_upper_bound=radius
        )

        samples = []

        for i in range(sample_num):
            source_node = sample_indices[i]
            valid_nbr_indices = knn_idx[i, knn_idx[i, :] < nmsed_point_num]
            valid_nbr_indices = valid_nbr_indices[1:]  # 去掉自身
            target_nodes = [nmsed_indices[ni] for ni in valid_nbr_indices]

            # BFS 在完整图上做 (目标: 预测完整路网)
            reached_nodes = graph_utils.bfs_with_conditions(
                self.full_graph_subdivide, source_node, set(target_nodes),
                radius // self.subdivide_resolution
            )

            pairs = []
            shall_connect = []
            valid_list = []
            source_nmsed_idx = sample_indices_in_nmsed[i]

            for j, target_graph_idx in enumerate(target_nodes):
                target_nmsed_idx = valid_nbr_indices[j]
                pairs.append((source_nmsed_idx, target_nmsed_idx))

                # 检查是否在已知图中已有该边
                edge_key = (min(source_node, target_graph_idx),
                            max(source_node, target_graph_idx))
                is_known_edge = edge_key in self.known_edge_set_subdivide

                if is_known_edge:
                    # 已知边: 不参与 topo loss
                    shall_connect.append(False)
                    valid_list.append(False)
                else:
                    # 未知边: 正常计算
                    shall_connect.append(target_graph_idx in reached_nodes)
                    valid_list.append(True)

            # zero-pad
            for _ in range(len(pairs), max_nbr_queries):
                pairs.append((source_nmsed_idx, source_nmsed_idx))
                shall_connect.append(False)
                valid_list.append(False)

            samples.append((pairs, shall_connect, valid_list))

        # 坐标变换 (与原版相同)
        nmsed_points -= np.array([x0, y0])[np.newaxis, :]
        nmsed_points = np.concatenate(
            [nmsed_points, np.ones((nmsed_point_num, 1), dtype=nmsed_points.dtype)], axis=1
        )
        trans = np.array([
            [1, 0, -0.5 * self.config.PATCH_SIZE],
            [0, 1, -0.5 * self.config.PATCH_SIZE],
            [0, 0, 1],
        ], dtype=np.float32)
        rot = np.array([
            [0, 1, 0],
            [-1, 0, 0],
            [0, 0, 1],
        ], dtype=np.float32)
        nmsed_points = nmsed_points @ trans.T @ np.linalg.matrix_power(rot.T, rot_index) @ np.linalg.inv(trans.T)
        nmsed_points = nmsed_points[:, :2]

        # 加噪声 (与原版相同)
        noise_scale = 1.0
        nmsed_points += np.random.normal(0.0, noise_scale, size=nmsed_points.shape)

        return nmsed_points, samples, known_edge_index


def completion_graph_collate_fn(batch):
    """补全数据集的 collate 函数, 处理 graph_points, known_edge_index, traj_heatmap 等"""
    keys = batch[0].keys()
    collated = {}
    for key in keys:
        if key == 'graph_points':
            tensors = [item[key] for item in batch]
            max_point_num = max([x.shape[0] for x in tensors])
            padded = []
            for x in tensors:
                pad_num = max_point_num - x.shape[0]
                padded_x = torch.concat([x, torch.zeros(pad_num, 2)], dim=0)
                padded.append(padded_x)
            collated[key] = torch.stack(padded, dim=0)
        elif key == 'known_edge_index':
            # edge_index [2, E] 需要按最大边数 padding
            # 用 -1 填充无效边 (GNN 中 valid_edge 检查会过滤 src < 0 的边)
            tensors = [item[key] for item in batch]
            max_edge_num = max([x.shape[1] for x in tensors]) if any(x.shape[1] > 0 for x in tensors) else 0
            if max_edge_num > 0:
                padded = []
                for x in tensors:
                    if x.shape[1] < max_edge_num:
                        pad_num = max_edge_num - x.shape[1]
                        padded_x = torch.concat([x, torch.full((2, pad_num), -1, dtype=torch.long)], dim=1)
                    else:
                        padded_x = x
                    padded.append(padded_x)
                collated[key] = torch.stack(padded, dim=0)  # [B, 2, max_E]
            else:
                # 空 edge_index
                collated[key] = torch.zeros(len(batch), 2, 0, dtype=torch.long)
        else:
            collated[key] = torch.stack([item[key] for item in batch], dim=0)
    return collated


class SatMapCompletionDataset(Dataset):
    """
    路网补全数据集 v2

    v2 变更:
      1. road_feature_map: 2通道 (mask + 节点位置)
      2. 动态 keep_ratio: 每次 refresh 随机 U[0.2, 0.8]
      3. 支持 traj_heatmap: Xian 加载 traj.png (真实GPS轨迹), 其他返回全零
      4. known_edge_index: 正确映射到 NMS 节点索引
      5. 特征图一致性: 渲染和标签使用同一组删边
      6. 模态 Dropout: 20%清空所有先验
    """

    def __init__(self, config, is_train, dev_run=False):
        self.config = config
        self.is_train = is_train
        self.keep_ratio_range = (
            getattr(config, 'KEEP_RATIO_MIN', 0.2),
            getattr(config, 'KEEP_RATIO_MAX', 0.8),
        )
        self.modality_dropout_prob = getattr(config, 'MODALITY_DROPOUT_PROB', 0.2)
        self.traj_dropout_prob = getattr(config, 'TRAJ_DROPOUT_PROB', 0.2)
        # USE_TRAJ 开关: False 时不加载 traj (训练无 traj 的 completion ckpt, 用于消融)
        # 默认 True (didi_xian 加载 traj.png, 其他数据集无 traj 文件本就全零)
        # 注: addict.Dict 对不存在的字段返回空Dict(假值), 用 'USE_TRAJ' in config 判断
        self.use_traj = config.get('USE_TRAJ', True) if hasattr(config, 'get') else True
        if self.use_traj == {} or self.use_traj is None:
            self.use_traj = True

        assert self.config.DATASET in {'cityscale', 'spacenet', 'didi', 'didi_xian', 'porto'}

        if self.config.DATASET == 'cityscale':
            self.IMAGE_SIZE = 2048
            self.SAMPLE_MARGIN = 64
            rgb_pattern = './datasets/cityscale/20cities/region_{}_sat.png'
            keypoint_mask_pattern = './datasets/cityscale/processed/keypoint_mask_{}.png'
            road_mask_pattern = './datasets/cityscale/processed/road_mask_{}.png'
            gt_graph_pattern = './datasets/cityscale/20cities/region_{}_refine_gt_graph.p'
            # CityScale 无 traj
            active_mask_pattern = None
            train, val, test = cityscale_data_partition()
            coord_transform = lambda v: v[:, ::-1]

        elif self.config.DATASET == 'spacenet':
            self.IMAGE_SIZE = 400
            self.SAMPLE_MARGIN = 0
            rgb_pattern = './datasets/spacenet/RGB_1.0_meter/{}__rgb.png'
            keypoint_mask_pattern = './datasets/spacenet/processed/keypoint_mask_{}.png'
            road_mask_pattern = './datasets/spacenet/processed/road_mask_{}.png'
            gt_graph_pattern = './datasets/spacenet/RGB_1.0_meter/{}__gt_graph.p'
            # SpaceNet 无 traj
            active_mask_pattern = None
            train, val, test = spacenet_data_partition()
            coord_transform = lambda v: np.stack([v[:, 1], 400 - v[:, 0]], axis=1)

        elif self.config.DATASET == 'didi' or self.config.DATASET == 'didi_xian':
            self.IMAGE_SIZE = 400
            self.SAMPLE_MARGIN = 0
            # 路径与 dataset.py 已修复的 didi_xian 分支保持一致 (相对项目根)
            rgb_pattern = 'datasets/didi/xian/2019_400/region_{}_sat.png'
            keypoint_mask_pattern = 'datasets/didi/xian/processed/keypoint_mask_{}.png'
            road_mask_pattern = 'datasets/didi/xian/processed/road_mask_{}.png'
            # gt_graph 文件在 xian_2019_400/ 子目录里, 不是 2019_400/ 顶层
            gt_graph_pattern = 'datasets/didi/xian/2019_400/region_{}_refine_gt_graph.p'
            # Xian 有 traj: traj.png (DelvMap 真实快递员 GPS 轨迹二值图, 已对齐)
            # USE_TRAJ=False 时不加载 (训练无 traj ckpt, 用于消融对比 traj 训练增强价值)
            active_mask_pattern = 'datasets/didi/xian/2019_400/region_{}_traj.png' if self.use_traj else None
            train, val, test = didi_data_partition()
            # DiDi Xian uses (row, col) coordinate format, same as Cityscale (NOT SpaceNet's (y_up, x))
            coord_transform = lambda v: v[:, ::-1]

        elif self.config.DATASET == 'porto':
            self.IMAGE_SIZE = 400
            self.SAMPLE_MARGIN = 0
            rgb_pattern = 'datasets/porto/2014_400/region_{}_sat.png'
            keypoint_mask_pattern = 'datasets/porto/processed/keypoint_mask_{}.png'
            road_mask_pattern = 'datasets/porto/processed/road_mask_{}.png'
            gt_graph_pattern = 'datasets/porto/2014_400/region_{}_refine_gt_graph.p'
            # Porto traj: 真实 GPS 轨迹点 (closed 版, 已对齐黑边). USE_TRAJ=False 时不加载
            active_mask_pattern = 'datasets/porto/2014_400/region_{}_traj.png' if self.use_traj else None
            train, val, test = porto_data_partition()
            # Porto 与 xian 同构: (row, col) 图像坐标系, 左上原点, swap
            coord_transform = lambda v: v[:, ::-1]

        train_split = train + val
        test_split = test
        tile_indices = train_split if self.is_train else test_split
        self.tile_indices = tile_indices

        # 存储数据
        self.rgbs, self.keypoint_masks, self.road_masks = [], [], []
        self.gt_graph_adjs = []  # 保存完整 GT 图
        self.graph_label_generators = []
        self.active_masks = []  # traj 热力图 (Xian 有, 其他为 None)
        self.has_traj = []  # 每个 tile 是否有 traj

        if dev_run:
            tile_indices = tile_indices[:4]

        for tile_idx in tile_indices:
            print(f'loading tile {tile_idx}')
            rgb_path = rgb_pattern.format(tile_idx)
            road_mask_path = road_mask_pattern.format(tile_idx)
            keypoint_mask_path = keypoint_mask_pattern.format(tile_idx)
            gt_graph_path = gt_graph_pattern.format(tile_idx)

            gt_graph_adj = pickle.load(open(gt_graph_path, 'rb'))
            if len(gt_graph_adj) == 0:
                print(f'===== skipped empty tile {tile_idx} =====')
                continue

            self.rgbs.append(read_rgb_img(rgb_path))
            self.road_masks.append(cv2.imread(road_mask_path, cv2.IMREAD_GRAYSCALE))
            self.keypoint_masks.append(cv2.imread(keypoint_mask_path, cv2.IMREAD_GRAYSCALE))
            self.gt_graph_adjs.append(gt_graph_adj)

            graph_label_generator = CompletionGraphLabelGenerator(
                config, gt_graph_adj, coord_transform,
                keep_ratio_range=self.keep_ratio_range
            )
            self.graph_label_generators.append(graph_label_generator)

            # 加载 traj 热力图 (仅 Xian)
            if active_mask_pattern is not None:
                active_mask_path = active_mask_pattern.format(tile_idx)
                active_img = cv2.imread(active_mask_path, cv2.IMREAD_GRAYSCALE)
                if active_img is not None:
                    self.active_masks.append(active_img)
                    self.has_traj.append(True)
                else:
                    self.active_masks.append(None)
                    self.has_traj.append(False)
            else:
                self.active_masks.append(None)
                self.has_traj.append(False)

        self.sample_min = self.SAMPLE_MARGIN
        self.sample_max = self.IMAGE_SIZE - (self.config.PATCH_SIZE + self.SAMPLE_MARGIN)

        if not self.is_train:
            eval_patches_per_edge = math.ceil(
                (self.IMAGE_SIZE - 2 * self.SAMPLE_MARGIN) / self.config.PATCH_SIZE
            )
            self.eval_patches = []
            for i in range(len(self.rgbs)):
                self.eval_patches += get_patch_info_one_img(
                    i, self.IMAGE_SIZE, self.SAMPLE_MARGIN,
                    self.config.PATCH_SIZE, eval_patches_per_edge
                )

    def refresh_known_graphs(self):
        """每个 epoch 调用, 重新随机删边"""
        for gen in self.graph_label_generators:
            gen.refresh_known_graph()

    def __len__(self):
        if self.is_train:
            if self.config.DATASET == 'cityscale':
                return max(1, int(self.IMAGE_SIZE / self.config.PATCH_SIZE)) ** 2 * 2500
            elif self.config.DATASET == 'spacenet':
                return 84667
            elif self.config.DATASET == 'didi' or self.config.DATASET == 'didi_xian':
                # 339 = train(302)+val(37) tile 数 (新数据 378 块, NW 编号);
                # 50 = 每 tile 期望采样 patch 数, 覆盖率 ~16x, 与
                # cityscale (16.3x) / spacenet (16.0x) 对齐, 避免 epoch 过长。
                return 339 * 50
            elif self.config.DATASET == 'porto':
                # 913 = train(812)+val(101) tile 数 (1015 块)
                return 913 * 50
        else:
            return len(self.eval_patches)

    def __getitem__(self, idx):
        # 采样 patch
        if self.is_train:
            img_idx = np.random.randint(low=0, high=len(self.rgbs))
            begin_x = np.random.randint(low=self.sample_min, high=self.sample_max + 1)
            begin_y = np.random.randint(low=self.sample_min, high=self.sample_max + 1)
            end_x = begin_x + self.config.PATCH_SIZE
            end_y = begin_y + self.config.PATCH_SIZE
        else:
            img_idx, (begin_x, begin_y), (end_x, end_y) = self.eval_patches[idx]

        # 裁剪
        rgb_patch = self.rgbs[img_idx][begin_y:end_y, begin_x:end_x, :]
        keypoint_mask_patch = self.keypoint_masks[img_idx][begin_y:end_y, begin_x:end_x]
        road_mask_patch = self.road_masks[img_idx][begin_y:end_y, begin_x:end_x]

        # ---- traj_heatmap (路径A) ----
        if self.active_masks[img_idx] is not None:
            traj_heatmap = self.active_masks[img_idx][begin_y:end_y, begin_x:end_x].astype(np.float32) / 255.0
        else:
            traj_heatmap = np.zeros((self.config.PATCH_SIZE, self.config.PATCH_SIZE), dtype=np.float32)

        # 数据增强: 旋转
        rot_index = 0
        if self.is_train:
            rot_index = np.random.randint(0, 4)
            rgb_patch = np.rot90(rgb_patch, rot_index, [0, 1]).copy()
            keypoint_mask_patch = np.rot90(keypoint_mask_patch, rot_index, [0, 1]).copy()
            road_mask_patch = np.rot90(road_mask_patch, rot_index, [0, 1]).copy()
            traj_heatmap = np.rot90(traj_heatmap, rot_index, [0, 1]).copy()

        # 采样图标签 (包含 known_edge_index)
        patch = ((begin_x, begin_y), (end_x, end_y))
        graph_points, topo_samples, known_edge_index = self.graph_label_generators[img_idx].sample_patch(
            patch, rot_index
        )

        pairs, connected, valid = zip(*topo_samples)

        # ---- 渲染已知路网的几何特征图 (2通道) ----
        # 使用 gen.known_edges_original 确保一致性
        known_graph_adj = self.graph_label_generators[img_idx].get_known_adj_for_render()
        road_feature_map = render_graph_feature_map(
            known_graph_adj, begin_x, begin_y, self.config.PATCH_SIZE
        )
        # 同步旋转
        if self.is_train and rot_index != 0:
            for ch in range(2):
                road_feature_map[:, :, ch] = np.rot90(
                    road_feature_map[:, :, ch], rot_index, [0, 1]
                ).copy()

        # ---- 模态 Dropout ----
        drop_all = False
        if self.is_train and random.random() < self.modality_dropout_prob:
            # 20% 概率清空所有先验: 模型退化为纯 Extraction
            road_feature_map = np.zeros_like(road_feature_map)
            known_edge_index = torch.zeros(2, 0, dtype=torch.long)
            traj_heatmap = np.zeros_like(traj_heatmap)
            drop_all = True
            # P0-1 修复: 先验已清零, 必须恢复 valid 让真实候选对重新参与 loss.
            # 否则这些步既无先验辅助、又只监督 ~50% 的候选边(已知边被 mask 掉),
            # 严格劣于原版 SAM-Road 的纯 extraction 训练步.
            # connected 保持原始 BFS 标签 (真实候选对的可达性, padding 自环为 False);
            # 仅把 valid 在"真实候选对 (src!=tgt)"处恢复为 True, zero-pad 自环保持 False.
            # 注: 此处 pairs/valid 为 list-of-tuple 结构 (zip(*topo_samples) 产物, 非 tensor),
            #     逐样本按真实候选对 (src!=tgt) 恢复 valid.
            valid = tuple(
                tuple((p[0] != p[1]) for p in sample_pairs)
                for sample_pairs in pairs
            )
            # connected 已含真实候选对的 BFS 标签 + padding 自环的 False, 无需改动

        # ---- traj 热力图增强 (路径A捷径缓解) ----
        if self.is_train and not drop_all and self.has_traj[img_idx]:
            rand_val = random.random()
            if rand_val < self.traj_dropout_prob:
                # 20% 概率: 全黑 (保持纯视觉能力)
                traj_heatmap = np.zeros_like(traj_heatmap)
            elif rand_val < 3 * self.traj_dropout_prob:
                # 40% 概率: 腐蚀 (打断捷径)
                traj_heatmap = traj_heatmap.copy()
                h, w = traj_heatmap.shape
                for _ in range(random.randint(3, 10)):
                    erase_w = random.randint(16, 64)
                    erase_h = random.randint(16, 64)
                    ex = random.randint(0, max(1, w - erase_w))
                    ey = random.randint(0, max(1, h - erase_h))
                    traj_heatmap[ey:ey + erase_h, ex:ex + erase_w] = 0
            # else: 40% 概率: 完整 traj (学习信任先验)

        # 验证集: 清空 traj, 保持 road_feature_map 和 known_edge_index
        if not self.is_train:
            traj_heatmap = np.zeros_like(traj_heatmap)

        # traj_heatmap: [H, W] → [H, W, 1]
        traj_heatmap = traj_heatmap[:, :, np.newaxis]  # [H, W, 1]

        return {
            'rgb': torch.tensor(rgb_patch, dtype=torch.float32),  # [H, W, 3]
            'traj_heatmap': torch.tensor(traj_heatmap, dtype=torch.float32),  # [H, W, 1]
            'keypoint_mask': torch.tensor(keypoint_mask_patch, dtype=torch.float32) / 255.0,
            'road_mask': torch.tensor(road_mask_patch, dtype=torch.float32) / 255.0,
            'road_feature_map': torch.tensor(road_feature_map, dtype=torch.float32).permute(2, 0, 1),  # [2, H, W]
            'graph_points': torch.tensor(graph_points, dtype=torch.float32),
            'pairs': torch.tensor(pairs, dtype=torch.int32),
            'connected': torch.tensor(connected, dtype=torch.bool),
            'valid': torch.tensor(valid, dtype=torch.bool),
            'known_edge_index': known_edge_index,  # [2, E]
        }
