"""
Dataset Registry for SAM-Road.
Centralizes all dataset-specific configurations: paths, coordinate transforms,
image sizes, etc. Eliminates scattered if/elif chains across the codebase.

"""

import numpy as np
import json


def _cityscale_data_partition():
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


def _json_data_partition(json_path):
    with open(json_path, "r") as jf:
        data_list = json.load(jf)
    return data_list["train"], data_list["validation"], data_list["test"]


# Tool functions for coordinate conversion
def convert_pred_nodes_to_sat2graph(pred_nodes, dataset_name, image_size):
    """Convert predicted nodes from image (r,c) back to sat2graph pickle coods."""
    if dataset_name in ("spacenet", "didi_xian"):
        return np.stack([image_size - pred_nodes[:, 0], pred_nodes[:, 1]], axis=1)
    else:
        return pred_nodes


def convert_gt_nodes_from_sat2graph(gt_nodes, dataset_name, image_size):
    """Convert GT nodes from sat2graph pickle coods to image (r,c)."""
    if dataset_name in ("spacenet", "didi_xian"):
        gt_nodes = np.stack([gt_nodes[:, 1], image_size - gt_nodes[:, 0]], axis=1)
        gt_nodes = gt_nodes[::, :]
    return gt_nodes


DATASET_REGISTRY = {
    "cityscale": {
        "image_size": 2048,
        "sample_margin": 64,
        "coord_origin": "top-left",
        "coord_format": "(row,col)",
        "coord_transform": lambda v: v[:, ::-1],
        "rgb_pattern": "datasets/cityscale/20cities/region_{}_sat.png",
        "keypoint_mask_pattern": "datasets/cityscale/processed/keypoint_mask_{}.png",
        "road_mask_pattern": "datasets/cityscale/processed/road_mask_{}.png",
        "gt_graph_pattern": "datasets/cityscale/20cities/region_{}_refine_gt_graph.p",
        "gt_graph_eval_pattern": "datasets/cityscale/20cities/region_{}_graph_gt.pickle",
        "data_partition": _cityscale_data_partition,
        "train_epoch_size": None,
        "need_y_flip_for_sat2graph": False,
        "active_mask_pattern": None,
    },
    "spacenet": {
        "image_size": 400,
        "sample_margin": 0,
        "coord_origin": "bottom-left",
        "coord_format": "(y_up,x)",
        "coord_transform": lambda v: np.stack([v[:, 1], 400 - v[:, 0]], axis=1),
        "rgb_pattern": "datasets/spacenet/RGB_1.0_meter/{}__rgb.png",
        "keypoint_mask_pattern": "datasets/spacenet/processed/keypoint_mask_{}.png",
        "road_mask_pattern": "datasets/spacenet/processed/road_mask_{}.png",
        "gt_graph_pattern": "datasets/spacenet/RGB_1.0_meter/{}__gt_graph.p",
        "gt_graph_eval_pattern": "datasets/spacenet/RGB_1.0_meter/{}__gt_graph.p",
        "data_partition": lambda: _json_data_partition("datasets/spacenet/data_split.json"),
        "train_epoch_size": 84667,
        "need_y_flip_for_sat2graph": True,
        "active_mask_pattern": None,
    },    "didi_xian": {
        "image_size": 400,
        "sample_margin": 0,
        "coord_origin": "bottom-left",
        "coord_format": "(y_up,x)",
        "coord_transform": lambda v: np.stack([v[:, 1], 400 - v[:, 0]], axis=1),
        "rgb_pattern": "datasets/didi/xian/2019_400/region_{}_sat.png",
        "active_mask_pattern": "datasets/didi/xian/2019_400/region_{}_traj.png",
        "keypoint_mask_pattern": "datasets/didi/xian/processed/keypoint_mask_{}.png",
        "road_mask_pattern": "datasets/didi/xian/processed/road_mask_{}.png",
        "gt_graph_pattern": "datasets/didi/xian/2019_400/region_{}_refine_gt_graph.p",
        "gt_graph_eval_pattern": "datasets/didi/xian/2019_400/region_{}_graph_gt.pickle",
        "data_partition": lambda: _json_data_partition("datasets/didi/xian/data_split.json"),
        "train_epoch_size": 339 * 50,   # train(302)+val(37), 与 dataset*.py __len__ 统一 *50 口径
        "need_y_flip_for_sat2graph": True,
    },
    "porto": {
        "image_size": 400,
        "sample_margin": 0,
        # Porto 与 xian 同构: (row, col) 图像坐标系, 左上原点, swap (同 cityscale/didi_xian 在 dataset.py 的实际行为)
        "coord_origin": "top-left",
        "coord_format": "(row,col)",
        "coord_transform": lambda v: v[:, ::-1],
        "rgb_pattern": "datasets/porto/2014_400/region_{}_sat.png",
        "active_mask_pattern": "datasets/porto/2014_400/region_{}_traj.png",
        "keypoint_mask_pattern": "datasets/porto/processed/keypoint_mask_{}.png",
        "road_mask_pattern": "datasets/porto/processed/road_mask_{}.png",
        "gt_graph_pattern": "datasets/porto/2014_400/region_{}_refine_gt_graph.p",
        "gt_graph_eval_pattern": "datasets/porto/2014_400/region_{}_graph_gt.pickle",
        "data_partition": lambda: _json_data_partition("datasets/porto/data_split.json"),
        "train_epoch_size": 913 * 50,   # train(812)+val(101), 1015 块
        "need_y_flip_for_sat2graph": False,
    },
}


# Legacy aliases
_DATASET_ALIASES = {
    "didi": "didi_xian",
    "xian": "didi_xian",
}


def get_dataset_config(dataset_name):
    """Get dataset configuration by name."""
    resolved = _DATASET_ALIASES.get(dataset_name, dataset_name)
    if resolved not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Available: {list(DATASET_REGISTRY.keys())} "
            f"(aliases: {list(_DATASET_ALIASES.keys())})"
        )
    return DATASET_REGISTRY[resolved]


def get_data_partition(dataset_name):
    """Get train/val/test split for a dataset."""
    cfg = get_dataset_config(dataset_name)
    return cfg["data_partition"]()
