"""
实验编排注册表 (Experiment Orchestration Registry)
====================================================
定义 TASKS 和 DATASETS 两张注册表, 供 run.py 编排层使用.

新增数据集: 在 DATASETS 加一项, 填好 input_graph_dir/traj_dir/eval_dataset_name.
新增 task (模型变体): 在 TASKS 加一项, 指向新的 train/infer/test 脚本.

run.py 通过这两张表把 "task + dataset" 映射到具体的脚本和路径, 自身零分支逻辑.
"""

import os


# ---------------------------------------------------------------------------
# 路径根 (所有实验产物归一到 runs/{run_id}/ 下)
# ---------------------------------------------------------------------------
RUNS_ROOT = 'runs'


def run_root(run_id):
    """单个实验的根目录: runs/{run_id}/"""
    return os.path.join(RUNS_ROOT, run_id)


def run_paths(run_id):
    """返回一个实验所有子目录的相对路径字典, 三步共享."""
    root = run_root(run_id)
    return {
        'run_root': root,
        'train_dir': os.path.join(root, 'train'),
        'ckpt_dir': os.path.join(root, 'train', 'checkpoints'),
        'train_log': os.path.join(root, 'train', 'log.txt'),
        'train_csv': os.path.join(root, 'train', 'csv'),
        'infer_dir': os.path.join(root, 'infer'),
        'eval_dir': os.path.join(root, 'eval'),
        'profile': os.path.join(root, 'profile.yaml'),
        'step_done_dir': os.path.join(root, '.step_done'),
        'best_ckpt': os.path.join(root, 'best_ckpt.txt'),
    }


# ---------------------------------------------------------------------------
# TASKS: 模型变体 → 脚本/模型类/默认config
# ---------------------------------------------------------------------------
# train_script / infer_script / test_script 用模块路径 (python -m 形式)
# ckpt_prefix: ModelCheckpoint filename 的前缀, 用于 best ckpt 自动选择时识别
# config_default: 给定 dataset 名, 返回默认 config 路径的函数
TASKS = {
    'extraction': {
        'train_script': 'engine.train',
        'infer_script': 'engine.inferencer',
        'test_script': 'engine.test',
        'model_class': 'SAMRoad',
        'ckpt_prefix': 'epoch',
        'ckpt_load_strict': True,
    },
    'completion': {
        'train_script': 'engine.train_completion',
        'infer_script': 'engine.inferencer_completion',
        'test_script': 'engine.test_completion',
        'model_class': 'SAMRoadCompletion',
        'ckpt_prefix': 'completion',
        'ckpt_load_strict': False,
    },
    '4ch': {
        'train_script': 'engine.train_4ch',
        'infer_script': 'engine.inferencer_4ch',
        'test_script': 'engine.test',  # 4ch 复用 test.py (同 SAMRoad 接口)
        'model_class': 'SAMRoad',
        'ckpt_prefix': 'epoch',
        'ckpt_load_strict': True,
    },
    # 未来新增 task 在此添加, 例如:
    # 'completion_v2': { 'train_script': 'engine.train_completion_v2', ... },
}


# ---------------------------------------------------------------------------
# DATASETS: 数据集 → 推理/评估所需属性
# ---------------------------------------------------------------------------
# eval_dataset_name: metrics/eval.py 的 --dataset 参数取值 (与 key 一致)
# infer_input_graph_dir: completion 推理时 --input_graph_dir 的默认值 (None=纯提取)
# infer_traj_dir: completion 推理时 --traj_dir 的默认值 (None=无轨迹)
# config 文件名规则: config/toponet_vitb_256_{dataset}[_completion].yaml, key 即文件名用词
DATASETS = {
    'spacenet': {
        'eval_dataset_name': 'spacenet',
        'infer_input_graph_dir': 'datasets/spacenet/RGB_1.0_meter',
        'infer_traj_dir': None,
    },
    'didi_xian': {
        'eval_dataset_name': 'didi_xian',
        'infer_input_graph_dir': 'datasets/didi/xian/2019_400',
        'infer_traj_dir': 'datasets/didi/xian/2019_400',
    },
    'cityscale': {
        'eval_dataset_name': 'cityscale',
        'infer_input_graph_dir': 'datasets/cityscale/20cities',
        'infer_traj_dir': None,
    },
    'porto': {
        'eval_dataset_name': 'porto',
        # completion 推理: partial 输入图目录 (region_*_refine_gt_graph_partial.p)
        'infer_input_graph_dir': 'datasets/porto/2014_400',
        # completion 推理: traj 目录 (region_*_traj.png, 同 input_graph_dir)
        'infer_traj_dir': 'datasets/porto/2014_400',
    },
    # 未来新增数据集在此添加
}


def get_task(task):
    if task not in TASKS:
        raise ValueError(f"未知 task '{task}', 可选: {list(TASKS.keys())}")
    return TASKS[task]


def get_dataset(dataset):
    if dataset not in DATASETS:
        raise ValueError(f"未知 dataset '{dataset}', 可选: {list(DATASETS.keys())}")
    return DATASETS[dataset]


def default_config_for(task, dataset):
    """返回 task 在该 dataset 下的默认 config 路径.

    config 文件名规则:
      extraction: config/toponet_vitb_256_{dataset}.yaml
      completion: config/toponet_vitb_256_{dataset}_completion.yaml
      4ch:        config/toponet_vitb_256_{dataset}.yaml (复用 extraction config)
    dataset key 即 config 文件名用词 (已统一, 如 didi_xian).
    """
    if task == 'completion':
        return f'config/toponet_vitb_256_{dataset}_completion.yaml'
    else:
        # extraction / 4ch 共用同一 config (4ch 模型读 4 通道但 config 里 BATCH_SIZE 等通用)
        return f'config/toponet_vitb_256_{dataset}.yaml'
