import numpy as np
import os
import imageio
import torch
import cv2

from utils import load_config
from dataset import cityscale_data_partition, read_rgb_img, get_patch_info_one_img
from model import SAMRoad
import graph_extraction
import graph_utils
import triage
# from triage import visualize_image_and_graph, rasterize_graph
import pickle
import scipy
import rtree
from collections import defaultdict

from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument(
    "--checkpoint", default=None, help="checkpoint of the model to test."
)
parser.add_argument(
    "--config", default=None, help="model config."
)
parser.add_argument("--device", default="cuda", help="device to use for training")
args = parser.parse_args()


def get_img_paths(root_dir, image_indices):
    img_paths = []

    for ind in image_indices:
        img_paths.append(os.path.join(root_dir, f"region_{ind}_sat.png"))

    return img_paths



def crop_img_patch(img, x0, y0, x1, y1):
    return img[y0:y1, x0:x1, :]


def get_batch_img_patches(img, batch_patch_info):
    patches = []
    for _, (x0, y0), (x1, y1) in batch_patch_info:
        patch = crop_img_patch(img, x0, y0, x1, y1)
        patches.append(torch.tensor(patch, dtype=torch.float32))
    batch = torch.stack(patches, 0).contiguous()
    return batch


def infer_one_img(net, img, config, img_name, out_dir):
    # TODO(congrui): centralize these configs
    image_size = img.shape[0]

    batch_size = config.INFER_BATCH_SIZE
    # list of (i, (x_begin, y_begin), (x_end, y_end))
    all_patch_info = get_patch_info_one_img(
        0, image_size, config.SAMPLE_MARGIN, config.PATCH_SIZE, config.INFER_PATCHES_PER_EDGE)
    patch_num = len(all_patch_info)
    batch_num = (
        patch_num // batch_size
        if patch_num % batch_size == 0
        else patch_num // batch_size + 1
    )

    # RGB already
    viz_img = np.copy(img)

    # [IMG_H, IMG_W]
    fused_keypoint_mask = torch.zeros(img.shape[0:2], dtype=torch.float32).to(args.device, non_blocking=False)
    fused_road_mask = torch.zeros(img.shape[0:2], dtype=torch.float32).to(args.device, non_blocking=False)
    pixel_counter = torch.zeros(img.shape[0:2], dtype=torch.float32).to(args.device, non_blocking=False)

    # stores img embeddings for toponet
    # list of [B, D, h, w], len=batch_num
    img_features = list()

    for batch_index in range(batch_num):
        offset = batch_index * batch_size
        batch_patch_info = all_patch_info[offset : offset + batch_size]
        # tensor [B, H, W, C]
        batch_img_patches = get_batch_img_patches(img, batch_patch_info)

        with torch.no_grad():
            batch_img_patches = batch_img_patches.to(args.device, non_blocking=False)
            # [B, H, W, 2]
            mask_scores, patch_img_features = net.infer_masks_and_img_features(batch_img_patches)
            img_features.append(patch_img_features)
        # Aggregate masks
        for patch_index, patch_info in enumerate(batch_patch_info):
            _, (x0, y0), (x1, y1) = patch_info
            keypoint_patch, road_patch = mask_scores[patch_index, :, :, 0], mask_scores[patch_index, :, :, 1]
            fused_keypoint_mask[y0:y1, x0:x1] += keypoint_patch
            fused_road_mask[y0:y1, x0:x1] += road_patch
            pixel_counter[y0:y1, x0:x1] += torch.ones(road_patch.shape[0:2], dtype=torch.float32, device=args.device)
    
    fused_keypoint_mask /= pixel_counter
    fused_road_mask /= pixel_counter
    # range 0-1 -> 0-255
    fused_keypoint_mask = (fused_keypoint_mask * 255).to(torch.uint8).cpu().numpy()
    fused_road_mask = (fused_road_mask * 255).to(torch.uint8).cpu().numpy()
    
    ## Extract sample points from masks
    graph_points = graph_extraction.extract_graph_points(fused_keypoint_mask, fused_road_mask, config)
    # for box query
    graph_rtree = rtree.index.Index()
    for i, v in enumerate(graph_points):
        x, y = v
        # hack to insert single points
        graph_rtree.insert(i, (x, y, x, y))
    
    ## Pass 2: infer toponet to predict topology of points from stored img features
    edge_scores = defaultdict(float)
    edge_counts = defaultdict(float)
    for batch_index in range(batch_num):
        offset = batch_index * batch_size
        batch_patch_info = all_patch_info[offset : offset + batch_size]

        topo_data = {
            'points': [],
            'pairs': [],
            'valid': [],
        }
        idx_maps = []


        # prepares pairs queries
        for patch_info in batch_patch_info:
            _, (x0, y0), (x1, y1) = patch_info
            patch_point_indices = list(graph_rtree.intersection((x0, y0, x1, y1)))
            idx_patch2all = {patch_idx : all_idx for patch_idx, all_idx in enumerate(patch_point_indices)}
            patch_point_num = len(patch_point_indices)
            # normalize into patch
            patch_points = graph_points[patch_point_indices, :] - np.array([[x0, y0]], dtype=graph_points.dtype)
            # for knn and circle query
            patch_kdtree = scipy.spatial.KDTree(patch_points)

            # k+1 because the nearest one is always self
            # idx is to the patch subgraph
            knn_d, knn_idx = patch_kdtree.query(patch_points, k=config.MAX_NEIGHBOR_QUERIES + 1, distance_upper_bound=config.NEIGHBOR_RADIUS)
            # [patch_point_num, n_nbr]
            knn_idx = knn_idx[:, 1:]  # removes self
            # [patch_point_num, n_nbr] idx is to the patch subgraph
            src_idx = np.tile(
                np.arange(patch_point_num)[:, np.newaxis],
                (1, config.MAX_NEIGHBOR_QUERIES)
            )
            valid = knn_idx < patch_point_num
            tgt_idx = np.where(valid, knn_idx, src_idx)
            # [patch_point_num, n_nbr, 2]
            pairs = np.stack([src_idx, tgt_idx], axis=-1)

            topo_data['points'].append(patch_points)
            topo_data['pairs'].append(pairs)
            topo_data['valid'].append(valid)
            idx_maps.append(idx_patch2all)
        
        # collate
        collated = {}
        for key, x_list in topo_data.items():
            length = max([x.shape[0] for x in x_list])
            collated[key] = np.stack([
                np.pad(x, [(0, length - x.shape[0])] + [(0, 0)] * (len(x.shape) - 1))
                for x in x_list
            ], axis=0)
        
        # infer toponet
        # [B, D, h, w]
        batch_features = img_features[batch_index]
        batch_points = torch.tensor(collated['points'], device=args.device)
        batch_pairs = torch.tensor(collated['pairs'], device=args.device)
        batch_valid = torch.tensor(collated['valid'], device=args.device)
        with torch.no_grad():
            # [B, N_samples, N_pairs, 1]
            topo_scores = net.infer_toponet(batch_features, batch_points, batch_pairs, batch_valid)
        # all-invalid (padded, no neighbors) queries returns nan scores
        # [B, N_samples, N_pairs]
        topo_scores = torch.where(torch.isnan(topo_scores), -100.0, topo_scores).squeeze(-1).cpu().numpy()

        # aggregate edge scores
        batch_size, n_samples, n_pairs = batch_valid.shape
        for bi in range(batch_size):
            for si in range(n_samples):
                for pi in range(n_pairs):
                    if not collated['valid'][bi, si, pi]:
                        continue
                    # idx to the full graph
                    src_idx_patch, tgt_idx_patch = collated['pairs'][bi, si, pi, :]
                    src_idx_all, tgt_idx_all = idx_maps[bi][src_idx_patch], idx_maps[bi][tgt_idx_patch]
                    edge_score = topo_scores[bi, si, pi]
                    assert 0.0 <= edge_score <= 1.0
                    edge_scores[(src_idx_all, tgt_idx_all)] += edge_score
                    edge_counts[(src_idx_all, tgt_idx_all)] += 1.0

    # avg edge scores and filter
    pred_edges = []
    for edge, score_sum in edge_scores.items():
        score = score_sum / edge_counts[edge] 
        if score > 0.5:
            pred_edges.append(edge)
    pred_edges = np.array(pred_edges)
    pred_nodes = graph_points[:, ::-1]  # to rc

        

    
    
    
    
    ### Astar graph extraction
    # pred_graph = graph_extraction.extract_graph_astar(fused_keypoint_mask, fused_road_mask)
    # # Doing this conversion to reuse copied code
    # pred_nodes, pred_edges = graph_utils.convert_from_nx(pred_graph)
    ### Astar graph extraction

    # Visualizes the diff between rasterized pred/gt graphs.
    # region_8_sat.png -> 8
    region_id = img_name.split('_')[1]
    gt_graph_path = f'cityscale/20cities/region_{region_id}_graph_gt.pickle'
    gt_graph = pickle.load(open(gt_graph_path, "rb"))
    gt_nodes, gt_edges = graph_utils.convert_from_sat2graph_format(gt_graph)
    img_size = viz_img.shape[0]
    rast_pred = triage.rasterize_graph(pred_nodes / img_size, pred_edges, img_size, dilation_radius=1)
    rast_pred_dilate = triage.rasterize_graph(pred_nodes / img_size, pred_edges, img_size, dilation_radius=5)
    rast_gt = triage.rasterize_graph(gt_nodes / img_size, gt_edges, img_size, dilation_radius=1)
    rast_gt_dilate = triage.rasterize_graph(gt_nodes / img_size, gt_edges, img_size, dilation_radius=5)

    fp_pred = (np.less_equal(rast_gt_dilate, 0) * np.greater(rast_pred, 0)).astype(np.uint8)
    missed_gt = (np.less_equal(rast_pred_dilate, 0) * np.greater(rast_gt, 0)).astype(np.uint8)

    diff_img = np.array(viz_img)
    # FP in blue, missed in red (BGR for opencv)
    diff_img = diff_img * np.less_equal(fp_pred, 0) + fp_pred * np.array([255, 0, 0], dtype=np.uint8)
    diff_img = diff_img * np.less_equal(missed_gt, 0) + missed_gt * np.array([0, 0, 255], dtype=np.uint8)

    diff_save_dir = os.path.join(out_dir, 'diff')
    if not os.path.exists(diff_save_dir):
        os.makedirs(diff_save_dir)
    cv2.imwrite(os.path.join(diff_save_dir, img_name), diff_img)

    # Visualizes merged large map
    viz_save_dir = os.path.join(out_dir, 'viz')
    if not os.path.exists(viz_save_dir):
        os.makedirs(viz_save_dir)
    viz_img = triage.visualize_image_and_graph(viz_img, pred_nodes / 2048, pred_edges, viz_img.shape[0])
    cv2.imwrite(os.path.join(viz_save_dir, img_name), viz_img)

    # Saves the large map
    large_map_sat2graph_format = graph_utils.convert_to_sat2graph_format(pred_nodes, pred_edges)
    # To suit code from RNGDet++: region_8_sat.png -> 8.p
    region_id = img_name.split('_')[1]
    graph_save_dir = os.path.join(out_dir, 'graph')
    if not os.path.exists(graph_save_dir):
        os.makedirs(graph_save_dir)
    graph_save_path = os.path.join(graph_save_dir, f'{region_id}.p')
    with open(graph_save_path, 'wb') as file:
        pickle.dump(large_map_sat2graph_format, file)
    
    print(f'Done for {img_name}.')


if __name__ == "__main__":
    config = load_config(args.config)
    
    # Builds eval model    
    device = torch.device("cuda") if args.device == "cuda" else torch.device("cpu")
    # Good when model architecture/input shape are fixed.
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    net = SAMRoad(config)

    # load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    print(f'##### Loading Trained CKPT {args.checkpoint} #####')
    net.load_state_dict(checkpoint["state_dict"], strict=True)
    net.eval()
    net.to(device)
    
    
    _, _, test_img_indices = cityscale_data_partition()
    test_img_paths = get_img_paths("./cityscale/20cities/", test_img_indices)
    
    # TODO: auto timestamp and store config
    output_dir = './save/cityscale_toponet_0'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for path in test_img_paths:
        # [H, W, C] RGB
        img = read_rgb_img(path)
        
        infer_one_img(net, img, config, os.path.basename(path), output_dir)