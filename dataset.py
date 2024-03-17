import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import math
import graph_utils
import rtree
import scipy
import pickle
import os
import addict

IMAGE_SIZE = 2048
SAMPLE_MARGIN = 64

def read_rgb_img(path):
    bgr = cv2.imread(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb

def cityscale_data_partition():
    # dataset partition
    indrange_train = []
    indrange_test = []
    indrange_validation = []

    for x in range(180):
        if x % 10 < 8 :
            indrange_train.append(x)

        if x % 10 == 9:
            indrange_test.append(x)

        if x % 20 == 18:
            indrange_validation.append(x)

        if x % 20 == 8:
            indrange_test.append(x)
    return indrange_train, indrange_validation, indrange_test


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


class GraphLabelGenerator():
    def __init__(self, config, full_graph):
        self.config = config
        # full_graph: sat2graph format
        # convert to igraph for high performance
        self.full_graph_origin = graph_utils.igraph_from_sat2graph_format(full_graph)
        # find crossover points, we'll avoid predicting these as keypoints
        self.crossover_points = graph_utils.find_crossover_points(self.full_graph_origin)
        # subdivide version
        # TODO: check proper resolution
        self.subdivide_resolution = 4
        self.full_graph_subdivide = graph_utils.subdivide_graph(self.full_graph_origin, self.subdivide_resolution)
        # np array, maybe faster
        self.subdivide_points = np.array(self.full_graph_subdivide.vs['point'])
        # pre-build spatial index
        # rtree for box queries
        self.graph_rtee = rtree.index.Index()
        for i, v in enumerate(self.subdivide_points):
            x, y = v
            # hack to insert single points
            self.graph_rtee.insert(i, (x, y, x, y))
        # kdtree for spherical query
        self.graph_kdtree = scipy.spatial.KDTree(self.subdivide_points)

        # pre-exclude points near crossover points
        crossover_exclude_radius = 4
        exclude_indices = set()
        for p in self.crossover_points:
            nearby_indices = self.graph_kdtree.query_ball_point(p, crossover_exclude_radius)
            exclude_indices.update(nearby_indices)
        self.exclude_indices = exclude_indices

        # Find intersection points, these will always be kept in nms
        itsc_indices = set()
        point_num = len(self.full_graph_subdivide.vs)
        for i in range(point_num):
            if self.full_graph_subdivide.degree(i) != 2:
                itsc_indices.add(i)
        self.nms_score_override = np.zeros((point_num, ), dtype=np.float32)
        self.nms_score_override[np.array(list(itsc_indices))] = 2.0  # itsc points will always be kept

        # Points near crossover and intersections are interesting.
        # they will be more frequently sampled
        interesting_indices = set()
        interesting_radius = 32
        # near itsc
        for i in itsc_indices:
            p = self.subdivide_points[i]
            nearby_indices = self.graph_kdtree.query_ball_point(p, interesting_radius)
            interesting_indices.update(nearby_indices)
        for p in self.crossover_points:
            nearby_indices = self.graph_kdtree.query_ball_point(np.array(p), interesting_radius)
            interesting_indices.update(nearby_indices)
        self.sample_weights = np.full((point_num, ), 0.1, dtype=np.float32)
        self.sample_weights[list(interesting_indices)] = 0.9
    
    def sample_patch(self, patch, rot_index = 0):
        (x0, y0), (x1, y1) = patch
        query_box = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
        patch_indices_all = set(self.graph_rtee.intersection(query_box))
        patch_indices = patch_indices_all - self.exclude_indices

        # Use NMS to downsample, params shall resemble inference time
        patch_indices = np.array(list(patch_indices))
        patch_points = self.subdivide_points[patch_indices, :]
        
        # random scores to emulate different random configurations that all share a
        # similar spacing between sampled points
        # raise scores for intersction points so they are always kept
        nms_scores = np.random.uniform(low=0.9, high=1.0, size=patch_indices.shape[0])
        nms_score_override = self.nms_score_override[patch_indices]
        nms_scores = np.maximum(nms_scores, nms_score_override)
        # TODO: use config to align with inference
        nms_radius = 16
        
        # kept_indces are into the patch_points array
        nmsed_points, kept_indices = graph_utils.nms_points(patch_points, nms_scores, radius=nms_radius, return_indices=True)
        # now this is into the subdivide graph
        nmsed_indices = patch_indices[kept_indices]
        nmsed_point_num = nmsed_points.shape[0]


        # TODO: this shall be in config and tuned
        sample_num = 512  # has to be greater than 1
        sample_weights = self.sample_weights[nmsed_indices]
        # indices into the nmsed points in the patch
        sample_indices_in_nmsed = np.random.choice(
            np.arange(start=0, stop=nmsed_points.shape[0], dtype=np.int32),
            size=sample_num, replace=True, p=sample_weights / np.sum(sample_weights))
        # indices into the subdivided graph
        sample_indices = nmsed_indices[sample_indices_in_nmsed]
        
        # TODO: in config
        radius = 64
        max_nbr_queries = 16  # has to be greater than 1
        nmsed_kdtree = scipy.spatial.KDTree(nmsed_points)
        sampled_points = self.subdivide_points[sample_indices, :]
        # [n_sample, n_nbr]
        # k+1 because the nearest one is always self
        knn_d, knn_idx = nmsed_kdtree.query(sampled_points, k=max_nbr_queries + 1, distance_upper_bound=radius)


        samples = []

        for i in range(sample_num):
            source_node = sample_indices[i]
            valid_nbr_indices = knn_idx[i, knn_idx[i, :] < nmsed_point_num]
            valid_nbr_indices = valid_nbr_indices[1:] # the nearest one is self so remove
            target_nodes = [nmsed_indices[ni] for ni in valid_nbr_indices]  

            ### BFS to find immediate neighbors on graph
            reached_nodes = graph_utils.bfs_with_conditions(self.full_graph_subdivide, source_node, set(target_nodes), radius // self.subdivide_resolution)
            shall_connect = [t in reached_nodes for t in target_nodes]
            ###

            pairs = []
            valid = []
            source_nmsed_idx = sample_indices_in_nmsed[i]
            for target_nmsed_idx in valid_nbr_indices:
                pairs.append((source_nmsed_idx, target_nmsed_idx))
                valid.append(True)

            # zero-pad
            for i in range(len(pairs), max_nbr_queries):
                pairs.append((source_nmsed_idx, source_nmsed_idx))
                shall_connect.append(False)
                valid.append(False)

            samples.append((pairs, shall_connect, valid))

        # Transform points
        # [N, 2]
        nmsed_points -= np.array([x0, y0])[np.newaxis, :]
        # homo for rot
        # [N, 3]
        nmsed_points = np.concatenate([nmsed_points, np.ones((nmsed_point_num, 1), dtype=nmsed_points.dtype)], axis=1)
        trans = np.array([
            [1, 0, -0.5 * self.config.PATCH_SIZE],
            [0, 1, -0.5 * self.config.PATCH_SIZE],
            [0, 0, 1],
        ], dtype=np.float32)
        # ccw 90 deg in img (x, y)
        rot = np.array([
            [0, 1, 0],
            [-1, 0, 0],
            [0, 0, 1],
        ], dtype=np.float32)
        nmsed_points = nmsed_points @ trans.T @ np.linalg.matrix_power(rot.T, rot_index) @ np.linalg.inv(trans.T)
        nmsed_points = nmsed_points[:, :2]
            
        # Add noise
        noise_scale = 1.0  # pixels
        nmsed_points += np.random.normal(0.0, noise_scale, size=nmsed_points.shape)

        return nmsed_points, samples
    

def test_graph_label_generator():
    if not os.path.exists('debug'):
        os.mkdir('debug')

    rgb_path = './cityscale/20cities/region_166_sat.png'
    # Load GT Graph
    gt_graph = pickle.load(open(f"./cityscale/20cities/region_166_refine_gt_graph.p",'rb'))
    rgb = read_rgb_img(rgb_path)
    config = addict.Dict()
    config.PATCH_SIZE = 512
    gen = GraphLabelGenerator(config, gt_graph)
    patch = ((x0, y0), (x1, y1)) = ((64+512, 64), (64+512+config.PATCH_SIZE, 64+config.PATCH_SIZE))
    test_num = 64
    for i in range(test_num):
        rot_index = np.random.randint(0, 4)
        points, samples = gen.sample_patch(patch, rot_index=rot_index)
        rgb_patch = rgb[y0:y1, x0:x1, ::-1].copy()
        rgb_patch = np.rot90(rgb_patch, rot_index, (0, 1)).copy()
        for pairs, shall_connect, valid in samples:
            color = tuple(int(c) for c in np.random.randint(0, 256, size=3))

            for (src, tgt), connected, is_valid in zip(pairs, shall_connect, valid):
                if not is_valid:
                    continue
                p0, p1 = points[src], points[tgt]
                cv2.circle(rgb_patch, p0.astype(np.int32), 4, color, -1)
                cv2.circle(rgb_patch, p1.astype(np.int32), 2, color, -1)
                if connected:
                    cv2.line(
                        rgb_patch,
                        (int(p0[0]), int(p0[1])),
                        (int(p1[0]), int(p1[1])),
                        (255, 255, 255),
                        1,
                    )
        cv2.imwrite(f'debug/viz_{i}.png', rgb_patch)

        



class CityScaleDataset(Dataset):
    def __init__(self, config, is_train):
        self.config = config
        
        rgb_pattern = './cityscale/20cities/region_{}_sat.png'
        keypoint_mask_pattern = './cityscale/processed/keypoint_mask_{}.png'
        road_mask_pattern = './cityscale/processed/road_mask_{}.png'
        gt_graph_pattern = './cityscale/20cities/region_{}_refine_gt_graph.p'
        

        train, val, test = cityscale_data_partition()

        self.is_train = is_train

        train_split = train + val
        test_split = test

        tile_indices = train_split if self.is_train else test_split
        self.tile_indices = tile_indices
        
        # Stores all imgs in memory.
        self.rgbs, self.keypoint_masks, self.road_masks = [], [], []
        # For graph label generation.
        self.graph_label_generators = []

        for tile_idx in tile_indices:
            rgb_path = rgb_pattern.format(tile_idx)
            road_mask_path = road_mask_pattern.format(tile_idx)
            keypoint_mask_path = keypoint_mask_pattern.format(tile_idx)
            self.rgbs.append(read_rgb_img(rgb_path))
            self.road_masks.append(cv2.imread(road_mask_path, cv2.IMREAD_GRAYSCALE))
            self.keypoint_masks.append(cv2.imread(keypoint_mask_path, cv2.IMREAD_GRAYSCALE))

            # graph label gen
            # gt graph: dict for adj list, keys are (r, c) nodes, values are list of (r, c) nodes
            gt_graph_adj = pickle.load(open(gt_graph_pattern.format(tile_idx),'rb'))
            graph_label_generator = GraphLabelGenerator(config, gt_graph_adj)
            self.graph_label_generators.append(graph_label_generator)
            
        
        self.sample_min = SAMPLE_MARGIN
        self.sample_max = IMAGE_SIZE - (self.config.PATCH_SIZE + SAMPLE_MARGIN)

        eval_patches_per_edge = math.ceil((IMAGE_SIZE - 2 * SAMPLE_MARGIN) / self.config.PATCH_SIZE)
        self.eval_patches = []
        for i in range(len(test_split)):
            self.eval_patches += get_patch_info_one_img(
                i, IMAGE_SIZE, SAMPLE_MARGIN, self.config.PATCH_SIZE, eval_patches_per_edge
            )

    def __len__(self):
        if self.is_train:
            return (IMAGE_SIZE // self.config.PATCH_SIZE)**2 * 2500
        else:
            return len(self.eval_patches)

    def __getitem__(self, idx):
        # Sample a patch.
        if self.is_train:
            img_idx = np.random.randint(low=0, high=len(self.rgbs))
            begin_x = np.random.randint(low=self.sample_min, high=self.sample_max+1)
            begin_y = np.random.randint(low=self.sample_min, high=self.sample_max+1)
            end_x, end_y = begin_x + self.config.PATCH_SIZE, begin_y + self.config.PATCH_SIZE
        else:
            # Returns eval patch
            img_idx, (begin_x, begin_y), (end_x, end_y) = self.eval_patches[idx]
        
        # Crop patch imgs and masks
        rgb_patch = self.rgbs[img_idx][begin_y:end_y, begin_x:end_x, :]
        keypoint_mask_patch = self.keypoint_masks[img_idx][begin_y:end_y, begin_x:end_x]
        road_mask_patch = self.road_masks[img_idx][begin_y:end_y, begin_x:end_x]

        # Augmentation
        rot_index = 0
        if self.is_train:
            rot_index = np.random.randint(0, 4)
            # CCW
            rgb_patch = np.rot90(rgb_patch, rot_index, [0,1]).copy()
            keypoint_mask_patch = np.rot90(keypoint_mask_patch, rot_index, [0, 1]).copy()
            road_mask_patch = np.rot90(road_mask_patch, rot_index, [0, 1]).copy()
        
        # Sample graph labels from patch
        patch = ((begin_x, begin_y), (end_x, end_y))
        graph_samples = self.graph_label_generators[img_idx].sample_patch(patch, rot_index)

        
        # rgb: [H, W, 3] 0-255
        # masks: [H, W] 0-1
        return torch.tensor(rgb_patch, dtype=torch.float32), torch.tensor(keypoint_mask_patch, dtype=torch.float32) / 255.0, torch.tensor(road_mask_patch, dtype=torch.float32) / 255.0


if __name__ == '__main__':
    test_graph_label_generator()