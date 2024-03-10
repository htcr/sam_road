import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import math
import tcod
from sklearn.neighbors import KDTree
from skimage.draw import line
import networkx as nx


IMAGE_SIZE = 2048
SAMPLE_MARGIN = 64

def read_rgb_img(path):
    bgr = cv2.imread(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb

def nms_points(points, scores, radius):
    sorted_indices = np.argsort(scores)[::-1]
    sorted_points = points[sorted_indices, :]
    kept = np.ones(sorted_indices.shape[0], dtype=bool)
    tree = KDTree(sorted_points)
    for idx, p in enumerate(sorted_points):
        if not kept[idx]:
            continue
        neighbor_indices = tree.query_radius(p[np.newaxis, :], r=radius)[0]
        kept[neighbor_indices] = False
        kept[idx] = True
    return sorted_points[kept]

# returns (x, y)
def get_points_and_scores_from_mask(mask, threshold):
    rcs = np.column_stack(np.where(mask > threshold))
    xys = rcs[:, ::-1]
    scores = mask[mask > threshold]
    return xys, scores


def draw_points_on_image(image, points, radius):
    """
    Draws points on a square image using OpenCV.

    Parameters:
    - size: The size of the square image (width and height) in pixels.
    - points: A list of tuples, where each tuple represents the (x, y) coordinates of a point in pixel coordinates.
    - radius: The radius of the circles to be drawn for each point, in pixels.

    Returns:
    - A square image with the given points drawn as filled circles.
    """
    
    # Iterate through the list of points
    for point in points:
        cv2.circle(image, point, radius, (0, 255, 0), -1)

    return image


def draw_points_on_grayscale_image(image, points, radius):
    """
    Draws points on a square image using OpenCV.

    Parameters:
    - size: The size of the square image (width and height) in pixels.
    - points: A list of tuples, where each tuple represents the (x, y) coordinates of a point in pixel coordinates.
    - radius: The radius of the circles to be drawn for each point, in pixels.

    Returns:
    - A square image with the given points drawn as filled circles.
    """
    
    # Iterate through the list of points
    for point in points:
        cv2.circle(image, point, radius, 255, -1)

    return image


# takes xy
def is_connected_bresenham(cost, start, end):
    c0, r0 = start
    c1, r1 = end
    rr, cc = line(r0, c0, r1, c1)
    kp_block_radius = 4
    cv2.circle(cost, start, kp_block_radius, 0, -1)
    cv2.circle(cost, end, kp_block_radius, 0, -1)
    
    # mean_cost = np.mean(cost[rr, cc])
    max_cost = np.max(cost[rr, cc])

    cv2.circle(cost, start, kp_block_radius, 255, -1)
    cv2.circle(cost, end, kp_block_radius, 255, -1)

    return max_cost < 255

def create_cost_field(sample_pts, road_mask):
    # road mask shall be uint8 normalized to 0-255
    cost_field = np.zeros(road_mask.shape, dtype=np.uint8)
    kp_block_radius = 4
    for point in sample_pts:
        cv2.circle(cost_field, point, kp_block_radius, 255, -1)
    cost_field = np.maximum(cost_field, 255 - road_mask)
    return cost_field

def extract_graph(keypoint_mask, road_mask):
    kp_candidates, kp_scores = get_points_and_scores_from_mask(keypoint_mask, 128)
    kps_0 = nms_points(kp_candidates, kp_scores, 8)
    kp_candidates, kp_scores = get_points_and_scores_from_mask(road_mask, 128)
    kps_1 = nms_points(kp_candidates, kp_scores, 16)
    kps = np.concatenate([kps_0, kps_1], axis=0)

    cost_field = create_cost_field(kps, road_mask)

    tree = KDTree(kps)
    graph = nx.Graph()
    checked = set()
    for p in kps:
        neighbor_indices = tree.query_radius(p[np.newaxis, :], r=40)[0]
        for n_idx in neighbor_indices:
            n = kps[n_idx]
            start, end = (int(p[0]), int(p[1])), (int(n[0]), int(n[1]))
            if (start, end) in checked:
                continue
            if is_connected_bresenham(cost_field, p, n):
                graph.add_edge(start, end)
            checked.add((start, end))
    return graph

# takes xys    
def visualize_image_and_graph(img, graph):
    # Draw nodes as green squares
    for node in graph.nodes():
        x, y = node
        cv2.rectangle(
            img, (int(x) - 2, int(y) - 2), (int(x) + 2, int(y) + 2), (0, 255, 0), -1
        )
    # Draw edges as white lines
    for start_node, end_node in graph.edges():
        cv2.line(
            img,
            (int(start_node[0]), int(start_node[1])),
            (int(end_node[0]), int(end_node[1])),
            (255, 255, 255),
            1,
        )
    return img
    


rgb_pattern = './cityscale/20cities/region_{}_sat.png'
keypoint_mask_pattern = './cityscale/processed/keypoint_mask_{}.png'
road_mask_pattern = './cityscale/processed/road_mask_{}.png'

index = 0
rgb = read_rgb_img(rgb_pattern.format(index))
road_mask = cv2.imread(road_mask_pattern.format(index), cv2.IMREAD_GRAYSCALE)
keypoint_mask = cv2.imread(keypoint_mask_pattern.format(index), cv2.IMREAD_GRAYSCALE)

graph = extract_graph(keypoint_mask, road_mask)
viz = visualize_image_and_graph(rgb, graph)
cv2.imwrite('test_graph_6.png', viz)

# # kp_candidates, kp_scores = get_points_and_scores_from_mask(keypoint_mask, 128)
# # kps = nms_points(kp_candidates, kp_scores, 8)
# kp_candidates, kp_scores = get_points_and_scores_from_mask(road_mask, 128)
# kps = nms_points(kp_candidates, kp_scores, 16)

# draw_points_on_image(rgb, kps, radius=3)
# cv2.imwrite('test_2.png', rgb)

