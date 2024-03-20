import numpy as np
from scipy import interpolate
from shapely.geometry import LineString
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import networkx as nx
from sklearn.cluster import DBSCAN
from shapely.geometry import Point, LineString
from shapely.strtree import STRtree
from collections import deque
import unittest

import igraph as ig
import rtree
import scipy


def inspect_graph(node_array, edge_array):
    # node_array: [N_node, 2] coordinates of nodes.
    # edge_array: [N_edge, 2] (src_idx, dst_idx) tuples.
    edge_set = set()
    for edge in edge_array:
        src, dst = edge[0], edge[1]
        edge_set.add((src, dst))
    one_way_edge_count = 0
    for src, dst in edge_set:
        if (dst, src) not in edge_set:
            one_way_edge_count += 1
    print(f"DEBUG: One-way-edge count {one_way_edge_count}")

    node_dist_matrix = node_array[:, np.newaxis, :] - node_array[np.newaxis, :, :]
    node_dist_matrix = np.sum(node_dist_matrix**2, axis=-1)**0.5
    node_num = node_array.shape[0]
    pair_is_close = node_dist_matrix < 0.1
    duplicate_node_count = (np.sum(pair_is_close.astype(int)) - node_num) / 2

    print(f"DEBUG: duplicate_node_count: {duplicate_node_count}")


def filter_nodes(node_array, edge_array, keep_node):
    # Filters nodes, removes edges connecting to them,
    # and updates indices in edges.
    # node_array: [N_node, 2] coordinates of nodes.
    # edge_array: [N_edge, 2] (src_idx, dst_idx) tuples.
    # keep_node: [N_node, ] boolean mask.
    new_nodes = node_array[keep_node, :]
    old_node_num = node_array.shape[0]
    keep_indices = np.where(keep_node)[0]
    new_node_num = keep_indices.shape[0]
    old_to_new_indices = np.full((old_node_num,) , fill_value=-1, dtype=np.int32)
    old_to_new_indices[keep_indices] = np.arange(start=0, stop=new_node_num, step=1, dtype=np.int32)
    # Replaces node indices in edge_array
    edge_nodes = edge_array.flatten()
    new_edge_nodes = old_to_new_indices[edge_nodes]
    new_edges = new_edge_nodes.reshape(-1, 2)
    # Filters disconnected edge
    keep_edge = np.all(new_edges > -1, axis=-1)
    new_edges = new_edges[keep_edge, :]
    return new_nodes, new_edges



def edge_list_to_adj_table(edges):
    # edges: [[src_idx, dst_idx], ...] node indices must start from 0 and
    # be continuous.
    # Returns:
    # adj_table: list of sets. len(adj_table) = num_nodes, adj_table[i] 
    # = neighbor node indices of node i. Empty if no neighbors.
    nodes = set()
    for edge in edges:
        start_idx, end_idx = edge[0], edge[1]
        nodes.add(start_idx)
        nodes.add(end_idx)
    node_num = len(nodes)
    adj_table = [set() for i in range(node_num)]
    for edge in edges:
        start_idx, end_idx = edge[0], edge[1]
        adj_table[start_idx].add(end_idx)
    return adj_table


def edge_list_to_adj_table(nodes, edges):
    # edges: [[src_idx, dst_idx], ...] node indices must start from 0 and
    # be continuous.
    # Returns:
    # adj_table: list of sets. len(adj_table) = num_nodes, adj_table[i] 
    # = neighbor node indices of node i. Empty if no neighbors.
    node_num = len(nodes)
    adj_table = [set() for i in range(node_num)]
    for edge in edges:
        start_idx, end_idx = edge[0], edge[1]
        adj_table[start_idx].add(end_idx)
    return adj_table
    

def trace_segment(start_edge, adj_table):
    segment_nodes = [start_edge[0], start_edge[1]]
    visited_nodes = set(segment_nodes)
    while True:
        curr_node = segment_nodes[-1]
        unvisited_neighbor_num = 0
        next_node = -1
        for neighbor in adj_table[curr_node]:
            if neighbor not in visited_nodes:
                unvisited_neighbor_num += 1
                next_node = neighbor
        if unvisited_neighbor_num != 1:
            break
        segment_nodes.append(next_node)
        visited_nodes.add(next_node)
    return segment_nodes


def unique_edge(src, dst):
    return (min(src, dst), max(src, dst))


def find_segments_in_road_graph(adj_table):
    # adj_table: road graph represented as adj table of nodes, as produced
    # by edge_list_to_adj_table.
    # Returns:
    # segments: list of lists, segments[i] = list of nodes forming the i-th
    # segment.
    segments = list()
    visited_edges = set()
    # Goes over each edge in the graph.
    node_num = len(adj_table)
    for node in range(node_num):
        # See if node is a segment end point.
        if len(adj_table[node]) == 2:
            continue
        # Trace down an unvisited edge.
        for neighbor in adj_table[node]:
            edge = unique_edge(node, neighbor)
            if edge in visited_edges:
                continue
            # Needs edge direction for correct tracing.
            segment = trace_segment((node, neighbor), adj_table)
            for i in range(len(segment) - 1):
                visited_edge = unique_edge(segment[i], segment[i+1])
                visited_edges.add(visited_edge)
            segments.append(segment)

    all_unique_edges = set()
    for node in range(node_num):
        for neighbor in adj_table[node]:
            all_unique_edges.add(unique_edge(node, neighbor))
    total_edge_num = len(all_unique_edges)
    if len(visited_edges) < total_edge_num:
        diff = total_edge_num - len(visited_edges)
        print(f'!!! Warning: Isolated loop detected. {diff} edges are missing.')

    return segments


def normalize_segments(coords, segments):
    # A segment has two endpoints. Makes sure the one with smaller x goes
    # first. If tie, The one with smaller y goes first.
    # coords: [N_node, 2] node coords.
    # segments: [list_of_segment_node_indices, ...]
    normalized_segments = []
    for i in range(len(segments)):
        segment = segments[i]
        first = coords[segment[0], :]
        last = coords[segment[-1], :]
        
        if first[0] > last[0] or (
            first[0] == last[0] and first[1] > last[1]):
            segment = segment[::-1]
            
        normalized_segments.append(segment)
        
    return normalized_segments
    

def get_resampled_polylines(coords, segments, num_points):
    # Uniformly resamples each polyline defined by segments to num_points.
    # coords: [N_node, 2] node coords.
    # segments: [list_of_segment_node_indices, ...]
    # Returns:
    # list of [num_points, 2].
    
    resampled_polylines = []
    
    for segment in segments:
        polyline_coords = coords[segment]
        polyline = LineString(polyline_coords)
        
        # Uniform parameter values
        dists = np.linspace(0, polyline.length, num_points)
        
        # Resample polyline
        resampled_polyline = np.array([list(polyline.interpolate(d).coords)[0] for d in dists])
        
        resampled_polylines.append(resampled_polyline)
    
    return resampled_polylines


def get_polylines_from_road_graph(coords, edges, num_points_per_segment):
    adj_table = edge_list_to_adj_table(edges)
    segments = find_segments_in_road_graph(adj_table)
    segments = normalize_segments(coords, segments)
    polylines = get_resampled_polylines(
        coords, segments, num_points_per_segment)
    return polylines


def get_polyline_connectivity(polylines, dist_threhsold):
    # Gets undirected connectivity between polylines by checking if endpoints
    # overlap.
    # polylines: list of [N_points, 2] arrays.
    # dist_threshold: points closer than this are considered connected.
    # Returns:
    # connected_pairs: [N_pairs, 2] (src_idx, dst_idx). The reverse pair will also be
    # here.
    # connected_point_indices: [N_pairs, 2] indices of overlapping points in
    # their polylines.
    connected_pairs = []
    connected_point_indices = []
    polyline_num = len(polylines)
    for i in range(polyline_num):
        for j in range(i+1, polyline_num):
            a, b = polylines[i], polylines[j]
            endpoint_indices = [
                (0, 0), (0, b.shape[0]-1),
                (a.shape[0]-1, 0), (a.shape[0]-1, b.shape[0]-1)]
            for a_idx, b_idx in endpoint_indices:
                if np.linalg.norm(a[a_idx] - b[b_idx]) < dist_threhsold:
                    connected_pairs.append((i, j))
                    connected_pairs.append((j, i))
                    connected_point_indices.append((a_idx, b_idx))
                    connected_point_indices.append((b_idx, a_idx))
    return connected_pairs, connected_point_indices


def visualize_polylines(image, polylines):
    # Draws all polylines on the image, each with a different color.
    # image: [H, W, C]
    # polylines: list of [length, 2] float arrays, each entry a (row, col)
    # tuple in pixel coordinates.

    # Generate a color map with as many colors as there are polylines
    cmap = plt.cm.get_cmap('hsv', len(polylines))

    # Display the image
    plt.imshow(image)
    
    # Draw each polyline with a different color
    for idx, polyline in enumerate(polylines):
        plt.plot(polyline[:, 1], polyline[:, 0], color=cmap(idx), linewidth=2)
    
    plt.show()


def visualize_polyline_graph(
        image, polylines, connected_pairs, connected_point_indices):
    # Draws each connected pair, one by one, red->green
    for pair_idx, (pair, endpoints) in enumerate(zip(connected_pairs, connected_point_indices)):
        print(f'pair {pair_idx+1}/{len(connected_pairs)}')
        plt.imshow(image)
        idx_a, idx_b = pair
        line_a, line_b = polylines[idx_a], polylines[idx_b]
        plt.plot(line_a[:, 1], line_a[:, 0], color='red', linewidth=2)
        plt.plot(line_b[:, 1], line_b[:, 0], color='green', linewidth=2)
        end_a, end_b = line_a[endpoints[0], :], line_b[endpoints[1], :]
        plt.plot(end_a[1], end_a[0], marker='o', markersize=8, color='blue')
        plt.plot(end_b[1], end_b[0], marker='o', markersize=8, color='blue')
        plt.show()
    

## Utils for aggregating the large map.
def remove_isolate_nodes(nodes, edges):
    node_indices = np.arange(nodes.shape[0])
    graph = nx.Graph()
    graph.add_nodes_from(node_indices)
    graph.add_edges_from(edges)
    
    isolated_nodes = list(nx.isolates(graph))
    graph.remove_nodes_from(isolated_nodes)

    remaining_node_indices = list(graph.nodes())
    remaining_node_indices.sort()
    remaining_nodes = nodes[remaining_node_indices, :]

    new_graph = nx.convert_node_labels_to_integers(graph)
    new_edges = list(new_graph.edges())

    return remaining_nodes, new_edges


def merge_nodes(nodes, edges, distance_threshold):
    clustering = DBSCAN(eps=distance_threshold, min_samples=1).fit(nodes)
    node_cluster_indices = clustering.labels_
    num_clusters = len(np.unique(node_cluster_indices))
    cluster_centers = np.zeros((num_clusters, 2), dtype=np.float32)
    cluster_size = np.zeros((num_clusters, ), dtype=np.float32)
    for node_index, node in enumerate(nodes):
        cluster_index = node_cluster_indices[node_index]
        cluster_centers[cluster_index, :] += node
        cluster_size[cluster_index] += 1
    cluster_centers = cluster_centers / cluster_size[:, np.newaxis]
    unique_edges = set()
    for (start, end) in edges:
        new_start = node_cluster_indices[start]
        new_end = node_cluster_indices[end]

        # Removes self-loops
        if new_start == new_end:
            continue

        new_edge = (min(new_start, new_end), max(new_start, new_end))
        unique_edges.add(new_edge)
    return cluster_centers, list(unique_edges)


def split_edges(nodes, edges, distance_threshold):
    points = [Point(x, y) for x, y in nodes]
    point_tree = STRtree(points)

    edge_queue = deque()
    for edge in edges:
        edge_queue.appendleft(edge)

    new_edges = list()

    while len(edge_queue) > 0:
        start, end = edge_queue.pop()
        start_pt, end_pt = nodes[start, :], nodes[end, :]
        line_segment = LineString([start_pt, end_pt])
        nearby_region = line_segment.buffer(distance=distance_threshold, cap_style='flat')
        nearby_point_indices = point_tree.query(nearby_region).tolist()
        min_dist = distance_threshold + 88.8
        nearest_point_index = None
        for index in nearby_point_indices:
            if index == start or index == end:
                continue
            point = points[index]
            dist = line_segment.distance(point)
            if dist < min_dist:
                min_dist, nearest_point_index = dist, index

        if nearest_point_index is None or min_dist >= distance_threshold:
            new_edges.append((start, end))
            continue
        else:
            e1, e2 = (start, nearest_point_index), (nearest_point_index, end)
            edge_queue.appendleft(e1)
            edge_queue.appendleft(e2)
    
    # TODO(congrui): share the edge dedup logic
    unique_edges = set()
    for (start, end) in new_edges:
        new_edge = (min(start, end), max(start, end))
        unique_edges.add(new_edge)
    
    return nodes, list(unique_edges)


def combine_graphs(graphs):
    # graphs: list of (nodes, edges)
    offset = 0
    combined_nodes, combined_edges = [], []
    for nodes, edges in graphs:
        combined_nodes.append(nodes)
        edges_np = np.array(edges)
        edges_np += offset
        combined_edges.append(edges_np)
        offset += nodes.shape[0]
    combined_nodes = np.concatenate(combined_nodes, axis=0)
    combined_edges = np.concatenate(combined_edges, axis=0)
    return combined_nodes, combined_edges


def merge_into_large_graph(nodes, edges, merge_node_dist_thresh, split_edge_dist_thresh):
    nodes1, edges1 = remove_isolate_nodes(nodes, edges)
    nodes2, edges2 = merge_nodes(nodes1, edges1, distance_threshold=merge_node_dist_thresh)
    nodes3, edges3 = split_edges(nodes2, edges2, distance_threshold=split_edge_dist_thresh)
    nodes4, edges4 = remove_isolate_nodes(nodes3, edges3)
    return nodes4, edges4


def convert_to_sat2graph_format(nodes, edges):
    # Converts a graph to the same format as the labels
    # in Sat2Graph.
    # nodes: [N_node, 2] of the (row, col) image coordinates.
    # edges: [N_edge, 2] pairs of (start, end) node indices.
    # Returns: A dict. Keys are (row, col) coordinates of each node. Float inputs will be rounded to int.
    # Values are lists, each item being a (row, col) of a neighbor node.
    # Edges are not directed. Input edges will be combined with reverse edges.
    reverse_edges = edges[:, ::-1]
    all_edges = np.concatenate((edges, reverse_edges), axis=0)

    adj_table = edge_list_to_adj_table(nodes, all_edges)
    
    int_nodes = [(round(x), round(y)) for x, y in nodes]
    
    result = dict()
    for node_idx, neighbor_indices in enumerate(adj_table):
        # Notice, we expect the input graph has gone through node-merging so
        # there shouldn't be two nodes at the same pixel location.
        key = int_nodes[node_idx]
        value = [int_nodes[neighbor_idx] for neighbor_idx in neighbor_indices]
        result[key] = value
    return result


def convert_from_sat2graph_format(graph):
    # Converts a graph from the Sat2Graph label format to nodes and edges.
    # graph: A dict. Keys are (row, col) coordinates of each node. Float inputs will be rounded to int.
    # Values are lists, each item being a (row, col) of a neighbor node.
    # Edges are not directed and ARE NOT DE-DUPLICATED.
    # Returns:
    # nodes: [N_node, 2] of the (row, col) image coordinates.
    # edges: [N_edge, 2] pairs of (start, end) node indices.
    node_to_idx = dict()
    for node, neighbors in graph.items():
        if node not in node_to_idx.keys():
            node_to_idx[node] = len(node_to_idx)
        for neighbor in neighbors:
            if neighbor not in node_to_idx.keys():
                node_to_idx[neighbor] = len(node_to_idx)
    
    edges = list()
    for node, neighbors in graph.items():
        for neighbor in neighbors:
            src_idx, dst_idx = node_to_idx[node], node_to_idx[neighbor]
            edges.append((src_idx, dst_idx))
    
    num_nodes = len(node_to_idx)
    nodes = [None] * num_nodes
    for node, idx in node_to_idx.items():
        nodes[idx] = node
    return np.array(nodes), edges


def convert_from_nx(graph):
    # nx graph, node being (x, y)
    # Returns:
    # nodes: [N_node, 2] of the (row, col) image coordinates.
    # edges: [N_edge, 2] pairs of (start, end) node indices.
    node_to_idx = dict()
    nodes = list()
    edges = list()
    for node in graph.nodes():
        if node not in node_to_idx.keys():
            node_to_idx[node] = len(node_to_idx)
        x, y = node
        nodes.append((y, x))  # to rc
    for node_0, node_1 in graph.edges():
        edges.append((node_to_idx[node_0], node_to_idx[node_1]))
    
    return np.array(nodes), np.array(edges)


### igraph utils for performance

def igraph_from_adj_dict(graph, coord_transform):
    # Edges will be de-duped
    nodes, edges = convert_from_sat2graph_format(graph)
    n_vertices = nodes.shape[0]
    if n_vertices == 0:
        nodes = np.zeros((0, 2), dtype=nodes.dtype)
    edges = set([(min(src, tgt), max(src, tgt)) for src, tgt in edges])
    g = ig.Graph(n_vertices, list(edges))
    try:
        g.vs['point'] = coord_transform(nodes)  # to xy
    except Exception:
        print("==================")
        print(nodes.shape)
        print(nodes)
        import pdb
        pdb.set_trace()
    return g

def get_line_bbox(line):
    (x0, y0), (x1, y1) = line
    l = min(x0, x1) - 1
    b = min(y0, y1) - 1
    r = max(x0, x1) + 1
    t = max(y0, y1) + 1
    return (l, b, r, t)

def find_intersection(segment1, segment2):
    """
    Finds the intersection point of two line segments, if it exists.

    Parameters:
        segment1 (tuple): A tuple representing the first line segment ((x1, y1), (x2, y2)).
        segment2 (tuple): A tuple representing the second line segment ((x3, y3), (x4, y4)).

    Returns:
        A tuple (x, y) representing the intersection point, or None if there is no intersection.
    """
    (x1, y1), (x2, y2) = segment1
    (x3, y3), (x4, y4) = segment2
    line1 = LineString([segment1[0], segment1[1]])
    line2 = LineString([segment2[0], segment2[1]])

    # Check for intersection
    intersection = line1.intersection(line2)

    if not intersection.is_empty and intersection.geom_type == 'Point':
        if not (
            intersection.equals(Point(x1, y1)) or
            intersection.equals(Point(x2, y2)) or
            intersection.equals(Point(x3, y3)) or
            intersection.equals(Point(x4, y4))
        ):
            return (intersection.x, intersection.y)
    # geom_type could be line if two parallel lines overlap
    # or just no intersection
    # or intersection is at endpoints
    return None

def find_crossover_points(graph):
    # takes igraph
    # y axis shall point upwards for rtree to work properly
    # crossover points are counted twice: A cross B, B cross A
    # - which is fine for now just be aware
    points = graph.vs['point']
    edges = graph.es
    lines = [(points[edge.source], points[edge.target]) for edge in edges]
    line_bboxes = [get_line_bbox(line) for line in lines]
    line_index = rtree.index.Index()
    for idx, bbox in enumerate(line_bboxes):
        line_index.insert(idx, bbox)

    crossover_points = []
    tested_pairs = set()
    for i, line_0 in enumerate(lines):
        bbox = line_bboxes[i]
        nearby_indices = list(line_index.intersection(bbox))
        for ni in nearby_indices:
            pair = (min(i, ni), max(i, ni))
            if pair in tested_pairs:
                continue
            line_1 = lines[ni]
            itsc = find_intersection(line_0, line_1)
            if itsc is not None:
                crossover_points.append(itsc)
            tested_pairs.add(pair)

    return crossover_points

def subdivide_graph(graph, resolution):
    # takes igraph
    new_points = [p for p in graph.vs['point']]
    new_edges = []
    for edge in graph.es:
        p0, p1 = graph.vs['point'][edge.source], graph.vs['point'][edge.target]
        length = np.linalg.norm(p1 - p0) 
        sample_pieces = max(1, int(length / resolution))
        # [N, ]
        samples = np.linspace(0.0, 1.0, sample_pieces + 1, endpoint=True)
        # [N, 2] = [1, 2] + [N, 1] @ [1, 2]
        sampled_pts = np.expand_dims(np.array(p0), axis=0) + np.expand_dims(samples, axis=1) @ np.expand_dims(p1 - p0, axis=0)
        # [N-2, 2]
        sampled_pts = sampled_pts[1:-1, :]
        new_point_indices = []
        for new_pt in sampled_pts:
            new_point_indices.append(len(new_points))
            new_points.append(new_pt)
        new_edges_sources = [edge.source] + new_point_indices
        new_edges_targets = new_point_indices + [edge.target]
        new_edges += list(zip(new_edges_sources, new_edges_targets))

    new_graph = ig.Graph(len(new_points), new_edges)
    new_graph.vs['point'] = np.array(new_points)
    return new_graph
        
def nms_points(points, scores, radius, return_indices=False):
    # if score > 1.0, the point is forced to be kept regardless
    sorted_indices = np.argsort(scores)[::-1]
    sorted_points = points[sorted_indices, :]
    sorted_scores = scores[sorted_indices]
    kept = np.ones(sorted_indices.shape[0], dtype=bool)
    tree = scipy.spatial.KDTree(sorted_points)
    for idx, p in enumerate(sorted_points):
        if not kept[idx]:
            continue
        # neighbor_indices = tree.query_radius(p[np.newaxis, :], r=radius)[0]
        neighbor_indices = tree.query_ball_point(p, r=radius)
        neighbor_scores = sorted_scores[neighbor_indices]
        keep_nbr = np.greater(neighbor_scores, 1.0)
        kept[neighbor_indices] = keep_nbr
        kept[idx] = True
    if return_indices:
        return sorted_points[kept], sorted_indices[kept]
    else:
        return sorted_points[kept]

    
def bfs_with_conditions(graph, start_node, stop_nodes, max_depth):
    """
    Perform BFS on an igraph graph (directed or undirected) from a given start node.
    The search stops if it visits a node from a given set of stop nodes or if the depth reaches a threshold.
    The function returns the set of visited nodes, including stop nodes if encountered.
    
    Args:
    - graph (ig.Graph): The graph to search.
    - start_node (int): The index of the node to start the BFS from.
    - stop_nodes (set): A set of node indices where the search will stop if visited.
    - max_depth (int): The maximum depth to search.
    
    Returns:
    - set: The set of visited node indices.
    """
    visited = set()  # To keep track of visited nodes
    queue = deque()
    queue.append((start_node, 0))  # Queue of (node, depth)
    
    while queue:
        current_node, current_depth = queue.popleft()  # Dequeue the next node and its depth
        
        # Mark node as visited
        visited.add(current_node)
        
        # Check if the current node is a stop node or if the current depth exceeds max_depth
        if current_node in stop_nodes or current_depth >= max_depth:
            # Stop condition met, do not extend
            continue  
        
        # Get neighbors and enqueue them with incremented depth, considering all edges
        neighbors = graph.neighbors(current_node, mode="all")
        for neighbor in neighbors:
            if neighbor not in visited:
                queue.append((neighbor, current_depth + 1))
    
    return visited



##### Unit tests #####
class TestGraphUtils(unittest.TestCase):
    def test_remove_isolated_nodes(self):
        nodes = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        edges = [[0, 2]]
        new_nodes, new_edges = remove_isolate_nodes(nodes, edges)
        gt_new_nodes = np.array([[0.0, 0.0], [2.0, 2.0]])
        gt_new_edges = np.array([[0, 1]])
        np.testing.assert_array_equal(new_nodes, gt_new_nodes)
        np.testing.assert_array_equal(np.array(new_edges), gt_new_edges)

    def test_merge_nodes(self):
        nodes = np.array([[0.0, 0.0], [1.0, 1.0], [1.1, 1.1], [2.0, 2.0], [0.1, 0.1]])
        edges = [[0, 1], [1, 2], [1, 3], [2, 3], [2, 4]]
        new_nodes, new_edges = merge_nodes(nodes, edges, 0.2)
        gt_new_nodes = np.array([[0.05, 0.05], [1.05, 1.05], [2.0, 2.0]])
        gt_new_edges = np.array([[0, 1], [1, 2]])
        np.testing.assert_almost_equal(new_nodes, gt_new_nodes)
        np.testing.assert_array_equal(np.array(new_edges), gt_new_edges)

    def test_split_edges(self):
        nodes = np.array([[0.0, 0.0], [1.01, 1.01], [2.0, 2.0], [2.0, 0.0]])
        edges = [[0, 1], [1, 2], [0, 2], [2, 3]]
        new_nodes, new_edges = split_edges(nodes, edges, 0.2)
        gt_new_nodes = nodes
        gt_new_edges = np.array([[0, 1], [1, 2], [2, 3]])
        np.testing.assert_almost_equal(new_nodes, gt_new_nodes)
        np.testing.assert_array_equal(np.array(new_edges), gt_new_edges)

    def test_combine_graphs(self):
        nodes0 = np.array([[0.0, 0.0], [1.0, 0.0]])
        edges0 = [[0, 1]]
        nodes1 = np.array([[2.0, 2.0], [3.0, 3.0]])
        edges1 = [[0, 1]]
        new_nodes, new_edges = combine_graphs([(nodes0, edges0), (nodes1, edges1)])
        gt_new_nodes = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 2.0], [3.0, 3.0]])
        gt_new_edges = np.array([[0, 1], [2, 3]])
        np.testing.assert_almost_equal(new_nodes, gt_new_nodes)
        np.testing.assert_array_equal(np.array(new_edges), gt_new_edges)

    def test_buffer_region(self):
        start_pt, end_pt = np.array([0.0, 0.0]), np.array([10.0, 0.0])
        line_segment = LineString([start_pt, end_pt])
        nearby_region = line_segment.buffer(distance=2.0, cap_style='flat')

        # Get the vertices of the polygon as a list of tuples
        vertices_list = list(nearby_region.exterior.coords)
        # Convert the list of tuples to a NumPy array
        vertices_array = np.array(vertices_list)

        gt_vertices = np.array([[10.0, 2.0], [10.0, -2.0], [0.0, -2.0], [0.0, 2.0], [10.0, 2.0]])
        np.testing.assert_almost_equal(vertices_array, gt_vertices)

    def test_convert_to_sat2graph_format(self):
        nodes = np.array([[0.0, 0.0], [1.1, 1.1], [1.6, 1.6]])
        edges = np.array([[0, 1], [1, 2]])
        result = convert_to_sat2graph_format(nodes, edges)
        gt_result = {(0, 0): [(1, 1)], (1, 1): [(0, 0), (2, 2)], (2, 2): [(1, 1)]}
        for k, v in result.items():
            self.assertTrue(k in gt_result.keys())
            self.assertSetEqual(set(v), set(gt_result[k]))

    def test_convert_from_sat2graph_format(self):
        graph = {(0, 0): [(1, 1)], (1, 1): [(0, 0), (2, 2)], (2, 2): [(1, 1)]}
        nodes, edges = convert_from_sat2graph_format(graph)
        gt_nodes = np.array([[0, 0], [1, 1], [2, 2]])
        gt_edges = np.array([[0, 1], [1, 0], [1, 2], [2, 1]])
        np.testing.assert_almost_equal(nodes, gt_nodes)
        np.testing.assert_almost_equal(np.array(edges), gt_edges)

    def test_convert_from_nx(self):
        graph = nx.Graph()
        graph.add_edge((1, 2), (3, 4))
        graph.add_edge((3, 4), (5, 6))
        nodes, edges = convert_from_nx(graph)
        gt_nodes = np.array([[2, 1], [4, 3], [6, 5]])
        gt_edges = np.array([[0, 1], [1, 2]])
        np.testing.assert_almost_equal(nodes, gt_nodes)
        np.testing.assert_almost_equal(edges, gt_edges)

    def test_igraph_from_sat2graph_format(self):
        adj = {
            (1, 2) : [(3, 4), (5, 6)],
            (3, 4) : [(1, 2), (5, 6)],
        }
        rc2xy = lambda x : x[:, ::-1]
        g = igraph_from_adj_dict(adj, rc2xy)
        self.assertEqual(len(g.es), 3)
        self.assertEqual(len(g.vs), 3)
        self.assertEqual(g.vs[0]['point'][0], 2)
        self.assertEqual(g.vs[0]['point'][1], 1)

    def test_find_crossover_points(self):
        adj = {
            (0, 1) : [(10, 1), ],
            (2, -2) : [(2, 10), ],
            (10, 1) : [(20, 1), ],
        }
        rc2xy = lambda x : x[:, ::-1]
        g = igraph_from_adj_dict(adj, rc2xy)
        pts = find_crossover_points(g)
        self.assertEqual(len(pts), 1)
        gt = np.array([1.0, 2.0])
        pd = np.array(pts[0])
        np.testing.assert_almost_equal(gt, pd)

    def test_subdivide_graph(self):
        adj = {
            (0, 0) : [(10, 0), ],
            (10, 0) : [(20, 0), ]
        }
        rc2xy = lambda x : x[:, ::-1]
        g = igraph_from_adj_dict(adj, rc2xy)
        g1 = subdivide_graph(g, resolution=2.0)
        self.assertEqual(len(g1.vs['point']), 11)
        self.assertEqual(len(g1.es), 10)


if __name__ == '__main__':
    unittest.main()




