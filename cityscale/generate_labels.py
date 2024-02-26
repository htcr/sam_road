import os
import numpy as np 
import shutil
import pickle
import networkx as nx
import cv2


IMAGE_SIZE = 2048
KEYPOINT_RADIUS = 3
ROAD_WIDTH = 3
output_dir = './processed'

def create_directory(dir,delete=False):
    if os.path.isdir(dir) and delete:
        shutil.rmtree(dir)
    os.makedirs(dir,exist_ok=True)


def draw_points_on_image(size, points, radius):
    """
    Draws points on a square image using OpenCV.

    Parameters:
    - size: The size of the square image (width and height) in pixels.
    - points: A list of tuples, where each tuple represents the (x, y) coordinates of a point in pixel coordinates.
    - radius: The radius of the circles to be drawn for each point, in pixels.

    Returns:
    - A square image with the given points drawn as filled circles.
    """
    
    # Create a square image of the given size, initialized to zeros (black), with one channel (grayscale), and dtype uint8
    image = np.zeros((size, size), dtype=np.uint8)

    # Iterate through the list of points
    for point in points:
        # Draw each point as a filled circle on the image
        # The circle is drawn with center at 'point', radius as specified, color 255 (white), and filled (thickness=-1)
        cv2.circle(image, point, radius, 255, -1)

    return image

def draw_line_segments_on_image(size, line_segments, width):
    """
    Draws line segments on a square image using OpenCV.

    Parameters:
    - size: The size of the square image (width and height) in pixels.
    - line_segments: A list of tuples, where each tuple represents a line segment as ((x1, y1), (x2, y2)).
    - width: The width of the lines to be drawn, in pixels.

    Returns:
    - A square image with the given line segments drawn.
    """
    
    # Create a square image of the given size, initialized to zeros (black)
    # with one channel (grayscale), and dtype uint8
    image = np.zeros((size, size), dtype=np.uint8)

    # Iterate through the list of line segments
    for segment in line_segments:
        # Unpack the start and end points of the line segment
        (x1, y1), (x2, y2) = segment

        # Draw the line segment on the image
        # The line is drawn with color 255 (white) and the specified width
        cv2.line(image, (x1, y1), (x2, y2), 255, width)

    return image


create_directory(output_dir,delete=True)

for tile_index in range(180):
    print(f'Processing cityscale tile {tile_index}.')
    vertices = []
    edges = []
    vertex_flag = True

    # Load GT Graph
    gt_graph = pickle.load(open(f"./20cities/region_{tile_index}_refine_gt_graph.p",'rb'))
    graph = nx.Graph()  # undirected
    for n, neis in gt_graph.items():
        for nei in neis:
            graph.add_edge((int(n[1]), int(n[0])), (int(nei[1]), int(nei[0])))
    
    # Collect key nodes (degree != 2)
    key_nodes = []
    for node, degree in graph.degree():
        if degree != 2:
            key_nodes.append(node)

    # Create key point mask
    keypoint_mask = draw_points_on_image(size=IMAGE_SIZE, points=key_nodes, radius=KEYPOINT_RADIUS)

    # Create road mask
    road_mask = draw_line_segments_on_image(
        size=IMAGE_SIZE, line_segments=graph.edges(), width=ROAD_WIDTH)

    cv2.imwrite(os.path.join(output_dir, f'keypoint_mask_{tile_index}.png'), keypoint_mask)
    cv2.imwrite(os.path.join(output_dir, f'road_mask_{tile_index}.png'), road_mask)