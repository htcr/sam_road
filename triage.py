import os
import pickle
import random
import cv2
import numpy as np


def visualize_image_and_graph(img, nodes, edges, viz_img_size=512):
    # img is rgb
    # Node coordinates in [0, 1], representing the normalized (r, c)
    # (r, c) -> (x, y)
    nodes = nodes[:, ::-1]

    # Resize the image to the specified visualization size, RGB->BGR
    img = cv2.resize(img, (viz_img_size, viz_img_size))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Draw edges
    for edge in edges:
        start_node = nodes[edge[0]] * viz_img_size
        end_node = nodes[edge[1]] * viz_img_size
        cv2.line(
            img,
            (int(start_node[0]), int(start_node[1])),
            (int(end_node[0]), int(end_node[1])),
            (15, 160, 253),
            4,
        )
    
    # Draw nodes
    for node in nodes:
        x, y = node * viz_img_size
        cv2.circle(img, (int(x), int(y)), 4, (0, 255, 255), -1)

    return img


def rasterize_graph(nodes, edges, viz_img_size, dilation_radius):
    # Rasterize the graph.
    # Node coordinates in [0, 1], representing the normalized (r, c)

    # (r, c) -> (x, y)
    nodes = nodes[:, ::-1]

    # Creates the canvas
    img = np.zeros((viz_img_size, viz_img_size, 3), dtype=np.uint8)

    # Draw predicted nodes as white squares
    for node in nodes:
        x, y = node * viz_img_size
        cv2.rectangle(
            img,
            (int(x) - dilation_radius, int(y) - dilation_radius),
            (int(x) + dilation_radius, int(y) + dilation_radius),
            (255, 255, 255),
            -1,
        )

    # Draw predicted edges as white lines
    for edge in edges:
        start_node = nodes[edge[0]] * viz_img_size
        end_node = nodes[edge[1]] * viz_img_size
        cv2.line(
            img,
            (int(start_node[0]), int(start_node[1])),
            (int(end_node[0]), int(end_node[1])),
            (255, 255, 255),
            dilation_radius * 2,
        )

    return img


def visualize_pred_gt_pair(result):
    img = cv2.imread(result["img_path"])
    pred_img = visualize_image_and_graph(
        img, result["pred_nodes"], result["pred_edges"]
    )
    gt_img = visualize_image_and_graph(img, result["gt_nodes"], result["gt_edges"])
    pair_img = np.concatenate((pred_img, gt_img), axis=1)
    return pair_img


if __name__ == '__main__':
    # Deserializing the list from the binary file
    with open("inference_results.pickle", "rb") as file:
        inference_results = pickle.load(file)

    result_length = len(inference_results)
    worst_ratio = 0.1
    sample_num = 200

    output_dir = "triage/below_average"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    sorted_results = sorted(inference_results, key=lambda x: -x["smd"])

    # selected_results = sorted_results[:int(result_length * worst_ratio)]
    selected_results = [x for x in inference_results if x["smd"] > 0.05]

    sampled_results = random.sample(selected_results, sample_num)

    sampled_results = sorted(sampled_results, key=lambda x: -x["smd"])

    for x in sampled_results:
        pair_img = visualize_pred_gt_pair(x)
        smd = x["smd"]
        img_name = os.path.basename(x["img_path"])
        output_name = f"smd_{smd:.6f}_{img_name}"
        cv2.imwrite(os.path.join(output_dir, output_name), pair_img)
