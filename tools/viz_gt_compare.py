"""
可视化对比 graph_gt.pickle (评测GT) vs refine_gt_graph.p (训练GT, 裁剪后)
画在 sat img 上, 看 black edge region 差异
"""
import os, sys, pickle, json
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.chdir('/Users/highee/research/sam_road')

BASE = 'datasets/didi/xian/2019_400'

# 黑边 region + 非黑边 region 各看几个
black_ids = ['373', '376', '370', '377', '17']  # 黑边
normal_ids = ['171', '304', '100', '92']  # 非黑边(掉幅大/正常)

def draw_graph_on_sat(sat, adj, edge_color=(0,255,255), node_color=(255,0,0), node_r=3, edge_w=2):
    img = sat.copy()
    drawn = set()
    for node, neighbors in adj.items():
        for nb in neighbors:
            edge = tuple(sorted((node, nb)))
            if edge in drawn: continue
            drawn.add(edge)
            p0 = (int(node[1]), int(node[0]))
            p1 = (int(nb[1]), int(nb[0]))
            cv2.line(img, p0, p1, edge_color, edge_w)
    for node in adj.keys():
        cv2.circle(img, (int(node[1]), int(node[0])), node_r, node_color, -1)
    return img

all_ids = black_ids + normal_ids
fig, axes = plt.subplots(len(all_ids), 3, figsize=(12, 4*len(all_ids)))

for r, img_id in enumerate(all_ids):
    sat = cv2.imread(f'{BASE}/region_{img_id}_sat.png')[:,:,::-1]
    # graph_gt.pickle (评测GT)
    gt_path = f'{BASE}/region_{img_id}_graph_gt.pickle'
    if not os.path.exists(gt_path):
        gt_path = f'{BASE}/region_{img_id}_graph_gt.p'
    gt_adj = pickle.load(open(gt_path, 'rb')) if os.path.exists(gt_path) else {}
    # refine_gt_graph.p (训练GT, 裁剪后)
    rf_adj = pickle.load(open(f'{BASE}/region_{img_id}_refine_gt_graph.p', 'rb'))

    is_black = '黑边' if img_id in black_ids else '正常'
    gt_img = draw_graph_on_sat(sat, gt_adj) if gt_adj else sat.copy()
    rf_img = draw_graph_on_sat(sat, rf_adj) if rf_adj else sat.copy()

    axes[r,0].imshow(sat)
    axes[r,0].set_title(f'region_{img_id} ({is_black}) sat', fontsize=9)
    axes[r,0].axis('off')

    axes[r,1].imshow(gt_img)
    axes[r,1].set_title(f'graph_gt.pickle (评测GT)\n{len(gt_adj)}节点', fontsize=9, color='blue')
    axes[r,1].axis('off')

    axes[r,2].imshow(rf_img)
    axes[r,2].set_title(f'refine_gt_graph.p (训练GT,裁后)\n{len(rf_adj)}节点', fontsize=9, color='red')
    axes[r,2].axis('off')

plt.suptitle('评测GT (graph_gt.pickle) vs 训练GT (refine_gt_graph.p, 裁剪后)\n黄线=edge 红点=node', fontsize=12)
plt.tight_layout()
plt.savefig('docs/imgs/gt_vs_refine_compare.png', dpi=100, bbox_inches='tight')
print('✓ docs/imgs/gt_vs_refine_compare.png')
