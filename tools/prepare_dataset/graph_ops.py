import math
import numpy as np
import scipy.misc
from PIL import Image
import cv2
import sys
import pickle
from rtree import index
from common import *


def GPSDistance(p1, p2):
	a = p1[0] - p2[0]
	b = (p1[1] - p2[1]) * math.cos(math.radians(p1[0]))

	return math.sqrt(a * a + b * b)


def graphInsert(node_neighbor, n1key, n2key):
	if n1key != n2key:
		if n1key in node_neighbor:
			if n2key in node_neighbor[n1key]:
				pass
			else:
				node_neighbor[n1key].append(n2key)
		else:
			node_neighbor[n1key] = [n2key]

		if n2key in node_neighbor:
			if n1key in node_neighbor[n2key]:
				pass
			else:
				node_neighbor[n2key].append(n1key)
		else:
			node_neighbor[n2key] = [n1key]

	return node_neighbor


def graphDensify(node_neighbor, density=0.00020):
	visited = []

	new_node_neighbor = {}

	for node, node_nei in node_neighbor.items():  # python3取消iteritems
		if len(node_nei) == 1 or len(node_nei) > 2:
			if node in visited:
				continue

			# search node_nei 

			for next_node in node_nei:
				if next_node in visited:
					continue

				node_list = [node, next_node]

				current_node = next_node

				while True:
					if len(node_neighbor[node_list[-1]]) == 2:
						if node_neighbor[node_list[-1]][0] == node_list[-2]:
							node_list.append(node_neighbor[node_list[-1]][1])
						else:
							node_list.append(node_neighbor[node_list[-1]][0])
					else:
						break

				for i in range(len(node_list) - 1):
					if node_list[i] not in visited:
						visited.append(node_list[i])

				# interpolate
				# partial distance 
				pd = [0]

				for i in range(len(node_list) - 1):
					pd.append(pd[-1] + GPSDistance(node_list[i], node_list[i + 1]))

				interpolate_N = int(pd[-1] / density)

				last_loc = node_list[0]

				for i in range(interpolate_N):
					int_d = pd[-1] / (interpolate_N + 1) * (i + 1)
					for j in range(len(node_list) - 1):
						if pd[j] <= int_d and pd[j + 1] > int_d:
							a = (int_d - pd[j]) / (pd[j + 1] - pd[j])

							loc = ((1 - a) * node_list[j][0] + a * node_list[j + 1][0],
							       (1 - a) * node_list[j][1] + a * node_list[j + 1][1])

							new_node_neighbor = graphInsert(new_node_neighbor, last_loc, loc)
							last_loc = loc

				new_node_neighbor = graphInsert(new_node_neighbor, last_loc, node_list[-1])

	return new_node_neighbor

# ==========================================
# 新增：Cohen-Sutherland 裁剪算法辅助常量与函数
# ==========================================

# 定义区域码 (Bit codes)
INSIDE = 0  # 0000
LEFT   = 1  # 0001
RIGHT  = 2  # 0010
BOTTOM = 4  # 0100
TOP    = 8  # 1000

def compute_out_code(x, y, w, h):
    """计算点 (x,y) 相对于矩形 (0,0,w,h) 的区域码"""
    code = INSIDE
    if x < 0:           # 到左侧的左边
        code |= LEFT
    elif x > w:         # 到右侧的右边
        code |= RIGHT
    if y < 0:           # 到下侧的下面 (注意：这里假设y=0是上边界还是下边界取决于坐标系，通用算法只需保证一致)
        code |= BOTTOM  # 在图像坐标系中，通常0是Top，但这里我们只需处理边界即可
    elif y > h:         # 到上侧的上面
        code |= TOP
    return code

def cohen_sutherland_clip(x1, y1, x2, y2, size):
    """
    使用 Cohen-Sutherland 算法将线段 (x1, y1)-(x2, y2) 裁剪到矩形 (0, 0, size, size) 内。
    返回:
        None: 如果线段完全在外面
        ((nx1, ny1), (nx2, ny2)): 裁剪后的新线段坐标
    """
    xmax = size
    ymax = size
    xmin = 0
    ymin = 0

    code1 = compute_out_code(x1, y1, xmax, ymax)
    code2 = compute_out_code(x2, y2, xmax, ymax)
    accept = False

    while True:
        # 情况1: 两个点都在矩形内 (0 & 0 == 0)
        if (code1 == 0) and (code2 == 0):
            accept = True
            break
        # 情况2: 两个点都在矩形外的同一侧 (例如都在左边 0001 & 0001 != 0)
        elif (code1 & code2) != 0:
            break
        # 情况3: 线段穿过边界，需要计算交点
        else:
            # 挑一个在界外的点进行计算 (至少有一个非0)
            x = 0.0
            y = 0.0
            out_code = code1 if code1 != 0 else code2

            # 计算交点公式
            # 这里的顺序非常重要：Top/Bottom/Right/Left
            if out_code & TOP:   # 超过最大Y (下方)
                x = x1 + (x2 - x1) * (ymax - y1) / (y2 - y1)
                y = ymax
            elif out_code & BOTTOM: # 小于最小Y (上方)
                x = x1 + (x2 - x1) * (ymin - y1) / (y2 - y1)
                y = ymin
            elif out_code & RIGHT:  # 超过最大X (右方)
                y = y1 + (y2 - y1) * (xmax - x1) / (x2 - x1)
                x = xmax
            elif out_code & LEFT:   # 小于最小X (左方)
                y = y1 + (y2 - y1) * (xmin - x1) / (x2 - x1)
                x = xmin

            # 用交点替换界外的点，并重新计算区域码
            if out_code == code1:
                x1 = x
                y1 = y
                code1 = compute_out_code(x1, y1, xmax, ymax)
            else:
                x2 = x
                y2 = y
                code2 = compute_out_code(x2, y2, xmax, ymax)

    if accept:
        return ((x1, y1), (x2, y2))
    else:
        return None
	
# ==========================================
# 修改：使用裁剪逻辑的 graph2RegionCoordinate
# ==========================================

def graph2RegionCoordinate(node_neighbor, region, size=2048):
    new_node_neighbor = {}
    
    # 用集合记录已处理过的边，防止无向图重复计算 (A->B 和 B->A)
    processed_edges = set()

    for node, nei in node_neighbor.items():
        loc0 = node
        
        # 1. 先将经纬度转换为“原始”像素坐标 (可能包含负数或大于size的数)
        raw_x0 = (loc0[1] - region[1]) / (region[3] - region[1]) * size
        raw_y0 = (region[2] - loc0[0]) / (region[2] - region[0]) * size
        
        for loc1 in nei:
            # 防止重复处理 A-B 和 B-A
            # 将两个点坐标转为 tuple 并排序，作为边的唯一 ID
            edge_id = tuple(sorted((loc0, loc1)))
            if edge_id in processed_edges:
                continue
            processed_edges.add(edge_id)

            raw_x1 = (loc1[1] - region[1]) / (region[3] - region[1]) * size
            raw_y1 = (region[2] - loc1[0]) / (region[2] - region[0]) * size

            # 2. 调用 Cohen-Sutherland 算法计算交点并裁剪
            clipped_segment = cohen_sutherland_clip(raw_x0, raw_y0, raw_x1, raw_y1, size)

            # 3. 如果线段有效（在区域内或穿过区域），则插入新图
            if clipped_segment is not None:
                (new_x0, new_y0), (new_x1, new_y1) = clipped_segment
                
                # 注意：graphInsert 内部处理的是邻接表，所以要把裁剪后的两个端点互联
                # graphInsert 需要 Key 是 (y, x) 格式，注意顺序
                n1key = (new_y0, new_x0)
                n2key = (new_y1, new_x1)

                new_node_neighbor = graphInsert(new_node_neighbor, n1key, n2key)

    return new_node_neighbor

# def graph2RegionCoordinate(node_neighbor, region, size=2048):	# hhy modify size 2026-01-26
# 	new_node_neighbor = {}

# 	for node, nei in node_neighbor.items():  # python3取消iteritems
# 		loc0 = node
# 		for loc1 in nei:
# 			x0 = (loc0[1] - region[1]) / (region[3] - region[1]) * size	# hhy modify size 2026-01-26
# 			y0 = (region[2] - loc0[0]) / (region[2] - region[0]) * size	# hhy modify size 2026-01-26
# 			x1 = (loc1[1] - region[1]) / (region[3] - region[1]) * size	# hhy modify size 2026-01-26
# 			y1 = (region[2] - loc1[0]) / (region[2] - region[0]) * size	# hhy modify size 2026-01-26

# 			n1key = (y0, x0)
# 			n2key = (y1, x1)

# 			new_node_neighbor = graphInsert(new_node_neighbor, n1key, n2key)

# 	return new_node_neighbor


def graphVis2048(node_neighbor, region, filename, size=2048):	# hhy modify size 2026-01-26
	img = np.zeros((size, size, 3), dtype=np.uint8)	# hhy modify size 2026-01-26
	img = img + 255

	for node, nei in node_neighbor.iteritems():
		loc0 = node
		for loc1 in nei:
			x0 = int((loc0[1] - region[1]) / (region[3] - region[1]) * size)	# hhy modify size 2026-01-26
			y0 = int((region[2] - loc0[0]) / (region[2] - region[0]) * size)	# hhy modify size 2026-01-26
			x1 = int((loc1[1] - region[1]) / (region[3] - region[1]) * size)	# hhy modify size 2026-01-26
			y1 = int((region[2] - loc1[0]) / (region[2] - region[0]) * size)	# hhy modify size 2026-01-26

			cv2.line(img, (x0, y0), (x1, y1), (0, 0, 0), 2)

	for node, nei in node_neighbor.iteritems():
		loc0 = node
		x0 = int((loc0[1] - region[1]) / (region[3] - region[1]) * size)	# hhy modify size 2026-01-26
		y0 = int((region[2] - loc0[0]) / (region[2] - region[0]) * size)	# hhy modify size 2026-01-26

		cv2.circle(img, (x0, y0), 3, (0, 0, 255), -1)

	cv2.imwrite(filename, img)


def graphVis2048Segmentation(node_neighbor, region, filename, size=2048):
	img = np.zeros((size, size), dtype=np.uint8)

	for node, nei in node_neighbor.items():  # python3取消iteritems
		loc0 = node
		for loc1 in nei:
			x0 = int((loc0[1] - region[1]) / (region[3] - region[1]) * size)
			y0 = int((region[2] - loc0[0]) / (region[2] - region[0]) * size)
			x1 = int((loc1[1] - region[1]) / (region[3] - region[1]) * size)
			y1 = int((region[2] - loc1[0]) / (region[2] - region[0]) * size)

			cv2.line(img, (x0, y0), (x1, y1), (255), 2)

	cv2.imwrite(filename, img)


def graphVisStackingRoad(node_neighbor, region, filename, size=2048):
	img = np.zeros((size, size), dtype=np.uint8)

	crossing_point, adjustment = locate_stacking_road(node_neighbor)

	for ip in crossing_point.values():
		loc0 = ip
		x0 = int((loc0[1] - region[1]) / (region[3] - region[1]) * size)
		y0 = int((region[2] - loc0[0]) / (region[2] - region[0]) * size)

		cv2.circle(img, (x0, y0), 5, (255), -1)

	cv2.imwrite(filename, img)


def graphVisIntersection(node_neighbor, region, filename, size=2048):
	img = np.zeros((size, size), dtype=np.uint8)

	for node, nei in node_neighbor.iteritems():
		loc0 = node

		if len(nei) != 2:
			x0 = int((loc0[1] - region[1]) / (region[3] - region[1]) * size)
			y0 = int((region[2] - loc0[0]) / (region[2] - region[0]) * size)

			cv2.circle(img, (x0, y0), 5, (255), -1)

	cv2.imwrite(filename, img)


def locate_stacking_road(graph):
	idx = index.Index()

	edges = []

	for n1, v in graph.items():
		for n2 in v:
			if (n1, n2) in edges or (n2, n1) in edges:
				continue

			x1 = min(n1[0], n2[0])
			x2 = max(n1[0], n2[0])

			y1 = min(n1[1], n2[1])
			y2 = max(n1[1], n2[1])

			idx.insert(len(edges), (x1, y1, x2, y2))

			edges.append((n1, n2))

	adjustment = {}

	crossing_point = {}

	for edge in edges:
		n1 = edge[0]
		n2 = edge[1]

		x1 = min(n1[0], n2[0])
		x2 = max(n1[0], n2[0])

		y1 = min(n1[1], n2[1])
		y2 = max(n1[1], n2[1])

		candidates = list(idx.intersection((x1, y1, x2, y2)))

		for _candidate in candidates:
			# todo mark the overlap point 
			candidate = edges[_candidate]

			if n1 == candidate[0] or n1 == candidate[1] or n2 == candidate[0] or n2 == candidate[1]:
				continue

			if intersect(n1, n2, candidate[0], candidate[1]):

				ip = intersectPoint(n1, n2, candidate[0], candidate[1])

				if (candidate, edge) not in crossing_point:
					crossing_point[(edge, candidate)] = ip

				#release points 
				d = distance(ip, n1)
				thr = 9.5  # was 5.0
				if d < thr:
					vec = neighbors_norm(graph, n1, n2)
					weight = (thr - d) / thr
					vec = (vec[0] * weight, vec[1] * weight)

					if n1 not in adjustment:
						adjustment[n1] = [vec]
					else:
						adjustment[n1].append(vec)

				d = distance(ip, n2)
				if d < thr:
					vec = neighbors_norm(graph, n2, n1)
					weight = (thr - d) / thr
					vec = (vec[0] * weight, vec[1] * weight)

					if n2 not in adjustment:
						adjustment[n2] = [vec]
					else:
						adjustment[n2].append(vec)

				c1 = candidate[0]
				c2 = candidate[1]

				d = distance(ip, c1)
				if d < thr:
					vec = neighbors_norm(graph, c1, c2)
					weight = (thr - d) / thr
					vec = (vec[0] * weight, vec[1] * weight)

					if c1 not in adjustment:
						adjustment[c1] = [vec]
					else:
						adjustment[c1].append(vec)

				d = distance(ip, c2)
				if d < thr:
					vec = neighbors_norm(graph, c2, c1)
					weight = (thr - d) / thr
					vec = (vec[0] * weight, vec[1] * weight)

					if c2 not in adjustment:
						adjustment[c2] = [vec]
					else:
						adjustment[c2].append(vec)

	# apply adjustment 
	# move 2 pixels each time 

	return crossing_point, adjustment


def locate_parallel_road(graph):
	idx = index.Index()

	edges = []

	for n1, v in graph.items():
		for n2 in v:
			if (n1, n2) in edges or (n2, n1) in edges:
				continue

			x1 = min(n1[0], n2[0])
			x2 = max(n1[0], n2[0])

			y1 = min(n1[1], n2[1])
			y2 = max(n1[1], n2[1])

			idx.insert(len(edges), (x1, y1, x2, y2))

			edges.append((n1, n2))

	adjustment = {}

	crossing_point = {}

	parallel_road = []

	for edge in edges:
		n1 = edge[0]
		n2 = edge[1]

		if distance(n1, n2) < 10:
			continue

		x1 = min(n1[0], n2[0]) - 20
		x2 = max(n1[0], n2[0]) + 20

		y1 = min(n1[1], n2[1]) - 20
		y2 = max(n1[1], n2[1]) + 20

		candidates = list(idx.intersection((x1, y1, x2, y2)))

		for _candidate in candidates:
			# todo mark the overlap point 
			candidate = edges[_candidate]

			if n1 == candidate[0] or n1 == candidate[1] or n2 == candidate[0] or n2 == candidate[1]:
				continue

			flag = False

			for nei in graph[n1]:
				if candidate[0] in graph[nei]:
					flag = True
					continue

				if candidate[1] in graph[nei]:
					flag = True
					continue

			for nei in graph[n2]:
				if candidate[0] in graph[nei]:
					flag = True
					continue

				if candidate[1] in graph[nei]:
					flag = True
					continue

			if flag:
				continue

			p = abs(neighbors_cos(graph, (0, 0), (n2[0] - n1[0], n2[1] - n1[1]),
			                      (candidate[1][0] - candidate[0][0], candidate[1][1] - candidate[0][1])))

			if p > 0.985:
				if n1 not in parallel_road:
					parallel_road.append(n1)

	return parallel_road


def apply_adjustment(graph, adjustment):
	current_graph = graph
	counter = 0

	for k, v in adjustment.items():
		#print(k)
		new_graph = {}

		vec = [0, 0]

		for vv in v:
			vec[0] += vv[0]
			vec[1] += vv[1]

		vl = vec[0] * vec[0] + vec[1] * vec[1]

		vl = np.sqrt(vl)

		if vl == 0:
			continue

		if vl > 1.0:
			vec[0] /= vl
			vec[1] /= vl

		for l in [1.5, 1.0]:

			#new_k = (k[0] + int(vec[0]*l), k[1] + int(vec[1]*l))
			new_k = (k[0] + (vec[0] * l), k[1] + (vec[1] * l))

			if new_k == k:
				continue

			if new_k not in current_graph:

				# k 可能在前次 adjustment 迭代中被 del (不同 adjustment 条目涉及同一节点)
				if k not in current_graph:
					continue

				neighbors = list(current_graph[k])

				del current_graph[k]

				current_graph[new_k] = neighbors

				for nei in neighbors:
					# nei 可能在本次/前次 adjustment 中已被 del, 跳过避免 KeyError
					# (Porto 路网密集, 邻居也被调整的情况比 xian 常见)
					if nei not in current_graph:
						continue
					new_nei = []

					for n in current_graph[nei]:
						if n == k:
							new_nei.append(new_k)
						else:
							new_nei.append(n)

					current_graph[nei] = new_nei

				#print(k, "-->", new_k)

				counter += 1

				break
			else:
				continue

	print("adjusted ", counter, " nodes")

	return current_graph, counter

	pass


def graph_move_node(graph, old_n, new_n):
	nei = list(graph[old_n])
	del graph[old_n]

	graph[new_n] = nei

	for nn in nei:
		for i in range(len(graph[nn])):
			if graph[nn][i] == old_n:
				graph[nn][i] = new_n

	return graph


def apply_adjustment_delete_closeby_nodes(graph, adjustment):
	# delete the node and push its two neighbors closer ...

	# thr = 9.5 
	thr = 9.5
	for k, v in adjustment.items():
		if len(v) >= 4:  # duplicated ...
			ds = []
			for vv in v:
				ds.append((1.0 - distance(vv, (0, 0))) * thr)
			sorted(ds)
			gap = sum(ds[0:4]) / 2.0

			# delete the node and push its two neighbors closer ...
			# k / nei 可能已被前次 adjustment 删除 (Porto 路网密集), 全程加存在性检查
			if k not in graph:
				continue
			if gap < 12 and len(graph[k]) == 2:
				nei1 = graph[k][0]
				nei2 = graph[k][1]

				del graph[k]
				print("delete a node", k)

				if nei1 in graph:
					for i in range(len(graph[nei1])):
						if graph[nei1][i] == k:
							graph[nei1][i] = nei2

				if nei2 in graph:
					for i in range(len(graph[nei2])):
						if graph[nei2][i] == k:
							graph[nei2][i] = nei1

				# move nei1/nei2 closer? (k 已删, neighbors_norm 内部会安全处理)
				if nei1 in graph and nei1 not in adjustment:
					vec = neighbors_norm(graph, k, nei1)
					new_nei1 = (nei1[0] + vec[0] * 5.0, nei1[1] + vec[1] * 5.0)
					graph = graph_move_node(graph, nei1, new_nei1)

				if nei2 in graph and nei2 not in adjustment:
					vec = neighbors_norm(graph, k, nei2)
					new_nei2 = (nei2[0] + vec[0] * 5.0, nei2[1] + vec[1] * 5.0)
					graph = graph_move_node(graph, nei2, new_nei2)

	return graph


def graphGroundTruthPreProcess(graph):
	for it in range(40):  # was 8
		cp, adj = locate_stacking_road(graph)
		if it % 5 == 0 and it != 0:
			graph = apply_adjustment_delete_closeby_nodes(graph, adj)
		else:
			graph, c = apply_adjustment(graph, adj)
			if c == 0:
				break

	sample_points = {}

	sample_points['parallel_road'] = locate_parallel_road(graph)
	sample_points['complicated_intersections'] = []

	for k, v in graph.items():
		degree = len(v)
		if degree > 4:
			sample_points['complicated_intersections'].append(k)

	sample_points['overpass'] = []

	for k, v in cp.items():
		sample_points['overpass'].append((int(v[0]), int(v[1])))

	return graph, sample_points
