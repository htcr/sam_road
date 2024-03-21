import numpy as np
import math
import sys
import scipy.ndimage
import pickle
import graph as splfy
import code
import random
import showTOPO
from rtree import index
from time import time 
from hopcroftkarp import HopcroftKarp
from subprocess import Popen


def latlonNorm(p1, lat = 40):

    p11 = p1[1] * math.cos(math.radians(lat))

    l = np.sqrt(p11 * p11 + p1[0] * p1[0])

    return p1[0]/l, p11/l

def pointToLineDistance(p1,p2,p3):
    # p1 --> p2 is the line
    # p1 is (0,0)

    dist = np.sqrt(p2[0] * p2[0] + p2[1] * p2[1]) 

    proj_length = (p2[0] * p3[0] + p2[1] * p3[1]) / dist 

    if proj_length > dist :
        a = p3[0] - p2[0]
        b = p3[1] - p2[1]

        return np.sqrt(a*a + b*b)

    if proj_length < 0 :
        a = p3[0] - p1[0]
        b = p3[1] - p1[1]

        return np.sqrt(a*a + b*b)


    alpha = proj_length / dist

    p4 = [0,0]

    p4[0] = alpha * p2[0]
    p4[1] = alpha * p2[1]

    a = p3[0] - p4[0]
    b = p3[1] - p4[1]

    return np.sqrt(a*a + b*b)

def pointToLineDistanceLatLon(p1,p2,p3):
    pp2 = [0,0]
    pp3 = [0,0]

    pp2[0] = p2[0] - p1[0]
    pp2[1] = (p2[1] - p1[1]) * math.cos(math.radians(p1[0]))

    pp3[0] = p3[0] - p1[0]
    pp3[1] = (p3[1] - p1[1]) * math.cos(math.radians(p1[0]))

    return pointToLineDistance((0,0), pp2, pp3)


def Coord2Pixels(lat, lon, min_lat, min_lon, max_lat, max_lon, sizex, sizey):
    #print(max_lat, min_lat, sizex)
    ilat = sizex - int((lat-min_lat) / ((max_lat - min_lat)/sizex))
    #ilat = int((lat-min_lat) / ((max_lat - min_lat)/sizex))
    ilon = int((lon-min_lon) / ((max_lon - min_lon)/sizey))

    return ilat, ilon


def distance(p1, p2):
    a = p1[0] - p2[0]
    b = (p1[1] - p2[1])*math.cos(math.radians(p1[0]))
    return np.sqrt(a*a + b*b)


def angleDistance(p1, p2):
    l1 = np.sqrt(p1[0] * p1[0] + p1[1] * p1[1])
    l2 = np.sqrt(p2[0] * p2[0] + p2[1] * p2[1])

    if l1 == 0 or l2 == 0:
        return 100000

    a = (p1[0]/l1 - p2[0]/l2)
    b = (p1[1]/l1 - p2[1]/l2)

    return np.sqrt(a*a + b * b)




def TOPOGenerateStartingPoints(OSMMap, check = True, density = 0.00050, region = None, image = None, direction = False, metaData = None, mergin=0.07):
    result = []

    tunnel_skip_num = 0

    svgEdges = []

    if image != 'NULL':
        img = scipy.ndimage.imread(image)
        sizex = np.shape(img)[0]
        sizey = np.shape(img)[1]

        if len(np.shape(img)) > 2:
            img = img[:,:,3].reshape((sizex, sizey))


    else:	
        img = None


    

    def Coord2Pixels(lat, lon, min_lat, min_lon, max_lat, max_lon, sizex, sizey):
        ilat = sizex - int((lat-min_lat) / ((max_lat - min_lat)/sizex))
        ilon = int((lon-min_lon) / ((max_lon - min_lon)/sizey))
        return ilat, ilon



    visitedNodes = []
    for nodeid in OSMMap.nodes.keys():
        if nodeid in visitedNodes:
            continue

        cur_node = nodeid  

        next_nodes = {}

        for nn in OSMMap.nodeLink[cur_node] + OSMMap.nodeLinkReverse[cur_node]:
            next_nodes[nn] = 1


        if len(next_nodes.keys()) == 2:
            continue 


        for nextnode in next_nodes.keys():
            if nextnode in visitedNodes:
                continue

            node_list = [nodeid]
            cur_node = nextnode 
            while True:
                node_list.append(cur_node)

                neighbor = {}
                for nn in OSMMap.nodeLink[cur_node] + OSMMap.nodeLinkReverse[cur_node]:
                    neighbor[nn] = 1

                if len(neighbor.keys()) != 2:
                    break


                if node_list[-2] == list(neighbor.keys())[0] :
                    cur_node = list(neighbor.keys())[1]
                else:
                    cur_node = list(neighbor.keys())[0]


            for i in range(1, len(node_list)-1):
                visitedNodes.append(node_list[i])



            dists = []
            dist = 0
            for i in range(0, len(node_list)-1):
                dists.append(dist)
                dist += distance(OSMMap.nodes[node_list[i]],OSMMap.nodes[node_list[i+1]])

            dists.append(dist)

            if dist < density/2:
                continue

            n = max(int(dist / density),1)

            alphas = [float(x+1)/float(n+1) for x in range(n)]

            
            



            for alpha in alphas:
                for j in range(len(node_list)-1):

                    # Don't add starting locations in the tunnel
                    if metaData is not None:
                        nnn1 = OSMMap.nodeHashReverse[node_list[j]]
                        nnn2 = OSMMap.nodeHashReverse[node_list[j+1]]

                        if metaData.edgeProperty[metaData.edge2edgeid[(nnn1,nnn2)]]['layer'] < 0:
                            tunnel_skip_num += 1
                            continue

                    if alpha * dist >= dists[j] and alpha * dist <= dists[j+1]:
                        a = (alpha * dist - dists[j]) / (dists[j+1] - dists[j])
                        lat = (1-a)*OSMMap.nodes[node_list[j]][0] + a * OSMMap.nodes[node_list[j+1]][0]
                        lon = (1-a)*OSMMap.nodes[node_list[j]][1] + a * OSMMap.nodes[node_list[j+1]][1]

                        if img != None:
                            x,y = Coord2Pixels(lat, lon, region[0], region[1], region[2], region[3], sizex, sizey)
                            if x>0 and x<sizex and y>0 and y < sizey:
                                if img[x,y] > 0:
                                    result.append((lat, lon, node_list[j], node_list[j+1], alpha * dist - dists[j], dists[j+1] - alpha * dist))
                        else:
                            lat_mergin = mergin*(region[2]-region[0])
                            lon_mergin = mergin*(region[3]-region[1])

                            # These was 0.00100 and 0.00150 for lat and lon
                            if lat-region[0] > lat_mergin and region[2] - lat > lat_mergin and lon-region[1] > lon_mergin and region[3] - lon > lon_mergin:
                                result.append((lat, lon, node_list[j], node_list[j+1], alpha * dist - dists[j], dists[j+1] - alpha * dist))


    for _,edge in OSMMap.edges.items():

        svgEdges.append((OSMMap.nodes[edge[0]][0],OSMMap.nodes[edge[0]][1], OSMMap.nodes[edge[1]][0], OSMMap.nodes[edge[1]][1]))


    showTOPO.RenderRegion(result, svgEdges, region, "gt.svg")



    print(len(result))
    print("Skipped tunnels ", tunnel_skip_num)
    return result





def TOPOGeneratePairs(GPSMap, OSMMap, OSMList, threshold = 0.00010, region = None, single = False, edgeids = None):

    result = {}

    matchedLoc = []

    idx = index.Index()
    if edgeids is not None:
        for edgeid in edgeids:	
            if edgeid not in GPSMap.edges.keys():
                continue
            n1 = GPSMap.edges[edgeid][0]
            n2 = GPSMap.edges[edgeid][1]

            lat1 = GPSMap.nodes[n1][0]
            lon1 = GPSMap.nodes[n1][1]

            lat2 = GPSMap.nodes[n2][0]
            lon2 = GPSMap.nodes[n2][1]

            idx.insert(edgeid, (min(lat1, lat2), min(lon1, lon2), max(lat1, lat2), max(lon1, lon2)))

            
    else:
        for edgeid in GPSMap.edges.keys():	
            n1 = GPSMap.edges[edgeid][0]
            n2 = GPSMap.edges[edgeid][1]

            lat1 = GPSMap.nodes[n1][0]
            lon1 = GPSMap.nodes[n1][1]

            lat2 = GPSMap.nodes[n2][0]
            lon2 = GPSMap.nodes[n2][1]

            idx.insert(edgeid, (min(lat1, lat2), min(lon1, lon2), max(lat1, lat2), max(lon1, lon2)))

            
    #for item in OSMList:
    for i in range(len(OSMList)):
        item = OSMList[i]

        lat = item[0]
        lon = item[1]

        possible_edges = list(idx.intersection((lat-threshold*2,lon-threshold*2, lat+threshold*2, lon+threshold*2)))

        min_dist = 10000
        min_edge = -1

        for edgeid in possible_edges:
            n1 = GPSMap.edges[edgeid][0]
            n2 = GPSMap.edges[edgeid][1]

            n3 = item[2]
            n4 = item[3]


            lat1 = GPSMap.nodes[n1][0]
            lon1 = GPSMap.nodes[n1][1]

            lat2 = GPSMap.nodes[n2][0]
            lon2 = GPSMap.nodes[n2][1]


            lat3 = OSMMap.nodes[n3][0]
            lon3 = OSMMap.nodes[n3][1]

            lat4 = OSMMap.nodes[n4][0]
            lon4 = OSMMap.nodes[n4][1]

            nlat1, nlon1 = latlonNorm((lat2-lat1,lon2-lon1))
            nlat2, nlon2 = latlonNorm((lat4-lat3,lon4-lon3))



            dist = pointToLineDistanceLatLon((lat1,lon1), (lat2, lon2), (lat,lon))
            if dist < threshold and dist < min_dist:
                angle_dist = 1.0 - abs(nlat1 * nlat2 + nlon1 * nlon2)
                #angle_dist = angleDistance((nlat1, nlon1), (nlat2, nlon2))

                #if angle_dist  < 0.1 or angle_dist > 1.9 :
                if edgeids is None:
                    #if angle_dist  < 0.25 or angle_dist > 1.75 :
                    # print(angle_dist)
                    #if angle_dist  < 0.13 : # 30 degrees
                    if angle_dist  < 0.04 : # 15 degrees
                        min_edge = edgeid 
                        min_dist = dist 
                else:
                    min_edge = edgeid 
                    min_dist = dist 


        if min_edge != -1 :
            edgeid = min_edge

            n1 = GPSMap.edges[edgeid][0]
            n2 = GPSMap.edges[edgeid][1]

            lat1 = GPSMap.nodes[n1][0]
            lon1 = GPSMap.nodes[n1][1]

            lat2 = GPSMap.nodes[n2][0]
            lon2 = GPSMap.nodes[n2][1]

            


            result[i] = [edgeid, n1, n2, distance((lat1,lon1),(lat, lon)), distance((lat2,lon2),(lat, lon)), lat,lon]
            matchedLoc.append((lat, lon))

            if single == True :
                return result


    
    svgEdges = []

    for _,edge in OSMMap.edges.items():
        svgEdges.append((OSMMap.nodes[edge[0]][0],OSMMap.nodes[edge[0]][1], OSMMap.nodes[edge[1]][0], OSMMap.nodes[edge[1]][1]))


    if region is not None:
        showTOPO.RenderRegion2(OSMList, matchedLoc, svgEdges, region, "coverage.svg")



    return result







def TOPOGenerateList(GPSMap, OSMMap, check = True, threshold = 0.00010, region = None, image = None, direction = False):
    result = {}

    
    img = scipy.ndimage.imread(image)

    sizex = np.shape(img)[0]
    sizey = np.shape(img)[1]

    if len(np.shape(img)) > 2:
        img = img[:,:,0].reshape((sizex, sizey))

    def Coord2Pixels(lat, lon, min_lat, min_lon, max_lat, max_lon, sizex, sizey):
        ilat = sizex - int((lat-min_lat) / ((max_lat - min_lat)/sizex))
        ilon = int((lon-min_lon) / ((max_lon - min_lon)/sizey))
        return ilat, ilon


    idx = index.Index()
    for idthis in OSMMap.nodes.keys():	
        x,y = Coord2Pixels(OSMMap.nodes[idthis][0], OSMMap.nodes[idthis][1], region[0], region[1], region[2], region[3], sizex, sizey)
        if x>0 and x<sizex and y>0 and y < sizey:
            if img[x,y] > 0:
                idx.insert(idthis, (OSMMap.nodes[idthis][0], OSMMap.nodes[idthis][1],OSMMap.nodes[idthis][0]+0.000001, OSMMap.nodes[idthis][1]+0.000001))
        


    candidateNode = {}

    for edgeId, edge in GPSMap.edges.items():

        n1 = edge[0]
        n2 = edge[1]

        if check :
            if n1 in GPSMap.deletedNodes.keys() or n2 in GPSMap.deletedNodes.keys():
                continue

            if GPSMap.nodeScore[n1] < 1 or GPSMap.nodeScore[n2] < 1 :
                continue

            if n1 in GPSMap.nodeTerminate.keys() or n2 in GPSMap.nodeTerminate.keys():
                continue

            score = GPSMap.edgeScore[GPSMap.edgeHash[n1*10000000 + n2]]
            if score <1:
                continue


        candidateNode[n1] = 1
        candidateNode[n2] = 1

    for nid in candidateNode.keys():
        lat = GPSMap.nodes[nid][0]
        lon = GPSMap.nodes[nid][1]

        input_dir = []

        for nnode in GPSMap.nodeLink[nid]:
            nlat = GPSMap.nodes[nnode][0]
            nlon = GPSMap.nodes[nnode][1]

            input_dir.append((nlat-lat, nlon-lon))

            if direction == False:
                input_dir.append((-nlat+lat, -nlon+lon))



        possible_nodes = list(idx.intersection((lat-threshold,lon-threshold, lat+threshold, lon+threshold)))

        min_dist = 100000
        min_node = -1



        for pnode in possible_nodes:
            latp = OSMMap.nodes[pnode][0]
            lonp = OSMMap.nodes[pnode][1]

            target_dir = []

            for nnode in OSMMap.nodeLink[pnode]:
                nlat = OSMMap.nodes[nnode][0]
                nlon = OSMMap.nodes[nnode][1]

                target_dir.append((nlat-latp, nlon-lonp))

                if direction == False:
                    target_dir.append((-nlat+latp, -nlon+lonp))


            match_dir = False

            for dir1 in input_dir:
                for dir2 in target_dir:
                    if angleDistance(dir1,dir2) < 0.1:
                        match_dir = True
                        break

            if match_dir == False:
                continue



            d = distance((lat,lon),(latp, lonp))

            if d < min_dist:
                min_dist = d
                min_node = pnode


        #print(nid, lat, lon, len(possible_nodes), min_dist)

        if min_node == -1 or min_dist > threshold:
            continue

        result[min_node] = nid



    return result


def TOPO(GPSMap, OSMMap, step = 0.00005, r = 0.00300, num = 1000, threshold = 0.00020, region = None):
    idx = index.Index()
    for idthis in OSMMap.nodes.keys():	
        idx.insert(idthis, (OSMMap.nodes[idthis][0], OSMMap.nodes[idthis][1],OSMMap.nodes[idthis][0]+0.000001, OSMMap.nodes[idthis][1]+0.000001))
    
    candidateNode = {}

    for edgeId, edge in GPSMap.edges.items():

        n1 = edge[0]
        n2 = edge[1]

        # if n1 in GPSMap.deletedNodes.keys() or n2 in GPSMap.deletedNodes.keys():
        # 	continue

        # if GPSMap.nodeScore[n1] < 1 or GPSMap.nodeScore[n2] < 1 :
        # 	continue

        # if n1 in GPSMap.nodeTerminate.keys() or n2 in GPSMap.nodeTerminate.keys():
        # 	continue

        # score = GPSMap.edgeScore[GPSMap.edgeHash[n1*10000000 + n2]]
        # if score <1:
        # 	continue


        candidateNode[n1] = 1
        candidateNode[n2] = 1


    precesion_sum = 0
    recall_sum = 0


    # print(len(candidateNode))

    for i in range(num):
        while True:
            nid = random.choice(candidateNode.keys())

            lat = GPSMap.nodes[nid][0]
            lon = GPSMap.nodes[nid][1]

            possible_nodes = list(idx.intersection((lat-threshold,lon-threshold, lat+threshold, lon+threshold)))

            min_dist = 100000
            min_node = -1



            for pnode in possible_nodes:
                latp = OSMMap.nodes[pnode][0]
                lonp = OSMMap.nodes[pnode][1]

                d = distance((lat,lon),(latp, lonp))

                if d < min_dist:
                    min_dist = d
                    min_node = pnode


            #print(nid, lat, lon, len(possible_nodes), min_dist)

            if min_node == -1 or min_dist > threshold:
                continue


            marbles = GPSMap.TOPOWalk(nid, step = step, r = r)
            holes = OSMMap.TOPOWalk(min_node, step = step, r = r+step)

            matchedNum = 0

            for marble in marbles:
                for hole in holes:
                    if distance(marble, hole) < threshold:
                        matchedNum += 1
                        break

            precesion = float(matchedNum) / len(marbles)


            matchedNum = 0

            for hole in holes:
                for marble in marbles:
                    if distance(marble, hole) < threshold:
                        matchedNum += 1
                        break
            recall = float(matchedNum) / len(holes)

            precesion_sum += precesion
            recall_sum += recall

            # print(i, "MapNodeID", nid, "OSMNodeID", pnode, "Precesion",precesion, "Recall",recall, "Avg Precesion", precesion_sum/(i+1),"Avg Recall", recall_sum/(i+1)) 

            break



def BipartiteGraphMatching(graph):
    cost = 0

    def getKey(item):
        return item[2]

    graph_ = sorted(graph, key=getKey)

    matched_marbles = []
    matched_holes = []


    for item in graph_:
        if item[0] not in matched_marbles and item[1] not in matched_holes:
            matched_marbles.append(item[0])
            matched_holes.append(item[1])
            cost += item[2]


    return matched_marbles, matched_holes, cost 


def TOPO121(topo_result, roadgraph):
    # create index
    rtree_index = index.Index()
    for ind in range(len(topo_result)):
        r = 0.000001
        lat = topo_result[ind][0]
        lon = topo_result[ind][1]

        rtree_index.insert(ind, [lat - r, lon - r, lat + r, lon + r])


    new_list = []

    # create dependency
    for ind in range(len(topo_result)):
        lat = topo_result[ind][0]
        lon = topo_result[ind][1]
        r_lat = 0.00030
        r_lon = 0.00030 / math.cos(math.radians(lat))
        
        candidate = list(rtree_index.intersection([lat-r_lat, lon-r_lon, lat+r_lat, lon+r_lon]))

        competitors = []

        gpsn1, gpsn2, gpsd1, gpsd2 = topo_result[ind][4], topo_result[ind][5], topo_result[ind][6], topo_result[ind][7] 

        for can_id in candidate:
            t_gpsn1, t_gpsn2, t_gpsd1, t_gpsd2 = topo_result[can_id][4], topo_result[can_id][5], topo_result[can_id][6], topo_result[can_id][7]

            d = roadgraph.distanceBetweenTwoLocation((gpsn1, gpsn2, gpsd1, gpsd2),(t_gpsn1, t_gpsn2, t_gpsd1, t_gpsd2), max_distance = 0.00030)

            if d < 0.00020:
                competitors.append(can_id)

        new_list.append((topo_result[ind], ind, competitors))

    # find maximum matching 
    # TODO

    def get_key(item):
        return item[0][2] # precision

    new_list = sorted(new_list, key = get_key)
    result = []
    mark = {}

    for ind in range(len(new_list)-1, -1, -1):
        if new_list[ind][1] in mark:
            # print(new_list[ind][0][2])
            if new_list[ind][0][2] < 0.9:
                continue 
        
        result.append(new_list[ind][0])
        for cc in new_list[ind][2]:
            mark[cc]=1

    # print(len(topo_result), ' now is ', len(result))

    return result 

def topoAvg(topo_result):
    p = 0
    r = 0 
    for item in topo_result:
        p = p + item[2]
        r = r + item[3]
    
    if len(topo_result) == 0 :
        return 0, 0
    return p/len(topo_result), r/len(topo_result)


def TOPOWithPairs(GPSMap, OSMMap, GPSList, OSMList, step = 0.00005, r = 0.00300, threshold = 0.00015, region = None, outputfile = "tmp.txt", one2oneMatching = True, metaData = None):
    
    i = 0
    precesion_sum = 0
    recall_sum = 0


    # print(len(OSMList), len(GPSList.keys()))

    # rrr = float(len(GPSList.keys())) / float(len(OSMList))

    # print("Overall Coverage", rrr)

    returnResult = []

    for k,itemGPS in GPSList.items():


        itemOSM = OSMList[k]

        gpsn1, gpsn2, gpsd1, gpsd2 = itemGPS[1],itemGPS[2],itemGPS[3],itemGPS[4]
        osmn1, osmn2, osmd1, osmd2 = itemOSM[2],itemOSM[3],itemOSM[4],itemOSM[5]

        osm_start_lat,osm_start_lon = itemOSM[0], itemOSM[1]
        gps_start_lat, gps_start_lon =  itemGPS[5], itemGPS[6]

        # nid = pairs[min_node]

        # lat = GPSMap.nodes[nid][0]
        # lon = GPSMap.nodes[nid][1]

        lat = itemOSM[0]
        lon = itemOSM[1]

        

        ts1 = time()
        marbles = GPSMap.TOPOWalk(1, step = step, r = r, direction = False, newstyle = True, nid1=gpsn1, nid2=gpsn2, dist1=gpsd1, dist2= gpsd2)
        
        # for recall
        holes = OSMMap.TOPOWalk(1, step = step, r = r, direction = False, newstyle = True, nid1=osmn1, nid2=osmn2, dist1=osmd1, dist2= osmd2, metaData = metaData) # remove holes in tunnel 
        
        # for precision
        holes_bidirection = OSMMap.TOPOWalk(1, step = step, r = r, direction = False, newstyle = True, nid1=osmn1, nid2=osmn2, dist1=osmd1, dist2= osmd2, bidirection = True, metaData = None) # don't remove holes in tunnel
        ts2 = time()

        

        idx_marbles = index.Index()
        idx_holes = index.Index()
        idx_holes_bidirection = index.Index()


        for j in range(len(marbles)):
            idx_marbles.insert(j, (marbles[j][0]-0.00001, marbles[j][1]-0.00001, marbles[j][0]+0.00001, marbles[j][1]+0.00001))

        for j in range(len(holes)):
            idx_holes.insert(j, (holes[j][0]-0.00001, holes[j][1]-0.00001, holes[j][0]+0.00001, holes[j][1]+0.00001))

        for j in range(len(holes_bidirection)):
            idx_holes_bidirection.insert(j, (holes_bidirection[j][0]-0.00001, holes_bidirection[j][1]-0.00001, holes_bidirection[j][0]+0.00001, holes_bidirection[j][1]+0.00001))

        # holes_bidirection  = holes
        # idx_holes_bidirection = idx_holes




        matchedNum = 0
        bigraph = {}
        matched_marbles = []
        bipartite_graph = []

        cost_map = {}

        for marble in marbles:
            rr = threshold * 1.8
            possible_holes = list(idx_holes_bidirection.intersection((marble[0]-rr, marble[1]-rr, marble[0]+rr, marble[1]+rr)))
            for hole_id in possible_holes:
                hole = holes_bidirection[hole_id]
                ddd = distance(marble, hole)

                n1 = latlonNorm((marble[2], marble[3]))
                n2 = latlonNorm((hole[2], hole[3]))

                #ddd += (1.0 - abs(n1[0] * n2[0] + n1[1] * n2[1])) * threshold * 5
                #ddd -= threshold / 2
                #ddd = max(ddd, 0)

                if marble[2] != marble[3] and hole[2] != hole[3]:
                    angle_d = 1.0 - abs(n1[0] * n2[0] + n1[1] * n2[1])
                else:
                    angle_d = 0.0

                #angle_d = 0.0

                if ddd < threshold and angle_d < 0.29:   # 0.03 --> 15 degrees  0.13 --> 30 degrees 0.29 --> 45 degrees
                    #cost_map[(marble, hole_id)] = ddd 

                    if marble in bigraph.keys():
                        bigraph[marble].add(hole_id)
                    else:
                        bigraph[marble] = set([hole_id])

                    bipartite_graph.append((marble, hole_id, ddd))

                    matchedNum += 1
                    matched_marbles.append(marble)
                    #break
        
        soft_matchedNum = 0

        if one2oneMatching == True:
            matches = HopcroftKarp(bigraph).maximum_matching()

            matchedNum = len(matches.keys()) / 2

            # for k,v in matches.items():
            # 	if (k,v) in cost_map.keys():
            # 		soft_matchedNum += max(min(((threshold - cost_map[(k,v)]) / threshold),1.0),0.0)


        
        #matched_marbles, matched_holes, _ = BipartiteGraphMatching(bipartite_graph)
        #matched_holes = [(holes_bidirection[item][0], holes_bidirection[item][1]) for item in matched_holes]
        




        #matched_marbles = [(marbles[item][0], marbles[item][1]) for item in matched_marbles]

        # for item in HopcroftKarp(bigraph).maximum_matching().keys():
        # 	if type(item) is not int :
        # 		matched_marbles.append(item) 

        

        # print(i, len(marbles), len(holes))
        if len(marbles)==0 or len(holes)==0:
            continue

        #precesion = float(soft_matchedNum) / len(marbles)
        precesion = float(matchedNum) / len(marbles)

        # TOPO Debug 
        #showTOPO.RenderSVG(marbles, holes, matched_marbles,matched_holes,  lat, lon, 0.00300, "svg/nn"+outputfile.split('/')[-1]+"_%.6f_"%precesion+str(i)+"_"+str(lat)+"_"+str(lon)+".svg", OSMMap= OSMMap, starts=(osm_start_lat,osm_start_lon,gps_start_lat,gps_start_lon))

        matchedNum = 0
        bigraph = {}

        cost_map = {}


        for hole in holes:
            rr = threshold * 1.8
            possible_marbles = list(idx_marbles.intersection((hole[0]-rr, hole[1]-rr, hole[0]+rr, hole[1]+rr)))
            for marble_id in possible_marbles:
                marble = marbles[marble_id]

                ddd = distance(marble, hole)

                n1 = latlonNorm((marble[2], marble[3]))
                n2 = latlonNorm((hole[2], hole[3]))

                #ddd += (1.0 - abs(n1[0] * n2[0] + n1[1] * n2[1])) * threshold * 5
                #ddd -= threshold / 2
                #ddd = max(ddd, 0)

                if marble[2] != marble[3] and hole[2] != hole[3]:
                    angle_d = 1.0 - abs(n1[0] * n2[0] + n1[1] * n2[1])
                else:
                    angle_d = 0.0

                #angle_d = 0.0

                if ddd < threshold and angle_d < 0.29:
                    #cost_map[(hole, marble_id)] = ddd 

                    if hole in bigraph.keys():
                        bigraph[hole].add(marble_id)
                    else:
                        bigraph[hole] = set([marble_id])
                    matchedNum += 1
                    #break

        soft_matchedNum = 0

        if one2oneMatching == True:
            #matchedNum = len(HopcroftKarp(bigraph).maximum_matching().keys()) / 2

            matches = HopcroftKarp(bigraph).maximum_matching()

            matchedNum = len(matches.keys()) / 2

            # for k,v in matches.items():
            # 	if (k,v) in cost_map.keys():
            # 		soft_matchedNum += max(min(((threshold - cost_map[(k,v)]) / threshold),1.0),0.0)


        #recall = float(soft_matchedNum) / len(holes)
        recall = float(matchedNum) / len(holes)

        precesion_sum += precesion
        recall_sum += recall

        ts3 = time()

        with open(outputfile, "a") as fout:
            fout.write(str(i)+ " " + str(lat)+" "+str(lon)+" "+str(gpsn1)+ " "+str(gpsn2)+ " Precesion " + str(precesion)+ " Recall "+str(recall)+ " Avg Precesion "+ str(precesion_sum/(i+1)) + " Avg Recall " + str(recall_sum/(i+1))+" \n")
        
        # print(i, "Precesion",precesion, "Recall",recall, "Avg Precesion", precesion_sum/(i+1),"Avg Recall", recall_sum/(i+1), rrr, ts2-ts1, ts3-ts2) 


        returnResult.append((lat, lon, precesion, recall, gpsn1, gpsn2, gpsd1, gpsd2))

        i = i + 1
        #if i > 100:
        #	break

    # try:
    # 	with open(outputfile, "a") as fout:
    # 		fout.write(str(precesion_sum/i)+" "+str(recall_sum/i)+" "+str(rrr)+ " "+ str(rrr * recall_sum/i) +"\n")
    # except:
    # 	 with open(outputfile, "a") as fout:
    #                     fout.write(str(0)+" "+str(0)+" "+str(0)+ " "+ "0.0" +"\n")


    #with open("TOPOResultSummary.txt","a") as fout:
    #	fout.write(str(precesion_sum/i)+" "+str(recall_sum/i)+" "+str(rrr)+ " "+ str(rrr * recall_sum/i) +"\n")



    new_topoResult = TOPO121(returnResult, GPSMap)


    # Debug svg 
    # for rr in returnResult:
    # 	if rr not in new_topoResult:
    # 		print("remove rr")
    # 		Popen("rm svg/*%s*.svg" % (str(rr[0])+"_"+str(rr[1])),shell=True).wait()



    #print(topoAvg(returnResult), len(returnResult)/float(len(OSMList)))
    # print(topoAvg(new_topoResult), len(new_topoResult)/float(len(OSMList)))
    p,r = topoAvg(new_topoResult)

    # with open(outputfile, "a") as fout:
    # 	fout.write(str(p)+" "+str(r)+" "+str(len(new_topoResult)/float(len(OSMList)))+"\n")

    # print("precision="+ str(p)+ " overall-recall="+ str(r * len(new_topoResult)/float(len(OSMList))))

    try:
        with open(outputfile, "a") as fout:
            fout.write(str(p)+" "+str(r)+" "+str(len(new_topoResult)/float(len(OSMList)))+ " " + str(r * len(new_topoResult)/float(len(OSMList)))  +"\n")
            fout.write("precision="+ str(p)+ " overall-recall="+ str(r * len(new_topoResult)/float(len(OSMList))))
    except:
        with open(outputfile, "a") as fout:
            fout.write(str(0)+" "+str(0)+" "+str(0)+ " " + str(0)  +"\n")



    return new_topoResult

def TOPOWithPairsNew(GPSMap, OSMMap, GPSList, OSMList, step = 0.00005, r = 0.00300, threshold = 0.00015, region = None, outputfile = "tmp.txt", one2oneMatching = True, base_n = None, svgname = "", soft = True, CheckGPS = None):
    
    i = 0
    precesion_sum = 0
    recall_sum = 0


    #print(len(OSMList), len(GPSList.keys()))

    rrr = float(len(GPSList.keys())) / float(len(OSMList))

    #print("Overall Coverage", rrr)

    total_score = 0
    total_f = 0

    cost = 0
    matchedNum = 0

    marbles =[]


    number_of_holes = []

    for k,itemGPS in GPSList.items():


        itemOSM = OSMList[k]

        gpsn1, gpsn2, gpsd1, gpsd2 = itemGPS[1],itemGPS[2],itemGPS[3],itemGPS[4]
        osmn1, osmn2, osmd1, osmd2 = itemOSM[2],itemOSM[3],itemOSM[4],itemOSM[5]


        # nid = pairs[min_node]

        # lat = GPSMap.nodes[nid][0]
        # lon = GPSMap.nodes[nid][1]

        lat = itemOSM[0]
        lon = itemOSM[1]

        ts1 = time()

        if gpsn1 in GPSMap.nodes.keys():
            marbles = GPSMap.TOPOWalk(1, step = step, r = r, direction = False, newstyle = True, nid1=gpsn1, nid2=gpsn2, dist1=gpsd1, dist2= gpsd2)
        else:
            marbles = []

        holes = OSMMap.TOPOWalk(1, step = step, r = r, direction = False, newstyle = True, nid1=osmn1, nid2=osmn2, dist1=osmd1, dist2= osmd2, CheckGPS = CheckGPS)
        
        number_of_holes.append(len(holes))
        #holes_bidirection = OSMMap.TOPOWalk(1, step = step, r = r, direction = False, newstyle = True, nid1=osmn1, nid2=osmn2, dist1=osmd1, dist2= osmd2, bidirection = True)
        ts2 = time()

        holes_bidirection = holes	

        idx_marbles = index.Index()
        idx_holes = index.Index()
        #idx_holes_bidirection = index.Index()


        for j in range(len(marbles)):
            idx_marbles.insert(j, (marbles[j][0]-0.00001, marbles[j][1]-0.00001, marbles[j][0]+0.00001, marbles[j][1]+0.00001))

        for j in range(len(holes)):
            idx_holes.insert(j, (holes[j][0]-0.00001, holes[j][1]-0.00001, holes[j][0]+0.00001, holes[j][1]+0.00001))

        idx_holes_bidirection = idx_holes
        # for j in range(len(holes_bidirection)):
        # 	idx_holes_bidirection.insert(j, (holes_bidirection[j][0]-0.00001, holes_bidirection[j][1]-0.00001, holes_bidirection[j][0]+0.00001, holes_bidirection[j][1]+0.00001))


        
        bigraph = {}
        matched_marbles = []
        bipartite_graph = []


        for marble in marbles:
            rr = threshold * 1.8
            possible_holes = list(idx_holes_bidirection.intersection((marble[0]-rr, marble[1]-rr, marble[0]+rr, marble[1]+rr)))
            for hole_id in possible_holes:
                hole = holes_bidirection[hole_id]
                ddd = distance(marble, hole)

                n1 = latlonNorm((marble[2], marble[3]))
                n2 = latlonNorm((hole[2], hole[3]))

                angle_d = (1.0 - abs(n1[0] * n2[0] + n1[1] * n2[1]))



                if ddd < threshold and angle_d < 0.3:
                    if marble in bigraph.keys():
                        bigraph[marble].add(hole_id)
                    else:
                        bigraph[marble] = set([hole_id])

                    n1 = latlonNorm((marble[2], marble[3]))
                    n2 = latlonNorm((hole[2], hole[3]))

                    
                    ddd -= threshold / 3
                    ddd = max(ddd*1.5, 0)

                    ddd += angle_d * threshold * 0.5


                    bipartite_graph.append((marble, hole, ddd))

                    matchedNum += 1
                    matched_marbles.append(marble)
                    #break
        
        #if one2oneMatching == True:
        #	matchedNum = len(HopcroftKarp(bigraph).maximum_matching().keys()) / 2

        matched_marbles, matched_holes, cost = BipartiteGraphMatching(bipartite_graph)
        matchedNum = len(matched_marbles)

        if soft == False:
            cost = 0

        showTOPO.RenderSVG(marbles, holes_bidirection, matched_marbles,matched_holes,  lat, lon, 0.00500, "svg/"+svgname +str(i)+"_"+str(lat)+"_"+str(lon)+".svg")

        score = cost + (len(marbles) - matchedNum) * threshold * 1.15
        total_score += score

        #print(i, len(marbles), len(holes), score, matchedNum, cost)
        

        i = i + 1

    
        #if base_n == None:
        
        base_n = len(holes)


        if len(marbles) == 0:
            f = 0
        else:
            smooth_precision = 1.0 - (cost+ (len(marbles) - matchedNum) * threshold * 1.15) / (len(marbles) * threshold * 1.15)

            smooth_recall = 1.0 - (cost+ (base_n - matchedNum) * threshold * 1.15) / (base_n * threshold * 1.15)

            # print(smooth_precision, smooth_recall, len(marbles), len(holes))


            if smooth_recall + smooth_precision == 0:
                f = 0
            else:
                f = 2*smooth_precision*smooth_recall/(smooth_recall + smooth_precision)

        total_f += f 


    total_f /= i 


    return total_score, total_f, number_of_holes #cost+ (len(marbles) - matchedNum) * threshold * 3
    #return total_score, 10


def TOPOWithList(GPSMap, OSMMap, pairs, step = 0.00005, r = 0.00300, threshold = 0.00015, region = None, outputfile = "tmp.txt"):
    
    i = 0
    precesion_sum = 0
    recall_sum = 0


    for min_node in pairs.keys():
        
        nid = pairs[min_node]

        lat = GPSMap.nodes[nid][0]
        lon = GPSMap.nodes[nid][1]

        
        marbles = GPSMap.TOPOWalk(nid, step = step, r = r, direction = False)
        holes = OSMMap.TOPOWalk(min_node, step = step, r = r, direction = False)


        showTOPO.RenderSVG(marbles, holes, lat, lon, 0.00500, "svg/"+str(i)+"_"+str(lat)+"_"+str(lon)+".svg")


        matchedNum = 0

        for marble in marbles:
            for hole in holes:
                if distance(marble, hole) < threshold:
                    matchedNum += 1
                    break
        
        if len(marbles)==0 or len(holes)==0:
            continue
        precesion = float(matchedNum) / len(marbles)


        matchedNum = 0

        for hole in holes:
            for marble in marbles:
                if distance(marble, hole) < threshold:
                    matchedNum += 1
                    break
        recall = float(matchedNum) / len(holes)

        precesion_sum += precesion
        recall_sum += recall


        with open(outputfile, "a") as fout:
            fout.write(str(i)+ " MapNodeID "+ str(nid)+ " OSMNodeID "+str(min_node)+ " Precesion " + str(precesion)+ " Recall "+str(recall)+ " Avg Precesion "+ str(precesion_sum/(i+1)) + " Avg Recall " + str(recall_sum/(i+1))+" \n")
        
        # print(i, "MapNodeID", nid, "OSMNodeID", min_node, "Precesion",precesion, "Recall",recall, "Avg Precesion", precesion_sum/(i+1),"Avg Recall", recall_sum/(i+1)) 

        i = i + 1


    with open(outputfile, "a") as fout:
        fout.write(str(precesion_sum/i)+" "+str(recall_sum/i)+"\n")

    with open("TOPOResultSummary.txt","a") as fout:
        fout.write(str(precesion_sum/i)+" "+str(recall_sum/i)+"\n")



    
