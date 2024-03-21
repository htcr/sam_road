import numpy as np
import math
import sys
import pickle 

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


class RoadGraph:
    def __init__(self, filename=None, region = None):
        self.nodeHash = {} # [tree_idx*10000000 + local_id] ->  id 
        self.nodeHashReverse = {} 
        self.nodes = {}	# id -> [lat,lon]
        self.edges = {} # id -> [n1, n2]
        self.nodeLink = {}   # id -> list of next node
        self.nodeID = 0 
        self.edgeID = 0
        self.edgeHash = {} # [nid1 * 10000000 + nid2] -> edge id 
        self.edgeScore = {}
        self.nodeTerminate = {}
        self.nodeScore = {}
        self.nodeLocations = {}

        if filename is not None:

            dumpDat = pickle.load(open(filename, "rb"))

            forest = dumpDat[1]

            self.forest = forest
            tid = 0
            for t in forest:
                for n in t:
                    idthis = tid*10000000 + n['id']

                    thislat = n['lat']
                    thislon = n['lon']

                    if region is not None:
                        if thislat < region[0] or thislon < region[1] or thislat > region[2] or thislon > region[3]:
                            continue

                    #if n['edgeScore'] < 7.0 : # skip those low confidential edges
                    #
                    #	continue

                    if n['similarWith'][0] != -1:
                        idthis = n['similarWith'][0]*10000000 + n['similarWith'][1]

                        thislat = forest[n['similarWith'][0]][n['similarWith'][1]]['lat']
                        thislon = forest[n['similarWith'][0]][n['similarWith'][1]]['lon']

                        

                    if n['OutRegion'] == 1:
                        self.nodeTerminate[tid*10000000+n['parent']] = 1


                    idparent = tid*10000000 + n['parent']
                    parentlat = t[n['parent']]['lat']
                    parentlon = t[n['parent']]['lon']

                    # if n['parent'] == 0:
                    #     print(tid, n['id'])


                    self.addEdge(idparent, parentlat, parentlon, idthis, thislat, thislon)



                tid += 1

        



    def addEdge(self, nid1,lat1,lon1,nid2,lat2,lon2, reverse=False, nodeScore1 = 0, nodeScore2 = 0, edgeScore = 0):  #n1d1->n1d2
        

        if nid1 not in self.nodeHash.keys():
            self.nodeHash[nid1] = self.nodeID
            self.nodeHashReverse[self.nodeID] = nid1
            self.nodes[self.nodeID] = [lat1, lon1]
            self.nodeLink[self.nodeID] = []
            #self.nodeLinkReverse[self.nodeID] = []
            self.nodeScore[self.nodeID] = nodeScore1
            self.nodeID += 1

        if nid2 not in self.nodeHash.keys():
            self.nodeHash[nid2] = self.nodeID
            self.nodeHashReverse[self.nodeID] = nid2
            self.nodes[self.nodeID] = [lat2, lon2]
            self.nodeLink[self.nodeID] = []
            #self.nodeLinkReverse[self.nodeID] = []
            self.nodeScore[self.nodeID] = nodeScore2
            self.nodeID += 1

        localid1 = self.nodeHash[nid1]
        localid2 = self.nodeHash[nid2]

        if localid1 * 10000000 + localid2 in self.edgeHash.keys():
            print("Duplicated Edge !!!", nid1, nid2)

            return 

        self.edges[self.edgeID] = [localid1, localid2]
        self.edgeHash[localid1 * 10000000 + localid2] = self.edgeID
        self.edgeScore[self.edgeID] = edgeScore
        self.edgeID += 1

        if localid2 not in self.nodeLink[localid1]:
            self.nodeLink[localid1].append(localid2)

        if reverse == True:
            if localid2 not in self.nodeLinkReverse.keys():
                self.nodeLinkReverse[localid2] = []

            if localid1 not in self.nodeLinkReverse[localid2]:
                self.nodeLinkReverse[localid2].append(localid1)


    def addEdgeToOneExistedNode(self, nid1,lat1,lon1,nid2, reverse=False, nodeScore1 = 0, edgeScore = 0):  #n1d1->n1d2
        

        if nid1 not in self.nodeHash.keys():
            self.nodeHash[nid1] = self.nodeID
            self.nodeHashReverse[self.nodeID] = nid1
            self.nodes[self.nodeID] = [lat1, lon1]
            self.nodeLink[self.nodeID] = []
            self.nodeLinkReverse[self.nodeID] = []
            self.nodeScore[self.nodeID] = nodeScore1
            self.nodeID += 1

        localid1 = self.nodeHash[nid1]
        localid2 = nid2

        self.edges[self.edgeID] = [localid1, localid2]
        self.edgeHash[localid1 * 10000000 + localid2] = self.edgeID
        self.edgeScore[self.edgeID] = edgeScore
        self.edgeID += 1

        if localid2 not in self.nodeLink[localid1]:
            self.nodeLink[localid1].append(localid2)

        if localid1 not in self.nodeLinkReverse[localid2]:
            self.nodeLinkReverse[localid2].append(localid1)


    def BiDirection(self):
        edgeList = list(self.edges.values())

        for edge in edgeList:
            localid1 = edge[1]
            localid2 = edge[0]

            self.edges[self.edgeID] = [localid1, localid2]
            self.edgeHash[localid1 * 10000000 + localid2] = self.edgeID
            self.edgeScore[self.edgeID] = self.edgeScore[self.edgeHash[localid2 * 10000000 + localid1]]
            self.edgeID += 1

            if localid2 not in self.nodeLink[localid1]:
                self.nodeLink[localid1].append(localid2)

    def ReverseDirectionLink(self):
        edgeList = list(self.edges.values())

        self.nodeLinkReverse = {}

        for edge in edgeList:
            localid1 = edge[1]
            localid2 = edge[0]

            if localid1 not in self.nodeLinkReverse :
                self.nodeLinkReverse[localid1] = [localid2]
            else:
                if localid2 not in self.nodeLinkReverse[localid1]:
                    self.nodeLinkReverse[localid1].append(localid2)

        for nodeId in self.nodes.keys():
            if nodeId not in self.nodeLinkReverse.keys():
                self.nodeLinkReverse[nodeId] = []

    # DFS
    def TOPOWalkDFS(self, nodeid, step = 0.00005, r = 0.00300, direction = False):

        localNodeList = {}
        localNodeDistance = {}

        mables = []

        localEdges = {}


        #localNodeList[nodeid] = 1
        #localNodeDistance[nodeid] = 0

        def explore(node_cur, node_prev, dist):
            old_node_dist = 1
            if node_cur in localNodeList.keys():
                old_node_dist = localNodeDistance[node_cur]
                if localNodeDistance[node_cur] <= dist:
                    return

            if dist > r :
                return

                  

            lat1 = self.nodes[node_cur][0]
            lon1 = self.nodes[node_cur][1]

            localNodeList[node_cur] = 1
            localNodeDistance[node_cur] = dist
            
            #mables.append((lat1, lon1))

            if node_cur not in self.nodeLinkReverse.keys():
                self.nodeLinkReverse[node_cur] = []

            reverseList = []

            if direction == False:
                reverseList = self.nodeLinkReverse[node_cur]

            for next_node in self.nodeLink[node_cur] + reverseList:

                edgeS = 0

                if node_cur * 10000000 + next_node in self.edgeHash.keys():
                    edgeS = self.edgeScore[self.edgeHash[node_cur * 10000000 + next_node]]
                
                if next_node * 10000000 + node_cur in self.edgeHash.keys():
                    edgeS = max(edgeS, self.edgeScore[self.edgeHash[next_node * 10000000 + node_cur]])


                if self.nodeScore[next_node] > 0 and edgeS > 0:
                    pass
                else:
                    continue

                if next_node == node_prev :
                    continue

                lat0 = 0
                lon0 = 0

                lat1 = self.nodes[node_cur][0]
                lon1 = self.nodes[node_cur][1]

                lat2 = self.nodes[next_node][0]
                lon2 = self.nodes[next_node][1]

                #TODO check angle of next_node


                localEdgeId = node_cur * 10000000 + next_node

                # if localEdgeId not in localEdges.keys():
                # 	localEdges[localEdgeId] = 1

                l = distance((lat2,lon2), (lat1,lon1))
                num = int(math.ceil(l / step))


                bias = step * math.ceil(dist / step) - dist
                cur = bias



                if old_node_dist + l < r :
                    explore(next_node, node_cur, dist + l)
                else:

                    while cur < l:
                        alpha = cur / l 
                #for a in range(1,num):
                #	alpha = float(a)/num 
                        if dist + l * alpha > r :
                            break

                        latI = lat2 * alpha + lat1 * (1-alpha)
                        lonI = lon2 * alpha + lon1 * (1-alpha)

                        if (latI, lonI) not in mables:
                            mables.append((latI, lonI))

                        cur += step

                    l = distance((lat2,lon2), (lat1,lon1))

                    explore(next_node, node_cur, dist + l)



        explore(nodeid, -1, 0)


        return mables


    def distanceBetweenTwoLocation(self, loc1, loc2, max_distance):
        localNodeList = {}
        localNodeDistance = {}

        #mables = []

        localEdges = {}


        edge_covered = {}  # (s,e) --> distance from s and distance from e 


        if loc1[0] == loc2[0] and loc1[1] == loc2[1] :
            return abs(loc1[2] - loc2[2])

        elif loc1[0] == loc2[1] and loc1[1] == loc2[0]:
            return abs(loc1[2] - loc2[3])

        ans_dist = 100000

        Queue = [(loc1[0], -1, loc1[2]), (loc1[1], -1, loc1[2])]

        while True:

            if len(Queue) == 0:
                break

            args = Queue.pop(0)

            node_cur, node_prev, dist = args[0], args[1], args[2]

            old_node_dist = 1
            if node_cur in localNodeList.keys():
                old_node_dist = localNodeDistance[node_cur]
                if localNodeDistance[node_cur] <= dist:
                    continue

            if dist > max_distance :
                continue

            lat1 = self.nodes[node_cur][0]
            lon1 = self.nodes[node_cur][1]

            localNodeList[node_cur] = 1
            localNodeDistance[node_cur] = dist
            
            #mables.append((lat1, lon1))

            if node_cur not in self.nodeLinkReverse.keys():
                self.nodeLinkReverse[node_cur] = []

            reverseList = []
            reverseList = self.nodeLinkReverse[node_cur]

            visited_next_node = []
            for next_node in self.nodeLink[node_cur] + reverseList:
                if next_node == node_prev:
                    continue

                if next_node == node_cur :
                    continue

                if next_node == loc1[0] or next_node == loc1[1] :
                    continue

                if next_node in visited_next_node:
                    continue 

                visited_next_node.append(next_node)



                edgeS = 0

                

                lat0 = 0
                lon0 = 0

                lat1 = self.nodes[node_cur][0]
                lon1 = self.nodes[node_cur][1]

                lat2 = self.nodes[next_node][0]
                lon2 = self.nodes[next_node][1]

                localEdgeId = node_cur * 10000000 + next_node

                # if localEdgeId not in localEdges.keys():
                # 	localEdges[localEdgeId] = 1


                if node_cur == loc2[0] and next_node == loc2[1]:
                    new_ans = dist + loc2[2]
                    if new_ans < ans_dist :
                        ans_dist = new_ans 
                elif node_cur == loc2[1] and next_node == loc2[0]:
                    new_ans = dist + loc2[3]
                    if new_ans < ans_dist :
                        ans_dist = new_ans




                l = distance((lat2,lon2), (lat1,lon1))
                Queue.append((next_node, node_cur, dist + l))
                
                


        


        return ans_dist


    # BFS (much faster)
    def TOPOWalk(self, nodeid, step = 0.00005, r = 0.00300, direction = False, newstyle = False, nid1=0, nid2=0, dist1=0, dist2= 0, bidirection = False, CheckGPS = None, metaData = None):

        localNodeList = {}
        localNodeDistance = {}

        mables = []

        localEdges = {}


        edge_covered = {}  # (s,e) --> distance from s and distance from e 


        #localNodeList[nodeid] = 1
        #localNodeDistance[nodeid] = 0

        if newstyle == False:
            Queue = [(nodeid, -1, 0)]

        else:
            Queue = [(nid1, -1, dist1), (nid2, -1, dist2)]


        # Add holes between nid1 and nid2 


        lat1 = self.nodes[nid1][0]
        lon1 = self.nodes[nid1][1]

        lat2 = self.nodes[nid2][0]
        lon2 = self.nodes[nid2][1]

        l = distance((lat2,lon2), (lat1,lon1))
        num = int(math.ceil(l / step))

        alpha = 0 

        while True:
            latI = lat1*alpha + lat2*(1-alpha)
            lonI = lon1*alpha + lon2*(1-alpha)

            d1 = distance((latI,lonI),(lat1,lon1))
            d2 = distance((latI,lonI),(lat2,lon2))

            if dist1 - d1 < r or dist2 -d2 < r:
                if (latI, lonI, lat2 - lat1, lon2 - lon1) not in mables:
                    mables.append((latI, lonI, lat2 - lat1, lon2 - lon1)) # add direction

                    if bidirection == True:
                        if nid1 in self.nodeLink[nid2] and nid2 in self.nodeLink[nid1]:
                            mables.append((latI+0.00001, lonI+0.00001, lat2 - lat1, lon2 - lon1))  #Add another mables

            alpha += step/l

            if alpha > 1.0:
                break




        while True:

            if len(Queue) == 0:
                break

            args = Queue.pop(0)

            node_cur, node_prev, dist = args[0], args[1], args[2]

            old_node_dist = 1
            if node_cur in localNodeList.keys():
                old_node_dist = localNodeDistance[node_cur]
                if localNodeDistance[node_cur] <= dist:
                    continue

            if dist > r :
                continue

                  

            lat1 = self.nodes[node_cur][0]
            lon1 = self.nodes[node_cur][1]

            localNodeList[node_cur] = 1
            localNodeDistance[node_cur] = dist
            
            #mables.append((lat1, lon1))

            if node_cur not in self.nodeLinkReverse.keys():
                self.nodeLinkReverse[node_cur] = []

            reverseList = []

            if direction == False:
                reverseList = self.nodeLinkReverse[node_cur]

            visited_next_node = []
            for next_node in self.nodeLink[node_cur] + reverseList:
                if next_node == node_prev:
                    continue

                if next_node == node_cur :
                    continue

                if next_node == nid1 or next_node == nid2 :
                    continue

                if next_node in visited_next_node:
                    continue 




                visited_next_node.append(next_node)



                edgeS = 0

                # if node_cur * 10000000 + next_node in self.edgeHash.keys():
                # 	edgeS = self.edgeScore[self.edgeHash[node_cur * 10000000 + next_node]]
                
                # if next_node * 10000000 + node_cur in self.edgeHash.keys():
                # 	edgeS = max(edgeS, self.edgeScore[self.edgeHash[next_node * 10000000 + node_cur]])


                # if self.nodeScore[next_node] > 0 and edgeS > 0:
                # 	pass
                # else:
                # 	continue

                # if next_node == node_prev :
                # 	continue

                lat0 = 0
                lon0 = 0

                lat1 = self.nodes[node_cur][0]
                lon1 = self.nodes[node_cur][1]

                lat2 = self.nodes[next_node][0]
                lon2 = self.nodes[next_node][1]

                #TODO check angle of next_node


                localEdgeId = node_cur * 10000000 + next_node

                # if localEdgeId not in localEdges.keys():
                # 	localEdges[localEdgeId] = 1

                l = distance((lat2,lon2), (lat1,lon1))
                num = int(math.ceil(l / step))


                bias = step * math.ceil(dist / step) - dist
                cur = bias



                if old_node_dist + l < r :
                    Queue.append((next_node, node_cur, dist + l))
                    #explore(next_node, node_cur, dist + l)
                else:

                    start_limitation = 0
                    end_limitation = l 
                    if (node_cur, next_node) in edge_covered.keys():
                        start_limitation = edge_covered[(node_cur, next_node)]

                    #if next_node == node_cur :
                        #print("BUG")

                    if (next_node, node_cur) in edge_covered.keys():
                        end_limitation = l-edge_covered[(next_node, node_cur)]

                    #end_limitation = l

                    #if next_node not in localNodeDistance.keys(): # Should we remove this ?


                    turnnel_edge = False
                    if metaData is not None:
                        nnn1 = self.nodeHashReverse[next_node]
                        nnn2 = self.nodeHashReverse[node_cur]

                        if metaData.edgeProperty[metaData.edge2edgeid[(nnn1,nnn2)]]['layer'] < 0:
                            turnnel_edge  = True
                            



                    while cur < l:
                        alpha = cur / l 
            
                        if dist + l * alpha > r :
                            break

                        if l * alpha < start_limitation:
                            cur += step
                            continue 

                        if l * alpha > end_limitation:
                            break

                        latI = lat2 * alpha + lat1 * (1-alpha)
                        lonI = lon2 * alpha + lon1 * (1-alpha)


                        
                        if (latI, lonI, lat2 - lat1, lon2 - lon1) not in mables and turnnel_edge is False:
                            mables.append((latI, lonI, lat2 - lat1, lon2 - lon1)) # add direction


                            if bidirection == True:
                                if next_node in self.nodeLink[node_cur] and node_cur in self.nodeLink[next_node] and turnnel_edge is False:
                                    mables.append((latI+0.00001, lonI+0.00001, lat2 - lat1, lon2 - lon1))  #Add another mables


                        cur += step


                    if (node_cur, next_node) in edge_covered.keys():
                        #if cur-step < edge_covered[(node_cur, next_node)]:
                        #	print(node_cur, edge_covered[(node_cur, next_node)], cur-step)

                        edge_covered[(node_cur, next_node)] = cur - step #max(cur, edge_covered[(node_cur, next_node)])
                        #edge_covered[(node_cur, next_node)] = cur
                    else:
                        edge_covered[(node_cur, next_node)] = cur - step
                        #edge_covered[(node_cur, next_node)] = cur




                    l = distance((lat2,lon2), (lat1,lon1))
                    Queue.append((next_node, node_cur, dist + l))
                    #explore(next_node, node_cur, dist + l)


        result_marbles = []

        if CheckGPS is None:
            result_marbles = mables
        else:
            for mable in mables:
                if CheckGPS(mable[0], mable[1]) == True:
                    result_marbles.append(mable)



        #explore(nodeid, -1, 0)


        return result_marbles


    def removeNode(self, nodeid):
        for next_node in self.nodeLink[nodeid]:
            edgeid = self.edgeHash[nodeid * 10000000 + next_node]

            del self.edges[edgeid]
            del self.edgeScore[edgeid]
            del self.edgeHash[nodeid * 10000000 + next_node]

            if nodeid in self.nodeLinkReverse[next_node]:
                self.nodeLinkReverse[next_node].remove(nodeid)


        for prev_node in self.nodeLinkReverse[nodeid]:
            edgeid = self.edgeHash[prev_node * 10000000 + nodeid]

            del self.edges[edgeid]
            del self.edgeScore[edgeid]
            del self.edgeHash[prev_node * 10000000 + nodeid]

            if nodeid in self.nodeLink[prev_node]:
                self.nodeLink[prev_node].remove(nodeid)


        del self.nodes[nodeid]
        del self.nodeScore[nodeid]
        del self.nodeLink[nodeid]
        del self.nodeLinkReverse[nodeid]



    def removeDeadEnds(self, oneround = False):
        deleted = 0
        for nodeid in self.nodes.keys():
            if self.nodeHashReverse[nodeid] in self.nodeTerminate.keys():
                continue

            if self.nodeHashReverse[nodeid] % 10000000 == 0:
                continue

            d = self.NumOfNeighbors(nodeid)
            if d == 1 or len(self.nodeLink[nodeid]) == 0 or len(self.nodeLinkReverse[nodeid]) == 0:
                self.removeNode(nodeid)
                deleted += 1

        return deleted 

    def NumOfNeighbors(self, nodeid):
        neighbor = {}

        for next_node in self.nodeLink[nodeid] + self.nodeLinkReverse[nodeid]:
            neighbor[next_node] = 1

        return len(neighbor.keys())

    def getNeighbors(self,nodeid):
        neighbor = {}

        for next_node in self.nodeLink[nodeid] + self.nodeLinkReverse[nodeid]:
            if next_node != nodeid:
                neighbor[next_node] = 1

        return neighbor.keys()



def edgeIntersection(baseX, baseY, dX, dY, n1X, n1Y, n2X, n2Y):
    t = dX * n1Y + dY * n2X - dX * n2Y - dY * n1X

    c = n2X * n1Y - n1X * n2Y + baseX * (n2Y - n1Y) + baseY * (n1X -n2X)

    if t == 0 :
        return 0,0,0,0

    alpha = c / t

    if alpha < 0 : 
        return 0,0,0,0

    iX = baseX + alpha * dX
    iY = baseY + alpha * dY

    d = (iX - n1X)*(n2X - iX) + (iY - n1Y) * (n2Y - iY)

    if d < 0 :
        return 0,0,0,0

    extend_length = np.sqrt(alpha * dX * alpha * dX + alpha * dY * alpha * dY)

    return iX, iY, extend_length, 1



if __name__ == "__main__":

    dumpDat = pickle.load(open(sys.argv[1], "rb"))
