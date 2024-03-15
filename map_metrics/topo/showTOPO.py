import svgwrite
import math

def Coord2Pixels(lat, lon, min_lat, min_lon, max_lat, max_lon, sizex, sizey):
    #print(max_lat, min_lat, sizex)
    ilat = sizex - int((lat-min_lat) / ((max_lat - min_lat)/sizex))
    #ilat = int((lat-min_lat) / ((max_lat - min_lat)/sizex))
    ilon = int((lon-min_lon) / ((max_lon - min_lon)/sizey))

    return ilat, ilon



def RenderSVG(marbles, holes, matched_marbles, matched_holes, clat, clon, r, filename, OSMMap=None, starts = None ):
    dwg = svgwrite.Drawing(filename, profile='tiny')
    


    min_lat = clat - r
    max_lat = clat + r 
    min_lon = clon - r / math.cos(math.radians(clat))
    max_lon = clon + r / math.cos(math.radians(clat))

    sizex = 1000
    sizey = 1000 




    if OSMMap is not None:
        roadmap = OSMMap 
        for edgeId, edge in roadmap.edges.iteritems():
            n1 = edge[0]
            n2 = edge[1]

            lat1 = roadmap.nodes[n1][0]
            lon1 = roadmap.nodes[n1][1]

            lat2 = roadmap.nodes[n2][0]
            lon2 = roadmap.nodes[n2][1]

            ilat2, ilon2 = Coord2Pixels(lat1, lon1, min_lat, min_lon, max_lat, max_lon, sizex, sizey)
            ilat, ilon = Coord2Pixels(lat2, lon2, min_lat, min_lon, max_lat, max_lon, sizex, sizey)

            latold = lat1
            lonold = lon1

            #dwg.add(dwg.line((ilon2, ilat2), (ilon, ilat),stroke='rgb(180,180,180)',style = "stroke-width: 2"))
            dwg.add(dwg.line((ilon2, ilat2), (ilon, ilat),stroke='rgb(180,180,180)'))



    #Starting Point#
    if starts is not None:
        x,y = Coord2Pixels(starts[0], starts[1], min_lat, min_lon, max_lat, max_lon, sizex, sizey)
        dwg.add(dwg.circle(center = (y,x), r = 3, stroke='orange', fill='none'))

        x,y = Coord2Pixels(starts[2], starts[3], min_lat, min_lon, max_lat, max_lon, sizex, sizey)
        dwg.add(dwg.circle(center = (y,x), r = 3, stroke='black', fill='none'))



    for hole in holes:
        # x = 1000-int((hole[0] - clat)/r * 500 + 500)
        # y = int((hole[1] - clon)/r * 500 + 500)

        x,y = Coord2Pixels(hole[0], hole[1], min_lat, min_lon, max_lat, max_lon, sizex, sizey)

    #	print(x,y)
        
        dwg.add(dwg.circle(center = (y,x), r = 2, stroke='blue', fill='none'))


    for marble in marbles:
        # x = 1000-int((marble[0] - clat)/r * 500 + 500)
        # y = int((marble[1] - clon)/r * 500 + 500)

        x,y = Coord2Pixels(marble[0], marble[1], min_lat, min_lon, max_lat, max_lon, sizex, sizey)

    #	print(x,y)

        dwg.add(dwg.circle(center = (y,x), r = 1, stroke='red',fill='red'))

    for marble in matched_marbles:
        # x = 1000-int((marble[0] - clat)/r * 500 + 500)
        # y = int((marble[1] - clon)/r * 500 + 500)

        x,y = Coord2Pixels(marble[0], marble[1], min_lat, min_lon, max_lat, max_lon, sizex, sizey)

    #	print(x,y)

        dwg.add(dwg.circle(center = (y,x), r = 1, stroke='green',fill='green'))

    for i in range(len(matched_marbles)):
        x1 = 1000 - int((matched_marbles[i][0] - clat) / r * 500 + 500)
        y1 = int((matched_marbles[i][1] - clon) / r * 500 + 500)

        x2 = 1000 - int((matched_holes[i][0] - clat) / r * 500 + 500)
        y2 = int((matched_holes[i][1] - clon) / r * 500 + 500)

        x1,y1 = Coord2Pixels(matched_marbles[i][0], matched_marbles[i][1], min_lat, min_lon, max_lat, max_lon, sizex, sizey)
        x2,y2 = Coord2Pixels(matched_holes[i][0], matched_holes[i][1], min_lat, min_lon, max_lat, max_lon, sizex, sizey)


        dwg.add(dwg.line((y1,x1),(y2,x2),stroke='blue'))


    dwg.save()

    

def RenderRegion(points, lines,  region, filename):
    dwg = svgwrite.Drawing(filename, profile='tiny')
    
    for line in lines:
        x1 = 1000 - int((line[0] - region[0]) / (region[2] - region[0]) * 1000)
        y1 = int((line[1] - region[1]) / (region[3] - region[1]) * 1000)

        x2 = 1000 - int((line[2] - region[0]) / (region[2] - region[0]) * 1000)
        y2 = int((line[3] - region[1]) / (region[3] - region[1]) * 1000)

        dwg.add(dwg.line((y1,x1),(y2,x2),stroke='blue'))

    for p in points:
        x = 1000 - int((p[0] - region[0]) / (region[2] - region[0]) * 1000)
        y = int((p[1] - region[1]) / (region[3] - region[1]) * 1000)
        dwg.add(dwg.circle(center = (y,x), r = 1, stroke='red'))

    dwg.save()

def RenderRegion2(points, points2, lines,  region, filename):
    dwg = svgwrite.Drawing(filename, profile='tiny')
    
    for line in lines:
        x1 = 1000 - int((line[0] - region[0]) / (region[2] - region[0]) * 1000)
        y1 = int((line[1] - region[1]) / (region[3] - region[1]) * 1000)

        x2 = 1000 - int((line[2] - region[0]) / (region[2] - region[0]) * 1000)
        y2 = int((line[3] - region[1]) / (region[3] - region[1]) * 1000)

        dwg.add(dwg.line((y1,x1),(y2,x2),stroke='blue'))

    for p in points:
        x = 1000 - int((p[0] - region[0]) / (region[2] - region[0]) * 1000)
        y = int((p[1] - region[1]) / (region[3] - region[1]) * 1000)
        dwg.add(dwg.circle(center = (y,x), r = 2, stroke='green', fill='green'))

    # for p in points2:
    # 	x = 1000 - int((p[0] - region[0]) / (region[2] - region[0]) * 1000)
    # 	y = int((p[1] - region[1]) / (region[3] - region[1]) * 1000)
    # 	dwg.add(dwg.circle(center = (y,x), r = 3, stroke='red', fill='red'))

    dwg.save()

    #exit()
