# TOPO Usage
```bash
python main.py -graph_gt example/gt.p -graph_prop example/prop.p -output toporesult.txt
```

Here, the graph files gt.p and prop.p are all in the same format as what we used in Sat2Graph - a python dictionary where each key is the coordinate of a vertex (denoted by x) and the corresponding value is a list of x's neighboring vertices.  


# TOPO Parameters
Parameters | Note 
--------------------- | -------------
Propagation Distance  | 300 meters for large tiles and 150 meters for small tiles (see line 127-130 in main.py)
Propagation Interval  | Default is 5 meters. Config with -interval flag.
Matching Distance Threshold | Default is 10 meters. Config with -matching_threshold flag.
Matching Angle Threshold | 30 degrees 
One-to-One Matching | True


# Dependency
* hopcroftkarp
* rtree
