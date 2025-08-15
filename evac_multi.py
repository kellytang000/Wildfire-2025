from __future__ import annotations
from shapely.ops import unary_union
from shapely.geometry import Point, LineString
from pyproj import Transformer
from pathlib import Path
from typing import Tuple, List
import geopandas as gpd
import networkx as nx
import argparse, sys, math, folium, heapq, itertools

TARGET_CRS = "EPSG:32610"
HIWAY_TYPES = {"motorway", "trunk"}
IS_LINK = lambda rt: rt and rt.endswith("_link")
INF = float("inf")
SEA2SKY = ("sea-to-sky highway", "Sea-to-Sky Highway")

# highway or not according to the name
def is_hiway(rt: str, name: str):
    return (rt in HIWAY_TYPES) or ("highway" in name)

# edge penalty, forbidden sea-to-sky highway to normal road, vice versa
def edge_penalized_weight(curr_attr, prev_attr, curr_node=None, prev_node=None, prev_prev_node=None, beta=1.5, gamma=2.0):
    # rise the weight according to rho
    load, cap = curr_attr["vehicles"], curr_attr["capacity"]
    rho = load / cap # ρ = traffic / capacity
    cong = 1 + beta * (rho ** gamma) # (1+β·ρ^γ)
    weight = curr_attr["weight"] * cong
    if prev_attr is None: 
        return weight
    
    rt_c  = curr_attr.get("road_type", "")
    nm_c  = curr_attr.get("road_name", "")
    hi_c  = is_hiway(rt_c, nm_c)
    link_c= IS_LINK(rt_c)

    rt_p  = prev_attr.get("road_type", "")
    nm_p  = prev_attr.get("road_name", "")
    hi_p  = is_hiway(rt_p, nm_p)
    link_p= IS_LINK(rt_p)

    # sea-to-sky highway specific restriction
    sea_in  = any(k in nm_c for k in SEA2SKY)
    sea_out = any(k in nm_p for k in SEA2SKY)
    if sea_in ^ sea_out: # if *_link to sea-to-sky highway, then set the forbidden restriction
        if not (IS_LINK(rt_c) or IS_LINK(rt_p)):
            return INF
    
    if hi_c and not (hi_p or link_p):
        return weight * 10
    if hi_p and not (hi_c or link_c):
        return weight * 10
    
    if prev_prev_node: # calculate the turning angle if we have 3 points
        v1 = (prev_node[0]-prev_prev_node[0], prev_node[1]-prev_prev_node[1])
        v2 = (curr_node[0]-prev_node[0], curr_node[1]-prev_node[1])
        # compute the angle (via cosine)
        dot  = v1[0]*v2[0] + v1[1]*v2[1] 
        norm = ((v1[0]**2+v1[1]**2)**0.5 * (v2[0]**2+v2[1]**2)**0.5)
        if norm > 0 and dot/norm < -0.9: # angle > ~155° (≈ U-turn)
            # if the turn happens on a trunk/motorway, forbid it outright, on ordinary roads we just give it a heavy penalty
            if curr_attr["road_type"] in {"trunk", "motorway"}:
                return float("inf")
            else:
                return weight * 20
    return weight

def to_wgs84(x: float, y: float) -> Tuple[float, float]:
    return Transformer.from_crs(TARGET_CRS, 4326, always_xy=True).transform(x, y)

def load_inputs(mesh_dir: Path):
    edges = gpd.read_file(mesh_dir / "mesh_edges.geojson").to_crs(TARGET_CRS)
    nodes = gpd.read_file(mesh_dir / "mesh_nodes.geojson").to_crs(TARGET_CRS)
    shelters = gpd.read_file(mesh_dir / "shelters.geojson").to_crs(TARGET_CRS)
    return edges, nodes, shelters

# mesh_edges to NetworkX DiGraph
def build_graph(edges: gpd.GeoDataFrame, fire_union, alpha: float) -> nx.Graph:
    G = nx.DiGraph()
    # collect link endpoints
    link_eps: set[tuple[float,float]] = set()
    for _, r in edges.iterrows():
        rt = r.get("road_type","")
        if rt.endswith("_link"):
            pts = list(r.geometry.coords)
            link_eps.add(tuple(pts[0]))
            link_eps.add(tuple(pts[-1]))
    for _, row in edges.iterrows():
        # if it's motorway or trunk
        # and endpoint not in link_eps, skip it  
        rt = row.get("road_type","")
        a = tuple(row.geometry.coords[0])
        b = tuple(row.geometry.coords[-1])
        # use mesh_edges.geojson (oneway_dir) to build edges
        dir_tag = ("FWD" if row.get("direction") == "forward" else "REV") if rt.endswith("_link") else row.get("oneway_dir", "BIDIR")
        base_cost = row.get("cost", row.length)
        phys_time = row.length / (row.speed_kph / 3.6)
        inter_len = row.geometry.intersection(fire_union).length
        ratio     = inter_len / row.length
        weight    = base_cost * (1 + alpha * ratio)
        cap       = row.get("cap", 30)
        if dir_tag == "FWD":
            # forward
            G.add_edge(a, b,
                cost=base_cost, exp_ratio=ratio,
                weight=weight, road_type=rt,
                phys_time=phys_time,
                length_m=row.length,
                road_name=(row.get("road_name","") or "").lower(),
                capacity=cap,        # NEW
                vehicles=0,
                oneway=True)
        elif dir_tag == "REV":
            # reverse
            G.add_edge(b, a,
                cost=base_cost, exp_ratio=ratio,
                weight=weight, road_type=rt,
                phys_time=phys_time,          # ← added
                length_m=row.length,
                road_name=(row.get("road_name","") or "").lower(),
                capacity=cap,        # NEW
                vehicles=0,
                oneway=True)
        else:
            G.add_edge(a, b,
                cost=base_cost, exp_ratio=ratio,
                weight=weight, road_type=rt,
                phys_time=phys_time,
                length_m=row.length,
                road_name=(row.get("road_name","") or "").lower(),
                capacity=cap,        # NEW
                vehicles=0,
                oneway=True)
            G.add_edge(b, a,
                cost=base_cost, exp_ratio=ratio,
                weight=weight, road_type=rt,
                phys_time=phys_time,
                length_m=row.length,
                road_name=(row.get("road_name","") or "").lower(),
                capacity=cap,        # NEW
                vehicles=0,
                oneway=True)
    return G

def nearest_node(pt: Point, nodes: list[tuple]) -> tuple:
    # return nearest node near pt (x, y, layer)
    min_d, closest = float("inf"), None
    x0, y0 = pt.x, pt.y
    for node in nodes:
        x, y = node[0], node[1]    # only consider the first two 
        d = (x - x0) ** 2 + (y - y0) ** 2
        if d < min_d:
            min_d, closest = d, node
    return closest

def nodes_list(nodes_gdf: gpd.GeoDataFrame) -> List[Tuple[float, float]]:
    return [(geom.x, geom.y) for geom in nodes_gdf.geometry]

def add_route_map(route: List[Tuple[float, float]], fires: gpd.GeoDataFrame, start_pt: Tuple[float, float], goal_pt: Tuple[float, float], out_path: Path):
    lon_c, lat_c = to_wgs84(*start_pt)
    m = folium.Map(location=[lat_c, lon_c], zoom_start=12)
    folium.GeoJson(fires.to_crs(4326), name="fires", style_function=lambda _: {"color": "#ff6600", "weight": 2, "fillColor": "#ff6600", "fillOpacity": 0.3}).add_to(m)
    line_latlon = [to_wgs84(x, y)[::-1] for x, y in route]
    folium.PolyLine(line_latlon, color="#2686CC", weight=4, opacity=0.9, tooltip=f"Path length {len(route)} nodes").add_to(m)
    folium.Marker(to_wgs84(*start_pt)[::-1], icon=folium.Icon(color="green"), tooltip="Start").add_to(m)
    folium.Marker(to_wgs84(*goal_pt)[::-1], icon=folium.Icon(color="red"), tooltip="Shelter").add_to(m)
    folium.LayerControl().add_to(m)
    m.save(out_path)

def build_parser():
    p = argparse.ArgumentParser(description="Evacuation route planner")
    p.add_argument("--lat", type=float, nargs="+", default=None, help="latitude list, e.g. --lat 49.7641 49.7670")
    p.add_argument("--lon", type=float, nargs="+", default=None, help="longitude list, same order as lat")
    p.add_argument("--mesh-dir", default="multi_outputs", type=str)
    p.add_argument("--timestep", type=int, default=None, help="Fire timestep number (leave empty=load fires.geojson)")
    p.add_argument("--out-dir", default="multi_outputs", type=str)
    p.add_argument("--alpha", type=float, default=3.0, help="Fire cost multiplier alpha (default=3.0)")
    p.add_argument("--beta",  type=float, default=1.5, help="Congestion penalty β")
    p.add_argument("--gamma", type=float, default=2.0, help="Congestion penalty γ")
    p.add_argument("--coord-file", type=Path, help="txt/CSV file containing lat,lon list, each line lat,lon")
    return p

def constrained_shortest_path(G: nx.DiGraph, source, target, beta, gamma):
    # return cost and path list
    counter = itertools.count()
    pq   = [(0.0, next(counter), source, None, None, None)]
    dist = {source: 0.0}
    parent = {} # node to parent 

    while pq:
        du, _, u, prev_node, prev_prev, attr_u = heapq.heappop(pq)
        if u == target:
            break
        if du != dist[u]:
            continue

        # DiGraph  
        for _, v, attr in G.out_edges(u, data=True):
            w = edge_penalized_weight(
                    attr,           # curr_attr
                    attr_u,         # prev_attr
                    v,              # curr_node
                    u,              # prev_node
                    prev_node,       # prev_prev_node
                    beta=beta, 
                    gamma=gamma
                )
            if w == INF:
                continue
            dv = du + w
            if dv < dist.get(v, INF):
                dist[v] = dv
                parent[v] = (u, attr)
                heapq.heappush(
                    pq,
                    (dv, next(counter), v,     # current to next level node
                     u, prev_node,             # new prev_node / prev_prev
                     attr)
                )

    if target not in dist:
        raise nx.NetworkXNoPath

    path = [target]
    node = target
    while node != source:
        node, _ = parent[node]
        path.append(node)
    path.reverse()
    return dist[target], path

def path_to_latlon(path, swap=True):
    if swap:
        return [to_wgs84(x, y)[::-1] for x, y in path] # (lat, lon)
    else:
        return [to_wgs84(x, y) for x, y in path] # (lon, lat)

def run_evac(lats, lons, mesh_dir="outputs", timestep=None, alpha=3.0, beta=1.5, gamma=2.0):
    mesh_dir = Path(mesh_dir)
    edges, nodes_gdf, shelters_gdf = load_inputs(mesh_dir)

    # read fire data
    fires = gpd.read_file(
        mesh_dir / (f"fires_t{timestep}.geojson" if timestep is not None else "fires.geojson")
    ).to_crs(TARGET_CRS)
    fire_union = unary_union(fires.geometry)

    # build graph
    G = build_graph(edges, fire_union, alpha)

    # start and end points
    start_nodes = []
    for lat, lon in zip(lats, lons):
        x, y = Transformer.from_crs(4326, TARGET_CRS, always_xy=True).transform(lon, lat)
        start_nodes.append(nearest_node(Point(x, y), list(G.nodes())))
    
    print(f"Processing {len(start_nodes)} vehicles")
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    routes = [] # save best result for each vehicle
    for idx, start_node in enumerate(start_nodes):
        best_cost, best_path = math.inf, None
        print(f"\nVehicle {idx+1}:")
        print(f"  Start: {start_node}, out_degree={G.out_degree(start_node)}")
        
        # Track which shelters are reachable
        reachable_shelters = 0
        for _, srow in shelters_gdf.iterrows():
            sx, sy = srow.geometry.coords[0]
            shelter_node = nearest_node(Point(sx, sy), list(G.nodes()))
            try:
                cost, path = constrained_shortest_path(G, start_node, shelter_node, beta=beta, gamma=gamma)
                if cost < best_cost:
                    best_cost, best_path, goal_node = cost, path, shelter_node
                reachable_shelters += 1
            except nx.NetworkXNoPath:
                continue

        if best_path is None:
            print(f"  No path found! All {len(shelters_gdf)} shelters unreachable")
            raise RuntimeError(f"No path for car #{idx}")
        else:
            print(f"  Found path: {reachable_shelters}/{len(shelters_gdf)} shelters reachable")
            print(f"  Best path: cost={best_cost:.1f}s, length={len(best_path)} nodes")
            
        for u, v in zip(best_path[:-1], best_path[1:]):
            G[u][v]["vehicles"] += 1
        routes.append((best_cost, best_path, start_node, goal_node))
    
    # Write statistics
    import csv, statistics
    stats_path = Path(f"evac_stats_beta{beta:g}_gamma{gamma:g}.csv")

    with stats_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "cost_s", "n_edges", "avg_speed_kph", "phy_speed_kph"])

        cost_list = []
        for i, (cost_s, path, _, _) in enumerate(routes):
            # average physical speed = road length / cost_s
            total_len = sum(G[u][v]["cost"] for u, v in zip(path[:-1], path[1:]))
            total_dis = sum(G[u][v]["length_m"] for u, v in zip(path[:-1], path[1:]))
            avg_speed = (total_dis / cost_s) * 3.6 if cost_s > 0 else 0   # m/s to km/h
            phys_sec = sum(G[u][v]["phys_time"] for u, v in zip(path[:-1], path[1:]))
            phys_speed_kph = (total_dis / phys_sec) * 3.6 if phys_sec > 0 else 0
            writer.writerow([i, round(cost_s, 1), len(path), round(avg_speed, 1), round(phys_speed_kph, 1)])
            cost_list.append(cost_s)

    # calculate statistics
    if cost_list:
        p95 = statistics.quantiles(cost_list, n=20)[-1] if len(cost_list) > 1 else cost_list[0]
        print(f"\n*** Summary: max(cost_s) = {max(cost_list):.1f}s  |  p95 = {p95:.1f}s ***")

    return routes, fires

def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.coord_file:
        lats, lons = [], []
        with args.coord_file.open() as f:
            for line in f:
                if not line.strip():
                    continue
                lat, lon = map(float, line.replace(',', ' ').split())
                lats.append(lat); lons.append(lon)
    else:                                           # command line --lat / --lon
        lats, lons = args.lat, args.lon

    routes, fires = run_evac(lats, lons,
                             mesh_dir=args.mesh_dir,
                             timestep=args.timestep,
                             alpha=args.alpha,
                             beta=args.beta,
                             gamma=args.gamma)

    print(f"\nSaving {len(routes)} route outputs...")
    for i, (cost, utm_path, s_node, g_node) in enumerate(routes):
        latlon_path = path_to_latlon(utm_path)
        print(f"[Car {i+1}] cost={cost:.1f}s, path_length={len(latlon_path)} points")
        
        # output geojson & html
        out_dir = Path(args.out_dir); out_dir.mkdir(exist_ok=True)
        gpd.GeoDataFrame({"cost_s":[cost]},
                         geometry=[LineString([(lon, lat) for lat, lon in latlon_path])])\
            .set_crs(4326).to_file(out_dir / f"route_{i}.geojson", driver="GeoJSON")
        add_route_map(utm_path, fires, s_node, g_node, out_dir / f"evac_{i}.html")
    
    print(f"✓ All outputs saved to {args.out_dir}/")

if __name__ == "__main__":
    main(sys.argv[1:])