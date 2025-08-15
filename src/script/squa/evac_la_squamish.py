from __future__ import annotations
from shapely.ops import unary_union
from shapely.geometry import Point, LineString
from pyproj import Transformer
from pathlib import Path
from typing import Tuple, List
import heapq, itertools
import geopandas as gpd
import networkx as nx
import folium
import argparse, sys, math
import sqlite3
import pandas as pd

REGIONS: dict[str, dict] = {
    "squamish": {
        "target_crs": "EPSG:32610",
        "bbox": (-123.30, 49.60, -122.90, 50.00),  # (min_lon, min_lat, max_lon, max_lat)
        "mesh_edges": "mesh_edges.geojson",
        "mesh_nodes": "mesh_nodes.geojson",
        "shelters": "shelters.geojson",
    },
    "la": {
        "target_crs": "EPSG:32611",
        "bbox": (-119.00, 33.00, -117.00, 35.00),
        "mesh_edges": "la_mesh_edges.geojson",
        "mesh_nodes": "la_mesh_nodes.geojson",
        "shelters": "la_shelters.geojson",
    },
}
HIWAY_TYPES = {"motorway", "trunk"}
IS_LINK = lambda rt: rt and rt.endswith("_link")
INF = float("inf")
SEA2SKY = ("sea-to-sky highway", "Sea-to-Sky Highway")

def detect_region(lat: float, lon: float) -> str:
    """Return region key based on bounding boxes or raise ValueError."""
    for name, cfg in REGIONS.items():
        min_lon, min_lat, max_lon, max_lat = cfg["bbox"]
        if min_lon <= lon <= max_lon and min_lat <= lat <= max_lat:
            return name
    raise ValueError("Coordinates outside supported regions; use --region to override.")


def to_wgs84(x: float, y: float, target_crs: str) -> Tuple[float, float]:
    return Transformer.from_crs(target_crs, 4326, always_xy=True).transform(x, y)

def load_shelters_from_db(region: str, target_crs: str, verbose: bool = True) -> gpd.GeoDataFrame:
    """Load shelters from SQLite database and convert to GeoDataFrame."""
    # 数据库文件路径（相对于脚本位置）
    db_path = Path(__file__).parent.parent.parent.parent / "prisma" / "sql.db"

    try:
        # 连接数据库
        conn = sqlite3.connect(db_path)

        # 查询指定地区的shelter数据
        query = """
        SELECT shelterId, lat, lng, capacity, hexId, region
        FROM Shelter
        WHERE region = ?
        ORDER BY shelterId
        """

        # 读取数据到DataFrame
        df = pd.read_sql_query(query, conn, params=(region,))
        conn.close()

        if df.empty:
            if verbose:
                print(f"WARNING: No shelter data found in database for {region} region")
            # 返回空的GeoDataFrame但保持正确的结构
            return gpd.GeoDataFrame(columns=['shelter_id', 'capacity', 'hex_id', 'geometry'])

        # 创建Point几何对象
        geometry = [Point(lng, lat) for lat, lng in zip(df['lat'], df['lng'])]

        # 创建GeoDataFrame
        gdf = gpd.GeoDataFrame({
            'shelter_id': df['shelterId'],
            'capacity': df['capacity'],
            'hex_id': df['hexId'],
            'geometry': geometry
        }, crs='EPSG:4326')  # WGS84坐标系

        # 转换到目标坐标系
        gdf = gdf.to_crs(target_crs)

        if verbose:
            print(f"SUCCESS: Loaded {len(gdf)} shelters from database for {region} region")
        return gdf

    except Exception as e:
        if verbose:
            print(f"ERROR: Failed to load shelters from database: {e}")
            print(f"   Database path: {db_path}")
            # 如果数据库读取失败，回退到GeoJSON文件
            print("   Falling back to GeoJSON file...")
        return None

def load_inputs(mesh_dir: Path, cfg: dict, region: str, verbose: bool = True):
    """Load edges/nodes/shelters and project to region CRS. Shelters loaded from database."""
    edges = gpd.read_file(mesh_dir / cfg["mesh_edges"]).to_crs(cfg["target_crs"])
    nodes = gpd.read_file(mesh_dir / cfg["mesh_nodes"]).to_crs(cfg["target_crs"])

    # 尝试从数据库加载shelter数据
    shelters = load_shelters_from_db(region, cfg["target_crs"], verbose)

    # 如果数据库加载失败，回退到GeoJSON文件
    if shelters is None:
        if verbose:
            print(f"   Using GeoJSON file: {cfg['shelters']}")
        shelters = gpd.read_file(mesh_dir / cfg["shelters"]).to_crs(cfg["target_crs"])

    return edges, nodes, shelters

def load_fire_perimeter(path: Path, target_crs: str) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)
    # 保留 type 含 "Heat" 的多边形；若字段缺失则全部保留
    gdf = gdf[gdf.get("type").fillna("").str.contains("Heat", case=False)]
    if gdf.empty:
        raise ValueError(f"No heat perimeter polygons found in {path}")
    merged = unary_union(gdf.geometry)
    return gpd.GeoDataFrame({"geometry": [merged]}, crs=gdf.crs).to_crs(target_crs)

# ---------------------------------------------------------------------------
# Graph construction & helpers
# ---------------------------------------------------------------------------

def is_hiway(rt: str, name: str):
    return (rt in HIWAY_TYPES) or ("highway" in name)


def edge_penalized_weight(curr_attr, prev_attr, curr_node=None, prev_node=None, prev_prev_node=None):
    """Weight with penalties for highway transitions & U‑turns (borrowed from both original scripts)."""
    if prev_attr is None:
        return curr_attr["weight"]

    rt_c, nm_c = curr_attr.get("road_type", ""), curr_attr.get("road_name", "")
    rt_p, nm_p = prev_attr.get("road_type", ""), prev_attr.get("road_name", "")
    hi_c, hi_p = is_hiway(rt_c, nm_c), is_hiway(rt_p, nm_p)
    link_c, link_p = IS_LINK(rt_c), IS_LINK(rt_p)

    # Sea‑to‑Sky specific restriction – harmless for LA data
    sea_in = any(k in nm_c for k in SEA2SKY)
    sea_out = any(k in nm_p for k in SEA2SKY)
    if sea_in ^ sea_out and not (link_c or link_p):
        return INF

    # Trunk/motorway ↔ ordinary road transitions (discourage)
    if hi_c and not (hi_p or link_p):
        return curr_attr["weight"] * 20
    if hi_p and not (hi_c or link_c):
        return curr_attr["weight"] * 20

    # Heavy penalty (or forbid) for U‑turns (>155°)
    if prev_prev_node is not None:
        v1 = (prev_node[0] - prev_prev_node[0], prev_node[1] - prev_prev_node[1])
        v2 = (curr_node[0] - prev_node[0], curr_node[1] - prev_node[1])
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        norm = ((v1[0]**2 + v1[1]**2)**0.5 * (v2[0]**2 + v2[1]**2)**0.5)
        if norm and dot / norm < -0.9:  # ≈155°
            if curr_attr.get("road_type") in HIWAY_TYPES:
                return INF
            return curr_attr["weight"] * 20
    return curr_attr["weight"]


# mesh_edges to NetworkX DiGraph
def build_graph(edges: gpd.GeoDataFrame, fire_union, alpha: float) -> nx.Graph:
    G = nx.DiGraph()
    # collect link end
    link_eps: set[tuple[float,float]] = set()
    for _, r in edges.iterrows():
        rt = r.get("road_type","")
        if rt.endswith("_link"):
            pts = list(r.geometry.coords)
            link_eps.add(tuple(pts[0]))
            link_eps.add(tuple(pts[-1]))
    for _, row in edges.iterrows():
        # if it's motorway or trunk
        # and end not in link_eps, get over it  
        rt = row.get("road_type","")
        a = tuple(row.geometry.coords[0])
        b = tuple(row.geometry.coords[-1])
        # use mesh_edges.geojson (oneway_dir) built edges
        dir_tag = ("FWD" if row.get("direction") == "forward" else "REV") if rt.endswith("_link") else row.get("oneway_dir", "BIDIR")
        base_cost = row.get("cost", row.length)
        inter_len = row.geometry.intersection(fire_union).length
        ratio     = inter_len / row.length
        weight    = base_cost * (1 + alpha * ratio)

        if dir_tag == "FWD":
            # 顺向单行
            G.add_edge(a, b,
                cost=base_cost, exp_ratio=ratio,
                weight=weight, road_type=rt,
                road_name=(row.get("road_name","") or "").lower(),
                oneway=True)
        elif dir_tag == "REV":
            # 逆向单行
            G.add_edge(b, a,
                cost=base_cost, exp_ratio=ratio,
                weight=weight, road_type=rt,
                road_name=(row.get("road_name","") or "").lower(),
                oneway=True)
        else:
            G.add_edge(a, b,
                cost=base_cost, exp_ratio=ratio,
                weight=weight, road_type=rt,
                road_name=(row.get("road_name","") or "").lower(),
                oneway=True)
            G.add_edge(b, a,
                cost=base_cost, exp_ratio=ratio,
                weight=weight, road_type=rt,
                road_name=(row.get("road_name","") or "").lower(),
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


def constrained_shortest_path(G: nx.DiGraph, source, target):
    # return cost and path list
    counter = itertools.count()
    pq   = [(0.0, next(counter), source,  None,     None,     None)]
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
                    prev_node       # prev_prev_node
                )
            if w == INF:
                continue
            dv = du + w
            if dv < dist.get(v, INF):
                dist[v] = dv
                parent[v] = (u, attr)
                heapq.heappush(
                    pq,
                    (dv, next(counter), v,     # 当前→下一层 node
                     u, prev_node,             # 新 prev_node / prev_prev
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


def add_route_map(route, fires, start_pt, goal_pt, out_path: Path, cfg: dict):
    lon_c, lat_c = to_wgs84(*start_pt, cfg["target_crs"])
    m = folium.Map(location=[lat_c, lon_c], zoom_start=14)
    folium.GeoJson(
        fires.to_crs(4326),
        name="fires",
        style_function=lambda _: {
            "color": "#ff6600",
            "weight": 2,
            "fillColor": "#ff6600",
            "fillOpacity": 0.3,
        },
    ).add_to(m)
    line_latlon = [to_wgs84(x, y, cfg["target_crs"])[::-1] for x, y in route]
    folium.PolyLine(line_latlon, color="#2686CC", weight=4, opacity=0.9).add_to(m)
    folium.Marker(line_latlon[0], icon=folium.Icon(color="green"), tooltip="Start").add_to(m)
    folium.Marker(line_latlon[-1], icon=folium.Icon(color="red"), tooltip="Shelter").add_to(m)
    folium.LayerControl().add_to(m)
    m.save(out_path)


# ---------------------------------------------------------------------------
# Main run & CLI
# ---------------------------------------------------------------------------

def run_evac(lat: float, lon: float, *, mesh_dir: str | Path = "outputs", fire_file: str = "fire_combined.geojson", alpha: float = 3.0, region: str | None = None, verbose: bool = True):
    if region is None or region == "auto":
        region = detect_region(lat, lon)
    if region not in REGIONS:
        raise ValueError(f"Unsupported region '{region}'. Choose from {list(REGIONS)}")
    cfg = REGIONS[region]

    mesh_dir = Path(mesh_dir)
    edges, nodes_gdf, shelters_gdf = load_inputs(mesh_dir, cfg, region, verbose)
    fires = load_fire_perimeter(Path(mesh_dir) / fire_file, cfg["target_crs"])
    fire_union = unary_union(fires.geometry)

    # Build graph
    G = build_graph(edges, fire_union, alpha)

    # Start & shelters
    x0, y0 = Transformer.from_crs(4326, cfg["target_crs"], always_xy=True).transform(lon, lat)
    start_node = nearest_node(Point(x0, y0), list(G.nodes()))

    best_cost, best_path, goal_node = math.inf, None, None
    for _, srow in shelters_gdf.iterrows():
        sx, sy = srow.geometry.coords[0]
        shelter_node = nearest_node(Point(sx, sy), list(G.nodes()))
        try:
            cost, path = constrained_shortest_path(G, start_node, shelter_node)
            if cost < best_cost:
                best_cost, best_path, goal_node = cost, path, shelter_node
        except nx.NetworkXNoPath:
            continue

    if best_path is None:
        raise RuntimeError("No path to any shelter")

    # Convert path to (lat, lon)
    latlon_path = [to_wgs84(x, y, cfg["target_crs"])[::-1] for x, y in best_path]
    return latlon_path, best_cost, fires, start_node, goal_node, best_path, cfg


def build_parser():
    p = argparse.ArgumentParser("Unified evacuation planner")
    p.add_argument("lat", type=float, help="Start latitude (WGS84)")
    p.add_argument("lon", type=float, help="Start longitude (WGS84)")
    p.add_argument("--region", choices=[*REGIONS.keys(), "auto"], default="auto", help="Region override (default=auto‑detect)")
    p.add_argument("--mesh-dir", default="outputs")
    p.add_argument("--fire-file", default="fire_combined.geojson")
    p.add_argument("--out-dir", default="outputs")
    p.add_argument("--alpha", type=float, default=3.0, help="Fire cost multiplier alpha")
    p.add_argument("--json", action="store_true", help="Print JSON only")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    mesh_dir = Path(args.mesh_dir).expanduser()
    if not mesh_dir.is_absolute():          # 仍然是相对路径
        mesh_dir = Path(__file__).parent / mesh_dir

    latlon_path, best_cost, fires, start_node, goal_node, utm_path, cfg = run_evac(
        args.lat,
        args.lon,
        mesh_dir=mesh_dir,
        fire_file=args.fire_file,
        alpha=args.alpha,
        region=args.region,
        verbose=not args.json,  # 在JSON模式下禁用verbose输出
    )

    resp = dict(
        route = latlon_path,
        cost_s= round(best_cost, 1),
        n_points=len(latlon_path),
        start = latlon_path[0],
        destination = latlon_path[-1]
    )

    if args.json:
        import json, sys
        json.dump(resp, sys.stdout, ensure_ascii=False)
        sys.stdout.flush()
        return
    
if __name__ == "__main__":
    main(sys.argv[1:])