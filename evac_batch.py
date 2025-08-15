"""
evac_batch.py – Multi-batch evacuation simulation
    python evac_batch.py --coord-file coords.txt --batch-size 20 \
        --mesh-dir outputs --out-dir multi_outputs
"""
from __future__ import annotations
import argparse, sys, math
from pathlib import Path
from typing import List
from shapely.ops import unary_union
from shapely.geometry import Point, LineString
from pyproj import Transformer
import geopandas as gpd
import networkx as nx
import folium

# ---------- Reused functions ------------------------------------------------------
from evac_multi import (load_inputs, build_graph, constrained_shortest_path,
                        path_to_latlon, add_route_map, nearest_node)

DECAY = 0.6
TARGET_CRS = "EPSG:32610"
TRANS_4326_TO_UTM = Transformer.from_crs("EPSG:4326", TARGET_CRS, always_xy=True)

# -------------------------------------------------------------------------
def read_coords(coord_file: Path) -> tuple[List[float], List[float]]:
    """coords.txt each line: lat,lon (comma or space separated)"""
    lats, lons = [], []
    with coord_file.open() as f:
        for ln in f:
            if not ln.strip():
                continue
            parts = ln.replace(",", " ").split()
            if len(parts) != 2:
                raise ValueError(f"Bad line: {ln.strip()}")
            lat, lon = map(float, parts)
            lats.append(lat); lons.append(lon)
    return lats, lons

# -------------------------------------------------------------------------
def solve_batch(G, shelters_gdf, start_nodes, *, beta, gamma):
    """
    Returns two items:
      routes       = [(cost, path, s_node, g_node), ...]  (reachable)
      bad_starts   = [s_node1, s_node2, ...]              (unreachable)
    """
    routes, bad_starts = [], []

    for s_node in start_nodes:
        best_cost, best_path, best_goal = math.inf, None, None
        for _, srow in shelters_gdf.iterrows():
            g_node = nearest_node(srow.geometry, G.nodes())
            
            # Check if start and goal are the same
            if s_node == g_node:
                # Start is at shelter, create a path with two identical points
                best_cost = 0
                best_path = [s_node, s_node]  # Ensure at least 2 points
                best_goal = g_node
                break
                
            try:
                cost, path = constrained_shortest_path(
                    G, s_node, g_node, beta=beta, gamma=gamma
                )
                if cost < best_cost:
                    best_cost, best_path, best_goal = cost, path, g_node
            except nx.NetworkXNoPath:
                continue

        if best_path is None:
            bad_starts.append(s_node)
            continue

        # Ensure path has at least 2 points
        if len(best_path) < 2:
            best_path = [best_path[0], best_path[0]] if best_path else [s_node, s_node]

        # Congestion accumulation (only for actual moving edges)
        if len(best_path) > 1 and best_path[0] != best_path[-1]:
            for u, v in zip(best_path[:-1], best_path[1:]):
                if u != v and G.has_edge(u, v):  # Ensure edge exists and no self-loop
                    G[u][v]["vehicles"] = G[u][v].get("vehicles", 0) + 1
                    
        routes.append((best_cost, best_path, s_node, best_goal))

    return routes, bad_starts


# -------------------------------------------------------------------------
def write_outputs(routes, fires_gdf, *, batch_id: int, out_dir: Path):
    out_dir.mkdir(exist_ok=True, parents=True)
    
    # Batch save flag to avoid duplicate printing
    saved_count = 0
    skipped_count = 0
    
    for i, (_c, path, s_node, g_node) in enumerate(routes):
        # Check path length
        if not path or len(path) < 2:
            print(f"    Warning: Route {i} has insufficient points ({len(path) if path else 0}), skipping")
            skipped_count += 1
            continue
            
        # Convert coordinates
        try:
            latlon = path_to_latlon(path)
            
            # Check converted coordinates again
            if len(latlon) < 2:
                print(f"    Warning: Converted route {i} has insufficient points, skipping")
                skipped_count += 1
                continue
                
            # Create LineString
            line = LineString([(lon, lat) for lat, lon in latlon])
            
            # Save GeoJSON
            gpd.GeoDataFrame({"batch": [batch_id], "car": [i]},
                             geometry=[line], crs="EPSG:4326")\
                .to_file(out_dir / f"route_b{batch_id}_c{i}.geojson", driver="GeoJSON")

            # HTML preview
            add_route_map(path, fires_gdf, s_node, g_node,
                          out_dir / f"evac_b{batch_id}_c{i}.html")
            saved_count += 1
            
        except Exception as e:
            print(f"    Error processing route {i}: {e}")
            skipped_count += 1
            continue
    
    # Print summary only once
    if saved_count > 0:
        print(f"  Saved {saved_count} routes for batch {batch_id}")
    if skipped_count > 0:
        print(f"  Skipped {skipped_count} invalid routes")

# -------------------------------------------------------------------------
def run_batches(lats, lons, *, batch_size, mesh_dir, timestep,
                alpha, beta, gamma, out_dir):
    param_tag = f"beta{beta}_gamma{gamma}".replace('.', '_')
    mesh_dir = Path(mesh_dir)
    out_dir  = Path(out_dir)
    edges, nodes_gdf, shelters_gdf = load_inputs(mesh_dir)
    fires_gdf = gpd.read_file(
        mesh_dir / (f"fires_t{timestep}.geojson" if timestep is not None else "fires.geojson")
    )
    G = build_graph(edges, unary_union(fires_gdf.geometry), alpha)

    # Start points → nearest graph nodes
    starts = []
    unreachable_coords = 0
    for lat, lon in zip(lats, lons):
        try:
            utm_x, utm_y = TRANS_4326_TO_UTM.transform(lon, lat)
            node = nearest_node(Point(utm_x, utm_y), list(G.nodes()))
            if node:
                starts.append(node)
            else:
                unreachable_coords += 1
        except:
            unreachable_coords += 1
            
    if unreachable_coords > 0:
        print(f"Warning: {unreachable_coords} coordinates could not be mapped to nodes")
        
    batches = [starts[i:i+batch_size] for i in range(0, len(starts), batch_size)]
    stats_rows = [] 
    
    print(f"\nRunning {len(batches)} batches with {len(starts)} total vehicles")
    print(f"Parameters: alpha={alpha}, beta={beta}, gamma={gamma}")
    
    for b, start_nodes in enumerate(batches):
        print(f"\nBatch {b+1}/{len(batches)} - {len(start_nodes)} cars")
        routes, bad = solve_batch(G, shelters_gdf, start_nodes,
                          beta=beta, gamma=gamma)
        if bad:
            print(f"  Warning: Skipped {len(bad)} unreachable starts")

        write_outputs(routes, fires_gdf, batch_id=b, out_dir=out_dir)

        # -------- Generate statistics rows (one row per car) --------------------------
        for i, (cost, path, s_node, g_node) in enumerate(routes):
            # Calculate path length, average saturation ρ, average speed
            edge_rhos = []  
            edge_len  = []
            
            # Process path (skip self-loops)
            for u, v in zip(path[:-1], path[1:]):
                if u != v and G.has_edge(u, v):
                    attr = G[u][v]
                    edge_rhos.append(attr.get("vehicles", 0) / attr.get("capacity", 30))
                    edge_len.append(attr.get("length_m", attr.get("length", 10)))
                    
            length_m    = sum(edge_len) if edge_len else 0
            travel_time = cost if cost < math.inf else 0
            rho_mean    = sum(edge_rhos) / len(edge_rhos) if edge_rhos else 0
            speed_kph   = 3.6 * length_m / travel_time if travel_time > 0 else 0

            stats_rows.append({
                "batch":  b,
                "car":    i,
                "nodes":  len(path),
                "length_m": round(length_m, 1),
                "travel_time": round(travel_time, 2),
                "rho_mean":   round(rho_mean, 3),
                "speed_kph":  round(speed_kph, 1),
                "start_x": s_node[0], "start_y": s_node[1],
                "goal_x":  g_node[0], "goal_y":  g_node[1],
            })
        
        # Update congestion state
        if G.number_of_edges() > 0:
            max_rho_before = max((d.get("vehicles", 0)/d.get("capacity", 30) 
                                 for _,_,d in G.edges(data=True)), default=0)
            for _, _, data in G.edges(data=True):
                data["vehicles"] = data.get("vehicles", 0) * DECAY
            max_rho_after  = max((d.get("vehicles", 0)/d.get("capacity", 30) 
                                 for _,_,d in G.edges(data=True)), default=0)
            print(f"  Max rho: before={max_rho_before:.2f}, after decay={max_rho_after:.2f}")

    # ----------- Write CSV summary ----------------------------------
    if stats_rows:
        import pandas as pd
        df = pd.DataFrame(stats_rows)

        # Details
        detail_csv = out_dir / f"evac_stats_{param_tag}.csv"
        df.to_csv(detail_csv, index=False)
        print(f"\nDetail stats saved to {detail_csv}")

        # Batch aggregation
        gb = df.groupby("batch")
        summary = pd.DataFrame({
            "cars": gb.size(),
            "mean_time" : gb["travel_time"].mean().round(2),
            "p95_time"  : gb["travel_time"].quantile(0.95).round(2),
            "mean_rho"  : gb["rho_mean"].mean().round(3),
            "mean_speed": gb["speed_kph"].mean().round(1),
        })

        # Relative coefficients
        if len(summary) > 0:
            base = summary["mean_time"].iloc[0]
            if base > 0:
                summary["ratio_to_batch0"] = (summary["mean_time"] / base).round(3)

        summary_csv = out_dir / f"evac_summary_{param_tag}.csv"
        summary.to_csv(summary_csv)
        print(f"Summary stats saved to {summary_csv}")
        
        # Print summary statistics
        print(f"\n=== Batch Summary ===")
        print(summary.to_string())

# -------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(description="Multi-batch evacuation simulation")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--coord-file", type=Path, help="Text coordinate file, each line lat,lon")
    g.add_argument("--coords", nargs="+", help="Direct coordinates e.g. 49.72,-123.1 49.73,-123.09")
    p.add_argument("--batch-size", type=int, required=True, help="Vehicles per batch")
    p.add_argument("--mesh-dir", default="outputs", help="Mesh/fire data directory")
    p.add_argument("--out-dir",  default="multi_outputs", help="Output directory")
    p.add_argument("--timestep", type=int, help="Fire timestep (optional)")
    p.add_argument("--alpha", type=float, default=3.0)
    p.add_argument("--beta",  type=float, default=1.5)
    p.add_argument("--gamma", type=float, default=2.0)
    return p

def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.coord_file:
        lats, lons = read_coords(args.coord_file)
    else:
        if len(args.coords) % 2:
            sys.exit("coords must be given in pairs: lat lon")
        coords = list(map(float, args.coords))
        lats = coords[::2]
        lons = coords[1::2]

    print(f"Processing {len(lats)} vehicles in batches of {args.batch_size}")
    
    run_batches(lats, lons,
                batch_size=args.batch_size,
                mesh_dir=args.mesh_dir,
                timestep=args.timestep,
                alpha=args.alpha, beta=args.beta, gamma=args.gamma,
                out_dir=args.out_dir)

if __name__ == "__main__":
    main()