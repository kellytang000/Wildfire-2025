"""
Data preprocessing - Extract subregion from large-scale road network that meets fire evacuation standards
"""
import geopandas as gpd
from shapely.geometry import Point, box, LineString
from shapely.ops import unary_union
import networkx as nx
from pathlib import Path
import numpy as np
import random
import pandas as pd

def extract_subregion(input_dir="outputs", output_dir="preprocess_data", 
                     center_point=None, radius=500, max_nodes=50,
                     num_shelters=3, num_fire_sources=2):
    """
    Extract subregion from large-scale network that meets fire evacuation standards
    
    Args:
        input_dir: Input directory
        output_dir: Output directory
        center_point: Center point coordinates
        radius: Extraction radius (meters)
        max_nodes: Maximum number of nodes
        num_shelters: Number of shelters
        num_fire_sources: Number of fire sources
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("="*70)
    print("Fire Evacuation Data Preprocessing")
    print("="*70)
    print(f"Configuration:")
    print(f"  Input directory: {input_dir}")
    print(f"  Output directory: {output_dir}")
    print(f"  Extraction radius: {radius} meters")
    print(f"  Max nodes: {max_nodes}")
    print(f"  Number of shelters: {num_shelters}")
    print(f"  Number of fire sources: {num_fire_sources}")
    
    edges_file = input_dir / "mesh_edges.geojson"
    if not edges_file.exists():
        raise FileNotFoundError(f"Error: Input file not found {edges_file}")
    
    edges_gdf = gpd.read_file(edges_file).to_crs("EPSG:32610")
    print(f"Original edges loaded: {len(edges_gdf)}")
    
    nodes_file = input_dir / "mesh_nodes.geojson"
    if nodes_file.exists():
        original_nodes_gdf = gpd.read_file(nodes_file).to_crs("EPSG:32610")
        print(f"Original nodes loaded: {len(original_nodes_gdf)}")
    
    G = nx.Graph()
    for _, edge in edges_gdf.iterrows():
        coords = list(edge.geometry.coords)
        p1, p2 = coords[0], coords[-1]
        
        edge_data = edge.to_dict()
        
        if 'road_type' not in edge_data or pd.isna(edge_data.get('road_type')):
            if 'highway' in str(edge_data.get('road_name', '')).lower():
                edge_data['road_type'] = 'trunk'
            elif edge_data.get('speed_kph', 30) > 60:
                edge_data['road_type'] = 'primary'
            elif edge_data.get('speed_kph', 30) > 40:
                edge_data['road_type'] = 'secondary'
            else:
                edge_data['road_type'] = 'residential'
        
        if 'speed_kph' not in edge_data or pd.isna(edge_data.get('speed_kph')):
            speed_map = {
                'motorway': 100, 'trunk': 80, 'primary': 60,
                'secondary': 50, 'tertiary': 40, 'residential': 30
            }
            edge_data['speed_kph'] = speed_map.get(edge_data.get('road_type', 'residential'), 30)
        
        if 'cap' not in edge_data or pd.isna(edge_data.get('cap')):
            cap_map = {
                'motorway': 150, 'trunk': 100, 'primary': 60,
                'secondary': 40, 'tertiary': 30, 'residential': 20
            }
            edge_data['cap'] = cap_map.get(edge_data.get('road_type', 'residential'), 30)
        
        if 'length' not in edge_data:
            edge_data['length'] = edge.geometry.length
        if 'cost' not in edge_data:
            edge_data['cost'] = edge_data['length']
        
        G.add_edge(p1, p2, **edge_data)
    
    print(f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    if center_point is None:
        degree_centrality = nx.degree_centrality(G)
        top_nodes = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)
        top_5_percent = max(1, len(top_nodes) // 20)
        candidates = [node for node, _ in top_nodes[:top_5_percent]]
        center_node = random.choice(candidates[:min(10, len(candidates))])
        center_point = center_node
    else:
        nodes = list(G.nodes())
        distances = [Point(n).distance(Point(center_point)) for n in nodes]
        center_node = nodes[np.argmin(distances)]
        center_point = center_node
    
    print(f"Center point selected: {center_point[:2] if len(center_point) > 2 else center_point}")
    
    subgraph_nodes = set()
    visited = set()
    queue = [(center_node, 0, 1.0)]
    
    node_degrees = dict(G.degree())
    max_degree = max(node_degrees.values()) if node_degrees else 1
    
    while queue and len(subgraph_nodes) < max_nodes:
        queue.sort(key=lambda x: -x[2])
        node, dist, priority = queue.pop(0)
        
        if node in visited:
            continue
            
        visited.add(node)
        node_point = Point(node[:2] if len(node) > 2 else node)
        center_pt = Point(center_point[:2] if len(center_point) > 2 else center_point)
        
        if node_point.distance(center_pt) <= radius:
            subgraph_nodes.add(node)
            
            for neighbor in G.neighbors(node):
                if neighbor not in visited:
                    neighbor_degree = node_degrees.get(neighbor, 1)
                    degree_priority = neighbor_degree / max_degree
                    
                    neighbor_pt = Point(neighbor[:2] if len(neighbor) > 2 else neighbor)
                    distance_factor = 1.0 / (1 + neighbor_pt.distance(center_pt) / radius)
                    
                    combined_priority = degree_priority * distance_factor
                    queue.append((neighbor, dist + 1, combined_priority))
    
    print(f"Nodes extracted: {len(subgraph_nodes)}")
    
    subgraph = G.subgraph(subgraph_nodes).copy()
    
    if not nx.is_connected(subgraph):
        components = list(nx.connected_components(subgraph))
        large_components = [c for c in components if len(c) > 10]
        
        if large_components:
            for comp in large_components:
                if center_node in comp:
                    subgraph = subgraph.subgraph(comp).copy()
                    break
            else:
                largest = max(large_components, key=len)
                subgraph = subgraph.subgraph(largest).copy()
        else:
            largest_cc = max(nx.connected_components(subgraph), key=len)
            subgraph = subgraph.subgraph(largest_cc).copy()
        
        print(f"Adjusted nodes: {subgraph.number_of_nodes()}")
    
    sub_edges = []
    for u, v, data in subgraph.edges(data=True):
        edge_data = data.copy()
        edge_data['geometry'] = edge_data.get('geometry', LineString([u, v]))
        
        edge_data['road_type'] = edge_data.get('road_type', 'residential')
        edge_data['road_name'] = edge_data.get('road_name', f'Road_{len(sub_edges)}')
        edge_data['speed_kph'] = edge_data.get('speed_kph', 30)
        edge_data['cost'] = edge_data.get('cost', edge_data['geometry'].length)
        edge_data['length'] = edge_data.get('length', edge_data['geometry'].length)
        edge_data['cap'] = edge_data.get('cap', 30)
        
        if 'oneway_dir' not in edge_data:
            if edge_data['road_type'] in ['trunk', 'primary', 'motorway']:
                edge_data['oneway_dir'] = 'BIDIR'
            else:
                edge_data['oneway_dir'] = random.choice(['BIDIR'] * 4 + ['FWD'])
        
        sub_edges.append(edge_data)
    
    sub_edges_gdf = gpd.GeoDataFrame(sub_edges, crs="EPSG:32610")
    
    nodes_data = []
    for i, node in enumerate(subgraph.nodes()):
        node_point = Point(node[:2] if len(node) > 2 else node)
        nodes_data.append({
            'node_id': i,
            'x': node_point.x,
            'y': node_point.y,
            'degree': subgraph.degree(node),
            'geometry': node_point
        })
    
    nodes_gdf = gpd.GeoDataFrame(nodes_data, crs="EPSG:32610")
    
    print("Selecting shelter locations...")
    shelter_candidates = []
    bounds = box(*nodes_gdf.total_bounds)
    
    for node in subgraph.nodes():
        node_point = Point(node[:2] if len(node) > 2 else node)
        dist_to_boundary = node_point.distance(bounds.boundary)
        if dist_to_boundary < radius * 0.1:
            shelter_candidates.append((node, -dist_to_boundary, 'boundary'))
    
    node_degrees = dict(subgraph.degree())
    sorted_by_degree = sorted(node_degrees.items(), key=lambda x: x[1], reverse=True)
    for node, degree in sorted_by_degree[:15]:
        if degree >= 3:
            shelter_candidates.append((node, degree * 100, 'hub'))
    
    center_pt = Point(center_point[:2] if len(center_point) > 2 else center_point)
    directions = [
        (1, 0), (-1, 0), (0, 1), (0, -1),
        (0.7, 0.7), (-0.7, 0.7), (0.7, -0.7), (-0.7, -0.7)
    ]
    
    for dx, dy in directions:
        target_x = center_pt.x + dx * radius * 0.7
        target_y = center_pt.y + dy * radius * 0.7
        target_point = Point(target_x, target_y)
        
        best_node = None
        min_dist = float('inf')
        for node in subgraph.nodes():
            node_point = Point(node[:2] if len(node) > 2 else node)
            dist = node_point.distance(target_point)
            if dist < min_dist and node_point.distance(center_pt) > radius * 0.3:
                min_dist = dist
                best_node = node
        
        if best_node:
            shelter_candidates.append((best_node, -min_dist, 'directional'))
    
    unique_shelters = {}
    for node, score, strategy in shelter_candidates:
        if node not in unique_shelters or abs(score) > abs(unique_shelters[node][0]):
            unique_shelters[node] = (score, strategy)
    
    selected_shelters = []
    min_shelter_distance = radius * 0.2
    
    for node, (score, strategy) in sorted(unique_shelters.items(), 
                                         key=lambda x: abs(x[1][0]), reverse=True):
        node_point = Point(node[:2] if len(node) > 2 else node)
        
        too_close = False
        for selected in selected_shelters:
            selected_point = Point(selected[:2] if len(selected) > 2 else selected)
            if node_point.distance(selected_point) < min_shelter_distance:
                too_close = True
                break
        
        if not too_close:
            selected_shelters.append(node)
            if len(selected_shelters) >= num_shelters:
                break
    
    if len(selected_shelters) < num_shelters:
        remaining_nodes = [n for n in subgraph.nodes() if n not in selected_shelters]
        random.shuffle(remaining_nodes)
        for node in remaining_nodes:
            if len(selected_shelters) >= num_shelters:
                break
            node_point = Point(node[:2] if len(node) > 2 else node)
            too_close = False
            for selected in selected_shelters:
                selected_point = Point(selected[:2] if len(selected) > 2 else selected)
                if node_point.distance(selected_point) < min_shelter_distance * 0.5:
                    too_close = True
                    break
            if not too_close:
                selected_shelters.append(node)
    
    shelter_types = ['Hospital', 'School', 'Stadium', 'Community_Center', 
                     'Mall', 'Park', 'Government_Building', 'Church']
    
    shelters_data = []
    for i, node in enumerate(selected_shelters[:num_shelters]):
        node_point = Point(node[:2] if len(node) > 2 else node)
        shelters_data.append({
            'shelter_id': i,
            'name': f'{shelter_types[i % len(shelter_types)]}_{i}',
            'capacity': random.randint(100, 500),
            'type': shelter_types[i % len(shelter_types)],
            'geometry': node_point
        })
    
    shelters_gdf = gpd.GeoDataFrame(shelters_data, crs="EPSG:32610")
    
    print("Generating fire scenarios...")
    fires_data = []
    
    main_fire_center = Point(center_point[:2] if len(center_point) > 2 else center_point)
    main_fire = main_fire_center.buffer(radius * 0.1)
    fires_data.append({
        'fire_id': 0,
        'type': 'primary',
        'intensity': 'high',
        'geometry': main_fire
    })
    
    used_locations = [main_fire_center]
    for i in range(1, num_fire_sources):
        attempts = 0
        while attempts < 50:
            fire_node = random.choice(list(subgraph.nodes()))
            fire_point = Point(fire_node[:2] if len(fire_node) > 2 else fire_node)
            
            min_dist_to_fire = min(fire_point.distance(loc) for loc in used_locations)
            if min_dist_to_fire > radius * 0.15:
                fire_size = random.uniform(radius * 0.05, radius * 0.08)
                fires_data.append({
                    'fire_id': i,
                    'type': 'secondary',
                    'intensity': random.choice(['medium', 'high']),
                    'geometry': fire_point.buffer(fire_size)
                })
                used_locations.append(fire_point)
                break
            attempts += 1
    
    fires_gdf = gpd.GeoDataFrame(fires_data, crs="EPSG:32610")
    
    x_min, y_min, x_max, y_max = nodes_gdf.total_bounds
    cell_size = radius / 5
    
    mesh_cells = []
    cell_id = 0
    for x in np.arange(x_min, x_max, cell_size):
        for y in np.arange(y_min, y_max, cell_size):
            cell = box(x, y, min(x + cell_size, x_max), min(y + cell_size, y_max))
            nodes_in_cell = sum(1 for n in subgraph.nodes() 
                              if cell.contains(Point(n[:2] if len(n) > 2 else n)))
            mesh_cells.append({
                'cell_id': cell_id,
                'nodes_count': nodes_in_cell,
                'geometry': cell
            })
            cell_id += 1
    
    mesh_gdf = gpd.GeoDataFrame(mesh_cells, crs="EPSG:32610")
    
    print("\nSaving output files...")
    nodes_gdf.to_file(output_dir / "mesh_nodes.geojson", driver="GeoJSON")
    sub_edges_gdf.to_file(output_dir / "mesh_edges.geojson", driver="GeoJSON")
    shelters_gdf.to_file(output_dir / "shelters.geojson", driver="GeoJSON")
    fires_gdf.to_file(output_dir / "fires.geojson", driver="GeoJSON")
    mesh_gdf.to_file(output_dir / "adaptive_mesh.geojson", driver="GeoJSON")
    
    print("\n" + "="*70)
    print("Data Preprocessing Complete")
    print("="*70)
    print(f"Output directory: {output_dir}/")
    print(f"Generated files:")
    print(f"  - mesh_nodes.geojson    : {len(nodes_gdf)} nodes")
    print(f"  - mesh_edges.geojson    : {len(sub_edges_gdf)} edges")
    print(f"  - shelters.geojson      : {len(shelters_gdf)} shelters")
    print(f"  - fires.geojson         : {len(fires_gdf)} fire sources")
    print(f"  - adaptive_mesh.geojson : {len(mesh_gdf)} mesh cells")
    
    if 'road_type' in sub_edges_gdf.columns:
        road_stats = sub_edges_gdf['road_type'].value_counts()
        print(f"\nRoad type distribution:")
        for road_type, count in road_stats.items():
            print(f"  {road_type:15s}: {count:4d} ({count/len(sub_edges_gdf)*100:5.1f}%)")
    
    degrees = nodes_gdf['degree'].values
    print(f"\nNode connectivity statistics:")
    print(f"  Average degree: {np.mean(degrees):.2f}")
    print(f"  Maximum degree: {np.max(degrees)}")
    print(f"  Minimum degree: {np.min(degrees)}")
    print(f"  Nodes with degree >= 3: {np.sum(degrees >= 3)}")
    
    print(f"\nShelter configuration:")
    total_capacity = 0
    for _, shelter in shelters_gdf.iterrows():
        print(f"  {shelter['name']:20s}: capacity {shelter['capacity']:3d} people")
        total_capacity += shelter['capacity']
    print(f"  Total capacity: {total_capacity} people")
    
    area_km2 = np.pi * (radius/1000) ** 2
    print(f"\nCoverage area:")
    print(f"  Radius: {radius} meters")
    print(f"  Estimated area: {area_km2:.2f} square kilometers")
    print("="*70)
    
    return output_dir


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fire evacuation data preprocessing")
    parser.add_argument("--input-dir", default="outputs", 
                       help="Input directory")
    parser.add_argument("--output-dir", default="preprocess_data", 
                       help="Output directory")
    parser.add_argument("--radius", type=int, default=500, 
                       help="Extraction radius in meters")
    parser.add_argument("--max-nodes", type=int, default=50, 
                       help="Maximum number of nodes")
    parser.add_argument("--num-shelters", type=int, default=3, 
                       help="Number of shelters")
    parser.add_argument("--num-fires", type=int, default=2, 
                       help="Number of fire sources")
    parser.add_argument("--center", type=float, nargs=2, default=None,
                       help="Center point coordinates x y")
    
    args = parser.parse_args()
    
    extract_subregion(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        center_point=tuple(args.center) if args.center else None,
        radius=args.radius,
        max_nodes=args.max_nodes,
        num_shelters=args.num_shelters,
        num_fire_sources=args.num_fires
    )