"""
Fire spread simulation script - supports wind direction and multiple timesteps
"""
from __future__ import annotations
import argparse, random, sys, uuid, math
from pathlib import Path
from typing import Tuple, List

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, Polygon, LineString
from shapely.ops import unary_union
from shapely.affinity import rotate, scale, translate
from pyproj import Transformer
import folium

TARGET_CRS = "EPSG:32610"
DEFAULT_FIRES  = 2
DEFAULT_MIN_R  = 500.0
DEFAULT_MAX_R  = 500.0
DEFAULT_TIMESTEPS = 5
DEFAULT_WIND_DIR = 90  # degrees clockwise from north

transformer_to_utm = Transformer.from_crs("EPSG:4326", TARGET_CRS, always_xy=True)
transformer_to_wgs = Transformer.from_crs(TARGET_CRS, "EPSG:4326", always_xy=True)

def load_mesh(mesh_dir: Path) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Load mesh data"""
    edges_gj = mesh_dir / "mesh_edges.geojson"
    mesh_gj  = mesh_dir / "adaptive_mesh.geojson"
    
    if not edges_gj.exists():
        raise FileNotFoundError(f"mesh_edges.geojson not found in {mesh_dir}")
    if not mesh_gj.exists():
        raise FileNotFoundError(f"adaptive_mesh.geojson not found in {mesh_dir}")
    
    edges = gpd.read_file(edges_gj).to_crs(TARGET_CRS)
    mesh  = gpd.read_file(mesh_gj).to_crs(TARGET_CRS)
    return mesh, edges


def get_fire_centers_from_existing(mesh_dir: Path) -> List[Tuple[float, float]]:
    """Get fire source centers from existing fire data"""
    fires_file = mesh_dir / "fires.geojson"
    if fires_file.exists():
        fires_gdf = gpd.read_file(fires_file).to_crs(TARGET_CRS)
        centers = []
        for _, fire in fires_gdf.iterrows():
            centroid = fire.geometry.centroid
            centers.append((centroid.x, centroid.y))
        return centers
    else:
        # If no existing fire file, return default center point
        return [(490314.0, 5515000.0)]  # Example coordinates


def generate_irregular_fire(center: Tuple[float, float],
                           rng: random.Random,
                           base_r: float,
                           wind_dir_deg: float,
                           intensity: float = 1.0) -> Polygon:
    """Generate irregular fire shape, considering wind direction effects"""
    # Create base elliptical shape, wind direction determines stretching direction
    main = Point(center).buffer(1.0)
    
    # Stretch based on wind direction
    # Convert wind direction angle to radians
    wind_rad = math.radians(wind_dir_deg)
    
    # Stretch more in wind direction
    xfact = 1.0 + 0.5 * abs(math.sin(wind_rad)) * intensity
    yfact = 1.0 + 0.5 * abs(math.cos(wind_rad)) * intensity
    
    main = scale(main, xfact=xfact, yfact=yfact, origin=center)
    main = rotate(main, wind_dir_deg, origin=center)
    main = scale(main, base_r, base_r, origin=center)

    # Add irregular edges (fire tongues)
    lobes = []
    num_lobes = rng.randint(2, 4)
    for i in range(num_lobes):
        # Fire tongues mainly spread in wind direction
        angle_offset = rng.uniform(-30, 30)  # Wind direction ±30 degrees
        lobe_angle = wind_dir_deg + angle_offset
        
        # Fire tongue distance
        dist = rng.uniform(0.3, 0.8) * base_r * intensity
        
        # Calculate fire tongue position
        lobe_rad = math.radians(lobe_angle)
        dx = dist * math.sin(lobe_rad)
        dy = -dist * math.cos(lobe_rad)  # Negative because y-axis points down
        
        # Create fire tongue
        lobe = Point(center).buffer(base_r * 0.3 * rng.uniform(0.5, 1.0))
        lobe = scale(lobe, xfact=0.6, yfact=1.2, origin='center')
        lobe = rotate(lobe, lobe_angle, origin='center')
        lobe = translate(lobe, xoff=dx, yoff=dy)
        lobes.append(lobe)

    # Merge main body and fire tongues
    return unary_union([main, *lobes])


def compute_exposure(mesh: gpd.GeoDataFrame, fires: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Calculate mesh fire exposure"""
    if len(fires) == 0:
        mesh["exposure_area"] = 0
        mesh["exposure_ratio"] = 0
        mesh["fire_exposed"] = False
        return mesh
    
    fire_union = unary_union(fires.geometry)
    inter_area = mesh.geometry.intersection(fire_union).area
    mesh["exposure_area"]  = inter_area
    mesh["exposure_ratio"] = (inter_area / mesh.geometry.area).clip(upper=1.0)
    mesh["fire_exposed"]   = mesh["exposure_area"] > 0
    return mesh.fillna({"exposure_area": 0, "exposure_ratio": 0, "fire_exposed": False})


def mesh_style(feat):
    """Mesh style function"""
    ratio = feat["properties"].get("exposure_ratio", 0.0)
    if ratio == 0:
        color = "#2686CC"
    elif ratio < 0.25:
        color = "#fef0d9"
    elif ratio < 0.5:
        color = "#fdcc8a"
    elif ratio < 0.75:
        color = "#fc8d59"
    else:
        color = "#d73027"
    return {"color": color, "weight": 1, "fillColor": color, "fillOpacity": 0.3}


def fire_style(_):
    """Fire style function"""
    return {"color": "#ff6600", "weight": 2, "fillColor": "#ff6600", "fillOpacity": 0.4}


def build_arg_parser():
    p = argparse.ArgumentParser(description="Generate animated wildfire spread with wind direction")
    p.add_argument("--n-fires", type=int, default=None,
                   help="Number of fires (default: use existing fires.geojson)")
    p.add_argument("--min-r", type=float, default=DEFAULT_MIN_R,
                   help="Initial min fire radius")
    p.add_argument("--max-r", type=float, default=DEFAULT_MAX_R,
                   help="Initial max fire radius")
    p.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS,
                   help="Number of fire growth steps")
    p.add_argument("--wind", type=float, default=DEFAULT_WIND_DIR,
                   help="Wind direction in degrees (0=North, 90=East, 180=South, 270=West)")
    p.add_argument("--mesh-dir", default="preprocess_data", type=str,
                   help="Directory containing mesh data")
    p.add_argument("--out-dir", default="outputs", type=str,
                   help="Output directory")
    p.add_argument("--seed", type=int, default=None,
                   help="Random seed for reproducibility")
    p.add_argument("--growth-rate", type=float, default=0.15,
                   help="Fire growth rate per timestep (default: 0.15)")
    return p


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    rng = random.Random(args.seed)

    mesh_dir = Path(args.mesh_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load mesh data
    try:
        mesh_base, edges = load_mesh(mesh_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"Please ensure {mesh_dir} contains required files")
        sys.exit(1)

    # Get fire source centers
    centers = get_fire_centers_from_existing(mesh_dir)
    
    # If number of fires specified, adjust center points
    if args.n_fires is not None:
        if args.n_fires > len(centers):
            # Need to add more fire sources
            bounds = mesh_base.total_bounds
            for _ in range(args.n_fires - len(centers)):
                x = rng.uniform(bounds[0], bounds[2])
                y = rng.uniform(bounds[1], bounds[3])
                centers.append((x, y))
        else:
            # Reduce number of fire sources
            centers = centers[:args.n_fires]
    
    # Set initial radius for each fire source
    base_r_list = [rng.uniform(args.min_r, args.max_r) for _ in centers]

    print(f"Fire Spread Simulation")
    print(f"  Timesteps: {args.timesteps}")
    print(f"  Fire sources: {len(centers)}")
    print(f"  Wind direction: {args.wind} degrees (0=North, 90=East)")
    print(f"  Growth rate: {args.growth_rate * 100:.0f}% per timestep")
    print(f"  Output directory: {out_dir}")

    # Generate fire for each timestep
    all_timestep_data = []
    
    for t in range(args.timesteps):
        print(f"\n[Timestep {t}]")
        
        # Generate fire shapes
        fires = []
        for i, (c, base_r) in enumerate(zip(centers, base_r_list)):
            # Fire grows over time
            current_r = base_r * (1 + args.growth_rate * t)
            
            # Fire intensity increases over time
            intensity = 1.0 + 0.1 * t
            
            fire_poly = generate_irregular_fire(
                c, rng, 
                base_r=current_r, 
                wind_dir_deg=args.wind,
                intensity=intensity
            )
            fires.append(fire_poly)
        
        # Create fire GeoDataFrame
        fire_data = []
        for i, fire_poly in enumerate(fires):
            fire_data.append({
                'fire_id': i,
                'timestep': t,
                'area': fire_poly.area,
                'geometry': fire_poly
            })
        
        fire_gdf = gpd.GeoDataFrame(fire_data, crs=TARGET_CRS)
        
        # Calculate mesh exposure
        mesh = compute_exposure(mesh_base.copy(), fire_gdf)
        
        # Save GeoJSON files
        mesh_file = out_dir / f"mesh_with_fire_t{t}.geojson"
        fire_file = out_dir / f"fires_t{t}.geojson"
        
        # Convert to WGS84 for saving
        mesh.to_crs("EPSG:4326").to_file(mesh_file, driver="GeoJSON")
        fire_gdf.to_crs("EPSG:4326").to_file(fire_file, driver="GeoJSON")
        
        print(f"  Saved: mesh_with_fire_t{t}.geojson")
        print(f"  Saved: fires_t{t}.geojson")
        
        # Statistics
        exposed_cells = mesh[mesh["fire_exposed"] == True]
        total_fire_area = fire_gdf.geometry.area.sum()
        
        print(f"  Fire statistics:")
        print(f"    - Exposed cells: {len(exposed_cells)}/{len(mesh)} ({len(exposed_cells)/len(mesh)*100:.1f}%)")
        print(f"    - Total fire area: {total_fire_area:.0f} sq meters")
        print(f"    - Average exposure: {mesh['exposure_ratio'].mean():.2%}")
        
        # Create HTML preview
        if len(fire_gdf) > 0:
            # Calculate center point
            fire_union = unary_union(fire_gdf.geometry)
            cx, cy = fire_union.centroid.coords[0]
            lon_c, lat_c = transformer_to_wgs.transform(cx, cy)
        else:
            bounds = mesh.total_bounds
            cx = (bounds[0] + bounds[2]) / 2
            cy = (bounds[1] + bounds[3]) / 2
            lon_c, lat_c = transformer_to_wgs.transform(cx, cy)
        
        # Create Folium map
        m = folium.Map(location=[lat_c, lon_c], zoom_start=14)
        
        # Add mesh layer
        folium.GeoJson(
            mesh.to_crs("EPSG:4326").to_json(),
            name="Mesh",
            style_function=mesh_style,
            tooltip=folium.GeoJsonTooltip(fields=['exposure_ratio'], aliases=['Exposure:'])
        ).add_to(m)
        
        # Add fire layer
        folium.GeoJson(
            fire_gdf.to_crs("EPSG:4326").to_json(),
            name="Fire",
            style_function=fire_style,
            tooltip=folium.GeoJsonTooltip(fields=['fire_id', 'area'], aliases=['Fire ID:', 'Area (sq m):'])
        ).add_to(m)
        
        # Add wind direction marker (top right corner of map)
        wind_html = f'''
        <div style="position: fixed; top: 80px; right: 10px; z-index: 1000; 
                    background: white; padding: 10px; border-radius: 5px; 
                    box-shadow: 0 0 10px rgba(0,0,0,0.3);">
            <b>Wind: {args.wind} deg</b><br>
            <svg width="40" height="40">
                <circle cx="20" cy="20" r="18" fill="none" stroke="black" stroke-width="2"/>
                <line x1="20" y1="20" x2="20" y2="5" stroke="red" stroke-width="3" 
                      transform="rotate({args.wind}, 20, 20)"/>
                <polygon points="20,5 15,10 25,10" fill="red" 
                         transform="rotate({args.wind}, 20, 20)"/>
            </svg>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(wind_html))
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        # Save HTML
        preview_file = out_dir / f"fire_preview_t{t}.html"
        m.save(str(preview_file))
        print(f"  Saved: fire_preview_t{t}.html")
        
        # Collect data for summary
        all_timestep_data.append({
            't': t,
            'exposed_cells': len(exposed_cells),
            'total_cells': len(mesh),
            'exposure_pct': len(exposed_cells)/len(mesh)*100,
            'fire_area': total_fire_area,
            'avg_exposure': mesh['exposure_ratio'].mean()
        })
    
    # Create summary HTML file
    print("\nCreating summary page...")
    index_html = out_dir / "fire_simulation_index.html"
    
    with open(index_html, 'w', encoding='utf-8') as f:
        f.write("""
<!DOCTYPE html>
<html>
<head>
    <title>Fire Simulation Results</title>
    <meta charset="utf-8">
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #ff6600; }}
        .params {{ background: #f0f0f0; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        .preview-list {{ list-style: none; padding: 0; }}
        .preview-list li {{ 
            display: inline-block; 
            margin: 10px; 
            padding: 10px; 
            background: #fff; 
            border: 1px solid #ddd;
            border-radius: 5px;
        }}
        .preview-list a {{ text-decoration: none; color: #0066cc; font-weight: bold; }}
        .stats-table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        .stats-table th, .stats-table td {{ 
            border: 1px solid #ddd; 
            padding: 8px; 
            text-align: center; 
        }}
        .stats-table th {{ background: #ff6600; color: white; }}
        .stats-table tr:nth-child(even) {{ background: #f9f9f9; }}
    </style>
</head>
<body>
    <h1>Fire Spread Simulation Results</h1>
    
    <div class="params">
        <h2>Simulation Parameters</h2>
        <ul>
            <li><b>Timesteps:</b> {timesteps}</li>
            <li><b>Fire sources:</b> {n_fires}</li>
            <li><b>Wind direction:</b> {wind} degrees (0=North, 90=East, 180=South, 270=West)</li>
            <li><b>Growth rate:</b> {growth_rate}% per timestep</li>
            <li><b>Data source:</b> {mesh_dir}</li>
        </ul>
    </div>
    
    <h2>Statistics</h2>
    <table class="stats-table">
        <tr>
            <th>Timestep</th>
            <th>Exposed Cells</th>
            <th>Exposure Rate</th>
            <th>Fire Area (sq m)</th>
            <th>Avg Exposure</th>
        </tr>
        {stats_rows}
    </table>
    
    <h2>Preview Maps</h2>
    <ul class="preview-list">
        {preview_links}
    </ul>
    
    <h2>Data Files</h2>
    <p>Files generated for each timestep:</p>
    <ul>
        <li><code>fires_t[N].geojson</code> - Fire boundary data</li>
        <li><code>mesh_with_fire_t[N].geojson</code> - Mesh with exposure data</li>
        <li><code>fire_preview_t[N].html</code> - Interactive map preview</li>
    </ul>
</body>
</html>
        """.format(
            timesteps=args.timesteps,
            n_fires=len(centers),
            wind=args.wind,
            growth_rate=int(args.growth_rate * 100),
            mesh_dir=args.mesh_dir,
            stats_rows='\n'.join([
                f"""
                <tr>
                    <td>T{d['t']}</td>
                    <td>{d['exposed_cells']}/{d['total_cells']}</td>
                    <td>{d['exposure_pct']:.1f}%</td>
                    <td>{d['fire_area']:.0f}</td>
                    <td>{d['avg_exposure']:.1%}</td>
                </tr>
                """ for d in all_timestep_data
            ]),
            preview_links='\n'.join([
                f'<li><a href="fire_preview_t{t}.html" target="_blank">Timestep {t}</a></li>'
                for t in range(args.timesteps)
            ])
        ))
    
    print(f"\nFire simulation complete!")
    print(f"All files saved in: {out_dir}/")
    print(f"Open {index_html} to view summary")


if __name__ == "__main__":
    main()