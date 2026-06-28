"""
Spatial Sampling Strategies for Fine-tuning Experiments
Implements three sampling strategies and computes spatial metrics.
"""

import os
import json
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon, MultiPoint
from sklearn.metrics.pairwise import haversine_distances
from typing import List, Dict, Tuple, Optional
import warnings
import matplotlib.pyplot as plt
import contextily as ctx
from matplotlib.patches import Rectangle
warnings.filterwarnings('ignore')


def load_mapping_json(metadata_dir: str) -> List[Dict]:
    """Load consolidated mapping.json file"""
    mapping_path = os.path.join(metadata_dir, 'mapping.json')
    with open(mapping_path, 'r') as f:
        return json.load(f)


def load_tile_metadata(city: str, resolution: str, metadata_dir: str) -> pd.DataFrame:
    """
    Load tile metadata CSV for given city and resolution
    
    Args:
        city: City name (chicago, miami, phoenix)
        resolution: 'highres' or 'midres'
        metadata_dir: Path to metadata directory
    
    Returns:
        DataFrame with tile metadata
    """
    if resolution == 'highres':
        csv_path = os.path.join(metadata_dir, f'{city}30.csv')
    else:  # midres
        csv_path = os.path.join(metadata_dir, f'{city}.csv')
    
    # Read with latin1 encoding as specified
    df = pd.read_csv(csv_path, encoding='latin1')
    return df

def load_original_split(city: str, resolution: str, base_data_root: str) -> Tuple[List[str], List[str]]:
    """
    Load the original train/val split from data directories
    
    Args:
        city: City name
        resolution: 'highres' or 'midres'
        base_data_root: Base data directory
    
    Returns:
        (train_filenames, val_filenames)
    """
    data_root = os.path.join(base_data_root, city, resolution)
    
    train_dir = os.path.join(data_root, 'train', 'images')
    val_dir = os.path.join(data_root, 'val', 'images')
    
    train_files = sorted([f for f in os.listdir(train_dir) if f.endswith('.png')])
    val_files = sorted([f for f in os.listdir(val_dir) if f.endswith('.png')])
    
    return train_files, val_files

def create_tile_geodataframe(metadata_df: pd.DataFrame) -> gpd.GeoDataFrame:
    """
    Convert metadata DataFrame to GeoDataFrame using corner coordinates
    
    Args:
        metadata_df: DataFrame with tile corner coordinates
    
    Returns:
        GeoDataFrame with tile polygons
    """
    geometries = []
    
    for idx, row in metadata_df.iterrows():
        # Get corner coordinates (decimal degrees)
        nw_lat, nw_lon = row['NW Corner Lat dec'], row['NW Corner Long dec']
        ne_lat, ne_lon = row['NE Corner Lat dec'], row['NE Corner Long dec']
        se_lat, se_lon = row['SE Corner Lat dec'], row['SE Corner Long dec']
        sw_lat, sw_lon = row['SW Corner Lat dec'], row['SW Corner Long dec']
        
        # Create polygon from corner coordinates (counter-clockwise)
        polygon = Polygon([
            (nw_lon, nw_lat),  # NW
            (sw_lon, sw_lat),  # SW
            (se_lon, se_lat),  # SE
            (ne_lon, ne_lat),  # NE
            (nw_lon, nw_lat)   # Close the polygon
        ])
        
        geometries.append(polygon)
    
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(metadata_df, geometry=geometries, crs='EPSG:4326')
    return gdf

def create_tile_based_boundary(tile_gdf: gpd.GeoDataFrame, city_boundary: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Create boundary from union of all tiles that intersect city boundary
    
    Args:
        tile_gdf: GeoDataFrame of NAIP tiles
        city_boundary: GeoDataFrame of city boundary
    
    Returns:
        GeoDataFrame with unioned tile boundary
    """
    # Find tiles that intersect city boundary
    intersecting_tiles = tile_gdf[tile_gdf.intersects(city_boundary.union_all())]
    
    # Create union of all intersecting tiles
    tile_union = intersecting_tiles.union_all()
    
    # Return as GeoDataFrame
    return gpd.GeoDataFrame(geometry=[tile_union], crs=tile_gdf.crs)

def load_city_boundary(city: str, metadata_dir: str) -> gpd.GeoDataFrame:
    """Load city boundary shapefile"""
    shp_path = os.path.join(metadata_dir, f'{city}.geojson')
    return gpd.read_file(shp_path)


def filter_patches(mapping_data: List[Dict], city: str, resolution: str, 
                   base_data_root: str) -> List[Dict]:
    """
    Filter patches by city and resolution, checking which ones exist in train OR val directories
    
    Args:
        mapping_data: List of patch metadata dictionaries
        city: Target city
        resolution: 'highres' or 'midres'
        base_data_root: Base data directory
    
    Returns:
        Filtered list of patch metadata with 'available_in_split' field added
    """
    # First filter by city and resolution
    city_patches = [p for p in mapping_data 
                    if p['city'] == city and p['resolution'] == resolution]
    
    # Check which split each patch exists in
    data_root = os.path.join(base_data_root, city, resolution)
    train_dir = os.path.join(data_root, 'train', 'images')
    val_dir = os.path.join(data_root, 'val', 'images')
    
    filtered = []
    for patch in city_patches:
        filename = patch['original_filename']
        
        # Check if file exists in train or val
        if os.path.exists(os.path.join(train_dir, filename)):
            patch['available_in_split'] = 'train'
            filtered.append(patch)
        elif os.path.exists(os.path.join(val_dir, filename)):
            patch['available_in_split'] = 'val'
            filtered.append(patch)
        # If not in either, skip it
    
    return filtered


# =============================================================================
# STRATEGY 1: RANDOM SAMPLING
# =============================================================================

def select_patches_random(patches: List[Dict], n_samples: int, 
                          split_ratio: Tuple[float, float] = (0.75, 0.25),
                          random_seed: int = 42) -> Tuple[List[str], List[str]]:
    """
    Random sampling strategy
    
    Args:
        patches: List of available patch metadata
        n_samples: Total number of samples to select
        split_ratio: (train_ratio, val_ratio) tuple
        random_seed: Random seed for reproducibility
    
    Returns:
        (train_filenames, val_filenames)
    """
    np.random.seed(random_seed)
    
    # Shuffle patches
    shuffled = patches.copy()
    np.random.shuffle(shuffled)
    
    # Split into train/val
    n_train = int(n_samples * split_ratio[0])
    n_val = n_samples - n_train
    
    # Select filenames
    train_files = [p['original_filename'] for p in shuffled[:n_train]]
    val_files = [p['original_filename'] for p in shuffled[n_train:n_train + n_val]]
    
    return train_files, val_files


# =============================================================================
# STRATEGY 2: CLUSTERED SAMPLING
# =============================================================================

def find_adjacent_tiles(tile_gdf: gpd.GeoDataFrame, seed_tile_id: str) -> List[str]:
    """
    Find all tiles adjacent to (touching) the seed tile
    
    Args:
        tile_gdf: GeoDataFrame of tiles
        seed_tile_id: NAIP Entity ID of seed tile
    
    Returns:
        List of adjacent tile IDs (including seed tile)
    """
    # Get seed tile geometry
    seed_tile = tile_gdf[tile_gdf['NAIP Entity ID'].str.lower() == seed_tile_id.lower()].iloc[0]
    seed_geom = seed_tile.geometry
    
    # Find touching tiles
    adjacent_ids = [seed_tile_id.lower()]  # Include seed tile
    
    for idx, row in tile_gdf.iterrows():
        if row['NAIP Entity ID'].lower() != seed_tile_id.lower():
            if seed_geom.touches(row.geometry) or seed_geom.intersects(row.geometry):
                adjacent_ids.append(row['NAIP Entity ID'].lower())
    
    return adjacent_ids


def select_patches_clustered(patches: List[Dict], n_samples: int,
                             city: str, resolution: str, metadata_dir: str,
                             split_ratio: Tuple[float, float] = (0.75, 0.25),
                             random_seed: int = 42) -> Tuple[List[str], List[str]]:
    """
    Clustered sampling strategy - sample all patches from adjacent tiles
    
    Args:
        patches: List of available patch metadata
        n_samples: Total number of samples to select
        city: City name
        resolution: 'highres' or 'midres'
        metadata_dir: Path to metadata directory
        split_ratio: (train_ratio, val_ratio) tuple
        random_seed: Random seed
    
    Returns:
        (train_filenames, val_filenames)
    """
    np.random.seed(random_seed)
    
    # Load tile metadata
    tile_df = load_tile_metadata(city, resolution, metadata_dir)
    tile_gdf = create_tile_geodataframe(tile_df)
    
    # Get unique tiles that have patches
    tiles_with_patches = list(set([p['tile_name'].lower() for p in patches]))
    
    # Randomly select a seed tile
    seed_tile = np.random.choice(tiles_with_patches)
    
    # Find adjacent tiles
    cluster_tiles = find_adjacent_tiles(tile_gdf, seed_tile)
    
    # Get patches from cluster
    cluster_patches = [p for p in patches if p['tile_name'] in cluster_tiles]
    
    # Expand cluster if we don't have enough patches
    max_iterations = 10
    iteration = 0
    while len(cluster_patches) < n_samples and iteration < max_iterations:
        # Add more adjacent tiles
        new_cluster = []
        for tile_id in cluster_tiles:
            new_cluster.extend(find_adjacent_tiles(tile_gdf, tile_id))
        cluster_tiles = list(set(new_cluster))
        
        cluster_patches = [p for p in patches if p['tile_name'].lower() in cluster_tiles]
        iteration += 1
    
    # Randomly sample from cluster
    np.random.shuffle(cluster_patches)
    
    # Split into train/val
    n_train = int(n_samples * split_ratio[0])
    n_val = n_samples - n_train
    
    train_files = [p['original_filename'] for p in cluster_patches[:n_train]]
    val_files = [p['original_filename'] for p in cluster_patches[n_train:n_train + n_val]]
    
    return train_files, val_files


# =============================================================================
# STRATEGY 3: DISPERSED SAMPLING
# =============================================================================

def create_spatial_grid(city_boundary: gpd.GeoDataFrame, grid_size_m: float,
                       crs_epsg: str = 'EPSG:3857') -> gpd.GeoDataFrame:
    """
    Create a spatial grid over city boundary
    
    Args:
        city_boundary: GeoDataFrame of city boundary
        grid_size_m: Grid cell size in meters
        crs_epsg: Equal-area CRS for grid creation (default: Web Mercator)
    
    Returns:
        GeoDataFrame of grid cells
    """
    # Project to equal-area CRS
    boundary_proj = city_boundary.to_crs(crs_epsg)
    
    # Get bounds
    minx, miny, maxx, maxy = boundary_proj.total_bounds
    
    # Create grid
    grid_cells = []
    grid_ids = []
    
    x = minx
    grid_id = 0
    while x < maxx:
        y = miny
        while y < maxy:
            # Create cell polygon
            cell = Polygon([
                (x, y),
                (x + grid_size_m, y),
                (x + grid_size_m, y + grid_size_m),
                (x, y + grid_size_m),
                (x, y)
            ])
            
            # Check if cell intersects city boundary
            if boundary_proj.intersects(cell).any():
                grid_cells.append(cell)
                grid_ids.append(grid_id)
                grid_id += 1
            
            y += grid_size_m
        x += grid_size_m
    
    # Create GeoDataFrame
    grid_gdf = gpd.GeoDataFrame({'grid_id': grid_ids}, 
                                 geometry=grid_cells, 
                                 crs=crs_epsg)
    
    # Project back to WGS84
    grid_gdf = grid_gdf.to_crs('EPSG:4326')
    
    return grid_gdf


def select_patches_dispersed(patches: List[Dict], n_samples: int,
                             city: str, resolution: str, metadata_dir: str,
                             split_ratio: Tuple[float, float] = (0.75, 0.25),
                             random_seed: int = 42,
                             initial_grid_size_m: Optional[float] = None) -> Tuple[List[str], List[str]]:
    """
    Dispersed sampling strategy - maximize geographic spread with grid-based sampling
    
    Args:
        patches: List of available patch metadata
        n_samples: Total number of samples to select
        city: City name
        metadata_dir: Path to metadata directory
        split_ratio: (train_ratio, val_ratio) tuple
        random_seed: Random seed
        initial_grid_size_m: Initial grid size in meters (auto-computed if None)
    
    Returns:
        (train_filenames, val_filenames)
    """
    np.random.seed(random_seed)
    
    # Load city boundary
    # Load city boundary
    city_boundary = load_city_boundary(city, metadata_dir)
    
    # Load tile metadata and create tile-based boundary
    tile_df = load_tile_metadata(city, resolution, metadata_dir)
    tile_gdf = create_tile_geodataframe(tile_df)
    
    # Use tile-based boundary for grid creation (matches random/clustered sampling space)
    boundary_for_grid = create_tile_based_boundary(tile_gdf, city_boundary)
    
    # Project to equal-area CRS to compute area
    boundary_proj = boundary_for_grid.to_crs('EPSG:3857')
    city_area_km2 = boundary_proj.area.sum() / 1e6  # in km²
    
    # Auto-compute initial grid size if not provided
    if initial_grid_size_m is None:
        # Aim for ~2x more cells than samples needed (to have flexibility)
        target_cells = n_samples
        # Grid size = sqrt(area / target_cells)
        initial_grid_size_m = np.sqrt(city_area_km2 * 1e6 / target_cells)
        # Round to nearest 100m
        initial_grid_size_m = max(100, int(initial_grid_size_m / 100) * 100)
    
    grid_size_m = initial_grid_size_m
    
    # Try creating grid and assigning patches
    max_iterations = 10
    iteration = 0
    
    while iteration < max_iterations:
        # Create grid
        grid_gdf = create_spatial_grid(boundary_for_grid, grid_size_m)
        
        # Assign each patch to a grid cell
        patch_points = []
        for p in patches:
            point = Point(p['center_lon'], p['center_lat'])
            patch_points.append(point)
        
        patches_gdf = gpd.GeoDataFrame(patches, 
                                       geometry=patch_points,
                                       crs='EPSG:4326')
        
        # Spatial join to find which grid cell each patch belongs to
        joined = gpd.sjoin(patches_gdf, grid_gdf, how='left', predicate='within')
        
        # Group by grid_id
        patches_by_grid = {}
        for idx, row in joined.iterrows():
            grid_id = row.get('grid_id', None)
            if pd.notna(grid_id):
                if grid_id not in patches_by_grid:
                    patches_by_grid[grid_id] = []
                patches_by_grid[grid_id].append(patches[idx])
        
        # Check if we have enough non-empty cells
        n_nonempty_cells = len(patches_by_grid)
        
        if n_nonempty_cells >= n_samples:
            # We have enough cells!
            break
        else:
            # Reduce grid size by half
            grid_size_m = grid_size_m / 2
            iteration += 1
    
    if len(patches_by_grid) < n_samples:
        print(f"Warning: Only {len(patches_by_grid)} non-empty grid cells available for {n_samples} samples.")
        print(f"Final grid size: {grid_size_m:.1f}m")
    
    # Sample one patch per grid cell
    grid_ids = list(patches_by_grid.keys())
    np.random.shuffle(grid_ids)
    
    selected_patches = []
    for grid_id in grid_ids[:n_samples]:
        # Randomly select one patch from this grid cell
        cell_patches = patches_by_grid[grid_id]
        selected_patch = cell_patches[np.random.randint(len(cell_patches))]
        selected_patches.append(selected_patch)
    
    # Split into train/val
    n_train = int(n_samples * split_ratio[0])
    n_val = n_samples - n_train
    
    train_files = [p['original_filename'] for p in selected_patches[:n_train]]
    val_files = [p['original_filename'] for p in selected_patches[n_train:n_train + n_val]]
    
    return train_files, val_files


# =============================================================================
# SPATIAL METRICS COMPUTATION
# =============================================================================

def compute_spatial_metrics(selected_patches: List[Dict], 
                            metadata_dir: str) -> Dict[str, float]:
    """
    Compute 5 spatial metrics for selected patches:
    1. Mean Pairwise Distance (MPD) in km
    2. Mean Minimum Distance (nearest neighbor) in km
    3. Standard Distance (spatial dispersion)
    4. Convex Hull Area in km²
    5. Number of Unique Tiles
    
    Args:
        selected_patches: List of selected patch metadata dicts
        metadata_dir: Path to metadata directory (for projections)
    
    Returns:
        Dictionary of spatial metrics
    """
    if len(selected_patches) == 0:
        return {
            'mean_pairwise_distance_km': 0.0,
            'mean_min_distance_km': 0.0,
            'median_min_distance_km': 0.0,
            'standard_distance': 0.0,
            'convex_hull_area_km2': 0.0,
            'unique_tiles': 0
        }
    
    # Extract coordinates
    coordinates = np.array([[p['center_lon'], p['center_lat']] for p in selected_patches])
    
    # 1. Mean Pairwise Distance (haversine)
    coords_rad = np.radians(coordinates)
    distances_km = haversine_distances(coords_rad) * 6371  # Earth radius in km
    
    # Get upper triangle (excluding diagonal)
    upper_triangle_indices = np.triu_indices_from(distances_km, k=1)
    pairwise_distances = distances_km[upper_triangle_indices]
    
    mpd = np.mean(pairwise_distances) if len(pairwise_distances) > 0 else 0.0
    
    # 2. Minimum Distance (nearest neighbor)
    np.fill_diagonal(distances_km, np.inf)  # Exclude self
    min_distances = distances_km.min(axis=1)
    mean_min_dist = np.mean(min_distances)
    median_min_dist = np.median(min_distances)
    
    # 3. Standard Distance
    lon_mean, lat_mean = coordinates.mean(axis=0)
    sd = np.sqrt(((coordinates[:, 0] - lon_mean)**2 + 
                  (coordinates[:, 1] - lat_mean)**2).sum() / len(coordinates))
    
    # 4. Convex Hull Area
    if len(coordinates) >= 3:
        points = MultiPoint([(lon, lat) for lon, lat in coordinates])
        hull = points.convex_hull
        
        # Project to equal-area CRS for area calculation
        gdf = gpd.GeoDataFrame(geometry=[hull], crs='EPSG:4326')
        gdf_projected = gdf.to_crs('EPSG:3857')  # Web Mercator
        hull_area_km2 = gdf_projected.geometry.area.values[0] / 1e6
    else:
        hull_area_km2 = 0.0
    
    # 5. Unique Tiles
    unique_tiles = len(set([p['tile_name'] for p in selected_patches]))
    
    return {
        'mean_pairwise_distance_km': float(mpd),
        'mean_min_distance_km': float(mean_min_dist),
        'median_min_distance_km': float(median_min_dist),
        'standard_distance': float(sd),
        'convex_hull_area_km2': float(hull_area_km2),
        'unique_tiles': int(unique_tiles),
        'n_samples': len(selected_patches)
    }

def visualize_selected_patches(
    selected_patches: List[Dict],
    train_filenames: List[str],
    val_filenames: List[str],
    city: str,
    strategy: str,
    n_samples: int,
    metadata_dir: str,
    output_path: str
):
    """
    Visualize selected patches on city map with background
    
    Args:
        selected_patches: List of selected patch metadata
        train_filenames: List of training filenames
        val_filenames: List of validation filenames
        city: City name
        strategy: Strategy name (for title)
        n_samples: Number of samples (for title)
        metadata_dir: Path to metadata directory
        output_path: Path to save figure
    """
    # Load city boundary
    city_boundary = load_city_boundary(city, metadata_dir)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot city boundary
    city_boundary.plot(ax=ax, facecolor='none', edgecolor='black', 
                       linewidth=2, alpha=0.7, zorder=2)
    
    # Separate train and val patches
    train_set = set(train_filenames)
    train_patches = [p for p in selected_patches if p['original_filename'] in train_set]
    val_patches = [p for p in selected_patches if p['original_filename'] not in train_set]
    
    # Plot training patches
    if train_patches:
        train_lons = [p['center_lon'] for p in train_patches]
        train_lats = [p['center_lat'] for p in train_patches]
        ax.scatter(train_lons, train_lats, c='red', s=50, alpha=0.7, 
                  label=f'Train (n={len(train_patches)})', zorder=3, 
                  edgecolors='darkred', linewidths=1)
    
    # Plot validation patches
    if val_patches:
        val_lons = [p['center_lon'] for p in val_patches]
        val_lats = [p['center_lat'] for p in val_patches]
        ax.scatter(val_lons, val_lats, c='blue', s=50, alpha=0.7,
                  label=f'Val (n={len(val_patches)})', zorder=3,
                  edgecolors='darkblue', linewidths=1)
    
    # Add basemap
    try:
        ctx.add_basemap(ax, crs=city_boundary.crs.to_string(), 
                       source=ctx.providers.OpenStreetMap.Mapnik,
                       alpha=0.5, zorder=1)
    except Exception as e:
        print(f"Warning: Could not add basemap: {e}")
    
    # Format plot
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title(f'{city.title()} - {strategy.title()} Strategy (N={n_samples})',
                fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved: {output_path}")

# =============================================================================
# MAIN SELECTION FUNCTION
# =============================================================================

def select_patches_by_strategy(
    city: str,
    resolution: str,
    n_samples: int,
    strategy: str = 'random',
    random_seed: int = 42,
    metadata_dir: str = os.path.join(os.environ["PROJECT_ROOT"], 'data', 'Final_data_test', 'metadata') + os.sep,
    base_data_root: str = os.path.join(os.environ["PROJECT_ROOT"], 'data', 'Final_data_test') + os.sep,
    split_ratio: Tuple[float, float] = (0.75, 0.25),
    save_visualization: Optional[str] = None
) -> Dict:
    """
    Main function to select patches using specified spatial strategy
    
    Args:
        city: Target city (chicago, miami, phoenix)
        resolution: 'highres' or 'midres'
        n_samples: Total number of samples to select
        strategy: 'random', 'clustered', or 'dispersed'
        random_seed: Random seed for reproducibility
        metadata_dir: Path to metadata directory
        split_ratio: (train_ratio, val_ratio) tuple
    
    Returns:
        Dictionary with:
            - train_filenames: List of training image filenames
            - val_filenames: List of validation image filenames
            - spatial_metrics: Dictionary of spatial metrics
            - strategy: Strategy used
            - n_samples: Number of samples
    """
    # Load mapping data
    mapping_data = load_mapping_json(metadata_dir)

    # Special case: N=600 means use original split
    if n_samples == 600:
        train_files, val_files = load_original_split(city, resolution, base_data_root)
        
        # Get patch metadata for metrics
        all_patches = filter_patches(mapping_data, city, resolution, base_data_root)
        all_selected_files = train_files + val_files
        selected_patches = [p for p in all_patches if p['original_filename'] in all_selected_files]
        
        spatial_metrics = compute_spatial_metrics(selected_patches, metadata_dir)
        spatial_metrics['note'] = 'Original split (N=600)'
        
        if save_visualization:
            visualize_selected_patches(
                selected_patches=selected_patches,
                train_filenames=train_files,
                val_filenames=val_files,
                city=city,
                strategy='original',
                n_samples=n_samples,
                metadata_dir=metadata_dir,
                output_path=save_visualization
            )
        
        return {
            'train_filenames': train_files,
            'val_filenames': val_files,
            'spatial_metrics': spatial_metrics,
            'strategy': 'original',
            'n_samples': n_samples,
            'random_seed': random_seed
        }
    
    # Filter patches for target city and resolution
    all_patches = filter_patches(mapping_data, city, resolution, base_data_root)

    if len(all_patches) == 0:
        raise ValueError(f"No patches found for {city} {resolution}")
    
    # Select patches based on strategy
    if strategy == 'random':
        train_files, val_files = select_patches_random(
            all_patches, n_samples, split_ratio, random_seed
        )
    elif strategy == 'clustered':
        train_files, val_files = select_patches_clustered(
            all_patches, n_samples, city, resolution, metadata_dir, 
            split_ratio, random_seed
        )
    elif strategy == 'dispersed':
        train_files, val_files = select_patches_dispersed(
            all_patches, n_samples, city, resolution, metadata_dir,
            split_ratio, random_seed
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Must be 'random', 'clustered', or 'dispersed'")
    
    # Get selected patch metadata for metrics
    all_selected_files = train_files + val_files
    selected_patches = [p for p in all_patches if p['original_filename'] in all_selected_files]
    
    # Compute spatial metrics
    spatial_metrics = compute_spatial_metrics(selected_patches, metadata_dir)

    if save_visualization:
        visualize_selected_patches(
            selected_patches=selected_patches,
            train_filenames=train_files,
            val_filenames=val_files,
            city=city,
            strategy=strategy,
            n_samples=n_samples,
            metadata_dir=metadata_dir,
            output_path=save_visualization
        )
    
    return {
        'train_filenames': train_files,
        'val_filenames': val_files,
        'spatial_metrics': spatial_metrics,
        'strategy': strategy,
        'n_samples': n_samples,
        'random_seed': random_seed
    }


if __name__ == '__main__':
    # Test the implementation
    print("Testing spatial sampling strategies...")
    
    result = select_patches_by_strategy(
        city='chicago',
        resolution='midres',
        n_samples=100,
        strategy='random',
        random_seed=42
    )
    
    print(f"\nStrategy: {result['strategy']}")
    print(f"N samples: {result['n_samples']}")
    print(f"Train files: {len(result['train_filenames'])}")
    print(f"Val files: {len(result['val_filenames'])}")
    print(f"\nSpatial Metrics:")
    for key, value in result['spatial_metrics'].items():
        print(f"  {key}: {value}")