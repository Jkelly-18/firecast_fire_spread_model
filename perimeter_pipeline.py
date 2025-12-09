"""
This file converts FIRMS satellite detections into predicted fire perimeters
using density-based clustering and polygon construction methods.

The pipeline takes preprocessed FIRMS detections and transforms them into
polygon shapes representing estimated fire boundaries. It supports two modes:
'tune' for final perimeter generation (used in parameter optimization) and
'full' for generating perimeters at each observation window (used for
progression tracking and dashboard visualization).

Steps:
1. density_filter(points_gdf, percentile, radius)
   - Removes low-density outlier points using BallTree neighbor counting.

2. merge_clusters(points_gdf, clusters, merge_dist)
   - Merges nearby clusters within a distance threshold to unify fragmented fire fronts.

3. make_polygon(survivors, polygon, density_pct, eps, min_samples, ...)
   - Applies DBSCAN clustering, merges clusters, filters by density, and constructs
     polygons using either concave hull or alpha shape methods.

4. process_fire(fire_data, mode, polygon, ...)
   - Main entry point. In 'tune' mode, returns final perimeter for evaluation.
     In 'full' mode, returns perimeters for each observation window.

Output:
Returns dictionary (tune mode) or list of dictionaries (full mode) containing:
    geometry, n_points, timestamp
"""

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import BallTree
from shapely.geometry import MultiPoint
from shapely import concave_hull
from shapely.ops import unary_union
import alphashape

def density_filter(points_gdf, percentile, radius):
    """Remove lowest density points"""

    if percentile is None:
        return points_gdf
    coords = np.array([(g.x, g.y) for g in points_gdf.geometry])
    tree = BallTree(coords)
    counts = tree.query_radius(coords, r=radius, count_only=True)
    mask = counts > np.percentile(counts, percentile)
    return points_gdf[mask]

def merge_clusters(points_gdf, clusters, merge_dist):
    """Merge clusters if closest points within threshold"""

    coords = np.array([(g.x, g.y) for g in points_gdf.geometry])
    cluster_ids = [c for c in set(clusters) if c >= 0]
    
    if len(cluster_ids) <= 1:
        return clusters
    
    cluster_coords = {c: coords[clusters == c] for c in cluster_ids}
    final_group = {c: c for c in cluster_ids}
    
    for i, c1 in enumerate(cluster_ids):
        for c2 in cluster_ids[i+1:]:
            tree = BallTree(cluster_coords[c2])
            min_dist = tree.query(cluster_coords[c1], k=1)[0].min()
            
            if min_dist <= merge_dist:
                group_to_replace = final_group[c2]
                group_to_keep = final_group[c1]
                for c in final_group:
                    if final_group[c] == group_to_replace:
                        final_group[c] = group_to_keep
    
    new_clusters = clusters
    for old_id, new_id in final_group.items():
        if old_id != new_id:
            new_clusters[clusters == old_id] = new_id
    
    return new_clusters


def make_polygon(survivors, polygon, density_pct, eps, min_samples, merge_dist, 
                 concave_ratio, alpha_value, density_radius):
    """Process to create perimeter: DBSCAN, merge clusters, density filter, create polygon"""

    if len(survivors) < 3:
        return None
    
    coords = np.array([(g.x, g.y) for g in survivors.geometry])
    clusters = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(coords)
    clusters = merge_clusters(survivors, clusters, merge_dist)
    
    polygons = []
    for cluster_id in set(clusters):
        if cluster_id < 0:
            continue

        cluster = survivors[clusters == cluster_id]
        
        if density_pct:
            cluster = density_filter(cluster, density_pct, density_radius)
        
        if len(cluster) < 3:
            continue

        cluster_coords = [(g.x, g.y) for g in cluster.geometry]
        
        if polygon == 'alpha':
            poly = alphashape.alphashape(cluster_coords, alpha_value)
        elif polygon == 'concave':
            poly = concave_hull(MultiPoint(cluster_coords), ratio=concave_ratio)     
        
        polygons.append(poly)

    return unary_union(polygons)

def process_fire(fire_data, mode='tune', polygon='concave', density_pct=2,
                 eps=750, min_samples=3, merge_dist=2000, 
                 concave_ratio=0.3, alpha_value=0.001, density_radius=750,
                 fire_id=None, output_dir=None):
    """Processes fire through all windows."""

    # Tune: only create final polygon for evaluation to cal fire final perimeter
    if mode == 'tune':
        poly = make_polygon(fire_data, polygon, density_pct, eps, min_samples,
                        merge_dist, concave_ratio, alpha_value, density_radius)
        return {
            'geometry': poly,
            'n_points': len(fire_data),
            'timestamp': fire_data['acq_datetime'].max()
        }

    # Full mode: create polygon for this window
    if mode == 'full':
        window_ids = sorted(fire_data['window_id'].unique())
        window_results = []
        
        for wid in window_ids:
            # Cumulative points up to this window
            cumulative = fire_data[fire_data['window_id'] <= wid]
            
            poly = make_polygon(cumulative, polygon, density_pct, eps, min_samples,
                               merge_dist, concave_ratio, alpha_value, density_radius)
            
            window_data = fire_data[fire_data['window_id'] == wid]
            window_results.append({
                'timestamp': window_data['acq_datetime'].max(),
                'geometry': poly,
                'n_points': len(cumulative)
            })
        
        return window_results
    