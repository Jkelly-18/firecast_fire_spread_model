"""
This script exports fire perimeter data and evaluation metrics to JSON files
for use by the interactive web dashboard.

It loads the generated window perimeters, calculates evaluation metrics by
comparing final predicted perimeters to CAL FIRE boundaries, and outputs
GeoJSON files for map visualization along with metadata for the dashboard UI.

Steps:
1. Load window perimeters from parquet and convert to required coordinate systems
   (EPSG:3310 for area calculations, EPSG:4326 for web mapping).

2. For each fire:
   - Calculate evaluation metrics (IoU, precision, recall, F1.25) by comparing
     the final predicted perimeter to the CAL FIRE actual perimeter.
   - Build a GeoJSON FeatureCollection containing all window perimeters and
     the actual CAL FIRE perimeter.
   - Extract metadata including fire name, year, centroid, areas, and metrics.

3. Save individual perimeter GeoJSON files for each fire.

4. Save combined fire metadata JSON for dashboard fire selection and display.

Output:
Saves to dashboard/dashboard_data/:
    - perimeters/{fire_id}.json: GeoJSON with window perimeters and CAL FIRE actual
    - fire_data.json: Metadata for all fires (name, year, metrics, window stats)
"""

import json
import geopandas as gpd
import pandas as pd

# Load predicted perimeters
perimeters = gpd.read_parquet("data/perimeters/window_perimeters.parquet")

# Calculate area
perimeters_m = perimeters.to_crs(epsg=3310)
perimeters_m['area_km2'] = perimeters_m.geometry.area / 1e6

perimeters_wgs = perimeters.to_crs(epsg=4326)
perimeters_wgs['area_km2'] = perimeters_m['area_km2'].values  # Copy over calculated area

# Load calfire for metadata and actual perimeters
calfire = gpd.read_file("data/calfire_data/California_Fire_Perimeters_(all).shp")
calfire_wgs = calfire.to_crs(epsg=4326)
calfire_m = calfire.to_crs(epsg=3310)
calfire_m["fire_id"] = calfire_m["FIRE_NAME"] + "_" + calfire_m["INC_NUM"]
calfire_wgs["fire_id"] = calfire_wgs["FIRE_NAME"] + "_" + calfire_wgs["INC_NUM"]

fire_ids = perimeters["fire_id"].unique()
fires_data = []

for fire_id in fire_ids:
    # Updated: use window_id instead of window_idx
    fire_perims_wgs = perimeters_wgs[perimeters_wgs["fire_id"] == fire_id].sort_values("timestamp")
    fire_perims_m = perimeters_m[perimeters_m["fire_id"] == fire_id].sort_values("timestamp")
    
    calfire_row_m = calfire_m[calfire_m["fire_id"] == fire_id]
    calfire_row_wgs = calfire_wgs[calfire_wgs["fire_id"] == fire_id]
    if len(calfire_row_m) == 0:
        continue
    calfire_row_m = calfire_row_m.iloc[0]
    calfire_row_wgs = calfire_row_wgs.iloc[0]
    
    final_perim_m = fire_perims_m.iloc[-1].geometry
    final_perim_wgs = fire_perims_wgs.iloc[-1].geometry
    if final_perim_m is None:
        continue
    
    centroid = final_perim_wgs.centroid
    
    # Calculate IoU, recall, precision (in meters CRS)
    actual_geom_m = calfire_row_m.geometry
    intersection = final_perim_m.intersection(actual_geom_m).area
    union = final_perim_m.union(actual_geom_m).area
    
    iou = intersection / union if union > 0 else 0
    recall = intersection / actual_geom_m.area if actual_geom_m.area > 0 else 0
    precision = intersection / final_perim_m.area if final_perim_m.area > 0 else 0

    beta = 1.25
    f125 = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall) if (precision + recall) > 0 else 0
    
    # Areas
    final_area_km2 = fire_perims_m.iloc[-1]["area_km2"]
    actual_area_km2 = calfire_row_m["GIS_ACRES"] * 0.00404686 if pd.notna(calfire_row_m["GIS_ACRES"]) else 0
    
    # Build windows list (updated: n_points instead of cumulative_points)
    windows = []
    for idx, (_, row) in enumerate(fire_perims_wgs.iterrows()):
        if row.geometry is None:
            continue
        windows.append({
            "area_km2": round(fire_perims_m.iloc[idx]["area_km2"], 2),
            "timestamp": row["timestamp"].isoformat() if pd.notna(row["timestamp"]) else None,
            "cumulative_points": int(row["n_points"]) if pd.notna(row.get("n_points")) else 0,
        })
    
    # Build GeoJSON for perimeters file (updated: use idx for window_idx)
    features = []
    for idx, (_, row) in enumerate(fire_perims_wgs.iterrows()):
        if row.geometry is None or row.geometry.is_empty:
            continue
        if fire_perims_m.iloc[idx]["area_km2"] <= 0:
            continue
        features.append({
            "type": "Feature",
            "properties": {
                "window_idx": idx,
                "timestamp": row["timestamp"].isoformat() if pd.notna(row["timestamp"]) else None,
                "area_km2": round(fire_perims_m.iloc[idx]["area_km2"], 2),
            },
            "geometry": row.geometry.__geo_interface__
        })
    
    # Add actual calfire perimeter
    features.append({
        "type": "Feature",
        "properties": {"type": "calfire_actual", "area_km2": round(actual_area_km2, 2)},
        "geometry": calfire_row_wgs.geometry.__geo_interface__
    })
    
    with open(f"dashboard/dashboard_data/perimeters/{fire_id}.json", "w") as f:
        json.dump({"type": "FeatureCollection", "features": features}, f)
    
    # Fire metadata
    fires_data.append({
        "fire_id": fire_id,
        "name": calfire_row_m["FIRE_NAME"],
        "year": int(calfire_row_m["YEAR_"]),
        "centroid": [centroid.y, centroid.x],
        "final_area_km2": round(final_area_km2, 2),
        "actual_area_km2": round(actual_area_km2, 2),
        "f125": round(f125, 3),
        "iou": round(iou, 3),
        "windows": windows
    })

with open("dashboard/dashboard_data/fire_data.json", "w") as f:
    json.dump(fires_data, f, indent=2)

print(f"{len(fires_data)} fires exported")
