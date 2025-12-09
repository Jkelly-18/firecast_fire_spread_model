"""
This file loads and preprocesses FIRMS satellite detections and CAL FIRE
perimeter data to create a clean dataset for fire perimeter reconstruction.

The pipeline mimics real-time conditions by using a buffer around final
perimeters to capture detections, then filters cross-fire contamination
and groups detections into observation windows based on satellite pass timing.

Steps:
1. load_raw_data(years)
   - Loads CAL FIRE perimeters and FIRMS detections from three VIIRS satellites
     (NOAA-20, NOAA-21, Suomi NPP), standardizes identifiers and timestamps.

2. match_firms_to_fires(firms, calfire, buffer)
   - Applies 5km buffer around CAL FIRE perimeters and uses spatial join to
     associate FIRMS detections with fire events within active date ranges.

3. filter_crossfire(firms_matched, calfire)
   - Identifies fires active within 10km during overlapping time periods and
     removes detections that lie closer to a neighboring fire's perimeter.

4. apply_final_filters(firms_filtered, calfire, min_points, min_windows, min_area_km2)
   - Assigns observation windows based on 2-hour gaps between detections.
   - Filters fires requiring minimum windows (4), points (150), and area (1 km²).

5. load_data(years, buffer, min_points, min_windows)
   - Main entry point. Runs full pipeline and saves output files.

Output:
Saves two parquet files:
    - firms_filtered.parquet: Cleaned FIRMS detections with fire_id and window_id
    - calfire_filtered.parquet: Matched CAL FIRE perimeters
"""

import numpy as np
import pandas as pd
import geopandas as gpd

def assign_windows(firms):
    """Assign window_id to each detection based on 2hr gap"""

    firms = firms.sort_values(['fire_id', 'acq_datetime'])
    
    window_ids = []
    for fire_id in firms['fire_id'].unique():
        fire_mask = firms['fire_id'] == fire_id
        fire_times = firms.loc[fire_mask, 'acq_datetime'].values
        
        window_id = 0
        fire_windows = [window_id]
        
        for i in range(1, len(fire_times)):
            gap = (fire_times[i] - fire_times[i-1]) / np.timedelta64(1, 'h')
            if gap > 2:
                window_id += 1
            fire_windows.append(window_id)
        
        window_ids.extend(fire_windows)
    
    firms['window_id'] = window_ids
    return firms

def load_raw_data(years):
    """Load FIRMS and CalFire data"""

    calfire = gpd.read_file("data/calfire_data/California_Fire_Perimeters_(all).shp")
    calfire = calfire[calfire["YEAR_"].isin(years)]
    calfire = calfire.dropna(subset=["FIRE_NAME", "INC_NUM"])
    calfire["ALARM_DATE"] = pd.to_datetime(calfire["ALARM_DATE"])
    calfire["CONT_DATE"] = pd.to_datetime(calfire["CONT_DATE"])
    calfire["fire_id"] = calfire["FIRE_NAME"] + "_" + calfire["INC_NUM"]
    calfire = calfire.to_crs(epsg=3310)
    
    firms_files = {
        "J1": ("DL_FIRE_J1V-C2_679500.zip", "fire_archive_J1V-C2_679500.shp"),
        "J2": ("DL_FIRE_J2V-C2_679501.zip", "fire_nrt_J2V-C2_679501.shp"),
        "SUOMI": ("DL_FIRE_SV-C2_679502.zip", "fire_archive_SV-C2_679502.shp"),
    }
    
    firms_list = []
    for sat, (zip_name, shp_name) in firms_files.items():
        filepath = f"zip://data/firms_data/{zip_name}!{shp_name}"
        gdf = gpd.read_file(filepath)
        gdf["satellite"] = sat
        gdf["ACQ_DATE"] = pd.to_datetime(gdf["ACQ_DATE"])
        gdf = gdf[gdf["ACQ_DATE"].dt.year.isin(years)]
        firms_list.append(gdf)
    
    firms = pd.concat(firms_list, ignore_index=True)
    firms = firms.to_crs(epsg=3310)
    
    time_str = firms["ACQ_TIME"].astype(int).astype(str).str.zfill(4)
    firms["acq_datetime"] = firms["ACQ_DATE"] + pd.to_timedelta(
        time_str.str[:2] + ':' + time_str.str[2:] + ':00'
    )
    
    return firms, calfire

def match_firms_to_fires(firms, calfire, buffer):
    """Spatial join with buffer + temporal filter"""
    
    calfire_buffered = calfire.copy()
    calfire_buffered["geometry"] = calfire_buffered.geometry.buffer(buffer)
    
    firms_joined = gpd.sjoin(
        firms,
        calfire_buffered[["geometry", "fire_id", "FIRE_NAME", "ALARM_DATE", "CONT_DATE"]],
        how="inner",
        predicate="within"
    )
    
    firms_matched = firms_joined[
        (firms_joined["ACQ_DATE"] >= firms_joined["ALARM_DATE"]) &
        (firms_joined["ACQ_DATE"] <= firms_joined["CONT_DATE"])
    ]
    
    fire_counts = firms_matched.groupby("fire_id").size()
    valid_fires = fire_counts[fire_counts >= 50].index
    firms_matched = firms_matched[firms_matched["fire_id"].isin(valid_fires)]
    
    return firms_matched

def get_concurrent_fires(fire_id, det_start, det_end, this_geom, calfire):
    """Find fires active during same time period and within 10km"""

    concurrent = []
    
    for _, other in calfire[calfire['fire_id'] != fire_id].iterrows():
        other_start = other['ALARM_DATE']
        other_end = other['CONT_DATE']
        
        # Check actual time overlap (no buffer)
        if not ((det_start <= other_end) and (det_end >= other_start)):
            continue
        
        # Check within 10km
        if this_geom.distance(other.geometry) > 10000:
            continue
            
        concurrent.append(other)
    return concurrent

def filter_fire(fire_id, fire_dets, calfire):
    """Remove detections closer to another fire's perimeter"""

    if len(fire_dets) == 0:
        return fire_dets
    
    this_fire = calfire[calfire['fire_id'] == fire_id].copy()
    
    this_geom = this_fire.iloc[0].geometry
    concurrent = get_concurrent_fires(fire_id, fire_dets['acq_datetime'].min(), 
                                       fire_dets['acq_datetime'].max(), this_geom, calfire)
    if len(concurrent) == 0:
        return fire_dets
    
    dist_to_this = np.array([pt.distance(this_geom.boundary) for pt in fire_dets.geometry])
    inside_this = fire_dets.geometry.within(this_geom).values
    
    keep_mask = np.ones(len(fire_dets), dtype=bool)
    
    for other in concurrent:
        other_geom = other.geometry
        dist_to_other = np.array([pt.distance(other_geom.boundary) for pt in fire_dets.geometry])
        should_remove = (dist_to_other < dist_to_this) & ~inside_this
        keep_mask = keep_mask & ~should_remove
    
    return fire_dets[keep_mask]

def filter_crossfire(firms_matched, calfire):
    """Remove detections closer to other concurrent fires"""
    
    fire_ids = firms_matched['fire_id'].unique()
    all_kept = []
    
    for i, fire_id in enumerate(fire_ids):
        fire_dets = firms_matched[firms_matched['fire_id'] == fire_id]
        kept = filter_fire(fire_id, fire_dets, calfire)
        all_kept.append(kept)
    
    firms_filtered = pd.concat(all_kept, ignore_index=True)
    return firms_filtered

def apply_final_filters(firms_filtered, calfire, min_points, min_windows, min_area_km2=1):
    """Apply minimum point, window, and area requirements"""
    
    firms_filtered = assign_windows(firms_filtered.copy())

    valid_fires = []
    for fire_id in firms_filtered['fire_id'].unique():
        fire_data = firms_filtered[firms_filtered['fire_id'] == fire_id]
        n_windows = fire_data['window_id'].nunique()
        
        # Check calfire match exists
        calfire_match = calfire[calfire['fire_id'] == fire_id]
        if len(calfire_match) == 0:
            continue
        
        # Check area requirement in kilometers
        actual_area_km2 = calfire_match.iloc[0].geometry.area / 1e6
        if actual_area_km2 < min_area_km2:
            continue
        
        if n_windows >= min_windows and len(fire_data) >= min_points:
            valid_fires.append(fire_id)
    
    firms_final = firms_filtered[firms_filtered['fire_id'].isin(valid_fires)]
    return firms_final

def load_data(years=[2021, 2022, 2023, 2024, 2025], buffer=5000, min_points=150, min_windows=4):
    """Load FIRMS and CalFire, filter cross-fire contamination"""
    
    firms, calfire = load_raw_data(years)
    firms_matched = match_firms_to_fires(firms, calfire, buffer)
    firms_filtered = filter_crossfire(firms_matched, calfire)
    firms_final = apply_final_filters(firms_filtered, calfire, min_points, min_windows)
    
    # Save both files
    firms_final.to_parquet("data/firms_filtered.parquet")
    calfire.to_parquet("data/calfire_filtered.parquet")

    print(f"Saved {firms_final['fire_id'].nunique()} fires to data/firms_filtered.parquet")
    print("Saved calfire to data/calfire_filtered.parquet")
    return None
