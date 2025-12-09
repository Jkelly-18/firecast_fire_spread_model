# FireCast: Satellite-Based Wildfire Perimeter Construction

## Overview

Wildfires have become an increasingly prominent challenge in California, with several fires causing massive devastation to homeowners and the environment. Effective wildfire management depends on timely response and accurate information about current conditions, yet there is currently a lack of real-time automated reporting for wildfire perimeters.

FireCast is an algorithm that estimates wildfire perimeter progression through the lifecycle of a fire using NASA FIRMS satellite detections and geospatial analysis techniques. The system clusters thermal detections using DBSCAN, merges nearby fire fronts, filters outlier points, and constructs polygon boundaries using concave hull methods. By testing on 93 historical California wildfires from 2021 through early 2025, we establish a foundation to later adapt the system to real-time perimeter prediction during active wildfires.

Our algorithm achieved a mean F1.25 score of 0.870 and IoU of 0.751 across all evaluated fires, demonstrating strong geometric agreement with official CAL FIRE perimeter records.

**Live Dashboard:** https://firecast-wildfire-tracking.netlify.app/

The interactive dashboard allows you to explore all 93 fires, filter by year, step through observation windows to watch fire progression, and compare predicted perimeters against official CAL FIRE boundaries.

For full methodology and analysis, see the project report: `reports/FireCast_Final_Report.pdf`

## Data Access

This project uses two data sources that must be downloaded separately:

- **NASA FIRMS VIIRS Detections**: Download from https://firms.modaps.eosdis.nasa.gov/download/
  - Requires free NASA Earthdata account (https://urs.earthdata.nasa.gov/users/new)
  - Select VIIRS (S-NPP, NOAA-20, NOAA-21) for California region
  - Select date range: January 2021 – February 2025
  - Place downloaded zip files in `data/firms_data/`

- **CAL FIRE Perimeters**: Download from https://gis.data.ca.gov/datasets/CALFIRE-Forestry::california-fire-perimeters-all/explore
  - Click the download icon and select "Shapefile"
  - Unzip and place contents in `data/calfire_data/`

Data is owned by NASA and CAL FIRE respectively and used here for educational purposes.

## How to Run

Install dependencies:
```bash
pip install -r requirements.txt
```

The following steps run the full pipeline from data preprocessing through dashboard export:

1. **Preprocess data**:
```python
   from data_preprocessing import load_data
   load_data()
```
   This filters and joins the raw FIRMS and CAL FIRE data. Pre-saved parquet files (`firms_filtered.parquet`, `calfire_filtered.parquet`) are included in the repository, so this step can be skipped.

2. **Run the analysis notebook**:
   Open `FireCast_Perimeter_Analysis.ipynb` and run cells sequentially. The notebook walks through parameter tuning, perimeter generation, and evaluation. Pre-computed results in `data/perimeters/` can be loaded to skip computationally expensive cells, as noted in the notebook.

3. **Export dashboard data**:
```bash
   python export_dashboard.py
```
   This exports perimeter geometries and fire metadata to JSON format in `dashboard/dashboard_data/` for use by the web dashboard. To run the dashboard locally, open `dashboard/index.html` in a browser.

## File Structure
```
FireCast/
├── README.md
├── data_preprocessing.py                 # FIRMS/CAL FIRE data loading and filtering
├── perimeter_pipeline.py                 # Perimeter construction algorithm
├── export_dashboard.py                   # Export data for web dashboard
├── firecast_perimeter_analysis.ipynb     # Full analysis notebook
├── requirements.txt
├── data/
│   ├── firms_data/
│   │   └── README.md                     # FIRMS download instructions
│   ├── calfire_data/
│   │   └── README.md                     # CAL FIRE download instructions
│   ├── firms_filtered.parquet            # Preprocessed FIRMS detections
│   ├── calfire_filtered.parquet          # Filtered CAL FIRE perimeters
│   └── perimeters/
│       ├── fire_evaluation.parquet       # Evaluation metrics per fire
│       └── window_perimeters.parquet     # Perimeters at each observation window
└── dashboard/                            # Interactive web dashboard and exported data
    ├── index.html
    ├── styles.css
    ├── app.js
    └── dashboard_data/
        ├── fire_data.json
        └── perimeters/
```

## Authors

- Jonathan Kelly
- Michael Michelini

University of Michigan - SIADS 699 Capstone Project
