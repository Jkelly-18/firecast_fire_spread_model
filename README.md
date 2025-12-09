# FireCast: Satellite-Based Wildfire Perimeter Construction

## Overview

Wildfires have become an increasingly prominent challenge in California, with several fires causing massive devastation to homeowners and the environment. Effective wildfire management depends on timely response and accurate information about current conditions, yet there is currently a lack of real-time automated reporting for wildfire perimeters.

FireCast is an algorithm that estimates wildfire perimeter progression through the lifecycle of a fire using NASA FIRMS satellite detections and geospatial analysis techniques. The system clusters thermal detections using DBSCAN, merges nearby fire fronts, filters outlier points, and constructs polygon boundaries using concave hull methods. By testing on 93 historical California wildfires from 2021-2025, we establish a foundation to adapt the system to real-time perimeter prediction during active wildfires.

Our algorithm achieved a mean F1.25 score of 0.870 and IoU of 0.751 across all evaluated fires, demonstrating strong geometric agreement with official CAL FIRE perimeter records.

**Live Dashboard:** https://firecast-wildfire-monitor.netlify.app/

The interactive dashboard allows you to explore all 93 fires, filter by year, scrub through observation windows to watch fire progression, and compare predicted perimeters against official CAL FIRE boundaries.

For full methodology and analysis, see the project report: `reports/FireCast_Final_Report.pdf`

## Installation
```bash
pip install -r requirements.txt
```

## Data Access

This project uses two data sources that must be downloaded separately:

- **NASA FIRMS VIIRS Detections**: Download from https://firms.modaps.eosdis.nasa.gov/download/
  - Select VIIRS (S-NPP, NOAA-20, NOAA-21) for California region
  - Place downloaded files in `data/firms_data/`

- **CAL FIRE Perimeters**: Download from https://www.fire.ca.gov/what-we-do/fire-resource-assessment-program/fire-perimeters
  - Download "California Fire Perimeters (all)"
  - Place shapefile in `data/calfire_data/`

Data is owned by NASA and CAL FIRE respectively and used here for educational purposes.

## How to Run

1. **Preprocess data** (or use pre-saved parquet files):
```python
   from data_preprocessing import load_data
   load_data()
```

2. **Run the analysis notebook**:
   Open `FireCast_Perimeter_Analysis.ipynb` and run cells sequentially. Pre-computed results are available in `data/perimeters/` to skip computationally expensive cells.

3. **Export dashboard data**:
```bash
   python export_dashboard.py
```

## File Structure
```
FireCast/
├── data_preprocessing.py                 # FIRMS/CAL FIRE data loading and filtering
├── perimeter_pipeline.py                 # Perimeter construction algorithm
├── export_dashboard.py                   # Export data for web dashboard
├── FireCast_Perimeter_Analysis.ipynb     # Full analysis notebook
├── requirements.txt
├── data/
│   ├── firms_filtered.parquet            # Preprocessed FIRMS detections
│   ├── calfire_filtered.parquet          # Filtered CAL FIRE perimeters
│   └── perimeters/
│       ├── fire_evaluation.parquet       # Evaluation metrics per fire
│       └── window_perimeters.parquet     # Perimeters at each observation window
├── dashboard/
│   ├── index.html
│   ├── styles.css
│   ├── app.js
│   └── dashboard_data/
│       ├── fire_data.json
│       └── perimeters/
└── reports/
    └── FireCast_Final_Report.pdf
```

## Authors

- Michael Michelini
- Jonathan Kelly

University of Michigan - SIADS 699 Capstone Project
