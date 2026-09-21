# Zomato Food Delivery – Operations & Delivery Time Analysis

## Project Title
**Zomato Food Delivery – Operations & Delivery Time Analysis**

---

## Problem Statement
Food delivery platforms like Zomato operate in a highly competitive, time-sensitive environment where customer satisfaction is tightly coupled to delivery speed and reliability. Understanding the factors that influence delivery time — traffic, weather, festivals, distance, vehicle type, rider behaviour — is essential for operational improvement and accurate ETA estimation.

---

## Objective
To perform a comprehensive exploratory analysis of Zomato's food delivery dataset in order to:
- Understand the distribution and key drivers of delivery time
- Identify operational bottlenecks through 12 targeted business questions
- Derive evidence-based business insights and actionable recommendations
- Establish a reproducible, documented analysis workflow

---

## Dataset Source
**File:** `data/Zomato Dataset.csv`  
**Source:** Publicly available Zomato food delivery dataset  
**Size:** 45,584 delivery records × 20 original columns
 dataset link:https://www.kaggle.com/datasets/saurabhbadole/zomato-delivery-operations-analytics-dataset

---

## Dataset Description

| Column | Description |
|---|---|
| `ID` | Unique delivery ID |
| `Delivery_person_ID` | Rider identifier |
| `Delivery_person_Age` | Rider age (years) |
| `Delivery_person_Ratings` | Rider rating (1–5 scale) |
| `Restaurant_latitude/longitude` | GPS coordinates of restaurant |
| `Delivery_location_latitude/longitude` | GPS coordinates of delivery address |
| `Order_Date` | Date of order (DD-MM-YYYY) |
| `Time_Orderd` | Time order was placed (HH:MM) |
| `Time_Order_picked` | Time order was picked up by rider (HH:MM) |
| `Weather_conditions` | Weather at time of delivery |
| `Road_traffic_density` | Traffic level (Low / Medium / High / Jam) |
| `Vehicle_condition` | Condition of vehicle (0=worst, 3=best) |
| `Type_of_order` | Food category (Meal, Snack, Drinks, Buffet) |
| `Type_of_vehicle` | Vehicle used (motorcycle, scooter, bicycle, electric_scooter) |
| `multiple_deliveries` | Number of simultaneous deliveries (0–3) |
| `Festival` | Whether delivery was during a festival (Yes/No) |
| `City` | City tier (Metropolitan, Urban, Semi-Urban) |
| `Time_taken (min)` | **Target variable** – total delivery time in minutes |

---

## Methodology

1. **Load** raw CSV without modification
2. **Inspect** shape, dtypes, missing values, unique categories, value ranges
3. **Clean** using documented, defensible treatments for each issue
4. **Transform** dtypes and rename columns for clarity
5. **Feature Engineering** – derive Distance_km, Order_Hour, Pickup_Duration_min, Time_of_Day, date features
6. **EDA** – answer 12 business questions with statistical analysis and visualisations
7. **KPIs** – compute aggregate performance indicators
8. **Insights** – translate findings into business implications and actions

---

## Data Cleaning Steps

| Issue | Count | Treatment |
|---|---|---|
| NaN-like strings | 0 found | Defensive scan applied |
| `Metropolitian` typo in City | 34,087 rows | Renamed to `Metropolitan` |
| Missing `Delivery_person_Age` | 1,854 | Median imputation (30.0 years) |
| Missing `Delivery_person_Ratings` | 1,908 | Median imputation (4.7) |
| Missing `multiple_deliveries` | 993 | Median imputation (1) |
| Missing `Weather_conditions` | 616 | Mode imputation (Fog) |
| Missing `Road_traffic_density` | 601 | Mode imputation (Low) |
| Missing `Festival` | 228 | Mode imputation (No) |
| Missing `City` | 1,200 | Mode imputation (Metropolitan) |
| Missing `Time_Orderd` | 1,731 | ffill/bfill within rider; fallback = pickup time |
| `Age == 15` | 38 rows | Capped to 18 (minimum plausible working age) |
| `Ratings > 5` | 53 rows | Capped to 5.0 (data entry error; scale is 1–5) |
| `Restaurant_latitude == 0` | 3,640 rows | Flagged in `coord_quality_flag`; `Distance_km = NaN` |
| Sign-flipped GPS coordinates | 431 rows | Absolute values used in Haversine; distances > 50 km nullified |

**No rows were deleted. Final cleaned shape: 45,584 × 28.**

---

## Feature Engineering

| Feature | Description | Method |
|---|---|---|
| `Distance_km` | Straight-line restaurant-to-customer distance | Haversine formula (abs coords, cap 50 km) |
| `Order_Hour` | Hour of day order was placed (0–23) | Split `Time_Orderd` on `:` |
| `Pickup_Duration_min` | Time from order to rider pickup | `Time_Order_picked` − `Time_Orderd` (midnight-safe) |
| `Time_of_Day` | Categorical bucket (Morning/Lunch/Afternoon/Evening/Late Night) | Derived from `Order_Hour` |
| `Order_DayOfWeek` | Day name from `Order_Date` | `dt.day_name()` |
| `Order_Month` | Month number | `dt.month` |
| `Order_Week` | ISO week number | `dt.isocalendar().week` |
| `coord_quality_flag` | 1 = GPS not recorded (lat=0) | Boolean flag |

---

## Key Findings

1. **Average delivery time: 26.3 min; Median: 26.0 min** – distribution is near-symmetric
2. **Festival orders take ~75% longer** (45.5 min) than non-festival (25.9 min) – the most extreme factor observed
3. **Traffic Jam conditions add ~10 min** over Low traffic (31.2 vs 21.5 min) – strongest continuous operational driver
4. **Multiple simultaneous deliveries sharply increase time**: 0 deliveries = 22.9 min, 2 = 40.5 min, 3 = 47.8 min
5. **Distance moderately correlates with delivery time** (Spearman r = 0.32)
6. **Higher-rated riders show shorter delivery times** (Spearman r = −0.28)
7. **Cloudy and Foggy weather adds 2–7 min** vs Sunny conditions
8. **Metropolitan areas are slower than Urban** (27.1 vs 23.0 min)

---

## How to Run the Project

### Prerequisites
- Python 3.11
- `.venv` already created and packages installed (see `requirements.txt`)

### Step 1 – Activate the virtual environment
```powershell
.venv\Scripts\Activate.ps1
```

### Step 2 – Run data cleaning and feature engineering
```powershell
python scripts/clean_and_engineer.py
```

### Step 3 – Run EDA and generate figures
```powershell
python scripts/eda_analysis.py
```

### Step 4 – Open the Jupyter notebook
```powershell
jupyter notebook notebooks/zomato_delivery_analysis.ipynb
```
Or execute programmatically:
```powershell
python scripts/execute_notebook.py
```

---

## Requirements

```
pandas==3.0.6
matplotlib==3.11.2
seaborn==0.13.2
scipy==1.17.1
nbformat==5.11.1
ipykernel==7.3.0
numpy==2.4.6
```

Install with:
```powershell
.venv\Scripts\pip.exe install -r requirements.txt
```

---

## Project Structure

```
Zomato-Food-Delivery-Analysis/
├── data/
│   ├── Zomato Dataset.csv           # Raw dataset (never modified)
│   └── cleaned/
│       ├── zomato_cleaned.csv       # Cleaned + feature-engineered dataset
│       └── cleaning_log.txt         # Complete audit trail of all cleaning steps
├── notebooks/
│   └── zomato_delivery_analysis.ipynb   # Full reproducible analysis notebook
├── report/
│   ├── figures/                     # All 13 analysis figures (PNG)
│   │   ├── 01_delivery_time_distribution.png
│   │   ├── 02_delivery_time_by_city.png
│   │   ├── 03_delivery_time_by_traffic.png
│   │   ├── 04_delivery_time_by_weather.png
│   │   ├── 05_delivery_time_by_vehicle.png
│   │   ├── 06_delivery_time_by_vehicle_condition.png
│   │   ├── 07_delivery_time_by_multiple_deliveries.png
│   │   ├── 08_delivery_time_festival.png
│   │   ├── 09_delivery_time_by_rating.png
│   │   ├── 10_delivery_time_by_distance.png
│   │   ├── 11_pickup_traffic_weather_vs_delivery.png
│   │   ├── 12_correlation_heatmap.png
│   │   └── 13_delivery_time_by_hour.png
│   ├── kpis.json                    # Computed KPIs
│   ├── insights.txt                 # Business insights text
│   └── Zomato_Delivery_Analysis_Report.md   # Full analysis report
├── scripts/
│   ├── clean_and_engineer.py        # Data cleaning & feature engineering
│   ├── eda_analysis.py              # EDA, figures, KPIs, insights
│   ├── build_notebook.py            # Generates the .ipynb file
│   └── execute_notebook.py          # Executes the notebook programmatically
├── requirements.txt
└── README.md
```

---

## Data Integrity

- The raw file `data/Zomato Dataset.csv` was **never modified, overwritten, or deleted**.
- All transformations were applied to in-memory copies and written to `data/cleaned/`.
- Every cleaning decision is documented in `data/cleaned/cleaning_log.txt` and in the notebook.

---

*Analysis conducted using Python 3.11. All findings are observational; causal conclusions require controlled experimentation.*
