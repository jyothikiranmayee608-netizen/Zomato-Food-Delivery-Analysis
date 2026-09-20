"""
Zomato Food Delivery - Data Cleaning & Feature Engineering
==========================================================
Reads:  data/Zomato Dataset.csv   (raw - never modified)
Writes: data/cleaned/zomato_cleaned.csv
        data/cleaned/cleaning_log.txt

All cleaning decisions are documented inline.
"""

import math
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

RAW_PATH   = "data/Zomato Dataset.csv"
CLEAN_PATH = "data/cleaned/zomato_cleaned.csv"
LOG_PATH   = "data/cleaned/cleaning_log.txt"

log_lines = []

def log(msg):
    print(msg)
    log_lines.append(msg)

# ── 1. Load with explicit dtype to avoid pandas 3 StringDtype issues ──────────
df = pd.read_csv(RAW_PATH, dtype=object)  # load everything as object first
# Convert numeric columns explicitly
numeric_cols = [
    "Delivery_person_Age", "Delivery_person_Ratings",
    "Restaurant_latitude", "Restaurant_longitude",
    "Delivery_location_latitude", "Delivery_location_longitude",
    "Vehicle_condition", "multiple_deliveries", "Time_taken (min)"
]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

log(f"[LOAD] Raw shape: {df.shape}")

# ── 2. NaN-like strings -> proper NaN ────────────────────────────────────────
str_cols = [c for c in df.columns if c not in numeric_cols]
for col in str_cols:
    mask = df[col].astype(str).str.strip().isin(["NaN", "nan", "NA", "N/A", "null", "NULL", ""])
    n = mask.sum()
    if n:
        df.loc[mask, col] = np.nan
        log(f"[NaN-STRING] {col}: {n} NaN-like strings converted to NaN")
    # Strip whitespace
    df[col] = df[col].where(df[col].isna(), df[col].astype(str).str.strip())

log(f"[LOAD] Missing after NaN-string cleanup:\n"
    + df.isnull().sum()[df.isnull().sum() > 0].to_string())

# ── 3. City: fix typo 'Metropolitian' -> 'Metropolitan' ──────────────────────
typo_count = (df["City"] == "Metropolitian").sum()
df["City"] = df["City"].replace("Metropolitian", "Metropolitan")
log(f"[TYPO-FIX] City 'Metropolitian' -> 'Metropolitan': {typo_count} rows corrected")

# ── 4. Missing value treatment ────────────────────────────────────────────────
# Strategy:
#   Numeric (Age, Ratings, multiple_deliveries) -> median imputation (robust to outliers)
#   Categorical (Weather, Traffic, Festival, City) -> mode imputation
#   Time_Orderd -> ffill/bfill within delivery person, fallback = pickup time
# Rationale: imputation preferred over deletion to preserve ~45k rows

# 4a. Delivery_person_Age
age_median = df["Delivery_person_Age"].median()
n_age = df["Delivery_person_Age"].isna().sum()
df["Delivery_person_Age"] = df["Delivery_person_Age"].fillna(age_median)
log(f"[IMPUTE] Delivery_person_Age: {n_age} NaN -> median {age_median:.1f}")

# 4b. Delivery_person_Ratings
rat_median = df["Delivery_person_Ratings"].median()
n_rat = df["Delivery_person_Ratings"].isna().sum()
df["Delivery_person_Ratings"] = df["Delivery_person_Ratings"].fillna(rat_median)
log(f"[IMPUTE] Delivery_person_Ratings: {n_rat} NaN -> median {rat_median:.1f}")

# 4c. multiple_deliveries
mul_median = df["multiple_deliveries"].median()
n_mul = df["multiple_deliveries"].isna().sum()
df["multiple_deliveries"] = df["multiple_deliveries"].fillna(mul_median)
log(f"[IMPUTE] multiple_deliveries: {n_mul} NaN -> median {mul_median:.0f}")

# 4d. Weather_conditions
weather_mode = df["Weather_conditions"].dropna().mode()[0]
n_weath = df["Weather_conditions"].isna().sum()
df["Weather_conditions"] = df["Weather_conditions"].fillna(weather_mode)
log(f"[IMPUTE] Weather_conditions: {n_weath} NaN -> mode '{weather_mode}'")

# 4e. Road_traffic_density
traffic_mode = df["Road_traffic_density"].dropna().mode()[0]
n_traf = df["Road_traffic_density"].isna().sum()
df["Road_traffic_density"] = df["Road_traffic_density"].fillna(traffic_mode)
log(f"[IMPUTE] Road_traffic_density: {n_traf} NaN -> mode '{traffic_mode}'")

# 4f. Festival
fest_mode = df["Festival"].dropna().mode()[0]
n_fest = df["Festival"].isna().sum()
df["Festival"] = df["Festival"].fillna(fest_mode)
log(f"[IMPUTE] Festival: {n_fest} NaN -> mode '{fest_mode}'")

# 4g. City
city_mode = df["City"].dropna().mode()[0]
n_city = df["City"].isna().sum()
df["City"] = df["City"].fillna(city_mode)
log(f"[IMPUTE] City: {n_city} NaN -> mode '{city_mode}'")

# 4h. Time_Orderd - ffill/bfill within same person, then fallback to pickup time
n_time = df["Time_Orderd"].isna().sum()
df["Time_Orderd"] = (
    df.groupby("Delivery_person_ID")["Time_Orderd"]
    .transform(lambda s: s.ffill().bfill())
)
df["Time_Orderd"] = df["Time_Orderd"].fillna(df["Time_Order_picked"])
log(f"[IMPUTE] Time_Orderd: {n_time} NaN -> ffill/bfill within person, else pickup time")

# ── 5. Suspicious value treatment ─────────────────────────────────────────────

# 5a. Age == 15: below plausible legal working age. Cap to 18.
n_age15 = (df["Delivery_person_Age"] == 15).sum()
df.loc[df["Delivery_person_Age"] == 15, "Delivery_person_Age"] = 18
log(f"[SUSPICIOUS] Delivery_person_Age==15 ({n_age15} rows) -> capped to 18 (minimum plausible working age)")

# 5b. Ratings > 5: valid scale is 1-5; cap to 5.0 (data entry error)
n_rat6 = (df["Delivery_person_Ratings"] > 5).sum()
df.loc[df["Delivery_person_Ratings"] > 5, "Delivery_person_Ratings"] = 5.0
log(f"[SUSPICIOUS] Delivery_person_Ratings > 5 ({n_rat6} rows) -> capped to 5.0")

# 5c. Zero coordinates: GPS not recorded. Flag rows, distance will be NaN.
n_zero_coord = (df["Restaurant_latitude"] == 0).sum()
df["coord_quality_flag"] = (df["Restaurant_latitude"] == 0).astype(int)
log(f"[SUSPICIOUS] Restaurant_latitude==0 ({n_zero_coord} rows) -> flagged in 'coord_quality_flag', distance=NaN")

# ── 6. Data types ──────────────────────────────────────────────────────────────
df["Order_Date"] = pd.to_datetime(df["Order_Date"], dayfirst=True, errors="coerce")
df["Delivery_person_Age"]    = df["Delivery_person_Age"].astype(float).astype(int)
df["Vehicle_condition"]      = df["Vehicle_condition"].astype(float).astype(int)
df["multiple_deliveries"]    = df["multiple_deliveries"].astype(float).astype(int)
df["Time_taken (min)"]       = df["Time_taken (min)"].astype(float).astype(int)

for cat_col in ["Weather_conditions", "Road_traffic_density", "Type_of_order",
                "Type_of_vehicle", "Festival", "City"]:
    df[cat_col] = df[cat_col].astype("category")

log("[DTYPE] Parsed Order_Date; converted numeric columns; categorised 6 columns")

# ── 7. Feature Engineering ────────────────────────────────────────────────────

# 7a. Haversine distance (km)
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(max(0, min(1, a))))

def haversine_row(r):
    if r["coord_quality_flag"] == 1:
        return np.nan
    try:
        # Use absolute coordinates to avoid sign-flip GPS artifacts
        dist = haversine(
            abs(float(r["Restaurant_latitude"])),  abs(float(r["Restaurant_longitude"])),
            abs(float(r["Delivery_location_latitude"])), abs(float(r["Delivery_location_longitude"]))
        )
        # Null out implausible distances (>50 km for a food delivery)
        return dist if dist <= 50 else np.nan
    except Exception:
        return np.nan

df["Distance_km"] = df.apply(haversine_row, axis=1).round(3)
n_implausible = df["Distance_km"].isna().sum() - (df["coord_quality_flag"] == 1).sum()
log(f"[FEATURE] Distance_km via Haversine (abs coords); "
    f"{(df['coord_quality_flag']==1).sum()} zero-coord NaN; "
    f"~{max(0,n_implausible)} implausible (>50 km) also set to NaN")

# 7b. Order_Hour
def parse_hour(t):
    try:
        return int(str(t).split(":")[0])
    except Exception:
        return np.nan

df["Order_Hour"] = df["Time_Orderd"].apply(parse_hour)
log(f"[FEATURE] Order_Hour extracted; {df['Order_Hour'].isna().sum()} NaN")

# 7c. Pickup_Duration_min
def time_to_min(t):
    try:
        parts = str(t).split(":")
        return int(parts[0]) * 60 + int(parts[1])
    except Exception:
        return np.nan

orderd_min   = df["Time_Orderd"].apply(time_to_min)
pickedup_min = df["Time_Order_picked"].apply(time_to_min)
diff = pickedup_min - orderd_min
# Correct for midnight crossover
diff = diff.apply(lambda x: x + 1440 if pd.notna(x) and x < 0 else x)
df["Pickup_Duration_min"] = diff
log(f"[FEATURE] Pickup_Duration_min; {df['Pickup_Duration_min'].isna().sum()} NaN; "
    f"range [{df['Pickup_Duration_min'].min():.0f}, {df['Pickup_Duration_min'].max():.0f}] min")

# 7d. Date features
df["Order_DayOfWeek"] = df["Order_Date"].dt.day_name()
df["Order_Month"]     = df["Order_Date"].dt.month
df["Order_Week"]      = df["Order_Date"].dt.isocalendar().week.astype("Int64")
log("[FEATURE] Order_DayOfWeek, Order_Month, Order_Week extracted")

# 7e. Time-of-day bucket
def time_bucket(h):
    if pd.isna(h): return np.nan
    h = int(h)
    if 5  <= h < 11: return "Morning"
    if 11 <= h < 15: return "Lunch"
    if 15 <= h < 19: return "Afternoon"
    if 19 <= h < 23: return "Evening"
    return "Late Night"

df["Time_of_Day"] = df["Order_Hour"].apply(time_bucket).astype("category")
log("[FEATURE] Time_of_Day bucket created")

# 7f. Rename target column
df.rename(columns={"Time_taken (min)": "Delivery_Time_min"}, inplace=True)
log("[RENAME] 'Time_taken (min)' -> 'Delivery_Time_min'")

# ── 8. Final checks ───────────────────────────────────────────────────────────
dupes = df.duplicated().sum()
log(f"[CHECK] Duplicate rows after cleaning: {dupes}")
log(f"[CHECK] Final shape: {df.shape}")
remaining_missing = df.isnull().sum()[df.isnull().sum() > 0]
if len(remaining_missing):
    log(f"[CHECK] Remaining NaN (expected - derived features only):\n{remaining_missing.to_string()}")
else:
    log("[CHECK] No missing values remaining in core columns")

# ── 9. Save ───────────────────────────────────────────────────────────────────
df.to_csv(CLEAN_PATH, index=False)
log(f"[SAVE] Cleaned dataset -> {CLEAN_PATH}")

with open(LOG_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(log_lines))
log(f"[SAVE] Cleaning log -> {LOG_PATH}")

print("\n=== DONE ===")
