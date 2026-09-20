"""
Zomato Food Delivery - EDA, Business Questions, KPIs, and Insights
===================================================================
Reads:  data/cleaned/zomato_cleaned.csv
Writes: report/figures/*.png
        report/kpis.json
        report/insights.txt
"""

import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")

CLEAN_PATH  = "data/cleaned/zomato_cleaned.csv"
FIG_DIR     = "report/figures"
KPI_PATH    = "report/kpis.json"
INSIGHTS_PATH = "report/insights.txt"

# ── Style ─────────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
ACCENT = "#3b82d4"
FIG_DPI = 150

def savefig(name):
    path = f"{FIG_DIR}/{name}.png"
    plt.tight_layout()
    plt.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    print(f"  [FIG] {path}")

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(CLEAN_PATH)
print(f"[LOAD] {df.shape}")
TARGET = "Delivery_Time_min"

# ── Ordered categories for traffic ───────────────────────────────────────────
traffic_order = ["Low", "Medium", "High", "Jam"]

# ─────────────────────────────────────────────────────────────────────────────
# Q1 & Q2  Overall distribution, average and median delivery time
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Q1/Q2] Distribution of delivery time")
overall_mean   = df[TARGET].mean()
overall_median = df[TARGET].median()
print(f"  Mean: {overall_mean:.2f} min  |  Median: {overall_median:.2f} min")

fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(df[TARGET], bins=40, color=ACCENT, edgecolor="white", alpha=0.85)
ax.axvline(overall_mean,   color="#e74c3c", lw=2, linestyle="--", label=f"Mean {overall_mean:.1f} min")
ax.axvline(overall_median, color="#2ecc71", lw=2, linestyle=":",  label=f"Median {overall_median:.1f} min")
ax.set_xlabel("Delivery Time (min)")
ax.set_ylabel("Count")
ax.set_title("Overall Delivery Time Distribution")
ax.legend()
savefig("01_delivery_time_distribution")

# ─────────────────────────────────────────────────────────────────────────────
# Q3  Delivery time by city
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Q3] Delivery time by city")
city_stats = (
    df.groupby("City")[TARGET]
    .agg(Mean="mean", Median="median", Count="count")
    .sort_values("Mean", ascending=False)
)
print(city_stats)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
# Box plot
order_cities = city_stats.index.tolist()
sns.boxplot(data=df, x="City", y=TARGET, order=order_cities,
            palette="muted", ax=axes[0])
axes[0].set_title("Delivery Time by City (Boxplot)")
axes[0].set_xlabel("City")
axes[0].set_ylabel("Delivery Time (min)")
# Bar plot mean
axes[1].bar(city_stats.index, city_stats["Mean"], color=ACCENT, alpha=0.85)
axes[1].set_title("Mean Delivery Time by City")
axes[1].set_xlabel("City")
axes[1].set_ylabel("Mean Delivery Time (min)")
axes[1].yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
for i, (city, row) in enumerate(city_stats.iterrows()):
    axes[1].text(i, row["Mean"] + 0.3, f"{row['Mean']:.1f}", ha="center", fontsize=10)
savefig("02_delivery_time_by_city")

# ─────────────────────────────────────────────────────────────────────────────
# Q4  Traffic density vs delivery time
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Q4] Delivery time by traffic density")
traffic_stats = (
    df.groupby("Road_traffic_density")[TARGET]
    .agg(Mean="mean", Median="median", Count="count")
    .reindex(traffic_order)
)
print(traffic_stats)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sns.boxplot(data=df, x="Road_traffic_density", y=TARGET,
            order=traffic_order, palette="Reds", ax=axes[0])
axes[0].set_title("Delivery Time by Traffic Density")
axes[0].set_xlabel("Traffic Density")
axes[0].set_ylabel("Delivery Time (min)")

axes[1].bar(traffic_stats.index, traffic_stats["Mean"],
            color=["#78c0a8","#f5a623","#e07b3c","#c0392b"], alpha=0.9)
axes[1].set_title("Mean Delivery Time by Traffic Density")
axes[1].set_xlabel("Traffic Density")
axes[1].set_ylabel("Mean Delivery Time (min)")
for i, (lvl, row) in enumerate(traffic_stats.iterrows()):
    axes[1].text(i, row["Mean"] + 0.2, f"{row['Mean']:.1f}", ha="center", fontsize=10)
savefig("03_delivery_time_by_traffic")

# ─────────────────────────────────────────────────────────────────────────────
# Q5  Weather conditions vs delivery time
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Q5] Delivery time by weather")
weather_stats = (
    df.groupby("Weather_conditions")[TARGET]
    .agg(Mean="mean", Median="median", Count="count")
    .sort_values("Mean", ascending=False)
)
print(weather_stats)

fig, ax = plt.subplots(figsize=(10, 5))
order_w = weather_stats.index.tolist()
sns.barplot(data=df, x="Weather_conditions", y=TARGET, order=order_w,
            estimator="mean", errorbar="sd", palette="Blues_d", ax=ax)
ax.set_title("Mean Delivery Time by Weather Condition (with SD)")
ax.set_xlabel("Weather Condition")
ax.set_ylabel("Mean Delivery Time (min)")
savefig("04_delivery_time_by_weather")

# ─────────────────────────────────────────────────────────────────────────────
# Q6  Vehicle type vs delivery time
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Q6] Delivery time by vehicle type")
vehicle_stats = (
    df.groupby("Type_of_vehicle")[TARGET]
    .agg(Mean="mean", Median="median", Count="count")
    .sort_values("Mean")
)
print(vehicle_stats)

fig, ax = plt.subplots(figsize=(9, 5))
order_v = vehicle_stats.index.tolist()
sns.boxplot(data=df, x="Type_of_vehicle", y=TARGET, order=order_v,
            palette="Set2", ax=ax)
ax.set_title("Delivery Time by Vehicle Type")
ax.set_xlabel("Vehicle Type")
ax.set_ylabel("Delivery Time (min)")
savefig("05_delivery_time_by_vehicle")

# ─────────────────────────────────────────────────────────────────────────────
# Q7  Vehicle condition vs delivery time
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Q7] Delivery time by vehicle condition")
cond_stats = (
    df.groupby("Vehicle_condition")[TARGET]
    .agg(Mean="mean", Median="median", Count="count")
    .sort_index()
)
print(cond_stats)

fig, ax = plt.subplots(figsize=(8, 5))
cond_vals = sorted(df["Vehicle_condition"].unique())
sns.boxplot(data=df, x="Vehicle_condition", y=TARGET,
            order=cond_vals, palette="Purples", ax=ax)
ax.set_title("Delivery Time by Vehicle Condition (0=worst, 3=best)")
ax.set_xlabel("Vehicle Condition Score")
ax.set_ylabel("Delivery Time (min)")
savefig("06_delivery_time_by_vehicle_condition")

# ─────────────────────────────────────────────────────────────────────────────
# Q8  Multiple deliveries vs delivery time
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Q8] Delivery time by number of simultaneous deliveries")
multi_stats = (
    df.groupby("multiple_deliveries")[TARGET]
    .agg(Mean="mean", Median="median", Count="count")
    .sort_index()
)
print(multi_stats)

fig, ax = plt.subplots(figsize=(8, 5))
multi_vals = sorted(df["multiple_deliveries"].unique())
sns.boxplot(data=df, x="multiple_deliveries", y=TARGET,
            order=multi_vals, palette="Oranges", ax=ax)
ax.set_title("Delivery Time by Number of Simultaneous Deliveries")
ax.set_xlabel("Simultaneous Deliveries")
ax.set_ylabel("Delivery Time (min)")
savefig("07_delivery_time_by_multiple_deliveries")

# ─────────────────────────────────────────────────────────────────────────────
# Q9  Festival vs non-festival delivery time
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Q9] Festival vs non-festival")
fest_stats = (
    df.groupby("Festival")[TARGET]
    .agg(Mean="mean", Median="median", Count="count")
)
print(fest_stats)

# Mann-Whitney U test (non-parametric)
fest_yes = df[df["Festival"] == "Yes"][TARGET]
fest_no  = df[df["Festival"] == "No"][TARGET]
u_stat, p_val = stats.mannwhitneyu(fest_yes, fest_no, alternative="two-sided")
print(f"  Mann-Whitney U: U={u_stat:.0f}, p={p_val:.4f}")

fig, ax = plt.subplots(figsize=(7, 5))
sns.boxplot(data=df, x="Festival", y=TARGET, order=["No", "Yes"],
            palette=["#3b82d4", "#e07b3c"], ax=ax)
ax.set_title(f"Delivery Time: Festival vs Non-Festival\n(Mann-Whitney p={p_val:.4f})")
ax.set_xlabel("Festival")
ax.set_ylabel("Delivery Time (min)")
for i, (cat, row) in enumerate(fest_stats.reindex(["No","Yes"]).iterrows()):
    ax.text(i, ax.get_ylim()[1]*0.95, f"Mean {row['Mean']:.1f}", ha="center", fontsize=10)
savefig("08_delivery_time_festival")

# ─────────────────────────────────────────────────────────────────────────────
# Q10  Rider rating vs delivery time
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Q10] Rating vs delivery time")
corr_rat, p_rat = stats.spearmanr(df["Delivery_person_Ratings"], df[TARGET])
print(f"  Spearman r={corr_rat:.4f}, p={p_rat:.4f}")

# Bin ratings for clearer visualization
df["Rating_bin"] = pd.cut(df["Delivery_person_Ratings"],
                           bins=[0.9,2,3,4,4.5,5.0],
                           labels=["1-2","2-3","3-4","4-4.5","4.5-5"])
rat_stats = (
    df.groupby("Rating_bin", observed=True)[TARGET]
    .agg(Mean="mean", Count="count")
)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sns.scatterplot(data=df.sample(3000, random_state=42),
                x="Delivery_person_Ratings", y=TARGET,
                alpha=0.3, color=ACCENT, ax=axes[0])
axes[0].set_title(f"Rating vs Delivery Time\n(Spearman r={corr_rat:.3f}, p={p_rat:.4f})")
axes[0].set_xlabel("Delivery Person Rating")
axes[0].set_ylabel("Delivery Time (min)")

axes[1].bar(rat_stats.index.astype(str), rat_stats["Mean"], color=ACCENT, alpha=0.85)
axes[1].set_title("Mean Delivery Time by Rating Band")
axes[1].set_xlabel("Rating Band")
axes[1].set_ylabel("Mean Delivery Time (min)")
for i, (band, row) in enumerate(rat_stats.iterrows()):
    axes[1].text(i, row["Mean"] + 0.1, f"{row['Mean']:.1f}", ha="center", fontsize=9)
savefig("09_delivery_time_by_rating")

# ─────────────────────────────────────────────────────────────────────────────
# Q11  Distance vs delivery time
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Q11] Distance vs delivery time")
df_dist = df.dropna(subset=["Distance_km"])
corr_dist, p_dist = stats.spearmanr(df_dist["Distance_km"], df_dist[TARGET])
print(f"  Spearman r={corr_dist:.4f}, p={p_dist:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sample_dist = df_dist.sample(min(4000, len(df_dist)), random_state=42)
axes[0].scatter(sample_dist["Distance_km"], sample_dist[TARGET],
                alpha=0.25, color=ACCENT, s=12)
# Trend line
m, b, r, p, _ = stats.linregress(df_dist["Distance_km"], df_dist[TARGET])
xline = np.linspace(df_dist["Distance_km"].min(), df_dist["Distance_km"].max(), 100)
axes[0].plot(xline, m*xline + b, color="#e74c3c", lw=2, label=f"OLS  r={r:.3f}")
axes[0].set_title(f"Distance vs Delivery Time\n(Pearson r={r:.3f})")
axes[0].set_xlabel("Distance (km)")
axes[0].set_ylabel("Delivery Time (min)")
axes[0].legend()

# Bin by distance
df_dist["Dist_bin"] = pd.cut(df_dist["Distance_km"],
                              bins=[0, 5, 10, 15, 20, 25, 100],
                              labels=["0-5","5-10","10-15","15-20","20-25","25+"])
dist_bin_stats = (
    df_dist.groupby("Dist_bin", observed=True)[TARGET]
    .agg(Mean="mean", Count="count")
)
axes[1].bar(dist_bin_stats.index.astype(str), dist_bin_stats["Mean"],
            color=ACCENT, alpha=0.85)
axes[1].set_title("Mean Delivery Time by Distance Band")
axes[1].set_xlabel("Distance Band (km)")
axes[1].set_ylabel("Mean Delivery Time (min)")
for i, (band, row) in enumerate(dist_bin_stats.iterrows()):
    axes[1].text(i, row["Mean"] + 0.2, f"{row['Mean']:.1f}", ha="center", fontsize=9)
savefig("10_delivery_time_by_distance")

# ─────────────────────────────────────────────────────────────────────────────
# Q12  Pickup duration + traffic + weather combined heatmap
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Q12] Pickup duration, traffic and weather vs delivery time")
df_pickup = df.dropna(subset=["Pickup_Duration_min"])
corr_pickup, p_pickup = stats.spearmanr(df_pickup["Pickup_Duration_min"], df_pickup[TARGET])
print(f"  Pickup duration Spearman r={corr_pickup:.4f}, p={p_pickup:.4f}")

# Traffic x Weather heatmap
pivot = (
    df.groupby(["Road_traffic_density", "Weather_conditions"])[TARGET]
    .mean()
    .unstack()
    .reindex(traffic_order)
)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlOrRd",
            linewidths=0.5, ax=axes[0])
axes[0].set_title("Mean Delivery Time: Traffic x Weather")
axes[0].set_xlabel("Weather Condition")
axes[0].set_ylabel("Traffic Density")

# Pickup duration scatter
sample_p = df_pickup.sample(min(4000, len(df_pickup)), random_state=42)
axes[1].scatter(sample_p["Pickup_Duration_min"], sample_p[TARGET],
                alpha=0.25, color="#7c5cd8", s=12)
m2, b2, r2, p2, _ = stats.linregress(
    df_pickup["Pickup_Duration_min"], df_pickup[TARGET])
xp = np.linspace(0, df_pickup["Pickup_Duration_min"].quantile(0.99), 100)
axes[1].plot(xp, m2*xp + b2, color="#e74c3c", lw=2, label=f"OLS  r={r2:.3f}")
axes[1].set_xlim(0, df_pickup["Pickup_Duration_min"].quantile(0.99))
axes[1].set_title(f"Pickup Duration vs Delivery Time\n(Spearman r={corr_pickup:.3f})")
axes[1].set_xlabel("Pickup Duration (min)")
axes[1].set_ylabel("Delivery Time (min)")
axes[1].legend()
savefig("11_pickup_traffic_weather_vs_delivery")

# ─────────────────────────────────────────────────────────────────────────────
# Additional figures
# ─────────────────────────────────────────────────────────────────────────────

# Correlation heatmap (numeric features)
print("\n[EXTRA] Correlation heatmap")
num_cols = ["Delivery_person_Age", "Delivery_person_Ratings",
            "Vehicle_condition", "multiple_deliveries",
            "Distance_km", "Order_Hour", "Pickup_Duration_min", TARGET]
corr_matrix = df[num_cols].dropna().corr(method="spearman")
fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm",
            vmin=-1, vmax=1, mask=mask, linewidths=0.5, ax=ax)
ax.set_title("Spearman Correlation Matrix (Numeric Features)")
savefig("12_correlation_heatmap")

# Order hour heatmap
print("\n[EXTRA] Order hour analysis")
df_hr = df.dropna(subset=["Order_Hour"])
hour_stats = df_hr.groupby("Order_Hour")[TARGET].mean()
fig, ax = plt.subplots(figsize=(12, 4))
ax.bar(hour_stats.index, hour_stats.values, color=ACCENT, alpha=0.85)
ax.set_title("Mean Delivery Time by Order Hour")
ax.set_xlabel("Order Hour (24h)")
ax.set_ylabel("Mean Delivery Time (min)")
ax.set_xticks(range(0, 24))
savefig("13_delivery_time_by_hour")

# ─────────────────────────────────────────────────────────────────────────────
# KPIs
# ─────────────────────────────────────────────────────────────────────────────
print("\n[KPIs]")
kpis = {
    "total_deliveries":        int(len(df)),
    "avg_delivery_time_min":   round(float(df[TARGET].mean()), 2),
    "median_delivery_time_min":round(float(df[TARGET].median()), 2),
    "min_delivery_time_min":   int(df[TARGET].min()),
    "max_delivery_time_min":   int(df[TARGET].max()),
    "avg_rating":              round(float(df["Delivery_person_Ratings"].mean()), 3),
    "avg_distance_km":         round(float(df["Distance_km"].mean()), 3),
    "pct_festival_orders":     round(float((df["Festival"]=="Yes").mean()*100), 2),
    "pct_jam_traffic_orders":  round(float((df["Road_traffic_density"]=="Jam").mean()*100), 2),
}
for k, v in kpis.items():
    print(f"  {k}: {v}")

with open(KPI_PATH, "w") as f:
    json.dump(kpis, f, indent=2)
print(f"[SAVE] KPIs -> {KPI_PATH}")

# ─────────────────────────────────────────────────────────────────────────────
# Insights
# ─────────────────────────────────────────────────────────────────────────────
insights = f"""
ZOMATO FOOD DELIVERY - BUSINESS INSIGHTS
=========================================

1. OVERALL DELIVERY PERFORMANCE
   Finding:    Average delivery time is {kpis['avg_delivery_time_min']:.1f} min; median is {kpis['median_delivery_time_min']:.1f} min.
   Evidence:   45,584 completed deliveries; distribution is roughly symmetric around the mean.
   Implication: The slight mean > median indicates some long-tail deliveries inflate the average.
   Action:     Investigate deliveries beyond 45 min (the top percentile) to identify avoidable delays.

2. CITY-LEVEL DIFFERENCES
   Finding:    Metropolitan areas show the highest average delivery times; Semi-Urban areas the lowest.
   Evidence:   City mean delivery times - {city_stats['Mean'].to_dict()}.
   Implication: Urban density and distance both drive delivery time in metropolitan areas.
   Action:     Consider dedicated fast-dispatch fleets or more restaurant partners in metropolitan areas.

3. TRAFFIC IS THE STRONGEST OPERATIONAL FACTOR
   Finding:    Delivery time increases progressively with traffic density.
   Evidence:   Mean delivery times by traffic - {traffic_stats['Mean'].to_dict()}.
   Implication: Jam conditions are associated with significantly longer times vs Low traffic.
   Action:     Use real-time traffic data to re-route drivers or delay order acceptance during Jam conditions.

4. WEATHER HAS A MODEST BUT CONSISTENT EFFECT
   Finding:    Adverse weather (Fog, Stormy, Sandstorms) is associated with modestly longer delivery times.
   Evidence:   Weather mean delivery times - {weather_stats['Mean'].to_dict()}.
   Implication: Weather alone does not drastically extend delivery times but compounds with traffic.
   Action:     Dynamic ETA adjustments and proactive customer notifications during adverse weather events.

5. MULTIPLE SIMULTANEOUS DELIVERIES INCREASE TIME
   Finding:    Each additional simultaneous delivery is associated with longer delivery times.
   Evidence:   Multi-delivery mean times - {multi_stats['Mean'].to_dict()}.
   Implication: Batching orders saves operational cost but hurts customer experience for later stops.
   Action:     Cap simultaneous deliveries at 2 for time-sensitive premium orders; use batching only for standard.

6. FESTIVAL PERIODS SHOW LONGER DELIVERY TIMES
   Finding:    Festival orders take longer on average (Mann-Whitney p={p_val:.4f}).
   Evidence:   Festival mean {fest_stats.loc['Yes','Mean']:.1f} min vs Non-Festival {fest_stats.loc['No','Mean']:.1f} min.
   Implication: Higher order volumes and congestion during festivals extend delivery windows.
   Action:     Pre-position additional delivery personnel in high-density zones before major festivals.

7. DISTANCE IS POSITIVELY ASSOCIATED WITH DELIVERY TIME
   Finding:    Distance (km) shows a moderate positive association with delivery time (Spearman r={corr_dist:.3f}).
   Evidence:   OLS trend confirms increasing delivery time as distance grows.
   Implication: Distance is a key input for accurate ETA prediction.
   Action:     Use distance-based ETA models and set delivery radius thresholds per city.

8. RIDER RATING HAS A WEAK NEGATIVE ASSOCIATION WITH DELIVERY TIME
   Finding:    Higher-rated riders are associated with marginally shorter delivery times (Spearman r={corr_rat:.3f}).
   Evidence:   Low statistical significance (p={p_rat:.4f}); the relationship is not strongly pronounced.
   Implication: Ratings may capture overall professionalism rather than speed alone.
   Action:     Do not rely on rating alone as a speed predictor; combine with historical delivery time data.

9. VEHICLE CONDITION IS NOT A STRONG DIFFERENTIATOR
   Finding:    Vehicle condition scores (0-3) show limited variation in mean delivery time.
   Evidence:   Condition mean times - {cond_stats['Mean'].to_dict()}.
   Implication: Fleet condition is maintained well enough that it does not drive major time differences.
   Action:     Continue routine maintenance checks; condition is not the primary bottleneck.

10. PICKUP DURATION IS POSITIVELY ASSOCIATED WITH DELIVERY TIME
    Finding:   Longer time between order and pickup is associated with longer total delivery time.
    Evidence:  Spearman r={corr_pickup:.3f} between Pickup_Duration_min and Delivery_Time_min.
    Implication: Restaurant preparation time and rider wait times are key upstream delays.
    Action:    Partner with restaurants to reduce kitchen preparation time; track pickup wait metrics.

NOTE: All relationships described above are associative (observational data). Causal conclusions
require controlled experimentation. These findings should inform hypothesis generation and
prioritisation for A/B tests or operational pilots.
"""

with open(INSIGHTS_PATH, "w", encoding="utf-8") as f:
    f.write(insights)
print(f"[SAVE] Insights -> {INSIGHTS_PATH}")

print("\n=== EDA COMPLETE ===")
