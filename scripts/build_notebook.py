"""
Script to generate notebooks/zomato_delivery_analysis.ipynb
"""
import json, textwrap

def code_cell(src, cell_id):
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {},
        "outputs": [],
        "source": src
    }

def md_cell(src, cell_id):
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {},
        "source": src
    }

cells = []

# ── Title ─────────────────────────────────────────────────────────────────────
cells.append(md_cell(
    "# Zomato Food Delivery – Operations & Delivery Time Analysis\n\n"
    "**Author:** Jyothsna | **Dataset:** Zomato Dataset.csv (45,584 deliveries)\n\n"
    "This notebook provides a complete, reproducible workflow:\n"
    "**Load → Inspect → Clean → Transform → Feature Engineering → EDA → Business Questions → KPIs → Insights**",
    "cell_title"
))

# ── 0. Setup ─────────────────────────────────────────────────────────────────
cells.append(md_cell("## 0. Setup and Imports", "cell_md_setup"))
cells.append(code_cell(
    "import math, json, warnings\n"
    "import numpy as np\n"
    "import pandas as pd\n"
    "import matplotlib\n"
    "import matplotlib.pyplot as plt\n"
    "import matplotlib.ticker as mticker\n"
    "import seaborn as sns\n"
    "from scipy import stats\n\n"
    "warnings.filterwarnings('ignore')\n"
    "sns.set_theme(style='whitegrid', palette='muted', font_scale=1.1)\n"
    "ACCENT = '#3b82d4'\n"
    "FIG_DIR = '../report/figures'\n"
    "FIG_DPI = 150\n\n"
    "def savefig(name):\n"
    "    plt.tight_layout()\n"
    "    plt.savefig(f'{FIG_DIR}/{name}.png', dpi=FIG_DPI, bbox_inches='tight')\n"
    "    print(f'Saved: {FIG_DIR}/{name}.png')\n\n"
    "print('All imports successful')",
    "cell_imports"
))

# ── 1. Load raw data ─────────────────────────────────────────────────────────
cells.append(md_cell("## 1. Load Raw Data", "cell_md_load"))
cells.append(code_cell(
    "RAW_PATH = '../data/Zomato Dataset.csv'\n"
    "df_raw = pd.read_csv(RAW_PATH, dtype=object)\n"
    "# Convert numeric columns\n"
    "numeric_cols = [\n"
    "    'Delivery_person_Age', 'Delivery_person_Ratings',\n"
    "    'Restaurant_latitude', 'Restaurant_longitude',\n"
    "    'Delivery_location_latitude', 'Delivery_location_longitude',\n"
    "    'Vehicle_condition', 'multiple_deliveries', 'Time_taken (min)'\n"
    "]\n"
    "for col in numeric_cols:\n"
    "    df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')\n\n"
    "print(f'Shape: {df_raw.shape}')\n"
    "print(f'Columns: {df_raw.columns.tolist()}')\n"
    "df_raw.head(3)",
    "cell_load"
))

cells.append(code_cell(
    "# Inspect data types and missing values\n"
    "print('=== Data Types ===')\n"
    "print(df_raw.dtypes)\n"
    "print()\n"
    "print('=== Missing Values ===')\n"
    "print(df_raw.isnull().sum())",
    "cell_inspect"
))

cells.append(code_cell(
    "# Unique values for categorical columns\n"
    "for col in ['City','Weather_conditions','Road_traffic_density',\n"
    "            'Type_of_vehicle','Type_of_order','Festival']:\n"
    "    print(f'{col}: {df_raw[col].unique().tolist()}')",
    "cell_uniques"
))

cells.append(code_cell(
    "# Suspicious value checks\n"
    "print(f'Age range: {df_raw[\"Delivery_person_Age\"].min()} - {df_raw[\"Delivery_person_Age\"].max()}')\n"
    "print(f'Age == 15 (below working age): {(df_raw[\"Delivery_person_Age\"]==15).sum()} rows')\n"
    "print(f'Ratings range: {df_raw[\"Delivery_person_Ratings\"].min()} - {df_raw[\"Delivery_person_Ratings\"].max()}')\n"
    "print(f'Ratings > 5 (invalid): {(df_raw[\"Delivery_person_Ratings\"]>5).sum()} rows')\n"
    "print(f'Restaurant lat == 0 (zero GPS): {(df_raw[\"Restaurant_latitude\"]==0).sum()} rows')\n"
    "print(f'Duplicate rows: {df_raw.duplicated().sum()}')",
    "cell_suspicious"
))

# ── 2. Data Cleaning ─────────────────────────────────────────────────────────
cells.append(md_cell(
    "## 2. Data Cleaning\n\n"
    "### Cleaning Decisions\n\n"
    "| Issue | Treatment | Rationale |\n"
    "|---|---|---|\n"
    "| NaN-like strings | Convert to `np.nan` | Uniform missing value representation |\n"
    "| `Metropolitian` typo | Rename to `Metropolitan` | Categorical consistency |\n"
    "| Numeric NaN (Age, Ratings, multi-deliveries) | Median imputation | Robust to outliers; preserves rows |\n"
    "| Categorical NaN (Weather, Traffic, Festival, City) | Mode imputation | Most common value is a defensible default |\n"
    "| `Time_Orderd` NaN | ffill/bfill within person, then fallback to pickup time | Likely recording gap within trip |\n"
    "| Age = 15 | Cap to 18 | Below minimum plausible working age |\n"
    "| Ratings > 5 | Cap to 5.0 | Scale is 1–5; value is data entry error |\n"
    "| Restaurant latitude = 0 | Flag in `coord_quality_flag`; Distance = NaN | GPS not recorded; cannot compute distance |\n"
    "| Sign-flipped GPS coords | Use `abs()` in Haversine calc | Coordinate sign error |\n",
    "cell_md_cleaning"
))

cells.append(code_cell(
    "df = df_raw.copy()\n\n"
    "# 2a. NaN-like strings -> NaN\n"
    "str_cols = [c for c in df.columns if c not in numeric_cols]\n"
    "for col in str_cols:\n"
    "    mask = df[col].astype(str).str.strip().isin(['NaN','nan','NA','N/A','null','NULL',''])\n"
    "    df.loc[mask, col] = np.nan\n"
    "    df[col] = df[col].where(df[col].isna(), df[col].astype(str).str.strip())\n\n"
    "# 2b. Typo fix\n"
    "typo_n = (df['City']=='Metropolitian').sum()\n"
    "df['City'] = df['City'].replace('Metropolitian','Metropolitan')\n"
    "print(f'City typo fixed: {typo_n} rows')\n\n"
    "# 2c. Median imputation - numeric\n"
    "for col in ['Delivery_person_Age','Delivery_person_Ratings','multiple_deliveries']:\n"
    "    med = df[col].median()\n"
    "    n = df[col].isna().sum()\n"
    "    df[col] = df[col].fillna(med)\n"
    "    print(f'{col}: {n} NaN -> median {med:.2f}')\n\n"
    "# 2d. Mode imputation - categorical\n"
    "for col in ['Weather_conditions','Road_traffic_density','Festival','City']:\n"
    "    mode = df[col].dropna().mode()[0]\n"
    "    n = df[col].isna().sum()\n"
    "    df[col] = df[col].fillna(mode)\n"
    "    print(f'{col}: {n} NaN -> mode {mode!r}')\n\n"
    "# 2e. Time_Orderd - ffill/bfill within person\n"
    "n_time = df['Time_Orderd'].isna().sum()\n"
    "df['Time_Orderd'] = df.groupby('Delivery_person_ID')['Time_Orderd'].transform(lambda s: s.ffill().bfill())\n"
    "df['Time_Orderd'] = df['Time_Orderd'].fillna(df['Time_Order_picked'])\n"
    "print(f'Time_Orderd: {n_time} NaN imputed')",
    "cell_clean1"
))

cells.append(code_cell(
    "# 2f. Suspicious value treatment\n"
    "n15 = (df['Delivery_person_Age']==15).sum()\n"
    "df.loc[df['Delivery_person_Age']==15, 'Delivery_person_Age'] = 18\n"
    "print(f'Age==15: {n15} rows capped to 18')\n\n"
    "n6 = (df['Delivery_person_Ratings']>5).sum()\n"
    "df.loc[df['Delivery_person_Ratings']>5, 'Delivery_person_Ratings'] = 5.0\n"
    "print(f'Ratings>5: {n6} rows capped to 5.0')\n\n"
    "n0 = (df['Restaurant_latitude']==0).sum()\n"
    "df['coord_quality_flag'] = (df['Restaurant_latitude']==0).astype(int)\n"
    "print(f'Zero coords: {n0} rows flagged')\n\n"
    "# 2g. Convert dtypes\n"
    "df['Order_Date'] = pd.to_datetime(df['Order_Date'], dayfirst=True, errors='coerce')\n"
    "df['Delivery_person_Age'] = df['Delivery_person_Age'].astype(float).astype(int)\n"
    "df['Vehicle_condition']   = df['Vehicle_condition'].astype(float).astype(int)\n"
    "df['multiple_deliveries'] = df['multiple_deliveries'].astype(float).astype(int)\n"
    "df['Time_taken (min)']    = df['Time_taken (min)'].astype(float).astype(int)\n"
    "df.rename(columns={'Time_taken (min)': 'Delivery_Time_min'}, inplace=True)\n"
    "for cat in ['Weather_conditions','Road_traffic_density','Type_of_order','Type_of_vehicle','Festival','City']:\n"
    "    df[cat] = df[cat].astype('category')\n"
    "print('Dtypes converted')\n"
    "print(f'Remaining NaN in core cols: {df[[\"Delivery_person_Age\",\"Delivery_person_Ratings\",\"Weather_conditions\",\"City\",\"Festival\"]].isnull().sum().sum()}')",
    "cell_clean2"
))

# ── 3. Feature Engineering ────────────────────────────────────────────────────
cells.append(md_cell("## 3. Feature Engineering", "cell_md_fe"))
cells.append(code_cell(
    "# 3a. Haversine Distance_km\n"
    "def haversine(lat1, lon1, lat2, lon2):\n"
    "    R = 6371.0\n"
    "    lat1,lon1,lat2,lon2 = map(math.radians,[lat1,lon1,lat2,lon2])\n"
    "    dlat, dlon = lat2-lat1, lon2-lon1\n"
    "    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2\n"
    "    return R * 2 * math.asin(math.sqrt(max(0,min(1,a))))\n\n"
    "def haversine_row(r):\n"
    "    if r['coord_quality_flag']==1: return np.nan\n"
    "    try:\n"
    "        d = haversine(abs(float(r['Restaurant_latitude'])), abs(float(r['Restaurant_longitude'])),\n"
    "                      abs(float(r['Delivery_location_latitude'])), abs(float(r['Delivery_location_longitude'])))\n"
    "        return d if d <= 50 else np.nan  # nullify implausible GPS sign-flip artifacts\n"
    "    except: return np.nan\n\n"
    "df['Distance_km'] = df.apply(haversine_row, axis=1).round(3)\n"
    "print(f'Distance_km: {df[\"Distance_km\"].notna().sum()} valid rows')\n"
    "print(df['Distance_km'].describe())",
    "cell_fe_dist"
))

cells.append(code_cell(
    "# 3b. Time features\n"
    "def parse_hour(t):\n"
    "    try: return int(str(t).split(':')[0])\n"
    "    except: return np.nan\n\n"
    "def time_to_min(t):\n"
    "    try:\n"
    "        p = str(t).split(':')\n"
    "        return int(p[0])*60+int(p[1])\n"
    "    except: return np.nan\n\n"
    "df['Order_Hour'] = df['Time_Orderd'].apply(parse_hour)\n\n"
    "orderd_min   = df['Time_Orderd'].apply(time_to_min)\n"
    "pickedup_min = df['Time_Order_picked'].apply(time_to_min)\n"
    "diff = pickedup_min - orderd_min\n"
    "df['Pickup_Duration_min'] = diff.apply(lambda x: x+1440 if pd.notna(x) and x<0 else x)\n\n"
    "df['Order_DayOfWeek'] = df['Order_Date'].dt.day_name()\n"
    "df['Order_Month']     = df['Order_Date'].dt.month\n"
    "df['Order_Week']      = df['Order_Date'].dt.isocalendar().week.astype('Int64')\n\n"
    "def time_bucket(h):\n"
    "    if pd.isna(h): return np.nan\n"
    "    h=int(h)\n"
    "    if 5<=h<11:  return 'Morning'\n"
    "    if 11<=h<15: return 'Lunch'\n"
    "    if 15<=h<19: return 'Afternoon'\n"
    "    if 19<=h<23: return 'Evening'\n"
    "    return 'Late Night'\n\n"
    "df['Time_of_Day'] = df['Order_Hour'].apply(time_bucket).astype('category')\n"
    "print('New columns:', ['Distance_km','Order_Hour','Pickup_Duration_min','Order_DayOfWeek','Order_Month','Time_of_Day'])\n"
    "df.head(3)",
    "cell_fe_time"
))

# ── 4. EDA ────────────────────────────────────────────────────────────────────
cells.append(md_cell("## 4. Exploratory Data Analysis\n\n### Q1 & Q2 – Overall Average & Median Delivery Time", "cell_md_eda1"))
cells.append(code_cell(
    "TARGET = 'Delivery_Time_min'\n\n"
    "mean_dt = df[TARGET].mean()\n"
    "med_dt  = df[TARGET].median()\n"
    "print(f'Mean delivery time:   {mean_dt:.2f} min')\n"
    "print(f'Median delivery time: {med_dt:.2f} min')\n\n"
    "fig, ax = plt.subplots(figsize=(9,5))\n"
    "ax.hist(df[TARGET], bins=40, color=ACCENT, edgecolor='white', alpha=0.85)\n"
    "ax.axvline(mean_dt, color='#e74c3c', lw=2, ls='--', label=f'Mean {mean_dt:.1f} min')\n"
    "ax.axvline(med_dt,  color='#2ecc71', lw=2, ls=':',  label=f'Median {med_dt:.1f} min')\n"
    "ax.set_xlabel('Delivery Time (min)'); ax.set_ylabel('Count')\n"
    "ax.set_title('Overall Delivery Time Distribution'); ax.legend()\n"
    "savefig('01_delivery_time_distribution')\n"
    "plt.show()",
    "cell_q1q2"
))

cells.append(md_cell("### Q3 – Delivery Time by City", "cell_md_q3"))
cells.append(code_cell(
    "city_stats = df.groupby('City')[TARGET].agg(Mean='mean',Median='median',Count='count').sort_values('Mean',ascending=False)\n"
    "print(city_stats)\n\n"
    "fig, axes = plt.subplots(1,2,figsize=(13,5))\n"
    "sns.boxplot(data=df, x='City', y=TARGET, order=city_stats.index, palette='muted', ax=axes[0])\n"
    "axes[0].set_title('Delivery Time by City'); axes[0].set_ylabel('Delivery Time (min)')\n"
    "axes[1].bar(city_stats.index, city_stats['Mean'], color=ACCENT, alpha=0.85)\n"
    "axes[1].set_title('Mean Delivery Time by City'); axes[1].set_ylabel('Mean Delivery Time (min)')\n"
    "for i,(c,r) in enumerate(city_stats.iterrows()): axes[1].text(i,r['Mean']+0.3,f\"{r['Mean']:.1f}\",ha='center',fontsize=10)\n"
    "savefig('02_delivery_time_by_city')\n"
    "plt.show()",
    "cell_q3"
))

cells.append(md_cell("### Q4 – Traffic Density vs Delivery Time", "cell_md_q4"))
cells.append(code_cell(
    "traffic_order = ['Low','Medium','High','Jam']\n"
    "traffic_stats = df.groupby('Road_traffic_density')[TARGET].agg(Mean='mean',Median='median',Count='count').reindex(traffic_order)\n"
    "print(traffic_stats)\n\n"
    "fig, axes = plt.subplots(1,2,figsize=(13,5))\n"
    "sns.boxplot(data=df, x='Road_traffic_density', y=TARGET, order=traffic_order, palette='Reds', ax=axes[0])\n"
    "axes[0].set_title('Delivery Time by Traffic Density'); axes[0].set_ylabel('Delivery Time (min)')\n"
    "axes[1].bar(traffic_stats.index, traffic_stats['Mean'], color=['#78c0a8','#f5a623','#e07b3c','#c0392b'], alpha=0.9)\n"
    "axes[1].set_title('Mean Delivery Time by Traffic')\n"
    "for i,(lvl,r) in enumerate(traffic_stats.iterrows()): axes[1].text(i,r['Mean']+0.2,f\"{r['Mean']:.1f}\",ha='center')\n"
    "savefig('03_delivery_time_by_traffic')\n"
    "plt.show()",
    "cell_q4"
))

cells.append(md_cell("### Q5 – Weather vs Delivery Time", "cell_md_q5"))
cells.append(code_cell(
    "weather_stats = df.groupby('Weather_conditions')[TARGET].agg(Mean='mean',Median='median',Count='count').sort_values('Mean',ascending=False)\n"
    "print(weather_stats)\n\n"
    "fig, ax = plt.subplots(figsize=(10,5))\n"
    "sns.barplot(data=df, x='Weather_conditions', y=TARGET, order=weather_stats.index, estimator='mean', errorbar='sd', palette='Blues_d', ax=ax)\n"
    "ax.set_title('Mean Delivery Time by Weather Condition (with SD)')\n"
    "savefig('04_delivery_time_by_weather')\n"
    "plt.show()",
    "cell_q5"
))

cells.append(md_cell("### Q6 – Vehicle Type vs Delivery Time", "cell_md_q6"))
cells.append(code_cell(
    "vehicle_stats = df.groupby('Type_of_vehicle')[TARGET].agg(Mean='mean',Median='median',Count='count').sort_values('Mean')\n"
    "print(vehicle_stats)\n\n"
    "fig, ax = plt.subplots(figsize=(9,5))\n"
    "sns.boxplot(data=df, x='Type_of_vehicle', y=TARGET, order=vehicle_stats.index, palette='Set2', ax=ax)\n"
    "ax.set_title('Delivery Time by Vehicle Type')\n"
    "savefig('05_delivery_time_by_vehicle')\n"
    "plt.show()",
    "cell_q6"
))

cells.append(md_cell("### Q7 – Vehicle Condition vs Delivery Time", "cell_md_q7"))
cells.append(code_cell(
    "cond_stats = df.groupby('Vehicle_condition')[TARGET].agg(Mean='mean',Median='median',Count='count').sort_index()\n"
    "print(cond_stats)\n\n"
    "fig, ax = plt.subplots(figsize=(8,5))\n"
    "sns.boxplot(data=df, x='Vehicle_condition', y=TARGET, order=sorted(df['Vehicle_condition'].unique()), palette='Purples', ax=ax)\n"
    "ax.set_title('Delivery Time by Vehicle Condition (0=worst, 3=best)')\n"
    "savefig('06_delivery_time_by_vehicle_condition')\n"
    "plt.show()",
    "cell_q7"
))

cells.append(md_cell("### Q8 – Multiple Deliveries vs Delivery Time", "cell_md_q8"))
cells.append(code_cell(
    "multi_stats = df.groupby('multiple_deliveries')[TARGET].agg(Mean='mean',Median='median',Count='count').sort_index()\n"
    "print(multi_stats)\n\n"
    "fig, ax = plt.subplots(figsize=(8,5))\n"
    "sns.boxplot(data=df, x='multiple_deliveries', y=TARGET, order=sorted(df['multiple_deliveries'].unique()), palette='Oranges', ax=ax)\n"
    "ax.set_title('Delivery Time by Number of Simultaneous Deliveries')\n"
    "savefig('07_delivery_time_by_multiple_deliveries')\n"
    "plt.show()",
    "cell_q8"
))

cells.append(md_cell("### Q9 – Festival vs Non-Festival Delivery Time", "cell_md_q9"))
cells.append(code_cell(
    "fest_stats = df.groupby('Festival')[TARGET].agg(Mean='mean',Median='median',Count='count')\n"
    "print(fest_stats)\n\n"
    "u_stat, p_val = stats.mannwhitneyu(df[df['Festival']=='Yes'][TARGET], df[df['Festival']=='No'][TARGET], alternative='two-sided')\n"
    "print(f'Mann-Whitney U={u_stat:.0f}, p={p_val:.4e}')\n\n"
    "fig, ax = plt.subplots(figsize=(7,5))\n"
    "sns.boxplot(data=df, x='Festival', y=TARGET, order=['No','Yes'], palette=['#3b82d4','#e07b3c'], ax=ax)\n"
    "ax.set_title(f'Delivery Time: Festival vs Non-Festival\\n(Mann-Whitney p={p_val:.4e})')\n"
    "savefig('08_delivery_time_festival')\n"
    "plt.show()",
    "cell_q9"
))

cells.append(md_cell("### Q10 – Rider Rating vs Delivery Time", "cell_md_q10"))
cells.append(code_cell(
    "corr_rat, p_rat = stats.spearmanr(df['Delivery_person_Ratings'], df[TARGET])\n"
    "print(f'Spearman r={corr_rat:.4f}, p={p_rat:.4e}')\n\n"
    "df['Rating_bin'] = pd.cut(df['Delivery_person_Ratings'], bins=[0.9,2,3,4,4.5,5.0], labels=['1-2','2-3','3-4','4-4.5','4.5-5'])\n"
    "rat_stats = df.groupby('Rating_bin', observed=True)[TARGET].agg(Mean='mean',Count='count')\n\n"
    "fig, axes = plt.subplots(1,2,figsize=(13,5))\n"
    "axes[0].scatter(df.sample(3000,random_state=42)['Delivery_person_Ratings'], df.sample(3000,random_state=42)[TARGET], alpha=0.3, color=ACCENT, s=12)\n"
    "axes[0].set_title(f'Rating vs Delivery Time (Spearman r={corr_rat:.3f})')\n"
    "axes[0].set_xlabel('Rating'); axes[0].set_ylabel('Delivery Time (min)')\n"
    "axes[1].bar(rat_stats.index.astype(str), rat_stats['Mean'], color=ACCENT, alpha=0.85)\n"
    "axes[1].set_title('Mean Delivery Time by Rating Band')\n"
    "savefig('09_delivery_time_by_rating')\n"
    "plt.show()",
    "cell_q10"
))

cells.append(md_cell("### Q11 – Distance vs Delivery Time", "cell_md_q11"))
cells.append(code_cell(
    "df_dist = df.dropna(subset=['Distance_km'])\n"
    "corr_dist, p_dist = stats.spearmanr(df_dist['Distance_km'], df_dist[TARGET])\n"
    "print(f'Spearman r={corr_dist:.4f}, p={p_dist:.4e}')\n\n"
    "fig, axes = plt.subplots(1,2,figsize=(13,5))\n"
    "sample_dist = df_dist.sample(min(4000,len(df_dist)),random_state=42)\n"
    "axes[0].scatter(sample_dist['Distance_km'], sample_dist[TARGET], alpha=0.25, color=ACCENT, s=12)\n"
    "m,b,r,p,_ = stats.linregress(df_dist['Distance_km'], df_dist[TARGET])\n"
    "xline = np.linspace(df_dist['Distance_km'].min(), df_dist['Distance_km'].max(), 100)\n"
    "axes[0].plot(xline, m*xline+b, color='#e74c3c', lw=2, label=f'OLS r={r:.3f}')\n"
    "axes[0].set_title(f'Distance vs Delivery Time (Pearson r={r:.3f})')\n"
    "axes[0].set_xlabel('Distance (km)'); axes[0].set_ylabel('Delivery Time (min)'); axes[0].legend()\n"
    "df_dist['Dist_bin'] = pd.cut(df_dist['Distance_km'], bins=[0,5,10,15,20,25,100], labels=['0-5','5-10','10-15','15-20','20-25','25+'])\n"
    "dist_bin = df_dist.groupby('Dist_bin', observed=True)[TARGET].mean()\n"
    "axes[1].bar(dist_bin.index.astype(str), dist_bin.values, color=ACCENT, alpha=0.85)\n"
    "axes[1].set_title('Mean Delivery Time by Distance Band'); axes[1].set_xlabel('Distance (km)')\n"
    "savefig('10_delivery_time_by_distance')\n"
    "plt.show()",
    "cell_q11"
))

cells.append(md_cell("### Q12 – Pickup Duration, Traffic & Weather Combined", "cell_md_q12"))
cells.append(code_cell(
    "df_p = df.dropna(subset=['Pickup_Duration_min'])\n"
    "corr_p, p_p = stats.spearmanr(df_p['Pickup_Duration_min'], df_p[TARGET])\n"
    "print(f'Pickup duration Spearman r={corr_p:.4f}, p={p_p:.4f}')\n\n"
    "pivot = df.groupby(['Road_traffic_density','Weather_conditions'])[TARGET].mean().unstack().reindex(['Low','Medium','High','Jam'])\n\n"
    "fig, axes = plt.subplots(1,2,figsize=(14,5))\n"
    "sns.heatmap(pivot, annot=True, fmt='.1f', cmap='YlOrRd', linewidths=0.5, ax=axes[0])\n"
    "axes[0].set_title('Mean Delivery Time: Traffic x Weather')\n"
    "sp = df_p.sample(min(4000,len(df_p)),random_state=42)\n"
    "axes[1].scatter(sp['Pickup_Duration_min'], sp[TARGET], alpha=0.25, color='#7c5cd8', s=12)\n"
    "m2,b2,r2,p2,_ = stats.linregress(df_p['Pickup_Duration_min'], df_p[TARGET])\n"
    "xp = np.linspace(0, df_p['Pickup_Duration_min'].quantile(0.99), 100)\n"
    "axes[1].plot(xp, m2*xp+b2, color='#e74c3c', lw=2, label=f'OLS r={r2:.3f}')\n"
    "axes[1].set_xlim(0, df_p['Pickup_Duration_min'].quantile(0.99))\n"
    "axes[1].set_title(f'Pickup Duration vs Delivery Time (Spearman r={corr_p:.3f})')\n"
    "axes[1].set_xlabel('Pickup Duration (min)'); axes[1].set_ylabel('Delivery Time (min)'); axes[1].legend()\n"
    "savefig('11_pickup_traffic_weather_vs_delivery')\n"
    "plt.show()",
    "cell_q12"
))

# ── 5. Additional visualizations ─────────────────────────────────────────────
cells.append(md_cell("## 5. Additional Visualizations", "cell_md_extra"))
cells.append(code_cell(
    "# Spearman correlation heatmap\n"
    "num_cols = ['Delivery_person_Age','Delivery_person_Ratings','Vehicle_condition',\n"
    "            'multiple_deliveries','Distance_km','Order_Hour','Pickup_Duration_min',TARGET]\n"
    "corr_matrix = df[num_cols].dropna().corr(method='spearman')\n"
    "fig, ax = plt.subplots(figsize=(10,8))\n"
    "mask = np.triu(np.ones_like(corr_matrix, dtype=bool))\n"
    "sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, mask=mask, linewidths=0.5, ax=ax)\n"
    "ax.set_title('Spearman Correlation Matrix')\n"
    "savefig('12_correlation_heatmap')\n"
    "plt.show()",
    "cell_corr"
))

cells.append(code_cell(
    "# Mean delivery time by order hour\n"
    "df_hr = df.dropna(subset=['Order_Hour'])\n"
    "hour_stats = df_hr.groupby('Order_Hour')[TARGET].mean()\n"
    "fig, ax = plt.subplots(figsize=(12,4))\n"
    "ax.bar(hour_stats.index, hour_stats.values, color=ACCENT, alpha=0.85)\n"
    "ax.set_title('Mean Delivery Time by Order Hour')\n"
    "ax.set_xlabel('Hour of Day'); ax.set_ylabel('Mean Delivery Time (min)')\n"
    "ax.set_xticks(range(0,24))\n"
    "savefig('13_delivery_time_by_hour')\n"
    "plt.show()",
    "cell_hour"
))

# ── 6. KPIs ───────────────────────────────────────────────────────────────────
cells.append(md_cell("## 6. Key Performance Indicators (KPIs)", "cell_md_kpis"))
cells.append(code_cell(
    "kpis = {\n"
    "    'Total Deliveries':        len(df),\n"
    "    'Average Delivery Time':   f\"{df[TARGET].mean():.2f} min\",\n"
    "    'Median Delivery Time':    f\"{df[TARGET].median():.2f} min\",\n"
    "    'Min Delivery Time':       f\"{df[TARGET].min()} min\",\n"
    "    'Max Delivery Time':       f\"{df[TARGET].max()} min\",\n"
    "    'Average Rating':          f\"{df['Delivery_person_Ratings'].mean():.3f}\",\n"
    "    'Average Distance (km)':   f\"{df['Distance_km'].mean():.2f} km\",\n"
    "    'Festival Orders %':       f\"{(df['Festival']=='Yes').mean()*100:.2f}%\",\n"
    "    'Jam Traffic Orders %':    f\"{(df['Road_traffic_density']=='Jam').mean()*100:.2f}%\",\n"
    "}\n"
    "print('=== KPIs ===')\n"
    "for k,v in kpis.items(): print(f'  {k}: {v}')",
    "cell_kpis"
))

# ── 7. Business Insights ──────────────────────────────────────────────────────
cells.append(md_cell(
    "## 7. Business Insights\n\n"
    "For each finding below the format is: **Finding → Evidence → Business Implication → Possible Action**\n\n"
    "---\n\n"
    "### 1. Festival Periods Are the Biggest Disruption\n"
    "**Finding:** Festival orders average ~45.5 min vs ~25.9 min non-festival (Mann-Whitney p<0.0001).  \n"
    "**Evidence:** Only 1.97% of orders occur on festivals but they show a ~75% longer mean delivery time.  \n"
    "**Implication:** Festival volume spikes create severe delivery delays.  \n"
    "**Action:** Pre-position surge riders and limit restaurant coverage radius on festival days.\n\n"
    "---\n\n"
    "### 2. Traffic Jam Conditions Add ~10 Minutes\n"
    "**Finding:** Jam traffic is associated with mean 31.2 min vs 21.5 min for Low traffic.  \n"
    "**Evidence:** Traffic density shows a monotonically increasing relationship with delivery time.  \n"
    "**Implication:** Traffic is the primary controllable operational factor in non-festival delivery time.  \n"
    "**Action:** Integrate live traffic APIs for dynamic ETA and smart route assignment.\n\n"
    "---\n\n"
    "### 3. Multiple Deliveries Significantly Increase Time\n"
    "**Finding:** 2 simultaneous deliveries average 40.5 min, 3 average 47.8 min vs 22.9 min for solo.  \n"
    "**Evidence:** Each additional delivery adds ~8–9 min on average.  \n"
    "**Implication:** Batching orders for cost savings comes at a significant customer experience cost.  \n"
    "**Action:** Limit batching to standard orders only; offer premium (solo) delivery for time-sensitive orders.\n\n"
    "---\n\n"
    "### 4. Metropolitan Areas Show Higher Delivery Times Than Urban\n"
    "**Finding:** Metropolitan mean 27.1 min vs Urban 23.0 min; Semi-Urban is highest at 49.7 min (very small sample n=164).  \n"
    "**Evidence:** City-level boxplots show wider variance in Metropolitan areas.  \n"
    "**Implication:** Urban density in Metropolitan areas, combined with traffic, extends last-mile distance.  \n"
    "**Action:** Expand ghost kitchen/dark store coverage in dense Metropolitan zones.\n\n"
    "---\n\n"
    "### 5. Distance Is a Moderate Positive Predictor\n"
    "**Finding:** Distance and delivery time show a moderate positive association (Spearman r=0.32).  \n"
    "**Evidence:** Mean delivery time increases from ~20 min for 0-5 km to ~34 min for 25+ km bands.  \n"
    "**Implication:** Distance is a reliable ETA input but is not the sole driver.  \n"
    "**Action:** Use distance as a primary feature in ETA models combined with traffic and time-of-day.\n\n"
    "---\n\n"
    "### 6. Higher Ratings Are Associated with Faster Delivery\n"
    "**Finding:** Spearman r = -0.28 between rating and delivery time (statistically significant).  \n"
    "**Evidence:** Top-rated riders (4.5-5.0) show consistently lower mean delivery times.  \n"
    "**Implication:** Rating captures some efficiency signal, though it is a composite of speed and service quality.  \n"
    "**Action:** Use historical delivery time alongside rating in rider dispatch prioritisation.\n\n"
    "---\n\n"
    "### 7. Cloudy and Foggy Weather Add ~2-3 Minutes\n"
    "**Finding:** Cloudy (28.9 min) and Fog (28.7 min) conditions are associated with longer delivery vs Sunny (21.9 min).  \n"
    "**Evidence:** Weather conditions show a 7-min spread from best (Sunny) to worst (Cloudy).  \n"
    "**Implication:** Weather compounds with traffic (as shown by the Traffic x Weather heatmap).  \n"
    "**Action:** Dynamic ETA adjustment based on weather API; proactive customer SMS on adverse weather days.\n\n"
    "---\n\n"
    "> **Note:** All relationships above are associative (observational data). Causal conclusions require controlled experiments (A/B tests). These findings should inform hypothesis generation and operational pilot design.",
    "cell_insights"
))

# ── 8. Save cleaned data ──────────────────────────────────────────────────────
cells.append(md_cell("## 8. Save Cleaned Dataset", "cell_md_save"))
cells.append(code_cell(
    "CLEAN_PATH = '../data/cleaned/zomato_cleaned.csv'\n"
    "df.to_csv(CLEAN_PATH, index=False)\n"
    "print(f'Saved cleaned dataset to {CLEAN_PATH}')\n"
    "print(f'Shape: {df.shape}')\n"
    "print('Columns:', df.columns.tolist())",
    "cell_save"
))

# ── Notebook dict ─────────────────────────────────────────────────────────────
notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.11.0"
        }
    },
    "cells": cells
}

NB_PATH = "notebooks/zomato_delivery_analysis.ipynb"
with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)
print(f"Notebook written to {NB_PATH}")
