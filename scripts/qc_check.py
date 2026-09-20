import os, json, nbformat

print("=== QUALITY CONTROL CHECKS ===")
print()

# 1. Raw dataset untouched
raw = "data/Zomato Dataset.csv"
if os.path.exists(raw):
    sz = os.path.getsize(raw)
    print(f"[OK] Raw dataset exists: {raw} ({sz:,} bytes)")
else:
    print("[FAIL] Raw dataset missing!")

# 2. Cleaned dataset exists
clean = "data/cleaned/zomato_cleaned.csv"
if os.path.exists(clean):
    import pandas as pd
    df = pd.read_csv(clean)
    print(f"[OK] Cleaned dataset: {clean} - shape {df.shape}")
    core_cols = ["Weather_conditions","Road_traffic_density","Festival","City",
                 "Delivery_person_Age","Delivery_person_Ratings"]
    core_missing = df[core_cols].isnull().sum().sum()
    print(f"[OK] Core column missing values: {core_missing} (should be 0)")
    tmin = df["Delivery_Time_min"].min()
    tmax = df["Delivery_Time_min"].max()
    print(f"[OK] Delivery_Time_min range: {tmin} - {tmax}")
    print(f"[OK] City values: {sorted(df['City'].dropna().unique().tolist())}")
    rmax = df["Delivery_person_Ratings"].max()
    amin = df["Delivery_person_Age"].min()
    davg = df["Distance_km"].mean()
    print(f"[OK] Ratings max: {rmax} (should be <= 5.0)")
    print(f"[OK] Age min: {amin} (should be >= 18)")
    print(f"[OK] Avg Distance_km: {davg:.2f} (should be ~9.7)")
else:
    print("[FAIL] Cleaned dataset missing!")

# 3. Figures
fig_dir = "report/figures"
figs = [f for f in os.listdir(fig_dir) if f.endswith(".png")]
print(f"[OK] Figures: {len(figs)} PNG files in {fig_dir}")
for f in sorted(figs):
    print(f"     {f}")

# 4. Notebook
nb_path = "notebooks/zomato_delivery_analysis.ipynb"
if os.path.exists(nb_path):
    nb = nbformat.read(nb_path, as_version=4)
    errors = [c for c in nb.cells if c.cell_type == "code" and
              any(o.get("output_type") == "error" for o in c.get("outputs", []))]
    code_cells_executed = sum(1 for c in nb.cells if c.cell_type == "code"
                              and c.get("execution_count") is not None)
    print(f"[OK] Notebook: {nb_path} - {len(nb.cells)} cells, {code_cells_executed} executed")
    if errors:
        print(f"[WARN] {len(errors)} cells with errors")
    else:
        print("[OK] No cell errors in notebook")
else:
    print("[FAIL] Notebook missing!")

# 5. KPIs
kpi_path = "report/kpis.json"
if os.path.exists(kpi_path):
    with open(kpi_path) as f:
        kpis = json.load(f)
    print(f"[OK] KPIs: {kpi_path}")
    for k, v in kpis.items():
        print(f"     {k}: {v}")
else:
    print("[FAIL] KPIs missing!")

# 6. Requirements
req = "requirements.txt"
if os.path.exists(req):
    lines = [l.strip() for l in open(req).readlines() if l.strip()]
    print(f"[OK] requirements.txt: {len(lines)} packages listed")
else:
    print("[FAIL] requirements.txt missing!")

# 7. README
readme = "README.md"
if os.path.exists(readme):
    content = open(readme, encoding="utf-8").read()
    sections = ["Problem Statement","Objective","Dataset","Methodology","Cleaning",
                "Feature Engineering","Key Findings","How to Run","Requirements","Project Structure"]
    missing = [s for s in sections if s not in content]
    if missing:
        print(f"[WARN] README missing sections: {missing}")
    else:
        print("[OK] README.md: all required sections present")
else:
    print("[FAIL] README.md missing!")

# 8. Report
report = "report/Zomato_Delivery_Analysis_Report.md"
if os.path.exists(report):
    content = open(report, encoding="utf-8").read()
    sections = ["Problem","Objective","Dataset","Cleaning","Analysis",
                "KPI","Insight","Recommendation","Conclusion"]
    missing = [s for s in sections if s not in content]
    if missing:
        print(f"[WARN] Report missing keywords: {missing}")
    else:
        print("[OK] Report: all required sections present")
else:
    print("[FAIL] Report missing!")

# 9. Log file
log = "data/cleaned/cleaning_log.txt"
if os.path.exists(log):
    lines = open(log, encoding="utf-8").readlines()
    print(f"[OK] Cleaning log: {len(lines)} lines")
else:
    print("[FAIL] Cleaning log missing!")

print()
print("=== QC COMPLETE ===")
