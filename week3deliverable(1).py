import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# === CONFIGURATION ===
# 🔴 FIX: Your files are in 'experimental/', not 'data/'
DATA_DIR = Path("./experimental")   # ✅ Corrected path
OUTPUT_DIR = Path("./week3_output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Expected input files from Week 2 (names match your Explorer)
SOLD_FILE = DATA_DIR / "Sold_Residential_Merged.csv"        # ✅ Exact name
LISTINGS_FILE = DATA_DIR / "Listings_Residential_Merged.csv" # ✅ Exact name

# FRED Mortgage Rate URL
FRED_MORTGAGE_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=MORTGAGE30US"

# === STEP 1: LOAD WEEK 2 FILTERED DATASETS ===
print("📂 Loading Week 2 filtered datasets...")
if not SOLD_FILE.exists() or not LISTINGS_FILE.exists():
    raise FileNotFoundError(
        f"Missing Week 2 output files. Expected:\n  {SOLD_FILE}\n  {LISTINGS_FILE}\n"
        "Please ensure Week 2 residential-filtered CSVs are saved in ./experimental/"
    )

sold = pd.read_csv(SOLD_FILE, low_memory=False)
listings = pd.read_csv(LISTINGS_FILE, low_memory=False)
print(f"   Sold: {sold.shape[0]:,} rows | Listings: {listings.shape[0]:,} rows")

# === STEP 2: FETCH & RESAMPLE FRED MORTGAGE RATES ===
print("\n📉 Fetching FRED MORTGAGE30US series...")
mortgage = pd.read_csv(FRED_MORTGAGE_URL, parse_dates=['observation_date'])
mortgage.columns = ['date', 'rate_30yr_fixed']

# Resample weekly → monthly average
mortgage['year_month'] = mortgage['date'].dt.to_period('M')
mortgage_monthly = (
    mortgage.groupby('year_month')['rate_30yr_fixed']
    .mean()
    .reset_index()
)
# Convert Period to string for reliable merging
mortgage_monthly['year_month'] = mortgage_monthly['year_month'].astype(str)
print(f"   ✅ Monthly mortgage rates available: {mortgage_monthly['year_month'].min()} to {mortgage_monthly['year_month'].max()}")

# === STEP 3: CREATE YEAR_MONTH KEYS ON MLS DATASETS ===
print("\n🔗 Creating year_month join keys...")

# Sold dataset: key off CloseDate
sold['CloseDate_dt'] = pd.to_datetime(sold['CloseDate'], errors='coerce')
sold['year_month'] = sold['CloseDate_dt'].dt.to_period('M').astype(str)

# Listings dataset: key off ListingContractDate
listings['ListingContractDate_dt'] = pd.to_datetime(listings['ListingContractDate'], errors='coerce')
listings['year_month'] = listings['ListingContractDate_dt'].dt.to_period('M').astype(str)

# === STEP 4: MERGE MORTGAGE RATES ===
print("🔄 Merging mortgage rates onto MLS datasets...")
sold_enriched = sold.merge(mortgage_monthly, on='year_month', how='left')
listings_enriched = listings.merge(mortgage_monthly, on='year_month', how='left')

# === STEP 5: VALIDATION CHECKS ===
print("\n✅ VALIDATION REPORT")
print("=" * 50)

sold_nulls = sold_enriched['rate_30yr_fixed'].isnull().sum()
list_nulls = listings_enriched['rate_30yr_fixed'].isnull().sum()
sold_total = len(sold_enriched)
list_total = len(listings_enriched)

print(f"Sold Dataset:")
print(f"  Total rows:          {sold_total:,}")
print(f"  Null mortgage rates: {sold_nulls:,} ({sold_nulls/sold_total*100:.2f}%)")
print(f"  Match rate:          {(1 - sold_nulls/sold_total)*100:.2f}%")

print(f"\nListings Dataset:")
print(f"  Total rows:          {list_total:,}")
print(f"  Null mortgage rates: {list_nulls:,} ({list_nulls/list_total*100:.2f}%)")
print(f"  Match rate:          {(1 - list_nulls/list_total)*100:.2f}%")

# Preview merged data
print("\n📋 Sample Enriched Sold Records:")
preview_cols = ['CloseDate', 'year_month', 'ClosePrice', 'rate_30yr_fixed']
available_preview = [c for c in preview_cols if c in sold_enriched.columns]
print(sold_enriched[available_preview].dropna(subset=['rate_30yr_fixed']).head(10).to_string(index=False))

if sold_nulls > 0 or list_nulls > 0:
    print("\n⚠️  WARNING: Some records have null mortgage rates.")
    print("   This typically occurs for months outside the FRED series range.")
    print("   Review year_month values in unmatched records.")
else:
    print("\n🎉 PERFECT MERGE: Zero null mortgage rates in both datasets!")

# === STEP 6: SAVE ENRICHED DATASETS ===
sold_out = OUTPUT_DIR / "sold_enriched_with_mortgage.csv"
list_out = OUTPUT_DIR / "listings_enriched_with_mortgage.csv"

# Drop helper datetime columns before saving
sold_enriched.drop(columns=['CloseDate_dt'], inplace=True)
listings_enriched.drop(columns=['ListingContractDate_dt'], inplace=True)

sold_enriched.to_csv(sold_out, index=False)
listings_enriched.to_csv(list_out, index=False)

print(f"\n💾 Enriched datasets saved:")
print(f"   • {sold_out}")
print(f"   • {list_out}")
print("\n✅ Week 3 Deliverable Complete!")