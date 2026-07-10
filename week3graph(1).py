import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# WEEK 3: Mortgage Rate Enrichment for MLS Datasets
# Goal:
# 1. Load Week 2 Residential-filtered MLS datasets
# 2. Fetch FRED 30-year fixed mortgage rate data
# 3. Convert weekly mortgage rates to monthly average rates
# 4. Merge monthly mortgage rates into sold and listings datasets
# 5. Validate merge quality
# 6. Save enriched CSVs and one visualization
# ============================================================

# === CONFIGURATION ===
DATA_DIR = Path("./experimental")
OUTPUT_DIR = Path("./week3_output")
FIGURE_DIR = OUTPUT_DIR / "figures"
REPORT_DIR = OUTPUT_DIR / "reports"

OUTPUT_DIR.mkdir(exist_ok=True)
FIGURE_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

SOLD_FILE = DATA_DIR / "Sold_Residential_Merged.csv"
LISTINGS_FILE = DATA_DIR / "Listings_Residential_Merged.csv"

FRED_MORTGAGE_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=MORTGAGE30US"
LOCAL_FRED_FILE = DATA_DIR / "MORTGAGE30US.csv"

RATE_COL = "rate_30yr_fixed"


# ============================================================
# Helper Functions
# ============================================================

def check_file_exists(file_path):
    """Check whether required input file exists."""
    if not file_path.exists():
        raise FileNotFoundError(
            f"Missing file: {file_path}\n"
            f"Please make sure the file is saved in: {DATA_DIR}"
        )


def require_columns(df, required_cols, dataset_name):
    """Check whether required columns exist in a dataframe."""
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise KeyError(
            f"{dataset_name} is missing required columns: {missing}\n"
            f"Available columns include: {list(df.columns[:30])}"
        )


def load_week2_datasets():
    """Load Week 2 Residential-filtered sold and listings datasets."""
    print("📂 STEP 1: Loading Week 2 Residential-filtered datasets...")

    check_file_exists(SOLD_FILE)
    check_file_exists(LISTINGS_FILE)

    sold_df = pd.read_csv(SOLD_FILE, low_memory=False)
    listings_df = pd.read_csv(LISTINGS_FILE, low_memory=False)

    require_columns(sold_df, ["CloseDate"], "Sold dataset")
    require_columns(listings_df, ["ListingContractDate"], "Listings dataset")

    print(f"   Sold rows:     {len(sold_df):,}")
    print(f"   Listings rows: {len(listings_df):,}")

    return sold_df, listings_df


def fetch_and_resample_mortgage_rates():
    """
    Fetch FRED MORTGAGE30US weekly mortgage rate data,
    then convert it to monthly average mortgage rates.
    """
    print("\n📉 STEP 2: Fetching FRED MORTGAGE30US mortgage rate data...")

    try:
        mortgage_raw = pd.read_csv(FRED_MORTGAGE_URL)
        source_used = "FRED online CSV"
    except Exception:
        if LOCAL_FRED_FILE.exists():
            mortgage_raw = pd.read_csv(LOCAL_FRED_FILE)
            source_used = f"Local fallback file: {LOCAL_FRED_FILE}"
        else:
            raise ConnectionError(
                "Could not fetch FRED data online, and no local fallback file was found.\n"
                f"Option: download MORTGAGE30US from FRED and save it as {LOCAL_FRED_FILE}"
            )

    print(f"   Source used: {source_used}")

    if "observation_date" not in mortgage_raw.columns:
        raise KeyError(
            "FRED file must contain an 'observation_date' column."
        )

    # FRED usually names the rate column as MORTGAGE30US
    rate_source_col = "MORTGAGE30US" if "MORTGAGE30US" in mortgage_raw.columns else mortgage_raw.columns[-1]

    mortgage = mortgage_raw.rename(
        columns={
            "observation_date": "date",
            rate_source_col: RATE_COL
        }
    )

    mortgage["date"] = pd.to_datetime(mortgage["date"], errors="coerce")
    mortgage[RATE_COL] = pd.to_numeric(mortgage[RATE_COL], errors="coerce")

    # Remove invalid FRED rows
    mortgage = mortgage.dropna(subset=["date", RATE_COL])

    # Convert weekly dates into monthly keys
    mortgage["year_month"] = mortgage["date"].dt.to_period("M").astype(str)

    # Monthly average mortgage rate
    mortgage_monthly = (
        mortgage
        .groupby("year_month", as_index=False)[RATE_COL]
        .mean()
    )

    if mortgage_monthly["year_month"].duplicated().any():
        raise ValueError("Duplicate year_month values found in mortgage_monthly.")

    print(
        f"   Monthly mortgage rate range: "
        f"{mortgage_monthly['year_month'].min()} to {mortgage_monthly['year_month'].max()}"
    )
    print(f"   Monthly rate rows: {len(mortgage_monthly):,}")

    return mortgage_monthly


def add_year_month_key(df, date_col, dataset_name):
    """
    Create a year_month key from a date column.
    This key allows monthly mortgage rates to merge into MLS records.
    """
    print(f"\n🔗 STEP 3: Creating year_month key for {dataset_name} using {date_col}...")

    require_columns(df, [date_col], dataset_name)

    output = df.copy()
    helper_col = f"{date_col}_dt"

    output[helper_col] = pd.to_datetime(output[date_col], errors="coerce")
    output["year_month"] = output[helper_col].dt.to_period("M").astype(str)

    # Avoid keeping 'NaT' as a fake month string
    output.loc[output[helper_col].isna(), "year_month"] = np.nan

    invalid_dates = output[helper_col].isna().sum()

    print(f"   Invalid or missing {date_col}: {invalid_dates:,}")
    print(f"   Unique valid months: {output['year_month'].nunique(dropna=True):,}")

    return output, helper_col


def merge_mortgage_rates(mls_df, mortgage_monthly, dataset_name):
    """
    Left merge mortgage rates onto MLS data using year_month.
    Row count should remain exactly the same after merge.
    """
    print(f"\n🔄 STEP 4: Merging monthly mortgage rates into {dataset_name}...")

    before_rows = len(mls_df)

    enriched = mls_df.merge(
        mortgage_monthly,
        on="year_month",
        how="left"
    )

    after_rows = len(enriched)

    if before_rows != after_rows:
        raise ValueError(
            f"Row count changed after merge for {dataset_name}.\n"
            f"Before: {before_rows:,}, After: {after_rows:,}"
        )

    print(f"   Row count preserved: {after_rows:,}")

    return enriched


def create_validation_report(df, dataset_name, helper_date_col):
    """Create validation metrics for merge completeness."""
    total_rows = len(df)
    null_rate_rows = df[RATE_COL].isna().sum()
    invalid_date_rows = df[helper_date_col].isna().sum()
    matched_rows = total_rows - null_rate_rows
    match_rate = matched_rows / total_rows * 100 if total_rows > 0 else 0

    return {
        "dataset": dataset_name,
        "total_rows": total_rows,
        "matched_rate_rows": matched_rows,
        "null_rate_rows": null_rate_rows,
        "invalid_date_rows": invalid_date_rows,
        "match_rate_percent": round(match_rate, 2)
    }


def save_unmatched_months(sold_enriched, listings_enriched):
    """Save unmatched year_month values for review."""
    sold_unmatched = sorted(
        sold_enriched.loc[
            sold_enriched[RATE_COL].isna() & sold_enriched["year_month"].notna(),
            "year_month"
        ].unique()
    )

    listings_unmatched = sorted(
        listings_enriched.loc[
            listings_enriched[RATE_COL].isna() & listings_enriched["year_month"].notna(),
            "year_month"
        ].unique()
    )

    unmatched_df = pd.concat(
        [
            pd.Series(sold_unmatched, name="sold_unmatched_year_month"),
            pd.Series(listings_unmatched, name="listings_unmatched_year_month")
        ],
        axis=1
    )

    unmatched_path = REPORT_DIR / "unmatched_year_months.csv"
    unmatched_df.to_csv(unmatched_path, index=False)

    print(f"\n🧾 Unmatched month report saved: {unmatched_path}")

    return unmatched_path


def create_mortgage_volume_chart(sold_enriched, listings_enriched, mortgage_monthly):
    """
    Generate one image showing:
    1. Monthly average 30-year fixed mortgage rate
    2. Monthly sold record count
    3. Monthly listing record count
    """
    print("\n🖼️ STEP 6: Creating mortgage rate and MLS volume chart...")

    sold_counts = (
        sold_enriched
        .dropna(subset=["year_month"])
        .groupby("year_month")
        .size()
        .reset_index(name="sold_records")
    )

    listings_counts = (
        listings_enriched
        .dropna(subset=["year_month"])
        .groupby("year_month")
        .size()
        .reset_index(name="listing_records")
    )

    plot_df = (
        mortgage_monthly
        .merge(sold_counts, on="year_month", how="outer")
        .merge(listings_counts, on="year_month", how="outer")
    )

    plot_df["month_dt"] = pd.to_datetime(plot_df["year_month"] + "-01", errors="coerce")
    plot_df = plot_df.dropna(subset=["month_dt"]).sort_values("month_dt")

    # Keep only the MLS date range, not the entire FRED history
    mls_months = pd.concat(
        [
            sold_enriched["year_month"].dropna(),
            listings_enriched["year_month"].dropna()
        ]
    )

    if mls_months.empty:
        raise ValueError("No valid MLS year_month values available for plotting.")

    start_month = pd.to_datetime(mls_months.min() + "-01")
    end_month = pd.to_datetime(mls_months.max() + "-01")

    plot_df = plot_df[
        (plot_df["month_dt"] >= start_month) &
        (plot_df["month_dt"] <= end_month)
    ]

    if plot_df.empty:
        raise ValueError("No overlapping months available for mortgage rate chart.")

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Left axis: mortgage rate
    ax1.plot(
        plot_df["month_dt"],
        plot_df[RATE_COL],
        marker="o",
        linewidth=2,
        label="30-Year Fixed Mortgage Rate (%)"
    )
    ax1.set_xlabel("Month")
    ax1.set_ylabel("Mortgage Rate (%)")
    ax1.grid(True, axis="y", alpha=0.3)

    # Right axis: MLS record counts
    ax2 = ax1.twinx()
    ax2.bar(
        plot_df["month_dt"],
        plot_df["sold_records"],
        width=20,
        alpha=0.25,
        label="Sold Records"
    )
    ax2.plot(
        plot_df["month_dt"],
        plot_df["listing_records"],
        marker="x",
        linestyle="--",
        linewidth=2,
        label="Listing Records"
    )
    ax2.set_ylabel("MLS Record Count")

    plt.title("Monthly Mortgage Rate and MLS Residential Activity")

    # Combine legends from both axes
    handles_1, labels_1 = ax1.get_legend_handles_labels()
    handles_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(handles_1 + handles_2, labels_1 + labels_2, loc="best")

    fig.autofmt_xdate()
    plt.tight_layout()

    figure_path = FIGURE_DIR / "mortgage_rate_and_mls_volume.png"
    plt.savefig(figure_path, dpi=300)
    plt.close()

    print(f"   Chart saved: {figure_path}")

    return figure_path


# ============================================================
# Main Script
# ============================================================

print("=" * 70)
print("WEEK 3 DELIVERABLE: Mortgage Rate Enrichment")
print("=" * 70)

# Step 1: Load MLS data from Week 2
sold, listings = load_week2_datasets()

# Step 2: Fetch and resample FRED mortgage rates
mortgage_monthly = fetch_and_resample_mortgage_rates()

# Step 3: Create year_month keys
sold_keyed, sold_helper_col = add_year_month_key(
    sold,
    date_col="CloseDate",
    dataset_name="Sold dataset"
)

listings_keyed, listings_helper_col = add_year_month_key(
    listings,
    date_col="ListingContractDate",
    dataset_name="Listings dataset"
)

# Step 4: Merge mortgage rates
sold_enriched = merge_mortgage_rates(
    sold_keyed,
    mortgage_monthly,
    dataset_name="Sold dataset"
)

listings_enriched = merge_mortgage_rates(
    listings_keyed,
    mortgage_monthly,
    dataset_name="Listings dataset"
)

# Step 5: Validation report
print("\n✅ STEP 5: Validation report")
print("=" * 70)

validation_report = pd.DataFrame(
    [
        create_validation_report(sold_enriched, "Sold dataset", sold_helper_col),
        create_validation_report(listings_enriched, "Listings dataset", listings_helper_col)
    ]
)

print(validation_report.to_string(index=False))

validation_report_path = REPORT_DIR / "mortgage_merge_validation_report.csv"
validation_report.to_csv(validation_report_path, index=False)

print(f"\nValidation report saved: {validation_report_path}")

# Save unmatched months for review
unmatched_path = save_unmatched_months(sold_enriched, listings_enriched)

# Preview sample sold records
print("\n📋 Sample enriched sold records:")

preview_cols = [
    "CloseDate",
    "year_month",
    "ClosePrice",
    RATE_COL
]

available_preview_cols = [col for col in preview_cols if col in sold_enriched.columns]

print(
    sold_enriched[available_preview_cols]
    .dropna(subset=[RATE_COL])
    .head(10)
    .to_string(index=False)
)

# Step 6: Create one image output
figure_path = create_mortgage_volume_chart(
    sold_enriched,
    listings_enriched,
    mortgage_monthly
)

# Step 7: Save enriched CSV files
print("\n💾 STEP 7: Saving enriched datasets...")

sold_out = OUTPUT_DIR / "sold_enriched_with_mortgage.csv"
listings_out = OUTPUT_DIR / "listings_enriched_with_mortgage.csv"

# Drop helper datetime columns before final CSV output
sold_final = sold_enriched.drop(columns=[sold_helper_col], errors="ignore")
listings_final = listings_enriched.drop(columns=[listings_helper_col], errors="ignore")

sold_final.to_csv(sold_out, index=False)
listings_final.to_csv(list_out := listings_out, index=False)

print(f"   Sold enriched CSV saved:     {sold_out}")
print(f"   Listings enriched CSV saved: {list_out}")
print(f"   Figure saved:                {figure_path}")
print(f"   Validation report saved:     {validation_report_path}")
print(f"   Unmatched month report saved:{unmatched_path}")

# Final warning if null mortgage rates exist
total_null_rates = (
    sold_final[RATE_COL].isna().sum() +
    listings_final[RATE_COL].isna().sum()
)

if total_null_rates == 0:
    print("\n🎉 PERFECT MERGE: Zero null mortgage rates in both datasets.")
else:
    print("\n⚠️ WARNING: Some records still have null mortgage rates.")
    print("   Check reports/unmatched_year_months.csv.")
    print("   Common reasons: invalid dates, missing dates, or MLS months outside FRED range.")

print("\n✅ Week 3 Deliverable Complete!")