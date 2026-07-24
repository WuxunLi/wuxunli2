import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# WEEK 5: Final Cleaning and Analysis-Ready Dataset Preparation
#
# Goal:
# 1. Load Week 4 flagged datasets
# 2. Reconfirm date and numeric data types
# 3. Recreate key validation flags if needed
# 4. Apply final cleaning rules
# 5. Save final analysis-ready datasets
# 6. Save final validation reports
#
# No chart output is required for Week 5.
# ============================================================


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_DIRS = [
    Path("./week4_output"),
    Path("./week3_output"),
    Path("./experimental"),
    Path(".")
]

OUTPUT_DIR = Path("./week5_output")
REPORT_DIR = OUTPUT_DIR / "reports"

OUTPUT_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Prefer Week 4 flagged files.
SOLD_PATTERNS = [
    "sold_week4_flagged_*.csv",
    "sold_week4_clean_*.csv",
    "sold_enriched_with_mortgage*.csv",
    "Sold_Residential_Merged.csv"
]

LISTINGS_PATTERNS = [
    "listings_week4_flagged_*.csv",
    "listings_week4_clean_*.csv",
    "listings_enriched_with_mortgage*.csv",
    "Listings_Residential_Merged.csv"
]


# ============================================================
# FIELD GROUPS
# ============================================================

DATE_COLS = [
    "CloseDate",
    "PurchaseContractDate",
    "ListingContractDate",
    "ContractStatusChangeDate"
]

NUMERIC_COLS = [
    "ClosePrice",
    "ListPrice",
    "OriginalListPrice",
    "DaysOnMarket",
    "LotSizeAcres",
    "BathroomsTotalInteger",
    "YearBuilt",
    "LivingArea",
    "BedroomsTotal",
    "Longitude",
    "Latitude",
    "rate_30yr_fixed"
]

SOLD_CRITICAL_FIELDS = [
    "ClosePrice",
    "OriginalListPrice",
    "LivingArea",
    "DaysOnMarket",
    "ListingContractDate",
    "CloseDate"
]

LISTINGS_CRITICAL_FIELDS = [
    "ListPrice",
    "LivingArea",
    "ListingContractDate"
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def find_preferred_file(patterns, input_dirs, dataset_name):
    """
    Find input file by priority order.
    This means the script uses Week 4 flagged files first if available.
    """
    for pattern in patterns:
        candidates = []

        for folder in input_dirs:
            if folder.exists():
                candidates.extend(folder.glob(pattern))

        candidates = [file for file in candidates if file.is_file()]

        if candidates:
            latest_file = max(candidates, key=lambda file: file.stat().st_mtime)
            print(f"   {dataset_name} input file found: {latest_file}")
            return latest_file

    raise FileNotFoundError(
        f"Could not find input file for {dataset_name}.\n"
        f"Checked folders: {[str(folder) for folder in input_dirs]}\n"
        f"Checked patterns: {patterns}"
    )


def standardize_column_names(df):
    """
    Fix small column-name inconsistencies.
    """
    output = df.copy()

    rename_map = {}

    if "PurchasecontractDate" in output.columns and "PurchaseContractDate" not in output.columns:
        rename_map["PurchasecontractDate"] = "PurchaseContractDate"

    if "CountryOrParish" in output.columns and "CountyOrParish" not in output.columns:
        rename_map["CountryOrParish"] = "CountyOrParish"

    if rename_map:
        output = output.rename(columns=rename_map)

    return output, rename_map


def clean_numeric_text(series):
    """
    Convert messy numeric text into clean numeric values.
    """
    return (
        series
        .astype("string")
        .str.replace(",", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.strip()
        .replace(["", "nan", "None", "NaN", "<NA>"], np.nan)
    )


def confirm_date_types(df, dataset_name):
    """
    Convert date columns into datetime format and report invalid dates.
    """
    output = df.copy()
    rows = []

    print(f"\n📅 Confirming date fields for {dataset_name}...")

    for col in DATE_COLS:
        if col in output.columns:
            output[col] = pd.to_datetime(output[col], errors="coerce")

            invalid_count = output[col].isna().sum()
            invalid_percent = round(invalid_count / len(output) * 100, 2) if len(output) > 0 else 0

            rows.append({
                "dataset": dataset_name,
                "column": col,
                "dtype_after": str(output[col].dtype),
                "missing_or_invalid_count": int(invalid_count),
                "missing_or_invalid_percent": invalid_percent
            })

            print(f"   {col}: {output[col].dtype}, invalid/missing = {invalid_count:,}")

    return output, pd.DataFrame(rows)


def confirm_numeric_types(df, dataset_name):
    """
    Convert numeric columns into numeric format and report invalid values.
    """
    output = df.copy()
    rows = []

    print(f"\n🔢 Confirming numeric fields for {dataset_name}...")

    for col in NUMERIC_COLS:
        if col in output.columns:
            output[col] = clean_numeric_text(output[col])
            output[col] = pd.to_numeric(output[col], errors="coerce")

            invalid_count = output[col].isna().sum()
            invalid_percent = round(invalid_count / len(output) * 100, 2) if len(output) > 0 else 0

            rows.append({
                "dataset": dataset_name,
                "column": col,
                "dtype_after": str(output[col].dtype),
                "missing_or_invalid_count": int(invalid_count),
                "missing_or_invalid_percent": invalid_percent
            })

            print(f"   {col}: {output[col].dtype}, invalid/missing = {invalid_count:,}")

    return output, pd.DataFrame(rows)


def add_final_numeric_flags(df):
    """
    Add final invalid numeric flags.
    These flags identify records that should not be used in final market analysis.
    """
    output = df.copy()

    output["final_invalid_close_price_flag"] = False
    if "ClosePrice" in output.columns:
        output["final_invalid_close_price_flag"] = (
            output["ClosePrice"].isna() |
            (output["ClosePrice"] <= 0)
        )

    output["final_invalid_list_price_flag"] = False
    if "ListPrice" in output.columns:
        output["final_invalid_list_price_flag"] = (
            output["ListPrice"].isna() |
            (output["ListPrice"] <= 0)
        )

    output["final_invalid_original_list_price_flag"] = False
    if "OriginalListPrice" in output.columns:
        output["final_invalid_original_list_price_flag"] = (
            output["OriginalListPrice"].isna() |
            (output["OriginalListPrice"] <= 0)
        )

    output["final_invalid_living_area_flag"] = False
    if "LivingArea" in output.columns:
        output["final_invalid_living_area_flag"] = (
            output["LivingArea"].isna() |
            (output["LivingArea"] <= 0)
        )

    output["final_invalid_days_on_market_flag"] = False
    if "DaysOnMarket" in output.columns:
        output["final_invalid_days_on_market_flag"] = (
            output["DaysOnMarket"].notna() &
            (output["DaysOnMarket"] < 0)
        )

    output["final_invalid_bedrooms_flag"] = False
    if "BedroomsTotal" in output.columns:
        output["final_invalid_bedrooms_flag"] = (
            output["BedroomsTotal"].notna() &
            (output["BedroomsTotal"] < 0)
        )

    output["final_invalid_bathrooms_flag"] = False
    if "BathroomsTotalInteger" in output.columns:
        output["final_invalid_bathrooms_flag"] = (
            output["BathroomsTotalInteger"].notna() &
            (output["BathroomsTotalInteger"] < 0)
        )

    output["final_invalid_year_built_flag"] = False
    if "YearBuilt" in output.columns:
        output["final_invalid_year_built_flag"] = (
            output["YearBuilt"].notna() &
            (
                (output["YearBuilt"] < 1800) |
                (output["YearBuilt"] > 2035)
            )
        )

    return output


def add_final_date_flags(df):
    """
    Add final date consistency flags.
    """
    output = df.copy()

    output["final_listing_after_close_flag"] = False
    output["final_purchase_after_close_flag"] = False
    output["final_negative_timeline_flag"] = False

    if "ListingContractDate" in output.columns and "CloseDate" in output.columns:
        valid = output["ListingContractDate"].notna() & output["CloseDate"].notna()
        output.loc[valid, "final_listing_after_close_flag"] = (
            output.loc[valid, "ListingContractDate"] >
            output.loc[valid, "CloseDate"]
        )

    if "PurchaseContractDate" in output.columns and "CloseDate" in output.columns:
        valid = output["PurchaseContractDate"].notna() & output["CloseDate"].notna()
        output.loc[valid, "final_purchase_after_close_flag"] = (
            output.loc[valid, "PurchaseContractDate"] >
            output.loc[valid, "CloseDate"]
        )

    if "ListingContractDate" in output.columns and "PurchaseContractDate" in output.columns:
        valid = output["ListingContractDate"].notna() & output["PurchaseContractDate"].notna()
        output.loc[valid, "final_negative_timeline_flag"] = (
            output.loc[valid, "ListingContractDate"] >
            output.loc[valid, "PurchaseContractDate"]
        )

    if "PurchaseContractDate" in output.columns and "CloseDate" in output.columns:
        valid = output["PurchaseContractDate"].notna() & output["CloseDate"].notna()
        output.loc[valid, "final_negative_timeline_flag"] = (
            output.loc[valid, "final_negative_timeline_flag"] |
            (
                output.loc[valid, "PurchaseContractDate"] >
                output.loc[valid, "CloseDate"]
            )
        )

    if "ListingContractDate" in output.columns and "CloseDate" in output.columns:
        valid = output["ListingContractDate"].notna() & output["CloseDate"].notna()
        output.loc[valid, "final_negative_timeline_flag"] = (
            output.loc[valid, "final_negative_timeline_flag"] |
            (
                output.loc[valid, "ListingContractDate"] >
                output.loc[valid, "CloseDate"]
            )
        )

    return output


def add_final_geo_flags(df):
    """
    Add final geographic quality flags.
    California coordinates should usually have:
    Latitude between about 32 and 42.5
    Longitude between about -125 and -113
    """
    output = df.copy()

    if "Latitude" in output.columns and "Longitude" in output.columns:
        output["final_missing_coordinates_flag"] = (
            output["Latitude"].isna() |
            output["Longitude"].isna()
        )

        output["final_zero_coordinates_flag"] = (
            output["Latitude"].eq(0) |
            output["Longitude"].eq(0)
        )

        output["final_positive_longitude_flag"] = output["Longitude"] > 0

        valid_coords = (
            output["Latitude"].notna() &
            output["Longitude"].notna() &
            ~output["Latitude"].eq(0) &
            ~output["Longitude"].eq(0)
        )

        output["final_implausible_ca_coordinates_flag"] = False

        output.loc[valid_coords, "final_implausible_ca_coordinates_flag"] = (
            (output.loc[valid_coords, "Latitude"] < 32.0) |
            (output.loc[valid_coords, "Latitude"] > 42.5) |
            (output.loc[valid_coords, "Longitude"] < -125.0) |
            (output.loc[valid_coords, "Longitude"] > -113.0)
        )

        output["final_geo_usable_flag"] = ~(
            output["final_missing_coordinates_flag"] |
            output["final_zero_coordinates_flag"] |
            output["final_positive_longitude_flag"] |
            output["final_implausible_ca_coordinates_flag"]
        )

    else:
        output["final_missing_coordinates_flag"] = True
        output["final_zero_coordinates_flag"] = False
        output["final_positive_longitude_flag"] = False
        output["final_implausible_ca_coordinates_flag"] = False
        output["final_geo_usable_flag"] = False

    return output


def create_final_clean_mask(df, dataset_name):
    """
    Create final clean mask.
    Sold data uses sold transaction fields.
    Listings data uses listing fields.
    """
    if dataset_name.lower() == "sold":
        required_columns = SOLD_CRITICAL_FIELDS

        required_missing_flags = []
        for col in required_columns:
            if col in df.columns:
                required_missing_flags.append(df[col].isna())

        invalid_flags = [
            "final_invalid_close_price_flag",
            "final_invalid_original_list_price_flag",
            "final_invalid_living_area_flag",
            "final_invalid_days_on_market_flag",
            "final_invalid_bedrooms_flag",
            "final_invalid_bathrooms_flag",
            "final_invalid_year_built_flag",
            "final_listing_after_close_flag",
            "final_purchase_after_close_flag",
            "final_negative_timeline_flag"
        ]

    else:
        required_columns = LISTINGS_CRITICAL_FIELDS

        required_missing_flags = []
        for col in required_columns:
            if col in df.columns:
                required_missing_flags.append(df[col].isna())

        invalid_flags = [
            "final_invalid_list_price_flag",
            "final_invalid_living_area_flag",
            "final_invalid_bedrooms_flag",
            "final_invalid_bathrooms_flag",
            "final_invalid_year_built_flag"
        ]

    existing_invalid_flags = [flag for flag in invalid_flags if flag in df.columns]

    if required_missing_flags:
        missing_required_mask = required_missing_flags[0]
        for mask in required_missing_flags[1:]:
            missing_required_mask = missing_required_mask | mask
    else:
        missing_required_mask = pd.Series(False, index=df.index)

    if existing_invalid_flags:
        invalid_mask = df[existing_invalid_flags].any(axis=1)
    else:
        invalid_mask = pd.Series(False, index=df.index)

    final_clean_mask = ~(missing_required_mask | invalid_mask)

    return final_clean_mask


def create_exclusion_reason_summary(df, clean_mask, dataset_name):
    """
    Count why records were excluded from the final clean dataset.
    """
    excluded = df[~clean_mask].copy()

    rows = []

    if excluded.empty:
        return pd.DataFrame([{
            "dataset": dataset_name,
            "exclusion_reason": "No excluded records",
            "excluded_rows": 0,
            "excluded_percent_of_total": 0
        }])

    total_rows = len(df)

    flag_cols = [
        col for col in excluded.columns
        if col.startswith("final_") and col.endswith("_flag")
    ]

    for col in flag_cols:
        count = int(excluded[col].sum())
        percent = round(count / total_rows * 100, 2) if total_rows > 0 else 0

        rows.append({
            "dataset": dataset_name,
            "exclusion_reason": col,
            "excluded_rows": count,
            "excluded_percent_of_total": percent
        })

    return pd.DataFrame(rows).sort_values("excluded_rows", ascending=False)


def create_flag_count_summary(df, dataset_name):
    """
    Summarize all final flag counts.
    """
    flag_cols = [
        col for col in df.columns
        if col.startswith("final_") and col.endswith("_flag")
    ]

    rows = []

    for col in flag_cols:
        count = int(df[col].sum())
        percent = round(count / len(df) * 100, 2) if len(df) > 0 else 0

        rows.append({
            "dataset": dataset_name,
            "flag": col,
            "flagged_rows": count,
            "flagged_percent": percent
        })

    return pd.DataFrame(rows)


def create_geo_summary(df, dataset_name):
    """
    Create geographic quality summary.
    """
    geo_flags = [
        "final_missing_coordinates_flag",
        "final_zero_coordinates_flag",
        "final_positive_longitude_flag",
        "final_implausible_ca_coordinates_flag"
    ]

    rows = []

    total_rows = len(df)

    for flag in geo_flags:
        if flag in df.columns:
            count = int(df[flag].sum())
            percent = round(count / total_rows * 100, 2) if total_rows > 0 else 0

            rows.append({
                "dataset": dataset_name,
                "geo_quality_check": flag,
                "flagged_rows": count,
                "flagged_percent": percent
            })

    if "final_geo_usable_flag" in df.columns:
        usable_count = int(df["final_geo_usable_flag"].sum())
        usable_percent = round(usable_count / total_rows * 100, 2) if total_rows > 0 else 0

        rows.append({
            "dataset": dataset_name,
            "geo_quality_check": "final_geo_usable_flag",
            "flagged_rows": usable_count,
            "flagged_percent": usable_percent
        })

    return pd.DataFrame(rows)


def create_date_summary(df, dataset_name):
    """
    Create date consistency summary.
    """
    date_flags = [
        "final_listing_after_close_flag",
        "final_purchase_after_close_flag",
        "final_negative_timeline_flag"
    ]

    rows = []

    total_rows = len(df)

    for flag in date_flags:
        if flag in df.columns:
            count = int(df[flag].sum())
            percent = round(count / total_rows * 100, 2) if total_rows > 0 else 0

            rows.append({
                "dataset": dataset_name,
                "date_quality_check": flag,
                "flagged_rows": count,
                "flagged_percent": percent
            })

    return pd.DataFrame(rows)


def create_row_count_summary(raw_df, flagged_df, clean_df, geo_clean_df, dataset_name):
    """
    Create before/after row count summary.
    """
    before_rows = len(raw_df)
    flagged_rows = len(flagged_df)
    clean_rows = len(clean_df)
    geo_clean_rows = len(geo_clean_df)

    return pd.DataFrame([{
        "dataset": dataset_name,
        "rows_loaded": before_rows,
        "rows_after_flag_creation": flagged_rows,
        "rows_final_analysis_ready": clean_rows,
        "rows_removed_from_final": before_rows - clean_rows,
        "removed_percent": round((before_rows - clean_rows) / before_rows * 100, 2)
        if before_rows > 0 else 0,
        "rows_geo_usable_subset": geo_clean_rows,
        "geo_usable_percent": round(geo_clean_rows / clean_rows * 100, 2)
        if clean_rows > 0 else 0
    }])


def process_dataset(raw_df, dataset_name):
    """
    Run Week 5 final cleaning for one dataset.
    """
    print("\n" + "=" * 80)
    print(f"PROCESSING {dataset_name.upper()} DATASET")
    print("=" * 80)

    df, rename_map = standardize_column_names(raw_df)

    if rename_map:
        print("\nStandardized column names:")
        for old, new in rename_map.items():
            print(f"   {old} -> {new}")

    # Confirm data types
    df, date_dtype_report = confirm_date_types(df, dataset_name)
    df, numeric_dtype_report = confirm_numeric_types(df, dataset_name)

    dtype_report = pd.concat(
        [date_dtype_report, numeric_dtype_report],
        ignore_index=True
    )

    # Add final validation flags
    print(f"\n🚩 Creating final numeric flags for {dataset_name}...")
    df = add_final_numeric_flags(df)

    print(f"\n🚩 Creating final date consistency flags for {dataset_name}...")
    df = add_final_date_flags(df)

    print(f"\n🚩 Creating final geographic quality flags for {dataset_name}...")
    df = add_final_geo_flags(df)

    # Create final clean dataset
    clean_mask = create_final_clean_mask(df, dataset_name)
    clean_df = df[clean_mask].copy()

    # Create geo-usable subset
    if "final_geo_usable_flag" in clean_df.columns:
        geo_clean_df = clean_df[clean_df["final_geo_usable_flag"]].copy()
    else:
        geo_clean_df = clean_df.copy()

    # Reports
    row_summary = create_row_count_summary(
        raw_df=raw_df,
        flagged_df=df,
        clean_df=clean_df,
        geo_clean_df=geo_clean_df,
        dataset_name=dataset_name
    )

    exclusion_summary = create_exclusion_reason_summary(
        df=df,
        clean_mask=clean_mask,
        dataset_name=dataset_name
    )

    flag_summary = create_flag_count_summary(df, dataset_name)
    geo_summary = create_geo_summary(df, dataset_name)
    date_summary = create_date_summary(df, dataset_name)

    print("\nRow count summary:")
    print(row_summary.to_string(index=False))

    print("\nTop final flags:")
    if not flag_summary.empty:
        print(flag_summary.sort_values("flagged_rows", ascending=False).head(10).to_string(index=False))

    return {
        "flagged_df": df,
        "clean_df": clean_df,
        "geo_clean_df": geo_clean_df,
        "dtype_report": dtype_report,
        "row_summary": row_summary,
        "exclusion_summary": exclusion_summary,
        "flag_summary": flag_summary,
        "geo_summary": geo_summary,
        "date_summary": date_summary
    }


# ============================================================
# MAIN SCRIPT
# ============================================================

print("=" * 80)
print("WEEK 5 DELIVERABLE: FINAL CLEANED ANALYSIS-READY DATASETS")
print("=" * 80)

# Step 1: Load Week 4 flagged datasets
print("\n📂 STEP 1: Loading Week 4 flagged datasets...")

sold_file = find_preferred_file(SOLD_PATTERNS, INPUT_DIRS, "Sold")
listings_file = find_preferred_file(LISTINGS_PATTERNS, INPUT_DIRS, "Listings")

sold_raw = pd.read_csv(sold_file, low_memory=False)
listings_raw = pd.read_csv(listings_file, low_memory=False)

print(f"\nSold rows loaded:     {len(sold_raw):,}")
print(f"Listings rows loaded: {len(listings_raw):,}")

# Step 2: Process sold and listings datasets
sold_results = process_dataset(sold_raw, "Sold")
listings_results = process_dataset(listings_raw, "Listings")

# Step 3: Save final datasets
print("\n💾 STEP 3: Saving final Week 5 datasets...")

sold_flagged_out = OUTPUT_DIR / f"sold_week5_final_flagged_{timestamp}.csv"
sold_clean_out = OUTPUT_DIR / f"sold_week5_analysis_ready_{timestamp}.csv"
sold_geo_out = OUTPUT_DIR / f"sold_week5_geo_usable_subset_{timestamp}.csv"

listings_flagged_out = OUTPUT_DIR / f"listings_week5_final_flagged_{timestamp}.csv"
listings_clean_out = OUTPUT_DIR / f"listings_week5_analysis_ready_{timestamp}.csv"
listings_geo_out = OUTPUT_DIR / f"listings_week5_geo_usable_subset_{timestamp}.csv"

sold_results["flagged_df"].to_csv(sold_flagged_out, index=False)
sold_results["clean_df"].to_csv(sold_clean_out, index=False)
sold_results["geo_clean_df"].to_csv(sold_geo_out, index=False)

listings_results["flagged_df"].to_csv(listings_flagged_out, index=False)
listings_results["clean_df"].to_csv(listings_clean_out, index=False)
listings_results["geo_clean_df"].to_csv(listings_geo_out, index=False)

print(f"   Sold final flagged dataset:      {sold_flagged_out.resolve()}")
print(f"   Sold analysis-ready dataset:     {sold_clean_out.resolve()}")
print(f"   Sold geo-usable subset:          {sold_geo_out.resolve()}")

print(f"   Listings final flagged dataset:  {listings_flagged_out.resolve()}")
print(f"   Listings analysis-ready dataset: {listings_clean_out.resolve()}")
print(f"   Listings geo-usable subset:      {listings_geo_out.resolve()}")

# Step 4: Save reports
print("\n🧾 STEP 4: Saving Week 5 validation reports...")

all_dtype = pd.concat(
    [sold_results["dtype_report"], listings_results["dtype_report"]],
    ignore_index=True
)

all_rows = pd.concat(
    [sold_results["row_summary"], listings_results["row_summary"]],
    ignore_index=True
)

all_exclusions = pd.concat(
    [sold_results["exclusion_summary"], listings_results["exclusion_summary"]],
    ignore_index=True
)

all_flags = pd.concat(
    [sold_results["flag_summary"], listings_results["flag_summary"]],
    ignore_index=True
)

all_geo = pd.concat(
    [sold_results["geo_summary"], listings_results["geo_summary"]],
    ignore_index=True
)

all_dates = pd.concat(
    [sold_results["date_summary"], listings_results["date_summary"]],
    ignore_index=True
)

dtype_out = REPORT_DIR / f"week5_dtype_confirmation_{timestamp}.csv"
row_out = REPORT_DIR / f"week5_before_after_row_counts_{timestamp}.csv"
exclusion_out = REPORT_DIR / f"week5_exclusion_reason_summary_{timestamp}.csv"
flag_out = REPORT_DIR / f"week5_final_quality_flag_counts_{timestamp}.csv"
geo_out = REPORT_DIR / f"week5_geographic_quality_summary_{timestamp}.csv"
date_out = REPORT_DIR / f"week5_date_consistency_summary_{timestamp}.csv"

all_dtype.to_csv(dtype_out, index=False)
all_rows.to_csv(row_out, index=False)
all_exclusions.to_csv(exclusion_out, index=False)
all_flags.to_csv(flag_out, index=False)
all_geo.to_csv(geo_out, index=False)
all_dates.to_csv(date_out, index=False)

print(f"   Data type confirmation:      {dtype_out.resolve()}")
print(f"   Before/after row counts:     {row_out.resolve()}")
print(f"   Exclusion reason summary:    {exclusion_out.resolve()}")
print(f"   Final quality flag counts:   {flag_out.resolve()}")
print(f"   Geographic quality summary:  {geo_out.resolve()}")
print(f"   Date consistency summary:    {date_out.resolve()}")

# Step 5: Final console summary
print("\n✅ FINAL WEEK 5 SUMMARY")
print("=" * 80)

print("\nBefore / after row counts:")
print(all_rows.to_string(index=False))

print("\nDate consistency summary:")
print(all_dates.to_string(index=False))

print("\nGeographic quality summary:")
print(all_geo.to_string(index=False))

print("\nOutput folder:")
print(OUTPUT_DIR.resolve())

print("\n✅ Week 5 Deliverable Complete!")