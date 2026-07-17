import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# WEEK 4: Data Cleaning and Preparation
# Based on:
# 1. IDX Exchange Week 4–5 handbook requirements
# 2. Google Doc column policy provided this week
#
# Main tasks:
# - Load Week 3 enriched sold and listing datasets
# - Drop unnecessary columns based on the Google Doc
# - Keep required analysis columns
# - Convert date fields
# - Convert numeric fields
# - Flag invalid numeric values
# - Flag date consistency issues
# - Flag geographic coordinate issues
# - Save full flagged datasets and cleaned datasets
# - Save validation reports and one chart output
# ============================================================


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_DIRS = [
    Path("./week3_output"),
    Path("./experimental"),
    Path(".")
]

OUTPUT_DIR = Path("./week4_output")
REPORT_DIR = OUTPUT_DIR / "reports"
FIGURE_DIR = OUTPUT_DIR / "figures"

OUTPUT_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)
FIGURE_DIR.mkdir(exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# The script will automatically find the latest matching Week 3 file.
SOLD_PATTERNS = [
    "sold_enriched_with_mortgage*.csv",
    "Sold_Residential_Merged.csv"
]

LISTINGS_PATTERNS = [
    "listings_enriched_with_mortgage*.csv",
    "Listings_Residential_Merged.csv"
]


# ============================================================
# COLUMN POLICY FROM GOOGLE DOC
# ============================================================

MUST_DROP_COLUMNS = [
    "BelowGradeFinishedArea",
    "WaterfrontYN",
    "BasementYN",
    "FireplacesTotal",
    "AboveGradeFinishedArea",
    "TaxAnnualAmount",
    "Latfilled",
    "Lonfilled",
    "BuilderName",
    "TaxYear",
    "ElementarySchoolDistrict",
    "CoBuyerAgentFirstName",
    "CoveredSpaces",
    "BusinessType",
    "LotSizeDimensions",
    "MiddleOrJuniorSchoolDistrict",
    "BuildingAreaTotal"
]

CAN_DROP_COLUMNS = [
    "ListAgentFirstName",
    "ListAgentLastName",
    "CoListAgentFirstName",
    "CoListAgentLastName"
]

CAN_RETAIN_COLUMNS = [
    "ElementarySchool",
    "MiddleOrJuniorSchool",
    "HighSchool",
    "HighSchoolDistrict",
    "ListAgentFullName",
    "ListAgentEmail",
    "ListOfficeName",
    "BuyerOfficeName",
    "BuyerAgentFirstName",
    "BuyerAgentLastName",
    "BuyerAgentMlsId",
    "BuyerAgencyCompensationType",
    "BuyerAgencyCompensation"
]

# Corrected based on your actual column list:
# Google Doc says PurchasecontractDate, but actual data has PurchaseContractDate.
# Google Doc says CountryOrParish, but actual data has CountyOrParish.
MUST_RETAIN_COLUMNS = [
    "ClosePrice",
    "ListPrice",
    "OriginalListPrice",
    "DaysOnMarket",
    "ListingContractDate",
    "PurchaseContractDate",
    "CloseDate",
    "LotSizeAcres",
    "BathroomsTotalInteger",
    "YearBuilt",
    "LivingArea",
    "BedroomsTotal",
    "City",
    "StateOrProvince",
    "PostalCode",
    "Longitude",
    "Latitude",
    "CountyOrParish"
]

# Keep Week 3 mortgage enrichment fields if they exist
WEEK3_RETAIN_COLUMNS = [
    "year_month",
    "rate_30yr_fixed"
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

CATEGORICAL_COLS_TO_FILL = [
    "City",
    "StateOrProvince",
    "PostalCode",
    "CountyOrParish",
    "PropertyType",
    "PropertySubType",
    "MLSAreaMajor",
    "ElementarySchool",
    "MiddleOrJuniorSchool",
    "HighSchool",
    "HighSchoolDistrict",
    "ListAgentFullName",
    "ListAgentEmail",
    "ListOfficeName",
    "BuyerOfficeName",
    "BuyerAgentFirstName",
    "BuyerAgentLastName",
    "BuyerAgentMlsId",
    "BuyerAgencyCompensationType",
    "BuyerAgencyCompensation"
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def find_latest_file(patterns, input_dirs, dataset_name):
    """Find the latest matching CSV file."""
    candidates = []

    for folder in input_dirs:
        if folder.exists():
            for pattern in patterns:
                candidates.extend(folder.glob(pattern))

    candidates = [file for file in candidates if file.is_file()]

    if not candidates:
        raise FileNotFoundError(
            f"Could not find input file for {dataset_name}.\n"
            f"Checked folders: {[str(folder) for folder in input_dirs]}\n"
            f"Checked patterns: {patterns}"
        )

    latest_file = max(candidates, key=lambda file: file.stat().st_mtime)
    print(f"   {dataset_name} file found: {latest_file}")
    return latest_file


def standardize_column_names(df):
    """
    Fix known column name issues.
    This protects the code from small spelling differences.
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


def apply_column_policy(df, dataset_name):
    """
    Apply the Week 4 Google Doc column policy:
    - Drop must-drop columns
    - Drop optional agent-name columns
    - Keep required and useful columns
    - Report missing required fields
    """
    print(f"\n🧹 Applying Week 4 column policy for {dataset_name}...")

    output = df.copy()

    columns_before = list(output.columns)

    # Drop unnamed index columns from previous CSV exports
    unnamed_cols = [col for col in output.columns if col.startswith("Unnamed:")]

    drop_candidates = unnamed_cols + MUST_DROP_COLUMNS + CAN_DROP_COLUMNS
    existing_drop_cols = [col for col in drop_candidates if col in output.columns]

    output = output.drop(columns=existing_drop_cols, errors="ignore")

    missing_required_cols = [
        col for col in MUST_RETAIN_COLUMNS
        if col not in output.columns
    ]

    existing_required_cols = [
        col for col in MUST_RETAIN_COLUMNS
        if col in output.columns
    ]

    existing_useful_cols = [
        col for col in CAN_RETAIN_COLUMNS
        if col in output.columns
    ]

    existing_week3_cols = [
        col for col in WEEK3_RETAIN_COLUMNS
        if col in output.columns
    ]

    print(f"   Columns before cleaning: {len(columns_before)}")
    print(f"   Columns dropped:         {len(existing_drop_cols)}")
    print(f"   Columns after cleaning:  {len(output.columns)}")

    if existing_drop_cols:
        print("   Dropped columns:")
        for col in existing_drop_cols:
            print(f"      - {col}")

    if missing_required_cols:
        print("   Missing required columns:")
        for col in missing_required_cols:
            print(f"      - {col}")
        print("   Note: Some fields may only exist in sold data or listing data.")
    else:
        print("   All required columns are present.")

    policy_report = pd.DataFrame([
        {
            "dataset": dataset_name,
            "columns_before": len(columns_before),
            "columns_after": len(output.columns),
            "dropped_columns_count": len(existing_drop_cols),
            "dropped_columns": ", ".join(existing_drop_cols) if existing_drop_cols else "None",
            "existing_required_columns": ", ".join(existing_required_cols) if existing_required_cols else "None",
            "missing_required_columns": ", ".join(missing_required_cols) if missing_required_cols else "None",
            "existing_useful_columns": ", ".join(existing_useful_cols) if existing_useful_cols else "None",
            "week3_enrichment_columns": ", ".join(existing_week3_cols) if existing_week3_cols else "None"
        }
    ])

    return output, policy_report


def convert_date_columns(df, dataset_name):
    """Convert existing date fields to datetime format."""
    print(f"\n📅 Converting date columns for {dataset_name}...")

    output = df.copy()
    converted_cols = []

    for col in DATE_COLS:
        if col in output.columns:
            output[col] = pd.to_datetime(output[col], errors="coerce")
            converted_cols.append(col)
            missing_count = output[col].isna().sum()
            print(f"   {col}: converted | missing or invalid = {missing_count:,}")

    if not converted_cols:
        print("   No expected date columns found.")

    return output, converted_cols


def clean_numeric_text(series):
    """
    Convert text-based numeric values into clean numeric format.
    Handles commas, dollar signs, and percent signs if they appear.
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


def convert_numeric_columns(df, dataset_name):
    """Convert existing numeric fields to numeric dtype."""
    print(f"\n🔢 Converting numeric columns for {dataset_name}...")

    output = df.copy()
    converted_cols = []

    for col in NUMERIC_COLS:
        if col in output.columns:
            output[col] = clean_numeric_text(output[col])
            output[col] = pd.to_numeric(output[col], errors="coerce")
            converted_cols.append(col)
            missing_count = output[col].isna().sum()
            print(f"   {col}: converted | missing or invalid = {missing_count:,}")

    if not converted_cols:
        print("   No expected numeric columns found.")

    return output, converted_cols


def fill_selected_categorical_missing(df, dataset_name):
    """
    Fill selected categorical fields with Unknown.
    Numeric fields are not filled because fake numeric values would distort analysis.
    """
    print(f"\n🧩 Filling selected categorical missing values for {dataset_name}...")

    output = df.copy()
    filled_cols = []

    for col in CATEGORICAL_COLS_TO_FILL:
        if col in output.columns:
            before_missing = output[col].isna().sum()
            output[col] = output[col].astype("string").fillna("Unknown")
            after_missing = output[col].isna().sum()
            filled_cols.append(col)

            print(
                f"   {col}: missing before = {before_missing:,}, "
                f"after = {after_missing:,}"
            )

    return output, filled_cols


def add_missing_value_flags(df):
    """Create missing-value flags for important analysis fields."""
    output = df.copy()

    for col in MUST_RETAIN_COLUMNS + WEEK3_RETAIN_COLUMNS:
        if col in output.columns:
            output[f"missing_{col}_flag"] = output[col].isna()

    return output


def add_invalid_numeric_flags(df):
    """
    Add invalid numeric value flags.
    Missing values are tracked separately, so they are not automatically treated as invalid here.
    """
    output = df.copy()

    output["invalid_close_price_flag"] = False
    if "ClosePrice" in output.columns:
        output.loc[output["ClosePrice"].notna(), "invalid_close_price_flag"] = (
            output.loc[output["ClosePrice"].notna(), "ClosePrice"] <= 0
        )

    output["invalid_list_price_flag"] = False
    if "ListPrice" in output.columns:
        output.loc[output["ListPrice"].notna(), "invalid_list_price_flag"] = (
            output.loc[output["ListPrice"].notna(), "ListPrice"] <= 0
        )

    output["invalid_original_list_price_flag"] = False
    if "OriginalListPrice" in output.columns:
        output.loc[output["OriginalListPrice"].notna(), "invalid_original_list_price_flag"] = (
            output.loc[output["OriginalListPrice"].notna(), "OriginalListPrice"] <= 0
        )

    output["invalid_living_area_flag"] = False
    if "LivingArea" in output.columns:
        output.loc[output["LivingArea"].notna(), "invalid_living_area_flag"] = (
            output.loc[output["LivingArea"].notna(), "LivingArea"] <= 0
        )

    output["invalid_days_on_market_flag"] = False
    if "DaysOnMarket" in output.columns:
        output.loc[output["DaysOnMarket"].notna(), "invalid_days_on_market_flag"] = (
            output.loc[output["DaysOnMarket"].notna(), "DaysOnMarket"] < 0
        )

    output["invalid_bedrooms_flag"] = False
    if "BedroomsTotal" in output.columns:
        output.loc[output["BedroomsTotal"].notna(), "invalid_bedrooms_flag"] = (
            output.loc[output["BedroomsTotal"].notna(), "BedroomsTotal"] < 0
        )

    output["invalid_bathrooms_flag"] = False
    if "BathroomsTotalInteger" in output.columns:
        output.loc[output["BathroomsTotalInteger"].notna(), "invalid_bathrooms_flag"] = (
            output.loc[output["BathroomsTotalInteger"].notna(), "BathroomsTotalInteger"] < 0
        )

    output["invalid_lot_size_flag"] = False
    if "LotSizeAcres" in output.columns:
        output.loc[output["LotSizeAcres"].notna(), "invalid_lot_size_flag"] = (
            output.loc[output["LotSizeAcres"].notna(), "LotSizeAcres"] < 0
        )

    output["invalid_year_built_flag"] = False
    if "YearBuilt" in output.columns:
        output.loc[output["YearBuilt"].notna(), "invalid_year_built_flag"] = (
            (output.loc[output["YearBuilt"].notna(), "YearBuilt"] < 1800) |
            (output.loc[output["YearBuilt"].notna(), "YearBuilt"] > 2035)
        )

    return output


def add_date_consistency_flags(df):
    """
    Add required date consistency flags:
    - ListingContractDate should not be after CloseDate
    - PurchaseContractDate should not be after CloseDate
    - Timeline should not go backwards
    """
    output = df.copy()

    output["listing_after_close_flag"] = False
    output["purchase_after_close_flag"] = False
    output["negative_timeline_flag"] = False

    if "ListingContractDate" in output.columns and "CloseDate" in output.columns:
        valid = output["ListingContractDate"].notna() & output["CloseDate"].notna()
        output.loc[valid, "listing_after_close_flag"] = (
            output.loc[valid, "ListingContractDate"] >
            output.loc[valid, "CloseDate"]
        )

    if "PurchaseContractDate" in output.columns and "CloseDate" in output.columns:
        valid = output["PurchaseContractDate"].notna() & output["CloseDate"].notna()
        output.loc[valid, "purchase_after_close_flag"] = (
            output.loc[valid, "PurchaseContractDate"] >
            output.loc[valid, "CloseDate"]
        )

    if "ListingContractDate" in output.columns and "PurchaseContractDate" in output.columns:
        valid = output["ListingContractDate"].notna() & output["PurchaseContractDate"].notna()
        output.loc[valid, "negative_timeline_flag"] = (
            output.loc[valid, "ListingContractDate"] >
            output.loc[valid, "PurchaseContractDate"]
        )

    if "PurchaseContractDate" in output.columns and "CloseDate" in output.columns:
        valid = output["PurchaseContractDate"].notna() & output["CloseDate"].notna()
        output.loc[valid, "negative_timeline_flag"] = (
            output.loc[valid, "negative_timeline_flag"] |
            (
                output.loc[valid, "PurchaseContractDate"] >
                output.loc[valid, "CloseDate"]
            )
        )

    if "ListingContractDate" in output.columns and "CloseDate" in output.columns:
        valid = output["ListingContractDate"].notna() & output["CloseDate"].notna()
        output.loc[valid, "negative_timeline_flag"] = (
            output.loc[valid, "negative_timeline_flag"] |
            (
                output.loc[valid, "ListingContractDate"] >
                output.loc[valid, "CloseDate"]
            )
        )

    return output


def add_geographic_quality_flags(df):
    """
    Add coordinate quality flags.
    CRMLS data should generally have California-like coordinates:
    - Latitude around 32 to 42.5
    - Longitude around -125 to -113
    """
    output = df.copy()

    if "Latitude" in output.columns and "Longitude" in output.columns:
        output["missing_coordinates_flag"] = (
            output["Latitude"].isna() |
            output["Longitude"].isna()
        )

        output["zero_coordinates_flag"] = (
            output["Latitude"].eq(0) |
            output["Longitude"].eq(0)
        )

        output["positive_longitude_flag"] = output["Longitude"] > 0

        valid_coords = (
            output["Latitude"].notna() &
            output["Longitude"].notna() &
            ~output["Latitude"].eq(0) &
            ~output["Longitude"].eq(0)
        )

        output["implausible_ca_coordinates_flag"] = False

        output.loc[valid_coords, "implausible_ca_coordinates_flag"] = (
            (output.loc[valid_coords, "Latitude"] < 32.0) |
            (output.loc[valid_coords, "Latitude"] > 42.5) |
            (output.loc[valid_coords, "Longitude"] < -125.0) |
            (output.loc[valid_coords, "Longitude"] > -113.0)
        )

        output["geo_usable_flag"] = ~(
            output["missing_coordinates_flag"] |
            output["zero_coordinates_flag"] |
            output["positive_longitude_flag"] |
            output["implausible_ca_coordinates_flag"]
        )

    else:
        output["missing_coordinates_flag"] = True
        output["zero_coordinates_flag"] = False
        output["positive_longitude_flag"] = False
        output["implausible_ca_coordinates_flag"] = False
        output["geo_usable_flag"] = False

    return output


def create_clean_dataset(df):
    """
    Create a cleaned dataset by removing clearly invalid rows.
    Missing-value rows are not automatically removed because some fields may be usable
    for other analysis tasks.
    """
    invalid_flags_for_removal = [
        "invalid_close_price_flag",
        "invalid_list_price_flag",
        "invalid_original_list_price_flag",
        "invalid_living_area_flag",
        "invalid_days_on_market_flag",
        "invalid_bedrooms_flag",
        "invalid_bathrooms_flag",
        "invalid_lot_size_flag",
        "invalid_year_built_flag",
        "listing_after_close_flag",
        "purchase_after_close_flag",
        "negative_timeline_flag"
    ]

    existing_flags = [flag for flag in invalid_flags_for_removal if flag in df.columns]

    if not existing_flags:
        return df.copy()

    clean_mask = ~df[existing_flags].any(axis=1)
    clean_df = df[clean_mask].copy()

    return clean_df


def create_missing_summary(df, dataset_name):
    """Create missing-value summary table."""
    total_rows = len(df)

    rows = []

    for col in df.columns:
        missing_count = df[col].isna().sum()
        missing_percent = round(missing_count / total_rows * 100, 2) if total_rows > 0 else 0

        rows.append({
            "dataset": dataset_name,
            "column": col,
            "missing_count": int(missing_count),
            "missing_percent": missing_percent,
            "above_90_percent_missing": missing_percent > 90
        })

    return pd.DataFrame(rows)


def create_flag_summary(df, dataset_name):
    """Create summary table for all flag columns."""
    flag_cols = [col for col in df.columns if col.endswith("_flag")]

    rows = []

    for col in flag_cols:
        flagged_count = int(df[col].sum())
        flagged_percent = round(flagged_count / len(df) * 100, 2) if len(df) > 0 else 0

        rows.append({
            "dataset": dataset_name,
            "flag": col,
            "flagged_rows": flagged_count,
            "flagged_percent": flagged_percent
        })

    return pd.DataFrame(rows)


def create_dtype_report(before_df, after_df, dataset_name):
    """Create before-and-after dtype report."""
    before_types = pd.DataFrame({
        "column": before_df.columns,
        "dtype_before": [str(before_df[col].dtype) for col in before_df.columns]
    })

    after_types = pd.DataFrame({
        "column": after_df.columns,
        "dtype_after": [str(after_df[col].dtype) for col in after_df.columns]
    })

    report = after_types.merge(before_types, on="column", how="left")
    report.insert(0, "dataset", dataset_name)

    return report


def create_row_summary(raw_df, flagged_df, clean_df, dataset_name):
    """Create row and column count summary."""
    rows_before = len(raw_df)
    rows_after_flagging = len(flagged_df)
    rows_after_cleaning = len(clean_df)

    return pd.DataFrame([
        {
            "dataset": dataset_name,
            "rows_before": rows_before,
            "rows_after_flagging": rows_after_flagging,
            "rows_after_cleaning": rows_after_cleaning,
            "rows_removed_in_clean_dataset": rows_before - rows_after_cleaning,
            "removed_percent": round((rows_before - rows_after_cleaning) / rows_before * 100, 2)
            if rows_before > 0 else 0,
            "columns_before": len(raw_df.columns),
            "columns_after": len(flagged_df.columns)
        }
    ])


def create_flag_chart(flag_summary):
    """Create one image output showing the top quality issues."""
    if flag_summary.empty:
        return None

    chart_df = (
        flag_summary
        .sort_values("flagged_rows", ascending=False)
        .head(20)
        .sort_values("flagged_rows", ascending=True)
        .copy()
    )

    chart_df["label"] = chart_df["dataset"] + " - " + chart_df["flag"]

    plt.figure(figsize=(12, 8))
    plt.barh(chart_df["label"], chart_df["flagged_rows"])
    plt.xlabel("Flagged Row Count")
    plt.ylabel("Data Quality Flag")
    plt.title("Week 4 Data Quality Flag Summary")
    plt.tight_layout()

    fig_path = FIGURE_DIR / f"week4_data_quality_flag_summary_{timestamp}.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()

    return fig_path


def process_dataset(raw_df, dataset_name):
    """Run full Week 4 cleaning pipeline for one dataset."""
    print("\n" + "=" * 80)
    print(f"PROCESSING {dataset_name.upper()} DATASET")
    print("=" * 80)

    original_df = raw_df.copy()

    # 1. Standardize column names
    df, rename_map = standardize_column_names(raw_df)

    if rename_map:
        print(f"\nColumn names standardized for {dataset_name}:")
        for old, new in rename_map.items():
            print(f"   {old} -> {new}")

    # 2. Apply column policy
    df, column_policy_report = apply_column_policy(df, dataset_name)

    # 3. Convert dates
    df, converted_dates = convert_date_columns(df, dataset_name)

    # 4. Convert numeric fields
    df, converted_numeric = convert_numeric_columns(df, dataset_name)

    # 5. Fill selected categorical missing values
    df, filled_categorical = fill_selected_categorical_missing(df, dataset_name)

    # 6. Add missing value flags
    print(f"\n🚩 Adding missing-value flags for {dataset_name}...")
    df = add_missing_value_flags(df)

    # 7. Add invalid numeric flags
    print(f"\n🚩 Adding invalid numeric flags for {dataset_name}...")
    df = add_invalid_numeric_flags(df)

    # 8. Add date consistency flags
    print(f"\n🚩 Adding date consistency flags for {dataset_name}...")
    df = add_date_consistency_flags(df)

    # 9. Add geographic quality flags
    print(f"\n🚩 Adding geographic quality flags for {dataset_name}...")
    df = add_geographic_quality_flags(df)

    # 10. Create cleaned dataset
    clean_df = create_clean_dataset(df)

    # 11. Reports
    missing_summary = create_missing_summary(df, dataset_name)
    flag_summary = create_flag_summary(df, dataset_name)
    dtype_report = create_dtype_report(original_df, df, dataset_name)
    row_summary = create_row_summary(original_df, df, clean_df, dataset_name)

    print(f"\n📊 {dataset_name} row summary:")
    print(row_summary.to_string(index=False))

    return {
        "flagged_df": df,
        "clean_df": clean_df,
        "column_policy_report": column_policy_report,
        "missing_summary": missing_summary,
        "flag_summary": flag_summary,
        "dtype_report": dtype_report,
        "row_summary": row_summary
    }


# ============================================================
# MAIN SCRIPT
# ============================================================

print("=" * 80)
print("WEEK 4 DELIVERABLE: DATA CLEANING AND PREPARATION")
print("=" * 80)

# Step 1: Load Week 3 outputs
print("\n📂 STEP 1: Finding Week 3 enriched datasets...")

sold_file = find_latest_file(SOLD_PATTERNS, INPUT_DIRS, "Sold")
listings_file = find_latest_file(LISTINGS_PATTERNS, INPUT_DIRS, "Listings")

sold_raw = pd.read_csv(sold_file, low_memory=False)
listings_raw = pd.read_csv(listings_file, low_memory=False)

print(f"\nSold rows loaded:     {len(sold_raw):,}")
print(f"Listings rows loaded: {len(listings_raw):,}")

# Step 2: Process both datasets
sold_results = process_dataset(sold_raw, "Sold")
listings_results = process_dataset(listings_raw, "Listings")

# Step 3: Save datasets
print("\n💾 STEP 3: Saving Week 4 datasets...")

sold_flagged_out = OUTPUT_DIR / f"sold_week4_flagged_{timestamp}.csv"
sold_clean_out = OUTPUT_DIR / f"sold_week4_clean_{timestamp}.csv"

listings_flagged_out = OUTPUT_DIR / f"listings_week4_flagged_{timestamp}.csv"
listings_clean_out = OUTPUT_DIR / f"listings_week4_clean_{timestamp}.csv"

sold_results["flagged_df"].to_csv(sold_flagged_out, index=False)
sold_results["clean_df"].to_csv(sold_clean_out, index=False)

listings_results["flagged_df"].to_csv(listings_flagged_out, index=False)
listings_results["clean_df"].to_csv(listings_clean_out, index=False)

print(f"   Sold flagged dataset saved:     {sold_flagged_out.resolve()}")
print(f"   Sold clean dataset saved:       {sold_clean_out.resolve()}")
print(f"   Listings flagged dataset saved: {listings_flagged_out.resolve()}")
print(f"   Listings clean dataset saved:   {listings_clean_out.resolve()}")

# Step 4: Save reports
print("\n🧾 STEP 4: Saving Week 4 reports...")

all_column_policy = pd.concat(
    [
        sold_results["column_policy_report"],
        listings_results["column_policy_report"]
    ],
    ignore_index=True
)

all_missing = pd.concat(
    [
        sold_results["missing_summary"],
        listings_results["missing_summary"]
    ],
    ignore_index=True
)

all_flags = pd.concat(
    [
        sold_results["flag_summary"],
        listings_results["flag_summary"]
    ],
    ignore_index=True
)

all_dtypes = pd.concat(
    [
        sold_results["dtype_report"],
        listings_results["dtype_report"]
    ],
    ignore_index=True
)

all_rows = pd.concat(
    [
        sold_results["row_summary"],
        listings_results["row_summary"]
    ],
    ignore_index=True
)

column_policy_out = REPORT_DIR / f"week4_column_policy_report_{timestamp}.csv"
missing_out = REPORT_DIR / f"week4_missing_value_summary_{timestamp}.csv"
flag_out = REPORT_DIR / f"week4_quality_flag_summary_{timestamp}.csv"
dtype_out = REPORT_DIR / f"week4_dtype_report_{timestamp}.csv"
row_out = REPORT_DIR / f"week4_row_count_summary_{timestamp}.csv"

all_column_policy.to_csv(column_policy_out, index=False)
all_missing.to_csv(missing_out, index=False)
all_flags.to_csv(flag_out, index=False)
all_dtypes.to_csv(dtype_out, index=False)
all_rows.to_csv(row_out, index=False)

print(f"   Column policy report saved: {column_policy_out.resolve()}")
print(f"   Missing value report saved: {missing_out.resolve()}")
print(f"   Quality flag report saved:  {flag_out.resolve()}")
print(f"   Dtype report saved:         {dtype_out.resolve()}")
print(f"   Row count report saved:     {row_out.resolve()}")

# Step 5: Save one chart output
print("\n🖼️ STEP 5: Creating chart output...")

figure_path = create_flag_chart(all_flags)

if figure_path:
    print(f"   Chart saved: {figure_path.resolve()}")
else:
    print("   No chart created because no flag summary was available.")

# Step 6: Final console summary
print("\n✅ FINAL WEEK 4 SUMMARY")
print("=" * 80)

print("\nRow count summary:")
print(all_rows.to_string(index=False))

print("\nTop quality flags:")
print(
    all_flags
    .sort_values("flagged_rows", ascending=False)
    .head(15)
    .to_string(index=False)
)

print("\nOutput folder:")
print(OUTPUT_DIR.resolve())

print("\n✅ Week 4 Deliverable Complete!")