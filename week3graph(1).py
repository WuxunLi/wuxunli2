import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# === CONFIGURATION ===
WEEK3_DIR = Path("./week3_output")
SOLD_FILE = WEEK3_DIR / "sold_enriched_with_mortgage.csv"
LISTINGS_FILE = WEEK3_DIR / "listings_enriched_with_mortgage.csv"
OUTPUT_PLOT = WEEK3_DIR / "week3_mortgage_validation.png"

# === LOAD ENRICHED DATA ===
print("📂 Loading Week 3 enriched datasets...")
sold = pd.read_csv(SOLD_FILE, parse_dates=['CloseDate_dt'])
listings = pd.read_csv(LISTINGS_FILE, parse_dates=['ListingContractDate_dt'])

# Create unified monthly rate series from both datasets for cross-validation
sold_monthly = sold.groupby('year_month')['rate_30yr_fixed'].mean().reset_index(name='sold_avg_rate')
list_monthly = listings.groupby('year_month')['rate_30yr_fixed'].mean().reset_index(name='list_avg_rate')
validation_df = sold_monthly.merge(list_monthly, on='year_month', how='outer')
validation_df['year_month'] = pd.to_datetime(validation_df['year_month'] + '-01')

# === GENERATE 3-PANEL VALIDATION FIGURE ===
fig, axes = plt.subplots(1, 3, figsize=(20, 5))
sns.set_theme(style="whitegrid", font_scale=0.9)

# PANEL 1: Mortgage Rate Time Series + Transaction Volume
ax1 = axes[0]
ax1.plot(validation_df['year_month'], validation_df['sold_avg_rate'], 
         marker='o', markersize=3, label='Sold Avg Rate', color='#2E86AB')
ax1.plot(validation_df['year_month'], validation_df['list_avg_rate'], 
         marker='s', markersize=3, label='Listings Avg Rate', color='#E8475F', linestyle='--')
ax1.set_title('Monthly 30-Yr Fixed Rate: Sold vs Listings', fontweight='bold')
ax1.set_ylabel('Rate (%)')
ax1.legend()
ax1.tick_params(axis='x', rotation=45)

# PANEL 2: Merge Coverage Heatmap (Null Rate by Year)
ax2 = axes[1]
for df, name, color in [(sold, 'Sold', '#2E86AB'), (listings, 'Listings', '#E8475F')]:
    yearly_nulls = df.assign(year=df['year_month'].str[:4]).groupby('year')['rate_30yr_fixed'].apply(
        lambda x: round(x.isnull().mean() * 100, 2)
    ).reset_index(name=f'{name}_Null_Pct')
    ax2.bar(yearly_nulls['year'] + (0.2 if name == 'Listings' else -0.2), 
            yearly_nulls[f'{name}_Null_Pct'], width=0.4, label=name, color=color, alpha=0.85)
ax2.set_title('Mortgage Rate Null Rate by Year (%)', fontweight='bold')
ax2.set_ylabel('Null %')
ax2.legend()
ax2.set_xticks(sorted(sold['year_month'].str[:4].unique()))

# PANEL 3: Distribution of Matched Rates (Sanity Check)
ax3 = axes[2]
sns.kdeplot(sold['rate_30yr_fixed'].dropna(), label='Sold', ax=ax3, color='#2E86AB', fill=True, alpha=0.3)
sns.kdeplot(listings['rate_30yr_fixed'].dropna(), label='Listings', ax=ax3, color='#E8475F', fill=True, alpha=0.3)
ax3.set_title('Distribution of Merged Mortgage Rates', fontweight='bold')
ax3.set_xlabel('30-Yr Fixed Rate (%)')
ax3.legend()

plt.suptitle('Week 3 Validation: Mortgage Rate Enrichment Quality', fontsize=14, y=1.05, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_PLOT, dpi=150, bbox_inches='tight')
plt.close()

print(f"✅ Validation chart saved to: {OUTPUT_PLOT}")