import pandas as pd
import glob
import os

# =============================================================================
# IDX Exchange Intern Handbook - Week 1: Monthly Dataset Aggregation
# =============================================================================

def load_and_merge(file_pattern):
    """读取匹配模式的所有CSV文件并合并为一个DataFrame"""
    files = sorted(glob.glob(file_pattern))
    if not files:
        raise FileNotFoundError(f"未找到匹配 '{file_pattern}' 的文件，请检查路径和文件名。")
    
    print(f"📂 找到 {len(files)} 个文件: {os.path.basename(files[0])} ... {os.path.basename(files[-1])}")
    dfs = [pd.read_csv(f) for f in files]
    before_merge_rows = sum(len(df) for df in dfs)
    merged_df = pd.concat(dfs, ignore_index=True)
    after_merge_rows = len(merged_df)
    
    return merged_df, before_merge_rows, after_merge_rows


def main():
    # ==========================
    # STEP 1: 加载并合并数据集
    # ==========================
    # 自动匹配所有 Listing 和 Sold 文件（兼容 _filled 后缀）
    listing_merged, list_before, list_after = load_and_merge("CRMLSListing*.csv")
    sold_merged, sold_before, sold_after = load_and_merge("CRMLSSold*.csv")

    # ==========================
    # STEP 2: 过滤 Residential
    # ==========================
    list_res = listing_merged[listing_merged['PropertyType'] == 'Residential'].copy()
    sold_res = sold_merged[sold_merged['PropertyType'] == 'Residential'].copy()

    list_res_count = len(list_res)
    sold_res_count = len(sold_res)

    # ==========================
    # STEP 3: 保存结果
    # ==========================
    list_output = "Listings_Residential_Merged.csv"
    sold_output = "Sold_Residential_Merged.csv"
    
    list_res.to_csv(list_output, index=False)
    sold_res.to_csv(sold_output, index=False)
    
    print(f"\n✅ 保存完成:")
    print(f"   {list_output} ({list_res_count:,} rows)")
    print(f"   {sold_output} ({sold_res_count:,} rows)")

    # =========================================================================
    # ⚠️ DELIVERABLE REQUIREMENT: 必须在脚本中包含以下4组计数注释
    # 运行脚本后，请用下方 print 输出的真实数值替换注释中的示例值！
    # =========================================================================
    print("\n" + "="*60)
    print("📊 请将以下真实数值填入脚本注释中作为交付物验证：")
    print("="*60)
    print(f"# Listing Dataset:")
    print(f"#   - Before merge (sum of individual files): {list_before:,} rows")
    print(f"#   - After merge (concatenated):             {list_after:,} rows")
    print(f"#   - Before filter (Residential):            {list_after:,} rows")
    print(f"#   - After filter (Residential):             {list_res_count:,} rows")
    print(f"#")
    print(f"# Sold Dataset:")
    print(f"#   - Before merge (sum of individual files): {sold_before:,} rows")
    print(f"#   - After merge (concatenated):             {sold_after:,} rows")
    print(f"#   - Before filter (Residential):            {sold_after:,} rows")
    print(f"#   - After filter (Residential):             {sold_res_count:,} rows")
    print("="*60)


if __name__ == "__main__":
    main()