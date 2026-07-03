# week2deliverable.py
# ✅ 已彻底移除 matplotlib/seaborn 依赖
# ✅ 自动加载当前目录下所有 CRMLSListing*.csv 文件
# ✅ 输出统计结果到 ./week2_output/ 目录（与你左侧文件夹一致）

import pandas as pd
import json
from pathlib import Path

# ==========================================
# 1. 自动查找并合并所有数据文件（适配你的实际路径）
# ==========================================
DATA_DIR = Path("./")  # 当前目录即 C:\Users\张十一\Desktop\csv\experimental\
csv_files = list(DATA_DIR.glob("CRMLSListing*.csv"))

if not csv_files:
    print("❌ 错误：未找到任何 CRMLSListing*.csv 文件！")
    print(f"   请确认文件位于：{DATA_DIR.resolve()}")
    print("   当前目录下的文件列表：")
    for f in DATA_DIR.iterdir():
        if f.is_file():
            print(f"     {f.name}")
    exit(1)

print(f"✅ 找到 {len(csv_files)} 个数据文件：")
for f in csv_files[:5]:  # 只打印前5个避免刷屏
    print(f"   - {f.name}")
if len(csv_files) > 5:
    print(f"   ... 共 {len(csv_files)} 个文件")

# 合并所有 CSV（假设结构相同）
dfs = []
for file in csv_files:
    try:
        df = pd.read_csv(file, low_memory=False)
        dfs.append(df)
    except Exception as e:
        print(f"⚠️  跳过文件 {file.name}（读取失败）: {e}")
        continue

if not dfs:
    print("❌ 错误：所有文件读取均失败！")
    exit(1)

df_combined = pd.concat(dfs, ignore_index=True)
print(f"\n✅ 合并完成！总记录数: {len(df_combined):,}")

# ==========================================
# 2. 核心数据处理（保留你原来的逻辑框架）
# ==========================================
# 示例：计算基本统计量（你可替换为自己的业务逻辑）
stats_result = {
    "总记录数": len(df_combined),
    "数值列均值": df_combined.select_dtypes(include="number").mean().round(2).to_dict(),
    "缺失值总数": df_combined.isnull().sum().sum(),
    "缺失值按列统计": df_combined.isnull().sum().to_dict()
}

# 示例：按年份分组统计（假设你有 'Year' 或 'ListDate' 列）
date_col = None
for col in ["ListDate", "Year", "Date"]:
    if col in df_combined.columns:
        date_col = col
        break

if date_col:
    try:
        # 尝试提取年份（兼容 YYYY-MM-DD 或 YYYY 格式）
        if df_combined[date_col].dtype == 'object':
            years = pd.to_datetime(df_combined[date_col], errors='coerce').dt.year
        else:
            years = df_combined[date_col]
        yearly_count = years.value_counts().sort_index().to_dict()
        stats_result["按年份统计"] = yearly_count
    except Exception as e:
        print(f"⚠️  年份统计失败: {e}")

# ==========================================
# 3. 结果导出（替代图表）
# ==========================================
OUTPUT_DIR = Path("./week2_output")
OUTPUT_DIR.mkdir(exist_ok=True)

# 导出详细统计 JSON
json_path = OUTPUT_DIR / "summary_stats.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(stats_result, f, ensure_ascii=False, indent=2)
print(f"\n✅ 统计摘要已保存至: {json_path}")

# 导出前10行样本（方便检查）
sample_path = OUTPUT_DIR / "sample_data.csv"
df_combined.head(10).to_csv(sample_path, index=False, encoding="utf-8-sig")
print(f"✅ 前10行样本已保存至: {sample_path}")

# 如果你想看完整列名和类型（调试用）
cols_info = {
    "列名": df_combined.columns.tolist(),
    "数据类型": df_combined.dtypes.astype(str).tolist()
}
with open(OUTPUT_DIR / "columns_info.json", "w", encoding="utf-8") as f:
    json.dump(cols_info, f, ensure_ascii=False, indent=2)

print("\n🎉 所有任务完成！结果已存入 ./week2_output/")
print("   无需 matplotlib，无需额外安装，纯 Python 标准库 + Pandas 即可运行。")