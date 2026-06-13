import pandas as pd

# 读取数据
df = pd.read_csv('cleanData_no_missing.csv')

# 保留2005年及之后的车辆
df_filtered = df[df['year'] >= 2005]

# 查看结果
print(f"原始数据量: {len(df)}")
print(f"过滤后数据量: {len(df_filtered)}")
print(f"删除了 {len(df) - len(df_filtered)} 条记录")

# 查看年份分布
print("\n年份分布（2005年及以后）:")
print(df_filtered['year'].value_counts().sort_index())

# 保存过滤后的数据（可选）
df_filtered.to_csv('cleanData_2005plus.csv', index=False)

print("已保存为: cleanData_2005plus.csv")