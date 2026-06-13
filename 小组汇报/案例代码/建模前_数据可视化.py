#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
二手车价格预测 - 建模前数据可视化
在特征工程完成后、模型构建前，进行全面的数据探索和可视化
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

print("="*70)
print("二手车价格预测 - 建模前数据可视化")
print("="*70)

# ========================================================
# 阶段1：数据读取与预处理
# ========================================================
print("\n" + "="*70)
print("阶段1：数据读取与预处理")
print("="*70)

df = pd.read_csv('cleanData_2005plus.csv')
print(f"原始数据量: {len(df)}")

# 创建age特征
df['age'] = 2021 - df['year']

# 选择特征
features = ['price', 'year', 'odometer', 'age', 'manufacturer', 'model',
            'condition', 'cylinders', 'fuel', 'transmission', 'type']
df_model = df[features].copy()

# 极端值处理
def remove_outliers_percentile(df, columns, lower=0.01, upper=0.99):
    df_clean = df.copy()
    for col in columns:
        lower_val = df_clean[col].quantile(lower)
        upper_val = df_clean[col].quantile(upper)
        df_clean = df_clean[(df_clean[col] >= lower_val) & (df_clean[col] <= upper_val)]
    return df_clean

df_clean = remove_outliers_percentile(df_model, ['price', 'odometer'], lower=0.01, upper=0.99)
df_clean = df_clean[df_clean['price'] > 0]

print(f"处理后数据量: {len(df_clean)}")
print(f"删除记录数: {len(df_model) - len(df_clean)}")

# LabelEncoder编码
categorical_cols = ['manufacturer', 'model', 'condition', 'cylinders', 'fuel', 'transmission', 'type']
df_encoded = df_clean.copy()
for col in categorical_cols:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))

# ========================================================
# 阶段2：目标变量分布可视化
# ========================================================
print("\n" + "="*70)
print("阶段2：目标变量（价格）分布可视化")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 2.1 价格直方图
ax1 = axes[0, 0]
ax1.hist(df_clean['price'], bins=50, color='steelblue', edgecolor='white', alpha=0.8)
ax1.axvline(df_clean['price'].mean(), color='red', linestyle='--', linewidth=2, label=f'均值: ${df_clean["price"].mean():,.0f}')
ax1.axvline(df_clean['price'].median(), color='orange', linestyle='--', linewidth=2, label=f'中位数: ${df_clean["price"].median():,.0f}')
ax1.set_xlabel('价格 (美元)', fontsize=12)
ax1.set_ylabel('频数', fontsize=12)
ax1.set_title('二手车价格分布', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)

# 2.2 价格箱线图
ax2 = axes[0, 1]
bp = ax2.boxplot(df_clean['price'], vert=True, patch_artist=True)
bp['boxes'][0].set_facecolor('lightblue')
ax2.set_ylabel('价格 (美元)', fontsize=12)
ax2.set_title('价格箱线图', fontsize=14, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

# 2.3 对数价格分布
ax3 = axes[1, 0]
log_price = np.log1p(df_clean['price'])
ax3.hist(log_price, bins=50, color='green', edgecolor='white', alpha=0.8)
ax3.set_xlabel('log(价格)', fontsize=12)
ax3.set_ylabel('频数', fontsize=12)
ax3.set_title('对数变换后的价格分布', fontsize=14, fontweight='bold')

# 2.4 价格统计摘要
ax4 = axes[1, 1]
ax4.axis('off')
stats_text = f"""
价格统计摘要
━━━━━━━━━━━━━━━━━━━
样本数量: {len(df_clean):,}
均值: ${df_clean['price'].mean():,.2f}
中位数: ${df_clean['price'].median():,.2f}
标准差: ${df_clean['price'].std():,.2f}
最小值: ${df_clean['price'].min():,.2f}
最大值: ${df_clean['price'].max():,.2f}
25%分位: ${df_clean['price'].quantile(0.25):,.2f}
75%分位: ${df_clean['price'].quantile(0.75):,.2f}
偏度: {df_clean['price'].skew():.2f}
峰度: {df_clean['price'].kurtosis():.2f}
"""
ax4.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
         verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
ax4.set_title('价格统计信息', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('建模前_价格分布可视化.png', dpi=150, bbox_inches='tight')
plt.close()
print("已保存: 建模前_价格分布可视化.png")

# ========================================================
# 阶段3：相关性热力图
# ========================================================
print("\n" + "="*70)
print("阶段3：相关性热力图")
print("="*70)

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# 3.1 完整相关性热力图（含year）
ax1 = axes[0]
corr_all = df_encoded.corr()
mask1 = np.triu(np.ones_like(corr_all, dtype=bool))
sns.heatmap(corr_all, mask=mask1, annot=True, cmap='coolwarm', fmt='.2f',
            annot_kws={'size': 9}, ax=ax1, vmin=-1, vmax=1,
            cbar_kws={'label': '相关系数'})
ax1.set_title('相关性热力图（含year）', fontsize=14, fontweight='bold')

# 3.2 移除year后的相关性热力图
ax2 = axes[1]
corr_no_year = df_encoded.drop(['year'], axis=1).corr()
mask2 = np.triu(np.ones_like(corr_no_year, dtype=bool))
sns.heatmap(corr_no_year, mask=mask2, annot=True, cmap='coolwarm', fmt='.2f',
            annot_kws={'size': 9}, ax=ax2, vmin=-1, vmax=1,
            cbar_kws={'label': '相关系数'})
ax2.set_title('相关性热力图（移除year后）', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('建模前_相关性热力图.png', dpi=150, bbox_inches='tight')
plt.close()
print("已保存: 建模前_相关性热力图.png")

# 输出price与各特征的相关性
print("\n【price与各特征的相关性】")
price_corr = corr_no_year['price'].sort_values(ascending=False)
print(price_corr.round(4))

# ========================================================
# 阶段4：数值特征分布
# ========================================================
print("\n" + "="*70)
print("阶段4：数值特征分布")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 4.1 车龄分布
ax1 = axes[0, 0]
ax1.hist(df_clean['age'], bins=30, color='coral', edgecolor='white', alpha=0.8)
ax1.axvline(df_clean['age'].mean(), color='red', linestyle='--', linewidth=2, label=f'均值: {df_clean["age"].mean():.1f}年')
ax1.set_xlabel('车龄 (年)', fontsize=12)
ax1.set_ylabel('频数', fontsize=12)
ax1.set_title('车龄分布', fontsize=14, fontweight='bold')
ax1.legend()

# 4.2 里程分布
ax2 = axes[0, 1]
ax2.hist(df_clean['odometer'], bins=30, color='mediumseagreen', edgecolor='white', alpha=0.8)
ax2.axvline(df_clean['odometer'].mean(), color='red', linestyle='--', linewidth=2, label=f'均值: {df_clean["odometer"].mean():,.0f}')
ax2.set_xlabel('行驶里程 (英里)', fontsize=12)
ax2.set_ylabel('频数', fontsize=12)
ax2.set_title('里程分布', fontsize=14, fontweight='bold')
ax2.legend()

# 4.3 车龄 vs 价格 散点图
ax3 = axes[1, 0]
sample = df_clean.sample(min(1000, len(df_clean)), random_state=42)
ax3.scatter(sample['age'], sample['price'], alpha=0.5, s=20, c='steelblue')
ax3.set_xlabel('车龄 (年)', fontsize=12)
ax3.set_ylabel('价格 (美元)', fontsize=12)
ax3.set_title('车龄 vs 价格', fontsize=14, fontweight='bold')
# 添加趋势线
z = np.polyfit(sample['age'], sample['price'], 1)
p = np.poly1d(z)
ax3.plot(sorted(sample['age']), p(sorted(sample['age'])), "r--", linewidth=2, label='趋势线')
ax3.legend()

# 4.4 里程 vs 价格 散点图
ax4 = axes[1, 1]
ax4.scatter(sample['odometer'], sample['price'], alpha=0.5, s=20, c='green')
ax4.set_xlabel('行驶里程 (英里)', fontsize=12)
ax4.set_ylabel('价格 (美元)', fontsize=12)
ax4.set_title('里程 vs 价格', fontsize=14, fontweight='bold')
# 添加趋势线
z2 = np.polyfit(sample['odometer'], sample['price'], 1)
p2 = np.poly1d(z2)
ax4.plot(sorted(sample['odometer']), p2(sorted(sample['odometer'])), "r--", linewidth=2, label='趋势线')
ax4.legend()

plt.tight_layout()
plt.savefig('建模前_数值特征分布.png', dpi=150, bbox_inches='tight')
plt.close()
print("已保存: 建模前_数值特征分布.png")

# ========================================================
# 阶段5：分类特征与价格关系
# ========================================================
print("\n" + "="*70)
print("阶段5：分类特征与价格关系")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 5.1 汽缸数 vs 价格
ax1 = axes[0, 0]
cylinders_order = ['3 cylinders', '4 cylinders', '5 cylinders', '6 cylinders', '8 cylinders', '10 cylinders', 'other']
cylinders_data = [df_clean[df_clean['cylinders'] == c]['price'].values for c in cylinders_order if c in df_clean['cylinders'].values]
bp1 = ax1.boxplot(cylinders_data, patch_artist=True)
colors1 = plt.cm.Blues(np.linspace(0.3, 0.9, len(cylinders_data)))
for patch, color in zip(bp1['boxes'], colors1):
    patch.set_facecolor(color)
ax1.set_xlabel('汽缸数', fontsize=12)
ax1.set_ylabel('价格 (美元)', fontsize=12)
ax1.set_title('不同汽缸数的车辆价格分布', fontsize=14, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)

# 5.2 车型 vs 价格
ax2 = axes[0, 1]
type_avg = df_clean.groupby('type')['price'].mean().sort_values(ascending=False)
type_data = [df_clean[df_clean['type'] == t]['price'].values for t in type_avg.index]
bp2 = ax2.boxplot(type_data, patch_artist=True)
colors2 = plt.cm.Greens(np.linspace(0.3, 0.9, len(type_data)))
for patch, color in zip(bp2['boxes'], colors2):
    patch.set_facecolor(color)
ax2.set_xticklabels(type_avg.index, rotation=45, ha='right')
ax2.set_xlabel('车辆类型', fontsize=12)
ax2.set_ylabel('价格 (美元)', fontsize=12)
ax2.set_title('不同车型的车辆价格分布', fontsize=14, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

# 5.3 品牌 vs 价格 (Top 15)
ax3 = axes[1, 0]
top_brands = df_clean.groupby('manufacturer')['price'].agg(['mean', 'count']).query('count >= 50').sort_values('mean', ascending=False).head(15)
colors3 = plt.cm.Reds(np.linspace(0.3, 0.9, 15))
bars = ax3.bar(range(len(top_brands)), top_brands['mean'], color=colors3)
ax3.set_xticks(range(len(top_brands)))
ax3.set_xticklabels(top_brands.index, rotation=45, ha='right')
ax3.set_xlabel('品牌', fontsize=12)
ax3.set_ylabel('平均价格 (美元)', fontsize=12)
ax3.set_title('TOP 15 品牌平均价格', fontsize=14, fontweight='bold')
ax3.grid(axis='y', alpha=0.3)
# 添加数值标签
for i, (idx, row) in enumerate(top_brands.iterrows()):
    ax3.text(i, row['mean'] + 500, f'${row["mean"]/1000:.0f}k', ha='center', va='bottom', fontsize=8)

# 5.4 车况 vs 价格
ax4 = axes[1, 1]
condition_order = ['excellent', 'good', 'like new', 'fair', 'salvage']
condition_data = [df_clean[df_clean['condition'] == c]['price'].values for c in condition_order if c in df_clean['condition'].values]
bp4 = ax4.boxplot(condition_data, patch_artist=True)
colors4 = plt.cm.Oranges(np.linspace(0.3, 0.9, len(condition_data)))
for patch, color in zip(bp4['boxes'], colors4):
    patch.set_facecolor(color)
ax4.set_xticklabels(condition_order)
ax4.set_xlabel('车况', fontsize=12)
ax4.set_ylabel('价格 (美元)', fontsize=12)
ax4.set_title('不同车况的车辆价格分布', fontsize=14, fontweight='bold')
ax4.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('建模前_分类特征与价格关系.png', dpi=150, bbox_inches='tight')
plt.close()
print("已保存: 建模前_分类特征与价格关系.png")

# ========================================================
# 阶段6：多变量关系图
# ========================================================
print("\n" + "="*70)
print("阶段6：多变量关系图")
print("="*70)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 6.1 车龄 x 里程 x 价格
ax1 = axes[0]
sample2 = df_clean.sample(min(2000, len(df_clean)), random_state=42)
scatter = ax1.scatter(sample2['age'], sample2['odometer'], c=sample2['price'],
                     cmap='RdYlGn_r', alpha=0.6, s=30)
cbar = plt.colorbar(scatter, ax=ax1)
cbar.set_label('价格 (美元)', fontsize=11)
ax1.set_xlabel('车龄 (年)', fontsize=12)
ax1.set_ylabel('行驶里程 (英里)', fontsize=12)
ax1.set_title('车龄 × 里程 × 价格', fontsize=14, fontweight='bold')

# 6.2 不同车型的价格分布（带抖动的小提琴图）
ax2 = axes[1]
top_types = df_clean['type'].value_counts().head(6).index
df_top_types = df_clean[df_clean['type'].isin(top_types)]
sns.violinplot(data=df_top_types, x='type', y='price', ax=ax2, palette='Set2')
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=30, ha='right')
ax2.set_xlabel('车辆类型', fontsize=12)
ax2.set_ylabel('价格 (美元)', fontsize=12)
ax2.set_title('主要车型价格分布（小提琴图）', fontsize=14, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('建模前_多变量关系图.png', dpi=150, bbox_inches='tight')
plt.close()
print("已保存: 建模前_多变量关系图.png")

# ========================================================
# 阶段7：生成可视化总结报告
# ========================================================
print("\n" + "="*70)
print("阶段7：可视化总结")
print("="*70)

# 关键发现
print("\n【建模前数据可视化关键发现】")
print("\n1. 价格分布特征:")
print(f"   - 价格范围: ${df_clean['price'].min():,.0f} - ${df_clean['price'].max():,.0f}")
print(f"   - 平均价格: ${df_clean['price'].mean():,.0f}")
print(f"   - 价格分布偏度: {df_clean['price'].skew():.2f}（正偏态，右偏）")
print(f"   - 建议：考虑对数变换处理偏态")

print("\n2. 相关性分析:")
print(f"   - year与age相关系数: {corr_all.loc['year', 'age']:.2f}（极强负相关）")
print(f"   - price与age相关系数: {corr_no_year.loc['age', 'price']:.2f}（负相关）")
print(f"   - price与odometer相关系数: {corr_no_year.loc['odometer', 'price']:.2f}（负相关）")
print(f"   - price与cylinders相关系数: {corr_no_year.loc['cylinders', 'price']:.2f}（正相关）")
print(f"   - 建议：移除year避免多重共线性")

print("\n3. 分类特征分析:")
print("   - 汽缸数: 8缸车辆价格明显高于其他类型")
print("   - 车型: pickup/truck类型价格最高")
print("   - 品牌: Lincoln/GMC/Ram等品牌平均价格最高")
print("   - 车况: excellent/good价格差异显著")

print("\n" + "="*70)
print("✅ 建模前数据可视化完成！")
print("="*70)
print("\n生成的可视化图表:")
print("  1. 建模前_价格分布可视化.png")
print("  2. 建模前_相关性热力图.png")
print("  3. 建模前_数值特征分布.png")
print("  4. 建模前_分类特征与价格关系.png")
print("  5. 建模前_多变量关系图.png")
print("="*70)
