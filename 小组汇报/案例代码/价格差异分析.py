#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

print("="*70)
print("二手车价格差异分析：汽缸数 / 车型 / 品牌")
print("="*70)

df = pd.read_csv('cleanData_最终处理版.csv')

# ========================================================
# 1. 汽缸数对价格的影响分析
# ========================================================
print("\n" + "="*70)
print("【一、汽缸数(Cylinders)对价格的影响】")
print("="*70)

cylinders_order = ['3 cylinders', '4 cylinders', '5 cylinders', '6 cylinders', '8 cylinders', '10 cylinders', 'other']
cylinders_avg = df.groupby('cylinders')['price'].agg(['mean', 'median', 'std', 'count']).sort_values('mean', ascending=False)

print("\n各汽缸数类型的平均价格：")
print("-" * 60)
for cyl in cylinders_avg.index:
    avg = cylinders_avg.loc[cyl, 'mean']
    med = cylinders_avg.loc[cyl, 'median']
    cnt = cylinders_avg.loc[cyl, 'count']
    std = cylinders_avg.loc[cyl, 'std']
    print(f"  {cyl:<15}: 平均 ${avg:>10,.0f} | 中位数 ${med:>10,.0f} | 样本 {cnt:>4}")

print("\n关键发现：")
print(f"  - 汽缸数最多(10缸)平均价格: ${cylinders_avg.loc['10 cylinders', 'mean']:,.0f}")
print(f"  - 汽缸数最少(3缸)平均价格: ${cylinders_avg.loc['3 cylinders', 'mean']:,.0f}")
print(f"  - 价格差异倍数: {cylinders_avg.loc['10 cylinders', 'mean'] / cylinders_avg.loc['3 cylinders', 'mean']:.1f}倍")

# ========================================================
# 2. 车型(type)对价格的影响分析
# ========================================================
print("\n" + "="*70)
print("【二、车型(Type)对价格的影响】")
print("="*70)

type_avg = df.groupby('type')['price'].agg(['mean', 'median', 'std', 'count']).sort_values('mean', ascending=False)

print("\n各车型的平均价格：")
print("-" * 60)
for t in type_avg.index:
    avg = type_avg.loc[t, 'mean']
    med = type_avg.loc[t, 'median']
    cnt = type_avg.loc[t, 'count']
    print(f"  {t:<12}: 平均 ${avg:>10,.0f} | 中位数 ${med:>10,.0f} | 样本 {cnt:>4}")

print("\n关键发现：")
print(f"  - 最贵车型: {type_avg.index[0]} (平均 ${type_avg.iloc[0]['mean']:,.0f})")
print(f"  - 最便宜车型: {type_avg.index[-1]} (平均 ${type_avg.iloc[-1]['mean']:,.0f})")
print(f"  - 价格差异倍数: {type_avg.iloc[0]['mean'] / type_avg.iloc[-1]['mean']:.1f}倍")

# 卡车 vs 轿车对比
truck_price = df[df['type'].isin(['truck', 'pickup'])]['price'].mean()
car_price = df[df['type'].isin(['sedan', 'coupe'])]['price'].mean()
print(f"\n  - 卡车/皮卡平均价格: ${truck_price:,.0f}")
print(f"  - 轿车/跑车平均价格: ${car_price:,.0f}")
print(f"  - 卡车比轿车贵: {(truck_price/car_price - 1)*100:.1f}%")

# ========================================================
# 3. 品牌(manufacturer)对价格的影响分析
# ========================================================
print("\n" + "="*70)
print("【三、品牌(Manufacturer)对价格的影响】")
print("="*70)

# 只分析样本量超过50的品牌
manufacturer_stats = df.groupby('manufacturer')['price'].agg(['mean', 'median', 'std', 'count'])
manufacturer_stats = manufacturer_stats[manufacturer_stats['count'] >= 50].sort_values('mean', ascending=False)

print("\n各品牌平均价格（样本量≥50）：")
print("-" * 60)
for mfr in manufacturer_stats.index:
    avg = manufacturer_stats.loc[mfr, 'mean']
    med = manufacturer_stats.loc[mfr, 'median']
    cnt = manufacturer_stats.loc[mfr, 'count']
    print(f"  {mfr:<15}: 平均 ${avg:>10,.0f} | 中位数 ${med:>10,.0f} | 样本 {cnt:>4}")

print("\n关键发现：")
top3 = manufacturer_stats.head(3).index.tolist()
bottom3 = manufacturer_stats.tail(3).index.tolist()
print(f"  - 最贵品牌TOP3: {', '.join(top3)}")
print(f"  - 最便宜品牌TOP3: {', '.join(bottom3)}")
print(f"  - 最高与最低品牌价差: ${manufacturer_stats.iloc[0]['mean'] - manufacturer_stats.iloc[-1]['mean']:,.0f}")

# 豪华品牌 vs 普通品牌
luxury_brands = ['bmw', 'mercedes-benz', 'lexus', 'cadillac', 'infiniti', 'acura', 'audi', 'porsche']
economy_brands = ['kia', 'hyundai', 'nissan', 'chevrolet', 'ford', 'dodge', 'chrysler']

luxury_avg = df[df['manufacturer'].isin(luxury_brands)]['price'].mean()
economy_avg = df[df['manufacturer'].isin(economy_brands)]['price'].mean()

print(f"\n  - 豪华品牌平均价格: ${luxury_avg:,.0f}")
print(f"  - 普通品牌平均价格: ${economy_avg:,.0f}")
print(f"  - 豪华品牌溢价: {(luxury_avg/economy_avg - 1)*100:.1f}%")

# ========================================================
# 4. 创建可视化图表
# ========================================================
print("\n" + "="*70)
print("【四、生成可视化图表】")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# 图1：汽缸数 vs 价格
ax1 = axes[0, 0]
cylinders_data = []
cylinders_labels = []
for cyl in cylinders_order:
    if cyl in df['cylinders'].values:
        cylinders_data.append(df[df['cylinders'] == cyl]['price'].values)
        cylinders_labels.append(cyl.replace(' cylinders', '\ncyl'))

bp1 = ax1.boxplot(cylinders_data, labels=cylinders_labels, patch_artist=True)
colors1 = plt.cm.Blues(np.linspace(0.3, 0.9, len(cylinders_labels)))
for patch, color in zip(bp1['boxes'], colors1):
    patch.set_facecolor(color)
ax1.set_title('Cylinders vs Price\n(汽缸数对价格的影响)', fontsize=14, fontweight='bold')
ax1.set_xlabel('Cylinders (汽缸数)', fontsize=12)
ax1.set_ylabel('Price ($)', fontsize=12)
ax1.tick_params(axis='y', labelsize=10)
ax1.grid(axis='y', alpha=0.3)

# 图2：车型 vs 价格
ax2 = axes[0, 1]
type_data = []
type_labels = []
type_order = type_avg.index.tolist()
for t in type_order:
    type_data.append(df[df['type'] == t]['price'].values)
    type_labels.append(t)

bp2 = ax2.boxplot(type_data, labels=type_labels, patch_artist=True)
colors2 = plt.cm.Greens(np.linspace(0.3, 0.9, len(type_labels)))
for patch, color in zip(bp2['boxes'], colors2):
    patch.set_facecolor(color)
ax2.set_title('Vehicle Type vs Price\n(车型对价格的影响)', fontsize=14, fontweight='bold')
ax2.set_xlabel('Type (车型)', fontsize=12)
ax2.set_ylabel('Price ($)', fontsize=12)
ax2.tick_params(axis='x', rotation=45, labelsize=9)
ax2.tick_params(axis='y', labelsize=10)
ax2.grid(axis='y', alpha=0.3)

# 图3：品牌 vs 价格（Top 15）
ax3 = axes[1, 0]
top_brands = manufacturer_stats.head(15)
bars = ax3.bar(range(len(top_brands)), top_brands['mean'], color=plt.cm.Reds(np.linspace(0.4, 0.9, 15)))
ax3.set_xticks(range(len(top_brands)))
ax3.set_xticklabels(top_brands.index, rotation=45, ha='right', fontsize=10)
ax3.set_title('Top 15 Brands by Average Price\n(品牌平均价格TOP15)', fontsize=14, fontweight='bold')
ax3.set_xlabel('Brand (品牌)', fontsize=12)
ax3.set_ylabel('Average Price ($)', fontsize=12)
ax3.tick_params(axis='y', labelsize=10)
ax3.grid(axis='y', alpha=0.3)

for i, (idx, row) in enumerate(top_brands.iterrows()):
    ax3.text(i, row['mean'] + 500, f'${row["mean"]/1000:.0f}k', ha='center', va='bottom', fontsize=8)

# 图4：品牌价格分布（箱线图，展示价格差异）
ax4 = axes[1, 1]
top10_brands = manufacturer_stats.head(10).index.tolist()
brand_data = []
brand_labels = []
for brand in top10_brands:
    brand_data.append(df[df['manufacturer'] == brand]['price'].values)
    brand_labels.append(brand)

bp4 = ax4.boxplot(brand_data, labels=brand_labels, patch_artist=True)
colors4 = plt.cm.Oranges(np.linspace(0.3, 0.9, len(brand_labels)))
for patch, color in zip(bp4['boxes'], colors4):
    patch.set_facecolor(color)
ax4.set_title('Top 10 Brands Price Distribution\n(品牌价格分布对比)', fontsize=14, fontweight='bold')
ax4.set_xlabel('Brand (品牌)', fontsize=12)
ax4.set_ylabel('Price ($)', fontsize=12)
ax4.set_xticklabels(brand_labels, rotation=45, ha='right', fontsize=10)
ax4.set_title('Top 10 Brands Price Distribution\n(品牌价格分布对比)', fontsize=14, fontweight='bold')
ax4.set_xlabel('Brand (品牌)', fontsize=12)
ax4.set_ylabel('Price ($)', fontsize=12)
ax4.tick_params(axis='y', labelsize=10)
ax4.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('价格差异分析_综合图.png', dpi=150, bbox_inches='tight')
print("  已保存: 价格差异分析_综合图.png")

# ========================================================
# 5. 统计检验：验证差异是否显著
# ========================================================
print("\n" + "="*70)
print("【五、统计显著性检验】")
print("="*70)

# 汽缸数的ANOVA检验
cylinders_groups = [df[df['cylinders'] == cyl]['price'].values for cyl in ['4 cylinders', '6 cylinders', '8 cylinders'] if cyl in df['cylinders'].values]
if len(cylinders_groups) == 3:
    f_stat, p_val = stats.f_oneway(*cylinders_groups)
    print(f"\n汽缸数差异检验 (ANOVA): F={f_stat:.2f}, p-value={p_val:.2e}")
    print(f"  → 汽缸数对价格的影响{'显著' if p_val < 0.05 else '不显著'} (α=0.05)")

# 车型的ANOVA检验
type_groups = [df[df['type'] == t]['price'].values for t in ['sedan', 'SUV', 'truck', 'pickup'] if t in df['type'].values]
if len(type_groups) == 4:
    f_stat, p_val = stats.f_oneway(*type_groups)
    print(f"\n车型差异检验 (ANOVA): F={f_stat:.2f}, p-value={p_val:.2e}")
    print(f"  → 车型对价格的影响{'显著' if p_val < 0.05 else '不显著'} (α=0.05)")

# 品牌的ANOVA检验
brand_groups = [df[df['manufacturer'] == m]['price'].values for m in ['ford', 'toyota', 'bmw', 'chevrolet'] if m in df['manufacturer'].values]
if len(brand_groups) == 4:
    f_stat, p_val = stats.f_oneway(*brand_groups)
    print(f"\n品牌差异检验 (ANOVA): F={f_stat:.2f}, p-value={p_val:.2e}")
    print(f"  → 品牌对价格的影响{'显著' if p_val < 0.05 else '不显著'} (α=0.05)")

# ========================================================
# 6. 交叉分析：汽缸数 x 车型
# ========================================================
print("\n" + "="*70)
print("【六、交叉分析：汽缸数 × 车型】")
print("="*70)

cross_tab = pd.pivot_table(df, values='price', index='type', columns='cylinders', aggfunc='mean')
print("\n各车型在不同汽缸数下的平均价格：")
print(cross_tab.round(0).to_string())

plt.show()
print("\n分析完成！")
