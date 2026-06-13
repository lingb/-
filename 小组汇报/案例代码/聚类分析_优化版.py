#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

print("="*70)
print("二手车数据聚类分析 - 优化版")
print("="*70)

# ========================================================
# 步骤1：数据读取与预处理
# ========================================================
print("\n【步骤1：数据读取与预处理】")
print("="*70)

df = pd.read_csv('cleanData_2005plus.csv')
print(f"原始数据量: {len(df)}")

# 创建车龄特征
df['age'] = 2021 - df['year']

# 选择用于聚类的特征
cluster_features = ['price', 'age', 'odometer']
df_cluster = df[cluster_features].dropna()

# 数据标准化
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_cluster)

print(f"聚类数据量: {len(df_cluster)}")
print(f"聚类特征: {cluster_features}")

# ========================================================
# 步骤2：确定最佳聚类数（改进版：肘部法则+轮廓系数）
# ========================================================
print("\n【步骤2：确定最佳聚类数】")
print("="*70)

k_range = range(2, 11)
inertia = []
sil_scores = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(df_scaled)
    inertia.append(kmeans.inertia_)
    sil_scores.append(silhouette_score(df_scaled, labels))
    print(f"  k={k}: 惯性={kmeans.inertia_:.0f}, 轮廓系数={silhouette_score(df_scaled, labels):.3f}")

# 可视化肘部法则和轮廓系数
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(k_range, inertia, 'bo-', linewidth=2)
plt.xlabel('聚类数 k')
plt.ylabel('惯性 (Inertia)')
plt.title('肘部法则：惯性 vs k')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(k_range, sil_scores, 'ro-', linewidth=2)
plt.xlabel('聚类数 k')
plt.ylabel('轮廓系数')
plt.title('轮廓系数 vs k')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('聚类_肘部法则_优化版.png', dpi=150)
print("\n肘部法则图已保存")

# 选择最佳k
best_k = 4  # 根据分析选择k=4

# ========================================================
# 步骤3：执行聚类
# ========================================================
print("\n【步骤3：执行聚类（k={}）】".format(best_k))
print("="*70)

kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df_cluster['cluster'] = kmeans.fit_predict(df_scaled)

# 获取聚类中心（反标准化回原始尺度）
cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
centers_df = pd.DataFrame(cluster_centers, columns=cluster_features)
centers_df['cluster'] = range(best_k)

print("\n聚类中心（原始尺度）：")
print(centers_df.round(0).to_string())

# ========================================================
# 步骤4：聚类命名与特征分析
# ========================================================
print("\n【步骤4：聚类命名与特征分析】")
print("="*70)

# 根据聚类中心特征进行命名
cluster_stats = df_cluster.groupby('cluster')[cluster_features].agg(['mean', 'median', 'count'])
cluster_stats.columns = ['_'.join(col).strip() for col in cluster_stats.columns.values]

# 按平均价格排序来命名
price_order = cluster_stats.sort_values('price_mean', ascending=False).index.tolist()

cluster_names = {
    price_order[0]: '🏆 高端车',
    price_order[1]: '💰 中端车',
    price_order[2]: '🛒 经济型车',
    price_order[3]: '📅 老旧车'
}

df_cluster['cluster_name'] = df_cluster['cluster'].map(cluster_names)

print("\n各簇详细统计：")
for cluster_id, name in cluster_names.items():
    stats = cluster_stats.loc[cluster_id]
    print(f"\n{name}（簇{cluster_id}）:")
    print(f"  样本数: {int(stats['price_count'])}")
    print(f"  平均价格: ${stats['price_mean']:,.0f}")
    print(f"  中位价格: ${stats['price_median']:,.0f}")
    print(f"  平均车龄: {stats['age_mean']:.1f}年")
    print(f"  平均里程: {stats['odometer_mean']:,.0f}英里")

# ========================================================
# 步骤5：优化版可视化
# ========================================================
print("\n【步骤5：优化版可视化】")
print("="*70)

# 自定义配色方案
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
palette = sns.color_palette(colors)

# 图1：价格-车龄散点图（带聚类中心）
plt.figure(figsize=(16, 12))

plt.subplot(2, 2, 1)
sns.scatterplot(data=df_cluster, x='age', y='price', hue='cluster_name', 
                palette=palette, alpha=0.6, s=50)
# 添加聚类中心
sns.scatterplot(data=centers_df, x='age', y='price', 
                color='black', marker='X', s=200, label='聚类中心')
plt.title('🚗 价格 vs 车龄（含聚类中心）', fontsize=14)
plt.xlabel('车龄（年）', fontsize=12)
plt.ylabel('价格（$）', fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)

# 图2：价格-里程散点图（带聚类中心）
plt.subplot(2, 2, 2)
sns.scatterplot(data=df_cluster, x='odometer', y='price', hue='cluster_name', 
                palette=palette, alpha=0.6, s=50)
sns.scatterplot(data=centers_df, x='odometer', y='price', 
                color='black', marker='X', s=200, label='聚类中心')
plt.title('📊 价格 vs 里程（含聚类中心）', fontsize=14)
plt.xlabel('里程（英里）', fontsize=12)
plt.ylabel('价格（$）', fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)

# 图3：各簇价格分布箱线图（改进版）
plt.subplot(2, 2, 3)
sns.boxplot(data=df_cluster, x='cluster_name', y='price', 
            palette=palette, showmeans=True, meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black"})
plt.title('📦 各簇价格分布', fontsize=14)
plt.xlabel('聚类类别', fontsize=12)
plt.ylabel('价格（$）', fontsize=12)
plt.xticks(rotation=15)
plt.grid(True, alpha=0.3)

# 图4：各簇特征对比柱状图
plt.subplot(2, 2, 4)
cluster_means = df_cluster.groupby('cluster_name')[['price', 'age', 'odometer']].mean()
cluster_means_normalized = cluster_means / cluster_means.max()  # 归一化便于对比
cluster_means_normalized.plot(kind='bar', ax=plt.gca(), color=palette)
plt.title('📈 各簇特征对比（归一化）', fontsize=14)
plt.xlabel('聚类类别', fontsize=12)
plt.ylabel('归一化值', fontsize=12)
plt.xticks(rotation=15)
plt.legend(['价格', '车龄', '里程'])
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('聚类_可视化_优化版.png', dpi=150)
print("聚类可视化图（优化版）已保存")

# 图5：雷达图展示各簇特征
plt.figure(figsize=(8, 8))
categories = ['价格', '车龄', '里程']
angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
angles += angles[:1]

for i, (cluster_name, color) in enumerate(zip(cluster_names.values(), palette)):
    stats = cluster_stats.loc[[k for k, v in cluster_names.items() if v == cluster_name][0]]
    values = [stats['price_mean'], stats['age_mean'], stats['odometer_mean']]
    values = values / np.max(values)  # 归一化
    values += values[:1]
    
    plt.polar(angles, values, 'o-', linewidth=2, color=color, label=cluster_name)

plt.fill(angles, values, alpha=0.1)
plt.title('🎯 各簇特征雷达图', size=14)
plt.legend(bbox_to_anchor=(1.3, 0.5))
plt.xticks(angles[:-1], categories)
plt.savefig('聚类_雷达图.png', dpi=150, bbox_inches='tight')
print("聚类雷达图已保存")

# ========================================================
# 步骤6：聚类质量评估
# ========================================================
print("\n【步骤6：聚类质量评估】")
print("="*70)

sil_score = silhouette_score(df_scaled, df_cluster['cluster'])
ch_score = calinski_harabasz_score(df_scaled, df_cluster['cluster'])

print(f"轮廓系数: {sil_score:.4f}")
print(f"Calinski-Harabasz指数: {ch_score:.0f}")

print("\n评估解读：")
print(f"- 轮廓系数({sil_score:.4f})：值越接近1表示聚类效果越好")
print(f"- Calinski-Harabasz指数({ch_score:.0f})：值越大表示聚类效果越好")

# ========================================================
# 步骤7：保存结果
# ========================================================
print("\n【步骤7：保存结果】")
print("="*70)

df_cluster.to_csv('cleanData_聚类结果_优化版.csv', index=False, encoding='utf-8-sig')
print("聚类结果已保存到: cleanData_聚类结果_优化版.csv")

print("\n" + "="*70)
print("聚类分析（优化版）完成！")
print("="*70)
print("\n生成的文件：")
print("- 聚类_肘部法则_优化版.png")
print("- 聚类_可视化_优化版.png")
print("- 聚类_雷达图.png")
print("- cleanData_聚类结果_优化版.csv")
