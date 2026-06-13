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
print("二手车数据聚类分析")
print("="*70)

df = pd.read_csv('cleanData_最终处理版.csv')
print(f"数据量: {len(df)} 条")

cluster_features = ['price', 'age', 'odometer']
df_cluster = df[cluster_features].copy()

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_cluster)

print("\n" + "="*70)
print("【步骤1：确定最佳聚类数】")
print("="*70)

inertia = []
sil_scores = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(df_scaled)
    inertia.append(kmeans.inertia_)
    sil_scores.append(silhouette_score(df_scaled, labels))
    print(f"  k={k}: 惯性={kmeans.inertia_:.0f}, 轮廓系数={silhouette_score(df_scaled, labels):.3f}")

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(k_range, inertia, 'bo-')
plt.title('肘部法则', fontsize=14)
plt.xlabel('聚类数 k', fontsize=12)
plt.ylabel('惯性值', fontsize=12)
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(k_range, sil_scores, 'ro-')
plt.title('轮廓系数', fontsize=14)
plt.xlabel('聚类数 k', fontsize=12)
plt.ylabel('轮廓系数', fontsize=12)
plt.grid(True)

plt.tight_layout()
plt.savefig('聚类_肘部法则.png', dpi=150)
print("\n肘部法则图已保存")

print("\n" + "="*70)
print("【步骤2：执行聚类】")
print("="*70)

best_k = 4
print(f"选择最佳聚类数: k={best_k}")

kmeans = KMeans(n_clusters=best_k, random_state=42)
df['cluster'] = kmeans.fit_predict(df_scaled)

print("\n聚类完成！各簇样本数：")
print(df['cluster'].value_counts().sort_index())

print("\n" + "="*70)
print("【步骤3：聚类结果分析】")
print("="*70)

cluster_stats = df.groupby('cluster')[['price', 'age', 'odometer']].agg(['mean', 'median', 'count'])
print("\n各簇统计特征：")
print(cluster_stats.round(0).to_string())

cluster_names = {
    0: '经济型车',
    1: '中端车',
    2: '高端车',
    3: '老旧车'
}
df['cluster_name'] = df['cluster'].map(cluster_names)

print("\n各簇特征描述：")
for cluster_id, name in cluster_names.items():
    stats = cluster_stats.loc[cluster_id]
    print(f"\n  {name} (簇{cluster_id}):")
    print(f"    样本数: {stats['price']['count']}")
    print(f"    平均价格: ${stats['price']['mean']:,.0f}")
    print(f"    平均车龄: {stats['age']['mean']:.1f}年")
    print(f"    平均里程: {stats['odometer']['mean']:,.0f}英里")

print("\n" + "="*70)
print("【步骤4：聚类可视化】")
print("="*70)

plt.figure(figsize=(14, 10))

plt.subplot(2, 2, 1)
sns.scatterplot(data=df, x='age', y='price', hue='cluster_name', palette='viridis', alpha=0.7)
plt.title('价格 vs 车龄', fontsize=14)
plt.xlabel('车龄（年）', fontsize=12)
plt.ylabel('价格（$）', fontsize=12)

plt.subplot(2, 2, 2)
sns.scatterplot(data=df, x='odometer', y='price', hue='cluster_name', palette='viridis', alpha=0.7)
plt.title('价格 vs 里程', fontsize=14)
plt.xlabel('里程（英里）', fontsize=12)
plt.ylabel('价格（$）', fontsize=12)

plt.subplot(2, 2, 3)
sns.scatterplot(data=df, x='age', y='odometer', hue='cluster_name', palette='viridis', alpha=0.7)
plt.title('里程 vs 车龄', fontsize=14)
plt.xlabel('车龄（年）', fontsize=12)
plt.ylabel('里程（英里）', fontsize=12)

plt.subplot(2, 2, 4)
sns.boxplot(data=df, x='cluster_name', y='price', palette='viridis')
plt.title('各簇价格分布', fontsize=14)
plt.xlabel('聚类', fontsize=12)
plt.ylabel('价格（$）', fontsize=12)
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('聚类_可视化.png', dpi=150)
print("聚类可视化图已保存")

print("\n" + "="*70)
print("【步骤5：聚类质量评估】")
print("="*70)

print(f"轮廓系数: {silhouette_score(df_scaled, df['cluster']):.4f}")
print(f"Calinski-Harabasz指数: {calinski_harabasz_score(df_scaled, df['cluster']):.0f}")

print("\n" + "="*70)
print("【步骤6：保存结果】")
print("="*70)

df.to_csv('cleanData_聚类结果.csv', index=False, encoding='utf-8-sig')
print("聚类结果已保存到: cleanData_聚类结果.csv")

print("\n" + "="*70)
print("聚类分析完成！")
print("="*70)
