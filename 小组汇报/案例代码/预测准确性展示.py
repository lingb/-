#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import Lasso

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ========================================================
# 1. 加载和处理数据
# ========================================================
print("="*70)
print("二手车价格预测模型 - 预测准确性展示")
print("="*70)

df = pd.read_csv('cleanData_最终处理版.csv')

# 编码
categorical_cols = ['manufacturer', 'model', 'condition', 'cylinders', 'fuel', 'transmission', 'type']
label_encoders = {}
df_encoded = df.copy()
for col in categorical_cols:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
    label_encoders[col] = le

# 划分数据
X = df_encoded.drop(['price', 'year'], axis=1)
y = df_encoded['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# 标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 训练模型
model = Lasso(alpha=54.62, max_iter=10000, random_state=42)
model.fit(X_train_scaled, y_train)

# 预测
y_pred = model.predict(X_test_scaled)

# ========================================================
# 2. 详细预测示例
# ========================================================
print("\n" + "="*70)
print("【预测示例展示】")
print("="*70)

# 创建结果DataFrame
results_df = pd.DataFrame({
    '实际价格': y_test.values,
    '预测价格': y_pred,
    '误差': y_pred - y_test.values,
    '误差百分比': ((y_pred - y_test.values) / y_test.values * 100)
})

# 添加原始特征信息
X_test_original = X_test.copy()
X_test_original['actual_price'] = y_test.values
X_test_original['predicted_price'] = y_pred

# 解码分类变量
for col in categorical_cols:
    X_test_original[col + '_decode'] = label_encoders[col].inverse_transform(X_test_original[col].astype(int))

# 选择20个代表性样本展示
print("\n【示例1：价格预测对比（20个样本）】")
print("-" * 100)
print(f"{'品牌':<12} {'车型':<20} {'车龄':<6} {'里程':<12} {'实际价格':<12} {'预测价格':<12} {'误差%':<8}")
print("-" * 100)

for i in range(20):
    row = X_test_original.iloc[i]
    actual = row['actual_price']
    pred = row['predicted_price']
    error_pct = ((pred - actual) / actual) * 100

    manufacturer = row['manufacturer_decode']
    model_name = row['model_decode'][:18]
    age = int(row['age'])
    odometer = int(row['odometer'])

    print(f"{manufacturer:<12} {model_name:<20} {age:<6} {odometer:<12,} ${actual:<11,.0f} ${pred:<11,.0f} {error_pct:>+6.1f}%")

# ========================================================
# 3. 误差分布分析
# ========================================================
print("\n" + "="*70)
print("【误差分布统计】")
print("="*70)

errors = y_pred - y_test.values
abs_errors = np.abs(errors)
error_pct = np.abs(errors) / y_test.values * 100

print(f"误差均值:     ${np.mean(errors):,.2f}")
print(f"误差标准差:   ${np.std(errors):,.2f}")
print(f"平均绝对误差: ${np.mean(abs_errors):,.2f}")
print(f"平均百分比误差: {np.mean(error_pct):.2f}%")
print(f"中位数绝对误差: ${np.median(abs_errors):,.2f}")

print(f"\n误差范围:")
print(f"  最小误差: ${np.min(errors):,.2f}")
print(f"  最大误差: ${np.max(errors):,.2f}")
print(f"  50%的预测误差在 ${np.percentile(abs_errors, 50):,.2f} 以内")
print(f"  80%的预测误差在 ${np.percentile(abs_errors, 80):,.2f} 以内")
print(f"  90%的预测误差在 ${np.percentile(abs_errors, 90):,.2f} 以内")

# ========================================================
# 4. 准确度分段统计
# ========================================================
print("\n" + "="*70)
print("【预测准确度分段统计】")
print("="*70)

within_5pct = np.sum(error_pct <= 5) / len(error_pct) * 100
within_10pct = np.sum(error_pct <= 10) / len(error_pct) * 100
within_20pct = np.sum(error_pct <= 20) / len(error_pct) * 100
within_30pct = np.sum(error_pct <= 30) / len(error_pct) * 100

print(f"误差 ≤ 5% 的样本:  {within_5pct:.1f}% ({np.sum(error_pct <= 5)} 条)")
print(f"误差 ≤ 10% 的样本: {within_10pct:.1f}% ({np.sum(error_pct <= 10)} 条)")
print(f"误差 ≤ 20% 的样本: {within_20pct:.1f}% ({np.sum(error_pct <= 20)} 条)")
print(f"误差 ≤ 30% 的样本: {within_30pct:.1f}% ({np.sum(error_pct <= 30)} 条)")

# ========================================================
# 5. 预测效果图展示
# ========================================================
print("\n" + "="*70)
print("【生成可视化图表】")
print("="*70)

# 图1：实际值 vs 预测值（带误差线）
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
# 随机抽取200个点展示
sample_idx = np.random.choice(len(y_test), 200, replace=False)
sample_actual = y_test.values[sample_idx]
sample_pred = y_pred[sample_idx]

plt.scatter(sample_actual, sample_pred, alpha=0.6, s=50)
plt.plot([0, 60000], [0, 60000], 'r--', linewidth=2, label='理想预测线')
plt.xlabel('实际价格（美元）', fontsize=12)
plt.ylabel('预测价格（美元）', fontsize=12)
plt.title('实际价格 vs 预测价格', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)

# 图2：误差百分比分布
plt.subplot(1, 2, 2)
error_pct_clipped = np.clip(error_pct, 0, 50)  # 截断到0-50%便于展示
plt.hist(error_pct_clipped, bins=50, edgecolor='black', alpha=0.7)
plt.axvline(x=10, color='r', linestyle='--', label='±10%误差线')
plt.axvline(x=20, color='orange', linestyle='--', label='±20%误差线')
plt.xlabel('误差百分比（%）', fontsize=12)
plt.ylabel('频数', fontsize=12)
plt.title('预测误差百分比分布', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('预测准确性_对比图.png', dpi=150, bbox_inches='tight')
plt.close()
print("  已保存: 预测准确性_对比图.png")

# 图3：误差分布直方图
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
errors_clipped = np.clip(errors, -15000, 15000)
plt.hist(errors_clipped, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
plt.axvline(x=0, color='red', linestyle='--', linewidth=2)
plt.axvline(x=-5000, color='orange', linestyle=':', linewidth=2)
plt.axvline(x=5000, color='orange', linestyle=':', linewidth=2)
plt.xlabel('预测误差（美元）', fontsize=12)
plt.ylabel('频数', fontsize=12)
plt.title('预测误差分布（截断到±$15,000）', fontsize=14)
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
# 误差百分比箱线图
error_pct_by_range = []
price_ranges = [(0, 10000), (10000, 20000), (20000, 30000), (30000, 50000), (50000, 100000)]
range_labels = ['$0-10k', '$10k-20k', '$20k-30k', '$30k-50k', '$50k-100k']

for low, high in price_ranges:
    mask = (y_test.values >= low) & (y_test.values < high)
    if np.sum(mask) > 0:
        error_pct_by_range.append(error_pct[mask])

plt.boxplot(error_pct_by_range, labels=range_labels)
plt.axhline(y=10, color='r', linestyle='--', label='±10%误差线')
plt.axhline(y=20, color='orange', linestyle='--', label='±20%误差线')
plt.xlabel('价格区间', fontsize=12)
plt.ylabel('误差百分比（%）', fontsize=12)
plt.title('不同价格区间的预测误差', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('预测准确性_误差分析.png', dpi=150, bbox_inches='tight')
plt.close()
print("  已保存: 预测准确性_误差分析.png")

# ========================================================
# 6. 典型案例分析
# ========================================================
print("\n" + "="*70)
print("【典型案例分析】")
print("="*70)

# 最佳预测案例
best_idx = np.argmin(np.abs(errors))
# 最差预测案例
worst_idx = np.argmax(np.abs(errors))

def print_case(idx, label):
    row = X_test_original.iloc[idx]
    actual = row['actual_price']
    pred = row['predicted_price']
    error = pred - actual
    error_pct = (error / actual) * 100

    print(f"\n{label}")
    print(f"  品牌: {row['manufacturer_decode']}")
    print(f"  车型: {row['model_decode']}")
    print(f"  车龄: {int(row['age'])}年")
    print(f"  里程: {int(row['odometer']):,}英里")
    print(f"  燃料: {row['fuel_decode']}")
    print(f"  变速箱: {row['transmission_decode']}")
    print(f"  实际价格: ${actual:,.0f}")
    print(f"  预测价格: ${pred:,.0f}")
    print(f"  误差: ${error:,.0f} ({error_pct:+.1f}%)")

print_case(best_idx, "【预测最准确的案例】")
print_case(worst_idx, "【预测误差最大的案例】")

# ========================================================
# 7. 价格区间预测效果
# ========================================================
print("\n" + "="*70)
print("【不同价格区间的预测效果】")
print("="*70)

price_ranges = [
    (0, 10000, '经济型 ($0-10k)'),
    (10000, 20000, '入门型 ($10k-20k)'),
    (20000, 30000, '中级型 ($20k-30k)'),
    (30000, 50000, '中高级 ($30k-50k)'),
    (50000, 100000, '高级型 ($50k-100k)')
]

print(f"\n{'价格区间':<20} {'样本数':<10} {'平均误差%':<12} {'R²':<10}")
print("-" * 55)

for low, high, name in price_ranges:
    mask = (y_test.values >= low) & (y_test.values < high)
    if np.sum(mask) > 0:
        range_actual = y_test.values[mask]
        range_pred = y_pred[mask]
        range_errors = range_pred - range_actual
        mean_error_pct = np.mean(np.abs(range_errors) / range_actual) * 100
        range_r2 = 1 - np.sum(range_errors**2) / np.sum((range_actual - np.mean(range_actual))**2)
        print(f"{name:<20} {np.sum(mask):<10} {mean_error_pct:<12.1f} {range_r2:<10.4f}")

# ========================================================
# 8. 总结
# ========================================================
print("\n" + "="*70)
print("【总结】")
print("="*70)

overall_r2 = 1 - np.sum(errors**2) / np.sum((y_test.values - np.mean(y_test.values))**2)
print(f"模型整体 R²: {overall_r2:.4f}")
print(f"平均绝对误差: ${np.mean(abs_errors):,.0f}")
print(f"预测准确度（误差≤10%）: {within_10pct:.1f}%")
print(f"预测准确度（误差≤20%）: {within_20pct:.1f}%")

print("\n【模型说明】")
print("  该模型基于Lasso回归算法，输入车辆特征后输出预测价格。")
print("  误差在10%以内的预测可视为较准确，20%以内为可接受范围。")
print("  对于稀有车型或极端情况，预测误差可能较大。")

print("\n" + "="*70)
print("预测准确性展示完成！")
print("="*70)