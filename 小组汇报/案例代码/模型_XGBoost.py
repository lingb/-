#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
二手车价格预测 - XGBoost模型
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

print("="*70)
print("XGBoost模型 - 二手车价格预测")
print("="*70)

# 数据读取与预处理
df = pd.read_csv('cleanData_2005plus.csv')
df['age'] = 2021 - df['year']

features = ['price', 'odometer', 'age', 'manufacturer', 'model',
            'condition', 'cylinders', 'fuel', 'transmission', 'type']
df_model = df[features].copy()

def remove_outliers_percentile(df, columns, lower=0.01, upper=0.99):
    df_clean = df.copy()
    for col in columns:
        lower_val = df_clean[col].quantile(lower)
        upper_val = df_clean[col].quantile(upper)
        df_clean = df_clean[(df_clean[col] >= lower_val) & (df_clean[col] <= upper_val)]
    return df_clean

df_clean = remove_outliers_percentile(df_model, ['price', 'odometer'], lower=0.01, upper=0.99)
df_clean = df_clean[df_clean['price'] > 0]

X = df_clean.drop(['price'], axis=1)
y = df_clean['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

categorical_cols = ['manufacturer', 'model', 'condition', 'cylinders', 'fuel', 'transmission', 'type']
X_train_encoded = X_train.copy()
X_test_encoded = X_test.copy()

for col in categorical_cols:
    target_mean = y_train.groupby(X_train[col]).mean()
    global_mean = y_train.mean()
    X_train_encoded[col] = X_train[col].map(target_mean).fillna(global_mean)
    X_test_encoded[col] = X_test[col].map(target_mean).fillna(global_mean)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_encoded)
X_test_scaled = scaler.transform(X_test_encoded)

# 模型训练
print("\n训练XGBoost模型...")
model = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1, verbosity=0)
model.fit(X_train_scaled, y_train)

y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

# 评估指标
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')

print(f"\n【模型评估结果】")
print(f"训练集R²: {train_r2:.4f}")
print(f"测试集R²: {test_r2:.4f}")
print(f"测试集RMSE: ${test_rmse:,.2f}")
print(f"测试集MAE: ${test_mae:,.2f}")
print(f"交叉验证R²均值: {cv_scores.mean():.4f}")
print(f"交叉验证R²标准差: {cv_scores.std():.4f}")

# 特征重要性
importance = pd.DataFrame({
    '特征': X.columns,
    '重要性': model.feature_importances_ * 100
}).sort_values('重要性', ascending=False)

print(f"\n【特征重要性】")
print(importance.round(2).to_string(index=False))

# 可视化
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

ax1 = axes[0]
ax1.scatter(y_test, y_test_pred, alpha=0.6, s=30, color='steelblue')
ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='理想拟合线')
ax1.set_title('XGBoost - 实际vs预测', fontsize=14, fontweight='bold')
ax1.set_xlabel('实际价格（美元）', fontsize=12)
ax1.set_ylabel('预测价格（美元）', fontsize=12)
ax1.legend()
ax1.grid(alpha=0.3)

ax2 = axes[1]
residuals = y_test - y_test_pred
sns.histplot(residuals, bins=50, kde=True, ax=ax2, color='steelblue')
ax2.axvline(x=0, color='red', linestyle='--')
ax2.set_title('XGBoost - 残差分布', fontsize=14, fontweight='bold')
ax2.set_xlabel('残差', fontsize=12)
ax2.set_ylabel('频数', fontsize=12)
ax2.grid(axis='y', alpha=0.3)

ax3 = axes[2]
sns.barplot(data=importance, x='重要性', y='特征', palette='coolwarm', ax=ax3)
ax3.set_title('XGBoost - 特征重要性', fontsize=14, fontweight='bold')
ax3.set_xlabel('重要性（%）', fontsize=12)
ax3.set_ylabel('特征', fontsize=12)

plt.tight_layout()
plt.savefig('XGBoost_结果.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"\n结果已保存: XGBoost_结果.png")
print("="*70)
