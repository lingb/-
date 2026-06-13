#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
二手车价格预测 - 多模型对比与评估（全模型优化版）
修复了命名KeyError、子图未绑定Bug，并升级了编码策略
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# 动态导入三大顶级Boosting库
try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("⚠️ XGBoost未安装，将跳过")

try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    print("⚠️ CatBoost未安装，将跳过")

try:
    from lightgbm import LGBMRegressor
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("⚠️ LightGBM未安装，将跳过")

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

print("="*70)
print("二手车价格预测 - 多模型对比与评估")
print("="*70)

# ========================================================
# 阶段1：数据读取与预处理
# ========================================================
print("\n" + "="*70)
print("阶段1：数据读取与预处理")
print("="*70)

df = pd.read_csv('cleanData_2005plus.csv')
print(f"原始数据量: {len(df)}")

# 创建车龄特征
df['age'] = 2021 - df['year']

# 选择特征（排除year，避免与age共线性）
features = ['price', 'odometer', 'age', 'manufacturer', 'model',
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

# ========================================================
# 阶段2：数据划分与目标编码（替换LabelEncoder，防止线性模型逻辑崩溃）
# ========================================================
print("\n" + "="*70)
print("阶段2：数据划分与学术规范特征工程")
print("="*70)

X = df_clean.drop(['price'], axis=1)
y = df_clean['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# 执行Target Encoding（使用训练集均值映射，严防数据泄露）
categorical_cols = ['manufacturer', 'model', 'condition', 'cylinders', 'fuel', 'transmission', 'type']
X_train_encoded = X_train.copy()
X_test_encoded = X_test.copy()

for col in categorical_cols:
    target_mean = y_train.groupby(X_train[col]).mean()
    global_mean = y_train.mean()
    X_train_encoded[col] = X_train[col].map(target_mean).fillna(global_mean)
    X_test_encoded[col] = X_test[col].map(target_mean).fillna(global_mean)

print(f"训练集: {len(X_train_encoded)} 条")
print(f"测试集: {len(X_test_encoded)} 条")

# 全局特征标准化（确保所有模型在同一起跑线，特别是Lasso和Ridge）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_encoded)
X_test_scaled = scaler.transform(X_test_encoded)

# ========================================================
# 阶段3：模型训练
# ========================================================
print("\n" + "="*70)
print("阶段3：模型训练")
print("="*70)

# 统一定义模型字典（注意：这里键名统一使用 '梯度提升' 避免KeyError）
models = {
    'Lasso回归': Lasso(alpha=0.01, max_iter=10000, random_state=42),
    '岭回归': Ridge(alpha=1.0, random_state=42),
    '随机森林': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
    '梯度提升': GradientBoostingRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
}

if HAS_XGBOOST:
    models['XGBoost'] = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1, verbosity=0)

if HAS_CATBOOST:
    models['CatBoost'] = CatBoostRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, verbose=0)

if HAS_LIGHTGBM:
    models['LightGBM'] = LGBMRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1, verbose=-1)

results = []
trained_models = {}

for name, model in models.items():
    print(f"正在训练: {name}...")
    
    # 统一使用标准化后的特征矩阵
    model.fit(X_train_scaled, y_train)
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    trained_models[name] = model
    
    # 计算评估指标
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    # 5折交叉验证稳定性评估
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2', n_jobs=-1)
    
    results.append({
        '模型': name,
        '训练RMSE': train_rmse,
        '测试RMSE': test_rmse,
        '训练MAE': train_mae,
        '测试MAE': test_mae,
        '训练R2': train_r2,
        '测试R2': test_r2,
        'CV R2均值': cv_scores.mean(),
        'CV R2标准差': cv_scores.std()
    })
    print(f"   测试R2: {test_r2:.4f} | 测试MAE: ${test_mae:,.2f}")

# ========================================================
# 阶段4：结果对比
# ========================================================
print("\n" + "="*70)
print("阶段4：结果对比")
print("="*70)

results_df = pd.DataFrame(results)
print("\n【模型性能对比】")
print(results_df.round(4).to_string(index=False))

results_df.to_csv('多模型对比结果.csv', index=False, encoding='utf-8-sig')

# ========================================================
# 阶段5：可视化对比
# ========================================================
print("\n" + "="*70)
print("阶段5：可视化对比")
print("="*70)

# 5.1 RMSE与R2对比图（修复：绑定ax参数）
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

sns.barplot(data=results_df, x='模型', y='测试RMSE', palette='viridis', ax=axes[0])
axes[0].set_title('各模型测试集RMSE对比', fontsize=14, fontweight='bold')
axes[0].set_xlabel('模型', fontsize=12)
axes[0].set_ylabel('RMSE (美元)', fontsize=12)
axes[0].tick_params(axis='x', rotation=45)
axes[0].grid(axis='y', alpha=0.3)

sns.barplot(data=results_df, x='模型', y='测试R2', palette='viridis', ax=axes[1])
axes[1].set_title('各模型测试集R2对比', fontsize=14, fontweight='bold')
axes[1].set_xlabel('模型', fontsize=12)
axes[1].set_ylabel('R2', fontsize=12)
axes[1].tick_params(axis='x', rotation=45)
axes[1].grid(axis='y', alpha=0.3)
axes[1].set_ylim(0, 1)

plt.tight_layout()
plt.savefig('多模型_RMSE_R2对比.png', dpi=150, bbox_inches='tight')
plt.close()

# 5.2 特征重要性（修复：Key统一为'梯度提升'）
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

rf_importance = pd.DataFrame({
    '特征': X.columns,
    '重要性': trained_models['随机森林'].feature_importances_
}).sort_values('重要性', ascending=False).head(10)
sns.barplot(data=rf_importance, x='重要性', y='特征', palette='coolwarm', ax=axes[0])
axes[0].set_title('随机森林特征重要性', fontsize=14, fontweight='bold')

gb_importance = pd.DataFrame({
    '特征': X.columns,
    '重要性': trained_models['梯度提升'].feature_importances_
}).sort_values('重要性', ascending=False).head(10)
sns.barplot(data=gb_importance, x='重要性', y='特征', palette='coolwarm', ax=axes[1])
axes[1].set_title('梯度提升特征重要性', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('多模型_特征重要性对比.png', dpi=150, bbox_inches='tight')
plt.close()

# 5.3 残差分布对比（修复：动态自适应网格，完美适配多模型数量）
num_models = len(models)
cols = 3
rows = (num_models + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(18, rows * 4))
axes = axes.flatten()

for i, name in enumerate(models.keys()):
    ax = axes[i]
    y_pred = trained_models[name].predict(X_test_scaled)
    residuals = y_test - y_pred
    sns.histplot(residuals, bins=50, kde=True, ax=ax, color='steelblue')
    ax.axvline(x=0, color='red', linestyle='--')
    ax.set_title(f'{name} 残差分布', fontsize=12, fontweight='bold')
    ax.set_xlabel('残差', fontsize=10)
    ax.set_ylabel('频数', fontsize=10)
    ax.grid(axis='y', alpha=0.3)

# 隐藏多余的空白子图占位符
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.savefig('多模型_残差分布对比.png', dpi=150, bbox_inches='tight')
plt.close()

# 5.4 实际vs预测散点图（最佳模型）
best_model_name = results_df.sort_values('测试R2', ascending=False).iloc[0]['模型']
print(f"\n最佳模型: {best_model_name}")

fig, ax = plt.subplots(figsize=(10, 6))
y_pred = trained_models[best_model_name].predict(X_test_scaled)

ax.scatter(y_test, y_pred, alpha=0.6, s=30, color='steelblue')
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='理想拟合线')
ax.set_title(f'{best_model_name} - 实际价格 vs 预测价格', fontsize=14, fontweight='bold')
ax.set_xlabel('实际价格', fontsize=12)
ax.set_ylabel('预测价格', fontsize=12)
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f'{best_model_name}_实际vs预测.png', dpi=150, bbox_inches='tight')
plt.close()

# ========================================================
# 阶段6-8：模型评估总结与分析（已修复命名匹配）
# ========================================================
print("\n" + "="*70)
print("阶段6：模型评估总结")
print("="*70)

sorted_results = results_df.sort_values('测试R2', ascending=False)

print("\n【模型性能排名】")
print("-" * 70)
print(f"{'排名':<6} {'模型':<10} {'测试R2':<10} {'测试RMSE':<15} {'测试MAE':<15}")
print("-" * 70)
for i, (_, row) in enumerate(sorted_results.iterrows(), 1):
    print(f"{i:<6} {row['模型']:<10} {row['测试R2']:<10.4f} ${row['测试RMSE']:<14,.2f} ${row['测试MAE']:<14,.2f}")
print("-" * 70)

best_result = sorted_results.iloc[0]
print(f"\n【最佳模型分析】")
print(f"模型名称: {best_result['模型']}")
print(f"测试集R2: {best_result['测试R2']:.4f}（可解释{best_result['测试R2']*100:.1f}%的价格变异）")
print(f"测试集MAE: ${best_result['测试MAE']:,.2f}")
print(f"交叉验证R2均值: {best_result['CV R2均值']:.4f}")

print("\n【模型对比分析】")
print(f"• 最佳模型与最差模型R2差距: {(best_result['测试R2'] - sorted_results.iloc[-1]['测试R2'])*100:.1f}个百分点")
print(f"• 集成树模型比正则化回归R2提升: {(sorted_results[sorted_results['模型']=='梯度提升']['测试R2'].values[0] - sorted_results[sorted_results['模型']=='Lasso回归']['测试R2'].values[0])*100:.1f}个百分点")

print("\n" + "="*70)
print("阶段7：特征重要性分析")
print("="*70)

print("\n【梯度提升特征重要性】")
gb_importance_full = pd.DataFrame({
    '特征': X.columns,
    '重要性': trained_models['梯度提升'].feature_importances_ * 100
}).sort_values('重要性', ascending=False)
print(gb_importance_full.round(2).to_string(index=False))

print("\n" + "="*70)
print("✅ 多模型对比与评估优化版运行成功！")
print("="*70)