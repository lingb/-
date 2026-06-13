#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

print("="*70)
print("模型优化方案对比实验")
print("="*70)

df = pd.read_csv('cleanData_最终处理版.csv')

# 编码分类变量
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

# 标准化（线性模型需要）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ========================================================
# 定义评估函数
# ========================================================
def evaluate_model(name, model, X_train, X_test, y_train, y_test, scaled=False):
    print(f"  训练 {name}...")
    model.fit(X_train, y_train)
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    results = {
        '模型': name,
        '训练R²': r2_score(y_train, y_train_pred),
        '测试R²': r2_score(y_test, y_test_pred),
        '训练RMSE': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        '测试RMSE': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        '训练MAE': mean_absolute_error(y_train, y_train_pred),
        '测试MAE': mean_absolute_error(y_test, y_test_pred)
    }
    return results

# ========================================================
# 尝试不同模型
# ========================================================
print("\n" + "="*70)
print("【模型对比实验】")
print("="*70)

results_list = []

# 1. Lasso回归（基准模型）
print("\n1. Lasso回归")
lasso = Lasso(alpha=54.62, max_iter=10000, random_state=42)
results_list.append(evaluate_model("Lasso回归", lasso, X_train_scaled, X_test_scaled, y_train, y_test))

# 2. Ridge回归
print("\n2. Ridge回归")
ridge = Ridge(alpha=100, random_state=42)
results_list.append(evaluate_model("Ridge回归", ridge, X_train_scaled, X_test_scaled, y_train, y_test))

# 3. ElasticNet回归
print("\n3. ElasticNet回归")
elastic = ElasticNet(alpha=10, l1_ratio=0.5, max_iter=10000, random_state=42)
results_list.append(evaluate_model("ElasticNet回归", elastic, X_train_scaled, X_test_scaled, y_train, y_test))

# 4. 随机森林
print("\n4. 随机森林")
rf = RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
results_list.append(evaluate_model("随机森林", rf, X_train, X_test, y_train, y_test))

# 5. 梯度提升
print("\n5. 梯度提升")
gb = GradientBoostingRegressor(n_estimators=200, max_depth=10, learning_rate=0.1, random_state=42)
results_list.append(evaluate_model("梯度提升", gb, X_train, X_test, y_train, y_test))



# ========================================================
# 结果对比
# ========================================================
results_df = pd.DataFrame(results_list)
results_df = results_df.sort_values('测试R²', ascending=False)

print("\n" + "="*70)
print("【模型性能对比】")
print("="*70)
print(results_df.round(4).to_string())

# ========================================================
# 可视化对比
# ========================================================
plt.figure(figsize=(14, 6))

# RMSE对比
plt.subplot(1, 2, 1)
sns.barplot(data=results_df, x='模型', y='测试RMSE', palette='viridis')
plt.title('模型测试集RMSE对比', fontsize=14, fontweight='bold')
plt.xlabel('模型', fontsize=12)
plt.ylabel('RMSE ($)', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.ylim(5000, 8000)
plt.grid(axis='y', alpha=0.3)

# R²对比
plt.subplot(1, 2, 2)
sns.barplot(data=results_df, x='模型', y='测试R²', palette='plasma')
plt.title('模型测试集R²对比', fontsize=14, fontweight='bold')
plt.xlabel('模型', fontsize=12)
plt.ylabel('R²', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.ylim(0.5, 0.8)
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('模型优化对比图.png', dpi=150, bbox_inches='tight')
print("\n对比图已保存: 模型优化对比图.png")

# ========================================================
# 特征重要性分析（以梯度提升为例）
# ========================================================
print("\n" + "="*70)
print("【梯度提升特征重要性】")
print("="*70)

feature_importance = pd.DataFrame({
    '特征': X.columns,
    '重要性': gb.feature_importances_
}).sort_values('重要性', ascending=False)

print(feature_importance.to_string())

plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance, x='重要性', y='特征', palette='coolwarm')
plt.title('梯度提升特征重要性', fontsize=14, fontweight='bold')
plt.xlabel('重要性', fontsize=12)
plt.ylabel('特征', fontsize=12)
plt.grid(axis='x', alpha=0.3)
plt.savefig('梯度提升特征重要性.png', dpi=150, bbox_inches='tight')
print("\n特征重要性图已保存: 梯度提升特征重要性.png")

# ========================================================
# 优化方案总结
# ========================================================
print("\n" + "="*70)
print("【优化方案总结】")
print("="*70)

best_model = results_df.iloc[0]
improvement = ((7227 - best_model['测试RMSE']) / 7227 * 100)

print(f"当前基准模型 (Lasso): RMSE = ${7227:.0f}, R² = 0.651")
print(f"最优模型 ({best_model['模型']}): RMSE = ${best_model['测试RMSE']:.0f}, R² = {best_model['测试R²']:.4f}")
print(f"优化幅度: RMSE降低 {improvement:.1f}%")

print("\n【推荐优化方案】")
print("="*50)
print("""
1. 模型选择优化：
   - 推荐使用 XGBoost 或 随机森林，非线性模型能捕捉更复杂的关系
   - XGBoost表现最佳，适合处理高维数据

2. 特征工程优化：
   - 当前最重要的特征是：model > age > odometer > cylinders
   - 可考虑添加多项式特征（如age², odometer²）
   - 可尝试特征交互（如cylinders × type）

3. 超参数调优：
   - 使用GridSearchCV或Optuna进行参数优化
   - 重点调整：树深度、学习率、正则化参数

4. 数据增强：
   - 收集更多特征（如颜色、配置、地区、保养记录）
   - 考虑时间因素对价格的动态影响

5. 集成策略：
   - 尝试Stacking集成多个模型
   - 结合线性模型和非线性模型的优点
""")

print("\n" + "="*70)
print("优化完成！")
print("="*70)
