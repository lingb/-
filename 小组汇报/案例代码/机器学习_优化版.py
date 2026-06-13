#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

print("="*70)
print("二手车价格预测系统 - 优化版（梯度提升模型）")
print("="*70)

# ========================================================
# 阶段1：数据读取
# ========================================================
print("\n" + "="*70)
print("阶段1：数据读取")
print("="*70)

df = pd.read_csv('cleanData_2005plus.csv')
print(f"原始数据量: {len(df)}")

# ========================================================
# 阶段2：极端值处理（百分位数方法 1%-99%）
# ========================================================
print("\n" + "="*70)
print("阶段2：极端值处理（百分位数方法 1%-99%）")
print("="*70)

def remove_outliers_percentile(df, columns, lower=0.01, upper=0.99):
    df_clean = df.copy()
    for col in columns:
        lower_val = df_clean[col].quantile(lower)
        upper_val = df_clean[col].quantile(upper)
        before = len(df_clean)
        df_clean = df_clean[(df_clean[col] >= lower_val) & (df_clean[col] <= upper_val)]
        removed = before - len(df_clean)
        print(f"  {col}: 原始{before}条 → 处理后{len(df_clean)}条（删除{removed}条极端值）")
    return df_clean

features = ['price', 'year', 'odometer', 'manufacturer', 'model',
            'condition', 'cylinders', 'fuel', 'transmission', 'type']
df_model = df[features].copy()

df_clean = remove_outliers_percentile(df_model, ['price', 'odometer'], lower=0.01, upper=0.99)
df_clean = df_clean[df_clean['price'] > 0]

df_clean['age'] = 2021 - df_clean['year']
print(f"\n最终数据量: {len(df_clean)}（共删除 {len(df_model) - len(df_clean)} 条）")

# ========================================================
# 阶段3：特征工程（添加交互特征）
# ========================================================
print("\n" + "="*70)
print("阶段3：特征工程（含交互特征）")
print("="*70)

categorical_cols = ['manufacturer', 'model', 'condition', 'cylinders', 'fuel', 'transmission', 'type']
label_encoders = {}
df_encoded = df_clean.copy()

for col in categorical_cols:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
    label_encoders[col] = le
    print(f"  编码: {col} → {len(le.classes_)} 个类别")

# 添加交互特征
print("\n  添加交互特征：")
df_encoded['age_odometer'] = df_encoded['age'] * df_encoded['odometer'] / 1000
print("    age × odometer（车龄里程交互项）")

df_encoded['cylinders_type'] = df_encoded['cylinders'] * df_encoded['type']
print("    cylinders × type（汽缸数车型交互项）")

# ========================================================
# 阶段4：数据划分
# ========================================================
print("\n" + "="*70)
print("阶段4：数据划分")
print("="*70)

X = df_encoded.drop(['price', 'year'], axis=1)
y = df_encoded['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

print(f"训练集: {len(X_train)} 条 ({len(X_train)/len(X):.1%})")
print(f"测试集: {len(X_test)} 条 ({len(X_test)/len(X):.1%})")

# ========================================================
# 阶段5：模型训练（梯度提升）
# ========================================================
print("\n" + "="*70)
print("阶段5：梯度提升模型训练")
print("="*70)

model = GradientBoostingRegressor(
    n_estimators=200,
    max_depth=10,
    learning_rate=0.1,
    random_state=42
)

print("  正在训练梯度提升模型...")
model.fit(X_train, y_train)
print("  模型训练完成")

# ========================================================
# 阶段6：模型评估
# ========================================================
print("\n" + "="*70)
print("阶段6：模型评估")
print("="*70)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
train_rmse = np.sqrt(train_mse)
test_rmse = np.sqrt(test_mse)
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print("\n【模型评估指标】")
print(f"  训练集 RMSE: ${train_rmse:,.2f}")
print(f"  测试集 RMSE: ${test_rmse:,.2f}")
print(f"  训练集 MAE: ${train_mae:,.2f}")
print(f"  测试集 MAE: ${test_mae:,.2f}")
print(f"  训练集 R²: {train_r2:.4f}")
print(f"  测试集 R²: {test_r2:.4f}")

# 5折交叉验证
cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
print(f"\n【5折交叉验证】")
print(f"  R²均值: {cv_scores.mean():.4f}")
print(f"  R²标准差: {cv_scores.std():.4f}")

# 保存评估结果
results_df = pd.DataFrame({
    '指标': ['训练RMSE', '测试RMSE', '训练MAE', '测试MAE', '训练R²', '测试R²', 'CV R²均值', 'CV R²标准差'],
    '数值': [train_rmse, test_rmse, train_mae, test_mae, train_r2, test_r2, cv_scores.mean(), cv_scores.std()]
})
results_df.to_csv('优化版_模型评估结果.csv', index=False, encoding='utf-8-sig')
print("\n评估结果已保存到: 优化版_模型评估结果.csv")

# ========================================================
# 阶段7：特征重要性分析
# ========================================================
print("\n" + "="*70)
print("阶段7：特征重要性分析")
print("="*70)

feature_importance = pd.DataFrame({
    '特征': X.columns,
    '重要性': model.feature_importances_
}).sort_values('重要性', ascending=False)

print("\n【特征重要性排名】")
for idx, row in feature_importance.iterrows():
    print(f"  {row['特征']:<15}: {row['重要性']*100:>6.2f}%")

# 可视化
plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance, x='重要性', y='特征', palette='coolwarm')
plt.title('梯度提升特征重要性', fontsize=14, fontweight='bold')
plt.xlabel('重要性', fontsize=12)
plt.ylabel('特征', fontsize=12)
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('优化版_特征重要性.png', dpi=150)
print("\n特征重要性图已保存: 优化版_特征重要性.png")

# ========================================================
# 阶段8：预测示例
# ========================================================
print("\n" + "="*70)
print("阶段8：预测示例")
print("="*70)

print("\n【预测示例】")
for i in range(5):
    idx = X_test.index[i]
    sample = df_clean.loc[idx]
    
    print(f"\n样本 {i+1}:")
    print(f"  品牌: {sample['manufacturer']}")
    print(f"  车型: {sample['model']}")
    print(f"  车龄: {int(sample['age'])}年, 里程: {int(sample['odometer']):,} 英里")
    print(f"  汽缸数: {sample['cylinders']}")
    print(f"  车辆类型: {sample['type']}")
    print(f"  实际价格: ${int(y_test.iloc[i]):,}")
    print(f"  预测价格: ${int(y_test_pred[i]):,}")
    error = y_test_pred[i] - y_test.iloc[i]
    error_pct = error / y_test.iloc[i] * 100
    print(f"  误差: ${int(error):,+} ({error_pct:+.1f}%)")

# ========================================================
# 阶段9：保存数据
# ========================================================
print("\n" + "="*70)
print("阶段9：保存数据")
print("="*70)

df_encoded.to_csv('cleanData_优化版.csv', index=False, encoding='utf-8-sig')
print("处理后的数据已保存到: cleanData_优化版.csv")

# ========================================================
# 阶段10：优化效果对比
# ========================================================
print("\n" + "="*70)
print("阶段10：优化效果对比")
print("="*70)

print("\n【优化前后对比】")
print("-" * 60)
print(f"{'指标':<15} {'优化前(Lasso)':<15} {'优化后(梯度提升)':<15} {'提升幅度':<10}")
print("-" * 60)
print(f"{'测试R²':<15} {'0.651':<15} {f'{test_r2:.3f}':<15} {f'+{((test_r2-0.651)/0.651*100):.1f}%':<10}")
print(f"{'测试RMSE':<15} {'$7,227':<15} {f'${test_rmse:,.0f}':<15} {f'-{((7227-test_rmse)/7227*100):.1f}%':<10}")
print(f"{'测试MAE':<15} {'$5,375':<15} {f'${test_mae:,.0f}':<15} {f'-{((5375-test_mae)/5375*100):.1f}%':<10}")
print("-" * 60)

print("\n" + "="*70)
print("✅ 优化版机器学习流程完成！")
print("="*70)
