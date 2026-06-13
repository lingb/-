#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import Lasso, LassoCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ========================================================
# 阶段1：数据读取
# ========================================================
print("="*70)
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
    """使用百分位数截断去除极端值"""
    df_clean = df.copy()
    for col in columns:
        lower_val = df_clean[col].quantile(lower)
        upper_val = df_clean[col].quantile(upper)
        before = len(df_clean)
        df_clean = df_clean[(df_clean[col] >= lower_val) & (df_clean[col] <= upper_val)]
        removed = before - len(df_clean)
        print(f"  {col}: 原始{before}条 → 处理后{len(df_clean)}条（删除{removed}条极端值）")
    return df_clean

# 选择特征（使用age代替year，去掉冗余特征）
features = ['price', 'year', 'odometer', 'manufacturer', 'model',
            'condition', 'cylinders', 'fuel', 'transmission', 'type']
df_model = df[features].copy()

# 处理极端值
df_clean = remove_outliers_percentile(df_model, ['price', 'odometer'], lower=0.01, upper=0.99)

# 确保价格为正数
df_clean = df_clean[df_clean['price'] > 0]

# 创建车龄特征（age = 2021 - year）
df_clean['age'] = 2021 - df_clean['year']

print(f"\n最终数据量: {len(df_clean)}（共删除 {len(df_model) - len(df_clean)} 条）")

# ========================================================
# 阶段3：特征工程
# ========================================================
print("\n" + "="*70)
print("阶段3：特征工程")
print("="*70)

# 编码分类变量
categorical_cols = ['manufacturer', 'model', 'condition', 'cylinders', 'fuel', 'transmission', 'type']
label_encoders = {}
df_encoded = df_clean.copy()

for col in categorical_cols:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
    label_encoders[col] = le
    print(f"  编码: {col} → {len(le.classes_)} 个类别")

# ========================================================
# 阶段4：数据划分（75%训练集，25%测试集）
# ========================================================
print("\n" + "="*70)
print("阶段4：数据划分")
print("="*70)

# 使用age，去掉year（避免冗余）
X = df_encoded.drop(['price', 'year'], axis=1)  # 移除year，保留age
y = df_encoded['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

print(f"训练集: {len(X_train)} 条 ({len(X_train)/len(X):.1%})")
print(f"测试集: {len(X_test)} 条 ({len(X_test)/len(X):.1%})")

# ========================================================
# 阶段5：特征标准化
# ========================================================
print("\n" + "="*70)
print("阶段5：特征标准化")
print("="*70)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("  标准化完成")

# ========================================================
# 阶段6：Lasso回归模型训练（带交叉验证）
# ========================================================
print("\n" + "="*70)
print("阶段6：Lasso回归模型训练")
print("="*70)

# 使用交叉验证选择最佳alpha
print("  正在进行5折交叉验证选择最佳alpha...")
lasso_cv = LassoCV(alphas=np.logspace(-4, 4, 100), cv=5, random_state=42, max_iter=10000)
lasso_cv.fit(X_train_scaled, y_train)
best_alpha = lasso_cv.alpha_

print(f"  最佳alpha: {best_alpha:.6f}")

# 使用最佳alpha训练最终模型
lasso_model = Lasso(alpha=best_alpha, max_iter=10000, random_state=42)
lasso_model.fit(X_train_scaled, y_train)

print("  模型训练完成")

# ========================================================
# 阶段7：模型评估
# ========================================================
print("\n" + "="*70)
print("阶段7：模型评估")
print("="*70)

# 预测
y_train_pred = lasso_model.predict(X_train_scaled)
y_test_pred = lasso_model.predict(X_test_scaled)

# 计算指标
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

# 交叉验证
cv_scores = cross_val_score(lasso_model, X_train_scaled, y_train, cv=5, scoring='r2')
print(f"\n【5折交叉验证】")
print(f"  R²均值: {cv_scores.mean():.4f}")
print(f"  R²标准差: {cv_scores.std():.4f}")

# 保存评估结果
results = {
    '指标': ['训练集MSE', '测试集MSE', '训练集RMSE', '测试集RMSE', '训练集MAE', '测试集MAE', '训练集R²', '测试集R²', 'CV_R²均值', 'CV_R²标准差'],
    '值': [train_mse, test_mse, train_rmse, test_rmse, train_mae, test_mae, train_r2, test_r2, cv_scores.mean(), cv_scores.std()]
}
pd.DataFrame(results).to_csv('机器学习_最终版_评估结果.csv', index=False, encoding='utf-8')
print("\n评估结果已保存到: 机器学习_最终版_评估结果.csv")

# ========================================================
# 阶段8：特征重要性分析
# ========================================================
print("\n" + "="*70)
print("阶段8：特征重要性分析")
print("="*70)

feature_importance = pd.DataFrame({
    '特征': X.columns,
    '系数': lasso_model.coef_,
    '绝对值': np.abs(lasso_model.coef_)
}).sort_values('绝对值', ascending=False)

print("\n【特征重要性排名】")
for i, row in feature_importance.iterrows():
    direction = "正向影响" if row['系数'] > 0 else "负向影响"
    print(f"  {row['特征']:15s}: {row['系数']:>10.2f} ({direction})")

# 可视化特征重要性
plt.figure(figsize=(12, 8))
colors = ['green' if x > 0 else 'red' for x in feature_importance['系数']]
sns.barplot(data=feature_importance, x='系数', y='特征', palette=colors)
plt.title('Lasso回归特征重要性（使用age替代year）')
plt.xlabel('系数值（绿色=正向影响，红色=负向影响）')
plt.ylabel('特征')
plt.tight_layout()
plt.savefig('机器学习_最终版_特征重要性.png', dpi=150)
plt.close()
print("\n特征重要性图已保存: 机器学习_最终版_特征重要性.png")

# ========================================================
# 阶段9：相关性热力图
# ========================================================
print("\n" + "="*70)
print("阶段9：相关性热力图")
print("="*70)

# 计算相关性
corr_matrix = df_encoded.drop(['year'], axis=1).corr()

plt.figure(figsize=(12, 10))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', fmt='.2f',
            annot_kws={'size': 10}, vmin=-1, vmax=1)
plt.title('变量相关性热力图')
plt.tight_layout()
plt.savefig('机器学习_最终版_相关性热力图.png', dpi=150, bbox_inches='tight')
plt.close()
print("相关性热力图已保存: 机器学习_最终版_相关性热力图.png")

# ========================================================
# 阶段10：残差分析
# ========================================================
print("\n" + "="*70)
print("阶段10：残差分析")
print("="*70)

# 计算残差
residuals = y_test - y_test_pred

# 残差分布直方图
plt.figure(figsize=(10, 6))
sns.histplot(residuals, bins=50, kde=True)
plt.title('残差分布')
plt.xlabel('残差（实际价格 - 预测价格）')
plt.ylabel('频数')
plt.axvline(x=0, color='red', linestyle='--')
plt.tight_layout()
plt.savefig('机器学习_最终版_残差分布.png', dpi=150)
plt.close()

# 实际值vs预测值
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_test_pred, alpha=0.5, s=30)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='理想拟合线')
plt.title('实际价格 vs 预测价格')
plt.xlabel('实际价格')
plt.ylabel('预测价格')
plt.legend()
plt.tight_layout()
plt.savefig('机器学习_最终版_实际vs预测.png', dpi=150)
plt.close()

# 残差vs预测值
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test_pred, y=residuals, alpha=0.5, s=30)
plt.axhline(y=0, color='red', linestyle='--')
plt.title('残差 vs 预测价格')
plt.xlabel('预测价格')
plt.ylabel('残差')
plt.tight_layout()
plt.savefig('机器学习_最终版_残差vs预测.png', dpi=150)
plt.close()

print("残差分析图已保存")

# ========================================================
# 阶段11：预测示例
# ========================================================
print("\n" + "="*70)
print("阶段11：预测示例")
print("="*70)

# 随机选择测试样本
sample_indices = np.random.choice(len(X_test), 5, replace=False)
sample_X = X_test.iloc[sample_indices]
sample_y = y_test.iloc[sample_indices]
sample_pred = lasso_model.predict(scaler.transform(sample_X))

print("\n【预测示例】")
for i, (idx, row) in enumerate(sample_X.iterrows(), 1):
    actual = sample_y.loc[idx]
    pred = sample_pred[i-1]
    error = pred - actual
    error_pct = (error / actual) * 100

    manufacturer = label_encoders['manufacturer'].inverse_transform([int(row['manufacturer'])])[0]
    model_name = label_encoders['model'].inverse_transform([int(row['model'])])[0]

    print(f"\n样本 {i}:")
    print(f"  品牌: {manufacturer}")
    print(f"  车型: {model_name}")
    print(f"  车龄: {int(row['age'])}年, 里程: {int(row['odometer']):,} 英里")
    print(f"  实际价格: ${actual:,.0f}")
    print(f"  预测价格: ${pred:,.0f}")
    print(f"  误差: ${error:,.0f} ({error_pct:+.1f}%)")

# ========================================================
# 阶段12：保存处理后的数据
# ========================================================
print("\n" + "="*70)
print("阶段12：保存数据")
print("="*70)

df_clean.to_csv('cleanData_最终处理版.csv', index=False, encoding='utf-8')
print("处理后的数据已保存到: cleanData_最终处理版.csv")

# ========================================================
# 总结
# ========================================================
print("\n" + "="*70)
print("总结")
print("="*70)
print(f"✅ 模型测试集 R²: {test_r2:.4f}")
print(f"✅ 模型测试集 RMSE: ${test_rmse:,.2f}")
print(f"✅ 最重要的特征: {feature_importance.iloc[0]['特征']} (系数={feature_importance.iloc[0]['系数']:.2f})")
print(f"✅ 最负向影响的特征: {feature_importance.iloc[-1]['特征']} (系数={feature_importance.iloc[-1]['系数']:.2f})")
print("\n" + "="*70)
print("机器学习流程（最终版）完成！")
print("="*70)