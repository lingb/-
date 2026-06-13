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
# 阶段2：极端值处理（百分位数方法）
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

# 选择特征
features = ['price', 'year', 'odometer', 'manufacturer', 'model', 
            'condition', 'cylinders', 'fuel', 'transmission', 'type']
df_model = df[features].copy()

# 处理极端值
df_clean = remove_outliers_percentile(df_model, ['price', 'odometer'], lower=0.01, upper=0.99)

# 确保价格为正数
df_clean = df_clean[df_clean['price'] > 0]

# 计算车龄
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

X = df_encoded.drop('price', axis=1)
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
metrics = {
    '数据集': ['训练集', '测试集'],
    '样本数': [len(y_train), len(y_test)],
    'MSE': [mean_squared_error(y_train, y_train_pred), mean_squared_error(y_test, y_test_pred)],
    'RMSE': [np.sqrt(mean_squared_error(y_train, y_train_pred)), np.sqrt(mean_squared_error(y_test, y_test_pred))],
    'MAE': [mean_absolute_error(y_train, y_train_pred), mean_absolute_error(y_test, y_test_pred)],
    'R²': [r2_score(y_train, y_train_pred), r2_score(y_test, y_test_pred)]
}

metrics_df = pd.DataFrame(metrics)
print("\n【模型评估指标】")
print(metrics_df.round(2).to_string(index=False))

# 交叉验证
cv_scores = cross_val_score(lasso_model, X_train_scaled, y_train, cv=5, scoring='r2')
print(f"\n【5折交叉验证】")
print(f"  R²均值: {cv_scores.mean():.4f}")
print(f"  R²标准差: {cv_scores.std():.4f}")

# 保存评估结果
metrics_df.to_csv('机器学习_极端值处理_评估结果.csv', index=False, encoding='utf-8')
print("\n评估结果已保存到: 机器学习_极端值处理_评估结果.csv")

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
print(feature_importance[['特征', '系数']].to_string(index=False))

# 可视化特征重要性
plt.figure(figsize=(12, 6))
sns.barplot(data=feature_importance.head(10), x='系数', y='特征', orient='h')
plt.title('Lasso回归特征重要性（极端值处理后）')
plt.xlabel('系数值')
plt.tight_layout()
plt.savefig('机器学习_极端值处理_特征重要性.png', dpi=150)
plt.close()
print("\n特征重要性图已保存")

# ========================================================
# 阶段9：残差分析
# ========================================================
print("\n" + "="*70)
print("阶段9：残差分析")
print("="*70)

# 计算残差
residuals = y_test - y_test_pred

# 残差分布直方图
plt.figure(figsize=(10, 6))
sns.histplot(residuals, bins=50, kde=True)
plt.title('残差分布（极端值处理后）')
plt.xlabel('残差（实际价格 - 预测价格）')
plt.ylabel('频数')
plt.tight_layout()
plt.savefig('机器学习_极端值处理_残差分布.png', dpi=150)
plt.close()

# 残差vs预测值
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test_pred, y=residuals, alpha=0.5)
plt.axhline(y=0, color='red', linestyle='--')
plt.title('残差vs预测值（极端值处理后）')
plt.xlabel('预测价格')
plt.ylabel('残差')
plt.tight_layout()
plt.savefig('机器学习_极端值处理_残差vs预测值.png', dpi=150)
plt.close()

# 实际值vs预测值
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_test_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title('实际值vs预测值（极端值处理后）')
plt.xlabel('实际价格')
plt.ylabel('预测价格')
plt.tight_layout()
plt.savefig('机器学习_极端值处理_实际vs预测.png', dpi=150)
plt.close()

print("残差分析图已保存")

# ========================================================
# 阶段10：预测示例
# ========================================================
print("\n" + "="*70)
print("阶段10：预测示例")
print("="*70)

# 随机选择测试样本
sample_indices = np.random.choice(len(X_test), 5, replace=False)
sample_X = X_test.iloc[sample_indices]
sample_y = y_test.iloc[sample_indices]
sample_pred = lasso_model.predict(scaler.transform(sample_X))

print("【预测示例】")
for i, (idx, row) in enumerate(sample_X.iterrows(), 1):
    actual = sample_y.loc[idx]
    pred = sample_pred[i-1]
    error = pred - actual
    error_pct = (error / actual) * 100
    
    # 解码分类变量
    manufacturer = label_encoders['manufacturer'].inverse_transform([int(row['manufacturer'])])[0]
    model = label_encoders['model'].inverse_transform([int(row['model'])])[0]
    
    print(f"\n样本 {i}:")
    print(f"  品牌: {manufacturer}")
    print(f"  车型: {model}")
    print(f"  年份: {int(row['year'])}, 里程: {int(row['odometer']):,} 英里")
    print(f"  实际价格: ${actual:,.0f}")
    print(f"  预测价格: ${pred:,.0f}")
    print(f"  误差: ${error:,.0f} ({error_pct:.1f}%)")

# ========================================================
# 阶段11：保存处理后的数据
# ========================================================
print("\n" + "="*70)
print("阶段11：保存处理后的数据")
print("="*70)

df_clean.to_csv('cleanData_极端值处理后.csv', index=False, encoding='utf-8')
print("处理后的数据已保存到: cleanData_极端值处理后.csv")

print("\n" + "="*70)
print("机器学习流程（极端值处理版）完成！")
print("="*70)