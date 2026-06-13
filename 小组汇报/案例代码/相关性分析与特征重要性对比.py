#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ========================================================
# 1. 读取数据
# ========================================================
print("="*70)
print("步骤1：读取数据")
print("="*70)

df = pd.read_csv('cleanData_2005plus.csv')
df['age'] = 2021 - df['year']

# ========================================================
# 2. 定义极端值处理函数
# ========================================================
def remove_outliers_percentile(df, columns, lower=0.01, upper=0.99):
    df_clean = df.copy()
    for col in columns:
        lower_val = df_clean[col].quantile(lower)
        upper_val = df_clean[col].quantile(upper)
        df_clean = df_clean[(df_clean[col] >= lower_val) & (df_clean[col] <= upper_val)]
    return df_clean

# ========================================================
# 3. 处理前的数据
# ========================================================
print("\n" + "="*70)
print("步骤2：处理前的数据")
print("="*70)

features = ['price', 'year', 'odometer', 'age', 'manufacturer', 'model', 
            'condition', 'cylinders', 'fuel', 'transmission', 'type']
df_before = df[features].copy()
df_before = df_before[df_before['price'] > 0]

# 编码
categorical_cols = ['manufacturer', 'model', 'condition', 'cylinders', 'fuel', 'transmission', 'type']
label_encoders_before = {}
df_encoded_before = df_before.copy()
for col in categorical_cols:
    le = LabelEncoder()
    df_encoded_before[col] = le.fit_transform(df_encoded_before[col].astype(str))
    label_encoders_before[col] = le

# ========================================================
# 4. 处理后的数据
# ========================================================
print("\n" + "="*70)
print("步骤3：处理后的数据")
print("="*70)

df_after = remove_outliers_percentile(df[features].copy(), ['price', 'odometer'], lower=0.01, upper=0.99)
df_after = df_after[df_after['price'] > 0]
df_after['age'] = 2021 - df_after['year']

# 编码
label_encoders_after = {}
df_encoded_after = df_after.copy()
for col in categorical_cols:
    le = LabelEncoder()
    df_encoded_after[col] = le.fit_transform(df_encoded_after[col].astype(str))
    label_encoders_after[col] = le

# ========================================================
# 5. 绘制相关性热力图（处理后）
# ========================================================
print("\n" + "="*70)
print("步骤4：绘制相关性热力图")
print("="*70)

# 计算相关性
corr_matrix = df_encoded_after.corr()

# 绘制热力图
plt.figure(figsize=(14, 12))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', fmt='.2f', 
            annot_kws={'size': 10}, cbar=True)
plt.title('变量相关性热力图（极端值处理后）', fontsize=14)
plt.tight_layout()
plt.savefig('相关性热力图.png', dpi=150, bbox_inches='tight')
plt.close()
print("  相关性热力图已保存: 相关性热力图.png")

# ========================================================
# 6. 训练模型并比较特征重要性
# ========================================================
print("\n" + "="*70)
print("步骤5：训练模型并比较特征重要性")
print("="*70)

def train_and_get_importance(df_encoded):
    X = df_encoded.drop('price', axis=1)
    y = df_encoded['price']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    lasso = Lasso(alpha=0.0001, max_iter=10000, random_state=42)
    lasso.fit(X_train_scaled, y_train)
    
    importance = pd.DataFrame({
        '特征': X.columns,
        '系数': lasso.coef_,
        '绝对值': np.abs(lasso.coef_)
    }).sort_values('绝对值', ascending=False)
    
    return importance

# 获取处理前后的特征重要性
importance_before = train_and_get_importance(df_encoded_before)
importance_after = train_and_get_importance(df_encoded_after)

# ========================================================
# 7. 输出对比结果
# ========================================================
print("\n【处理前特征重要性】")
print(importance_before[['特征', '系数']].round(2).to_string(index=False))

print("\n【处理后特征重要性】")
print(importance_after[['特征', '系数']].round(2).to_string(index=False))

# ========================================================
# 8. 可视化对比
# ========================================================
print("\n" + "="*70)
print("步骤6：可视化对比")
print("="*70)

# 合并对比数据
compare_df = pd.merge(
    importance_before[['特征', '系数']].rename(columns={'系数': '处理前'}),
    importance_after[['特征', '系数']].rename(columns={'系数': '处理后'}),
    on='特征'
)

compare_df['变化'] = compare_df['处理后'] - compare_df['处理前']

# 特征重要性对比图
plt.figure(figsize=(14, 8))
sns.barplot(data=compare_df, x='特征', y='处理后', label='处理后')
sns.barplot(data=compare_df, x='特征', y='处理前', label='处理前', alpha=0.5)
plt.title('处理前后特征重要性对比')
plt.xlabel('特征')
plt.ylabel('系数值')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('特征重要性对比图.png', dpi=150)
plt.close()
print("  特征重要性对比图已保存: 特征重要性对比图.png")

# 变化条形图
plt.figure(figsize=(14, 8))
sns.barplot(data=compare_df, x='特征', y='变化')
plt.title('特征重要性变化量')
plt.xlabel('特征')
plt.ylabel('系数变化')
plt.xticks(rotation=45)
plt.axhline(y=0, color='red', linestyle='--')
plt.tight_layout()
plt.savefig('特征重要性变化图.png', dpi=150)
plt.close()
print("  特征重要性变化图已保存: 特征重要性变化图.png")

# ========================================================
# 9. 解释差异原因
# ========================================================
print("\n" + "="*70)
print("步骤7：特征重要性差异原因分析")
print("="*70)

print("【为什么处理前后特征重要性差别大？】")
print("")
print("1. 极端值的影响被移除")
print("   - 极端高价或低价的异常数据会扭曲特征与价格的关系")
print("   - 例如：某些稀有车型的超高价格会夸大品牌的重要性")
print("")
print("2. 系数尺度变化")
print("   - 极端值会增大特征的标准差，影响标准化结果")
print("   - 处理后数据分布更集中，系数更能反映真实影响")
print("")
print("3. 年份与年龄的冗余")
print("   - year和age高度相关（相关系数≈-1）")
print("   - 处理前：year系数更高")
print("   - 处理后：cylinders成为最重要特征")
print("")
print("4. 噪声减少")
print("   - 极端值相当于噪声，干扰模型学习")
print("   - 去除后模型能学到更稳定的特征权重")
print("")
print("5. 数据分布变化")
print("   - 处理前：价格范围可能是[$1, $1M+]")
print("   - 处理后：价格范围更合理，特征重要性更真实")

print("\n" + "="*70)
print("分析完成！")
print("="*70)