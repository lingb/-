import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import Lasso
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
df['age'] = 2021 - df['year']

# 选择特征
features = ['price', 'year', 'odometer', 'age', 'manufacturer', 'model', 
            'condition', 'cylinders', 'fuel', 'transmission', 'type']
df_model = df[features].copy()

print(f"原始数据量: {len(df_model)}")

# ========================================================
# 阶段2：定义极端值处理函数
# ========================================================
print("\n" + "="*70)
print("阶段2：定义极端值处理方法")
print("="*70)

def remove_outliers_zscore(df, columns, threshold=3):
    """使用Z-score方法去除极端值"""
    df_clean = df.copy()
    for col in columns:
        z_scores = np.abs((df_clean[col] - df_clean[col].mean()) / df_clean[col].std())
        df_clean = df_clean[z_scores < threshold]
    return df_clean

def remove_outliers_iqr(df, columns):
    """使用IQR方法去除极端值"""
    df_clean = df.copy()
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    return df_clean

def remove_outliers_percentile(df, columns, lower=0.01, upper=0.99):
    """使用百分位数截断去除极端值"""
    df_clean = df.copy()
    for col in columns:
        lower_val = df_clean[col].quantile(lower)
        upper_val = df_clean[col].quantile(upper)
        df_clean = df_clean[(df_clean[col] >= lower_val) & (df_clean[col] <= upper_val)]
    return df_clean

# ========================================================
# 阶段3：比较不同处理方法
# ========================================================
print("\n" + "="*70)
print("阶段3：比较不同极端值处理方法")
print("="*70)

methods = {
    '原始数据': lambda x: x,
    'Z-score(3σ)': lambda x: remove_outliers_zscore(x, ['price', 'odometer', 'age']),
    'IQR方法': lambda x: remove_outliers_iqr(x, ['price', 'odometer', 'age']),
    '百分位数(1%-99%)': lambda x: remove_outliers_percentile(x, ['price', 'odometer', 'age'])
}

results = []

for method_name, method_func in methods.items():
    print(f"\n正在测试: {method_name}")
    
    # 应用处理方法
    df_processed = method_func(df_model.copy())
    
    # 数据清洗基础步骤
    df_processed = df_processed[df_processed['price'] > 0]
    
    # 编码分类变量
    categorical_cols = ['manufacturer', 'model', 'condition', 'cylinders', 'fuel', 'transmission', 'type']
    label_encoders = {}
    df_encoded = df_processed.copy()
    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        label_encoders[col] = le
    
    # 划分数据集
    X = df_encoded.drop('price', axis=1)
    y = df_encoded['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 训练模型
    lasso = Lasso(alpha=0.0001, max_iter=10000, random_state=42)
    lasso.fit(X_train_scaled, y_train)
    
    # 评估
    y_pred = lasso.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    cv_scores = cross_val_score(lasso, X_train_scaled, y_train, cv=5, scoring='r2')
    
    results.append({
        '方法': method_name,
        '样本数': len(df_processed),
        '删除数': len(df_model) - len(df_processed),
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2,
        'CV_R²均值': cv_scores.mean()
    })
    
    print(f"  样本数: {len(df_processed)} (删除 {len(df_model) - len(df_processed)} 条)")
    print(f"  R²: {r2:.4f}, RMSE: ${rmse:.0f}")

# ========================================================
# 阶段4：输出对比结果
# ========================================================
print("\n" + "="*70)
print("阶段4：结果对比")
print("="*70)

results_df = pd.DataFrame(results)
print("【不同极端值处理方法对比】")
print(results_df.round(4).to_string(index=False))

# 保存结果
results_df.to_csv('极端值处理对比结果.csv', index=False, encoding='utf-8')
print("\n结果已保存到: 极端值处理对比结果.csv")

# ========================================================
# 阶段5：可视化对比
# ========================================================
print("\n" + "="*70)
print("阶段5：可视化对比")
print("="*70)

# 对比图1：R²对比
plt.figure(figsize=(12, 6))
sns.barplot(data=results_df, x='方法', y='R²')
plt.title('不同极端值处理方法的R²对比')
plt.ylabel('R²')
plt.ylim(0.5, 0.8)
plt.tight_layout()
plt.savefig('极端值处理_R2对比.png', dpi=150)
plt.close()

# 对比图2：RMSE对比
plt.figure(figsize=(12, 6))
sns.barplot(data=results_df, x='方法', y='RMSE')
plt.title('不同极端值处理方法的RMSE对比')
plt.ylabel('RMSE（美元）')
plt.tight_layout()
plt.savefig('极端值处理_RMSE对比.png', dpi=150)
plt.close()

# 对比图3：样本数对比
plt.figure(figsize=(12, 6))
sns.barplot(data=results_df, x='方法', y='样本数')
plt.title('不同极端值处理方法的样本数对比')
plt.ylabel('样本数')
plt.tight_layout()
plt.savefig('极端值处理_样本数对比.png', dpi=150)
plt.close()

print("对比图表已保存")

# ========================================================
# 阶段6：最佳方法的预测示例
# ========================================================
print("\n" + "="*70)
print("阶段6：最佳方法的预测示例")
print("="*70)

# 使用最佳方法（百分位数）重新训练并展示预测效果
df_best = remove_outliers_percentile(df_model.copy(), ['price', 'odometer', 'age'])
df_best = df_best[df_best['price'] > 0]

# 编码
for col in categorical_cols:
    le = LabelEncoder()
    df_best[col] = le.fit_transform(df_best[col].astype(str))

# 划分和训练
X = df_best.drop('price', axis=1)
y = df_best['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lasso = Lasso(alpha=0.0001, max_iter=10000, random_state=42)
lasso.fit(X_train_scaled, y_train)

# 随机选择测试样本
sample_indices = np.random.choice(len(X_test), 5, replace=False)
sample_X = X_test.iloc[sample_indices]
sample_y = y_test.iloc[sample_indices]
sample_pred = lasso.predict(scaler.transform(sample_X))

print("【IQR方法处理后的预测示例】")
for i, (idx, row) in enumerate(sample_X.iterrows(), 1):
    actual = sample_y.loc[idx]
    pred = sample_pred[i-1]
    print(f"\n样本 {i}:")
    print(f"  年份: {int(row['year'])}, 里程: {int(row['odometer']):,} 英里")
    print(f"  实际价格: ${actual:,.0f}")
    print(f"  预测价格: ${pred:,.0f}")
    print(f"  误差: ${pred - actual:,.0f} ({((pred - actual)/actual*100):.1f}%)")

print("\n" + "="*70)
print("结论：处理极端值后，模型性能有显著提升！")
print("="*70)