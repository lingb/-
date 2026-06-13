import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ========== 1. 数据读取与基本信息 ==========
print("="*70)
print("阶段1：数据读取与基本信息")
print("="*70)
df = pd.read_csv('cleanData_2005plus.csv')
print(f"数据集形状: {df.shape}")
print(f"样本数: {df.shape[0]} 条")
print(f"变量数: {df.shape[1]} 个")

# ========== 2. 描述性统计 ==========
print("\n" + "="*70)
print("阶段2：描述性统计")
print("="*70)

# 连续变量统计
print("\n【连续变量统计】")
continuous_vars = ['price', 'year', 'odometer']
print(df[continuous_vars].describe().round(2))

# 分类变量统计
print("\n【分类变量统计】")
categorical_vars = ['manufacturer', 'model', 'condition', 'fuel', 'transmission', 'type']
for var in categorical_vars:
    print(f"\n{var}: {df[var].nunique()} 个类别")
    print(f"前5个: {df[var].value_counts().head(5).index.tolist()}")

# ========== 3. 数据预处理 ==========
print("\n" + "="*70)
print("阶段3：数据预处理")
print("="*70)

# 创建车龄特征
df['age'] = 2021 - df['year']
print(f"车龄特征已创建，车龄范围: {df['age'].min()} - {df['age'].max()} 年")

# 选择建模变量
df_model = df[['price', 'odometer', 'age', 'manufacturer', 'model', 'condition', 'fuel', 'transmission', 'type']].copy()

# ========== 4. 相关性分析 ==========
print("\n" + "="*70)
print("阶段4：相关性分析")
print("="*70)

# 连续变量相关矩阵
corr_matrix = df_model[['price', 'odometer', 'age']].corr()
print("【连续变量相关系数矩阵】")
print(corr_matrix.round(4))

# 可视化相关矩阵
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('连续变量相关系数矩阵')
plt.tight_layout()
plt.savefig('复现_相关系数矩阵.png', dpi=150)
plt.close()
print("相关系数矩阵图已保存")

# ========== 5. 可视化分析 ==========
print("\n" + "="*70)
print("阶段5：可视化分析")
print("="*70)

# 图1：价格分布直方图
print("\n生成图1：价格分布直方图...")
plt.figure(figsize=(12, 6))
sns.histplot(df_model['price'], bins=50, kde=True)
plt.xlabel('价格（美元）')
plt.ylabel('频数')
plt.title('二手车价格分布')
plt.tight_layout()
plt.savefig('复现_价格分布.png', dpi=150)
plt.close()

# 图2：价格与里程散点图
print("生成图2：价格与里程散点图...")
plt.figure(figsize=(12, 6))
sns.scatterplot(data=df_model, x='odometer', y='price', alpha=0.3, s=20)
plt.xlabel('里程（英里）')
plt.ylabel('价格（美元）')
plt.title('价格 vs. 里程')
plt.tight_layout()
plt.savefig('复现_价格vs里程.png', dpi=150)
plt.close()

# 图3：价格与车龄散点图
print("生成图3：价格与车龄散点图...")
plt.figure(figsize=(12, 6))
sns.scatterplot(data=df_model, x='age', y='price', alpha=0.3, s=20)
plt.xlabel('车龄（年）')
plt.ylabel('价格（美元）')
plt.title('价格 vs. 车龄')
plt.tight_layout()
plt.savefig('复现_价格vs车龄.png', dpi=150)
plt.close()

# 图4：不同品牌平均价格
print("生成图4：品牌平均价格对比...")
top10_brands = df_model['manufacturer'].value_counts().head(10).index
brand_price = df_model[df_model['manufacturer'].isin(top10_brands)].groupby('manufacturer')['price'].mean().sort_values(ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(x=brand_price.index, y=brand_price.values)
plt.xlabel('品牌')
plt.ylabel('平均价格（美元）')
plt.title('各品牌平均价格对比（前10品牌）')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('复现_品牌平均价格.png', dpi=150)
plt.close()

# 图5：车辆类型分布
print("生成图5：车辆类型分布...")
type_counts = df_model['type'].value_counts()
plt.figure(figsize=(12, 6))
sns.barplot(x=type_counts.index, y=type_counts.values)
plt.xlabel('车辆类型')
plt.ylabel('数量')
plt.title('车辆类型分布')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('复现_车辆类型分布.png', dpi=150)
plt.close()

# ========== 6. 多元回归建模 ==========
print("\n" + "="*70)
print("阶段6：多元回归建模")
print("="*70)

# 创建虚拟变量
brand_dummies = pd.get_dummies(df_model['manufacturer'], prefix='brand', drop_first=True).astype(int)
type_dummies = pd.get_dummies(df_model['type'], prefix='type', drop_first=True).astype(int)

# 构建模型矩阵
X = sm.add_constant(pd.concat([df_model[['odometer', 'age']], brand_dummies, type_dummies], axis=1))
y = df_model['price']

# 拟合模型
model = sm.OLS(y, X).fit()
print("\n【回归模型摘要】")
print(model.summary())

# 保存模型结果
with open('复现_回归结果.txt', 'w', encoding='utf-8') as f:
    f.write(str(model.summary()))
print("\n回归结果已保存到: 复现_回归结果.txt")

# ========== 7. 模型诊断 ==========
print("\n" + "="*70)
print("阶段7：模型诊断")
print("="*70)

# 图6：残差分布
print("\n生成图6：残差分布图...")
plt.figure(figsize=(12, 6))
sns.histplot(model.resid, bins=50, kde=True)
plt.xlabel('残差')
plt.ylabel('频数')
plt.title('回归残差分布')
plt.tight_layout()
plt.savefig('复现_残差分布.png', dpi=150)
plt.close()

# 图7：残差 vs 拟合值
print("生成图7：残差 vs 拟合值...")
plt.figure(figsize=(12, 6))
sns.scatterplot(x=model.fittedvalues, y=model.resid, alpha=0.3)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('拟合值')
plt.ylabel('残差')
plt.title('残差 vs 拟合值')
plt.tight_layout()
plt.savefig('复现_残差vs拟合值.png', dpi=150)
plt.close()

# ========== 8. 关键发现总结 ==========
print("\n" + "="*70)
print("阶段8：关键发现总结")
print("="*70)

print(f"1. 模型整体拟合度 R² = {model.rsquared:.4f}")
print(f"2. 调整后 R² = {model.rsquared_adj:.4f}")
print(f"3. 里程系数 = {model.params['odometer']:.4f}（每增加1英里，价格下降约 {abs(model.params['odometer']):.4f} 美元）")
print(f"4. 车龄系数 = {model.params['age']:.2f}（每增加1年，价格下降约 {abs(model.params['age']):.0f} 美元）")

# 品牌系数分析
brand_coefs = []
for col in model.params.index:
    if col.startswith('brand_'):
        brand_coefs.append((col.replace('brand_', ''), model.params[col]))
brand_coefs.sort(key=lambda x: abs(x[1]), reverse=True)

print("\n5. 影响最大的品牌效应（相对于 Chevrolet）：")
for brand, coef in brand_coefs[:5]:
    direction = "溢价" if coef > 0 else "折价"
    print(f"   {brand}: {direction} {abs(coef):,.0f} 美元")

print("\n" + "="*70)
print("分析完成！所有图表和结果已保存。")
print("="*70)