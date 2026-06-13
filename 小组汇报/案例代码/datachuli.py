import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# ========== 1. 读取数据 ==========
df = pd.read_csv(r'C:/Users/admin/Desktop/Python-Used-car-price-PredictSys-main/数据挖掘作业/cleanData_2005plus.csv')

# ========== 2. 创建车龄特征 ==========
df['age'] = 2021 - df['year']

# ========== 3. 选择变量 ==========
# 直接使用原始数据，不删除任何缺失值
df_model = df[['price', 'odometer', 'age', 'manufacturer', 'model']].copy()

print(f"原始数据量: {len(df_model)} 条")
print(f"品牌数量: {df_model['manufacturer'].nunique()}")
print(f"车型数量: {df_model['model'].nunique()}")

# ========== 4. 多元回归建模（三步骤）==========

# 4.1 对品牌和车型进行独热编码（drop_first 避免完全共线性）
# astype(int) 将布尔类型转换为整数类型，避免 statsmodels 报错
brand_dummies = pd.get_dummies(df_model['manufacturer'], prefix='brand', drop_first=True).astype(int)
model_dummies = pd.get_dummies(df_model['model'], prefix='model', drop_first=True).astype(int)

# 4.2 构建三个模型的 X 矩阵
X_continuous = df_model[['odometer', 'age']]

# Step 1: 只有连续变量
X_step1 = sm.add_constant(X_continuous)

# Step 2: 连续变量 + 品牌
X_step2 = sm.add_constant(pd.concat([X_continuous, brand_dummies], axis=1))

# Step 3: 连续变量 + 品牌 + 车型
X_step3 = sm.add_constant(pd.concat([X_continuous, brand_dummies, model_dummies], axis=1))

# 4.3 处理缺失值（statsmodels 需要，但不修改原数据逻辑，只删除建模所需行）
# 注意：这里的 dropna() 只是为了让模型能运行，不代表数据清洗
mask_step1 = X_step1.notna().all(axis=1) & df_model['price'].notna()
mask_step2 = X_step2.notna().all(axis=1) & df_model['price'].notna()
mask_step3 = X_step3.notna().all(axis=1) & df_model['price'].notna()

y = df_model['price']

# 4.4 拟合三个模型
print("\n" + "="*70)
print("Step 1: 基准模型（odometer + age）")
print("="*70)
model1 = sm.OLS(y[mask_step1], X_step1[mask_step1]).fit()
print(model1.summary())

print("\n" + "="*70)
print("Step 2: 加入品牌虚拟变量")
print("="*70)
model2 = sm.OLS(y[mask_step2], X_step2[mask_step2]).fit()
print(model2.summary())

print("\n" + "="*70)
print("Step 3: 完整模型（odometer + age + 品牌 + 车型）")
print("="*70)
model3 = sm.OLS(y[mask_step3], X_step3[mask_step3]).fit()
print(model3.summary())

# 获取残差（只用于可视化的部分）
df_model.loc[mask_step3, 'residual'] = model3.resid

# ========== 5. 可视化部分 ==========

# 设置图形风格
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

# 图1：价格 vs. 里程（按品牌分面）
print("\n正在生成图1：价格 vs. 里程（按品牌分面）...")

# 获取前6个最常出现的品牌
top6_brands = df_model['manufacturer'].value_counts().head(6).index
df_top6 = df_model[df_model['manufacturer'].isin(top6_brands)].copy()

g1 = sns.lmplot(
    data=df_top6, 
    x='odometer', 
    y='price', 
    col='manufacturer', 
    col_wrap=3,
    scatter_kws={'alpha': 0.2, 's': 5},
    line_kws={'color': 'red', 'linewidth': 2},
    height=4,
    aspect=1.2,
    ci=None  # 不显示置信区间，加快绘图
)
g1.fig.suptitle('价格 vs. 里程（按品牌分面）', y=1.02, fontsize=14)
plt.tight_layout()
plt.savefig('图1_价格vs里程_按品牌分面.png', dpi=150, bbox_inches='tight')
plt.show()

# 图2：价格 vs. 车龄（分车型）
print("\n正在生成图2：价格 vs. 车龄（按车型）...")

# 从数据中找出3个最常见的车型
top_models = df_model['model'].value_counts().head(10).index.tolist()
# 选择3个有代表性的（皮卡、跑车、SUV/轿车）
selected_models = []
for m in top_models:
    if 'silverado' in m.lower() and len(selected_models) < 1:
        selected_models.append(m)
    elif 'camaro' in m.lower() and len(selected_models) < 2:
        selected_models.append(m)
    elif 'wrangler' in m.lower() and len(selected_models) < 3:
        selected_models.append(m)

if len(selected_models) < 3:
    selected_models = top_models[:3]

df_selected = df_model[df_model['model'].isin(selected_models)].copy()

plt.figure(figsize=(12, 6))
for model_name in selected_models:
    subset = df_selected[df_selected['model'] == model_name]
    # 使用 LOWESS 平滑曲线
    sns.regplot(
        data=subset, 
        x='age', 
        y='price', 
        label=model_name[:30],  # 截取前30字符
        scatter=False, 
        ci=None, 
        lowess=True,
        line_kws={'linewidth': 2}
    )

plt.xlabel('车龄（年）')
plt.ylabel('价格（美元）')
plt.title('不同车型的价格 vs. 车龄（LOWESS 平滑曲线）')
plt.legend()
plt.tight_layout()
plt.savefig('图2_价格vs车龄_按车型.png', dpi=150, bbox_inches='tight')
plt.show(block=False)
plt.pause(2)
plt.close()

# 图3：品牌价格残差箱线图
print("\n正在生成图3：品牌价格残差箱线图...")

# 只取前15个最常见的品牌
top15_brands = df_model['manufacturer'].value_counts().head(15).index
df_top15_resid = df_model[df_model['manufacturer'].isin(top15_brands) & mask_step3].copy()

plt.figure(figsize=(16, 6))
sns.boxplot(
    data=df_top15_resid, 
    x='manufacturer', 
    y='residual',
    order=top15_brands
)
plt.xlabel('品牌')
plt.ylabel('回归残差（美元）')
plt.title('各品牌价格残差分布（控制了里程、车龄、车型后）')
plt.xticks(rotation=45, ha='right')
plt.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.7)
plt.tight_layout()
plt.savefig('图3_品牌价格残差箱线图.png', dpi=150, bbox_inches='tight')
plt.show(block=False)
plt.pause(2)
plt.close()

# ========== 6. 输出模型对比 ==========
print("\n" + "="*70)
print("模型对比")
print("="*70)
print(f"Step 1 (odometer + age):          R² = {model1.rsquared:.4f},  Adj.R² = {model1.rsquared_adj:.4f}")
print(f"Step 2 (+ 品牌):                   R² = {model2.rsquared:.4f},  Adj.R² = {model2.rsquared_adj:.4f}")
print(f"Step 3 (+ 品牌 + 车型):            R² = {model3.rsquared:.4f},  Adj.R² = {model3.rsquared_adj:.4f}")
print(f"\nR² 提升 (Step1 → Step2): {model2.rsquared - model1.rsquared:.4f}")
print(f"R² 提升 (Step2 → Step3): {model3.rsquared - model2.rsquared:.4f}")

# ========== 7. 输出最重要的品牌系数 ==========
print("\n" + "="*70)
print("品牌效应（相对于参考品牌 Chevrolet）")
print("="*70)

# 提取品牌系数
brand_coefs = []
for col in model3.params.index:
    if col.startswith('brand_'):
        brand_name = col.replace('brand_', '')
        brand_coefs.append({
            'brand': brand_name,
            'coefficient': model3.params[col],
            'pvalue': model3.pvalues[col]
        })

brand_df = pd.DataFrame(brand_coefs).sort_values('coefficient', ascending=False)
print(brand_df.head(10).to_string(index=False))