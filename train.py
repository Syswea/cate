import numpy as np
import pandas as pd
from econml.dml import CausalForestDML
from econml.inference import BootstrapInference
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
mpl.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
mpl.rcParams['axes.unicode_minus'] = False

# ==================== 1. 数据加载 ====================
print("=" * 60)
print("1. 数据加载")
print("=" * 60)

train_df = pd.read_feather("data/train_data.feather")
print(f"原始数据量：{len(train_df):,}")
print(f"\n数据类型分布:")
print(train_df.dtypes.value_counts())

# 随机采样（如果数据量太大）
if len(train_df) > 200000:
    train_df = train_df.sample(n=200000, random_state=42)
    print(f"\n采样后数据量：{len(train_df):,}")

# 删除不需要的列
if "area_id" in train_df.columns:
    train_df.drop(columns=["area_id"], inplace=True)
if "server_id" in train_df.columns:
    train_df.drop(columns=["server_id"], inplace=True)

# 重置索引
train_df.reset_index(drop=True, inplace=True)

# ==================== 2. 识别变量类型 ====================
print("\n" + "=" * 60)
print("2. 识别变量类型")
print("=" * 60)

Treatment = "gem_value"
Target = "order_data.price"

# 分离特征、Treatment、Target
X_df = train_df.drop(columns=[Treatment, Target]).copy()
T = train_df[Treatment].values.astype(float)
Y = train_df[Target].values.astype(float)

# 识别数值列和分类列
numeric_cols = X_df.select_dtypes(include=[np.number]).columns.tolist()
category_cols = X_df.select_dtypes(include=['category', 'object', 'bool']).columns.tolist()

print(f"数值特征 ({len(numeric_cols)}): {numeric_cols[:5]}...")
print(f"分类特征 ({len(category_cols)}): {category_cols}")

# ==================== 3. 数据清洗 ====================
print("\n" + "=" * 60)
print("3. 数据清洗")
print("=" * 60)

# 3.1 处理数值列的缺失值和无穷值
for col in numeric_cols:
    if X_df[col].isna().any():
        X_df[col].fillna(X_df[col].median(), inplace=True)
    if np.isinf(X_df[col]).any():
        X_df[col] = np.clip(X_df[col], -1e10, 1e10)

# 3.2 处理分类列的缺失值 ⚠️ 修复：统一转换为字符串
for col in category_cols:
    # ✅ 关键：先将Categorical转换为object，再全部转为字符串
    if X_df[col].dtype.name == 'category':
        X_df[col] = X_df[col].astype('object')
    
    # ✅ 关键：将所有值转换为字符串（避免混合类型）
    X_df[col] = X_df[col].astype(str)
    
    if X_df[col].isna().any() or (X_df[col] == 'nan').any():
        X_df[col] = X_df[col].replace('nan', '__MISSING__')
        n_missing = (X_df[col] == '__MISSING__').sum()
        print(f"  {col}: 填充 {n_missing} 个缺失值")

# 3.3 处理T和Y的缺失值
valid_mask = ~(np.isnan(T) | np.isnan(Y))
if not valid_mask.all():
    print(f"\n⚠️ 删除 {(~valid_mask).sum()} 行T/Y包含NaN的数据")
    X_df = X_df[valid_mask]
    T = T[valid_mask]
    Y = Y[valid_mask]

# 重置索引
X_df.reset_index(drop=True, inplace=True)

print(f"\n清洗后样本量：{len(Y):,}")
print(f"T: 均值={T.mean():.4f}, 标准差={T.std():.4f}, 范围=[{T.min():.4f}, {T.max():.4f}]")
print(f"Y: 均值={Y.mean():.4f}, 标准差={Y.std():.4f}, 范围=[{Y.min():.4f}, {Y.max():.4f}]")

# ==================== 4. 特征预处理（关键！） ====================
print("\n" + "=" * 60)
print("4. 特征预处理（保留分类变量信息）")
print("=" * 60)

# ✅ 添加：验证分类列类型统一
print("\n验证分类列数据类型...")
for col in category_cols:
    unique_types = set(type(x).__name__ for x in X_df[col].unique())
    print(f"  {col}: {unique_types}")
    if len(unique_types) > 1:
        print(f"    ⚠️ 发现混合类型，强制转换为字符串")
        X_df[col] = X_df[col].astype(str)

# 4.1 使用ColumnTransformer分别处理数值和分类特征
preprocessors = []

# 数值特征：标准化
if len(numeric_cols) > 0:
    numeric_transformer = StandardScaler()
    preprocessors.append(('num', numeric_transformer, numeric_cols))
    print(f"\n✓ 数值特征标准化：{len(numeric_cols)} 列")

# 分类特征：One-Hot编码（保留分类语义）
if len(category_cols) > 0:
    categorical_transformer = OneHotEncoder(
        handle_unknown='ignore',
        sparse_output=False,
        drop='first'  # 避免多重共线性
    )
    preprocessors.append(('cat', categorical_transformer, category_cols))
    print(f"✓ 分类特征One-Hot编码：{len(category_cols)} 列")

# 创建预处理器
preprocessor = ColumnTransformer(transformers=preprocessors)

# 拟合并转换
X_processed = preprocessor.fit_transform(X_df)

# 获取特征名称
feature_names = []
if len(numeric_cols) > 0:
    feature_names.extend(numeric_cols)
if len(category_cols) > 0:
    cat_encoder = preprocessor.named_transformers_['cat']
    cat_feature_names = cat_encoder.get_feature_names_out(category_cols)
    feature_names.extend(cat_feature_names)

print(f"\n原始特征数：{len(numeric_cols) + len(category_cols)}")
print(f"编码后特征数：{X_processed.shape[1]}")
print(f"特征名称示例：{feature_names[:10]}...")

# 4.2 删除低方差特征
variance_selector = VarianceThreshold(threshold=0.001)
X_filtered = variance_selector.fit_transform(X_processed)
kept_mask = variance_selector.get_support()
feature_names_filtered = [feature_names[i] for i in range(len(feature_names)) if kept_mask[i]]

print(f"低方差过滤后特征数：{X_filtered.shape[1]}")

# ==================== 5. 最终NaN检查 ====================
print("\n" + "=" * 60)
print("5. 最终NaN检查")
print("=" * 60)

nan_x = np.isnan(X_filtered).sum()
nan_y = np.isnan(Y).sum()
nan_t = np.isnan(T).sum()

print(f"X中NaN数量：{nan_x}")
print(f"Y中NaN数量：{nan_y}")
print(f"T中NaN数量：{nan_t}")

# 删除包含NaN的行
valid_mask = ~(np.isnan(X_filtered).any(axis=1) | np.isnan(Y) | np.isnan(T))
if not valid_mask.all():
    n_removed = (~valid_mask).sum()
    print(f"\n⚠️ 删除 {n_removed} 行包含NaN的数据")
    X_filtered = X_filtered[valid_mask]
    Y = Y[valid_mask]
    T = T[valid_mask]

print(f"\n最终样本量：{len(Y):,}")

# ==================== 6. 划分训练/测试集 ====================
print("\n" + "=" * 60)
print("6. 划分训练/测试集")
print("=" * 60)

X_train, X_test, Y_train, Y_test, T_train, T_test = train_test_split(
    X_filtered, Y, T, test_size=0.2, random_state=42
)

print(f"训练集：{len(X_train):,}, 测试集：{len(X_test):,}")

# ==================== 7. 训练模型 ====================
print("\n" + "=" * 60)
print("7. 训练因果森林模型")
print("=" * 60)

cf_dml = CausalForestDML(
    model_y=RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=10,
        min_samples_split=20,
        random_state=42,
        n_jobs=-1
    ),
    model_t=RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=10,
        min_samples_split=20,
        random_state=42,
        n_jobs=-1
    ),
    discrete_treatment=False,
    n_estimators=500,
    max_depth=None,
    min_samples_leaf=20,
    min_samples_split=40,
    max_samples=0.5,
    honest=True,
    cv=3,
    random_state=42,
    n_jobs=-1
)

print("正在训练...")
cf_dml.fit(Y=Y_train, T=T_train, X=X_train)
print("✅ 训练完成!")

# ==================== 8. 评估结果 ====================
print("\n" + "=" * 60)
print("8. 评估结果")
print("=" * 60)

cate_test = cf_dml.effect(X_test)

print(f"\n【1】CATE 统计:")
print(f"  均值：{cate_test.mean():.4f}")
print(f"  标准差：{cate_test.std():.4f}")
print(f"  最小值：{cate_test.min():.4f}")
print(f"  最大值：{cate_test.max():.4f}")
print(f"  唯一值数量：{len(np.unique(cate_test))}")

print(f"\n【2】ATE 置信区间:")
try:
    summary = cf_dml.summary(X=X_test, alpha=0.05)
    print(summary)
    ate_lb = summary['ate_lower']
    ate_ub = summary['ate_upper']
except Exception as e:
    print(f"  使用备用方法计算置信区间...")
    ate = cate_test.mean()
    ate_se = cate_test.std() / np.sqrt(len(cate_test))
    ate_lb = ate - 1.96 * ate_se
    ate_ub = ate + 1.96 * ate_se

print(f"  95% CI: [{ate_lb:.4f}, {ate_ub:.4f}]")
print(f"  统计显著：{'✅ 是' if (ate_lb > 0 or ate_ub < 0) else '❌ 否'}")

# 特征重要性
feature_importance = cf_dml.feature_importances_
importance_df = pd.DataFrame({
    'feature': feature_names_filtered,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print(f"\n【3】特征重要性 Top 10:")
print(importance_df.head(10))
print(f"  非零重要性特征数：{(feature_importance > 0).sum()}")

# Treatment 分位数效应
print(f"\n【4】Treatment 分位数效应:")
t_quantiles = np.percentile(T_test, [25, 50, 75])
print(f"  T 分布：Q25={t_quantiles[0]:.2f}, Q50={t_quantiles[1]:.2f}, Q75={t_quantiles[2]:.2f}")

mask_q1 = T_test <= t_quantiles[0]
mask_q2 = (T_test > t_quantiles[0]) & (T_test <= t_quantiles[1])
mask_q3 = (T_test > t_quantiles[1]) & (T_test <= t_quantiles[2])
mask_q4 = T_test > t_quantiles[2]

if mask_q1.sum() > 0:
    print(f"  Q1 (低剂量) 效应：{cate_test[mask_q1].mean():.4f} (n={mask_q1.sum()})")
if mask_q2.sum() > 0:
    print(f"  Q2 (中低剂量) 效应：{cate_test[mask_q2].mean():.4f} (n={mask_q2.sum()})")
if mask_q3.sum() > 0:
    print(f"  Q3 (中高剂量) 效应：{cate_test[mask_q3].mean():.4f} (n={mask_q3.sum()})")
if mask_q4.sum() > 0:
    print(f"  Q4 (高剂量) 效应：{cate_test[mask_q4].mean():.4f} (n={mask_q4.sum()})")

# ==================== 9. 可视化 ====================
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False
print("\n" + "=" * 60)
print("9. 生成可视化图表")
print("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(20, 10))

# 图1: CATE分布
axes[0, 0].hist(cate_test, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
axes[0, 0].axvline(x=cate_test.mean(), color='red', linestyle='--', linewidth=2, label=f'均值：{cate_test.mean():.4f}')
axes[0, 0].axvline(x=0, color='black', linestyle='-', linewidth=1, label='无效应')
axes[0, 0].set_xlabel('CATE')
axes[0, 0].set_ylabel('频数')
axes[0, 0].set_title(f'CATE 分布 (std={cate_test.std():.4f})')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 图2: 特征重要性
top_n = min(15, len(importance_df))
top_features = importance_df.head(top_n)
axes[0, 1].barh(range(len(top_features)), top_features['importance'].values, color='steelblue')
axes[0, 1].set_yticks(range(len(top_features)))
axes[0, 1].set_yticklabels(top_features['feature'].values, fontsize=8)
axes[0, 1].set_xlabel('重要性')
axes[0, 1].set_title(f'Top {top_n} 特征重要性')
axes[0, 1].invert_yaxis()
axes[0, 1].grid(True, alpha=0.3, axis='x')

# 图3: CATE箱线图
axes[0, 2].boxplot(cate_test, vert=True, patch_artist=True)
axes[0, 2].set_ylabel('CATE 值')
axes[0, 2].set_title(f'CATE 箱线图\n中位数：{np.median(cate_test):.4f}')
axes[0, 2].grid(True, alpha=0.3, axis='y')

# 图4: Treatment分布
axes[1, 0].hist(T_test, bins=50, edgecolor='black', alpha=0.7, color='lightgreen')
axes[1, 0].axvline(x=T_test.mean(), color='red', linestyle='--', linewidth=2, label=f'均值：{T_test.mean():.2f}')
axes[1, 0].set_xlabel('Treatment 值')
axes[1, 0].set_ylabel('频数')
axes[1, 0].set_title(f'Treatment 分布\n范围：[{T_test.min():.2f}, {T_test.max():.2f}]')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 图5: 剂量响应曲线
dose_levels = ['Q1\n(低)', 'Q2\n(中低)', 'Q3\n(中高)', 'Q4\n(高)']
dose_effects = []
dose_counts = []
for mask in [mask_q1, mask_q2, mask_q3, mask_q4]:
    if mask.sum() > 0:
        dose_effects.append(cate_test[mask].mean())
        dose_counts.append(mask.sum())
    else:
        dose_effects.append(np.nan)
        dose_counts.append(0)

axes[1, 1].plot(range(len(dose_effects)), dose_effects, marker='o', linewidth=2, markersize=8, color='darkorange')
axes[1, 1].set_xticks(range(len(dose_effects)))
axes[1, 1].set_xticklabels(dose_levels)
axes[1, 1].set_ylabel('CATE 效应')
axes[1, 1].set_title('剂量响应曲线')
axes[1, 1].grid(True, alpha=0.3)
for i, (eff, cnt) in enumerate(zip(dose_effects, dose_counts)):
    if not np.isnan(eff):
        axes[1, 1].annotate(f'{eff:.3f}\n(n={cnt})', (i, eff), textcoords="offset points", xytext=(0, 10), ha='center')

# 图6: 特征类型分布
if len(category_cols) > 0:
    cat_importance = []
    num_importance = []
    for feat, imp in zip(feature_names_filtered, feature_importance):
        is_cat = any(feat.startswith(col + '_') for col in category_cols)
        if is_cat:
            cat_importance.append(imp)
        else:
            num_importance.append(imp)
    
    axes[1, 2].pie(
        [sum(num_importance), sum(cat_importance)],
        labels=[f'数值特征\n{sum(num_importance):.3f}', f'分类特征\n{sum(cat_importance):.3f}'],
        autopct='%1.1f%%',
        colors=['steelblue', 'coral']
    )
    axes[1, 2].set_title('特征重要性按类型分布')
else:
    axes[1, 2].text(0.5, 0.5, '无分类特征', ha='center', va='center', fontsize=14)
    axes[1, 2].set_title('特征类型分布')

plt.tight_layout()
plt.savefig('causal_analysis_with_categorical.png', dpi=300, bbox_inches='tight')
plt.show()

# ==================== 10. 结论 ====================
print("\n" + "=" * 60)
print("10. 结论")
print("=" * 60)

if cate_test.std() > 0.001:
    print("✅ 模型已正常学习异质性!")
else:
    print("❌ 模型仍未学到异质性")

if (ate_lb > 0 or ate_ub < 0):
    print(f"✅ ATE 统计显著：{Treatment} 对 {Target} 的影响为 {cate_test.mean():.4f}")
else:
    print("⚠️ ATE 统计不显著")

if (feature_importance > 0).sum() > 0:
    print(f"✅ 有 {(feature_importance > 0).sum()} 个特征被模型使用")

print(f"\n【剂量响应分析】")
if len(dose_effects) >= 2 and not np.isnan(dose_effects[0]) and not np.isnan(dose_effects[-1]):
    if dose_effects[-1] > dose_effects[0]:
        print("✅ 高剂量 treatment 效应更强（正剂量响应）")
    elif dose_effects[-1] < dose_effects[0]:
        print("✅ 低剂量 treatment 效应更强（负剂量响应）")
    else:
        print("⚠️ 剂量响应不明显")

print(f"\n【分类变量处理】")
print(f"  分类特征数：{len(category_cols)}")
print(f"  One-Hot编码后特征数：{len(cat_feature_names) if len(category_cols) > 0 else 0}")
print(f"  分类特征总重要性：{sum(cat_importance) if len(category_cols) > 0 else 0:.4f}")

print("\n" + "=" * 60)
print("运行完成!")
print("=" * 60)