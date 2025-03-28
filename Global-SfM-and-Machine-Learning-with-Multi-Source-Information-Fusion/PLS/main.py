# 导入必要的库
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GridSearchCV, train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np



scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 加载数据（示例数据，需替换为实际数据）
X = 特征矩阵，y = 目标变量
示例数据：
X = np.random.rand(100, 10)  # 100个样本，10个特征
y = np.random.rand(100)      # 连续型目标变量

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 定义参数网格
ncomp_range = list(range(1, 11))  # 搜索n_components从1到10
param_grid = {'n_components': ncomp_range}

# 创建PLS模型和交叉验证对象
pls = PLSRegression()
kfold = KFold(n_splits=10, shuffle=True, random_state=42)

# 配置网格搜索
grid_search = GridSearchCV(
    estimator=pls,
    param_grid=param_grid,
    cv=kfold,
    scoring='neg_mean_squared_error',  # 常用回归指标
    verbose=1,
    n_jobs=-1  # 并行计算
)

# 执行网格搜索（在训练集上）
grid_search.fit(X_train, y_train)

# 输出最佳参数
print("最佳n_components:", grid_search.best_params_)
print("最佳模型得分（负MSE）:", grid_search.best_score_)

# 使用最佳模型进行预测（测试集评估）
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# 计算评估指标
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("\n测试集评估结果:")
print(f"MSE: {mse:.4f}")
print(f"R²: {r2:.4f}")

# 交叉验证结果分析（可选）
cv_results = grid_search.cv_results_
print("\n交叉验证详细结果:")
for mean_score, params in zip(cv_results['mean_test_score'], cv_results['params']):
    print(f"n_components={params['n_components']}: MSE={-mean_score:.4f}")