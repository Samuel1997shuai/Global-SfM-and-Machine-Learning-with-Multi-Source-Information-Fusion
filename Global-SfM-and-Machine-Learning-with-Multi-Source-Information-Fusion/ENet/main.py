import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_diabetes  # 示例数据集，可替换为您的数据

# 加载示例数据集
data = load_diabetes()
X, y = data.data, data.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 创建包含标准化和ENET模型的Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # 特征标准化
    ('enet', ElasticNet(max_iter=10000, random_state=42))  # 增加迭代次数确保收敛
])

# 定义参数网格
param_grid = {
    'enet__alpha': np.logspace(-4, 2, 20),  # 正则化强度（10^-4到10^2之间取对数）
    'enet__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]  # L1/L2混合比例
}

# 创建网格搜索对象
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=10,  # 10折交叉验证
    scoring='neg_mean_squared_error',  # 评估指标（负均方误差）
    n_jobs=-1,  # 使用所有CPU核心
    verbose=1  # 显示进度
)

# 执行网格搜索
grid_search.fit(X_train, y_train)

# 输出最佳参数
print("最佳参数组合：", grid_search.best_params_)
print("最佳模型分数（负MSE）：", grid_search.best_score_)

# 使用最佳模型评估测试集
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# 计算评估指标
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n测试集评估：")
print(f"MSE: {mse:.4f}")
print(f"R²: {r2:.4f}")

# 获取所有参数组合结果
results = pd.DataFrame(grid_search.cv_results_)
print("\n前5个参数组合结果：")
print(results[['params', 'mean_test_score', 'std_test_score']].head())