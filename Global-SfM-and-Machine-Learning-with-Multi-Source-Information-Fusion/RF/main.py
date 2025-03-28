import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing

# 加载示例数据集（替换为您的数据）
data = fetch_california_housing()
X = data.data
y = data.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 定义参数网格
param_grid = {
    'max_features': ['sqrt', 'log2', 0.5, 0.7],  # 对应mtry
    'n_estimators': [50, 100, 200]                # 对应ntree
}

# 初始化模型
rf = RandomForestRegressor(random_state=42)

# 设置10折交叉验证
kfold = KFold(n_splits=10, shuffle=True, random_state=42)

# 网格搜索
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=kfold,
    scoring='neg_mean_squared_error',  # 可根据需求改为'r2'
    verbose=1,
    n_jobs=-1  # 使用所有CPU核心
)

# 执行网格搜索
grid_search.fit(X_train, y_train)

# 输出最佳参数
print("最佳参数组合:", grid_search.best_params_)
print("最佳模型分数:", grid_search.best_score_)

# 用最佳模型测试
best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test)

# 评估测试集表现
print("\n测试集性能:")
print(f"R²分数: {r2_score(y_test, y_pred):.4f}")
print(f"均方误差: {mean_squared_error(y_test, y_pred):.4f}")

# 可选：输出特征重要性
feature_importances = best_rf.feature_importances_
print("\n特征重要性:")
for name, importance in zip(data.feature_names, feature_importances):
    print(f"{name}: {importance:.4f}")