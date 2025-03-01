import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import ssl
import pickle
# 设置图形风格
sns.set(style="whitegrid")
ssl.wrap_socket(ssl_version=ssl.PROTOCOL_TLSv1)
# 1. 加载数据
file_path = 'data.pkl'
# 使用pickle打开并读取数据
with open(file_path, 'rb') as file:
    data = pickle.load(file)
try:
    data = file_path
    print("数据加载成功！")
except Exception as e:
    print(f"加载数据失败：{e}")
    exit()

# 2. 数据检查与预览
print("数据基本信息：")
print(data.info())
print("数据预览：")
print(data.head())

# 检查缺失值
missing_values = data.isnull().sum()
print("每列的缺失值数量：")
print(missing_values[missing_values > 0])

# 检查数值列的基本统计
numeric_columns = data.select_dtypes(include=[np.number]).columns
print("数值列的基本统计：")
print(data[numeric_columns].describe())

# 检查分类型列的值分布
categorical_columns = data.select_dtypes(include=['object']).columns
for col in categorical_columns:
    print(f"{col}列的值分布：")
    print(data[col].value_counts())

# 3. 数据清洗
print("数据清洗开始...")

# 填充缺失值
for col in numeric_columns:
    data[col].fillna(data[col].mean(), inplace=True)
for col in categorical_columns:
    data[col].fillna(data[col].mode()[0], inplace=True)

# 删除重复行
data.drop_duplicates(inplace=True)

# 处理异常值（以 Z-Score 大于 3 为例）
z_scores = np.abs((data[numeric_columns] - data[numeric_columns].mean()) / data[numeric_columns].std())
data = data[(z_scores < 3).all(axis=1)]

print("数据清洗完成！")

# 4. 数据分析与可视化

# 4.1 数值列的分布
print("绘制数值列的分布图...")
for col in numeric_columns:
    plt.figure(figsize=(8, 5))
    sns.histplot(data[col], kde=True, bins=30, color='blue')
    plt.title(f'{col} 分布图')
    plt.xlabel(col)
    plt.ylabel("频数")
    plt.savefig(f"{col}_histogram.png")  # 保存图像
    plt.show()

# 4.2 分类型列的条形图
print("绘制分类型列的条形图...")
for col in categorical_columns:
    plt.figure(figsize=(8, 5))
    sns.countplot(data=data, x=col, order=data[col].value_counts().index, palette="Set2")
    plt.title(f'{col} 值分布条形图')
    plt.xlabel(col)
    plt.ylabel("频数")
    plt.savefig(f"{col}_barplot.png")  # 保存图像
    plt.show()

# 4.3 时间序列分析（假设有时间列）
if 'date' in data.columns:
    print("开始时间序列分析...")
    data['date'] = pd.to_datetime(data['date'])
    time_data = data.groupby(data['date'].dt.to_period("M")).size()
    plt.figure(figsize=(10, 6))
    time_data.plot(kind='line', marker='o')
    plt.title("每月数据量趋势")
    plt.xlabel("日期")
    plt.ylabel("频数")
    plt.grid(True)
    plt.savefig("time_series_analysis.png")  # 保存图像
    plt.show()

# 4.4 数据分组聚合
print("数据分组与聚合统计...")
if categorical_columns:
    group_column = categorical_columns[0]
    grouped = data.groupby(group_column)[numeric_columns].mean()
    print(f"按{group_column}分组的均值统计：")
    print(grouped)

    # 绘制分组均值图
    grouped.plot(kind='bar', figsize=(12, 6), cmap='viridis')
    plt.title(f"按 {group_column} 分组的均值")
    plt.ylabel("均值")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.savefig(f"grouped_means_{group_column}.png")  # 保存图像
    plt.show()

# 5. 数据保存
output_file = "cleaned_data.csv"
try:
    data.to_csv(output_file, index=False)
    print(f"清洗后的数据已保存到 {output_file}")
except Exception as e:
    print(f"保存失败：{e}")

# 6. 数据相关性分析
print("绘制数值列的散点图矩阵和配对关系图...")
if len(numeric_columns) > 1:
    # 散点图矩阵
    sns.pairplot(data[numeric_columns])
    plt.suptitle("数值列的散点图矩阵", y=1.02)
    plt.savefig("scatter_matrix.png")
    plt.show()

    # 相关性热图
    corr_matrix = data[numeric_columns].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.title("相关性热图")
    plt.savefig("correlation_heatmap.png")
    plt.show()

# 8. 数据分布的统计图表
print("检查长尾分布数据...")
for col in numeric_columns:
    if data[col].skew() > 1:  # 检测偏度大于1的列
        print(f"{col} 可能具有长尾分布 (偏度={data[col].skew():.2f})")
        plt.figure(figsize=(8, 5))
        sns.histplot(data[col], kde=True, bins=50, color='purple', log_scale=True)
        plt.title(f"{col} 长尾分布 (对数尺度)")
        plt.xlabel(col)
        plt.ylabel("频数 (对数)")
        plt.savefig(f"{col}_long_tail_distribution.png")
        plt.show()

# 9. 数据降维 (PCA)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

print("执行主成分分析 (PCA)...")
if len(numeric_columns) > 1:
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data[numeric_columns])

    pca = PCA(n_components=2)  # 降维到2维
    pca_result = pca.fit_transform(scaled_data)
    data['PCA1'] = pca_result[:, 0]
    data['PCA2'] = pca_result[:, 1]

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='PCA1', y='PCA2', hue=categorical_columns[0] if categorical_columns else None, data=data, palette="Set1")
    plt.title("主成分分析 (PCA) 结果")
    plt.xlabel("主成分1")
    plt.ylabel("主成分2")
    plt.savefig("pca_scatterplot.png")
    plt.show()


# 12. 异常检测
from sklearn.ensemble import IsolationForest

print("执行异常检测...")
if len(numeric_columns) > 1:
    isolation_forest = IsolationForest(contamination=0.05, random_state=42)
    data['Anomaly'] = isolation_forest.fit_predict(data[numeric_columns])
    anomaly_data = data[data['Anomaly'] == -1]

    print(f"检测到的异常数据条数：{len(anomaly_data)}")
    print(anomaly_data)

    # 可视化异常值 (以 PCA 降维后展示)
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaler.fit_transform(data[numeric_columns]))
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=pca_result[:, 0], y=pca_result[:, 1], hue=data['Anomaly'], palette={1: 'blue', -1: 'red'})
    plt.title("异常检测结果 (Isolation Forest)")
    plt.xlabel("主成分 1")
    plt.ylabel("主成分 2")
    plt.savefig("anomaly_detection.png")
    plt.show()

# 13. 时间序列预测 (ARIMA 模型)
from statsmodels.tsa.arima.model import ARIMA

if 'date' in data.columns and len(numeric_columns) > 0:
    print("执行时间序列预测...")
    time_col = numeric_columns[0]  # 假设数值列是时间序列
    time_series = data.set_index('date')[time_col]

    # 拆分训练集和测试集
    train = time_series[:-12]
    test = time_series[-12:]

    # 拟合 ARIMA 模型
    model = ARIMA(train, order=(5, 1, 0))
    model_fit = model.fit()

    # 预测
    forecast = model_fit.forecast(steps=12)

    # 可视化预测结果
    plt.figure(figsize=(10, 6))
    plt.plot(train, label='训练集')
    plt.plot(test, label='测试集', color='orange')
    plt.plot(forecast, label='预测值', color='green')
    plt.title(f"{time_col} 时间序列预测")
    plt.xlabel("时间")
    plt.ylabel(time_col)
    plt.legend()
    plt.savefig("time_series_forecast.png")
    plt.show()

# 14. 数据分类模型 (以决策树为例)
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

if len(categorical_columns) > 0 and len(numeric_columns) > 1:
    print("训练分类模型...")
    target_col = categorical_columns[0]
    X = data[numeric_columns]
    y = data[target_col]

    # 将类别变量转为数字编码
    y_encoded = pd.factorize(y)[0]

    # 数据集拆分
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # 训练决策树分类器
    classifier = DecisionTreeClassifier(random_state=42)
    classifier.fit(X_train, y_train)

    # 模型评估
    y_pred = classifier.predict(X_test)
    print("分类模型评估：")
    print(classification_report(y_test, y_pred))
    print("混淆矩阵：")
    print(confusion_matrix(y_test, y_pred))

    # 特征重要性可视化
    feature_importances = classifier.feature_importances_
    plt.figure(figsize=(8, 5))
    sns.barplot(x=X.columns, y=feature_importances, palette="coolwarm")
    plt.title("特征重要性")
    plt.xlabel("特征")
    plt.ylabel("重要性分数")
    plt.xticks(rotation=45)
    plt.savefig("feature_importances.png")
    plt.show()


# 14. 数据分类模型 (以决策树为例)
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

if len(categorical_columns) > 0 and len(numeric_columns) > 1:
    print("训练分类模型...")
    target_col = categorical_columns[0]
    X = data[numeric_columns]
    y = data[target_col]

    # 将类别变量转为数字编码
    y_encoded = pd.factorize(y)[0]

    # 数据集拆分
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # 训练决策树分类器
    classifier = DecisionTreeClassifier(random_state=42)
    classifier.fit(X_train, y_train)

    # 模型评估
    y_pred = classifier.predict(X_test)
    print("分类模型评估：")
    print(classification_report(y_test, y_pred))
    print("混淆矩阵：")
    print(confusion_matrix(y_test, y_pred))

    # 特征重要性可视化
    feature_importances = classifier.feature_importances_
    plt.figure(figsize=(8, 5))
    sns.barplot(x=X.columns, y=feature_importances, palette="coolwarm")
    plt.title("特征重要性")
    plt.xlabel("特征")
    plt.ylabel("重要性分数")
    plt.xticks(rotation=45)
    plt.savefig("feature_importances.png")
    plt.show()
