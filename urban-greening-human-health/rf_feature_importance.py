from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.font_manager import FontProperties

try:
    # On Windows, use Microsoft YaHei
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
except:
    # Fallback: use the default font, but Chinese may not display
    print("Warning: Chinese font setup failed; the chart may not display Chinese correctly")
# Specify the file path
file_path = "C:/Users/mmh27/Desktop/数据/0911_血压血糖.csv"

# Try reading the CSV with different encodings
try:
    df = pd.read_csv(file_path)
except UnicodeDecodeError:
    try:
        df = pd.read_csv(file_path, encoding="gbk")
    except:
        df = pd.read_csv(file_path, encoding="latin1")

df = df.dropna()

# Factor names
columns = [
    "小区建成时间",
    "三年平均PM2.5",
    "平均绿度",
    "人口密度",
    "地形高程",
    "夜光指数",
    "绿度聚集度",
    "离道路距离",
    "绿度季节变化",
    # "年龄",
]

# The target variable; the rest are features
target_col = "平均动脉压"
feature_cols = [col for col in columns if col != target_col]

# Extract features (X) and target (y)
y = df[target_col].to_numpy()
X = df[feature_cols].to_numpy()


# Train the model
us_model = RandomForestRegressor(random_state=42, max_depth=100, n_estimators=500)
us_model.fit(X, y)

# Get predictions and evaluation metrics
y_pred = us_model.predict(X)
print("Training Mean Absolute Error:", mean_absolute_error(y, y_pred))
print("Training Mean Squared Error:", mean_squared_error(y, y_pred))
print("Training R^2 Score:", r2_score(y, y_pred))

# Feature-importance analysis
importance_scores = us_model.feature_importances_
feature_importance = pd.DataFrame(
    {"Feature": feature_cols, "Importance": importance_scores}
).sort_values("Importance", ascending=False)

# Reset index
feature_importance = feature_importance.reset_index(drop=True)

# Set column width for alignment
pd.set_option("display.max_colwidth", 15)  # Maximum column width
pd.set_option("display.unicode.east_asian_width", True)  # Correctly compute width of Chinese characters

print(f"\nImportance ranking of factors for '{target_col}':")
print(
    feature_importance.to_string(
        index=True,
        float_format=lambda x: f"{x:.5f}",
        justify="left",  # left align
        col_space=12,  # column spacing
    )
)
# Visualisation (x-axis = factor, y-axis = importance)
plt.figure(figsize=(12, 8))
sns.barplot(x="Importance", y="Feature", data=feature_importance, palette="viridis")
plt.title(f"Feature Importance for Predicting '{target_col}'")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.tight_layout()
plt.show()
