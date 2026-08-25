import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# ==========================
# Chart font and display settings
# ==========================
plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]  # Chinese font support
plt.rcParams["axes.unicode_minus"] = False
sns.set(style="whitegrid", font="Microsoft YaHei", font_scale=1.1)

# ==========================
# Data reading
# ==========================
file_path = r"C:/Users/DELL/Desktop/杭州中心城区/杭州中心城区数据提取.csv"

for enc in ["utf-8", "gbk", "latin1"]:
    try:
        df = pd.read_csv(file_path, encoding=enc)
        break
    except Exception:
        continue

df.dropna(inplace=True)
print(f"Successfully read data: {len(df)} records.")

# ==========================
# Variable definition
# ==========================
feature_cols = [
    "植被年龄",
    "三年平均PM2.5",
    "平均绿度",
    "人口密度",
    "地形高程",
    "夜光指数",
    "绿度聚集度",
    "离道路距离",
    "绿度季节变化",
    "绿度范围"
]
target_col = "平均动脉压偏离度"
age_col = "年龄"

# ==========================
# 1. Relationship between age and MAP deviation
# ==========================
plt.figure(figsize=(8, 6))
sns.scatterplot(x=age_col, y=target_col, data=df, alpha=0.6)
plt.title("Relationship between age and MAP deviation", fontsize=15)
plt.xlabel("Age")
plt.ylabel("MAP deviation value")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("age_vs_map_deviation_relationship.png", dpi=300)
plt.show()

# ==========================
# 2. Polynomial-regression age correction
# ==========================
X = df[[age_col]].values
y = df[target_col].values

best_degree, best_r2 = 1, -np.inf
for degree in range(1, 6):
    poly = PolynomialFeatures(degree)
    X_poly = poly.fit_transform(X)
    model = LinearRegression()
    score = np.mean(cross_val_score(model, X_poly, y, cv=5, scoring="r2"))
    if score > best_r2:
        best_r2, best_degree = score, degree
    print(f"Polynomial degree {degree}: average R^2 = {score:.4f}")

print(f"\nBest polynomial degree: {best_degree} (R^2 = {best_r2:.4f})")

poly = PolynomialFeatures(best_degree)
X_poly = poly.fit_transform(X)
poly_model = LinearRegression().fit(X_poly, y)
y_pred = poly_model.predict(X_poly)
df["map_deviation_corrected"] = y - y_pred

# ==========================
# 3. Correction-effect comparison chart
# ==========================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

sns.regplot(x=age_col, y=target_col, data=df, order=best_degree,
            line_kws={"color": "red"}, ax=axes[0])
axes[0].set_title("Original relationship", fontsize=14)
axes[0].set_xlabel("Age")
axes[0].set_ylabel("MAP deviation")

sns.scatterplot(x=age_col, y="map_deviation_corrected", data=df, alpha=0.6, ax=axes[1])
sns.regplot(x=age_col, y="map_deviation_corrected", data=df, order=1,
            scatter=False, line_kws={"color": "red", "linestyle": "--"}, ax=axes[1])
axes[1].set_title("Corrected relationship", fontsize=14)
axes[1].set_xlabel("Age")
axes[1].set_ylabel("Corrected MAP deviation")

corr_after = df[[age_col, "map_deviation_corrected"]].corr().iloc[0, 1]
print(f"Correlation between corrected MAP deviation and age: {corr_after:.4f}")

# ==========================
# 4. Random-forest modelling
# ==========================
X = df[feature_cols].values
y_corrected = df["map_deviation_corrected"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y_corrected, test_size=0.2, random_state=42)

rf = RandomForestRegressor(n_estimators=200, max_depth=80, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("\nModel performance:")
print(f"MAE = {mean_absolute_error(y_corrected, y_pred_rf):.4f}")
print(f"MSE = {mean_squared_error(y_corrected, y_pred_rf):.4f}")
print(f"R^2  = {r2_score(y_corrected, y_pred_rf):.4f}")

# ==========================
# 5. Feature importance
# ==========================
importance = pd.DataFrame({
    "Feature": feature_cols,
    "Importance": rf.feature_importances_
}).sort_values("Importance", ascending=False).reset_index(drop=True)

print("\nImportance ranking of factors for MAP deviation:")
print(importance.to_string(index=True, float_format=lambda x: f"{x:.5f}"))

# ==========================
# 6. Visualise feature importance
# ==========================
plt.figure(figsize=(10, 6))
sns.barplot(data=importance, y="Feature", x="Importance", palette="viridis")
plt.title("Main factors affecting MAP deviation", fontsize=15, fontweight="bold")
plt.xlabel("Feature importance")
plt.ylabel("Feature name")
plt.tight_layout()
plt.show()
