import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from statsmodels.stats.outliers_influence import variance_inflation_factor
import os

# ==================== Basic setup ====================
try:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["xtick.labelfontfamily"] = "Times New Roman"
    plt.rcParams["ytick.labelfontfamily"] = "Times New Roman"
except:
    print("Warning: Chinese font setup failed")

# ==================== Original factor list (verify against your actual data column names) ====================
ALL_FACTORS = [
    "植被年龄",
    "三年平均PM2.5",
    "平均绿度",
    "人口密度",
    "地形高程",
    "夜光指数",
    "离道路距离",
    "绿度季节变化",
    "绿度聚集度",
    "绿度范围"
]

# ==================== Function definitions ====================
def calculate_vif(X, threshold=10.0, keep_factors=None):
    """Compute VIF and automatically remove highly collinear features."""
    X_vif = X.copy()
    keep_factors = keep_factors if keep_factors is not None else []

    while True:
        if "const" in X_vif.columns:
            X_vif = X_vif.drop("const", axis=1)

        factors_to_evaluate = [f for f in X_vif.columns if f not in keep_factors]
        if not factors_to_evaluate:
            break

        vif_indices = [X_vif.columns.get_loc(f) for f in factors_to_evaluate]
        vif = [variance_inflation_factor(X_vif.values, i) for i in vif_indices]
        max_vif = np.max(vif) if vif else 0

        if max_vif > threshold:
            max_idx = np.argmax(vif)
            feature_to_drop = factors_to_evaluate[max_idx]
            print(f"Removing high-collinearity feature: {feature_to_drop} (VIF = {max_vif:.2f})")
            X_vif = X_vif.drop(feature_to_drop, axis=1)
        else:
            break
    return X_vif


def preprocess_data(df):
    """Encode non-numeric factors as numeric."""
    df_processed = df.copy()
    for col in df_processed.select_dtypes(include=["object"]).columns:
        if col in ALL_FACTORS:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col].astype(str))
            print(f"Encoding non-numeric factor: '{col}'")
    return df_processed


def get_shap_ranking(shap_values, X_test):
    """Compute mean absolute SHAP value and sort in descending order."""
    shap_matrix = shap_values[0] if isinstance(shap_values, list) else shap_values

    present_factors = [f for f in ALL_FACTORS if f in X_test.columns]
    if len(present_factors) != len(ALL_FACTORS):
        missing = set(ALL_FACTORS) - set(present_factors)
        raise ValueError(f"Missing original factors in data: {missing}")

    factor_idx = [X_test.columns.get_loc(f) for f in present_factors]
    shap_mean_abs = np.mean(np.abs(shap_matrix[:, factor_idx]), axis=0)

    ranking_df = (
        pd.DataFrame({"original_factor": present_factors, "mean(|SHAP|)": shap_mean_abs})
        .sort_values(by="mean(|SHAP|)", ascending=False)
        .reset_index(drop=True)
    )
    ranking_df["importance_rank"] = range(1, len(ranking_df) + 1)
    return ranking_df


def plot_shap_ranking(ranking_df, save_path=None):
    """Plot the SHAP factor-importance ranking bar chart."""
    ranking_sorted = ranking_df.sort_values(by="mean(|SHAP|)", ascending=False).reset_index(drop=True)

    plt.figure(figsize=(12, 10))
    ax = plt.gca()

    y_pos = np.arange(len(ranking_sorted))
    bars = ax.barh(
        y_pos,
        ranking_sorted["mean(|SHAP|)"],
        color="#2E86AB",
        alpha=0.8,
        edgecolor="#1A5F7A",
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(ranking_sorted["original_factor"], fontsize=12)
    ax.invert_yaxis()

    ax.set_xlabel("mean(|SHAP|) value", fontsize=14, fontweight="bold")
    ax.set_title(
        f"SHAP importance ranking of {len(ranking_sorted)} original factors (top rank at the top)",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )

    for bar, value, rank in zip(bars, ranking_sorted["mean(|SHAP|)"], ranking_sorted["importance_rank"]):
        ax.text(
            bar.get_width() + 0.0005,
            bar.get_y() + bar.get_height() / 2,
            f"Rank {rank} | {value:.4f}",
            va="center",
            ha="left",
            fontsize=10,
            fontfamily="Times New Roman",
        )

    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Chart saved to: {save_path}")
    plt.show()


def main(data_path, target_var="血糖偏离度  ", save_plot=None):
    try:
        print("1. Reading data...")
        data = pd.read_csv(data_path)
        data_clean = data.replace([np.inf, -np.inf], np.nan).dropna()
        print(f"Data after cleaning: {len(data_clean)} rows")

        if target_var not in data_clean.columns:
            raise ValueError(f"Target variable not found: {target_var}")
        missing_factors = [f for f in ALL_FACTORS if f not in data_clean.columns]
        if missing_factors:
            raise ValueError(f"Missing original factors in data: {missing_factors}")

        print("\n2. Preprocessing data...")
        data_core = data_clean[ALL_FACTORS + [target_var]].copy()
        data_processed = preprocess_data(data_core)

        print("\n3. Handling collinearity and splitting dataset...")
        X = data_processed.drop(columns=[target_var])
        y = data_processed[target_var]
        X_vif = calculate_vif(X, keep_factors=ALL_FACTORS)
        X_train, X_test, y_train, y_test = train_test_split(
            X_vif, y, test_size=0.2, random_state=42
        )

        print("\n4. Training random forest and computing SHAP values...")
        rf = RandomForestRegressor(
            n_estimators=200,
            max_depth=80,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        )
        rf.fit(X_train, y_train)
        explainer = shap.TreeExplainer(rf)
        shap_vals = explainer.shap_values(X_test)

        print("\n5. Generating SHAP importance ranking...")
        shap_ranking = get_shap_ranking(shap_vals, X_test)
        print("\n[SHAP importance ranking (high to low)]")
        print(shap_ranking.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

        if save_plot:
            os.makedirs(os.path.dirname(save_plot), exist_ok=True)
        plot_shap_ranking(shap_ranking, save_path=save_plot)

        # Auto-export the CSV ranking table
        csv_path = os.path.splitext(save_plot)[0] + "_shap_ranking_table.csv"
        shap_ranking.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"Ranking table saved to: {csv_path}")

    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()


# ==================== Main entry point ====================
if __name__ == "__main__":
    DATA_PATH = "C:/Users/DELL/Desktop/杭州中心城区/杭州中心城区数据提取.csv"
    SAVE_PLOT_PATH = "C:/Users/DELL/Desktop/数据/SHAP图/1003/SHAP排名图.png"

    main(
        data_path=DATA_PATH,
        target_var="血糖偏离度  ",
        save_plot=SAVE_PLOT_PATH
    )
