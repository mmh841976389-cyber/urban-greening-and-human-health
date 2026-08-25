import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
from scipy.interpolate import interp1d
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from matplotlib.lines import Line2D

try:
    # Chinese display setup (titles/axis labels in Chinese; numeric labels set to Times New Roman later)
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
except:
    print("Warning: Chinese font setup failed; the chart may not display Chinese correctly")

# Define the list of available original factors
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


def calculate_vif(X, threshold=10.0, keep_factors=None):
    """Compute VIF and remove highly collinear features (keep all original factors)."""
    X_vif = X.copy()
    keep_factors = keep_factors if keep_factors is not None else []

    while True:
        X_temp = X_vif.drop("const", axis=1) if "const" in X_vif.columns else X_vif
        factors_to_evaluate = [f for f in X_temp.columns if f not in keep_factors]

        if not factors_to_evaluate:
            break

        vif_indices = [X_temp.columns.get_loc(f) for f in factors_to_evaluate]
        vif = [variance_inflation_factor(X_temp.values, i) for i in vif_indices]
        max_vif = np.max(vif) if vif else 0

        if max_vif > threshold:
            max_idx = np.argmax(vif)
            feature_to_drop = factors_to_evaluate[max_idx]
            print(f"Removing high-VIF feature (non-original factor): {feature_to_drop} (VIF = {max_vif:.2f})")
            X_vif = X_vif.drop(feature_to_drop, axis=1)
        else:
            break
    return X_vif


def create_interaction_features(X, factors):
    """Create interaction terms based on the target factor (greenness range)."""
    X_interact = X.copy()
    target_factor = "绿度范围"
    if target_factor in factors:
        for factor in factors:
            if factor != target_factor and factor in X.columns:
                new_col = f"{target_factor}_x_{factor}"
                X_interact[new_col] = X[target_factor] * X[factor]
                print(f"Creating interaction term: {new_col}")
    return X_interact


def add_polynomial_features(X, factor, max_degree=3):
    """Add polynomial features for the target factor (greenness range)."""
    X_poly = X.copy()
    if factor in X.columns:
        for d in range(2, max_degree + 1):
            new_col = f"{factor}^{d}"
            X_poly[new_col] = X[factor] ** d
            print(f"Creating polynomial feature: {new_col}")
    return X_poly


def calculate_original_factors_shap_mean_abs(shap_values, X_test, all_factors):
    """Compute mean(|SHAP|) of original factors (ignore factors missing from the test set)."""
    shap_matrix = shap_values[0] if isinstance(shap_values, list) else shap_values

    present_factors = [f for f in all_factors if f in X_test.columns]
    missing_factors = [f for f in all_factors if f not in X_test.columns]

    if missing_factors:
        print(f"Warning: original factors missing from test set, will be ignored: {', '.join(missing_factors)}")
    if not present_factors:
        return pd.DataFrame(columns=["original_factor", "mean(|SHAP|)", "importance_rank"])

    factor_indices = [X_test.columns.get_loc(f) for f in present_factors]
    shap_mean_abs = np.mean(np.abs(shap_matrix[:, factor_indices]), axis=0)

    return pd.DataFrame({
        "original_factor": present_factors,
        "mean(|SHAP|)": shap_mean_abs,
        "importance_rank": range(1, len(present_factors) + 1)
    }).sort_values(by="mean(|SHAP|)", ascending=False).reset_index(drop=True)


def main():
    try:
        # 1. Data reading and preprocessing
        file_path = r"C:/Users/DELL/Desktop/杭州中心城区/杭州中心城区数据提取.csv"
        data = pd.read_csv(file_path, encoding="utf-8")
        data_clean = data.replace([np.inf, -np.inf], np.nan).dropna()
        print(f"Data after cleaning: {len(data_clean)} rows")

        # Target factor and target variable
        factor_of_interest = "绿度范围"
        target_variable = "平均动脉压偏离度"
        zone_name = "High"  # Zone name, adjustable as needed
        print(data.head())  # Inspect the first few rows of data
        # Check that core columns exist
        if target_variable not in data_clean.columns:
            raise ValueError(f"Target variable '{target_variable}' not found")
        missing_in_data = [f for f in ALL_FACTORS if f not in data_clean.columns]
        if missing_in_data:
            raise ValueError(f"Factors missing from raw data: {', '.join(missing_in_data)}")

        # Keep only core data (original factors + target variable)
        data_filtered = data_clean[ALL_FACTORS + [target_variable]].copy()
        print(f"Analysing with {len(ALL_FACTORS)} original factors")

        # Preprocess non-numeric original factors
        def preprocess_data(df):
            df_processed = df.copy()
            for col in df_processed.select_dtypes(include=["object"]).columns:
                if col in ALL_FACTORS:
                    le = LabelEncoder()
                    df_processed[col] = le.fit_transform(df_processed[col].astype(str))
                    print(f"Label-encoding non-numeric original factor '{col}'")
            return df_processed
        data_processed = preprocess_data(data_filtered)

        # 2. Feature processing (split, VIF, feature engineering)
        X = data_processed.drop(columns=[target_variable])
        y = data_processed[target_variable]

        # Handle collinearity (keep all original factors)
        print("\nHandling feature collinearity (keep all original factors)...")
        X_vif_filtered = calculate_vif(X, threshold=10.0, keep_factors=ALL_FACTORS)

        # Feature engineering (interaction terms + polynomial features)
        print("\nPerforming feature engineering...")
        remaining_original_factors = [f for f in ALL_FACTORS if f in X_vif_filtered.columns]
        X_engineered = create_interaction_features(X_vif_filtered, remaining_original_factors)
        X_engineered = add_polynomial_features(X_engineered, factor_of_interest, max_degree=3)

        # Split train / test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X_engineered, y, test_size=0.2, random_state=42
        )
        print(f"\nTraining set: {X_train.shape[0]} samples x {X_train.shape[1]} features")
        print(f"Test set: {X_test.shape[0]} samples x {X_test.shape[1]} features")

        # 3. Model training and SHAP computation
        print("\nTraining random forest model...")
        model = RandomForestRegressor(
            n_estimators=500,
            max_depth=100,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        print("\nComputing SHAP values...")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        shap_matrix = shap_values[0] if isinstance(shap_values, list) else shap_values

        # Create a SHAP-value DataFrame for the new algorithm
        shap_df = pd.DataFrame(
            shap_matrix,
            columns=X_test.columns,
            index=X_test.index
        )

        # Output original-factor SHAP importance
        shap_importance = calculate_original_factors_shap_mean_abs(shap_values, X_test, ALL_FACTORS)
        print("\n[mean(|SHAP|) ranking of original factors]")
        print(shap_importance.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

        # 4. Plot SHAP dependence (LOWESS trend + 95% confidence interval)
        if factor_of_interest not in X_test.columns:
            raise ValueError(f"Target factor '{factor_of_interest}' not in test set; cannot plot")

        x = X_test[factor_of_interest].values
        y_shap = shap_df[factor_of_interest].values

        # LOWESS trend fitting
        z = sm.nonparametric.lowess(y_shap, x, frac=0.3, return_sorted=True)
        x_sorted_raw, y_smoothed_raw = z[:, 0], z[:, 1]
        x_sorted, idx = np.unique(x_sorted_raw, return_index=True)
        y_smoothed = y_smoothed_raw[idx]

        # Confidence interval estimation
        f_fit = interp1d(x_sorted, y_smoothed, bounds_error=False, fill_value="extrapolate")
        residuals = y_shap - f_fit(x)
        std_err = np.std(residuals)
        y_upper = y_smoothed + 1.96 * std_err
        y_lower = y_smoothed - 1.96 * std_err

        # R^2 (Lowess) goodness-of-fit
        y_pred = f_fit(x)
        mask_valid = ~np.isnan(y_pred)
        r2 = r2_score(y_shap[mask_valid], y_pred[mask_valid])

        # Color mapping
        norm = Normalize(vmin=np.min(x), vmax=np.max(x))
        cmap = cm.cool
        colors = cmap(norm(x))

        # Plot
        plt.figure(figsize=(12, 8))
        ax = plt.gca()

        # Scatter plot
        ax.scatter(
            x, y_shap,
            color=colors,
            alpha=0.6,
            s=50,  # enlarge scatter points
            # label='SHAP value'
        )

        # Trend line
        ax.plot(
            x_sorted, y_smoothed,
            color='darkred',
            linewidth=2.5,
            label='Trend line (LOWESS)'
        )

        # Confidence interval
        ax.fill_between(
            x_sorted, y_lower, y_upper,
            color='lightcoral',
            alpha=0.3,
            label='95% confidence interval'
        )

        # Colorbar (commented out)
        # smap = cm.ScalarMappable(cmap=cmap, norm=norm)
        # cbar = plt.colorbar(smap, ax=ax)
        # cbar.set_label(f'{factor_of_interest} value', fontsize=12)  # keep if needed
        # Set colorbar number font to Times New Roman, keep size 25
        # cbar.ax.tick_params(labelsize=25, labelfontfamily='Times New Roman')

        # Legend and R^2 value (commented out)
        # handles, labels = ax.get_legend_handles_labels()
        # r2_handle = Line2D([], [], color='none', label=f'$R^2_{{LOWESS}}$ = {r2:.2f}')
        # handles.append(r2_handle)
        # labels.append(r2_handle.get_label())
        #
        # ax.legend(
        #     handles, labels,
        #     fontsize=10,
        #     loc='center left',
        #     bbox_to_anchor=(1.05, -0.15),
        #     borderaxespad=0.,
        #     frameon=True
        # )

        # Axis and title settings (commented out)
        # ax.set_xlabel(factor_of_interest, fontsize=14, fontweight='bold')
        # ax.set_ylabel(f'SHAP value of {factor_of_interest}', fontsize=14, fontweight='bold')
        # ax.set_title(f'SHAP value effect of {factor_of_interest} on {target_variable} - {zone_name}',
        #              fontsize=16, fontweight='bold')

        # Set axis number font to Times New Roman
        ax.tick_params(
            axis='both',
            which='major',
            labelsize=36,
            labelfontfamily='Arial'
        )

        # Grid lines
        ax.grid(alpha=0.3, linestyle="--")

        # Adjust axis range
        x_range = x_sorted.max() - x_sorted.min()
        y_range = np.max(y_shap) - np.min(y_shap)
        plt.xlim(x_sorted.min() - x_range*0.05, x_sorted.max() + x_range*0.05)
        plt.ylim(np.min(y_shap) - y_range*0.05, np.max(y_shap) + y_range*0.05)
        # Keep original limits
        plt.xlim(0, 0.9)
        plt.ylim(-0.39, 0.3)

        plt.tight_layout()
        plt.show()

        # Output built-in random-forest importance of original factors
        print("\n[Built-in random-forest importance of original factors]")
        feature_importance = pd.Series(model.feature_importances_, index=X_engineered.columns)
        original_importance = feature_importance[feature_importance.index.isin(ALL_FACTORS)].sort_values(ascending=False)
        print(original_importance)

    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
