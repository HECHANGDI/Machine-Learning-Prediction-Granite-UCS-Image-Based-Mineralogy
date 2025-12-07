# -*- coding: utf-8 -*-
"""
This script implements:
- SHAP-based feature selection
- Explicit reporting of the feature-to-sample size ratio
- Hyperparameter optimisation with Optuna
- Final XGBoost model training and evaluation
- Repeated k-fold cross-validation with mean ± std of metrics
- Learning curve (training vs validation R²) for robustness
- Correlation matrix of selected features (multicollinearity check)
- Robustness analysis across different random seeds
- Export of key results to an Excel workbook
- Export of key results to a TXT report
- Visualisations saved as high-resolution figures
"""

import numpy as np
import pandas as pd
import shap
import optuna
import matplotlib.pyplot as plt

from xgboost import XGBRegressor
from sklearn.model_selection import (
    train_test_split,
    RepeatedKFold,
    KFold,
    learning_curve
)
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

# ============================================================
# 0. Global configuration & random seeds
# ============================================================
# Base seed for the main analysis (SHAP, Optuna, CV, learning curves)
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Additional seeds for robustness analysis (Section 12)
SEED_LIST = [0, 12, 22, 32, 42]

print(f"Base RANDOM_SEED for main experiments: {RANDOM_SEED}")
print(f"Random seeds for robustness analysis: {SEED_LIST}")

plt.rc('font', family='Times New Roman', size=7.5)

# ============================================================
# 1. Load dataset ONCE
# ============================================================
data = pd.read_excel("training_data_case_4.xlsx", header=0)

# Target (UCS): convert to MPa
y = data.iloc[:, -1] / 1e6
# Input features (all columns except first and target)
X_full = data.iloc[:, 1:-1]

print("Full data shape (samples, features):", X_full.shape)
print("Columns:", X_full.columns.tolist())

n_samples_full, n_features_full = X_full.shape
feature_to_sample_ratio_full = n_features_full / n_samples_full
print(
    f"\nOriginal feature-to-sample size ratio: "
    f"{n_features_full} features / {n_samples_full} samples "
    f"= {feature_to_sample_ratio_full:.3f} "
    f"(~{n_samples_full / n_features_full:.2f} samples per feature)"
)

# ============================================================
# 2. SHAP Feature Selection
# ============================================================
print("\n===== Stage 1: SHAP Feature Selection =====")

# Base model for SHAP importance (fixed seed)
base_model = XGBRegressor(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_SEED
)

base_model.fit(X_full, y)

explainer = shap.TreeExplainer(base_model)
shap_values = explainer.shap_values(X_full)

# Mean absolute SHAP value per feature
mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

shap_df = pd.DataFrame({
    "Feature": X_full.columns,
    "MeanAbsSHAP": mean_abs_shap
})

shap_df = shap_df.sort_values("MeanAbsSHAP", ascending=False).reset_index(drop=True)

# Relative and cumulative importance
max_shap = shap_df["MeanAbsSHAP"].iloc[0]
shap_df["RelativeImportance"] = shap_df["MeanAbsSHAP"] / max_shap
shap_df["CumulativeImportance"] = (
    shap_df["MeanAbsSHAP"].cumsum() / shap_df["MeanAbsSHAP"].sum()
)

# Selection thresholds
rel_th = 0.02   # ≥ 2% of max SHAP
cum_th = 0.98   # within top 98% cumulative importance

shap_df["Selected"] = (
    (shap_df["RelativeImportance"] >= rel_th) &
    (shap_df["CumulativeImportance"] <= cum_th)
)

print("\nSHAP-based feature importance:")
print(shap_df)

# Selected features
selected_features = shap_df.loc[shap_df["Selected"], "Feature"].tolist()

print("\nSelected features for machine learning:")
for f in selected_features:
    print(" -", f)
print("Number of selected features:", len(selected_features))

# Final input dataset (reduced feature set)
X = X_full[selected_features]

n_samples, n_features = X.shape
feature_to_sample_ratio = n_features / n_samples
print(
    f"\nReduced feature-to-sample size ratio (after SHAP selection): "
    f"{n_features} features / {n_samples} samples "
    f"= {feature_to_sample_ratio:.3f} "
    f"(~{n_samples / n_features:.2f} samples per feature)"
)

# ------------------------------------------------------------
# Simplify feature names for plotting
# ------------------------------------------------------------
shap_df["FeatureLabel"] = shap_df["Feature"]

exact_map = {
    "Biotite (mm^2)": "Bi",
    "Quartz (mm^2)": "Qz",
    "Plagioclase (mm^2)": "Pl",
    "K-feldspar (mm^2)": "Kf",
    "K-Feldspar (mm^2)": "Kf",
}
shap_df["FeatureLabel"] = shap_df["FeatureLabel"].replace(exact_map)

shap_df["FeatureLabel"] = (
    shap_df["FeatureLabel"]
    .str.replace(r"^Biotite_", "Bi_", regex=True)
    .str.replace(r"^Quartz_", "Qz_", regex=True)
    .str.replace(r"^Plagioclase_", "Pl_", regex=True)
    .str.replace(r"^K[- ]?[Ff]eldspar_", "Kf_", regex=True)
)

# ============================================================
# 2.1 Visualisation: SHAP feature importance (all features)
# ============================================================
plt.figure(figsize=(8/2.54, 10/2.54))
plt.barh(shap_df["Feature"], shap_df["MeanAbsSHAP"])
plt.xlabel("Mean |SHAP value|")
plt.ylabel("Feature")
plt.title("SHAP-based feature importance")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("shap_feature_importance_all_features.jpg", dpi=1000)
plt.show()

# ============================================================
# 2.2 Visualisation: Top-20 SHAP feature importance (simplified labels)
# ============================================================
top_k = 20
shap_top = shap_df.head(top_k)

plt.figure(figsize=(4.5/2.54, 7/2.54))
plt.barh(shap_top["FeatureLabel"], shap_top["MeanAbsSHAP"])
plt.xlabel("Mean |SHAP value|")
plt.ylabel("Feature")
plt.xlim(0, 10)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(f"shap_feature_importance_top{top_k}.jpg", dpi=1000)
plt.show()

# ============================================================
# 2.3 Visualisation: Correlation matrix of selected features
# ============================================================
corr_matrix = X.corr()

plt.figure(figsize=(10/2.54, 8/2.54))
im = plt.imshow(corr_matrix, vmin=-1, vmax=1, cmap='coolwarm')
plt.colorbar(im, fraction=0.046, pad=0.04)
plt.xticks(range(n_features), X.columns, rotation=90)
plt.yticks(range(n_features), X.columns)
plt.title("Correlation matrix of selected features")
plt.tight_layout()
plt.savefig("correlation_matrix_selected_features.jpg", dpi=1000)
plt.show()

# ============================================================
# 3. Train/Test Split (for final model illustration)
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_SEED
)

# ============================================================
# 4. Evaluation function
# ============================================================
def evaluate_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return r2, mae, mse, rmse, mape

# ============================================================
# 5. Optuna objective function (uses train/test split)
# ============================================================
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 20),
        "alpha": trial.suggest_float("alpha", 0, 20),
        "lambda": trial.suggest_float("lambda", 0, 20),
    }

    model = XGBRegressor(**params, random_state=RANDOM_SEED, n_jobs=1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Minimise 1 - R²
    return 1 - r2_score(y_test, y_pred)

# ============================================================
# 6. Run Optuna optimization
# ============================================================
print("\n===== Stage 2: Hyperparameter Optimization (Optuna) =====")
study = optuna.create_study(
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
)
study.optimize(objective, n_trials=8000)

best_params = study.best_params
print("\nOptimal Parameters:")
print(best_params)

# ============================================================
# 7. Train final model (single train/test split)
# ============================================================
model_final = XGBRegressor(**best_params, random_state=RANDOM_SEED, n_jobs=1)
model_final.fit(X_train, y_train)

y_train_pred = model_final.predict(X_train)
y_test_pred = model_final.predict(X_test)

# ============================================================
# 8. Print metrics (single split)
# ============================================================
r2_train, mae_train, mse_train, rmse_train, mape_train = evaluate_metrics(
    y_train, y_train_pred
)
r2_test, mae_test, mse_test, rmse_test, mape_test = evaluate_metrics(
    y_test, y_test_pred
)

print("\n===== Final Model Performance (Single Train/Test Split) =====")
print("Training Metrics:")
print(
    f"R²: {r2_train:.4f}, MAE: {mae_train:.4f}, "
    f"RMSE: {rmse_train:.4f}, MAPE: {mape_train:.2f}%"
)
print("Testing Metrics:")
print(
    f"R²: {r2_test:.4f}, MAE: {mae_test:.4f}, "
    f"RMSE: {rmse_test:.4f}, MAPE: {mape_test:.2f}%"
)

# Prepare single-split metrics DataFrame (for Excel export)
single_split_df = pd.DataFrame({
    "Set": ["Train", "Test"],
    "R2": [r2_train, r2_test],
    "RMSE": [rmse_train, rmse_test],
    "MAE": [mae_train, mae_test],
    "MAPE(%)": [mape_train, mape_test],
})

# ============================================================
# 9. Visualisation: Predicted vs Actual (Train/Test)
# ============================================================
plt.figure(figsize=(12/2.54, 6/2.54))

# Training plot
plt.subplot(1, 2, 1)
plt.scatter(y_train, y_train_pred, alpha=0.7)
plt.plot(
    [y_train.min(), y_train.max()],
    [y_train.min(), y_train.max()],
    'r--'
)
plt.xlabel("Actual UCS (MPa)")
plt.ylabel("Predicted UCS (MPa)")
plt.ylim(100, 182)
plt.title(f"Train (R² = {r2_train:.3f})")

# Testing plot
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_test_pred, alpha=0.7)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'r--'
)
plt.xlabel("Actual UCS (MPa)")
plt.ylabel("Predicted UCS (MPa)")
plt.ylim(100, 182)
plt.title(f"Test (R² = {r2_test:.3f})")

plt.tight_layout()
plt.savefig("predicted_vs_actual_SHAP_selected_features.jpg", dpi=1000)
plt.show()

# ============================================================
# 10. Repeated k-fold cross-validation for robustness
# ============================================================
print("\n===== Repeated k-fold Cross-Validation with Tuned Model =====")

n_splits = 5
n_repeats = 10
rkf = RepeatedKFold(
    n_splits=n_splits,
    n_repeats=n_repeats,
    random_state=RANDOM_SEED
)

r2_scores = []
mae_scores = []
rmse_scores = []
mape_scores = []

for fold_idx, (train_idx, test_idx) in enumerate(rkf.split(X, y), start=1):
    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

    model_cv = XGBRegressor(**best_params, random_state=RANDOM_SEED, n_jobs=1)
    model_cv.fit(X_tr, y_tr)
    y_te_pred = model_cv.predict(X_te)

    r2, mae, mse, rmse, mape = evaluate_metrics(y_te, y_te_pred)

    r2_scores.append(r2)
    mae_scores.append(mae)
    rmse_scores.append(rmse)
    mape_scores.append(mape)

r2_mean, r2_std = np.mean(r2_scores), np.std(r2_scores)
mae_mean, mae_std = np.mean(mae_scores), np.std(mae_scores)
rmse_mean, rmse_std = np.mean(rmse_scores), np.std(rmse_scores)
mape_mean, mape_std = np.mean(mape_scores), np.std(mape_scores)

print(
    f"\nRepeated {n_splits}-fold CV (n_repeats={n_repeats}) results "
    f"(mean ± SD over {n_splits * n_repeats} folds):"
)
print(f"R²   : {r2_mean:.4f} ± {r2_std:.4f}")
print(f"MAE  : {mae_mean:.4f} ± {mae_std:.4f}")
print(f"RMSE : {rmse_mean:.4f} ± {rmse_std:.4f}")
print(f"MAPE : {mape_mean:.2f}% ± {mape_std:.2f}%")

# Per-fold CV metrics DataFrame
cv_folds_df = pd.DataFrame({
    "Fold": np.arange(1, len(r2_scores) + 1),
    "R2": r2_scores,
    "RMSE": rmse_scores,
    "MAE": mae_scores,
    "MAPE(%)": mape_scores,
})

# Summary row for CV
cv_summary_df = pd.DataFrame({
    "Metric": ["R2", "RMSE", "MAE", "MAPE(%)"],
    "Mean": [r2_mean, rmse_mean, mae_mean, mape_mean],
    "Std": [r2_std, rmse_std, mae_std, mape_std],
})

# ------------------------------------------------------------
# 10.1 Visualisation: Distribution of R² across CV folds
# ------------------------------------------------------------
plt.figure(figsize=(6/2.54, 6/2.54))
plt.boxplot(r2_scores, vert=True)
plt.ylabel("R² (validation)")
plt.title(
    f"Repeated {n_splits}-fold CV\n"
    f"Distribution of validation R² (n = {n_splits * n_repeats} folds)"
)
plt.tight_layout()
plt.savefig("xgb_repeated_kfold_r2_distribution.jpg", dpi=1000)
plt.show()

# ============================================================
# 11. Learning curve (training vs validation R²)
# ============================================================
print("\n===== Learning Curve (Training vs Validation R²) =====")

cv_for_lc = KFold(
    n_splits=5, shuffle=True, random_state=RANDOM_SEED
)

estimator_lc = XGBRegressor(
    **best_params,
    random_state=RANDOM_SEED,
    n_jobs=1
)

train_sizes, train_scores, val_scores = learning_curve(
    estimator=estimator_lc,
    X=X,
    y=y,
    cv=cv_for_lc,
    scoring="r2",
    train_sizes=np.linspace(0.2, 1.0, 5),
    n_jobs=1
)

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
val_scores_mean = np.mean(val_scores, axis=1)
val_scores_std = np.std(val_scores, axis=1)

# Put learning-curve data into a DataFrame for exporting
lc_df = pd.DataFrame({
    "train_size": train_sizes,
    "R2_train_mean": train_scores_mean,
    "R2_train_std": train_scores_std,
    "R2_val_mean": val_scores_mean,
    "R2_val_std": val_scores_std,
})

plt.figure(figsize=(5/2.54, 5/2.54))

# Training curve
plt.fill_between(
    train_sizes,
    train_scores_mean - train_scores_std,
    train_scores_mean + train_scores_std,
    alpha=0.2
)
plt.plot(train_sizes, train_scores_mean, marker='o', markersize=4,
         label="Training R²")

for x_val, y_val in zip(train_sizes, train_scores_mean):
    plt.annotate(
        f"{y_val:.2f}",
        (x_val, y_val),
        textcoords="offset points",
        xytext=(0, 4),
        ha="center",
        fontsize=6
    )

# Validation curve
plt.fill_between(
    train_sizes,
    val_scores_mean - val_scores_std,
    val_scores_mean + val_scores_std,
    alpha=0.2
)
plt.plot(train_sizes, val_scores_mean,
         marker='s', markersize=4, markerfacecolor='none',
         label="Validation R²")

for x_val, y_val in zip(train_sizes, val_scores_mean):
    plt.annotate(
        f"{y_val:.2f}",
        (x_val, y_val),
        textcoords="offset points",
        xytext=(0, -8),
        ha="center",
        fontsize=6
    )

plt.xlabel("Number of training samples")
plt.ylabel("R²")
plt.ylim(-3.5, 1.2)
# plt.legend()
plt.tight_layout()
plt.savefig("xgb_learning_curve_r2_with_labels.jpg", dpi=1000)
plt.show()

print("\nLearning curve finished.")

# ============================================================
# 12. Robustness analysis across different random seeds
# ============================================================
print("\n===== Robustness analysis across random seeds =====")

robust_results = []

for seed in SEED_LIST:
    # Re-split training and test sets for each seed
    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
        X, y, test_size=0.25, random_state=seed
    )

    # Use the same optimised hyperparameters but different random_state
    model_seed = XGBRegressor(**best_params, random_state=seed, n_jobs=1)
    model_seed.fit(X_train_s, y_train_s)

    # Compute metrics
    y_train_pred_s = model_seed.predict(X_train_s)
    y_test_pred_s = model_seed.predict(X_test_s)

    r2_train_s, mae_train_s, mse_train_s, rmse_train_s, mape_train_s = \
        evaluate_metrics(y_train_s, y_train_pred_s)
    r2_test_s, mae_test_s, mse_test_s, rmse_test_s, mape_test_s = \
        evaluate_metrics(y_test_s, y_test_pred_s)

    robust_results.append({
        "seed": seed,
        "R2_train": r2_train_s,
        "R2_test": r2_test_s,
        "RMSE_train": rmse_train_s,
        "RMSE_test": rmse_test_s,
        "MAE_train": mae_train_s,
        "MAE_test": mae_test_s,
        "MAPE_train(%)": mape_train_s,
        "MAPE_test(%)": mape_test_s,
    })

robust_df = pd.DataFrame(robust_results)
print("\nPerformance across different random seeds (single train/test split):")
print(robust_df)

# 12.1 Visualise test R² across seeds (single line)
plt.figure(figsize=(8/2.54, 6/2.54))
plt.plot(
    robust_df["seed"],
    robust_df["R2_test"],
    marker="o",
    linestyle="-"
)
for x_val, y_val in zip(robust_df["seed"], robust_df["R2_test"]):
    plt.annotate(
        f"{y_val:.3f}",
        (x_val, y_val),
        textcoords="offset points",
        xytext=(0, 4),
        ha="center",
        fontsize=6
    )

plt.xlabel("Random seed")
plt.ylabel("Test R²")
plt.tight_layout()
plt.savefig("xgb_seed_robustness_R2_test.jpg", dpi=1000)
plt.show()

# 12.2 Visualisation: R² (train/test) and MAPE (train/test) in one 1×2 figure
print("\nPlotting combined R² + MAPE robustness figure...")

x_labels = robust_df["seed"].astype(str)
x = np.arange(len(x_labels))

fig, axes = plt.subplots(
    1, 2,
    figsize=(12/2.54, 5/2.54),
    sharex=True
)

# (a) R² train / test
ax = axes[0]
ax.plot(x, robust_df["R2_train"], marker='o', linestyle='-',
        label=r"$R^2_{\mathrm{train}}$")
ax.plot(x, robust_df["R2_test"], marker='s', linestyle='--',
        label=r"$R^2_{\mathrm{test}}$")

for xi, yt, yv in zip(x, robust_df["R2_train"], robust_df["R2_test"]):
    ax.annotate(f"{yt:.2f}", (xi, yt),
                textcoords="offset points", xytext=(0, -10),
                ha="center", fontsize=7.5)
    ax.annotate(f"{yv:.2f}", (xi, yv),
                textcoords="offset points", xytext=(-1, 8),
                ha="center", fontsize=7.5)

ax.set_xticks(x)
ax.set_xticklabels(x_labels)
ax.set_xlabel("Random seed")
ax.set_ylabel(r"$R^2$")
ax.grid(True, linestyle="--", alpha=0.4)
ax.legend(fontsize=7.5, loc="center")

# (b) MAPE train / test
ax = axes[1]
ax.plot(x, robust_df["MAPE_train(%)"], marker='o', linestyle='-',
        label="MAPE (train)")
ax.plot(x, robust_df["MAPE_test(%)"], marker='s', linestyle='--',
        label="MAPE (test)")

for xi, yt, yv in zip(x, robust_df["MAPE_train(%)"], robust_df["MAPE_test(%)"]):
    ax.annotate(f"{yt:.2f}", (xi, yt),
                textcoords="offset points", xytext=(0, 8),
                ha="center", fontsize=7.5)
    ax.annotate(f"{yv:.2f}", (xi, yv),
                textcoords="offset points", xytext=(0, -10),
                ha="center", fontsize=7.5)

ax.set_xticks(x)
ax.set_xticklabels(x_labels)
ax.set_xlabel("Random seed")
ax.set_ylabel("MAPE (\%)")
ax.grid(True, linestyle="--", alpha=0.4)
ax.legend(fontsize=7.5, loc="center", facecolor="white", framealpha=1.0)

plt.tight_layout()
plt.savefig("seed_robustness_R2_MAPE.jpg", dpi=1200)
plt.show()

print("\nRobustness analysis finished.")

# ============================================================
# 13. Export key results to Excel
# ============================================================
output_excel = "xgb_results_case4.xlsx"
print(f"\nSaving key results to Excel file: {output_excel}")

with pd.ExcelWriter(output_excel, engine="xlsxwriter") as writer:
    # SHAP importance (all features)
    shap_df.to_excel(writer, sheet_name="SHAP_all", index=False)

    # Top-20 SHAP
    shap_top.to_excel(writer, sheet_name="SHAP_top20", index=False)

    # Single train/test split metrics
    single_split_df.to_excel(writer, sheet_name="Single_split_metrics", index=False)

    # CV per-fold metrics
    cv_folds_df.to_excel(writer, sheet_name="CV_folds", index=False)

    # CV summary (mean ± std)
    cv_summary_df.to_excel(writer, sheet_name="CV_summary", index=False)

    # Learning curve data
    lc_df.to_excel(writer, sheet_name="Learning_curve", index=False)

    # Robustness across seeds
    robust_df.to_excel(writer, sheet_name="Seed_robustness", index=False)

print("Excel export finished.")

# ============================================================
# 14. Save key results to a TXT report
# ============================================================
report_path = "xgb_results_summary_case4.txt"

with open(report_path, "w") as f:
    f.write("===== XGBoost Modelling Report (Case 4, SHAP-selected features) =====\n\n")

    # --------------------------------------------------------
    # 1. Data overview
    # --------------------------------------------------------
    f.write("1. Data overview\n")
    f.write(f"Full data shape (samples, features): {X_full.shape}\n")
    f.write("Original feature-to-sample size ratio:\n")
    f.write(
        f"  {n_features_full} features / {n_samples_full} samples "
        f"= {feature_to_sample_ratio_full:.3f} "
        f"(~{n_samples_full / n_features_full:.2f} samples per feature)\n\n"
    )

    f.write("After SHAP selection (reduced feature set used for modelling):\n")
    f.write(
        f"  {n_features} features / {n_samples} samples "
        f"= {feature_to_sample_ratio:.3f} "
        f"(~{n_samples / n_features:.2f} samples per feature)\n\n"
    )

    # --------------------------------------------------------
    # 2. SHAP feature importance and selected features
    # --------------------------------------------------------
    f.write("2. SHAP-based Feature Importance (all features, sorted):\n")
    f.write(
        shap_df[
            ["Feature", "MeanAbsSHAP",
             "RelativeImportance",
             "CumulativeImportance",
             "Selected"]
        ].to_string(index=False)
    )
    f.write("\n\n")

    f.write("Selected features used in the final model:\n")
    for feat in selected_features:
        f.write(f"  - {feat}\n")
    f.write(f"Number of selected features: {len(selected_features)}\n\n")

    # --------------------------------------------------------
    # 3. Optimal hyperparameters (Optuna)
    # --------------------------------------------------------
    f.write("3. Optimal hyperparameters (Optuna, based on single train/test split):\n")
    for key, value in best_params.items():
        f.write(f"  {key}: {value}\n")
    f.write("\n")

    # --------------------------------------------------------
    # 4. Single train/test split performance
    # --------------------------------------------------------
    f.write("4. Final model performance (single train/test split)\n")
    f.write("Training metrics:\n")
    f.write(
        f"  R²   : {r2_train:.4f}\n"
        f"  MAE  : {mae_train:.4f}\n"
        f"  RMSE : {rmse_train:.4f}\n"
        f"  MAPE : {mape_train:.2f}%\n"
    )
    f.write("Testing metrics:\n")
    f.write(
        f"  R²   : {r2_test:.4f}\n"
        f"  MAE  : {mae_test:.4f}\n"
        f"  RMSE : {rmse_test:.4f}\n"
        f"  MAPE : {mape_test:.2f}%\n\n"
    )

    # --------------------------------------------------------
    # 5. Repeated k-fold cross-validation summary
    # --------------------------------------------------------
    f.write("5. Repeated k-fold cross-validation summary\n")
    f.write(
        f"  Setting: {n_splits}-fold, {n_repeats} repeats "
        f"({n_splits * n_repeats} total folds)\n"
    )
    f.write(
        f"  R²   : {r2_mean:.4f} ± {r2_std:.4f}\n"
        f"  MAE  : {mae_mean:.4f} ± {mae_std:.4f}\n"
        f"  RMSE : {rmse_mean:.4f} ± {rmse_std:.4f}\n"
        f"  MAPE : {mape_mean:.2f}% ± {mape_std:.2f}%\n\n"
    )

    # --------------------------------------------------------
    # 6. Learning curve summary
    # --------------------------------------------------------
    f.write("6. Learning curve (training vs validation R²)\n")
    f.write("Train size, mean R² (train), mean R² (validation):\n")
    for n_tr, r_tr, r_val in zip(train_sizes, train_scores_mean, val_scores_mean):
        f.write(
            f"  n_train = {int(n_tr):4d} | "
            f"R²_train = {r_tr:.4f}, "
            f"R²_val = {r_val:.4f}\n"
        )
    f.write("\n")

    # --------------------------------------------------------
    # 7. Robustness across random seeds
    # --------------------------------------------------------
    f.write("7. Robustness analysis across random seeds\n")
    f.write("Single train/test split per seed, using the same tuned hyperparameters:\n")
    f.write(robust_df.to_string(index=False))
    f.write("\n\n")

    # --------------------------------------------------------
    # 8. Excel export information
    # --------------------------------------------------------
    f.write("8. Excel export\n")
    f.write(f"Key tables have also been saved to Excel file: {output_excel}\n")
    f.write("  Sheets include:\n")
    f.write("    - SHAP_all\n")
    f.write("    - SHAP_top20\n")
    f.write("    - Single_split_metrics\n")
    f.write("    - CV_folds\n")
    f.write("    - CV_summary\n")
    f.write("    - Learning_curve\n")
    f.write("    - Seed_robustness\n\n")

    f.write("Report generated successfully.\n")

print(f'\nResults report written to: {report_path}')

print("\nScript finished successfully.")
