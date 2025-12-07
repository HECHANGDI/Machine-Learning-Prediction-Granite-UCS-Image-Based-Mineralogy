# -*- coding: utf-8 -*-
"""
This script implements:
- SHAP-based feature importance analysis (no feature reduction)
- Explicit reporting of feature-to-sample size ratio
- Hyperparameter optimisation with Optuna
- Final XGBoost model training and evaluation
- k-fold (repeated) cross-validation with mean ± std of metrics
- Learning curve (training vs validation R²) for robustness
- Correlation matrix of features (multicollinearity check)
- Visualisations saved as high-resolution figures
- Text report of key results saved into a .txt file

All random seeds are fixed for reproducibility.
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
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

plt.rc('font', family='Times New Roman', size=7.5)

# ============================================================
# 1. Load dataset ONCE
# ============================================================
data = pd.read_excel("training_data_case_7.xlsx", header=0)

# Target (UCS): convert to MPa
y = data.iloc[:, -1] / 1e6
# Input features (all columns except first and target)
X_full = data.iloc[:, 1:-1]

print("Full data shape (samples, features):", X_full.shape)
print("Columns:", X_full.columns.tolist())

n_samples_full, n_features_full = X_full.shape
feature_to_sample_ratio_full = n_features_full / n_samples_full
print(
    f"\nFeature-to-sample size ratio (using ALL features): "
    f"{n_features_full} features / {n_samples_full} samples "
    f"= {feature_to_sample_ratio_full:.3f} "
    f"(~{n_samples_full / n_features_full:.2f} samples per feature)"
)

# ============================================================
# 2. SHAP Feature Importance (NO feature reduction)
# ============================================================
print("\n===== Stage 1: SHAP Feature Importance (no feature reduction) =====")

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

# Optional: mark "top" features for interpretation (NOT used for modelling)
rel_th = 0.02   # e.g. ≥ 2% of max SHAP
cum_th = 0.98   # within top 98% cumulative importance

shap_df["Selected"] = (
    (shap_df["RelativeImportance"] >= rel_th) &
    (shap_df["CumulativeImportance"] <= cum_th)
)

print("\nSHAP-based feature importance:")
print(shap_df)

# "Selected" features here are only for interpretation, NOT for feature reduction
selected_features = shap_df.loc[shap_df["Selected"], "Feature"].tolist()

print("\nTop features based on SHAP (for interpretation only; model uses ALL features):")
for f in selected_features:
    print(" -", f)
print("Number of top features (interpretation):", len(selected_features))

# ============================================================
# 2A. Dataset for modelling (ALL features)
# ============================================================
X = X_full  # <- use all features for training
n_samples, n_features = X.shape  # identical to n_samples_full, n_features_full

#%%

# ------------------------------------------------------------
# Simplify feature names for plotting
# Biotite (mm^2) -> Bi
# Quartz (mm^2) -> Qz
# Plagioclase (mm^2) -> Pl
# K-feldspar (mm^2) / K-Feldspar (mm^2) -> Kf
# Biotite_ -> Bi_, Quartz_ -> Qz_, Plagioclase_ -> Pl_,
# K-feldspar_/K-Feldspar_ -> Kf_
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
# 2.1 Visualisation: SHAP feature importance (bar plot)
# ============================================================
plt.figure(figsize=(8/2.54, 10/2.54))
plt.barh(shap_df["Feature"], shap_df["MeanAbsSHAP"])
plt.xlabel("Mean |SHAP value|")
plt.ylabel("Feature")
plt.title("SHAP-based feature importance (all features)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("shap_feature_importance_all_features.jpg", dpi=1000)
plt.show()

# ============================================================
# 2.1 Visualisation: Top-20 SHAP feature importance (full names)
# ============================================================
top_k = 20
shap_top = shap_df.head(top_k)

plt.figure(figsize=(7/2.54, 7/2.54))
plt.barh(shap_top["Feature"], shap_top["MeanAbsSHAP"])
plt.xlabel("Mean |SHAP value|")
plt.ylabel("Feature")
plt.title(f"Top {top_k} SHAP-based feature importance")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(f"shap_feature_importance_top{top_k}_fullnames.jpg", dpi=1000)
plt.show()

# ============================================================
# 2.1 Visualisation: Top-20 SHAP feature importance (short labels)
# ============================================================
plt.figure(figsize=(4.5/2.54, 7/2.54))
plt.barh(shap_top["FeatureLabel"], shap_top["MeanAbsSHAP"])
plt.xlabel("Mean |SHAP value|")
plt.ylabel("Feature")
plt.xlim(0, 10)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(f"shap_feature_importance_top{top_k}_shortlabels.jpg", dpi=1000)
plt.show()

#%%

# ============================================================
# 2.2 Visualisation: Correlation matrix of ALL features
#      (to inspect multicollinearity)
# ============================================================
corr_matrix = X.corr()

plt.figure(figsize=(10/2.54, 8/2.54))
im = plt.imshow(corr_matrix, vmin=-1, vmax=1, cmap='coolwarm')
plt.colorbar(im, fraction=0.046, pad=0.04)
plt.xticks(range(n_features), X.columns, rotation=90)
plt.yticks(range(n_features), X.columns)
plt.title("Correlation matrix of all features")
plt.tight_layout()
plt.savefig("correlation_matrix_all_features.jpg", dpi=1000)
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

#%%

# ============================================================
# 9. Visualization: Predicted vs Actual (Train/Test)
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
plt.savefig("predicted_vs_actual_all_features.jpg", dpi=1000)
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

#%%

# ------------------------------------------------------------
# 10.1 Visualisation: Distribution of R² across CV folds
# ------------------------------------------------------------
plt.figure(figsize=(6/2.54, 6/2.54))
plt.boxplot(r2_scores, vert=True)
plt.ylabel("R² (validation)")
plt.title(
    f"Repeated {n_splits}-fold CV\nDistribution of validation R² "
    f"(n = {n_splits * n_repeats} folds)"
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

plt.figure(figsize=(5/2.54, 5/2.54))

plt.fill_between(
    train_sizes,
    train_scores_mean - train_scores_std,
    train_scores_mean + train_scores_std,
    alpha=0.2
)
plt.plot(train_sizes, train_scores_mean, marker='o', markersize=4, label="Training R²")

for x_val, y_val in zip(train_sizes, train_scores_mean):
    plt.annotate(
        f"{y_val:.2f}",
        (x_val, y_val),
        textcoords="offset points",
        xytext=(0, 4),
        ha="center",
        fontsize=6
    )

plt.fill_between(
    train_sizes,
    val_scores_mean - val_scores_std,
    val_scores_mean + val_scores_std,
    alpha=0.2
)
plt.plot(
    train_sizes,
    val_scores_mean,
    marker='s',
    markersize=4,
    markerfacecolor='none',
    label="Validation R²"
)

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

print("\nScript finished successfully.")

# ============================================================
# 12. Save key results to a TXT report
# ============================================================
report_path = "xgb_results_summary.txt"

with open(report_path, "w") as f:
    f.write("===== XGBoost Modelling Report =====\n\n")

    # --------------------------------------------------------
    # Basic data information
    # --------------------------------------------------------
    f.write("1. Data Overview\n")
    f.write(f"Full data shape (samples, features): {X_full.shape}\n")
    f.write("Feature-to-sample size ratio (ALL features used in the model):\n")
    f.write(
        f"  {n_features_full} features / {n_samples_full} samples "
        f"= {feature_to_sample_ratio_full:.3f} "
        f"(~{n_samples_full / n_features_full:.2f} samples per feature)\n\n"
    )

    # --------------------------------------------------------
    # SHAP feature importance and "top" features
    # --------------------------------------------------------
    f.write("2. SHAP-based Feature Importance (sorted, all features):\n")
    f.write(
        shap_df[
            ["Feature", "MeanAbsSHAP",
             "RelativeImportance",
             "CumulativeImportance",
             "Selected"]
        ].to_string(index=False)
    )
    f.write("\n\n")

    f.write("Top features based on SHAP (for interpretation; ALL features used in model):\n")
    for feat in selected_features:
        f.write(f"  - {feat}\n")
    f.write(f"Number of top features (interpretation): {len(selected_features)}\n\n")

    # --------------------------------------------------------
    # Optuna hyperparameters
    # --------------------------------------------------------
    f.write("3. Optimal Hyperparameters (Optuna):\n")
    for key, value in best_params.items():
        f.write(f"  {key}: {value}\n")
    f.write("\n")

    # --------------------------------------------------------
    # Single train/test split performance
    # --------------------------------------------------------
    f.write("4. Final Model Performance (Single Train/Test Split)\n")
    f.write("Training Metrics:\n")
    f.write(
        f"  R²   : {r2_train:.4f}\n"
        f"  MAE  : {mae_train:.4f}\n"
        f"  RMSE : {rmse_train:.4f}\n"
        f"  MAPE : {mape_train:.2f}%\n"
    )
    f.write("Testing Metrics:\n")
    f.write(
        f"  R²   : {r2_test:.4f}\n"
        f"  MAE  : {mae_test:.4f}\n"
        f"  RMSE : {rmse_test:.4f}\n"
        f"  MAPE : {mape_test:.2f}%\n\n"
    )

    # --------------------------------------------------------
    # Repeated k-fold cross-validation summary
    # --------------------------------------------------------
    f.write("5. Repeated k-fold Cross-Validation Summary\n")
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
    # Learning curve summary (numeric dump)
    # --------------------------------------------------------
    f.write("6. Learning Curve (Training vs Validation R²)\n")
    f.write("Train sizes and mean R²:\n")
    for n_val, r_tr, r_val in zip(train_sizes, train_scores_mean, val_scores_mean):
        f.write(
            f"  n_train = {int(n_val):4d} | "
            f"Training R² = {r_tr:.4f}, "
            f"Validation R² = {r_val:.4f}\n"
        )

    f.write("\nReport generated successfully.\n")

print(f"\nResults report written to: {report_path}")
