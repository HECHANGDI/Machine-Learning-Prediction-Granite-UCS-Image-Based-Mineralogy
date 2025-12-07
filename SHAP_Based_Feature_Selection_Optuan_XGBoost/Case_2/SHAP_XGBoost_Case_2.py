# -*- coding: utf-8 -*-
"""
This script implements:
- SHAP-based feature selection
- Explicit reporting of feature-to-sample size ratio
- Hyperparameter optimisation with Optuna
- Final XGBoost model training and evaluation
- k-fold (repeated) cross-validation with mean ± std of metrics
- Learning curve (training vs validation R²) for robustness
- Correlation matrix of selected features (multicollinearity check)
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
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

plt.rc('font', family='Times New Roman', size=7.5)

# ============================================================
# 1. Load dataset ONCE
# ============================================================
data = pd.read_excel("training_data_case_2.xlsx", header=0)

# Target (UCS): convert to MPa
y = data.iloc[:, -1] / 1e6
# Input features (all columns except target)
X_full = data.iloc[:, 1:-1]      

print("Full data shape (samples, features):", X_full.shape)
print("Columns:", X_full.columns.tolist())

n_samples_full, n_features_full = X_full.shape
feature_to_sample_ratio_full = n_features_full / n_samples_full
print(f"\nOriginal feature-to-sample size ratio: "
      f"{n_features_full} features / {n_samples_full} samples "
      f"= {feature_to_sample_ratio_full:.3f} "
      f"(~{n_samples_full / n_features_full:.2f} samples per feature)")

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

# Selection thresholds, with 0 and 1.0, all features are selected in Cases 1 to 3
# Setting specific thresholds to select top 15 features in Cases 4 to 7
rel_th = 0   # ≥ 0 of max SHAP
cum_th = 1.0   # within top 100% cumulative importance

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
print(f"\nReduced feature-to-sample size ratio (after SHAP selection): "
      f"{n_features} features / {n_samples} samples "
      f"= {feature_to_sample_ratio:.3f} "
      f"(~{n_samples / n_features:.2f} samples per feature)")
#%%
# ============================================================
# 2.0 Simplify feature names for plotting
# Biotite (mm^2) -> Bi
# Quartz (mm^2) -> Qz
# Plagioclase (mm^2) -> Pl
# K-feldspar (mm^2) / K-Feldspar (mm^2) -> Kf
# Biotite_ -> Bi_, Quartz_ -> Qz_, Plagioclase_ -> Pl_, K-feldspar_/K-Feldspar_ -> Kf_
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
# 2.1 Visualisation: Top-20 SHAP feature importance (bar plot)
# ============================================================
top_k = 20
shap_top = shap_df.head(top_k) 

plt.figure(figsize=(4.5/2.54, 7/2.54))
plt.barh(shap_top["FeatureLabel"], shap_top["MeanAbsSHAP"])
plt.xlabel("Mean |SHAP value|")
plt.ylabel("Feature")
plt.xlim(0, 20)
# plt.title(f"Top {top_k} SHAP-based feature importance")
plt.gca().invert_yaxis() 
plt.tight_layout()
plt.savefig(f"shap_feature_importance_top{top_k}.jpg", dpi=1000)
plt.show()

# ============================================================
# 2.2 Visualisation: Correlation matrix of selected features
#      (to inspect multicollinearity)
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
# 2.3 Quantifying multicollinearity: VIF before vs after SHAP
# ============================================================
from statsmodels.stats.outliers_influence import variance_inflation_factor

def compute_vif(df):
    """Compute Variance Inflation Factor (VIF) for each column in a DataFrame."""
    X_np = df.values
    vif_values = [variance_inflation_factor(X_np, i) for i in range(X_np.shape[1])]
    return pd.DataFrame({"Feature": df.columns, "VIF": vif_values})

# VIF using all features (before SHAP selection)
vif_full = compute_vif(X_full)

# VIF using SHAP-selected features (after selection)
vif_selected = compute_vif(X)

print("\nVIF before SHAP selection (all features):")
print(vif_full.sort_values("VIF", ascending=False).head(10))

print("\nVIF after SHAP selection (selected features):")
print(vif_selected.sort_values("VIF", ascending=False))


name_map = dict(zip(shap_df["Feature"], shap_df["FeatureLabel"]))

vif_full_labeled = vif_full.copy()
vif_full_labeled["FeatureLabel"] = (
    vif_full_labeled["Feature"].map(name_map).fillna(vif_full_labeled["Feature"])
)

vif_selected_labeled = vif_selected.copy()
vif_selected_labeled["FeatureLabel"] = (
    vif_selected_labeled["Feature"].map(name_map).fillna(vif_selected_labeled["Feature"])
)

# ------------------------------------------------------------
# 2.4 Visualisation: VIF distribution before vs after SHAP
# ------------------------------------------------------------
plt.figure(figsize=(8/2.54, 6/2.54))
plt.boxplot(
    [vif_full_labeled["VIF"], vif_selected_labeled["VIF"]],
    labels=["Before SHAP", "After SHAP"],
    showfliers=False
)
plt.ylabel("VIF")
plt.title("Distribution of VIF values\nbefore and after SHAP-based feature selection")
plt.tight_layout()
plt.savefig("vif_before_after_SHAP_boxplot.jpg", dpi=1000)
plt.show()

# -----------------------------------------------------------
# ------------------------------------------------------------
top_k_vif = 20  # plot top-20 most collinear features in the full set

vif_full_sorted = (
    vif_full_labeled
    .sort_values("VIF", ascending=False)
    .tail(top_k_vif)
)

plt.figure(figsize=(4.5/2.54, 7/2.54))
plt.barh(vif_full_sorted["FeatureLabel"], vif_full_sorted["VIF"])
plt.xlabel("VIF")
plt.ylabel("Feature")
plt.gca().invert_yaxis()   
plt.tight_layout()
plt.savefig("vif_top_features_before_SHAP.jpg", dpi=1000)
plt.show()

# ------------------------------------------------------------
# ------------------------------------------------------------
vif_selected_sorted = vif_selected_labeled.sort_values("VIF", ascending=False)

plt.figure(figsize=(4.5/2.54, 6/2.54))
plt.barh(vif_selected_sorted["FeatureLabel"], vif_selected_sorted["VIF"])
plt.xlabel("VIF")
plt.xlim(0,15)
plt.ylabel("Feature")
plt.gca().invert_yaxis()  
plt.tight_layout()
plt.savefig("vif_selected_features_after_SHAP.jpg", dpi=1000)
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
print(f"R²: {r2_train:.4f}, MAE: {mae_train:.4f}, "
      f"RMSE: {rmse_train:.4f}, MAPE: {mape_train:.2f}%")
print("Testing Metrics:")
print(f"R²: {r2_test:.4f}, MAE: {mae_test:.4f}, "
      f"RMSE: {rmse_test:.4f}, MAPE: {mape_test:.2f}%")

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
plt.savefig("predicted_vs_actual_SHAP_selected_features.jpg", dpi=1000)
plt.show()

# ============================================================
# 10. Repeated k-fold cross-validation for robustness
#     (addresses reviewer: k-fold CV + mean ± std of metrics)
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

print(f"\nRepeated {n_splits}-fold CV (n_repeats={n_repeats}) results "
      f"(mean ± SD over {n_splits * n_repeats} folds):")
print(f"R²   : {r2_mean:.4f} ± {r2_std:.4f}")
print(f"MAE  : {mae_mean:.4f} ± {mae_std:.4f}")
print(f"RMSE : {rmse_mean:.4f} ± {rmse_std:.4f}")
print(f"MAPE : {mape_mean:.2f}% ± {mape_std:.2f}%")

# ------------------------------------------------------------
# 11. Visualisation: Distribution of R² across CV folds
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
#%%
# ============================================================
# 12. Learning curve (training vs validation R²)
#     (addresses reviewer: learning curves for robustness)
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
plt.plot(train_sizes, train_scores_mean, marker='o', markersize=4,label="Training R²")


for x, y in zip(train_sizes, train_scores_mean):
    plt.annotate(
        f"{y:.2f}",
        (x, y),
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
plt.plot(train_sizes, val_scores_mean, marker='s', markersize=4, markerfacecolor='none',label="Validation R²")

for x, y in zip(train_sizes, val_scores_mean):
    plt.annotate(
        f"{y:.2f}",
        (x, y),
        textcoords="offset points",
        xytext=(0, -8),  
        ha="center",
        fontsize=6
    )

plt.xlabel("Number of training samples")
plt.ylabel("R²")
plt.ylim(-3.5,1.2)
# plt.title("Learning curve of XGBoost model")
plt.legend()
plt.tight_layout()
plt.savefig("xgb_learning_curve_r2_with_labels_1.jpg", dpi=1000)
plt.show()

print("\nScript finished successfully.")
#%%
# ============================================================
# 13. Save key results to a text file
# ============================================================
results_file = "xgb_shap_optuna_case1_results.txt"

with open(results_file, "w", encoding="utf-8") as f:
    f.write("===== Dataset & Feature Information =====\n")
    f.write(f"Full data shape (samples, features): {n_samples_full} x {n_features_full}\n")
    f.write(f"Original feature-to-sample size ratio: "
            f"{n_features_full} / {n_samples_full} = {feature_to_sample_ratio_full:.3f} "
            f"(~{n_samples_full / n_features_full:.2f} samples per feature)\n\n")

    f.write("Selected features after SHAP:\n")
    for feat in selected_features:
        f.write(f"  - {feat}\n")
    f.write(f"Number of selected features: {len(selected_features)}\n")
    f.write(f"Reduced feature-to-sample size ratio: "
            f"{n_features} / {n_samples} = {feature_to_sample_ratio:.3f} "
            f"(~{n_samples / n_features:.2f} samples per feature)\n\n")

    f.write("===== Hyperparameter Optimisation (Optuna) =====\n")
    f.write("Best parameters:\n")
    for k, v in best_params.items():
        f.write(f"  {k}: {v}\n")
    f.write("\n")

    f.write("===== Single Train/Test Split Performance =====\n")
    f.write("Training metrics:\n")
    f.write(f"  R²   : {r2_train:.4f}\n")
    f.write(f"  MAE  : {mae_train:.4f}\n")
    f.write(f"  RMSE : {rmse_train:.44f}\n")
    f.write(f"  MAPE : {mape_train:.2f}%\n\n")

    f.write("Testing metrics:\n")
    f.write(f"  R²   : {r2_test:.4f}\n")
    f.write(f"  MAE  : {mae_test:.4f}\n")
    f.write(f"  RMSE : {rmse_test:.4f}\n")
    f.write(f"  MAPE : {mape_test:.2f}%\n\n")

    f.write("===== Repeated k-fold Cross-Validation =====\n")
    f.write(f"Setting: {n_splits}-fold, {n_repeats} repeats "
            f"({n_splits * n_repeats} folds in total)\n")
    f.write(f"  R²   : {r2_mean:.4f} ± {r2_std:.4f}\n")
    f.write(f"  MAE  : {mae_mean:.4f} ± {mae_std:.4f}\n")
    f.write(f"  RMSE : {rmse_mean:.4f} ± {rmse_std:.4f}\n")
    f.write(f"  MAPE : {mape_mean:.2f}% ± {mape_std:.2f}%\n\n")

    f.write("===== Learning Curve (R²) =====\n")
    f.write("Train sizes (number of samples) and mean R²:\n")
    for n_tr, r2_tr, r2_val in zip(train_sizes, train_scores_mean, val_scores_mean):
        f.write(f"  n_train = {int(n_tr)}: "
                f"Train R² = {r2_tr:.4f}, "
                f"Validation R² = {r2_val:.4f}\n")

print(f"\nAll key results have been saved to '{results_file}'.")
print("\nScript finished successfully.")


