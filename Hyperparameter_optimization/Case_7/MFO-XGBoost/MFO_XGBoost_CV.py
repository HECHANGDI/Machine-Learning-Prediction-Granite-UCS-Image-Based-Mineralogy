# -*- coding: utf-8 -*-
"""
This script:
- Loads the dataset
- Uses SHAP to select the top 15 most important input features
- Optimizes XGBoost hyperparameters using Moth-Flame Optimisation (MFO)
- Trains the final XGBoost model with SHAP-selected features
- Evaluates single train/test performance
- Performs repeated k-fold cross-validation
- Computes and plots a learning curve (training vs validation R²)
- Saves metrics to a text file
- Plots predicted vs actual UCS for train and test sets
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    train_test_split,
    RepeatedKFold,
    KFold,
    learning_curve
)
from joblib import Parallel, delayed
import shap

# ------------------------------------------------------------
# 0. Global configuration
# ------------------------------------------------------------
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
plt.rc('font', family='Times New Roman', size=7.5)

# ------------------------------------------------------------
# 1. Load dataset
# ------------------------------------------------------------
data = pd.read_excel("training_data_case_7.xlsx", header=0)

# Target (UCS, converted to MPa)
y = data.iloc[:, -1] / 1e6

# All candidate input features (full feature set)
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

# ------------------------------------------------------------
# 2. SHAP-based feature selection (top 15)
# ------------------------------------------------------------
print("\n===== Stage 1: SHAP-based Feature Selection =====")

# Base XGBoost model for computing SHAP importance
base_model = XGBRegressor(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_SEED,
    n_jobs=-1
)

# Fit base model on the full feature set
base_model.fit(X_full, y)

# Compute SHAP values
explainer = shap.TreeExplainer(base_model)
shap_values = explainer.shap_values(X_full)

# Mean absolute SHAP value per feature
mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

shap_df = pd.DataFrame({
    "Feature": X_full.columns,
    "MeanAbsSHAP": mean_abs_shap
})

# Rank features by importance
shap_df = shap_df.sort_values("MeanAbsSHAP", ascending=False).reset_index(drop=True)

print("\nSHAP-based feature importance (top 20):")
print(shap_df.head(20))

# Select the top 15 most important features
top_k = 15
selected_features = shap_df["Feature"].head(top_k).tolist()

print(f"\nTop {top_k} SHAP-based features to be used as model inputs:")
for f in selected_features:
    print(" -", f)

# Reduced input matrix using only SHAP-selected features
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
# 3. Train/test split using SHAP-selected features
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_SEED
)

# Precompute the test data for faster evaluation
y_test_np = np.array(y_test)

# ------------------------------------------------------------
# 4. Objective function for MFO (uses SHAP-selected features)
# ------------------------------------------------------------
def mfo_objective(params):
    """Objective function for MFO: minimise 1 - R² on the test set."""
    params_dict = {
        "n_estimators": int(params[0]),
        "max_depth": int(params[1]),
        "learning_rate": params[2],
        "subsample": params[3],
        "colsample_bytree": params[4],
        "gamma": params[5],
        "alpha": params[6],
        "lambda": params[7],
    }
    model = XGBRegressor(
        **params_dict,
        random_state=RANDOM_SEED,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return 1 - r2_score(y_test_np, y_pred)  # Minimize 1 - R²

# ------------------------------------------------------------
# 5. Parameter bounds for MFO
# ------------------------------------------------------------
param_bounds = np.array([
    (50, 500),      # n_estimators
    (3, 12),        # max_depth
    (0.01, 0.3),    # learning_rate
    (0.6, 1.0),     # subsample
    (0.6, 1.0),     # colsample_bytree
    (0.0, 20.0),    # gamma
    (0.0, 20.0),    # alpha
    (0.0, 20.0),    # lambda
])

# ------------------------------------------------------------
# 6. Moth-Flame Optimisation (MFO) implementation
# ------------------------------------------------------------
def mfo_optimize(objective, bounds, iterations=8000, population_size=20):
    dim = bounds.shape[0]
    population = np.random.uniform(
        bounds[:, 0], bounds[:, 1],
        size=(population_size, dim)
    )
    flames = population.copy()
    best_score = float("inf")
    best_solution = None

    b = 1.0  # constant for logarithmic spiral

    for t in range(1, iterations + 1):
        # Evaluate all individuals
        scores = Parallel(n_jobs=-1)(
            delayed(objective)(ind) for ind in population
        )
        scores = np.array(scores)

        # Sort population by fitness (ascending)
        sorted_indices = np.argsort(scores)
        population = population[sorted_indices]
        scores = scores[sorted_indices]
        flames = population.copy()

        # Update global best
        if scores[0] < best_score:
            best_score = scores[0]
            best_solution = population[0].copy()

        # Update the position of moths around flames (simplified version)
        t_norm = t / iterations  # in (0, 1]
        for i in range(population_size):
            for j in range(dim):
                distance_to_flame = abs(flames[i, j] - population[i, j])
                population[i, j] = (
                    distance_to_flame
                    * np.exp(b * t_norm)
                    * np.cos(2 * np.pi * t_norm)
                    + flames[i, j]
                )

        # Ensure solutions stay within bounds
        population = np.clip(population, bounds[:, 0], bounds[:, 1])

        # Print progress
        print(f"Iteration {t}: Best Score = {best_score:.6f}")

    return best_solution

# ------------------------------------------------------------
# 7. Run MFO optimisation
# ------------------------------------------------------------
best_params_raw = mfo_optimize(
    mfo_objective,
    param_bounds,
    iterations=8000,
    population_size=20
)

optimized_params = {
    "n_estimators": int(best_params_raw[0]),
    "max_depth": int(best_params_raw[1]),
    "learning_rate": best_params_raw[2],
    "subsample": best_params_raw[3],
    "colsample_bytree": best_params_raw[4],
    "gamma": best_params_raw[5],
    "alpha": best_params_raw[6],
    "lambda": best_params_raw[7],
}

print("\nOptimised Parameters (MFO + top-15 SHAP features):")
print(optimized_params)

# ------------------------------------------------------------
# 8. Train the final XGBoost model with optimised parameters
# ------------------------------------------------------------
model = XGBRegressor(
    **optimized_params,
    random_state=RANDOM_SEED,
    n_jobs=-1
)
model.fit(X_train, y_train)

# Predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# ------------------------------------------------------------
# 9. Evaluation metrics (single train/test split)
# ------------------------------------------------------------
def evaluate_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return r2, mae, mse, rmse, mape

# Training metrics
r2_train, mae_train, mse_train, rmse_train, mape_train = evaluate_metrics(
    y_train, y_train_pred
)

# Testing metrics
r2_test, mae_test, mse_test, rmse_test, mape_test = evaluate_metrics(
    y_test, y_test_pred
)

print("\n===== Single train/test split performance =====")
print("Training Metrics:")
print(f"  R²: {r2_train:.4f}, MAE: {mae_train:.4f}, "
      f"MSE: {mse_train:.4f}, RMSE: {rmse_train:.4f}, "
      f"MAPE: {mape_train:.2f}%")
print("Testing Metrics:")
print(f"  R²: {r2_test:.4f}, MAE: {mae_test:.4f}, "
      f"MSE: {mse_test:.4f}, RMSE: {rmse_test:.4f}, "
      f"MAPE: {mape_test:.2f}%")

# Simple overfitting check
if r2_train - r2_test > 0.1:
    overfit_msg = ("Warning: Model might be overfitting. "
                   "Consider tuning regularization parameters further or adding more data.")
else:
    overfit_msg = "Model shows good generalization."

print(overfit_msg)

# ------------------------------------------------------------
# 10. Repeated k-fold cross-validation
# ------------------------------------------------------------
print("\n===== Repeated k-fold Cross-Validation with MFO-XGBoost =====")

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

    model_cv = XGBRegressor(
        **optimized_params,
        random_state=RANDOM_SEED,
        n_jobs=-1
    )
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

# Boxplot of validation R² across folds
plt.figure(figsize=(6/2.54, 6/2.54))
plt.boxplot(r2_scores, vert=True)
plt.ylabel("R² (validation)")
plt.title(
    f"Repeated {n_splits}-fold CV\n"
    f"Distribution of validation R² (n = {n_splits * n_repeats} folds)"
)
plt.tight_layout()
plt.savefig("xgb_distribution_MFO_SHAP_top15.jpg", dpi=1000)
plt.show()

# ------------------------------------------------------------
# 11. Learning curve (training vs validation R²)
# ------------------------------------------------------------
print("\n===== Learning curve (Training vs Validation R²) =====")

cv_for_lc = KFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_SEED
)

estimator_lc = XGBRegressor(
    **optimized_params,
    random_state=RANDOM_SEED,
    n_jobs=-1
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
    train_scores_mean - train_scores_std,
    train_scores_mean + train_scores_std,
    alpha=0.2
)
plt.plot(
    train_sizes, train_scores_mean,
    marker="o", markersize=4,
    label="Training R²"
)

for x, y in zip(train_sizes, val_scores_mean):
    plt.annotate(
        f"{y:.2f}",
        (x, y),
        textcoords="offset points",
        xytext=(0, -8),  
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
    train_sizes, val_scores_mean,
    marker="s", markersize=4, markerfacecolor="none",
    label="Validation R²"
)

plt.xlabel("Number of training samples")
plt.ylabel(r"R²")
plt.tight_layout()
plt.savefig("xgb_gwo_learning_curve_r2.jpg", dpi=1000)
plt.show()

print("\nLearning curve finished.")

# ------------------------------------------------------------
# 12. Save metrics and messages to a text file
# ------------------------------------------------------------
output_file = "model_metrics_output_MFO_SHAP_top15_full.txt"
with open(output_file, "w") as f:
    f.write("Optimised Parameters (MFO + top-15 SHAP features):\n")
    f.write(str(optimized_params) + "\n\n")

    f.write("Single train/test split metrics:\n")
    f.write(f"Training:\n")
    f.write(
        f"  R²: {r2_train:.4f}, MAE: {mae_train:.4f}, "
        f"MSE: {mse_train:.4f}, RMSE: {rmse_train:.4f}, "
        f"MAPE: {mape_train:.2f}%\n"
    )
    f.write("Testing:\n")
    f.write(
        f"  R²: {r2_test:.4f}, MAE: {mae_test:.4f}, "
        f"MSE: {mse_test:.4f}, RMSE: {rmse_test:.4f}, "
        f"MAPE: {mape_test:.2f}%\n\n"
    )
    f.write(overfit_msg + "\n\n")

    f.write("Repeated k-fold CV metrics "
            f"(n_splits={n_splits}, n_repeats={n_repeats}):\n")
    f.write(
        f"  R²   : {r2_mean:.4f} ± {r2_std:.4f}\n"
        f"  MAE  : {mae_mean:.4f} ± {mae_std:.4f}\n"
        f"  RMSE : {rmse_mean:.4f} ± {rmse_std:.4f}\n"
        f"  MAPE : {mape_mean:.2f}% ± {mape_std:.2f}%\n"
    )

print(f"\nMetrics and messages saved to {output_file}")

# ------------------------------------------------------------
# 13. Visualisation: Predicted vs Actual (Train/Test)
# ------------------------------------------------------------
plt.figure(figsize=(12/2.54, 6/2.54))

plt.subplot(1, 2, 1)
plt.scatter(y_train, y_train_pred, alpha=0.7, color="blue", label="Training data")
plt.plot(
    [y_train.min(), y_train.max()],
    [y_train.min(), y_train.max()],
    "r--", lw=1, label="Ideal fit"
)
plt.xlabel("Actual UCS (MPa)")
plt.ylabel("Predicted UCS (MPa)")
plt.title(f"Train (R² = {r2_train:.4f})")
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(y_test, y_test_pred, alpha=0.7, color="green", label="Testing data")
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    "r--", lw=1, label="Ideal fit"
)
plt.xlabel("Actual UCS (MPa)")
plt.ylabel("Predicted UCS (MPa)")
plt.title(f"Test (R² = {r2_test:.4f})")
plt.legend()

plt.tight_layout()
plt.savefig("predicted_vs_actual_fit4_SHAP_top15_GWO.jpg", dpi=1000, format="jpg")
plt.show()
