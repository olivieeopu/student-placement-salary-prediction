import pandas as pd
import optuna
import mlflow
import mlflow.sklearn
import joblib
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.linear_model import LinearRegression, Ridge, Lasso

from optuna.samplers import TPESampler

from src.preprocessing import FeatureEngineer, FeatureDropper, create_preprocessor
from src.data.data_ingestion import load_data, split_regression


RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Student Regression")


df = load_data("fitur_target/A.csv", "fitur_target/A_targets.csv")

X_reg, y_reg = split_regression(df)

# TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=RANDOM_STATE
)

# DROP COLUMNS
drop_cols_reg = [
    'sleep_hours',
    'attendance_percentage',
    'stress_level',
    'gender',
    'part_time_job',
    'city_tier',
    'extracurricular_involvement',
    'family_income_level',
    'tenth_percentage',
    'communication_skill_rating'
]

kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


# EVALUATION FUNCTION
def evaluate_model(model):
    pipe = Pipeline([
        ('feature_engineering', FeatureEngineer()),
        ('dropper', FeatureDropper(drop_cols_reg)),
        ('preprocessing', create_preprocessor()),
        ('model', model)
    ])

    scores = cross_validate(
        pipe,
        X_train,
        y_train,
        cv=kf,
        scoring={
            'rmse': 'neg_root_mean_squared_error',
            'r2': 'r2'
        }
    )

    rmse = -scores['test_rmse'].mean()
    r2 = scores['test_r2'].mean()

    return rmse, r2


# OPTUNA OBJECTIVES
def objective_lr(trial):
    fit_intercept = trial.suggest_categorical("fit_intercept", [True, False])
    rmse, _ = evaluate_model(LinearRegression(fit_intercept=fit_intercept))
    return rmse


def objective_ridge(trial):
    alpha = trial.suggest_float("alpha", 1e-3, 100, log=True)
    rmse, _ = evaluate_model(Ridge(alpha=alpha))
    return rmse


def objective_lasso(trial):
    alpha = trial.suggest_float("alpha", 1e-4, 1, log=True)
    rmse, _ = evaluate_model(Lasso(alpha=alpha, max_iter=5000))
    return rmse


sampler = TPESampler(seed=RANDOM_STATE)


with mlflow.start_run(run_name="optuna_regression"):

    print("\n=== OPTUNA REGRESSION ===")


    # LINEAR REGRESSION
    with mlflow.start_run(run_name="LinearRegression", nested=True):
        study_lr = optuna.create_study(direction="minimize", sampler=sampler)
        study_lr.optimize(objective_lr, n_trials=20)

        model_lr = LinearRegression(**study_lr.best_params)
        rmse_lr, r2_lr = evaluate_model(model_lr)

        pipe_lr = Pipeline([
            ('feature_engineering', FeatureEngineer()),
            ('dropper', FeatureDropper(drop_cols_reg)),
            ('preprocessing', create_preprocessor()),
            ('model', model_lr)
        ])

        pipe_lr.fit(X_train, y_train)

        mlflow.log_params(study_lr.best_params)
        mlflow.log_metric("best_rmse", rmse_lr)
        mlflow.log_metric("best_r2", r2_lr)
        mlflow.sklearn.log_model(pipe_lr, "model")


    # RIDGE
    with mlflow.start_run(run_name="Ridge", nested=True):
        study_ridge = optuna.create_study(direction="minimize", sampler=sampler)
        study_ridge.optimize(objective_ridge, n_trials=20)

        model_ridge = Ridge(**study_ridge.best_params)
        rmse_ridge, r2_ridge = evaluate_model(model_ridge)

        pipe_ridge = Pipeline([
            ('feature_engineering', FeatureEngineer()),
            ('dropper', FeatureDropper(drop_cols_reg)),
            ('preprocessing', create_preprocessor()),
            ('model', model_ridge)
        ])

        pipe_ridge.fit(X_train, y_train)

        mlflow.log_params(study_ridge.best_params)
        mlflow.log_metric("best_rmse", rmse_ridge)
        mlflow.log_metric("best_r2", r2_ridge)
        mlflow.sklearn.log_model(pipe_ridge, "model")


    # LASSO
    with mlflow.start_run(run_name="Lasso", nested=True):
        study_lasso = optuna.create_study(direction="minimize", sampler=sampler)
        study_lasso.optimize(objective_lasso, n_trials=20)

        model_lasso = Lasso(**study_lasso.best_params, max_iter=5000)
        rmse_lasso, r2_lasso = evaluate_model(model_lasso)

        pipe_lasso = Pipeline([
            ('feature_engineering', FeatureEngineer()),
            ('dropper', FeatureDropper(drop_cols_reg)),
            ('preprocessing', create_preprocessor()),
            ('model', model_lasso)
        ])

        pipe_lasso.fit(X_train, y_train)

        mlflow.log_params(study_lasso.best_params)
        mlflow.log_metric("best_rmse", rmse_lasso)
        mlflow.log_metric("best_r2", r2_lasso)
        mlflow.sklearn.log_model(pipe_lasso, "model")

    # AUTO BEST MODEL SELECTION
    models_results = {
        "LinearRegression": (rmse_lr, study_lr.best_params),
        "Ridge": (rmse_ridge, study_ridge.best_params),
        "Lasso": (rmse_lasso, study_lasso.best_params),
    }

    best_model_name = min(models_results, key=lambda x: models_results[x][0])
    best_rmse, best_params = models_results[best_model_name]

    print(f"Best Model: {best_model_name} | RMSE: {best_rmse}")


# FINAL MODEL (AUTO BEST)
with mlflow.start_run(run_name="final_model", nested=True):

    if best_model_name == "LinearRegression":
        best_model = LinearRegression(**best_params)

    elif best_model_name == "Ridge":
        best_model = Ridge(**best_params)

    elif best_model_name == "Lasso":
        best_model = Lasso(**best_params, max_iter=5000)

    final_pipe = Pipeline([
        ('feature_engineering', FeatureEngineer()),
        ('dropper', FeatureDropper(drop_cols_reg)),
        ('preprocessing', create_preprocessor()),
        ('model', best_model)
    ])

    final_pipe.fit(X_train, y_train)

    joblib.dump(final_pipe, "best_model_regression.pkl")

    mlflow.log_params(best_params)
    mlflow.log_param("model_type", best_model_name)

    y_pred = final_pipe.predict(X_test)
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))

    mlflow.log_metric("rmse_test", rmse)

    mlflow.sklearn.log_model(final_pipe, "best_model")


print("\n=== BEST REGRESSION ===")
print("LR:", study_lr.best_params)
print("Ridge:", study_ridge.best_params)
print("Lasso:", study_lasso.best_params)

print("\n=== DONE & LOGGED TO MLFLOW ===")