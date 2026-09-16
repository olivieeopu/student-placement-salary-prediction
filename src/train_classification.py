import pandas as pd
import numpy as np
import optuna
import mlflow
import mlflow.sklearn
import os
import joblib
import random

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, recall_score

from optuna.samplers import TPESampler

from preprocessing import FeatureEngineer, FeatureDropper, create_preprocessor
from data.data_ingestion import load_data, split_classification


RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

sampler = TPESampler(seed=RANDOM_STATE)

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Student Placement Classification")

print("TRACKING URI:", mlflow.get_tracking_uri())
print("WORKDIR:", os.getcwd())


df = load_data("fitur_target/A.csv", "fitur_target/A_targets.csv")

df['placement_status'] = df['placement_status'].str.strip()
df['placement_status'] = df['placement_status'].map({
    'Not Placed': 0,
    'Placed': 1
})

# SPLIT
X_class, y_class = split_classification(df)

X_train, X_test, y_train, y_test = train_test_split(
    X_class, y_class, test_size=0.2, random_state=RANDOM_STATE, stratify=y_class
)

# CONFIG CV
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

drop_cols_clf = [
    'sleep_hours','attendance_percentage','study_hours_per_day','stress_level',
    'gender','part_time_job','family_income_level','city_tier','extracurricular_involvement', 
    'tenth_percentage', 'communication_skill_rating'
]


# CLASS WEIGHT
classes = np.unique(y_train)
weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
class_weight_dict = dict(zip(classes, weights))

neg, pos = np.bincount(y_train)
scale_pos_weight = neg / pos


# OPTUNA OBJECTIVES
def objective_lr(trial):
    C = trial.suggest_float("C", 0.01, 10, log=True)
    cw = trial.suggest_categorical("class_weight", [None, "balanced", class_weight_dict])

    model = LogisticRegression(C=C, max_iter=1000, class_weight=cw)

    pipe = Pipeline([
        ('feature_engineering', FeatureEngineer()),
        ('dropper', FeatureDropper(drop_cols_clf)),
        ('preprocessing', create_preprocessor()),
        ('model', model)
    ])

    return cross_val_score(pipe, X_train, y_train, cv=skf, scoring='f1').mean()


def objective_rf(trial):
    n_estimators = trial.suggest_int("n_estimators", 100, 300)
    max_depth = trial.suggest_categorical("max_depth", [None, 5, 10])
    cw = trial.suggest_categorical("class_weight", [None, "balanced", class_weight_dict])

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight=cw,
        random_state=RANDOM_STATE
    )

    pipe = Pipeline([
        ('feature_engineering', FeatureEngineer()),
        ('dropper', FeatureDropper(drop_cols_clf)),
        ('preprocessing', create_preprocessor()),
        ('model', model)
    ])

    return cross_val_score(pipe, X_train, y_train, cv=skf, scoring='f1').mean()


def objective_xgb(trial):
    n_estimators = trial.suggest_int("n_estimators", 100, 300)
    max_depth = trial.suggest_int("max_depth", 3, 6)
    lr = trial.suggest_float("learning_rate", 0.01, 0.2)
    spw = trial.suggest_categorical("scale_pos_weight", [1, scale_pos_weight])

    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=lr,
        scale_pos_weight=spw,
        eval_metric='logloss',
        random_state=RANDOM_STATE
    )

    pipe = Pipeline([
        ('feature_engineering', FeatureEngineer()),
        ('dropper', FeatureDropper(drop_cols_clf)),
        ('preprocessing', create_preprocessor()),
        ('model', model)
    ])

    return cross_val_score(pipe, X_train, y_train, cv=skf, scoring='f1').mean()


# RUN MLFLOW
with mlflow.start_run(run_name="optuna_classification"):

    # LOGISTIC
    with mlflow.start_run(run_name="LogReg", nested=True):
        study_lr = optuna.create_study(direction="maximize", sampler=sampler)
        study_lr.optimize(objective_lr, n_trials=20)

        mlflow.log_params(study_lr.best_params)
        mlflow.log_metric("best_f1", study_lr.best_value)


    # RANDOM FOREST
    with mlflow.start_run(run_name="RandomForest", nested=True):
        study_rf = optuna.create_study(direction="maximize", sampler=sampler)
        study_rf.optimize(objective_rf, n_trials=20)

        mlflow.log_params(study_rf.best_params)
        mlflow.log_metric("best_f1", study_rf.best_value)


    # XGBOOST
    with mlflow.start_run(run_name="XGBoost", nested=True):
        study_xgb = optuna.create_study(direction="maximize", sampler=sampler)
        study_xgb.optimize(objective_xgb, n_trials=20)

        mlflow.log_params(study_xgb.best_params)
        mlflow.log_metric("best_f1", study_xgb.best_value)


    # AUTO BEST MODEL SELECTION
    models_results = {
        "LogisticRegression": (study_lr.best_value, study_lr.best_params),
        "RandomForest": (study_rf.best_value, study_rf.best_params),
        "XGBoost": (study_xgb.best_value, study_xgb.best_params),
    }

    best_model_name = max(models_results, key=lambda x: models_results[x][0])
    best_score, best_params = models_results[best_model_name]

    print(f"Best Model: {best_model_name} | F1: {best_score}")


    # FINAL MODEL
    with mlflow.start_run(run_name="final_model", nested=True):

        if best_model_name == "LogisticRegression":
            best_model = LogisticRegression(
                **best_params,
                random_state=RANDOM_STATE
            )

        elif best_model_name == "RandomForest":
            best_model = RandomForestClassifier(
                **best_params,
                random_state=RANDOM_STATE
            )

        elif best_model_name == "XGBoost":
            best_model = xgb.XGBClassifier(
                **best_params,
                eval_metric='logloss',
                random_state=RANDOM_STATE
            )

        pipe = Pipeline([
            ('feature_engineering', FeatureEngineer()),
            ('dropper', FeatureDropper(drop_cols_clf)),
            ('preprocessing', create_preprocessor()),
            ('model', best_model)
        ])

        pipe.fit(X_train, y_train)

        joblib.dump(pipe, "best_model_classification.pkl")

        y_pred = pipe.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

        cv_f1 = cross_val_score(pipe, X_train, y_train, cv=skf, scoring='f1').mean()

        mlflow.log_params(best_params)
        mlflow.log_param("model_type", best_model_name)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("cv_f1", cv_f1)

        mlflow.sklearn.log_model(pipe, "model")


print("\n=== DONE & LOGGED TO MLFLOW ===")