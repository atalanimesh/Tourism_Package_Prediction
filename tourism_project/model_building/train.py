"""
Model Training with Experiment Tracking (Production)
----------------------------------------------------
Loads the prepared splits produced by the previous workflow job, builds a
preprocessing + XGBoost pipeline, tunes it with GridSearchCV while logging every
trial to MLflow, evaluates the best model, and saves it into the deployment
folder so the workflow can commit it back to the repository. Streamlit Community
Cloud then serves that committed model.

MLflow uses a local file store (./mlruns), which is robust inside GitHub Actions
and needs no running server. The live MLflow UI screenshot for the report comes
from the development run in the notebook (exposed via ngrok).
"""
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report,
)
import xgboost as xgb
import joblib
import os

# Allow MLflow's local file store across versions (newer MLflow gates it behind
# this flag; with the pinned mlflow==3.0.1 this is simply a no-op).
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
import mlflow

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("tourism-package-prediction")

# Xtrain/Xtest/ytrain/ytest are downloaded from the previous job's artifact
Xtrain = pd.read_csv("Xtrain.csv")
Xtest = pd.read_csv("Xtest.csv")
ytrain = pd.read_csv("ytrain.csv").squeeze("columns")
ytest = pd.read_csv("ytest.csv").squeeze("columns")

numeric_features = [
    "Age", "CityTier", "DurationOfPitch", "NumberOfPersonVisiting",
    "NumberOfFollowups", "PreferredPropertyStar", "NumberOfTrips",
    "Passport", "PitchSatisfactionScore", "OwnCar",
    "NumberOfChildrenVisiting", "MonthlyIncome",
]
categorical_features = [
    "TypeofContact", "Occupation", "Gender", "ProductPitched",
    "MaritalStatus", "Designation",
]

# Handle class imbalance (~1 buyer for every ~4 non-buyers)
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]

preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore"), categorical_features),
)

xgb_model = xgb.XGBClassifier(
    scale_pos_weight=class_weight,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
)

pipeline = make_pipeline(preprocessor, xgb_model)

param_grid = {
    "xgbclassifier__n_estimators": [100, 200],
    "xgbclassifier__max_depth": [3, 5],
    "xgbclassifier__learning_rate": [0.05, 0.1],
    "xgbclassifier__subsample": [0.8, 1.0],
    "xgbclassifier__colsample_bytree": [0.8],
}

grid = GridSearchCV(pipeline, param_grid, cv=5, scoring="f1", n_jobs=-1, refit=True)

with mlflow.start_run(run_name="xgboost-gridsearch"):
    grid.fit(Xtrain, ytrain)

    # Log every parameter combination tried as a nested run
    results = grid.cv_results_
    for i in range(len(results["params"])):
        with mlflow.start_run(nested=True):
            mlflow.log_params(results["params"][i])
            mlflow.log_metric("mean_cv_f1", results["mean_test_score"][i])

    # Log the best configuration on the parent run
    mlflow.log_params(grid.best_params_)
    mlflow.log_metric("best_cv_f1", grid.best_score_)

    best_model = grid.best_estimator_
    pred = best_model.predict(Xtest)
    proba = best_model.predict_proba(Xtest)[:, 1]

    metrics = {
        "test_accuracy": accuracy_score(ytest, pred),
        "test_precision": precision_score(ytest, pred),
        "test_recall": recall_score(ytest, pred),
        "test_f1": f1_score(ytest, pred),
        "test_roc_auc": roc_auc_score(ytest, proba),
    }
    mlflow.log_metrics(metrics)

    print("Best parameters:", grid.best_params_)
    print("Best CV F1:", round(grid.best_score_, 4))
    print("\nTest metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    print("\nClassification report:\n", classification_report(ytest, pred, digits=3))

    # Save the best model into the deployment folder for the workflow to commit
    os.makedirs("tourism_project/deployment", exist_ok=True)
    model_path = "tourism_project/deployment/best_tourism_package_model_v1.joblib"
    joblib.dump(best_model, model_path)
    mlflow.log_artifact(model_path)
    print(f"\nSaved best model to {model_path}")
