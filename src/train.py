import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report
)
import os

# ── Load data ──────────────────────────────────────────────────────────────────

def load_splits():
    X_train = pd.read_csv("data/X_train.csv")
    X_val   = pd.read_csv("data/X_val.csv")
    y_train = pd.read_csv("data/y_train.csv").squeeze()
    y_val   = pd.read_csv("data/y_val.csv").squeeze()
    return X_train, X_val, y_train, y_val


# ── Train one model run ────────────────────────────────────────────────────────

def train(params: dict, X_train, y_train, X_val, y_val):
    """
    Train one XGBoost model with the given params.
    Logs everything to MLflow automatically.
    Returns the validation AUC score.
    """

    # scale_pos_weight tells XGBoost to pay more attention
    # to the minority class (readmitted patients)
    # Formula: number of negatives / number of positives
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos_weight = neg / pos
    print(f"scale_pos_weight: {scale_pos_weight:.2f}  "
          f"(neg={neg}, pos={pos})")

    with mlflow.start_run():

        # Log all hyperparameters
        mlflow.log_params(params)
        mlflow.log_param("scale_pos_weight", round(scale_pos_weight, 2))

        # Build and train the model
        model = XGBClassifier(
            **params,
            scale_pos_weight=scale_pos_weight,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42
        )
        model.fit(X_train, y_train)

        # Get predictions on validation set
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        y_pred       = model.predict(X_val)

        # Calculate metrics
        auc       = roc_auc_score(y_val, y_pred_proba)
        f1        = f1_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred)
        recall    = recall_score(y_val, y_pred)

        # Log metrics to MLflow
        mlflow.log_metric("val_auc",       round(auc, 4))
        mlflow.log_metric("val_f1",        round(f1, 4))
        mlflow.log_metric("val_precision", round(precision, 4))
        mlflow.log_metric("val_recall",    round(recall, 4))

        # Print results to terminal
        print(f"\nValidation Results:")
        print(f"  AUC:       {auc:.4f}")
        print(f"  F1:        {f1:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"\nClassification Report:")
        print(classification_report(y_val, y_pred,
              target_names=["Not Readmitted", "Readmitted <30d"]))

        # Save the model to MLflow
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name="readmission-model"
        )

        return auc


# ── Run experiments ────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("Loading data splits...")
    X_train, X_val, y_train, y_val = load_splits()
    print(f"Train: {X_train.shape} | Val: {X_val.shape}")

    # Tell MLflow to group all runs under this experiment name
    mlflow.set_experiment("readmission-prediction")

    # ── Experiment 1: Simple baseline ─────────────────────────────────────────
    print("\n--- Run 1: Baseline ---")
    train(
        params={
            "n_estimators":  100,
            "max_depth":     3,
            "learning_rate": 0.1,
        },
        X_train=X_train, y_train=y_train,
        X_val=X_val,     y_val=y_val
    )

    # ── Experiment 2: More trees, slightly deeper ──────────────────────────────
    print("\n--- Run 2: More trees ---")
    train(
        params={
            "n_estimators":  200,
            "max_depth":     4,
            "learning_rate": 0.1,
        },
        X_train=X_train, y_train=y_train,
        X_val=X_val,     y_val=y_val
    )

    # ── Experiment 3: Slower learning rate, more trees ─────────────────────────
    print("\n--- Run 3: Slower learning rate ---")
    train(
        params={
            "n_estimators":  300,
            "max_depth":     4,
            "learning_rate": 0.05,
        },
        X_train=X_train, y_train=y_train,
        X_val=X_val,     y_val=y_val
    )

    print("\n✓ All runs complete.")
    print("Run: mlflow ui")
    print("Then open: http://localhost:5000")