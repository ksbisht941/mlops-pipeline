import mlflow
import mlflow.tensorflow
import subprocess
import json
from tensorflow import keras


# =========================
# METADATA HELPERS
# =========================
def get_git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"]
        ).decode().strip()
    except:
        return "unknown"


def get_dvc_hash():
    try:
        return subprocess.check_output(
            ["dvc", "status", "--json"]
        ).decode()
    except:
        return "unknown"


# =========================
# TRAIN FUNCTION
# =========================
def train(model, train_ds, val_ds, epochs, config):
    with mlflow.start_run() as run:

        # -------------------------
        # LOG PARAMS (CRITICAL)
        # -------------------------
        mlflow.log_params({
            "epochs": epochs,
            "batch_size": config["data"]["batch_size"],
            "img_size": config["data"]["img_size"],
            "learning_rate": config["training"]["lr"],
            "backbone": config["model"]["backbone"],
            "trainable": config["model"]["trainable"],
        })

        # -------------------------
        # LINK EXPERIMENT → CODE + DATA
        # -------------------------
        mlflow.log_param("git_commit", get_git_commit())
        mlflow.log_text(get_dvc_hash(), "dvc_status.json")

        # -------------------------
        # CALLBACKS
        # -------------------------
        callbacks = [
            keras.callbacks.EarlyStopping(
                patience=5, restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(patience=3),
        ]

        # -------------------------
        # TRAIN
        # -------------------------
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=callbacks
        )

        # -------------------------
        # LOG METRICS (ALL, NOT PARTIAL)
        # -------------------------
        for metric_name, values in history.history.items():
            for epoch, value in enumerate(values):
                mlflow.log_metric(metric_name, value, step=epoch)

        # -------------------------
        # SAVE MODEL LOCALLY (DVC TRACKS THIS)
        # -------------------------
        model_path = "models/model.keras"
        model.save(model_path)

        mlflow.log_artifact(model_path)

        # -------------------------
        # LOG MODEL TO MLFLOW
        # -------------------------
        mlflow.tensorflow.log_model(
            model,
            artifact_path="model"
        )

        # -------------------------
        # REGISTER MODEL
        # -------------------------
        model_name = "cat-dog-classifier"

        result = mlflow.register_model(
            f"runs:/{run.info.run_id}/model",
            model_name
        )

        client = mlflow.tracking.MlflowClient()

        # Move to staging automatically
        client.transition_model_version_stage(
            name=model_name,
            version=result.version,
            stage="Staging"
        )

        print(f"Model registered: {model_name}, version: {result.version}")

        return model