import yaml
from src.utils.logger import setup_mlflow
from src.pipelines.training_pipeline import run_training_pipeline

# with open("configs/config.yaml") as f:
#     config = yaml.safe_load(f)

with open("params.yaml") as f:
    config = yaml.safe_load(f)

setup_mlflow(config["mlflow"]["experiment_name"])

run_training_pipeline(config)