import mlflow

def setup_mlflow(experiment_name):
    mlflow.set_experiment(experiment_name)