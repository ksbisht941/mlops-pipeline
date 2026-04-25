from src.data.loader import load_datasets
from src.model.model import build_model
from src.model.train import train

def run_training_pipeline(config):
    train_ds, val_ds = load_datasets(
        config["data"]["dir"],
        config["data"]["img_size"],
        config["data"]["batch_size"]
    )

    model = build_model(
        config["data"]["img_size"],
        config["training"]["learning_rate"],
        config["model"]["trainable"]
    )

    model = train(
        model,
        train_ds,
        val_ds,
        config["training"]["epochs"],
        config
    )


    return model