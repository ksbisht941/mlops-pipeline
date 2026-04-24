import mlflow
import mlflow.tensorflow

def train(model, train_ds, val_ds, epochs):
    with mlflow.start_run():

        callbacks = [
            keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(patience=3),
        ]

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=callbacks
        )

        # Log metrics
        for epoch in range(len(history.history["loss"])):
            mlflow.log_metric("loss", history.history["loss"][epoch], step=epoch)
            mlflow.log_metric("val_loss", history.history["val_loss"][epoch], step=epoch)
            mlflow.log_metric("accuracy", history.history["accuracy"][epoch], step=epoch)

        # Log model
        mlflow.tensorflow.log_model(model, "model")

        return model