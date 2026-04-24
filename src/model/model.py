from tensorflow import keras
from tensorflow.keras import layers

def build_model(img_size, lr, trainable=False):
    base_model = keras.applications.MobileNetV2(
        input_shape=tuple(img_size) + (3,),
        include_top=False,
        weights="imagenet"
    )

    base_model.trainable = trainable

    inputs = keras.Input(shape=tuple(img_size) + (3,))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")]
    )

    return model