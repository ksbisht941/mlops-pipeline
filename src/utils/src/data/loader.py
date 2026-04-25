import tensorflow as tf
from tensorflow import keras
import os

def load_datasets(data_dir, img_size, batch_size):
    train_ds = keras.utils.image_dataset_from_directory(
        os.path.join(data_dir, "train"),
        image_size=tuple(img_size),
        batch_size=batch_size,
        label_mode="binary"
    )

    val_ds = keras.utils.image_dataset_from_directory(
        os.path.join(data_dir, "validation"),
        image_size=tuple(img_size),
        batch_size=batch_size,
        label_mode="binary"
    )

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = (
        train_ds
        .map(lambda x, y: (x / 255.0, y))
        .shuffle(1000)
        .prefetch(AUTOTUNE)
    )

    val_ds = (
        val_ds
        .map(lambda x, y: (x / 255.0, y))
        .prefetch(AUTOTUNE)
    )

    return train_ds, val_ds