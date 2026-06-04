import tensorflow as tf
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense

def build_model(input_shape=(224, 224, 3)):
    # Load EfficientNetB0 pretrained on ImageNet
    base_model = EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape
    )

    # Freeze the base model
    base_model.trainable = False

    # Use functional API with training=False
    # This keeps BatchNorm in inference mode when base is frozen
    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.2)(x)
    output = Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs=inputs, outputs=output)

    return model, base_model


def compile_and_train(model, base_model, train_dataset, val_dataset,
                      train_labels):

    # Calculate class weights to handle imbalance
    classes = np.unique(train_labels)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=train_labels
    )
    class_weight_dict = dict(zip(classes.tolist(), weights.tolist()))
    print(f"Class weights: {class_weight_dict}")

    # Stage 1 — Train top layers only
    print("\n" + "=" * 50)
    print("STAGE 1: Training top layers only")
    print("=" * 50)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )

    callbacks_stage1 = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            patience=5,
            restore_best_weights=True,
            mode="max"
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            verbose=1
        )
    ]

    history1 = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=20,
        class_weight=class_weight_dict,
        callbacks=callbacks_stage1,
        verbose=1
    )

    # Stage 2 — Unfreeze and fine-tune
    print("\n" + "=" * 50)
    print("STAGE 2: Fine-tuning EfficientNet top layers")
    print("=" * 50)

    # Unfreeze base model
    base_model.trainable = True

    # Keep BatchNorm ALWAYS frozen — critical for EfficientNet
    for layer in base_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    # Keep bottom layers frozen too
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    # Recompile with much smaller learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )

    callbacks_stage2 = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            patience=5,
            restore_best_weights=True,
            mode="max"
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath="models/best_model.keras",
            monitor="val_auc",
            save_best_only=True,
            mode="max",
            verbose=1
        )
    ]

    history2 = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=20,
        class_weight=class_weight_dict,
        callbacks=callbacks_stage2,
        verbose=1
    )

    return model, history1, history2