import tensorflow as tf
import numpy as np

from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.models import Model
from sklearn.utils.class_weight import compute_class_weight

def build_model(input_shape=(224, 224, 3), num_classes=10):
    # Load the EfficientNetB0 pretrained on ImageNet
    base_model = EfficientNetB0(
        weights='imagenet', 
        include_top=False, # Means removing the original 1000-class output layer
        input_shape=input_shape
        )
    
    # Freeze the base model (So not touch pretrained weights yet)
    base_model.trainable = False
    
    # Build custom top layers on top of EfficientNetB0
    x = base_model.output # Get the output of the base model
    x = GlobalAveragePooling2D()(x) # Summarise feature maps into a single vector
    x = Dropout(0.2)(x) # Randomly switch 20% of the neurons to prevent overfitting
    x = Dense(128, activation='relu')(x) # Add a fully connected layer with 128 neurons and ReLU activation
    x = Dropout(0.2)(x) # Another dropout layer to further reduce overfitting
    output = Dense(1, activation='sigmoid')(x) # Final output layer with sigmoid activation for binary classification
    
    # Combine base model and custom top into a new model
    model = Model(inputs=base_model.input, outputs=output)
    
    return model, base_model

def compile_and_train(model, base_model, X_train, y_train, X_val, y_val):

    # Calculate class weights to handle class imbalance
    classess = np.unique(y_train)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classess,
        y=y_train
    )
    class_weight_dict = dict(zip(classes, weights))
    print(f"Class weights: {class_weight_dict}")
    
    # Stage 1 - Train only the top layers
    print("\n" + "=" * 50)
    print("STAGE 1: Training the top layers only")
    print("=" * 50)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3)
        loss="binary_crossentropy"
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
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=32,
        class_weight=class_weight_dict,
        callbacks=callbacks_stage1,
        verbose=1
    )

    # Stage 2 - Unfreeze and fine tune
    print("\n" + "=" * 50)
    print("STAGE 2: Fine-tuning top layers of EfficientNet")
    print("=" * 50)

    # Unfreeze the top 20 layers
    base_model.trainable = True
    for layer in base_model.layers[:-20]:
        layer.trainable = False
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5)
        loss="binary_crossentropy"
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
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=32,
        class_weight=class_weight_dict,
        callbacks=callbacks_stage1,
        verbose=1
    )

    return model, history1, history2
