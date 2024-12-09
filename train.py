# model_training.py

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.callbacks import TensorBoard  # Added import
import joblib

# Set the GPUs to be used
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

# Get the current directory (directory where the script is located)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Set the paths to your data and models relative to the current directory
TRAIN_ENCODED_CSV = os.path.join(current_dir, 'encoders', 'train_encoded.csv')
VAL_ENCODED_CSV = os.path.join(current_dir, 'encoders', 'val_encoded.csv')
LABEL_ENCODERS_PATH = os.path.join(current_dir, 'encoders', 'label_encoders.joblib')
IMAGE_DIR = os.path.join(current_dir, 'data', 'images', 'train_images')

# Path to save the model
MODEL_DIR = os.path.join(current_dir, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, 'efficientnet_model_checkpoint.h5')

# Path for TensorBoard logs
LOGS_DIR = os.path.join(current_dir, 'logs')  # Added for TensorBoard
os.makedirs(LOGS_DIR, exist_ok=True)          # Ensure the directory exists

# Batch size and number of epochs
BATCH_SIZE_PER_REPLICA = 128  # Adjust this based on your GPU memory
EPOCHS = 150  # Number of epochs for initial training
FINE_TUNE_EPOCHS = 20  # Additional epochs for fine-tuning

def main():
    # Set up the strategy for multi-GPU training
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    # Adjust the batch size for multiple GPUs
    GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

    # Load the data
    train_df = pd.read_csv(TRAIN_ENCODED_CSV)
    val_df = pd.read_csv(VAL_ENCODED_CSV)

    # Load label encoders
    label_encoders = joblib.load(LABEL_ENCODERS_PATH)

    # Define attribute columns
    attribute_columns = [f'attr_{i}' for i in range(1, 11)]

    # Number of classes per attribute
    num_classes = {attr: len(le.classes_) for attr, le in label_encoders.items()}

    # Prepare labels
    def prepare_labels(df):
        labels = {}
        for attr in attribute_columns:
            labels[attr] = tf.constant(tf.keras.utils.to_categorical(df[attr], num_classes=num_classes[attr]))
        return labels

    train_labels = prepare_labels(train_df)
    val_labels = prepare_labels(val_df)

    # Add 'image_path' column to DataFrames
    train_df['image_path'] = train_df['id'].apply(lambda x: os.path.join(IMAGE_DIR, f'{int(x):06d}.jpg'))
    val_df['image_path'] = val_df['id'].apply(lambda x: os.path.join(IMAGE_DIR, f'{int(x):06d}.jpg'))

    # Validate image paths
    def validate_image_paths(df):
        missing_images = []
        for path in df['image_path']:
            if not os.path.exists(path):
                missing_images.append(path)
        if missing_images:
            print(f"Missing images: {len(missing_images)}")
            for path in missing_images[:5]:  # Print first 5 missing paths
                print(f"Missing image: {path}")
            # Remove rows with missing images
            df = df[~df['image_path'].isin(missing_images)].reset_index(drop=True)
        else:
            print("All image paths are valid.")
        return df

    train_df = validate_image_paths(train_df)
    val_df = validate_image_paths(val_df)

    # Create data generators
    def create_dataset(df, labels, batch_size, is_training=True):
        image_paths = df['image_path'].values
        labels_dict = {attr: labels[attr] for attr in attribute_columns}

        dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels_dict))

        def load_and_preprocess_image(image_path, labels):
            # Read and preprocess the image
            image = tf.io.read_file(image_path)
            image = tf.image.decode_jpeg(image, channels=3)
            image = tf.image.resize(image, [224, 224])
            image = preprocess_input(image)
            return image, labels

        dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
        if is_training:
            dataset = dataset.shuffle(buffer_size=1000)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

    # Create datasets
    train_dataset = create_dataset(train_df, train_labels, GLOBAL_BATCH_SIZE, is_training=True)
    val_dataset = create_dataset(val_df, val_labels, GLOBAL_BATCH_SIZE, is_training=False)

    with strategy.scope():
        # Build the model
        base_model = EfficientNetB0(include_top=False, input_shape=(224, 224, 3), pooling='avg')
        base_model.trainable = False  # Freeze the base model for initial training

        # Input layer
        inputs = Input(shape=(224, 224, 3))

        # Pass inputs through the base model
        x = base_model(inputs)

        # Add dropout
        x = Dropout(0.5)(x)

        # Output layers for each attribute
        outputs = []
        for attr in attribute_columns:
            num_class = num_classes[attr]
            output = Dense(num_class, activation='softmax', name=attr)(x)
            outputs.append(output)

        # Define the model
        model = Model(inputs=inputs, outputs=outputs)

        # Define losses and metrics
        losses = {attr: 'categorical_crossentropy' for attr in attribute_columns}
        metrics = {attr: 'accuracy' for attr in attribute_columns}

        # Compile the model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss=losses,
            metrics=metrics
        )

    # Define callbacks
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        MODEL_PATH,
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=True,
        verbose=1
    )

    reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        verbose=1
    )

    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        verbose=1,
        restore_best_weights=True
    )

    # TensorBoard callback
    tensorboard_callback = TensorBoard(log_dir=LOGS_DIR)  # Added TensorBoard callback

    # Combine callbacks for initial training
    callbacks_initial = [
        checkpoint_callback,
        reduce_lr_callback,
        early_stopping_callback,
        tensorboard_callback  # Added to callbacks
    ]

    # Load the model weights from the checkpoint if it exists
    initial_epoch = 0
    if os.path.exists(MODEL_PATH):
        print("Loading model weights from checkpoint...")
        with strategy.scope():
            model.load_weights(MODEL_PATH)
        # Set initial_epoch to the epoch you stopped at
        # If resuming training, set initial_epoch accordingly
        # initial_epoch = 20  # Uncomment and adjust if resuming
    else:
        print("No checkpoint found. Starting training from scratch.")

    # Verify the dataset
    for images, labels in train_dataset.take(1):
        print(f"Batch images shape: {images.shape}")
        print(f"Labels type: {type(labels)}")
        print(f"Labels keys: {labels.keys()}")
        for attr in labels:
            print(f"Label '{attr}' shape: {labels[attr].shape}")

    # Initial Training Phase
    print("Starting initial training with base model frozen...")
    history_initial = model.fit(
        train_dataset,
        initial_epoch=initial_epoch,
        epochs=EPOCHS,  # Number of epochs for initial training
        validation_data=val_dataset,
        callbacks=callbacks_initial
    )

    # Fine-Tuning Phase
    print("Starting fine-tuning...")
    with strategy.scope():
        # Unfreeze the base model
        base_model.trainable = True

        # Optionally, freeze some layers
        # Fine-tune from this layer onwards
        fine_tune_at = 100  # Adjust based on the number of layers in the model
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False

        # Recompile the model with a lower learning rate
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss=losses,
            metrics=metrics
        )

    # Reset early stopping patience for fine-tuning
    early_stopping_fine_tune = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        verbose=1,
        restore_best_weights=True
    )

    # Combine callbacks for fine-tuning
    callbacks_fine_tune = [
        checkpoint_callback,
        reduce_lr_callback,
        early_stopping_fine_tune,
        tensorboard_callback  # Added to callbacks
    ]

    # Continue training
    total_epochs = EPOCHS + FINE_TUNE_EPOCHS
    history_fine_tune = model.fit(
        train_dataset,
        initial_epoch=EPOCHS,  # Start from where we left off
        epochs=total_epochs,
        validation_data=val_dataset,
        callbacks=callbacks_fine_tune
    )

    # Save the final model
    model.save(os.path.join(MODEL_DIR, 'efficientnet_model_final.h5'))
    print("Model training complete and saved.")

if __name__ == "__main__":
    main()