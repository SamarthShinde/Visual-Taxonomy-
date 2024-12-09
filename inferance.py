# model_inference.py

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
import joblib
import pyarrow.parquet as pq

# Set the GPUs to be used (if applicable)
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'  # Uncomment and adjust if needed

# Get the current directory (directory where the script is located)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Paths to test data and resources
TEST_CSV = os.path.join(current_dir, 'data','raw','test.csv')
TEST_IMAGE_DIR = os.path.join(current_dir, 'data', 'images', 'test_images')
LABEL_ENCODERS_PATH = os.path.join(current_dir, 'encoders', 'label_encoders.joblib')
MODEL_PATH = os.path.join(current_dir, 'models', 'efficientnet_model_final.h5')
CATEGORY_ATTRIBUTES_PATH = os.path.join(current_dir, 'logs', 'efficientnetb7_model_checkpoint.h5')

# Output file path for predictions
OUTPUT_CSV = os.path.join(current_dir, 'submission.csv')

# Batch size for inference
BATCH_SIZE = 32  # Adjust based on your system's memory

def main():
    # Load the test data
    test_df = pd.read_csv(TEST_CSV)
    
    # Load category attributes to get the number of attributes per category
    category_attributes = pq.read_table(CATEGORY_ATTRIBUTES_PATH).to_pandas()
    category_attr_count = category_attributes.groupby('Category').size().to_dict()
    
    # Add 'image_path' column to the test DataFrame
    test_df['image_path'] = test_df['id'].apply(lambda x: os.path.join(TEST_IMAGE_DIR, f'{int(x):06d}.jpg'))
    
    # Validate image paths
    def validate_image_paths(df):
        missing_images = []
        for path in df['image_path']:
            if not os.path.exists(path):
                missing_images.append(path)
        if missing_images:
            print(f"Missing images: {len(missing_images)}")
            for path in missing_images[:5]:  # Show first 5 missing paths
                print(f"Missing image: {path}")
            # Remove rows with missing images
            df = df[~df['image_path'].isin(missing_images)].reset_index(drop=True)
        else:
            print("All image paths are valid.")
        return df

    test_df = validate_image_paths(test_df)
    
    # Create the test dataset
    def create_test_dataset(df, batch_size):
        image_paths = df['image_path'].values
        dataset = tf.data.Dataset.from_tensor_slices(image_paths)

        def load_and_preprocess_image(image_path):
            # Read and preprocess the image
            image = tf.io.read_file(image_path)
            image = tf.image.decode_jpeg(image, channels=3)
            image = tf.image.resize(image, [224, 224])
            image = preprocess_input(image)
            return image

        dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

    test_dataset = create_test_dataset(test_df, BATCH_SIZE)
    
    # Load the trained model
    model = load_model(MODEL_PATH)

    # Load label encoders
    label_encoders = joblib.load(LABEL_ENCODERS_PATH)
    attribute_columns = [f'attr_{i}' for i in range(1, 11)]

    # Get predictions
    predictions = model.predict(test_dataset)

    # Decode predictions
    predicted_labels = {}
    for idx, attr in enumerate(attribute_columns):
        pred_indices = np.argmax(predictions[idx], axis=1)
        pred_labels = label_encoders[attr].inverse_transform(pred_indices)
        predicted_labels[attr] = pred_labels

    # Prepare the submission DataFrame
    submission_df = pd.DataFrame({
        'id': test_df['id'],
        'Category': test_df['Category']
    })

    # Add 'len' column representing the number of attributes per category
    submission_df['len'] = submission_df['Category'].map(lambda x: category_attr_count.get(x, 10))

    # Add predicted attributes to the DataFrame
    for attr in attribute_columns:
        submission_df[attr] = predicted_labels[attr]

    # Handle categories with fewer than 10 attributes
    for idx, row in submission_df.iterrows():
        category = row['Category']
        num_attrs = category_attr_count.get(category, 10)
        if num_attrs < 10:
            for i in range(num_attrs + 1, 11):
                submission_df.at[idx, f'attr_{i}'] = 'dummy_value'

    # Reorder columns as required
    submission_columns = ['id', 'Category', 'len'] + attribute_columns
    submission_df = submission_df[submission_columns]

    # Save predictions to CSV
    submission_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Predictions saved to '{OUTPUT_CSV}'.")

if __name__ == "__main__":
    main()