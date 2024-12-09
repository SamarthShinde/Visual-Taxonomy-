# scripts/encode_labels.py

import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib

# ================== CONFIGURATION SECTION ==================

# Path to your training split CSV file
TRAIN_SPLIT_CSV = '/Users/samarthshinde/Downloads/visual-taxonomy/data/processed/train_split.csv'

# Path to save the label encoders
LABEL_ENCODERS_PATH = '/Users/samarthshinde/Downloads/visual-taxonomy/data/processed/label_encoders.joblib'

# ===========================================================

def main():
    # Load the training data
    train_df = pd.read_csv(TRAIN_SPLIT_CSV)
    print(f"Loaded {len(train_df)} training records.")

    # Define attribute columns
    attribute_columns = [f'attr_{i}' for i in range(1, 11)]  # attr_1 to attr_10

    # Create label encoders for each attribute
    label_encoders = {}
    for attr in attribute_columns:
        le = LabelEncoder()
        # Handle missing values
        train_df[attr] = train_df[attr].fillna('nu').replace('', 'nu').astype(str)
        # Fit the label encoder
        le.fit(train_df[attr])
        label_encoders[attr] = le
        print(f"Encoded {attr} with {len(le.classes_)} classes.")

    # Save the label encoders to a file
    os.makedirs(os.path.dirname(LABEL_ENCODERS_PATH), exist_ok=True)
    joblib.dump(label_encoders, LABEL_ENCODERS_PATH)
    print(f"Saved label encoders to {LABEL_ENCODERS_PATH}")

if __name__ == "__main__":
    main()
