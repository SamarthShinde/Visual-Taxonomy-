import pandas as pd
from sklearn.model_selection import train_test_split

# ================== CONFIGURATION SECTION ==================

# Specify the path to your train.csv file
TRAIN_CSV_PATH = '/Users/samarthshinde/Downloads/visual-taxonomy/data/raw/train.csv'  # <-- Replace this with the actual path to your train.csv

# Specify where you want to save the training split CSV
PROCESSED_TRAIN_CSV = '/Users/samarthshinde/Downloads/visual-taxonomy/data/processed/train_split.csv'  # <-- Replace this with the desired save path

# Specify where you want to save the validation split CSV
PROCESSED_VALID_CSV = '/Users/samarthshinde/Downloads/visual-taxonomy/data/processed/valid_split.csv'  # <-- Replace this with the desired save path

# Set the validation split ratio (e.g., 0.2 for 20% validation data)
VALIDATION_SPLIT_RATIO = 0.2

# ===========================================================

def split_train_validation(train_csv_path, processed_train_csv, processed_valid_csv, split_ratio):
    # Load the data
    print(f"Loading training data from {train_csv_path}...")
    train_df = pd.read_csv(train_csv_path)
    print(f"Total records in training data: {len(train_df)}")

    # Split the data
    print(f"Splitting data with a validation ratio of {split_ratio}...")
    train_split, valid_split = train_test_split(
        train_df,
        test_size=split_ratio,
        random_state=42,
        shuffle=True
    )

    print(f"Training split records: {len(train_split)}")
    print(f"Validation split records: {len(valid_split)}")

    # Save the splits
    print(f"Saving training split to {processed_train_csv}...")
    train_split.to_csv(processed_train_csv, index=False)

    print(f"Saving validation split to {processed_valid_csv}...")
    valid_split.to_csv(processed_valid_csv, index=False)

    print("Data preparation completed successfully.")

if __name__ == "__main__":
    split_train_validation(
        TRAIN_CSV_PATH,
        PROCESSED_TRAIN_CSV,
        PROCESSED_VALID_CSV,
        VALIDATION_SPLIT_RATIO
    )