# category_attributes_mapping.py

import os
import pandas as pd

# Get the current directory (directory where the script is located)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Path to train.csv
TRAIN_CSV = os.path.join(current_dir, 'data','raw','train.csv')

def create_category_attributes_mapping(train_csv_path):
    train_df = pd.read_csv(train_csv_path)
    attribute_columns = [f'attr_{i}' for i in range(1, 11)]
    category_attr_count = {}

    # For each Category, find out which attributes are used
    for category, group in train_df.groupby('Category'):
        # For the group, check which attributes have meaningful values
        # Assuming that 'dummy_value', 'default', or NaN are not meaningful
        num_attrs = 0
        for attr in attribute_columns:
            # Check if the attribute column has at least one meaningful value in this category
            unique_values = group[attr].dropna().unique()
            meaningful_values = [v for v in unique_values if v not in ['dummy_value', 'default', 'no', 'yes', 'nan']]
            if len(meaningful_values) > 0:
                num_attrs += 1
        category_attr_count[category] = num_attrs

    return category_attr_count

def main():
    category_attr_count = create_category_attributes_mapping(TRAIN_CSV)
    # Save the mapping to a CSV file
    output_path = os.path.join(current_dir, 'category_attributes.csv')
    pd.DataFrame(list(category_attr_count.items()), columns=['Category', 'num_attributes']).to_csv(output_path, index=False)
    print(f"Category attributes mapping saved to '{output_path}'.")

if __name__ == "__main__":
    main()