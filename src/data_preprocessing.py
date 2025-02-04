import pandas as pd
import numpy as np
import os

# Load the data
filepath = os.path.join('data', 'raw', 'train.csv')
data = pd.read_csv(filepath, index_col='Id')

# Loại bỏ cột cuối cùng khỏi danh sách cột xử lý
columns_to_process = data.columns[:-1]

# Find columns with missing values
lst_of_missing = [col for col in columns_to_process if data[col].isnull().sum() > 0]

# Find numerical and categorical columns
lst_of_numerical = [col for col in columns_to_process if data[col].dtype != 'object']
lst_of_categorical = [col for col in columns_to_process if data[col].dtype == 'object']

# Print information about missing categorical columns
cat_missing = set(lst_of_categorical) & set(lst_of_missing)
print(f'There are {len(cat_missing)} categorical columns with missing values')
for col in cat_missing:
    print(f'{col:<13}: {data[col].isnull().sum(): <4} missing values - {data[col].isnull().sum() / len(data) * 100:.2f}% - {len(data[col].unique())} unique values')

# Save preprocessed data
output_dir = os.path.join('data', 'preprocessed')
os.makedirs(output_dir, exist_ok=True)
output_filepath = os.path.join(output_dir, 'train_preprocessed.csv')
data.to_csv(output_filepath)
print(f"Preprocessed data saved to {output_filepath}")
