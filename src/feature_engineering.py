import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
import os

# Load the data
filepath = './data/preprocessed/train_preprocessed.csv'
data = pd.read_csv(filepath, index_col='Id')

# Fill missing values
data['LotFrontage'] = data['LotFrontage'].fillna(data[data['LotFrontage'] < 300]['LotFrontage'].mean())
data['GarageYrBlt'] = data['GarageYrBlt'].fillna(data['GarageYrBlt'].interpolate())
data['MasVnrArea'] = data['MasVnrArea'].fillna(0)

# Fill missing values for categorical columns
data['MasVnrType'] = data['MasVnrType'].fillna('None')
data.drop(['MiscFeature', 'PoolQC', 'Fence', 'Alley'], axis=1, inplace=True)

# Encode categorical columns
label_encoders = {}
for col in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le  # Store encoders for potential inverse transform

# Standardize numerical features
scaler = StandardScaler()
data_standardized = pd.DataFrame(scaler.fit_transform(data), columns=data.columns, index=data.index)

# Save scaling parameters for SalePrice
scaling_params = {
    'mean': data['SalePrice'].mean(),
    'std': data['SalePrice'].std(),
    'min': data['SalePrice'].min(),
    'max': data['SalePrice'].max()
}
scaling_params_filepath = './data/preprocessed/scaling_params.txt'
os.makedirs(os.path.dirname(scaling_params_filepath), exist_ok=True)
with open(scaling_params_filepath, 'w') as f:
    for key, value in scaling_params.items():
        f.write(f'{key}: {value}\n')

# Save preprocessed data
output_filepath = './data/preprocessed/train_preprocessed.csv'
data_standardized.to_csv(output_filepath)
print(f"Preprocessed data saved to {output_filepath}")
