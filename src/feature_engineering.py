import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
import os

# Load the train data
train_filepath = './data/preprocessed/train_preprocessed.csv'
data = pd.read_csv(train_filepath, index_col='Id')

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

# Tách SalePrice để không áp dụng PCA lên cột này
sale_price = data_standardized['SalePrice']
features = data_standardized.drop(columns=['SalePrice'])

# Áp dụng PCA cho train set
pca = PCA(n_components=30)  
features_pca = pca.fit_transform(features)

# Chuyển về DataFrame
pca_columns = [f'feature{i+1}' for i in range(features_pca.shape[1])]
data_pca = pd.DataFrame(features_pca, columns=pca_columns, index=data_standardized.index)

# Ghép lại với SalePrice
data_pca['SalePrice'] = sale_price

# Lưu lại dữ liệu đã xử lý với PCA
train_pca_filepath = './data/preprocessed/train_preprocessed_pca.csv'
data_pca.to_csv(train_pca_filepath)

print(f"PCA processed train data saved to {train_pca_filepath}")

# Save preprocessed train data
train_preprocessed_filepath = './data/preprocessed/train_preprocessed.csv'
data_standardized.to_csv(train_preprocessed_filepath)
print(f"Preprocessed train data saved to {train_preprocessed_filepath}")

### ---- TEST SET ---- ###

# Load test data

# Load test data
test_filepath = './data/preprocessed/test_preprocessed.csv'
data = pd.read_csv(test_filepath, index_col='Id')

# Fill missing values using forward fill
data = data.ffill()

# Encode categorical columns
label_encoders = {}
for col in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le  # Store encoders for potential inverse transform

# Standardize numerical features (giả sử scaler đã được fit từ train set)
scaler = StandardScaler()
data_standardized = pd.DataFrame(scaler.fit_transform(data), columns=data.columns, index=data.index)

# Fit PCA 
pca = PCA(n_components=30)
features_pca = pca.fit_transform(data_standardized)

# Chuyển về DataFrame
pca_columns = [f'feature{i+1}' for i in range(features_pca.shape[1])]
data_pca = pd.DataFrame(features_pca, columns=pca_columns, index=data_standardized.index)

# Save preprocessed test data (có PCA)
test_pca_filepath = './data/preprocessed/test_preprocessed_pca.csv'
data_pca.to_csv(test_pca_filepath)

print(f"PCA processed test data saved to {test_pca_filepath}")