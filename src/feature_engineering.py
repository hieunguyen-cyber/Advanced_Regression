import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
import os

# Load the train data
train_filepath = './data/preprocessed/train_preprocessed.csv'
data = pd.read_csv(train_filepath, index_col='Id')

# Fill missing values for numerical columns
data['LotFrontage'] = data['LotFrontage'].fillna(data[data['LotFrontage'] < 300]['LotFrontage'].mean())
data['GarageYrBlt'] = data['GarageYrBlt'].fillna(data['GarageYrBlt'].interpolate())
data['MasVnrArea'] = data['MasVnrArea'].fillna(0)

# Fill missing values for categorical columns
data['MasVnrType'] = data['MasVnrType'].fillna('None')
data.drop(['MiscFeature', 'PoolQC', 'Fence', 'Alley'], axis=1, inplace=True, errors='ignore')

def fill_missing(df, col, i):
    # Step 1: Set point value
    point = {}
    for type in df[col].unique():
        num = df[df[col] == type][col].count()
        if num > 0:
            point[type] = 1/num
        else:
            point[type] = 0
    # Step 2: Find mode value for each range
    mode = {}
    for i in range(5):
        rang = [i*144020, (i+1)*144020]
        data = df[col][df['SalePrice'].between(rang[0], rang[1])]
        max = 0
        for type in data.unique():
            if data[data == type].count()*point[type] > max:
                mode[i] = type
                max = data[data == type].count()*point[type]
    # Step 3: Fill missing value
    if i in data[data[col].isnull()].index:
        for c in mode.keys():
            if df['SalePrice'][i] in range(c*144020, (c+1)*144020):
                df[col][i] = mode[c]

for col in ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Electrical', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:
    data[col] = data[col].apply(lambda x: fill_missing(data, col, x) if x == 'Nan' else x)

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
test_filepath = './data/preprocessed/test_preprocessed.csv'
test_data = pd.read_csv(test_filepath, index_col='Id')

# Fill missing values using forward fill
test_data = test_data.ffill()
test_data.drop(['MiscFeature', 'PoolQC', 'Fence', 'Alley'], axis=1, inplace=True, errors='ignore')

# Encode categorical columns
label_encoders = {}
for col in test_data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    test_data[col] = le.fit_transform(test_data[col])
    label_encoders[col] = le  # Store encoders for potential inverse transform

# Standardize numerical features (giả sử scaler đã được fit từ train set)
scaler = StandardScaler()
test_data_standardized = pd.DataFrame(scaler.fit_transform(test_data), columns=test_data.columns, index=test_data.index)

# Fit PCA 
pca = PCA(n_components=30)
features_pca = pca.fit_transform(test_data_standardized)

# Chuyển về DataFrame
pca_columns = [f'feature{i+1}' for i in range(features_pca.shape[1])]
test_data_pca = pd.DataFrame(features_pca, columns=pca_columns, index=test_data_standardized.index)

# Save preprocessed test data (có PCA)
test_pca_filepath = './data/preprocessed/test_preprocessed_pca.csv'
test_data_pca.to_csv(test_pca_filepath)

print(f"PCA processed test data saved to {test_pca_filepath}")

# Save preprocessed train data
test_preprocessed_filepath = './data/preprocessed/test_preprocessed.csv'
test_data_standardized.to_csv(test_preprocessed_filepath)
print(f"Preprocessed test data saved to {test_preprocessed_filepath}")