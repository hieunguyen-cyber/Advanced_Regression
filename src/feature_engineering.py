import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
import os

# Load train & test data
train_filepath = './data/preprocessed/train_preprocessed.csv'
test_filepath = './data/preprocessed/test_preprocessed.csv'
train = pd.read_csv(train_filepath, index_col='Id')
test = pd.read_csv(test_filepath, index_col='Id')

# Drop unwanted columns
cols_to_drop = ['MiscFeature', 'PoolQC', 'Fence', 'Alley']
train.drop(columns=cols_to_drop, inplace=True)
test.drop(columns=cols_to_drop, inplace=True)

# Fill missing values (Numerical)
train['LotFrontage'] = train['LotFrontage'].fillna(train.loc[train['LotFrontage'] < 300, 'LotFrontage'].mean())
test['LotFrontage'] = test['LotFrontage'].fillna(test.loc[test['LotFrontage'] < 300, 'LotFrontage'].mean())

train['GarageYrBlt'] = train['GarageYrBlt'].interpolate()
test['GarageYrBlt'] = test['GarageYrBlt'].interpolate()

train['MasVnrArea'] = train['MasVnrArea'].fillna(0)
test['MasVnrArea'] = test['MasVnrArea'].fillna(0)

# Fill missing values (Categorical)
train['MasVnrType'] = train['MasVnrType'].fillna('None')
test['MasVnrType'] = test['MasVnrType'].fillna('None')

# Separate SalePrice
sale_price = train.pop('SalePrice')

# Save SalePrice statistics
scaling_params = {
    'mean': sale_price.mean(),
    'std': sale_price.std(),
    'min': sale_price.min(),
    'max': sale_price.max()
}
scaling_params_filepath = './data/preprocessed/scaling_params.txt'
os.makedirs(os.path.dirname(scaling_params_filepath), exist_ok=True)
with open(scaling_params_filepath, 'w') as f:
    for key, value in scaling_params.items():
        f.write(f'{key}: {value}\n')

# Encode categorical columns (Bỏ qua NaN)
label_encoders = {}
for col in train.select_dtypes(include=['object']).columns:
    le = LabelEncoder()

    # Encode train
    train[col] = le.fit_transform(train[col].astype(str).fillna("MISSING"))

    # Cập nhật classes_ của LabelEncoder
    test[col] = test[col].astype(str).fillna("MISSING")
    unseen_values = set(test[col].unique()) - set(le.classes_)

    if unseen_values:
        le.classes_ = np.append(le.classes_, list(unseen_values))  # Thêm giá trị unseen vào classes_

    # Encode test
    test[col] = le.transform(test[col])

    label_encoders[col] = le

# Fill NaN for numerical columns
for col in train.select_dtypes(include=['number']).columns:
    train[col] = train[col].interpolate()
    test[col] = test[col].interpolate()

# Standardize numerical features
scaler = StandardScaler()
train_standardized = pd.DataFrame(scaler.fit_transform(train), columns=train.columns, index=train.index)
test_standardized = pd.DataFrame(scaler.transform(test), columns=test.columns, index=test.index)
sale_price_scaled = scaler.fit_transform(sale_price.values.reshape(-1, 1))
# Apply PCA (fit trên train, transform trên test)
pca = PCA(n_components=30)
features_pca_train = pca.fit_transform(train_standardized)
features_pca_test = pca.transform(test_standardized)

# Convert back to DataFrame
pca_columns = [f'feature{i+1}' for i in range(features_pca_train.shape[1])]
train_pca = pd.DataFrame(features_pca_train, columns=pca_columns, index=train_standardized.index)
test_pca = pd.DataFrame(features_pca_test, columns=pca_columns, index=test_standardized.index)

# Thêm lại SalePrice vào train PCA
train_pca['SalePrice'] = sale_price_scaled
train_standardized['SalePrice'] = sale_price_scaled

# Save PCA processed data
train_pca_filepath = './data/preprocessed/train_preprocessed_pca.csv'
train_pca.to_csv(train_pca_filepath)
print(f"PCA processed train data saved to {train_pca_filepath}")

test_pca_filepath = './data/preprocessed/test_preprocessed_pca.csv'
test_pca.to_csv(test_pca_filepath)
print(f"PCA processed test data saved to {test_pca_filepath}")

# Save standardized data before PCA
train_preprocessed_filepath = './data/preprocessed/train_preprocessed.csv'
train_standardized.to_csv(train_preprocessed_filepath)
print(f"Preprocessed train data saved to {train_preprocessed_filepath}")

test_preprocessed_filepath = './data/preprocessed/test_preprocessed.csv'
test_standardized.to_csv(test_preprocessed_filepath)
print(f"Preprocessed test data saved to {test_preprocessed_filepath}")