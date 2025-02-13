import pandas as pd
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def data_preprocessing():
    # Load the data
    filepath = os.path.join('data', 'raw', 'train.csv')
    data = pd.read_csv(filepath, index_col='Id')

    # Loại bỏ cột cuối cùng khỏi danh sách cột xử lý
    columns_to_process = data.columns

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

    # Load the data
    filepath = os.path.join('data', 'raw', 'test.csv')
    data = pd.read_csv(filepath, index_col='Id')

    # Loại bỏ cột cuối cùng khỏi danh sách cột xử lý
    columns_to_process = data.columns

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
    output_filepath = os.path.join(output_dir, 'test_preprocessed.csv')
    data.to_csv(output_filepath)
    print(f"Preprocessed data saved to {output_filepath}")
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
def predict():
    # Đọc dữ liệu
    train_filepath = './data/preprocessed/train_preprocessed.csv'
    test_filepath = './data/preprocessed/test_preprocessed.csv'

    train_data = pd.read_csv(train_filepath)
    test_data = pd.read_csv(test_filepath)

    # Tách features và target
    X = train_data.drop(columns=['SalePrice'])
    y = train_data['SalePrice']

    # Chia train - validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Cải thiện mô hình XGBoost
    # Cải thiện mô hình XGBoost
    model = xgb.XGBRegressor(
        n_estimators=31679,        # Giảm số vòng lặp để tránh overfitting
        learning_rate=0.027641540158987947,       # Giảm tốc độ học
        max_depth=7,              # Tăng chiều sâu
        min_child_weight= 2,
        subsample=0.651203956733307,           # Tăng tập con train
        colsample_bytree=0.8818190813201844,    # Tăng tập con feature
        reg_lambda=1.2021383673878119,           # L2 Regularization
        reg_alpha=0.36844579384256904,           # L1 Regularization
        gamma = 0.1820205101783258,          
        objective='reg:squarederror',
        eval_metric='rmse',
        random_state=42
    )

    # Huấn luyện mô hình với Early Stopping
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=100
    )

    # Dự đoán trên tập test
    test_predictions = model.predict(test_data)

    print("Training Completed!")

    with open('data/preprocessed/scaling_params.txt', 'r') as f:
        lines = f.readlines()
        mean = float(lines[0].split()[1])
        std = float(lines[1].split()[1])

    # Đảo ngược chuẩn hóa
    test_predictions = test_predictions * std + mean

    # Lưu kết quả
    output_dir = './data/output'
    os.makedirs(output_dir, exist_ok=True)
    output_filepath = os.path.join(output_dir, 'predictions.csv')
    test_ids = pd.read_csv(test_filepath)['Id']
    submission = pd.DataFrame({'Id': test_ids, 'SalePrice': test_predictions})
    submission.to_csv(output_filepath, index=False)

    print(f"Predictions saved to {output_filepath}")
if __name__ == "__main__":
    data_preprocessing()
    predict()