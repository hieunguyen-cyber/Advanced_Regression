import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Chuyển thư mục nếu cần
os.chdir(os.getcwd().replace('/notebooks', ''))

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