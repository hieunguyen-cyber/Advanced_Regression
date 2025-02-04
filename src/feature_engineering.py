import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

# Load the data
filepath = './data/preprocessed/train_preprocessed.csv'
data = pd.read_csv(filepath, index_col='Id')

# Tách cột cuối cùng ra khỏi dữ liệu xử lý
last_column = data.columns[-1]
data_last_col = data[[last_column]]
data = data.drop(columns=[last_column])

# Fill missing values
if 'LotFrontage' in data.columns:
    data['LotFrontage'] = data['LotFrontage'].fillna(data[data['LotFrontage'] < 300]['LotFrontage'].mean())
if 'GarageYrBlt' in data.columns:
    data['GarageYrBlt'] = data['GarageYrBlt'].fillna(data['GarageYrBlt'].interpolate())
if 'MasVnrArea' in data.columns:
    data['MasVnrArea'] = data['MasVnrArea'].fillna(0)

# Fill missing values for categorical columns
if 'MasVnrType' in data.columns:
    data['MasVnrType'] = data['MasVnrType'].fillna('None')
data.drop([col for col in ['MiscFeature','PoolQC','Fence','Alley'] if col in data.columns], axis=1, inplace=True)
lst_of_missing = [col for col in data.columns if data[col].isnull().sum() > 0]
        
# Fill missing function
def fill_missing(df, col):
    point = {}
    for type in df[col].unique():
        num = df[df[col] == type][col].count()
        if num > 0:
            point[type] = 1/num
        else:
            point[type] = 0
    mode = {}
    for i in range(5):
        rang = [i*144020, (i+1)*144020]
        data_sub = df[col][df['SalePrice'].between(rang[0], rang[1])]
        max_val = 0
        for type in data_sub.unique():
            if data_sub[data_sub == type].count()*point[type] > max_val:
                mode[i] = type
                max_val = data_sub[data_sub == type].count()*point[type]
    for i in df[df[col].isnull()].index:
        for c in mode.keys():
            if df['SalePrice'][i] in range(c*144020, (c+1)*144020):
                df.at[i, col] = mode[c]

for col in lst_of_missing:
    data[col] = data[col].apply(lambda x: fill_missing(data, col) if x == 'Nan' else x)

# Encode categorical columns
label_encoders = LabelEncoder()
for col in data.columns:
    if data[col].dtype == 'object':
        data[col] = label_encoders.fit_transform(data[col])
        
# Normalized data using StandardScaler
scaler = StandardScaler()
data_standardized = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

# Dimensionality reduction using PCA
pca = PCA(n_components=3)
pca.fit(data_standardized.T)
data_pca1 = pca.components_.T
print(data_pca1)

# Ghép lại cột cuối cùng
final_data = pd.concat([data_standardized.reset_index(drop=True), data_last_col.reset_index(drop=True)], axis=1)

# Save preprocessed data
output_filepath = './data/preprocessed/train_preprocessed.csv'
final_data.to_csv(output_filepath)
