"""
 Mọi người có thể đọc file features_engineering.ipynb để hiểu rõ hơn về các bước tiền xử lý dữ liệu 
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

# Load the data
# Chu y: thay doi duong dan file tuy theo vi tri luu file tren may tinh cua ban
filepath = r'C:\Users\Admin\OneDrive - Hanoi University of Science and Technology\Training đi thi\Project\Advanced_Regression\data\raw\train.csv'
data = pd.read_csv(filepath, index_col='Id')

# Fill missing values
# Fill missing values for numerical columns
data['LotFrontage'] = data['LotFrontage'].fillna(data[data['LotFrontage'] < 300]['LotFrontage'].mean())
data['GarageYrBlt'] = data['GarageYrBlt'].fillna(data['GarageYrBlt'].interpolate())
data['MasVnrArea'] = data['MasVnrArea'].fillna(0)

# Fill missing values for categorical columns
data['MasVnrType'] = data['MasVnrType'].fillna('None')
data.drop(['MiscFeature','PoolQC','Fence','Alley'],axis=1, inplace=True)
lst_of_missing = []
for col in data.columns:
    if data[col].isnull().sum() > 0:
        lst_of_missing.append(col)
        
# Fill missing function
def fill_missing(df, col):
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
    for i in df[df[col].isnull()].index:
        for c in mode.keys():
            if df['SalePrice'][i] in range(c*144020, (c+1)*144020):
                df[col][i] = mode[c]

for col in lst_of_missing:
    data[col] = data[col].apply(lambda x: fill_missing(data, col, x) if x == 'Nan' else x)
    
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