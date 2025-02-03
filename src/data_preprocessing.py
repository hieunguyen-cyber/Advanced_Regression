"""
 Mọi người có thể đọc file exploratory_data_analysis.ipynb để hiểu rõ hơn về các bước tiền xử lý dữ liệu và phân tích dữ liệu.   
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
# Chu y: thay doi duong dan file tuy theo vi tri luu file tren may tinh cua ban
filepath = r'C:\Users\Admin\OneDrive - Hanoi University of Science and Technology\Training đi thi\Project\Advanced_Regression\data\raw\train.csv'
data = pd.read_csv(filepath, index_col='Id')

# Find columns with missing values
lst_of_missing = []
for col in data.columns:
    if data[col].isnull().sum() > 0:
        lst_of_missing.append(col)

# Find numerical and categorical columns
lst_of_numerical = []
lst_of_categorical = []
for col in data.columns:
    if data[col].dtype == 'object':
        lst_of_categorical.append(col)
    else:
        lst_of_numerical.append(col)
        
# Plot the distribution of numerical columns with missing values
num_missing = set(lst_of_numerical)&set(lst_of_missing)
num_missing.add("SalePrice")

fig, ax = plt.subplots(1, 3, figsize=(15, 5))
sns.scatterplot(x='LotFrontage', y='SalePrice', data=data, ax=ax[0], color='r').set(title='LotFrontage')
sns.scatterplot(x='MasVnrArea', y='SalePrice', data=data, ax=ax[1], color='g').set(title='MasVnrArea')
sns.scatterplot(x='GarageYrBlt', y='SalePrice', data=data, ax=ax[2], color='b').set(title='GarageYrBlt')
plt.tight_layout()

# Plot heatmap, barplot of correlation matrix
correlation_matrix = data[lst_of_numerical].corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, cmap='coolwarm', linewidths=0.5)

corr_with_saleprice = correlation_matrix.loc[:'YrSold', 'SalePrice'].sort_values(ascending=True)
plt.figure(figsize=(12, 6))
sns.barplot(x=corr_with_saleprice.index,y=corr_with_saleprice.values, palette='coolwarm')
plt.xticks(rotation=90)
plt.yticks(ticks=np.arange(-0.2,1,0.2))

# Plot the distribution of categorical columns with missing values
cat_missing = set(lst_of_categorical)&set(lst_of_missing)
print(f'There are {len(cat_missing)} categorical columns with missing values')
for col in cat_missing:
    print(f'{col:<13}: {data[col].isnull().sum(): <4} missing values - {data[col].isnull().sum() / len(data) * 100:.2f}% - {len(data[col].unique())} unique values')
    
Bstmnt_cols = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
fig2, ax2 = plt.subplots(2, 3, figsize=(15, 10))
ax2.flat[-1].set_visible(False)
for r in range(2):
    for c in range(3):
        sns.stripplot(data=data,x=Bstmnt_cols[r*3+c-1] , y="SalePrice" , ax=ax2[r,c])
        
Garage_cols = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']
fig3, ax3 = plt.subplots(2, 2, figsize=(10, 10))
for r in range(2):
    for c in range(2):
        sns.stripplot(data=data,x=Garage_cols[r*2+c-1] , y="SalePrice" , ax=ax3[r,c])
        
sns.stripplot(data=data,x="FireplaceQu" , y="SalePrice")

plt.show()