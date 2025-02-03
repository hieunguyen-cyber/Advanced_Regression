# Advanced_Regression
Solving a House Price Prediction Competition for Beginners on Kaggle

[Competition Link 🔗](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)

# Project Directory Structure
First install requirements.txt by 
~~~
pip install -r requirements.
~~~


```Python
house-price-prediction/
│
├── data/                    # Dữ liệu thô và đã qua xử lý
│   ├── raw/                 # Dữ liệu thô ban đầu
│   ├── processed/           # Dữ liệu đã được xử lý
│   └── README.md            # Mô tả cách tổ chức và sử dụng dữ liệu
│
├── notebooks/               # Notebook Jupyter để phân tích và thử nghiệm
│   ├── 01-data-exploration.ipynb   # Khám phá dữ liệu
│   ├── 02-feature-engineering.ipynb  # Xử lý và chọn đặc trưng
│   ├── 03-model-training.ipynb     # Huấn luyện mô hình
│   └── 04-evaluation.ipynb         # Đánh giá mô hình
│
├── src/                     # Mã nguồn chính của dự án
│   ├── data_processing.py   # Xử lý dữ liệu
│   ├── feature_engineering.py  # Tạo và chọn đặc trưng
│   ├── model_training.py    # Huấn luyện mô hình
│   ├── evaluation.py        # Đánh giá mô hình
│   └── utils.py             # Các tiện ích dùng chung
│
├── models/                  # Mô hình đã được huấn luyện
│   ├── model.pkl            # File chứa mô hình lưu trữ (pickle hoặc joblib)
│   └── README.md            # Thông tin về mô hình
│
├── results/                 # Kết quả và báo cáo
│   ├── plots/               # Biểu đồ và hình ảnh trực quan
│   ├── metrics.txt          # Các chỉ số đánh giá mô hình
│   └── README.md            # Mô tả kết quả và cách tổ chức
│
├── tests/                   # Các script kiểm thử
│   ├── test_data_processing.py
│   ├── test_feature_engineering.py
│   ├── test_model_training.py
│   └── README.md
│
├── requirements.txt         # Danh sách thư viện Python cần thiết
├── README.md                # Tài liệu mô tả dự án
├── .gitignore               # Quy định các file/folder không cần commit
└── main.py                  # Chay file nay
