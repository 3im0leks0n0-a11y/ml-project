# Library
import matplotlib
matplotlib.use('TkAgg')
import pandas as pd # type: ignore 
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore
from sklearn.preprocessing import LabelEncoder # type: ignore

# Load dataset
data = pd.read_csv('dataset/thyroidDF.csv')

# print(data.info()) #Show the information of the dataset datatypes, non-null values, and memory usage
# print(data.isnull().sum()) #how the number of missing values in each column

# Fill missing values with the mean of each column for the numerical columns
numerical_cols = data.select_dtypes(include=[np.number]).columns
data[numerical_cols] = data[numerical_cols].fillna(data[numerical_cols].mean())
# For categorical columns, fill missing values with the mode (most frequent value)
categorical_cols = data.select_dtypes(include=['object', 'string']).columns #Select categorical columns based on object and string data types
data[categorical_cols] = data[categorical_cols].fillna(data[categorical_cols].mode().iloc[0])

#print(data.isnull().sum()) #Verify that there are no more missing values in the dataset

# Identify outliers using the IQR method
Q1 = data[numerical_cols].quantile(0.25)
Q3 = data[numerical_cols].quantile(0.75)
IQR = Q3 - Q1

# Remove outliers from the dataset
# Filter dataframe untuk hanya menyimpan baris yang tidak mengandung outliers pada kolom numerik
condition = ~((data[numerical_cols] < (Q1 - 1.5 * IQR)) | (data[numerical_cols] > (Q3 + 1.5 * IQR))).any(axis=1)
data_filtered_numeric = data.loc[condition, numerical_cols]
 
# Menggabungkan kembali dengan kolom kategorikal
categorical_features = data.select_dtypes(include=['object', 'string']).columns
data = pd.concat([data_filtered_numeric, data.loc[condition, categorical_features]], axis=1)

# Visualize the distribution of each feature using boxplots and histograms
# for feature in data.columns:
#     plt.figure(figsize=(10, 6))
#     sns.boxplot(x=data[feature])
#     plt.title(f'Boxplot of {feature}')
#     plt.hist(data[feature], bins=30, edgecolor='k')
#     plt.title(f'Histogram of {feature}')
#     plt.xlabel(feature)
#     plt.ylabel('Frequency')
#     plt.show()
#     plt.close()


# for col in numerical_cols:

#     before_scaling = data[col].copy()

#     scaler = StandardScaler()
#     after_scaling = scaler.fit_transform(data[[col]])

#     plt.figure(figsize=(12,5))

#     plt.subplot(1,2,1)
#     sns.boxplot(data=before_scaling)

#     plt.subplot(1,2,2)
#     sns.boxplot(data=after_scaling)

#     plt.show()

# print("Before scaling:")
# print(before_scaling.describe())

# print("\nAfter scaling:")
# print(after_scaling.describe())

# Identify and remove duplicate rows
# duplicates = data.duplicated()
 
# print("Baris duplikat:")
# print(data[duplicates])

# Hot encode categorical features using pd.get_dummies
category_features = data.select_dtypes(include=['object', 'string']).columns
data_one_hot = pd.get_dummies(data, columns=category_features)

# Encode categorical features using Label Encoding
label_encoder = LabelEncoder()
data_lencoder = pd.DataFrame(data)
for col in category_features:
    data_lencoder[col] = label_encoder.fit_transform(data_lencoder[col])

# # Menghitung jumlah variabel
# num_vars = data_lencoder.shape[1]
 
# # Menentukan jumlah baris dan kolom untuk grid subplot
# n_cols = 7  # Jumlah kolom yang diinginkan
# n_rows = -(-num_vars // n_cols)  # Ceiling division untuk menentukan jumlah baris
 
# # Membuat subplot
# fig, axes = plt.subplots(n_rows, n_cols, figsize=(30, n_rows * 7))
 
# # Flatten axes array untuk memudahkan iterasi jika diperlukan
# axes = axes.flatten()
 
# # Plot setiap variabel
# for i, column in enumerate(data_lencoder.columns):
#     data_lencoder[column].hist(ax=axes[i], bins=20, edgecolor='black')
#     axes[i].set_title(column)
#     axes[i].set_xlabel('Value')
#     axes[i].set_ylabel('Frequency')
 
# # Menghapus subplot yang tidak terpakai (jika ada)
# for j in range(i + 1, len(axes)):
#     fig.delaxes(axes[j])
 
# # Menyesuaikan layout agar lebih rapi
# plt.tight_layout()
# plt.subplots_adjust(hspace=0.6, wspace=0.4)
# plt.show()

# Visualisasi distribusi data untuk beberapa kolom
# columns_to_plot = ['age', 'TSH', 'T3', 'TT4', 'T4U', 'FTI']

# plt.figure(figsize=(16,10))

# for i, column in enumerate(columns_to_plot, 1):
#     plt.subplot(2, 3, i)
#     sns.histplot(data_lencoder[column], kde=True, bins=30)
#     plt.title(f'Distribution of {column}')

# plt.tight_layout()
# plt.show()

# Visualisasi korelasi antar variabel menggunakan heatmap
    plt.figure(figsize=(10,8))
    sns.heatmap(data_lencoder.corr(), annot=False, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Matrix')
    plt.show()

# target_corr = data_lencoder.corr()['age']
 
# # (Opsional) Mengurutkan hasil korelasi berdasarkan korelasi
# target_corr_sorted = target_corr.abs().sort_values(ascending=False)
 
# plt.figure(figsize=(10, 6))
# target_corr_sorted.plot(kind='bar')
# plt.title(f'Correlation with age')
# plt.xlabel('Variables')
# plt.ylabel('Correlation Coefficient')
# plt.show()