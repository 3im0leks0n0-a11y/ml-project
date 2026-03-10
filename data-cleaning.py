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

print(data_lencoder.head())

