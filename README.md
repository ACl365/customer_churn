# Telecom Churn Prediction Project

This project aims to predict customer churn in the telecom industry using machine learning techniques.

## Data Preparation

The data preparation step involves the following:
- Loading the training and holdout data using the `pandas` library.
- Removing the `CustomerID` column.
- Handling missing values by filling categorical features with "Unknown" and numerical features with the mean.
- Converting categorical features to numerical features using `LabelEncoder`. The LabelEncoder is fit on the combined data from both training and holdout sets to avoid errors due to unseen labels.
- Scaling numerical features using `StandardScaler`.
- Splitting the training data into training and validation sets.

The shape of the training data is (51047, 57) and the shape of the holdout data is (20000, 57). The training data is split into a training set with shape (40837, 56) and a validation set with shape (10210, 56).