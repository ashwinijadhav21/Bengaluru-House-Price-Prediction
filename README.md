# Bengaluru House Price Prediction

## 📌 Project Overview

Bengaluru House Price Prediction is a **Machine Learning regression project** that predicts residential property prices in Bengaluru based on features such as location, total square footage, number of bedrooms (BHK), bathrooms, balcony availability, property type, and availability status.

The project focuses on understanding and preparing real-world housing data, performing feature engineering and outlier treatment, encoding categorical variables, training regression models, and evaluating their performance using standard regression metrics.

---

## 🎯 Objective

The main objective of this project is to build a machine learning model that can predict house prices based on available property features.

The project covers the complete workflow:

* Data loading and exploration
* Missing-value treatment
* Data cleaning
* Feature engineering
* Outlier detection and removal
* Categorical feature encoding
* Train-test splitting
* Regression model training
* Model evaluation
* Feature importance analysis
* Model saving using Joblib

---

## 📊 Dataset

The project uses the **Bengaluru House Data** dataset.

### Original Features

| Feature        | Description                               |
| -------------- | ----------------------------------------- |
| `area_type`    | Type of property area                     |
| `availability` | Property availability status              |
| `location`     | Location of the property                  |
| `size`         | Property size represented in BHK          |
| `society`      | Society/project name                      |
| `total_sqft`   | Total area of the property in square feet |
| `bath`         | Number of bathrooms                       |
| `balcony`      | Number of balconies                       |
| `price`        | House price in lakhs — Target variable    |

### Dataset Size

* **Rows:** 13,320
* **Columns:** 9

---

## 🔧 Data Preprocessing

The dataset required several preprocessing steps before machine learning could be applied.

### 1. Missing Value Treatment

The following steps were performed:

* Dropped the `society` column because it contained a large number of missing values.
* Removed rows with missing `location` and `size`.
* Filled missing `bath` values using the median.
* Filled missing `balcony` values using the median.

### 2. BHK Feature Engineering

The original `size` column contained values such as:

```text
2 BHK
3 BHK
4 BHK
```

It was converted into a numerical `bhk` feature.

```python
df['bhk'] = df['size'].apply(lambda x: int(x.split(' ')[0]))
```

The original `size` column was then removed.

### 3. Total Square Feet Conversion

Some `total_sqft` values were provided as ranges, for example:

```text
1200-1500
```

These values were converted into numerical values by taking the average of the range.

Other non-numeric values that could not be converted were treated as missing and removed.

### 4. BHK Outlier Removal

Properties with unusually high numbers of bedrooms were removed.

```python
df = df[df['bhk'] <= 10]
```

### 5. Price Per Square Foot

A temporary `price_per_sqft` feature was created for identifying price-related outliers.

```python
df['price_per_sqft'] = (df['price'] * 100000) / df['total_sqft']
```

The **IQR (Interquartile Range)** method was used to identify and remove extreme values.

After outlier treatment, the temporary `price_per_sqft` feature was removed because it is directly derived from the target variable and should not be used as a model input.

### 6. Square Feet per BHK

A temporary `sqft_per_bhk` feature was created:

```python
df['sqft_per_bhk'] = df['total_sqft'] / df['bhk']
```

Properties with extremely low area per BHK were removed as potential outliers.

The temporary feature was then removed before model training.

---

## 🔤 Categorical Encoding

Machine learning models require numerical input, so categorical features were converted using **One-Hot Encoding**.

The following categorical features were encoded:

* `area_type`
* `location`
* `availability`

`handle_unknown='ignore'` was used for categorical encoding so that unseen categories can be handled safely.

---

## 🧪 Train-Test Split

The cleaned dataset was divided into training and testing datasets.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

* **80%** data → Training
* **20%** data → Testing

The target variable was:

```text
price
```

---

## 🤖 Machine Learning Models

Two regression algorithms were trained and compared.

### 1. Random Forest Regressor

```python
RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)
```

### 2. Gradient Boosting Regressor

```python
GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)
```

---

## 📈 Model Evaluation

The models were evaluated using:

* **MAE — Mean Absolute Error**
* **RMSE — Root Mean Squared Error**
* **R² Score — Coefficient of Determination**

### Results

| Model             |   MAE |  RMSE |    R² |
| ----------------- | ----: | ----: | ----: |
| Random Forest     | 16.80 | 35.63 | 0.733 |
| Gradient Boosting | 19.44 | 36.13 | 0.726 |

The Random Forest model achieved an **R² score of approximately 0.733** on the test dataset.

---

## 🌳 Feature Importance

Feature importance was extracted from the trained Random Forest model to understand which features contributed most to the model's predictions.

```python
feature_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': rf_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by='Importance',
    ascending=False
)
```

The top features were visualized using a horizontal bar chart.

---

## 💾 Model Saving

The trained Random Forest model was saved using **Joblib**:

```python
joblib.dump(
    rf_model,
    'bengaluru_house_price_model.pkl'
)
```

The `.pkl` file contains the trained model and can be loaded later for prediction.

> **Note:** The trained model file is approximately 206 MB and is therefore excluded from this GitHub repository because GitHub's standard file-size limit is 100 MB.

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* Joblib
* Jupyter Notebook
* Git & GitHub

---

## 📚 Machine Learning Concepts Used

This project demonstrates practical knowledge of:

* Data Cleaning
* Exploratory Data Analysis
* Missing Value Treatment
* Feature Engineering
* One-Hot Encoding
* Outlier Detection
* IQR Method
* Train-Test Split
* Regression
* Random Forest
* Gradient Boosting
* Model Evaluation
* MAE
* RMSE
* R² Score
* Feature Importance
* Model Serialization

---

## 📁 Project Structure

```text
Bengaluru-House-Price-Prediction/
│
├── price_pred.ipynb
├── README.md
├── .gitignore
└── bengaluru_house_price_model.pkl  # excluded from GitHub due to file size
```

---

## ▶️ How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/ashwinijadhav21/Bengaluru-House-Price-Prediction.git
```

### 2. Navigate to the project directory

```bash
cd Bengaluru-House-Price-Prediction
```

### 3. Install required libraries

```bash
pip install pandas numpy matplotlib scikit-learn joblib jupyter
```

### 4. Open the notebook

```bash
jupyter notebook price_pred.ipynb
```

Run the notebook cells sequentially to reproduce the data preprocessing, model training, evaluation, and feature-importance analysis.

---

## 🚀 Future Improvements

Possible future improvements include:

* Hyperparameter tuning
* Cross-validation
* Trying additional regression algorithms
* Building a prediction interface using Streamlit
* Deploying the model as a web application
* Improving model generalization
* Adding more detailed exploratory data analysis

---

## 👩‍💻 Author

**Ashwini Jadhav**

Data Science Aspirant

GitHub:
https://github.com/ashwinijadhav21
