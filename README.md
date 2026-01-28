# CodSoft-DataScience-Internship

# Task 1 Titanic Survival Prediction

## Project Overview
This project predicts the survival of passengers on the Titanic using Machine Learning techniques.
The model is trained on the Kaggle Titanic dataset.

## Problem Statement
The objective of this project is to build a machine learning model that predicts whether a passenger survived the Titanic disaster based on features such as age, gender, class, and fare.

##Dataset Information
Dataset Name: Titanic Dataset
Source: Kaggle
Total Records: 891 passengers
Target Variable: Survived (0 = No, 1 = Yes)

🔑 Features Used

Pclass – Passenger class
Sex – Gender
Age – Passenger age
SibSp – Number of siblings/spouses
Parch – Number of parents/children
Fare – Ticket fare

## Technologies Used
- Python
- Google Colab
- Pandas
- NumPy
- Matplotlib / Seaborn
- Scikit-learn
- Version Control: Git & GitHub

**  ##Machine Learning Algorithms**

The following models were implemented and evaluated:
-Logistic Regression
-Decision Tree Classifier
-Random Forest Classifier

## Steps Performed
1. Dataset loading and exploration
2. Data preprocessing and cleaning
3. Handling missing values
4. Feature encoding
5. Train-test split
6. Model training (Logistic Regression / Decision Tree)
7. Model evaluation using accuracy and confusion matrix


   ## How to Run
1. Clone the repository
2. Open the notebook in Google Colab
3. Upload the dataset (train.csv)
4. Run all cells sequentially


## Repository Contents
titanic_survival.ipynb – Complete Google Colab notebook
train.csv – Dataset file
README.md – Project documentation

## Result
The trained model successfully predicts passenger survival with reasonable accuracy.

## Author
KrishnaBhargavi



📌Task 2 - Movie Rating Prediction using python

## Project Overview

-The Movie Rating Prediction project focuses on predicting the rating of a movie based on its features such as genre, director, actors, and other attributes using machine learning regression techniques.

-This project demonstrates the complete data science workflow including data preprocessing, feature encoding, model training, and evaluation using Python.

## Objective

-To build a regression-based machine learning model that can accurately predict movie ratings from historical movie data.

## Dataset Information

-Dataset Type: CSV file
-File Name: movies.csv
-Content: Movie-related attributes and their corresponding ratings
-Target Variable: rating (continuous numeric value)

## Example Features
-Genre
-Director
-Cast / Actors
-Other movie attributes (based on dataset)

## Technologies & Tools Used

-Programming Language: Python
-Platform: Google Colab
-Libraries:
-Pandas
-NumPy
-Matplotlib
-Seaborn
-Scikit-learn
--Version Control: Git & GitHub

## Machine Learning Approach

-Problem Type: Regression
-Algorithm Used: Linear Regression

##Project Workflow

-Dataset loading and inspection
-Exploratory data analysis
-Handling missing values
-Encoding categorical variables
-Feature selection
-Train–test split
-Model training
-Rating prediction
-Model evaluation using regression metrics

##Model Evaluation Metrics

-Mean Squared Error (MSE) – Measures average prediction error
-R² Score – Indicates how well the model explains variance in movie ratings

##Results & Observations

-Categorical feature encoding played a key role in model performance
-Proper data cleaning improved prediction accuracy
-The regression model successfully learned patterns in movie rating data

##Conclusion

-This project shows how machine learning regression techniques can be effectively applied to predict movie ratings.
-With proper preprocessing and feature engineering, predictive models can provide meaningful insights into factors that influence movie ratings.

##Future Enhancements

-Use advanced models like Random Forest Regressor
-Perform hyperparameter tuning
-Add feature importance analysis
-Improve prediction accuracy with more data

## Author
KrishnaBhargavi




📌 Task-3 Sales Prediction using Python
 ## Project Overview

-The Sales Prediction project focuses on forecasting product sales based on historical data using Machine Learning techniques. Accurate sales prediction helps businesses in inventory management, revenue planning, and decision-making.

-This project demonstrates the complete data science workflow, including data preprocessing, exploratory data analysis, model training, and performance evaluation.

##🎯 Problem Statement

-The objective of this project is to build a machine learning model that predicts future sales based on factors such as advertising spend, product attributes, or historical sales patterns.

##📊 Dataset Information

-Dataset Type: CSV file
-Dataset Name: Sales Dataset
-Target Variable: Sales (continuous numeric value)
-Example Features
-Advertising spend (TV, Radio, Newspaper)
-Product / Store attributes
-Time-based features (if available)
-Other relevant independent variables

##🛠 Technologies & Tools Used

-Programming Language: Python
-Platform: Google Colab
-Libraries:
-Pandas
-NumPy
-Matplotlib
-Seaborn
-Scikit-learn
-Version Control: Git & GitHub

##🤖 Machine Learning Approach

-Problem Type: Regression
-Algorithms Used:
-Linear Regression
-(Optional) Decision Tree Regressor
-(Optional) Random Forest Regressor

##🔄 Project Workflow

-Dataset loading and inspection
-Exploratory Data Analysis (EDA)
-Data cleaning and preprocessing
-Handling missing values
-Feature selection
-Train–test split
-Model training
-Sales prediction
-Model evaluation

##📈 Model Evaluation Metrics

-Mean Squared Error (MSE) – Measures average prediction error
-Mean Absolute Error (MAE) – Measures average absolute error
-R² Score – Indicates how well the model explains variance in sales

##📂 Repository Contents

-sales_prediction.ipynb – Complete Google Colab notebook
-sales.csv – Dataset file
-README.md – Project documentation

##▶️ How to Run the Project

-Clone the repository
-Open the notebook in Google Colab
-Upload the dataset (sales.csv)
-Run all cells sequentially

##✅ Results

-The trained regression model successfully predicts sales with reasonable accuracy, showing the effectiveness of data preprocessing and regression techniques.

##🧾 Conclusion

-This project demonstrates how machine learning regression models can be used for sales forecasting. Proper feature selection and data preprocessing play a crucial role in improving prediction accuracy.

##🔮 Future Enhancements

-Use advanced regression models (XGBoost, Random Forest)
-Perform hyperparameter tuning
-Add feature importance analysis
-Incorporate time-series forecasting techniques
-Improve accuracy with larger datasets


## Author
KrishnaBhargavi



 📌  Task 4 Credit Card Fraud Detection using Machine Learning

## 📌 Project Overview
This project focuses on detecting fraudulent credit card transactions using Machine Learning techniques. Due to the highly imbalanced nature of fraud datasets, effective modeling and evaluation are crucial to accurately identify fraudulent activities while minimizing false positives.

The project uses the **Credit Card Fraud Detection dataset** and implements a **Random Forest Classifier** to classify transactions as legitimate or fraudulent.

---

## 🎯 Objective
- To build a machine learning model that can accurately detect fraudulent credit card transactions.
- To analyze model performance using appropriate evaluation metrics such as Precision, Recall, F1-score, Confusion Matrix, and ROC-AUC score.

---

## 📂 Dataset Information
- **Dataset Name:** Credit Card Fraud Detection
- **Source:** Kaggle
- **Total Transactions:** 284,807
- **Fraudulent Transactions:** 492
- **Legitimate Transactions:** 284,315
- **Features:** 30 numerical features (`V1` to `V28`, `Time`, `Amount`)
- **Target Variable:** `Class`
  - `0` → Legitimate Transaction  
  - `1` → Fraudulent Transaction

> Note: The dataset is highly imbalanced, making fraud detection a challenging task.

---

### Key Characteristics:
- Highly imbalanced dataset
- Fraud cases represent a very small percentage of total transactions
- Requires special evaluation metrics beyond accuracy

---

## 🛠 Tools & Technologies Used
- **Programming Language**: Python  
- **Platform**: Google Colab  
- **Version Control**: GitHub  
- **Libraries**:
  - NumPy
  - Pandas
  - Matplotlib
  - Seaborn
  - Scikit-learn
  - Imbalanced-learn (SMOTE)

---

## ⚙️ Methodology
1. Loaded and explored the dataset.
2. Performed feature–target separation.
3. Split the dataset into training and testing sets using an 80:20 ratio with stratification.
4. Trained a **Random Forest Classifier** with optimized hyperparameters.
5. Generated predictions on the test dataset.
6. Evaluated the model using standard classification metrics.

---

## 🤖 Machine Learning Model
- **Random Forest Classifier**
  - Handles non-linearity well
  - Performs effectively on imbalanced datasets
  - Reduces overfitting using ensemble learning

---

## 📊 Evaluation Metrics
The model performance was evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC–AUC Score

These metrics provide a comprehensive understanding of the model’s ability to detect fraudulent transactions.

---

## 📈 Results
- The Random Forest model achieved high accuracy.
- Precision and recall values indicate effective fraud detection.
- Confusion Matrix visualization highlights correct and incorrect classifications.
- ROC-AUC score demonstrates strong discriminative ability between fraud and non-fraud transactions.

---

## 📌 Conclusion
The Random Forest model proved to be effective in detecting fraudulent credit card transactions. Despite the class imbalance, the model achieved strong performance across evaluation metrics. This approach can assist financial institutions in reducing fraud-related losses.

---

## 🚀 Future Enhancements
- Apply SMOTE or other resampling techniques to handle class imbalance.
- Compare performance with other models such as Logistic Regression, XGBoost, or Neural Networks.
- Deploy the model as a web application for real-time fraud detection.

---

## 📁 Repository Contents
- `Credit_Card_Fraud_Detection.ipynb` – Complete implementation
- `creditcard.csv` – Dataset (optional)
- `README.md` – Project documentation

---

## 👤 Author
**Krishna Bhargavi**

 
