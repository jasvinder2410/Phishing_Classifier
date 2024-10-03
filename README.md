# Phishinng_classifier
This project aims to build a machine learning model to classify websites as either phishing or legitimate using various features extracted from URLs. The model is trained using supervised learning techniques and is evaluated on its ability to correctly identify phishing websites, which is crucial in cybersecurity.

__Project Overview__
Phishing is a type of cyber attack where attackers trick users into providing sensitive information by mimicking legitimate websites. To combat this, this project builds a classification model that can predict whether a website is phishing or legitimate based on features such as URL structure, domain name, etc.

__Key Features:__
Data Preprocessing: Data cleaning, handling missing values, and feature engineering.
Modeling: Training multiple models using RandomizedSearchCV and selecting the best one based on performance metrics.
Class Imbalance Handling: Addressing imbalance in the dataset using SMOTE (Synthetic Minority Over-sampling Technique).
Evaluation: Evaluating model performance using metrics like Accuracy, Recall.

__Project Structure__
Phishing.ipynb: The main notebook containing data preprocessing, model training, evaluation, and results.
web-page-phishing.csv: The dataset used for training and testing the model.
model.pkl: The saved trained model for future predictions.
README.md: This file, providing an overview of the project.

__Requirements__
The project uses the following Python libraries:
numpy
pandas
scikit-learn
xgboost
matplotlib
seaborn
imblearn

__Dataset__
The dataset used in this project consists of various features related to the structure and characteristics of URLs. Some key features include:
URL Length
Has IP Address in URL
Number of Dots
HTTPS token
Domain Age
The target variable is binary, indicating whether the website is phishing (1) or legitimate (0).

__Model Training and Evaluation__
The following machine learning models were trained and evaluated:
Logistic Regression
XGBoost Classifier

The best-performing model was XGBoost, which achieved:
Accuracy: 89%
Recall: 89%
The final model was saved using the pickle module for later predictions.

__Feature Scaling:__
Features were standardized using StandardScaler before training to ensure better performance.

__Handling Class Imbalance:__
Class imbalance was addressed using SMOTE, which oversampled the minority class (phishing websites) to create a more balanced dataset.

__Metrics:__
The model was evaluated using:
Accuracy: Overall correctness of predictions.
Recall: Focused on correctly identifying phishing websites.

__Results:__
The model showed high performance in identifying phishing websites with the following key metrics:
Accuracy: 89%
Recall: 89%
