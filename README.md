Insurance Cost Prediction App (Linear Regression)
📘 Overview

This project predicts medical insurance charges based on demographic and health-related features such as age, BMI, number of children, smoking habits, and region.
It uses Linear Regression to model the relationship between these variables and insurance costs.

This app is built with Streamlit and includes:

📊 Outlier detection and removal

⚙️ Automatic data preprocessing & encoding

🧠 Model training and evaluation

👨‍💻 Manual input prediction

📂 CSV upload for batch predictions

📈 Interactive visualizations

📁 Dataset Information

Dataset Name: insurance.csv
Columns:

Column	Description
age	Age of primary beneficiary
sex	Gender (male, female)
bmi	Body Mass Index (weight/height²)
children	Number of dependents covered by insurance
smoker	Smoking status (yes, no)
region	Residential area (northeast, northwest, southeast, southwest)
charges	Individual medical insurance cost (target variable)
🧠 Machine Learning Workflow

Data Loading – Load insurance.csv and display a quick summary

EDA (Exploratory Data Analysis) – Visualize distributions and correlations

Outlier Detection – Use Z-score method to detect and remove outliers

Encoding – Convert categorical variables to numeric:

sex: female → 0, male → 1

smoker: no → 0, yes → 1

region: One-hot encoding (drop_first=True)

Model Training – Train a Linear Regression model

Evaluation – Display:

Mean Absolute Error (MAE)

Root Mean Squared Error (RMSE)

R² Score

Prediction Options –

Enter values manually

Upload a test CSV for bulk predictions
