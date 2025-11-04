# ============================================================
# 💰 Insurance Cost Prediction using Linear Regression
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder

# ============================================================
# 🧩 Streamlit Page Configuration
# ============================================================
st.set_page_config(page_title="Insurance Cost Prediction", page_icon="💰", layout="wide")

st.title("💰 Insurance Cost Prediction App")
st.markdown("Predict individual medical insurance charges using **Linear Regression**.")

# ============================================================
# 📂 Load Dataset Automatically
# ============================================================
@st.cache_data
def load_data():
    df = pd.read_csv("insurance.csv")
    return df

df = load_data()
st.success("✅ Training dataset loaded successfully!")

# ============================================================
# 🔍 Exploratory Data Analysis
# ============================================================
with st.expander("📊 View Dataset Overview"):
    st.write(df.head())
    st.write("Shape:", df.shape)
    st.write(df.describe())

# ============================================================
# 🚨 Outlier Detection & Removal
# ============================================================
numeric_cols = ['age', 'bmi', 'children', 'charges']
z = np.abs(stats.zscore(df[numeric_cols]))
df_clean = df[(z < 3).all(axis=1)]
removed = df.shape[0] - df_clean.shape[0]

st.info(f"🧹 Removed {removed} outliers using Z-score method.")

# Boxplot before and after
col1, col2 = st.columns(2)
with col1:
    st.write("Before Outlier Removal")
    fig1, ax1 = plt.subplots()
    sns.boxplot(data=df[numeric_cols], ax=ax1)
    st.pyplot(fig1)

with col2:
    st.write("After Outlier Removal")
    fig2, ax2 = plt.subplots()
    sns.boxplot(data=df_clean[numeric_cols], ax=ax2)
    st.pyplot(fig2)

# ============================================================
# ⚙️ Data Preprocessing (Encoding)
# ============================================================
df_clean['sex'] = df_clean['sex'].map({'female': 0, 'male': 1})
df_clean['smoker'] = df_clean['smoker'].map({'no': 0, 'yes': 1})
df_clean = pd.get_dummies(df_clean, columns=['region'], drop_first=True)

# ============================================================
# 🧠 Model Training
# ============================================================
X = df_clean.drop('charges', axis=1)
y = df_clean['charges']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# ============================================================
# 📈 Model Evaluation
# ============================================================
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

st.subheader("📊 Model Evaluation")
st.write(f"**Mean Absolute Error (MAE):** {mae:.2f}")
st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.2f}")
st.write(f"**R² Score:** {r2:.2f}")

# Visualization: Actual vs Predicted
fig3, ax3 = plt.subplots()
sns.scatterplot(x=y_test, y=y_pred, alpha=0.7, color="blue")
plt.xlabel("Actual Charges")
plt.ylabel("Predicted Charges")
plt.title("Actual vs Predicted Insurance Charges")
st.pyplot(fig3)

# ============================================================
# 🔢 Prediction Options
# ============================================================
st.subheader("🔮 Make Predictions")

option = st.radio("Choose Prediction Mode:", ["Manual Input", "Upload CSV File"])

# ------------- Manual Input -------------------
if option == "Manual Input":
    st.markdown("Enter the details below:")

    age = st.slider("Age", 18, 100, 30)
    sex = st.radio("Sex", ["male", "female"])
    bmi = st.number_input("BMI", 10.0, 50.0, 25.0)
    children = st.slider("Children", 0, 5, 0)
    smoker = st.radio("Smoker", ["yes", "no"])
    region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

    # Encode manually entered data
    input_data = pd.DataFrame({
        'age': [age],
        'sex': [1 if sex == 'male' else 0],
        'bmi': [bmi],
        'children': [children],
        'smoker': [1 if smoker == 'yes' else 0],
        'region_northwest': [1 if region == 'northwest' else 0],
        'region_southeast': [1 if region == 'southeast' else 0],
        'region_southwest': [1 if region == 'southwest' else 0],
    })

    if st.button("Predict Insurance Cost"):
        pred = model.predict(input_data)[0]
        st.success(f"💰 Predicted Insurance Charges: **${pred:,.2f}**")

# ------------- CSV Upload -------------------
else:
    uploaded_file = st.file_uploader("Upload a CSV file for prediction", type=["csv"])
    if uploaded_file is not None:
        test_df = pd.read_csv(uploaded_file)

        st.write("📄 Uploaded File Preview:")
        st.write(test_df.head())

        # Encoding
        test_df['sex'] = test_df['sex'].map({'female': 0, 'male': 1})
        test_df['smoker'] = test_df['smoker'].map({'no': 0, 'yes': 1})
        test_df = pd.get_dummies(test_df, columns=['region'], drop_first=True)

        # Ensure same feature alignment
        missing_cols = set(X.columns) - set(test_df.columns)
        for c in missing_cols:
            test_df[c] = 0
        test_df = test_df[X.columns]

        preds = model.predict(test_df)
        test_df['Predicted_Charges'] = preds

        st.write("✅ Predictions Completed!")
        st.dataframe(test_df.head())

        csv = test_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Predictions", data=csv, file_name="insurance_predictions.csv")

# ============================================================
# 🎯 End
# ============================================================
st.markdown("---")
st.caption("Developed with ❤️ using Streamlit and scikit-learn.")
