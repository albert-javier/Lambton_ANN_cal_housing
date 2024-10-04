import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Load the California housing dataset
cal_data = fetch_california_housing()
df = pd.DataFrame(cal_data.data, columns=cal_data.feature_names)
df['Price'] = cal_data.target

# Split data into features and target
X = df.drop(['Price'], axis=1)
y = df['Price']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

# Define regression models
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(),
    "Lasso Regression": Lasso(),
    "ElasticNet Regression": ElasticNet()
}

# Streamlit app
st.title("California Housing Price Prediction App")

# Select regression model
model_name = st.selectbox("Select Regression Model", list(models.keys()))
model = models[model_name]

# Train the selected model
model.fit(X_train_sc, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_sc)

# Evaluate the model
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# Display evaluation metrics
st.subheader("Model Evaluation")
st.write(f"R-squared: {r2:.4f}")
st.write(f"Mean Absolute Error: {mae:.4f}")
st.write(f"Mean Squared Error: {mse:.4f}")
st.write(f"Root Mean Squared Error: {rmse:.4f}")

# Input features for prediction
st.subheader("Predict House Price")
input_features = {}
for feature in X.columns:
    # input_features[feature] = st.number_input(feature)
    input_features[feature] = st.slider(feature, min_value=X[feature].min(), max_value=X[feature].max(), value=X[feature].mean() )
    
    

# Create a DataFrame from input features
input_df = pd.DataFrame([input_features])


if st.button("Predict"):
    
    # Scale input features
    input_df_sc = scaler.transform(input_df)

    # Make prediction
    prediction = model.predict(input_df_sc)

    # Display prediction
    st.write(f"Predicted House Price: ${prediction[0]*100000:.2f}")

else:
    st.write("Click 'Predict' to get the prediction.")

