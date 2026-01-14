import streamlit as st
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗",
    layout="wide"
)

# Title and description
st.title("🚗 Car Price Prediction Model")
st.markdown("Predict car selling prices using Mileage and Age with MinMax Scaled Linear Regression")

# Load the trained model and scaler
try:
    with open('car_price_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
except FileNotFoundError:
    st.error("⚠️ Model files not found! Please run 'python regression_model.py' first.")
    st.stop()

# Sidebar for input
st.sidebar.header("📊 Input Parameters")

col1, col2 = st.columns(2)

with col1:
    mileage = st.slider(
        "Mileage (miles)",
        min_value=10000,
        max_value=100000,
        value=50000,
        step=1000
    )

with col2:
    age = st.slider(
        "Age (years)",
        min_value=1,
        max_value=15,
        value=5,
        step=1
    )

# Make prediction
input_data = np.array([[mileage, age]])
input_scaled = scaler.transform(input_data)
predicted_price = model.predict(input_scaled)[0]

# Display results
st.markdown("---")
st.subheader("📈 Prediction Results")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(label="Mileage", value=f"{mileage:,} miles")

with col2:
    st.metric(label="Age", value=f"{age} years")

with col3:
    st.metric(label="Predicted Price", value=f"${predicted_price:,.2f}")

# Model information
st.markdown("---")
st.subheader("ℹ️ Model Information")

col1, col2 = st.columns(2)

with col1:
    st.info(f"""
    **Model Details:**
    - Type: Linear Regression with MinMax Scaling
    - Scaler Range: [0, 1]
    - Mileage Coefficient: {model.coef_[0]:.6f}
    - Age Coefficient: {model.coef_[1]:.6f}
    - Intercept: ${model.intercept_:,.2f}
    """)

with col2:
    # Load training data for stats
    try:
        df = pd.read_csv('carprices (1) (1).csv')
        st.info(f"""
        **Dataset Statistics:**
        - Total Records: {len(df)}
        - Mileage Range: {df['Mileage'].min():,} - {df['Mileage'].max():,} miles
        - Age Range: {df['Age(yrs)'].min()} - {df['Age(yrs)'].max()} years
        - Price Range: ${df['Sell Price($)'].min():,} - ${df['Sell Price($)'].max():,}
        - Average Price: ${df['Sell Price($)'].mean():,.2f}
        """)
    except:
        pass

# Batch prediction
st.markdown("---")
st.subheader("📋 Batch Predictions")

if st.checkbox("Show batch prediction from CSV"):
    try:
        df = pd.read_csv('carprices (1) (1).csv')
        
        # Make predictions for all records
        X = df[['Mileage', 'Age(yrs)']].values
        X_scaled = scaler.transform(X)
        df['Predicted Price'] = model.predict(X_scaled)
        df['Actual Price'] = df['Sell Price($)']
        df['Error ($)'] = df['Actual Price'] - df['Predicted Price']
        df['Error (%)'] = (df['Error ($)'] / df['Actual Price'] * 100).round(2)
        
        # Display table
        st.dataframe(
            df[['Mileage', 'Age(yrs)', 'Actual Price', 'Predicted Price', 'Error ($)', 'Error (%)']],
            use_container_width=True
        )
        
        # Summary stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Mean Absolute Error", f"${df['Error ($)'].abs().mean():,.2f}")
        with col2:
            st.metric("Mean Error %", f"{df['Error (%)'].abs().mean():.2f}%")
        with col3:
            st.metric("Max Error $", f"${df['Error ($)'].abs().max():,.2f}")
            
    except Exception as e:
        st.error(f"Error loading data: {e}")

st.markdown("---")
st.caption("Built with Streamlit | Car Price Regression Model")
