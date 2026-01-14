import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle
import warnings
warnings.filterwarnings('ignore')

# Load the data
df = pd.read_csv('carprices (1) (1).csv')

print("Dataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())
print("\nDataset info:")
print(df.info())
print("\nStatistical summary:")
print(df.describe())

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Prepare features and target
X = df[['Mileage', 'Age(yrs)']].values
y = df['Sell Price($)'].values

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Apply MinMax Scaler
scaler = MinMaxScaler(feature_range=(0, 1))
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Make predictions
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

# Evaluate the model
print("\n" + "="*50)
print("MODEL PERFORMANCE")
print("="*50)

train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
train_rmse = np.sqrt(train_mse)
test_rmse = np.sqrt(test_mse)
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f"\nTraining Metrics:")
print(f"  MAE: ${train_mae:,.2f}")
print(f"  RMSE: ${train_rmse:,.2f}")
print(f"  R² Score: {train_r2:.4f}")

print(f"\nTest Metrics:")
print(f"  MAE: ${test_mae:,.2f}")
print(f"  RMSE: ${test_rmse:,.2f}")
print(f"  R² Score: {test_r2:.4f}")

print(f"\nModel Coefficients:")
print(f"  Mileage coefficient: {model.coef_[0]:.6f}")
print(f"  Age coefficient: {model.coef_[1]:.6f}")
print(f"  Intercept: ${model.intercept_:,.2f}")

# Save the model and scaler
with open('car_price_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("\n✓ Model and scaler saved successfully!")
print("✓ Ready to run Streamlit app: streamlit run app.py")
