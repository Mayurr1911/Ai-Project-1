import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib
import numpy as np

# Generate synthetic stock data for demonstration
np.random.seed(42)
dates = pd.date_range(start='2020-01-01', periods=1000, freq='D')
prices = np.cumsum(np.random.randn(1000)) + 100  # Random walk
data = pd.DataFrame({'Date': dates, 'Price': prices})

# Feature engineering: use previous prices as features
data['Prev_Price'] = data['Price'].shift(1)
data = data.dropna()

X = data[['Prev_Price']]
y = data['Price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse:.2f}")

# Save model
joblib.dump(model, 'stock_prediction_model.pkl')
