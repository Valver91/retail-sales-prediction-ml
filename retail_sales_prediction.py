# Predicting Retail Store Sales Using Machine Learning
# Author: Santiago FV

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras import Sequential
from keras.layers import Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error

# -------------------------------
# 1. Load Dataset
# -------------------------------

# Dataset: Sales data for a chain of retail stores
# It contains features like number of customers, promotions, open days, seasons, etc.

# Load CSV file (Make sure the CSV file is in the same directory)
dataset = pd.read_csv('#')

print(f"Dataset shape: {dataset.shape}")
print(dataset.head())

# -------------------------------
# 2. Data Preparation
# -------------------------------

# Separate target variable ('Sales') and features
X = dataset.drop('Sales', axis=1)
y = dataset[['Sales']]

# Split into train/test sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Further split training data into train/validation (90/10 split)
X_train_final, X_val, y_train_final, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

print(f"Training set: {X_train_final.shape}, Validation set: {X_val.shape}, Test set: {X_test.shape}")

# -------------------------------
# 3. Baseline Model
# -------------------------------

# Calculate mean sales and use as naive prediction
mean_sales = y_train_final['Sales'].mean()
y_pred_baseline = [mean_sales] * len(y_test)
mae_baseline = mean_absolute_error(y_test, y_pred_baseline)
print(f"Baseline MAE (using mean sales as prediction): {mae_baseline:.2f}")

# -------------------------------
# 4. Model Building: Deep Neural Network (MLP)
# -------------------------------

model = Sequential()

# Input layer + hidden layers
model.add(Dense(350, activation='relu', input_dim=X_train_final.shape[1]))
model.add(Dense(350, activation='relu'))
model.add(Dense(350, activation='relu'))
model.add(Dense(350, activation='relu'))
model.add(Dense(350, activation='relu'))

# Output layer
model.add(Dense(1, activation='linear'))

# Compile the model
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])

# -------------------------------
# 5. Model Training
# -------------------------------

history = model.fit(X_train_final, y_train_final, validation_data=(X_val, y_val), epochs=15, batch_size=16)

# -------------------------------
# 6. Model Evaluation
# -------------------------------

results = model.evaluate(X_test, y_test, return_dict=True)
print("\nTest Evaluation:")
for metric, value in results.items():
    print(f"{metric}: {value:.2f}")

# -------------------------------
# 7. Visualization: Loss Curves
# -------------------------------

plt.figure(figsize=(10,6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss During Training')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.show()

# -------------------------------
# 8. Predictions & Results
# -------------------------------

predictions = model.predict(X_test)

comparison_df = pd.DataFrame({
    'Actual Sales': y_test.values.flatten(),
    'Predicted Sales': predictions.flatten()
})
print("\nSample Predictions:")
print(comparison_df.head(15))

# Calculate final metrics
mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
print(f"\nFinal Test MSE: {mse:.2f}")
print(f"Final Test MAE: {mae:.2f}")
print(f"Mean Actual Sales: {y_test['Sales'].mean():.2f}")

# -------------------------------
# END
# -------------------------------
