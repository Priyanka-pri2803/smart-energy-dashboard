from preprocess import load_and_preprocess
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Preprocess data
X_train, X_test, y_train, y_test, preprocessor, feature_cols = load_and_preprocess()

#  Debugging: Print the shape of the training and test data
print("X_train shape:", X_train.shape)  # Print the number of rows and columns in X_train
print("X_test shape:", X_test.shape)    # Print the number of rows and columns in X_test
print("Training model now...")          # Confirm model training is about to start

#  Step 2: Train the model
print(" Training RandomForest model...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

#  Step 3: Evaluate model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n Model Evaluation Results:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R² Score: {r2:.2f}")

#  Step 4: Save model + preprocessor
joblib.dump(model, "energy_model.pkl")
joblib.dump(preprocessor, "preprocessor.pkl")  # Save the preprocessor instead of just the scaler
print("Model and preprocessor saved successfully!")

# Step 5: Feature Importance Plot
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Feature Importance in Energy Consumption Prediction")
plt.bar(range(len(feature_cols)), importances[indices], align="center")
plt.xticks(range(len(feature_cols)), np.array(feature_cols)[indices], rotation=90)
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()

print("Feature importance plot saved as 'feature_importance.png'")
