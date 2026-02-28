# train_model.py
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset (replace with your dataset path)
data = pd.read_csv("har_sample_561x15.csv")  # Ensure dataset CSV is ready

# Separate features and labels
X = data.drop("Activity", axis=1)
y = data["Activity"]

# Save feature names
joblib.dump(X.columns.tolist(), "features.pkl")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("Unique training labels:", y.unique())

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy*100:.2f}%")

# Save model
joblib.dump(model, "model.pkl")