import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset without header
df = pd.read_csv("data/har_dataset_combined.csv", header=None)

# Save as zip (optional, reduces size for GitHub)
df.to_csv("data/har_dataset_combined.zip", index=False, header=False, compression='zip')

# Features & Label
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

print("Dataset Shape:", data.shape)
print("Activities:", y.unique())

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Accuracy
pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))

# Save
joblib.dump(model, "model.pkl")
joblib.dump(X.columns.tolist(), "features.pkl")


print("Training Completed")
