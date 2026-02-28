from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load model & features
model = joblib.load("model.pkl")
features = joblib.load("features.pkl")

activity_labels = {
    0: "Walking",
    1: "Walking Upstairs",
    2: "Walking Downstairs",
    3: "Sitting",
    4: "Standing",
    5: "Laying"
}

@app.route("/", methods=["GET", "POST"])
def index():
    predictions = None
    if request.method == "POST":
        # Check if file part is present
        if "file" not in request.files:
            return "No file uploaded", 400
        file = request.files["file"]
        if file.filename == "":
            return "No file selected", 400

        # Read uploaded CSV
        df = pd.read_csv(file)

        # Keep only the features used for training
        df_features = df[features]

        # Predict
        preds = model.predict(df_features)
        # Use predictions directly (they are already strings)
        predictions = preds.tolist()
        
        # Add predictions to dataframe
        df["Predicted_Activity"] = predictions

        # Select 10 random rows
        df_random10 = df.sample(n=10, random_state=42)  # random_state ensures reproducibility

        # Save only these 10 rows to CSV
        df_random10.to_csv("predictions.csv", index=False)

    return render_template("index.html", predictions=predictions)
if __name__ == "__main__":
    app.run(debug=True)


    