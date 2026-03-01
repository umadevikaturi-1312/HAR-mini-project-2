from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load model & features
model = joblib.load("model.pkl")
features = joblib.load("features.pkl")

# Print features
print("Number of features:", len(features))
print(features[:10])  # first 10 feature names

activity_labels = {
    1: "Walking",
    2: "Walking Upstairs",
    3: "Walking Downstairs",
    4: "Sitting",
    5: "Standing",
    6: "Laying"
}

@app.route("/", methods=["GET", "POST"])
def index():
    predictions = None
    if request.method == "POST":
        if "file" not in request.files:
            return "No file uploaded", 400
        file = request.files["file"]
        if file.filename == "":
            return "No file selected", 400

        # Read uploaded CSV without headers
        df = pd.read_csv(file, header=None)

        # Make sure all feature columns exist
        if len(features) != df.shape[1]:
            for i in range(len(features) - df.shape[1]):
                df[i + df.shape[1]] = 0

        df.columns = features  # align column names with training features

        # Predict
        preds = model.predict(df)
        # Map numeric predictions to activity names
        predictions = [activity_labels[p] for p in preds]

        df["Predicted_Activity"] = predictions
        # Keep only the last 10 rows
        df_last10 = df.tail(10)
        df_last10.to_csv("predictions.csv", index=False)

    return render_template("index.html", predictions=predictions)

if __name__ == "__main__":
    app.run(debug=True)


    