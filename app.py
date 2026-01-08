from flask import Flask, render_template, request
import numpy as np
import pickle
import os

app = Flask(__name__)

# Load model and scaler
with open("knn_lung_cancer_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        # Get input values from form
        features = [
            int(request.form["gender"]),
            int(request.form["age"]),
            int(request.form["smoking"]),
            int(request.form["yellow_fingers"]),
            int(request.form["anxiety"]),
            int(request.form["peer_pressure"]),
            int(request.form["chronic_disease"]),
            int(request.form["fatigue"]),
            int(request.form["allergy"]),
            int(request.form["wheezing"]),
            int(request.form["alcohol"]),
            int(request.form["coughing"])
        ]

        features = np.array(features).reshape(1, -1)
        features_scaled = scaler.transform(features)

        result = model.predict(features_scaled)

        if result[0] == 1:
            prediction = "Lung Cancer Detected"
        else:
            prediction = "No Lung Cancer Detected"

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
