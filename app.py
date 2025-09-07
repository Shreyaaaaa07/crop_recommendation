from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import os

# Load model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "crop_recommendation_model.pkl")
model = joblib.load(MODEL_PATH)

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.form
        features = np.array([[
            float(data["N"]), float(data["P"]), float(data["K"]),
            float(data["temperature"]), float(data["humidity"]),
            float(data["ph"]), float(data["rainfall"])
        ]])
        prediction = model.predict(features)[0]
        return render_template("result.html", crop=prediction)
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
# Ensure you have the required packages installed:
# pip install flask joblib numpy



