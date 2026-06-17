import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from flask import Flask, request, jsonify
import numpy as np
import joblib
import tensorflow as tf
from flask_cors import CORS

CORS(app)

app = Flask(__name__)

model = tf.keras.models.load_model("model/water_quality_model.keras")
scaler = joblib.load("model/scaler.pkl")
encoder = joblib.load("model/label_encoder.pkl")

@app.route("/")
def home():
    return "AquaIntelX AI API is running"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        ph = float(data["ph"])
        temperature = float(data["temperature"])
        turbidity = float(data["turbidity"])
        tds = float(data["tds"])

        sample = np.array([[ph, temperature, turbidity, tds]])
        sample_scaled = scaler.transform(sample)

        prediction = model.predict(sample_scaled)
        class_index = prediction.argmax(axis=1)[0]

        risk_level = encoder.inverse_transform([class_index])[0]
        confidence = float(prediction[0][class_index] * 100)

        if confidence >= 80:
            status = "High Confidence"
        elif confidence >= 60:
            status = "Medium Confidence"
        else:
            status = "Low Confidence"

        return jsonify({
            "ph": ph,
            "temperature": temperature,
            "turbidity": turbidity,
            "tds": tds,
            "risk_level": risk_level,
            "confidence": round(confidence, 2),
            "status": status
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 400


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)