import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import joblib
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("model/water_quality_model.keras")
scaler = joblib.load("model/scaler.pkl")
encoder = joblib.load("model/label_encoder.pkl")

sample = np.array([[7.2, 26.5, 2.1, 180]])

sample_scaled = scaler.transform(sample)

prediction = model.predict(sample_scaled)

class_index = prediction.argmax(axis=1)[0]
confidence = prediction[0][class_index] * 100

risk = encoder.inverse_transform([class_index])[0]

if confidence >= 80:
    status = "High Confidence"
elif confidence >= 60:
    status = "Medium Confidence"
else:
    status = "Low Confidence"

print("Prediction:", risk)
print("Confidence:", round(confidence, 2), "%")
print("Status:", status)

print("\nClass Probabilities:")
for label, prob in zip(encoder.classes_, prediction[0]):
    print(f"{label}: {prob * 100:.2f}%")