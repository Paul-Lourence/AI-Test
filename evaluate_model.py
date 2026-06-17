import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import joblib
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split

df = pd.read_csv("water_quality_dataset.csv")
df = df.dropna()

X = df[["ph", "temperature", "turbidity", "tds"]]
y = df["risk_level"]

scaler = joblib.load("model/scaler.pkl")
encoder = joblib.load("model/label_encoder.pkl")
model = tf.keras.models.load_model("model/water_quality_model.keras")

X_scaled = scaler.transform(X)
y_encoded = encoder.transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

predictions = model.predict(X_test)
y_pred = predictions.argmax(axis=1)

print(classification_report(
    y_test,
    y_pred,
    target_names=encoder.classes_
))

cm = confusion_matrix(y_test, y_pred)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=encoder.classes_
)

display.plot()
plt.title("AquaIntelX Confusion Matrix")
plt.show()