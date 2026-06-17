import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import json
import joblib
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, BackupAndRestore


# ==============================
# FILE SETTINGS
# ==============================
CSV_FILE = "aquaintelx_measurements_separated.csv"

# Backup if your file still has the old name
if not os.path.exists(CSV_FILE):
    CSV_FILE = "water_quality_dataset.csv"

MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)


# ==============================
# LOAD DATASET
# ==============================
print("Loading dataset:", CSV_FILE)

df = pd.read_csv(CSV_FILE)
df.columns = df.columns.str.strip()

# Optional auto-rename for common column names
df = df.rename(columns={
    "pH": "ph",
    "PH": "ph",
    "Temp": "temperature",
    "Temperature": "temperature",
    "Water Temperature": "temperature",
    "Turbidity": "turbidity",
    "Turbidity (cm)": "turbidity",
    "TDS": "tds",
    "Total Dissolved Solids": "tds",
    "Risk Level": "risk_level",
    "Water Quality": "risk_level",
    "Target": "risk_level",
    "Potability": "risk_level"
})

required_columns = ["ph", "temperature", "turbidity", "tds", "risk_level"]

for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Missing column: {col}")

df = df[required_columns].copy()

df["ph"] = pd.to_numeric(df["ph"], errors="coerce")
df["temperature"] = pd.to_numeric(df["temperature"], errors="coerce")
df["turbidity"] = pd.to_numeric(df["turbidity"], errors="coerce")
df["tds"] = pd.to_numeric(df["tds"], errors="coerce")

df = df.dropna()

print("Dataset shape:", df.shape)
print("\nRisk level counts:")
print(df["risk_level"].value_counts())


# ==============================
# FEATURES AND LABEL
# ==============================
X = df[["ph", "temperature", "turbidity", "tds"]]
y = df["risk_level"]


# ==============================
# ENCODE LABELS
# ==============================
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

print("\nDetected risk labels:")
for i, label in enumerate(encoder.classes_):
    print(i, "=", label)


# ==============================
# SCALE FEATURES
# ==============================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ==============================
# SPLIT DATA
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)


# ==============================
# MODEL
# ==============================
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(4,)),

    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(16, activation="relu"),

    tf.keras.layers.Dense(len(encoder.classes_), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ==============================
# CALLBACKS
# ==============================
checkpoint_callback = ModelCheckpoint(
    filepath=os.path.join(MODEL_DIR, "best_water_quality_model.keras"),
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)

backup_callback = BackupAndRestore(
    backup_dir=os.path.join(MODEL_DIR, "training_backup")
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

callbacks = [
    checkpoint_callback,
    backup_callback,
    early_stop
]


# ==============================
# TRAIN MODEL
# ==============================
print("\nTraining model...")

history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=512,
    callbacks=callbacks,
    verbose=1
)


# ==============================
# SAVE LEARNING CURVES
# ==============================
plt.figure()
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.savefig(os.path.join(MODEL_DIR, "accuracy_curve.png"))
plt.close()

plt.figure()
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.savefig(os.path.join(MODEL_DIR, "loss_curve.png"))
plt.close()

print("\nLearning curve graphs saved:")
print("model/accuracy_curve.png")
print("model/loss_curve.png")


# ==============================
# EVALUATE MODEL
# ==============================
print("\nEvaluating model...")

y_pred_prob = model.predict(X_test)
y_pred = y_pred_prob.argmax(axis=1)

accuracy = accuracy_score(y_test, y_pred)

print("\nTest Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=encoder.classes_
))

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

print("\nLabel order:")
print(encoder.classes_)

cm_df = pd.DataFrame(
    cm,
    index=[f"Actual {label}" for label in encoder.classes_],
    columns=[f"Predicted {label}" for label in encoder.classes_]
)

cm_path = os.path.join(MODEL_DIR, "confusion_matrix.csv")
cm_df.to_csv(cm_path)

print("\nConfusion matrix saved:")
print("model/confusion_matrix.csv")


# ==============================
# SAVE FINAL MODEL FILES
# ==============================
model.save(os.path.join(MODEL_DIR, "water_quality_model.keras"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
joblib.dump(encoder, os.path.join(MODEL_DIR, "label_encoder.pkl"))

with open(os.path.join(MODEL_DIR, "labels.json"), "w") as file:
    json.dump(list(encoder.classes_), file, indent=4)

print("\nModel saved successfully.")
print("Saved files:")
print("model/water_quality_model.keras")
print("model/best_water_quality_model.keras")
print("model/scaler.pkl")
print("model/label_encoder.pkl")
print("model/labels.json")
print("model/accuracy_curve.png")
print("model/loss_curve.png")
print("model/confusion_matrix.csv")