import pandas as pd
import numpy as np

df = pd.read_csv("water_potability.csv")

new_df = pd.DataFrame()
new_df["ph"] = df["ph"]

np.random.seed(42)
new_df["temperature"] = np.random.uniform(20, 35, len(df))

new_df["turbidity"] = df["Turbidity"]
new_df["tds"] = df["Solids"] / 10

new_df = new_df.dropna()

def classify(row):
    score = 0

    if row["ph"] < 6.0 or row["ph"] > 9.0:
        score += 2
    elif row["ph"] < 6.5 or row["ph"] > 8.5:
        score += 1

    if row["turbidity"] > 10:
        score += 2
    elif row["turbidity"] > 5:
        score += 1

    if row["tds"] > 600:
        score += 2
    elif row["tds"] > 300:
        score += 1

    if score >= 4:
        return "Critical Risk"
    elif score >= 2:
        return "Moderate Risk"
    else:
        return "Low Risk"

new_df["risk_level"] = new_df.apply(classify, axis=1)

new_df.to_csv("water_quality_dataset.csv", index=False)

print("water_quality_dataset.csv created successfully")
print("Total rows:", len(new_df))
print(new_df["risk_level"].value_counts())