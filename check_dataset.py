import pandas as pd

df = pd.read_csv("water_quality_dataset.csv")

print("Dataset Preview:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nRisk Level Count:")
print(df["risk_level"].value_counts())

print("\nBasic Statistics:")
print(df.describe())