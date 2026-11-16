"""
Data Registration
------------------
Loads the raw tourism.csv that lives in the repository, validates that every
expected column is present, and prints a short summary (row/column counts and
the ProdTaken class balance). Registration is file based: the CSV lives in the
repo itself, so there is no external dataset store to manage.
"""
import pandas as pd

RAW_PATH = "tourism_project/data/tourism.csv"  # raw dataset inside the data folder

# Load the raw dataset
df = pd.read_csv(RAW_PATH)

# Validate that the expected columns are present before registering it
expected_columns = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier",
    "DurationOfPitch", "Occupation", "Gender", "NumberOfPersonVisiting",
    "NumberOfFollowups", "ProductPitched", "PreferredPropertyStar",
    "MaritalStatus", "NumberOfTrips", "Passport", "PitchSatisfactionScore",
    "OwnCar", "NumberOfChildrenVisiting", "Designation", "MonthlyIncome",
]
missing = [c for c in expected_columns if c not in df.columns]
if missing:
    raise ValueError(f"Dataset is missing expected columns: {missing}")

# Report a summary so the registration step is auditable in the workflow logs
print("Dataset registered successfully.")
print(f"Rows: {df.shape[0]}  |  Columns: {df.shape[1]}")
print("Columns:", list(df.columns))
print("\nProdTaken class balance:")
print(df["ProdTaken"].value_counts(dropna=False).to_string())
print("\nProdTaken class balance (proportion):")
print(df["ProdTaken"].value_counts(normalize=True).round(4).to_string())
