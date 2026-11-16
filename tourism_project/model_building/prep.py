"""
Data Preparation
----------------
Loads the registered CSV, performs light cleaning (drops the non-predictive
CustomerID column and fixes the 'Fe Male' data-entry typo in Gender), and
splits features/target into stratified train and test sets. The four CSVs are
written to the working directory so the GitHub Actions workflow can pass them
to the model-training job as an artifact.

NOTE: categorical columns are intentionally left as raw strings. The training
pipeline one-hot-encodes them, and the Streamlit app also sends raw category
values, so training and serving stay consistent.
"""
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("tourism_project/data/tourism.csv")

# Drop the unnamed index column if the CSV was saved with one
if df.columns[0].startswith("Unnamed"):
    df = df.drop(columns=df.columns[0])

# Drop the customer identifier: it is not a predictive feature
df.drop(columns=["CustomerID"], inplace=True)

# Fix an obvious data-entry typo so 'Fe Male' and 'Female' are one category
df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})

target = "ProdTaken"  # 1 if the customer purchased the package, else 0
X = df.drop(columns=[target])
y = df[target]

# Stratify on the target to preserve the class imbalance in both splits
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Data preparation complete.")
print(f"Xtrain: {Xtrain.shape}  Xtest: {Xtest.shape}")
print(f"ytrain positive rate: {ytrain.mean():.4f}  ytest positive rate: {ytest.mean():.4f}")
