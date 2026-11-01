import pandas as pd
from sklearn.model_selection import train_test_split

# Read the cleaned dataset
df = pd.read_csv("tourism_project/data/tourism.csv")

# Drop Unnamed: 0 and CustomerID if they exist
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])
if "CustomerID" in df.columns:
    df = df.drop(columns=["CustomerID"])

# Clean inconsistent categories to align with notebook
df.loc[df["Gender"] == "Fe Male", "Gender"] = "Female"
df.loc[df["MaritalStatus"] == "Unmarried", "MaritalStatus"] = "Single"

X = df.drop(columns=["ProdTaken"])
y = df["ProdTaken"]

# stratify=y keeps the (imbalanced) failure ratio consistent across splits
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Data prepared: train/test splits written.")
