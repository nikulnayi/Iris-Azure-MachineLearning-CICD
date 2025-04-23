import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
import os

# Load data
df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv")
X = df.drop("species", axis=1)
y = df["species"]

# Split & train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Save model cleanly
os.makedirs("outputs", exist_ok=True)
joblib.dump(model, "outputs/model.joblib")

print("✅ Model saved successfully.")
