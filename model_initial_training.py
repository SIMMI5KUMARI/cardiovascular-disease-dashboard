import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

DATA_PATH = "cardio_train.csv"

MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "heart_disease_model.pkl")
FEATURES_PATH = os.path.join(MODEL_DIR, "model_features.pkl")


def train_model():

    print("Starting model training...")

    df = pd.read_csv(DATA_PATH, sep=";")

    df = df.drop("id", axis=1)

    df.drop_duplicates(inplace=True)

    df = df[(df["ap_hi"] < 250) & (df["ap_lo"] < 200)]
    df = df[(df["ap_hi"] > 50) & (df["ap_lo"] > 50)]
    df = df[(df["ap_hi"] >= df["ap_lo"])]

    df["age"] = (df["age"] / 365.25).astype(int)

    target = "cardio"

    X = df.drop(target, axis=1)
    y = df[target]

    feature_list = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=1000, solver="liblinear")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    print("Model accuracy:", acc)

    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(feature_list, FEATURES_PATH)

    print("Model saved successfully")


if __name__ == "__main__":
    train_model()
