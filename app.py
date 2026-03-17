from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# load ML model
model = joblib.load("model/heart_disease_model.pkl")
features = joblib.load("model/model_features.pkl")


# ---------- Home Page ----------
@app.route("/")
def dashboard():
    return render_template("dashboard.html")


# ---------- ML Form Page ----------
@app.route("/predict_page")
def predict_page():
    return render_template("index.html")


# ---------- Prediction ----------
@app.route("/predict", methods=["POST"])
def predict():

    age = int(request.form["age"])
    gender = int(request.form["gender"])
    height = int(request.form["height"])
    weight = float(request.form["weight"])
    ap_hi = int(request.form["ap_hi"])
    ap_lo = int(request.form["ap_lo"])
    cholesterol = int(request.form["cholesterol"])
    gluc = int(request.form["gluc"])
    smoke = int(request.form["smoke"])
    alco = int(request.form["alco"])
    active = int(request.form["active"])

    input_data = np.array([[age, gender, height, weight, ap_hi, ap_lo,
                            cholesterol, gluc, smoke, alco, active]])

    input_df = pd.DataFrame(input_data, columns=features)

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        result = "High Risk of Cardiovascular Disease"
    else:
        result = "Low Risk of Cardiovascular Disease"

    return render_template(
        "results.html",
        prediction=result,
        probability=round(probability * 100, 2)
    )


if __name__ == "__main__":
    app.run(debug=True)
