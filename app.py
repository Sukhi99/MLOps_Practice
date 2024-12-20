from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Loading the model and and the encoders
with open("loan_clf_with_encoders.pkl", "rb") as f:
    rf_model, label_encoder, scaler = pickle.load(f)


@app.route("/")
def home():
    return {"message": "Yo Yo!! I am making an API endpoint"}


@app.route("/aboutus")
def info():
    return "<h1> THIS API IS OWNED BY BOSS </h1>"


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        # get the input in json format
        req_data = request.get_json()

        print(req_data)

        # label encode the request data
        for col, encoder in label_encoder.items():
            print(f"Column Name: {col}")
            print(f"encoder name: {encoder}")
            print(f"req_data[col]: {req_data[col]:}")
            req_data[col] = encoder.transform([req_data[col]])[0]

        # Separate categorical and numerical data
        numerical_cols = [
            "ApplicantIncome",
            "CoapplicantIncome",
            "LoanAmount",
            "Loan_Amount_Term",
            "Credit_History",
        ]
        categorical_data = [req_data[col] for col in label_encoder.keys()]
        numerical_data = [req_data[col] for col in numerical_cols]

        # Combine numerical and encoded categorical data
        X_input = categorical_data + numerical_data

        # Convert the input data into a DataFrame to ensure column names are consistent
        feature_names = list(label_encoder.keys()) + numerical_cols
        X_input_df = pd.DataFrame([X_input], columns=feature_names)

        # Standardize the input using the scaler
        X_input_scaled = scaler.transform(X_input_df)

        # req_data_scaled = scaler.transform(req_data)

        result = rf_model.predict(X_input_scaled)

        if result == 1:
            pred = "Approved"
        else:
            pred = "Rejected"

        return {"Loan Approval Status": pred}

        pass
    else:
        return "Provide the input in json format using POST request"
