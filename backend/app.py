import joblib
import pandas as pd
from flask import Flask, jsonify, request

# Initialise the Flask app
superkart_api = Flask("SuperKart Sales Forecaster")

# Load the trained pipeline (preprocessing + model) once at start-up
model = joblib.load("superkart_model.joblib")

FEATURES = [
    "Product_Weight",
    "Product_Sugar_Content",
    "Product_Allocated_Area",
    "Product_MRP",
    "Store_Size",
    "Store_Location_City_Type",
    "Store_Type",
    "Product_Id_char",
    "Store_Age_Years",
    "Product_Type_Category",
]


def missing_features(columns):
    return [f for f in FEATURES if f not in columns]


@superkart_api.get("/")
def home():
    """Health check."""
    return "Welcome to the SuperKart Sales Forecasting API!"


@superkart_api.post("/v1/predict")
def predict_sales():
    """Predict sales for a single product-store record sent as JSON."""
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400

    missing = missing_features(data.keys())
    if missing:
        return jsonify({"error": "Missing required features.", "missing": missing}), 400

    input_df = pd.DataFrame([{f: data[f] for f in FEATURES}])
    try:
        prediction = float(model.predict(input_df)[0])
    except Exception as exc:  # e.g. a non-numeric value in a numeric field
        return jsonify({"error": f"Could not score input: {exc}"}), 400

    return jsonify({"Predicted_Sales": round(prediction, 2)})


@superkart_api.post("/v1/predictbatch")
def predict_sales_batch():
    """Predict sales for every row of an uploaded CSV file."""
    if "file" not in request.files:
        return jsonify({"error": "Upload a CSV file in the 'file' form field."}), 400

    try:
        input_df = pd.read_csv(request.files["file"])
    except Exception as exc:
        return jsonify({"error": f"Could not read CSV: {exc}"}), 400

    missing = missing_features(input_df.columns)
    if missing:
        return jsonify({"error": "Missing required columns.", "missing": missing}), 400

    try:
        predictions = model.predict(input_df[FEATURES])
    except Exception as exc:
        return jsonify({"error": f"Could not score input: {exc}"}), 400

    # Key each prediction by its row index
    return jsonify({str(idx): round(float(p), 2) for idx, p in zip(input_df.index, predictions)})


if __name__ == "__main__":
    superkart_api.run(host="0.0.0.0", port=7860, debug=False)
