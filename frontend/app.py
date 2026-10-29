import os

import pandas as pd
import requests
import streamlit as st

# Backend address. Inside the Docker network the backend container is reachable by its name.
BACKEND_URL = os.getenv("BACKEND_URL", "http://superkart-backend:7860").rstrip("/")

st.set_page_config(page_title="SuperKart Sales Forecast", layout="centered")
st.title("SuperKart Sales Forecast")
st.write("Forecast the sales revenue of a product in a store.")

PERISHABLES = ["Dairy", "Meat", "Fruits and Vegetables", "Breakfast", "Breads", "Seafood"]
PRODUCT_TYPES = sorted(PERISHABLES + [
    "Snack Foods", "Frozen Foods", "Household", "Baking Goods", "Canned", "Health and Hygiene",
    "Soft Drinks", "Hard Drinks", "Starchy Foods", "Others",
])
REFERENCE_YEAR = 2025

single_tab, batch_tab = st.tabs(["Single prediction", "Batch prediction"])

with single_tab:
    with st.form("single"):
        st.subheader("Product")
        col1, col2 = st.columns(2)
        product_id = col1.text_input("Product ID", value="FD6114",
                                     help="Only the first two letters (FD, NC, DR) are used.")
        product_type = col2.selectbox("Product type", PRODUCT_TYPES)
        weight = col1.number_input("Product weight", min_value=0.0, max_value=50.0, value=12.66)
        mrp = col2.number_input("Product MRP", min_value=0.0, max_value=1000.0, value=117.08)
        sugar = col1.selectbox("Sugar content", ["Low Sugar", "Regular", "No Sugar"])
        area = col2.number_input("Allocated display area (ratio)", min_value=0.0, max_value=1.0,
                                 value=0.027, step=0.001, format="%.3f")

        st.subheader("Store")
        col3, col4 = st.columns(2)
        store_type = col3.selectbox("Store type", ["Supermarket Type2", "Supermarket Type1",
                                                   "Departmental Store", "Food Mart"])
        store_size = col4.selectbox("Store size", ["Medium", "High", "Small"])
        city_type = col3.selectbox("City tier", ["Tier 2", "Tier 1", "Tier 3"])
        est_year = col4.number_input("Establishment year", min_value=1950, max_value=REFERENCE_YEAR,
                                     value=2009, step=1)

        submitted = st.form_submit_button("Predict sales")

    if submitted:
        payload = {
            "Product_Weight": weight,
            "Product_Sugar_Content": sugar,
            "Product_Allocated_Area": area,
            "Product_MRP": mrp,
            "Store_Size": store_size,
            "Store_Location_City_Type": city_type,
            "Store_Type": store_type,
            "Product_Id_char": product_id.strip().upper()[:2],
            "Store_Age_Years": int(REFERENCE_YEAR - est_year),
            "Product_Type_Category": "Perishables" if product_type in PERISHABLES else "Non Perishables",
        }
        try:
            response = requests.post(f"{BACKEND_URL}/v1/predict", json=payload, timeout=30)
            if response.ok:
                st.success(f"Forecast sales: {response.json()['Predicted_Sales']:,.2f}")
            else:
                st.error(f"API error {response.status_code}: {response.text}")
        except requests.exceptions.RequestException as exc:
            st.error(f"Could not reach the backend at {BACKEND_URL}: {exc}")

with batch_tab:
    st.write("Upload a CSV with the columns: Product_Weight, Product_Sugar_Content, "
             "Product_Allocated_Area, Product_MRP, Store_Size, Store_Location_City_Type, "
             "Store_Type, Product_Id_char, Store_Age_Years, Product_Type_Category.")
    uploaded = st.file_uploader("CSV file", type="csv")

    if uploaded is not None and st.button("Predict batch"):
        try:
            response = requests.post(f"{BACKEND_URL}/v1/predictbatch",
                                     files={"file": uploaded.getvalue()}, timeout=60)
            if response.ok:
                predictions = response.json()
                result = pd.read_csv(uploaded, index_col=None)
                uploaded.seek(0)
                result["Predicted_Sales"] = [predictions[str(i)] for i in range(len(result))]
                st.dataframe(result)
                st.download_button("Download predictions", result.to_csv(index=False),
                                   file_name="superkart_predictions.csv", mime="text/csv")
            else:
                st.error(f"API error {response.status_code}: {response.text}")
        except requests.exceptions.RequestException as exc:
            st.error(f"Could not reach the backend at {BACKEND_URL}: {exc}")
