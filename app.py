# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import calendar
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import openpyxl

st.title("Sales Prediction & Visualization")

# ----------------- Show Rules -----------------
st.markdown("""
*⚠️ Please follow these rules before uploading your file:*

1. File must be *CSV or Excel (.xlsx)*.  
2. Required columns: product, price, quantity, date, age, gender, location.  
3. Extra columns are not being considered.  
4. Ensure numeric columns (price, quantity, age, date) do not contain text.  
5. Avoid having too much similar columns (like product catagory, item, or location, state etc)
6. Date :dd/mm/yyyy
7. Limit 200MB
""")


# ----------------- File Upload -----------------
uploaded_file = st.file_uploader("Upload your CSV or Excel", type=["csv", "xlsx"])
if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        user_data = pd.read_csv(uploaded_file, encoding="utf-8", on_bad_lines="skip")
    else:
        user_data = pd.read_excel(uploaded_file)

    st.subheader("Uploaded Data Preview")
    st.write(user_data.head())

    # ----------------- Data Cleaning -----------------
    # Drop columns with all unique or single values
    for col in user_data.columns:
        if user_data[col].nunique() in [1, len(user_data)]:
            user_data = user_data.drop(columns=[col])

    user_data.columns = user_data.columns.str.strip().str.lower()

    rename_dict = {
        "sex": "gender", "customer id": "customer_id", "customerid": "customer_id",
        "cust_id": "customer_id", "product category": "product", "prodcat": "product",
        "product_category": "product", "category": "product", "item": "product",
        "item_category": "product", "area": "location", "price_per_unit":"price",
        "price per unit": "price", "unit_price": "price", "total amount": "total",
        "total_amount": "total"
    }
    user_data.rename(columns=rename_dict, inplace=True)

    if "date" in user_data.columns:
        user_data["date"] = pd.to_datetime(user_data["date"], dayfirst=True, errors="coerce")
        user_data["year"] = user_data["date"].dt.year
        user_data["month"] = user_data["date"].dt.month
        user_data = user_data.drop(columns=["date"])
    else:
        user_data["month"] = 0
        user_data["year"] = 0

    if "age" in user_data.columns:
        user_data["age"] = user_data["age"].fillna(user_data["age"].median())

    needed_cols = ["product","price","month","quantity", "location", "age", "gender"]
    for col in needed_cols:
        if col not in user_data.columns:
            user_data[col] = 0

    # ----------------- Encoding -----------------
    column_mappings = {}
    categorical_cols = ['gender', 'product','location']
    for col in categorical_cols:
        if col in user_data.columns:
            mapping = column_mappings.get(col, {})
            new_values = [v for v in user_data[col].unique() if v not in mapping]
            if new_values:
                start_num = max(mapping.values(), default=0) + 1
                for i, val in enumerate(new_values):
                    mapping[val] = start_num + i
                column_mappings[col] = mapping
            user_data[col] = user_data[col].map(mapping)

    needed_cols = ["gender","product","price","month","quantity","age","location"]
    user_data = user_data[[col for col in user_data.columns if col in needed_cols]]

    X = user_data.drop(columns=["quantity"])
    y = user_data["quantity"]

    # ----------------- Train/Test Split -----------------
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ----------------- Cached Model Loading/Training -----------------
    @st.cache_resource
    def train_model(X_train, y_train):
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        return rf_model

    rf_model = train_model(X_train, y_train)

    # ----------------- Predictions -----------------
    user_data["Predicted_quantity"] = rf_model.predict(X)
    y_pred = rf_model.predict(X_test)

    # ----------------- Model Evaluation -----------------
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.subheader("Model Performance")
    st.write("Mean Squared Error:", mse)
    st.write("R-squared:", r2)

    # ----------------- Product Demand Plot -----------------
    inverse_product_map = {v: k for k, v in column_mappings['product'].items()}
    user_data['predicted_product'] = user_data['product'].map(inverse_product_map)

    st.subheader("🔹 Predicted Product Demand")
    product_summary = user_data.groupby('predicted_product')['Predicted_quantity'].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7,4))
    sns.barplot(x=product_summary.index, y=product_summary.values, palette="viridis", ax=ax)
    ax.set_title("Predicted Product Demand")
    ax.set_ylabel("Total Quantity")
    ax.set_xlabel("Product")
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)

    # ----------------- Age Group Pie Chart -----------------
    bins = [0, 18, 25, 35, 45, 55, 65, 100]
    labels = ['0-18', '18-25', '25-35', '35-45', '45-55', '55-65', '65+']
    user_data['age_group'] = pd.cut(user_data['age'], bins=bins, labels=labels, right=False)
    age_group_summary = user_data.groupby('age_group')['Predicted_quantity'].sum()

    st.subheader("🔹 Predicted Quantity by Age Group")
    fig2, ax2 = plt.subplots(figsize=(5,5))
    ax2.pie(age_group_summary, labels=age_group_summary.index, autopct='%1.1f%%', startangle=90)
    ax2.set_title("Predicted Quantity by Age Group")
    st.pyplot(fig2)

    # ----------------- Monthly Demand -----------------
    all_months = range(1, 13)
    all_products = user_data['predicted_product'].unique()
    full_index = pd.MultiIndex.from_product([all_months, all_products], names=["month", "predicted_product"])
    monthly_demand = user_data.groupby(["month", "predicted_product"])["Predicted_quantity"] \
        .sum().reindex(full_index, fill_value=0).reset_index()

    st.subheader("🔹 Monthly Demand by Product")
    fig3, ax3 = plt.subplots(figsize=(10,6))
    sns.lineplot(data=monthly_demand, x="month", y="Predicted_quantity",
                 hue="predicted_product", marker="o", ax=ax3)
    ax3.set_title("Monthly Demand by Product")
    ax3.set_xlabel("Month")
    ax3.set_ylabel("Predicted Quantity")
    ax3.set_xticks(list(all_months))
    ax3.set_xticklabels([calendar.month_abbr[m] for m in all_months])
    st.pyplot(fig3)

    # ----------------- Location vs Product -----------------
    inverse_location_map = {v: k for k, v in column_mappings['location'].items()}
    user_data['location_name'] = user_data['location'].map(inverse_location_map)
    top_location_products = user_data.groupby(['location_name','predicted_product'])['Predicted_quantity'] \
        .sum().reset_index()

    st.subheader("🔹 Predicted Product Quantity by Location")
    fig4, ax4 = plt.subplots(figsize=(12,6))
    sns.barplot(data=top_location_products, x='location_name',
                y='Predicted_quantity', hue='predicted_product', ax=ax4)
    ax4.set_title("Predicted Product Quantity by Location")
    ax4.set_xlabel("Location")
    ax4.set_ylabel("Total Predicted Quantity")
    ax4.tick_params(axis='x', rotation=45)
    st.pyplot(fig4)

   
