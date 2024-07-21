import streamlit as st
import pandas as pd
import pickle
import numpy as np

st.set_page_config(layout='wide')

if 'page' not in st.session_state:
    st.session_state['page'] = 'Home'

@st.cache_resource
def load_data():
    return pd.read_csv(r"K:\data\Industry copper.csv")

@st.cache_resource
def load_models():
    with open(r"K:\data\ML Model\Linear regression\Rit_ohe.pkl", "rb") as ft:
        it_ohe = pickle.load(ft)
    with open(r"K:\data\ML Model\Linear regression\Regression_scaler.pkl", "rb") as ft:
        scaler = pickle.load(ft)
    with open(r"K:\data\ML Model\Linear regression\Regression_model.pkl", "rb") as f:
        reg_model = pickle.load(f)
    with open(r"K:\data\ML Model\Linear regression\Rs_ohe.pkl", "rb") as f:
        s_ohe = pickle.load(f)
    with open(r"K:\data\ML Model\Logistic regression\classification_it_ohe.pkl", "rb") as ft:
        cls_ohe = pickle.load(ft)
    with open(r"K:\data\ML Model\Logistic regression\classification_scaler.pkl", "rb") as ft:
        cls_scaler = pickle.load(ft)
    with open(r"K:\data\ML Model\Logistic regression\classification_model.pkl", "rb") as f:
        cls_model = pickle.load(f)
    return it_ohe, scaler, reg_model, s_ohe, cls_ohe, cls_scaler, cls_model

df = load_data()
it_ohe, scaler, reg_model, s_ohe, cls_ohe, cls_scaler, cls_model = load_models()

def back():
    st.session_state.page = 'Home'

def change():
    st.session_state.page = 'SP'

def change1():
    st.session_state.page = 'status'

col1, col2, col3 = st.columns([0.2, 0.7, 0.1], gap='large')

if st.session_state.page == 'Home':
    col2.title(":orange[Industry Copper Modelling]")
    st.text("""
The copper industry often encounters challenges with sales and pricing data, which can be skewed and noisy. These issues
complicate manual predictions, leading to suboptimal pricing decisions and inefficient lead management. The aim of this
project is to develop two machine learning models:
""")

    container = st.container(border = True)
    container.text("""
Regression Model
             To predict the selling prices of copper products, accounting for variables such as quantity, customer
details, country, item type, application, and material properties.""")
    
    container.button("Regression Model", on_click=change)

    container.text("""
Classification Model
                To classify leads as 'won' or 'lost' based on similar variables, optimizing sales strategies and
improving lead conversion rates.""")
    container.button("Classification Model", on_click=change1)

    st.text("""
To achieve these objectives, the solution involves:

1. Data Preprocessing: Handling missing values, outlier detection, and skewness treatment.
2. Feature Engineering: Creating informative features and encoding categorical variables.
3. Model Building: Training and evaluating tree-based models for both regression and classification tasks.
4. Model Deployment: Developing a Streamlit application for real-time predictions, enabling users to input relevant data
   and receive accurate selling prices or lead statuses.
                   
This project aims to enhance decision-making in the copper industry by leveraging machine learning techniques to provide
reliable and efficient predictions, ultimately leading to better pricing strategies and improved lead management.
""")

if st.session_state.page == 'SP':
    c1, c2, c3 = st.columns([0.1, 0.8, 0.1], gap='large')
    c2.title(":orange[Regression Model: Selling Price Prediction]")
    c1.button("Back", on_click=back)

    co1, co2 = st.columns([0.5, 0.5], gap='small')

    with co1:
        container1 = st.container()
        container1.text("Categorical column")
        item = st.selectbox("Select Item Type", df['item type'].unique())
        status = st.selectbox("Select Status", df['status'].unique())
        country = st.selectbox("Select Country Code", df['country'].unique())
        application = st.selectbox("Select Application Code", df['application'])
        customer = st.selectbox("Select Customer ID", df['customer'])
        product = st.selectbox("Select Product Code", df['product_ref'])

    with co2:
        container2 = st.container()
        container2.text("Continuous column")
        thickness = st.number_input(f"Thickness, Min={df['thickness'].min()}, Max={df['thickness'].max()}", value=None, placeholder="Type a number...")
        width = st.number_input(f"Width, Min={df['width'].min()}, Max={df['width'].max()}", value=None, placeholder="Type a number...")
        quantity = st.number_input(f"Quantity, Min={df['quantity tons'].min()}, Max={df['quantity tons'].max()}", value=None, placeholder="Type a number...")
        v = st.button("Predict")
        if v:
            input_data = np.array([[np.log(quantity), customer, country, application, np.log(thickness), np.log(width), product, status, item]])
            status_ohe = s_ohe.transform(input_data[:, [7]]).toarray()
            item_ohe = it_ohe.transform(input_data[:, [8]]).toarray()
            features = np.concatenate((input_data[:, [0, 1, 2, 3, 4, 5, 6]], status_ohe, item_ohe), axis=1)
            features[:, [0, 4, 5]] = scaler.transform(features[:, [0, 4, 5]])
            prediction = reg_model.predict(features)
            st.success(np.exp(prediction[0]))

if st.session_state.page == 'status':
    col2.title(":orange[Classification Model: Status Prediction]")
    col1.button("Back", on_click=back)

    column1, column2 = st.columns([0.5, 0.5], gap='small')

    with column1:
        container1 = st.container()
        container1.text("Categorical column")
        item = st.selectbox("Select Item Type", df['item type'].unique())
        country = st.selectbox("Select Country Code", df['country'].unique())
        application = st.selectbox("Select Application Code", df['application'])
        customer = st.selectbox("Select Customer ID", df['customer'])
        product = st.selectbox("Select Product Code", df['product_ref'])

    with column2:
        container2 = st.container()
        container2.text("Continuous column")
        sp = st.number_input(f"Selling Price, Min={df['selling_price'].min()}, Max={df['selling_price'].max()}", value=None, placeholder="Type a number...")
        thickness = st.number_input(f"Thickness, Min={df['thickness'].min()}, Max={df['thickness'].max()}", value=None, placeholder="Type a number...")
        width = st.number_input(f"Width, Min={df['width'].min()}, Max={df['width'].max()}", value=None, placeholder="Type a number...")
        quantity = st.number_input(f"Quantity, Min={df['quantity tons'].min()}, Max={df['quantity tons'].max()}", value=None, placeholder="Type a number...")
        v = st.button("Predict")
        if v:
            input_data = np.array([[np.log(quantity), customer, country, application, np.log(thickness), np.log(width), product, np.log(sp), item]])
            item_ohe = cls_ohe.transform(input_data[:, [8]]).toarray()
            features = np.concatenate((input_data[:, [0, 1, 2, 3, 4, 5, 6, 7]], item_ohe), axis=1)
            features[:, [0, 4, 5, 7]] = cls_scaler.transform(features[:, [0, 4, 5, 7]])
            prediction = cls_model.predict(features)
            if prediction[0] == 1.0:
                st.success("Won")
            else:
                st.error("Lost")
