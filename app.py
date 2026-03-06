import pickle
import pandas as pd
import streamlit as st

# df[["Geography", "Gender","Card Type","Age","CreditScore","Tenure","Point Earned","Balance","EstimatedSalary", 
#     "NumOfProducts","IsActiveMember","HasCrCard","Satisfaction Score","Exited"]]

def load_data():
    df  = pd.read_csv("data/Churn Sampled.csv").sample(10)
    return df

def load_models():
    with open("models/GBC_Pipeline.pkl", "rb") as f:
        gbc = pickle.load(f)
    with open("models/XGB_Pipeline.pkl", "rb") as f:
        xgb = pickle.load(f)
    with open("models/RFC_Pipeline.pkl", "rb") as f:
        rfc = pickle.load(f)

    return gbc, xgb, rfc

def fetch_inputs():
    geography = st.selectbox("Country : ", ["Germany", "Spain", "France"])
    gender = st.selectbox("Gender :", ["Male", "Female"])
    card = st.selectbox("Card Type :", ["SILVER", "GOLD", "PLATIMUN", "DIAMOND"])
    age  = st.slider("Age :", min_value  = 18, max_value = 70, step = 30)
    creditscore = st.slider("Credit Score", min_value  = 100, max_value = 1000, step = 10)
    tenure = st.selectbox("Tenure : ", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    points = st.number_input("Earned Points :", min_value = 100, max_value = 1000, step = 10)
    balance = st.number_input("Current Balance :")
    salary = st.number_input("Estimated Salary :")
    products = st.selectbox("Number of Products :", [1, 2, 3, 4])
    isactive = st.selectbox("Are you Active Member", [0, 1])
    creditcard = st.selectbox("Do you have credit card :", [0, 1])
    satisfaction = st.selectbox("Satisfaction Score :", [1,2,3,4,5])

    return [[geography], [gender], [card], [age], [creditscore], [tenure], [points], [balance], [salary], [products], [isactive], [creditcard], [satisfaction]]

def create_df():
    data = fetch_inputs()
    df = pd.DataFrame(data, columns = ["Geography", "Gender","Card Type","Age","CreditScore","Tenure","Point Earned","Balance","EstimatedSalary", "NumOfProducts","IsActiveMember","HasCrCard","Satisfaction Score"])
    return df

def predict():
    gbc, xgb, rfc = load_models()

    gb = gbc.predict(create_df())
    xg = xgb.predict(create_df())
    rf = rfc.predict(create_df())

    return gb, xg, rf


def main():
    st.subheader("Data View : ")
    st.dataframe(load_data())

    fetch_inputs()

    st.subheader("User Values :")
    st.dataframe(create_df())

    gb, xg, rf = predict()

    st.markdown("Model predictions : ")
    st.markdown([gb, xg, rf])

    if gb == 1:
        st.markdown("Customer is going to leave the bank." )
    elif gb == 0:
        st.markdown("Customer is not going to leave the bank." )


if __name__ == "__main__":
    main()
