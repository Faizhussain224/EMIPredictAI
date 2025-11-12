import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.title("📊 Maximum EMI Estimation")

clf_model, reg_model, scaler_clf, scaler_reg = st.session_state.models

# --- reuse same user_input_form and feature engineering from previous file (copy-paste below) ---
def user_input_form():
    st.subheader("Enter Financial & Demographic Details")

    age = st.slider("Age", 25, 60, 30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married"])
    education = st.selectbox("Education", ["High School", "Graduate", "Post Graduate", "Professional"])
    employment_type = st.selectbox("Employment Type", ["Private", "Government", "Self-employed"])
    years_of_employment = st.slider("Years of Employment", 0, 40, 5)
    monthly_salary = st.number_input("Monthly Salary (INR)", 15000, 2000000000, 50000, step=5000)
    current_emi_amount = st.number_input("Current EMI Amount (INR)", 0, 100000000, 0, step=1000)
    existing_loans = st.selectbox("Existing Loans", ["Yes", "No"])
    dependents = st.slider("Dependents", 0, 5, 1)
    family_size = st.slider("Family Size", 1, 10, 3)
    monthly_rent = st.number_input("Monthly Rent (INR)", 0, 100000000, 10000, step=1000)
    school_fees = st.number_input("School Fees (INR)", 0, 100000000, 5000, step=1000)
    college_fees = st.number_input("College Fees (INR)", 0, 100000000, 3000, step=1000)
    travel_expenses = st.number_input("Travel Expenses (INR)", 0, 1000000000, 2000, step=1000)
    groceries_utilities = st.number_input("Groceries & Utilities (INR)", 0, 1000000000, 4000, step=1000)
    other_monthly_expenses = st.number_input("Other Monthly Expenses (INR)", 0, 1000000000, 2000, step=1000)
    credit_score = st.slider("Credit Score", 0, 1000, 500)
    bank_balance = st.number_input("Bank Balance (INR)", 0, 10000000000, 10000, step=1000)
    emergency_fund = st.number_input("Emergency Fund (INR)", 0, 10000000000, 5000, step=1000)

    gender = 1 if gender == "Male" else 0
    marital_status = 1 if marital_status == "Married" else 0
    employment_map = {"Private": 0, "Government": 1, "Self-employed": 2}
    education_map = {"High School": 0, "Graduate": 1, "Post Graduate": 2, "Professional": 3}

    df = pd.DataFrame({
        "age": [age],
        "gender": [gender],
        "marital_status": [marital_status],
        "education": [education_map[education]],
        "employment_type": [employment_map[employment_type]],
        "years_of_employment": [years_of_employment],
        "monthly_salary": [monthly_salary],
        "current_emi_amount": [current_emi_amount],
        "existing_loans": [1 if existing_loans == "Yes" else 0],
        "dependents": [dependents],
        "family_size": [family_size],
        "monthly_rent": [monthly_rent],
        "school_fees": [school_fees],
        "college_fees": [college_fees],
        "travel_expenses": [travel_expenses],
        "groceries_utilities": [groceries_utilities],
        "other_monthly_expenses": [other_monthly_expenses],
        "credit_score": [credit_score],
        "bank_balance": [bank_balance],
        "emergency_fund": [emergency_fund],
    })
    return df

def apply_feature_engineering(df):
    df["debt_to_income_ratio"] = df["current_emi_amount"] / (df["monthly_salary"] + 1)
    df["expense_to_income_ratio"] = (
        df["groceries_utilities"] + df["travel_expenses"] + df["school_fees"] +
        df["college_fees"] + df["other_monthly_expenses"]
    ) / (df["monthly_salary"] + 1)
    df["affordability_index"] = (
        (df["monthly_salary"]) * 0.7 -
        (df["current_emi_amount"] + df["groceries_utilities"] +
         df["travel_expenses"] + df["school_fees"] +
         df["college_fees"] + df["other_monthly_expenses"])
    ) / (df["monthly_salary"] + 1)
    return df.fillna(0)


user_df = user_input_form()

if st.button("Predict Maximum EMI"):
    try:
        user_df = apply_feature_engineering(user_df)
        viz_df = user_df.copy()

        for col in scaler_reg.feature_names_in_:
            if col not in user_df.columns:
                user_df[col] = 0
        user_df = user_df[scaler_reg.feature_names_in_]

        X_scaled = scaler_reg.transform(user_df)
        predicted_amount = reg_model.predict(X_scaled)[0]

        st.success(f"Predicted Maximum EMI: ₹{predicted_amount:,.2f}")

        # (then keep your visualization code below here)

    except Exception as e:
        st.error(f"⚠️ An error occurred during prediction:\n\n{e}")

    # 🧾 Visualizations use viz_df (not user_df)
    salary = viz_df["monthly_salary"].iloc[0]

    # 🎨 Salary vs EMI bar chart
    fig = go.Figure()
    fig.add_trace(go.Bar(x=["Monthly Salary"], y=[salary], name="Monthly Salary"))
    fig.add_trace(go.Bar(x=["Predicted EMI"], y=[predicted_amount], name="Predicted EMI"))
    fig.update_layout(barmode="group", title="Salary vs Predicted EMI", yaxis_title="INR")
    st.plotly_chart(fig, use_container_width=True)

    # 💡 Expense Breakdown Pie Chart
    exp_labels = ["Rent", "Groceries", "Travel", "School", "College", "Other"]
    exp_values = [
        viz_df["monthly_rent"].iloc[0],
        viz_df["groceries_utilities"].iloc[0],
        viz_df["travel_expenses"].iloc[0],
        viz_df["school_fees"].iloc[0],
        viz_df["college_fees"].iloc[0],
        viz_df["other_monthly_expenses"].iloc[0],
    ]
    fig2 = go.Figure(data=[go.Pie(labels=exp_labels, values=exp_values, hole=0.3)])
    fig2.update_layout(title_text="Monthly Expense Breakdown")
    st.plotly_chart(fig2, use_container_width=True)

    # 💰 Optional: Savings Gauge
    savings_ratio = 1 - (predicted_amount / (salary + 1))
    fig3 = go.Figure(go.Indicator(
        mode="gauge+number",
        value=savings_ratio * 100,
        title={"text": "Savings After EMI (%)"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "green" if savings_ratio > 0.4 else "orange" if savings_ratio > 0.2 else "red"},
            "steps": [
                {"range": [0, 20], "color": "red"},
                {"range": [20, 40], "color": "orange"},
                {"range": [40, 100], "color": "green"},
            ],
        },
    ))
    st.plotly_chart(fig3, use_container_width=True)


st.markdown("---")
st.subheader("💬 Share Your Feedback")

feedback = st.text_area("How was your experience with Maximum EMI Estimation?")
if st.button("Submit Feedback"):
    if feedback.strip():
        st.success("✅ Thank you for your valuable feedback!")
    else:
        st.warning("Please write something before submitting.")

