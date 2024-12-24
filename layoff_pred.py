import streamlit as st
import pickle
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# Load the saved models and components for future forecast
with open('trend_arima_model.pkl', 'rb') as f:
    model_trend_full = pickle.load(f)

with open('residual_arima_model.pkl', 'rb') as f:
    model_resid_full = pickle.load(f)

with open('seasonal_component.pkl', 'rb') as f:
    seasonal_full = pickle.load(f)

# Load the trained Random Forest model for attrition
with open('random_forest_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)

with open('label_encoders.pkl', 'rb') as f:
    encoders = pickle.load(f)


def main():
    
    st.title("Role of Artificial Intelligence in evolving the workforce in the IT sector")
    # Option menu
    menu = ["Future Forecasting", "Attrition Prediction"]
    choice = st.sidebar.selectbox("Select an Option", menu)

    if choice == "Future Forecasting":
        st.title("Future Forecasting of No. of Layoffs")
        st.write("This section forecasts future trends using STL ARIMA.")
        
        
        # User input for number of future steps
        future_steps = st.number_input("Enter the number of future steps to forecast:", min_value=1, max_value=12, value=12)

        # Generate forecast
        trend_forecast_full = model_trend_full.forecast(steps=future_steps)
        residual_forecast_full = model_resid_full.forecast(steps=future_steps)

        future_forecast_log = trend_forecast_full + seasonal_full[:future_steps] + residual_forecast_full
        future_forecast_values = np.expm1(future_forecast_log)
        future_forecast_values = np.clip(future_forecast_values, a_min=0, a_max=None)

        # Display forecast
        data=pd.read_csv('real_data.csv')
        
        data['Date'] = pd.to_datetime(data['Date'])  # Convert 'Date' to datetime
        data.set_index('Date', inplace=True)  # Set 'Date' as the index
        future_dates = pd.date_range(data.index[-1], periods=future_steps + 1, freq='W')[1:]
        forecast_df = pd.DataFrame({
                                     'Date': future_dates,
                                     'Forecasted_Laid_Off_Count': future_forecast_values
                                   })

        st.write("Forecasted Values:")
        st.write(forecast_df)

        # Plot forecast
        plt.figure(figsize=(12, 6))
        plt.plot(data.index, data['Laid_Off_Count'], label='Observed', color='blue')
        
        
        plt.plot(future_dates, future_forecast_values, label='Forecast (Future)', color='red', linestyle='-')
        plt.title('STL-ARIMA Forecast ')
        plt.xlabel('Date')
        plt.ylabel('Laid Off Count')
        plt.legend()
        plt.grid()
        st.write("chart")
        st.pyplot(plt)
        
        
    elif choice == "Attrition Prediction":
        st.title("Attrition Prediction")
        st.header("Feature Importance for Attrition Prediction")
        feature_importance_df = pd.DataFrame({
                      'Feature': ['MonthlyIncome','Age','YearsAtCompany','DistanceFromHome','NumCompaniesWorked','JobSatisfaction','EnvironmentSatisfaction'   ,'EducationField'   ,
                               'MaritalStatus','Education' ,'WorkLifeBalance','Department'],
                       'Importance': [   0.196761 , 0.13211 , 0.111539, 0.106697,0.076737, 0.064058,0.059736, 0.058995 ,0.056235,0.050554,0.050467,0.036103]})

        # Display the table  Streamlit
       
        st.write("Feature Importance Table:")
        st.write(feature_importance_df)

        # Plot the feature importance using Matplotlib
        plt.figure(figsize=(8, 4))
        plt.bar(feature_importance_df['Feature'], feature_importance_df['Importance'], color='skyblue')
        plt.title('Feature Importance for Attrition Prediction')
        plt.ylabel('Importance')
        plt.xlabel('Features')
        plt.xticks(rotation=90)

        # Display the graph in Streamlit
        st.write("Feature Importance Bar Chart:")
        st.pyplot(plt)
        st.header("Attrition prediction using Random forest model")
        marital_status_options = ["Single", "Married"]
        education_field_options = ['Life Sciences', 'Other', 'Medical', 'Marketing',
                               'Technical Degree', 'Human Resources']

        # Input fields for numerical columns
        monthly_income = st.number_input("Monthly Income", min_value=0, step=1000)
        age = st.number_input("Age", min_value=18, step=1)
        years_at_company = st.number_input("Years at Company", min_value=0, step=1)
        distance_from_home = st.number_input("Distance From Home (km)", min_value=0, step=1)
        num_companies_worked = st.number_input("Number of Companies Worked", min_value=0, step=1)
        job_satisfaction = st.slider("Job Satisfaction (1-4)", min_value=1, max_value=4)
        environment_satisfaction = st.slider("Environment Satisfaction (1-4)", min_value=1, max_value=4)
        education = st.slider("Education (1-5)", min_value=1, max_value=5)
        work_life_balance = st.slider("Work Life Balance (1-4)", min_value=1, max_value=4)

        # Input fields for categorical columns
        education_field = st.selectbox("Education Field", education_field_options)
        marital_status = st.selectbox("Marital Status", marital_status_options)

        # Preprocess the inputs
        input_data = {
            "MonthlyIncome": monthly_income,
            "Age": age,
            "YearsAtCompany": years_at_company,
            "DistanceFromHome": distance_from_home,
            "NumCompaniesWorked": num_companies_worked,
            "JobSatisfaction": job_satisfaction,
            "EnvironmentSatisfaction": environment_satisfaction,
            "EducationField": education_field,
            "MaritalStatus": marital_status,
            "Education": education,
            "WorkLifeBalance": work_life_balance
        }

        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])

        # Encode categorical columns
        for col in ["EducationField", "MaritalStatus"]:
            encoder = encoders.get(col)
            if encoder:
                input_df[col] = encoder.transform(input_df[col])

        # Make prediction
        if st.button("Predict Attrition"):
            prediction = rf_model.predict(input_df)[0]
            prediction_proba = rf_model.predict_proba(input_df)[0]

            if prediction == 1:
                st.error(f"The employee is likely to attrite with a probability of {prediction_proba[1]:.2f}.")
            else:
                st.success(f"The employee is unlikely to attrite with a probability of {prediction_proba[0]:.2f}.")
            
if __name__ == "__main__":
    main()

     