import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="AI Job Salary Predictor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_model_and_data():
    try:
        model = joblib.load('best_salary_model.pkl')
        
        scaler = joblib.load('salary_scaler.pkl')
        
        with open('model_info.pkl', 'rb') as f:
            model_info = pickle.load(f)
        
        return model, scaler, model_info
    except FileNotFoundError as e:
        st.error("Model files not found. Please ensure you have run the training script first.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

try:
    model, scaler, model_info = load_model_and_data()
except:
    st.error("Please run the training script first to generate model files.")
    st.stop()

st.title("💰 AI Job Salary Predictor")
st.markdown("### Predict your potential salary in the AI industry based on your qualifications and job characteristics")

st.sidebar.header("📊 Model Information")
st.sidebar.write(f"**Best Model:** {model_info['model_name']}")
st.sidebar.write(f"**R² Score:** {model_info['performance_metrics']['R2']:.4f}")
st.sidebar.write(f"**RMSE:** ${model_info['performance_metrics']['RMSE']:,.2f}")
st.sidebar.write(f"**MAE:** ${model_info['performance_metrics']['MAE']:,.2f}")

# Create tabs
tab1, tab2, tab3 = st.tabs(["🔮 Salary Prediction", "📈 Model Performance", "ℹ️ About"])

with tab1:
    st.header("Enter Your Job Characteristics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Personal Information")
        
        experience_level = st.selectbox(
            "Experience Level",
            options=['EN', 'MI', 'SE', 'EX'],
            format_func=lambda x: {
                'EN': 'Entry-level',
                'MI': 'Mid-level', 
                'SE': 'Senior-level',
                'EX': 'Executive-level'
            }[x]
        )
        
        years_experience = st.slider("Years of Experience", 0, 30, 5)
        
        education_required = st.selectbox(
            "Education Level",
            options=['Bachelor', 'Master', 'PhD', 'Associate', 'High School']
        )
        
        employee_residence = st.selectbox(
            "Your Country of Residence",
            options=['United States', 'Canada', 'United Kingdom', 'Germany', 'India', 
                    'France', 'Australia', 'Netherlands', 'Singapore', 'China', 'Other']
        )
        
        salary_currency = st.selectbox(
            "Preferred Salary Currency",
            options=['USD', 'EUR', 'GBP', 'CAD', 'AUD', 'SGD', 'INR']
        )
    
    with col2:
        st.subheader("Job Information")
        
        employment_type = st.selectbox(
            "Employment Type",
            options=['FT', 'PT', 'CT', 'FL'],
            format_func=lambda x: {
                'FT': 'Full-time',
                'PT': 'Part-time',
                'CT': 'Contract',
                'FL': 'Freelance'
            }[x]
        )
        
        company_size = st.selectbox(
            "Company Size",
            options=['S', 'M', 'L'],
            format_func=lambda x: {
                'S': 'Small (< 50 employees)',
                'M': 'Medium (50-250 employees)',
                'L': 'Large (> 250 employees)'
            }[x]
        )
        
        company_location = st.selectbox(
            "Company Location",
            options=['United States', 'Canada', 'United Kingdom', 'Germany', 'India',
                    'France', 'Australia', 'Netherlands', 'Singapore', 'China', 'Other']
        )
        
        industry = st.selectbox(
            "Industry",
            options=['Technology', 'Finance', 'Healthcare', 'Consulting', 'Education',
                    'Media', 'Automotive', 'Retail', 'Manufacturing', 'Other']
        )
        
        remote_ratio = st.slider("Remote Work Ratio (%)", 0, 100, 50)
        
        benefits_score = st.slider("Benefits Score (1-10)", 1.0, 10.0, 6.5, step=0.1)
    
    st.subheader("Skills")
    st.write("Select the skills you have:")
    
    skill_cols = st.columns(4)
    selected_skills = {}
    
    for i, skill in enumerate(model_info['top_skills']):
        col_idx = i % 4
        with skill_cols[col_idx]:
            selected_skills[skill] = st.checkbox(skill)
    
    if st.button("🔮 Predict Salary", type="primary", use_container_width=True):
        try:
            input_data = {}
            
            input_data['remote_ratio'] = remote_ratio
            input_data['years_experience'] = years_experience
            input_data['benefits_score'] = benefits_score
            
            input_data['company_name_frequency'] = 1  
            input_data['job_title_frequency'] = 1     

            for skill in model_info['top_skills']:
                skill_key = f"skill_{skill.lower().replace(' ', '_')}"
                input_data[skill_key] = 1 if selected_skills.get(skill, False) else 0
            
            for feature_name in model_info['feature_names']:
                if feature_name not in input_data:
                    if f'salary_currency_{salary_currency}' == feature_name:
                        input_data[feature_name] = 1
                    elif f'experience_level_{experience_level}' == feature_name:
                        input_data[feature_name] = 1
                    elif f'employment_type_{employment_type}' == feature_name:
                        input_data[feature_name] = 1
                    elif f'company_location_top_{company_location}' == feature_name:
                        input_data[feature_name] = 1
                    elif f'company_size_{company_size}' == feature_name:
                        input_data[feature_name] = 1
                    elif f'employee_residence_top_{employee_residence}' == feature_name:
                        input_data[feature_name] = 1
                    elif f'education_required_{education_required}' == feature_name:
                        input_data[feature_name] = 1
                    elif f'industry_{industry}' == feature_name:
                        input_data[feature_name] = 1
                    else:
                        input_data[feature_name] = 0
            
            input_df = pd.DataFrame([input_data])
            input_df = input_df.reindex(columns=model_info['feature_names'], fill_value=0)
            
            if model_info['model_name'] == 'Linear Regression':
                input_scaled = scaler.transform(input_df)
                prediction = model.predict(input_scaled)[0]
            else:
                prediction = model.predict(input_df)[0]
            
            st.success(f"### 💰 Predicted Salary: ${prediction:,.2f}")
            st.info(f"""
            **Prediction Details:**
            - Experience Level: {experience_level}
            - Years of Experience: {years_experience}
            - Company Size: {company_size}
            - Remote Work: {remote_ratio}%
            - Industry: {industry}
            """)
            
            mae = model_info['performance_metrics']['MAE']
            st.write(f"**Estimated Range:** ${prediction - mae:,.2f} - ${prediction + mae:,.2f}")
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.write("Please ensure all model files are properly generated from the training script.")

with tab2:
    st.header("📈 Model Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Metrics")
        st.metric("R² Score", f"{model_info['performance_metrics']['R2']:.4f}")
        st.metric("RMSE", f"${model_info['performance_metrics']['RMSE']:,.2f}")
        st.metric("MAE", f"${model_info['performance_metrics']['MAE']:,.2f}")
        st.metric("MSE", f"${model_info['performance_metrics']['MSE']:,.2f}")
    
    with col2:
        st.subheader("Model Information")
        st.write(f"**Selected Model:** {model_info['model_name']}")
        st.write(f"**Number of Features:** {len(model_info['feature_names'])}")
        st.write(f"**Top Skills Considered:** {', '.join(model_info['top_skills'][:5])}")
    
    if 'all_results' in model_info:
        st.subheader("Model Comparison")
        
        try:
            results = model_info['all_results']
            models = list(results.keys())
            r2_scores = [results[model]['R2'] for model in models]
            rmse_scores = [results[model]['RMSE'] for model in models]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold']
            bars1 = ax1.bar(models, r2_scores, color=colors[:len(models)])
            ax1.set_title('R² Score Comparison')
            ax1.set_ylabel('R² Score')
            ax1.tick_params(axis='x', rotation=45)
            
            for bar, score in zip(bars1, r2_scores):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{score:.3f}', ha='center', va='bottom')
            
            bars2 = ax2.bar(models, rmse_scores, color=colors[:len(models)])
            ax2.set_title('RMSE Comparison')
            ax2.set_ylabel('RMSE ($)')
            ax2.tick_params(axis='x', rotation=45)
            
            for bar, score in zip(bars2, rmse_scores):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'${score:,.0f}', ha='center', va='bottom')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()  
            
        except Exception as e:
            st.error(f"Error creating performance charts: {str(e)}")
    
    st.subheader("Performance Interpretation")
    r2_score = model_info['performance_metrics']['R2']
    
    if r2_score >= 0.8:
        st.success(f"Excellent model performance! The model explains {r2_score*100:.1f}% of the variance in salary predictions.")
    elif r2_score >= 0.6:
        st.info(f"Good model performance. The model explains {r2_score*100:.1f}% of the variance in salary predictions.")
    elif r2_score >= 0.4:
        st.warning(f"Moderate model performance. The model explains {r2_score*100:.1f}% of the variance in salary predictions.")
    else:
        st.error(f"The model has limited predictive power, explaining only {r2_score*100:.1f}% of the variance.")

with tab3:
    st.header("ℹ️ About This Application")
    
    st.markdown("""
    ### AI Job Salary Predictor
    
    This application uses machine learning to predict salaries in the AI industry based on various job characteristics and personal qualifications.
    
    **Features Used for Prediction:**
    - Experience level and years of experience
    - Education requirements
    - Company size and location
    - Employment type (full-time, part-time, contract, freelance)
    - Remote work ratio
    - Industry sector
    - Technical skills
    - Benefits score
    
    **Models Tested:**
    - Linear Regression
    - Decision Tree Regressor
    - Random Forest Regressor
    - Gradient Boosting Regressor
    
    **Data Processing:**
    - One-hot encoding for categorical variables
    - Feature scaling for linear models
    - Skill extraction from job requirements
    - Frequency encoding for high-cardinality features
    
    **Disclaimer:**
    This is a predictive model based on historical data and should be used as a general guideline. 
    Actual salaries may vary based on specific company policies, negotiation skills, market conditions, 
    and other factors not captured in the model.
    
    ---
    *Created for Machine Learning for Developers (CAI2C08) Project*
    """)
    
    st.subheader("💡 Tips for Better Predictions")
    st.markdown("""
    - Select all relevant skills you possess for more accurate predictions
    - Consider the cost of living differences between locations
    - Remote work ratio can significantly impact salary expectations
    - Company size often correlates with salary ranges and benefits
    - Experience level should match your years of experience
    """)

st.markdown("---")
st.markdown("*AI Job Salary Predictor - Powered by Machine Learning*")