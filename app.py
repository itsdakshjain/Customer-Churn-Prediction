import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

#  Page Setup 
st.set_page_config(page_title="Customer Churn Predictor", layout="wide",initial_sidebar_state="expanded")

# Custom CSS
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007bff; color: white; }
    </style>
    """, unsafe_allow_html=True)


# ( Load Models ) 
@st.cache_resource 
def load_assets():
    m_rfc = joblib.load('models/model.pkl')    # Random Forest Model, liked best , version 3 Model
    m_svm = joblib.load('models/model_2.pkl')  # Support Vector Machine , version 2 Model
    s = joblib.load('models/scaler.pkl')       # Data Scaler
    return m_rfc, m_svm, s

model_rfc, model_svm, scaler = load_assets()

# made a sidebar for info with image, title, caption, model choice, version, my name add github linkedin links
# center all elements in sidebar ke lia used st.markdown with html
with st.sidebar:
    #col_a, col_b, col_c = st.columns([1, 2, 1])
    #with col_b:
    #    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=80)
    
    # Centered Title and Caption using HTML
    st.markdown("<h1 style='text-align: center;font-size: 35px;'>CHURN INGIGHTS</h1>", unsafe_allow_html=True)
    st.divider()
    st.caption("<p style='text-align: center; font-size: 16px;'>This dashboard uses multiple machine learning models to predict the likelihood of customer departure based on historical behavioral patterns.</p>", unsafe_allow_html=True)

    #st.divider()
    st.markdown("<h3 style='text-align: center;'>Model Selection</h3>", unsafe_allow_html=True)
    
    #  toggle for switching between RFC and SVM logic
    model_choice = st.radio(
        "Choose Intelligence Engine:",
        ("Random Forest Classifier (RFC)", "Support Vector Machines (SVM)"),
        index=0, # This makes RFC the default
        help="RFC is better at handling category logic, while SVM looks for linear patterns."
    )
    st.divider()

    # my name 
    st.markdown("<h3 style='text-align: center;'>Developed by: Daksh Jain</h3>", unsafe_allow_html=True)
    #st.markdown("<h4 style='text-align: center;'>Daksh Jain</h4>", unsafe_allow_html=True)
    #st.divider()
    
    #  Centered GitHub linkedin links
    st.markdown("""
    <div style="text-align: center;">
        <a href="https://github.com/itsdakshjain" target="_blank" style="text-decoration: none; display: block; margin-bottom: 20px;">
            <img src="https://img.shields.io/badge/GitHub-itsdakshjain-181717?style=for-the-badge&logo=github&logoColor=white" />
        </a>
        <a href="https://www.linkedin.com/in/daksh-jain-6b31772b9/" target="_blank" style="text-decoration: none; display: block;">
            <img src="https://img.shields.io/badge/LinkedIn-Daksh%20Jain-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white" />
        </a>
    </div>
""", unsafe_allow_html=True)
    
    st.divider()
    # Model Version
    st.markdown("<p style='text-align: center;'><b>Model Version:</b>4.0 (Final Deployment)</p>", unsafe_allow_html=True)
    #st.divider()


# ( Main UI )
st.title("Customer Churn Prediction Dashboard")
st.divider()
st.write("Enter the customer details below to predict the risk of them leaving.")

# (Input Form )
#with st.container(): # expander better hai 
with st.expander(" Customer Profile.", expanded=True):
    with st.form("churn_form"):
        col1, col2, col3 = st.columns(3)
        #buttons banaye
        with col1:
            st.subheader("Demographics")
            age = st.number_input("Age", min_value=18, max_value=100, value=30)
            gender = st.selectbox("Gender", options=["Male", "Female"])   

        with col2:
            st.subheader("Contract & Service")
            contract = st.selectbox("Contract Type", options=["Month-to-month", "One-Year", "Two-Year"])
            tech_support = st.selectbox("Tech Support", options=["Yes", "No"])
            internet = st.selectbox("Internet Service", options=["DSL", "Fiber Optic", "No"])

        with col3:
            st.subheader("Tenure & Charges")
            tenure = st.number_input("Tenure (Months)", min_value=0, max_value=130, value=12)
            monthly_charges = st.number_input("Monthly Charges ($)", min_value=30, max_value=150, value=70)


        submit = st.form_submit_button("Predict Churn Risk")

# ( Prediction Logic with Animation and Scroll )
import time

if submit:
    st.markdown("<div id='results'></div>", unsafe_allow_html=True) # scrolling k lie bookmark

    with st.spinner('AI is analyzing customer patterns...'):
        time.sleep(2)
        
        gender_val = 1 if gender == "Female" else 0 # encoded data 
        tech_support_val = 1 if tech_support == "Yes" else 0
        contract_one = 1 if contract == "One-Year" else 0
        contract_two = 1 if contract == "Two-Year" else 0
        internet_dsl = 1 if internet == "DSL" else 0
        internet_fiber = 1 if internet == "Fiber Optic" else 0
        
        # same order of features as training data
        input_data = pd.DataFrame([[
            age, gender_val, tenure, monthly_charges, tech_support_val,
            contract_one, contract_two, internet_dsl, internet_fiber
        ]], columns=['Age', 'Gender', 'Tenure', 'MonthlyCharges', 'TechSupport', 
                    'ContractType_One-Year', 'ContractType_Two-Year', 
                    'InternetService_DSL', 'InternetService_Fiber Optic'])
  
        if model_choice == "Random Forest Classifier (RFC)":
            # Use RFC model (model_rfc) on raw data but still scaling as v3 me scaled data lia
            input_scaled = scaler.transform(input_data)
            probability = model_rfc.predict_proba(input_scaled)[0][1]
        else:
            # Use SVM model (model_svm)  on scaled data
            input_scaled = scaler.transform(input_data)
            probability = model_svm.predict_proba(input_scaled)[0][1]

    st.toast("Prediction Ready! Scroll down for analysis.")
    st.success("Prediction Ready! Scroll down for analysis.")

    # tells the browser to jump to the 'results' bookmark
    st.components.v1.html(
        f"""
        <script>
            window.parent.document.getElementById('results').scrollIntoView({{behavior: 'smooth'}});
        </script>
        """,
        height=0,)
    
    st.divider()
        
    res_col1, res_col2 = st.columns([1, 2])
        
    with res_col1: # show probability with metric and progress bar
        st.write("###") 
        st.metric(label="Calculated Risk", value=f"{probability:.2%}")
        st.progress(probability)
        # Visual Risk Zone banaya h inside coulmn 1
        fig, ax = plt.subplots(figsize=(6, 1.5))
        # Create a background color bar (Green, Yellow, Red)
        ax.barh(0, 0.45, color='#2ecc71', alpha=0.3)
        ax.barh(0, 0.30, left=0.45, color='#f1c40f', alpha=0.3)
        ax.barh(0, 0.25, left=0.75, color='#e74c3c', alpha=0.3)
        # Vertical line for current customer
        ax.axvline(probability, color='black', linewidth=3)
        ax.set_xlim(0, 1)
        ax.set_yticks([])
        ax.set_title("Churn Risk Zone", fontsize=10)
        st.pyplot(fig)

    with res_col2:
        if probability < 0.45:
            st.success("### Low Risk Profile")
            st.write("This customer is stable. Continue standard engagement.")
            st.balloons()
        elif probability < 0.75:
            st.warning("### Medium Risk Profile")
            st.write("Risk detected. Suggesting proactive outreach or loyalty discount.")
        else:
            st.error("### High Risk Profile")
            st.write("Critical Churn Risk. Immediate retention strategy required.")

    with st.expander("See Risk Analysis Factors",expanded=True): # one more expander for feature importance
        st.write("The model identifies **Contract Type**, **Monthly Charges**, and **Internet Service** as the primary drivers for this specific prediction.")

# outside the submit block , graphs banaye 

st.divider()
st.subheader("Key Factors Driving This Prediction") #
# which model's feature importance to show for bar graph
active_model = model_rfc if model_choice == "Random Forest Classifier (RFC)" else model_svm

# Feature Importance Visualization
if hasattr(active_model, 'feature_importances_'):
    importances = active_model.feature_importances_  # This is for RFC
elif hasattr(active_model, 'coef_'):                 # This is for Linear SVM
    importances = np.abs(active_model.coef_[0])
else:
    importances = None

if importances is not None:
    # We use a fixed list of names because 'input_data' isn't ready yet
    feature_names = ['Age', 'Gender', 'Tenure', 'MonthlyCharges', 'TechSupport', 
                     'ContractType_One-Year', 'ContractType_Two-Year', 
                     'InternetService_DSL', 'InternetService_Fiber Optic']
    
    feat_importances = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=True)

    st.bar_chart(feat_importances.set_index('Feature'))
    st.caption("This chart shows which variables had the most influence on the model's decision.")
else:
    st.info("Feature importance visualization is not available for this specific model type.")
    
#one more outside graph for general risk scale with marker

st.divider()
st.subheader("General Churn Risk Scale")
st.write("Use this scale to understand how we measure customer risk.")

# This part checks if a prediction has been made yet
current_risk = probability if 'probability' in locals() else 0

fig_end, ax_end = plt.subplots(figsize=(10, 1))
# Background Color Zones
ax_end.barh(0, 0.45, color='#2ecc71', alpha=0.3) # Safe Zone
ax_end.barh(0, 0.30, left=0.45, color='#f1c40f', alpha=0.3) # Warning Zone
ax_end.barh(0, 0.25, left=0.75, color='#e74c3c', alpha=0.3) # Danger Zone

# The Black Marker Line
ax_end.axvline(current_risk, color='black', linewidth=4)

# Label that only shows up after prediction 
if 'probability' in locals():
    ax_end.text(current_risk, 0.6, f"RESULT: {current_risk:.1%}", 
                ha='center', fontweight='bold')

# Formatting
ax_end.set_xlim(0, 1)
ax_end.set_yticks([])
ax_end.set_xticks([0, 0.45, 0.75, 1])
ax_end.set_xticklabels(['0%', 'Low Risk', 'High Risk', '100%'])
sns.despine(left=True)

st.pyplot(fig_end)




st.write("") # Add some space
st.write("") 
st.divider()

# Create three columns for the footer
f_col1, f_col2, f_col3 , f_col4 = st.columns([2, 0.5, 1,1])

with f_col1:
    st.caption("© 2026 | **Customer Churn Analytics Dashboard. All Rights Reserved**")
    
    
with f_col2:
    st.caption(f"**Version:** 4.0 ")
    
with f_col3:
    st.caption("Developed by **Daksh Jain**")

with f_col4:
    # Social links as small clickable text or icons
    st.markdown(f"""
        <div style="text-align: right;">
                <a href="https://github.com/itsdakshjain" target="_blank" style="text-decoration:none;">
                <img src="https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white" height="20">
            </a>
            <a href="https://www.linkedin.com/in/daksh-jain-6b31772b9/" target="_blank" style="text-decoration:none;">
                <img src="https://img.shields.io/badge/LinkedIn-0A66C2?style=flat&logo=linkedin&logoColor=white" height="20">
            </a>
                
        </div>
                
    """, unsafe_allow_html=True)