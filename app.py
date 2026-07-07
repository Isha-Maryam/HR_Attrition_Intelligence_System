import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.graph_objects as go
import os

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="HR Attrition Intelligence System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# CUSTOM CSS — Dark Premium HR Theme & Keyframe Animations
# ============================================================
st.markdown("""
<style>
    /* Global backgrounds and layout overrides */
    .stApp {
        background-color: #0F172A !important;
        color: #FFFFFF !important;
    }
    
    /* Ensure all text blocks and headings are white */
    h1, h2, h3, h4, h5, h6, p, span, div, label {
        color: #FFFFFF !important;
    }
    
    /* Input field label styles */
    div[data-testid="stWidgetLabel"] p {
        color: #E2E8F0 !important;
        font-weight: 500 !important;
        font-size: 14px !important;
    }
    
    /* Premium Banner Card Styling */
    .premium-banner {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        position: relative;
        padding: 32px 24px;
        border-radius: 12px;
        border: 1px solid #334155;
        margin-bottom: 30px;
        overflow: hidden;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
    }

    .premium-banner::before {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        background-image: radial-gradient(#475569 1.5px, transparent 1.5px);
        background-size: 20px 20px;
        opacity: 0.35;
        pointer-events: none;
    }

    /* Dark card container styling for bordered container components */
    div[data-testid="stBorderedContainer"] {
        background-color: #1E293B !important;
        border: 1px solid #334155 !important;
        border-radius: 12px !important;
        padding: 24px !important;
        margin-bottom: 24px !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06) !important;
        animation: fadeIn 0.6s ease-out;
    }
    
    /* Thin colored top borders on input cards via child markers */
    div[data-testid="stBorderedContainer"]:has(> div[data-testid="stMarkdownContainer"] > div.personal-card-marker) {
        border-top: 4px solid #3B82F6 !important;
    }
    div[data-testid="stBorderedContainer"]:has(> div[data-testid="stMarkdownContainer"] > div.job-card-marker) {
        border-top: 4px solid #8B5CF6 !important;
    }
    div[data-testid="stBorderedContainer"]:has(> div[data-testid="stMarkdownContainer"] > div.work-card-marker) {
        border-top: 4px solid #F97316 !important;
    }
    div[data-testid="stBorderedContainer"]:has(> div[data-testid="stMarkdownContainer"] > div.sat-card-marker) {
        border-top: 4px solid #10B981 !important;
    }
    
    /* Custom tab selector styles */
    .custom-tab-bar {
        display: none;
    }
    
    /* Base style for tab buttons inside columns */
    div:has(> .custom-tab-bar) ~ div[data-testid="column"] button {
        background-color: transparent !important;
        color: #94A3B8 !important;
        border: none !important;
        border-bottom: 3px solid #1E293B !important;
        border-radius: 0px !important;
        font-weight: 600 !important;
        padding: 12px 0px !important;
        margin: 0 !important;
        box-shadow: none !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
        animation: none !important;
    }
    
    /* Active tab button */
    div:has(> .custom-tab-bar) ~ div[data-testid="column"] button[kind="primary"] {
        color: #3B82F6 !important;
        border-bottom: 3px solid #3B82F6 !important;
        font-weight: 700 !important;
    }
    
    /* Inactive tab button hover */
    div:has(> .custom-tab-bar) ~ div[data-testid="column"] button:hover {
        color: #FFFFFF !important;
        border-bottom: 3px solid #475569 !important;
    }
    
    /* Active tab button hover overrides */
    div:has(> .custom-tab-bar) ~ div[data-testid="column"] button[kind="primary"]:hover {
        border-bottom: 3px solid #3B82F6 !important;
        color: #3B82F6 !important;
    }
    
    /* Custom marker to target the Analyze button */
    .custom-analyze-btn {
        display: none;
    }
    
    /* Glowing Green Analyze Button styling with active transitions */
    div:has(> .custom-analyze-btn) + div.stButton > button {
        background: linear-gradient(135deg, #059669 0%, #10B981 100%) !important;
        color: white !important;
        font-weight: 700 !important;
        font-size: 16px !important;
        padding: 14px 28px !important;
        border-radius: 30px !important;
        border: none !important;
        box-shadow: 0 0 15px rgba(16, 185, 129, 0.4) !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        animation: pulse 2s infinite !important;
        display: block !important;
        margin: 24px auto !important;
        width: 100% !important;
        max-width: 320px !important;
    }

    div:has(> .custom-analyze-btn) + div.stButton > button:hover {
        box-shadow: 0 0 25px rgba(16, 185, 129, 0.8) !important;
        transform: translateY(-2px) scale(1.02) !important;
        background: linear-gradient(135deg, #059669 0%, #34D399 100%) !important;
    }

    div:has(> .custom-analyze-btn) + div.stButton > button:active {
        transform: scale(0.95) !important;
        background: linear-gradient(135deg, #1D4ED8 0%, #1E40AF 100%) !important;
        box-shadow: 0 0 30px rgba(59, 130, 246, 0.9) !important;
    }
    
    /* Custom styled cards for Root Causes (Red left border) */
    .root-cause-card {
        background-color: #1E293B;
        border-left: 5px solid #EF4444;
        border-top: 1px solid #334155;
        border-right: 1px solid #334155;
        border-bottom: 1px solid #334155;
        padding: 16px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 12px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Custom styled cards for Action Plan (Blue left border) */
    .action-plan-card {
        background-color: #1E293B;
        border-left: 5px solid #3B82F6;
        border-top: 1px solid #334155;
        border-right: 1px solid #334155;
        border-bottom: 1px solid #334155;
        padding: 16px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 12px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Radio and widget custom overrides to enforce dark readability */
    div[data-testid="stRadio"] label {
        color: #FFFFFF !important;
        font-weight: 500 !important;
    }
    
    /* Fix potential white block in select sliders */
    div[data-testid="stSlider"] {
        color: #FFFFFF !important;
    }
    
    /* Custom alert success box */
    .success-alert {
        background-color: rgba(16, 185, 129, 0.15);
        border: 1px solid #10B981;
        color: #10B981 !important;
        padding: 16px;
        border-radius: 8px;
        margin-bottom: 24px;
        font-weight: 500;
        animation: fadeInDown 0.6s ease-out forwards;
    }
    
    /* ANIMATIONS */
    
    /* 1. Gauge container: fadeIn + scaleUp over 0.6s */
    .gauge-container {
        animation: fadeInScale 0.6s cubic-bezier(0.34, 1.56, 0.64, 1) forwards;
    }
    
    @keyframes fadeInScale {
        0% {
            opacity: 0;
            transform: scale(0.9);
        }
        100% {
            opacity: 1;
            transform: scale(1);
        }
    }
    
    /* 2. Each cause/action card: slideInLeft with 0.1s stagger delay */
    .stagger-card {
        opacity: 0;
        animation: slideInLeft 0.5s cubic-bezier(0.16, 1, 0.3, 1) forwards;
    }
    
    .stagger-1 { animation-delay: 0.08s; }
    .stagger-2 { animation-delay: 0.16s; }
    .stagger-3 { animation-delay: 0.24s; }
    .stagger-4 { animation-delay: 0.32s; }
    .stagger-5 { animation-delay: 0.40s; }
    .stagger-6 { animation-delay: 0.48s; }
    
    @keyframes slideInLeft {
        0% {
            opacity: 0;
            transform: translateX(-30px);
        }
        100% {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    /* 3. Analyze button pulse box-shadow breathing effect */
    @keyframes pulse {
        0% {
            box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.5);
        }
        70% {
            box-shadow: 0 0 0 10px rgba(16, 185, 129, 0);
        }
        100% {
            box-shadow: 0 0 0 0 rgba(16, 185, 129, 0);
        }
    }
    
    /* 4. Risk badge bounceIn when appearing */
    .risk-badge {
        display: inline-block;
        padding: 8px 24px;
        font-size: 15px;
        font-weight: 700;
        border-radius: 50px;
        letter-spacing: 1px;
        animation: bounceIn 0.8s cubic-bezier(0.215, 0.610, 0.355, 1) forwards;
    }
    
    @keyframes bounceIn {
        0% {
            opacity: 0;
            transform: scale3d(0.3, 0.3, 0.3);
        }
        20% {
            transform: scale3d(1.1, 1.1, 1.1);
        }
        40% {
            transform: scale3d(0.9, 0.9, 0.9);
        }
        60% {
            opacity: 1;
            transform: scale3d(1.03, 1.03, 1.03);
        }
        80% {
            transform: scale3d(0.97, 0.97, 0.97);
        }
        100% {
            opacity: 1;
            transform: scale3d(1, 1, 1);
        }
    }
    
    /* 5. Success message: fadeInDown */
    @keyframes fadeInDown {
        0% {
            opacity: 0;
            transform: translateY(-20px);
        }
        100% {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Simple general fadeIn for cards */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODEL & CONFIG
# ============================================================
@st.cache_resource
def load_model():
    model = joblib.load('models/xgboost_attrition_model.pkl')
    with open('models/model_config.json', 'r') as f:
        config = json.load(f)
    return model, config

try:
    model, config = load_model()
    threshold = config['best_threshold']
    feature_names = config['feature_names']
except Exception as e:
    st.error(f"⚠️ Model files not found. Please ensure models/ folder exists. Error: {e}")
    st.stop()

# ============================================================
# RETENTION RECOMMENDATION ENGINE
# ============================================================
def get_retention_plan(employee_data, risk_score):
    actions = []
    root_causes = []
    retention_boost = 0

    if employee_data['OverTime'] == 1:
        root_causes.append("Working overtime consistently")
        actions.append("Remove from overtime roster this quarter")
        retention_boost += 15

    if employee_data['Promotion_Lag'] > 5:
        root_causes.append("No promotion in 5+ years")
        actions.append("Schedule promotion review within 30 days")
        retention_boost += 20
    elif employee_data['Promotion_Lag'] > 3:
        root_causes.append("Promotion overdue (3-5 years)")
        actions.append("Add to promotion pipeline for next cycle")
        retention_boost += 10

    if employee_data['Satisfaction_Score'] < 2.0:
        root_causes.append("Very low satisfaction score")
        actions.append("Immediate 1-on-1 with HR manager")
        actions.append("Enroll in employee engagement program")
        retention_boost += 18
    elif employee_data['Satisfaction_Score'] < 2.5:
        root_causes.append("Below average satisfaction")
        actions.append("Schedule quarterly check-in meeting")
        retention_boost += 8

    if employee_data['WorkLife_Stress_Index'] >= 3:
        root_causes.append("High burnout risk detected")
        actions.append("Offer flexible working hours")
        actions.append("Consider mental health support resources")
        retention_boost += 15

    if employee_data['Salary_Growth_Rate'] < 1.0:
        root_causes.append("Salary growth not matching tenure")
        actions.append("Compensation review recommended")
        retention_boost += 12

    if employee_data['Manager_Relationship_Risk'] > 1.5:
        root_causes.append("Poor manager relationship detected")
        actions.append("Consider team/manager reassignment")
        retention_boost += 10

    if employee_data.get('Age', 30) < 26:
        root_causes.append("Early career employee — high mobility")
        actions.append("Assign mentor and clear growth roadmap")
        retention_boost += 7

    base_retention = 100 - (risk_score * 100)
    retention_if_act = min(base_retention + retention_boost, 92)

    if risk_score >= 0.70:
        risk_level = "CRITICAL RISK"
        risk_class = "risk-critical"
        risk_color = "#EF4444"
    elif risk_score >= threshold:
        risk_level = "MODERATE RISK"
        risk_class = "risk-moderate"
        risk_color = "#F59E0B"
    else:
        risk_level = "LOW RISK"
        risk_class = "risk-low"
        risk_color = "#10B981"

    return {
        'risk_score': round(risk_score * 100, 1),
        'risk_level': risk_level,
        'risk_class': risk_class,
        'risk_color': risk_color,
        'root_causes': root_causes if root_causes else ["No major risk factors detected"],
        'action_plan': actions if actions else ["Continue current HR practices"],
        'retention_if_act': round(retention_if_act, 1),
        'current_retention': round(base_retention, 1)
    }

# ============================================================
# GAUGE CHART
# ============================================================
def create_gauge(risk_pct, risk_color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_pct,
        number={'suffix': "%", 'font': {'size': 40, 'color': '#FFFFFF'}},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#94A3B8"},
            'bar': {'color': risk_color, 'thickness': 0.35},
            'bgcolor': "#334155",
            'borderwidth': 0,
            'steps': [
                {'range': [0, 40], 'color': 'rgba(16, 185, 129, 0.15)'},
                {'range': [40, 70], 'color': 'rgba(245, 158, 11, 0.15)'},
                {'range': [70, 100], 'color': 'rgba(239, 68, 68, 0.15)'}
            ],
            'threshold': {
                'line': {'color': risk_color, 'width': 4},
                'thickness': 0.8,
                'value': risk_pct
            }
        }
    ))
    fig.update_layout(
        height=240,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="#1E293B",
        font={'family': "Inter, sans-serif", 'color': '#FFFFFF'}
    )
    return fig

# ============================================================
# PREMIUM HEADER BANNER (with subtle dot grid overlay)
# ============================================================
st.markdown("""
<div class="premium-banner">
    <h1 style="margin: 0 0 10px 0; color: #FFFFFF; font-size: 28px; font-weight: 800; text-align: center;">HR Attrition Intelligence System</h1>
    <p style="margin: 0; color: #94A3B8 !important; font-size: 15px; text-align: center;">AI-powered employee retention prediction with actionable HR recommendations</p>
</div>
""", unsafe_allow_html=True)

# Initialize Session State
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None
if 'employee_data' not in st.session_state:
    st.session_state.employee_data = None
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = "Employee Input"
if 'show_success' not in st.session_state:
    st.session_state.show_success = False

# Render Custom Tab Bar
st.markdown('<div class="custom-tab-bar"></div>', unsafe_allow_html=True)
col_t1, col_t2, col_t3, _ = st.columns([1, 1, 1, 2])

with col_t1:
    is_active = (st.session_state.active_tab == "Employee Input")
    if st.button("Employee Input", type="primary" if is_active else "secondary", use_container_width=True):
        st.session_state.active_tab = "Employee Input"
        st.rerun()
with col_t2:
    is_active = (st.session_state.active_tab == "Results & Insights")
    if st.button("Results & Insights", type="primary" if is_active else "secondary", use_container_width=True):
        st.session_state.active_tab = "Results & Insights"
        st.rerun()
with col_t3:
    is_active = (st.session_state.active_tab == "About")
    if st.button("About", type="primary" if is_active else "secondary", use_container_width=True):
        st.session_state.active_tab = "About"
        st.rerun()

st.markdown("<br>", unsafe_allow_html=True)

# ============================================================
# TAB 1 — EMPLOYEE INPUT
# ============================================================
if st.session_state.active_tab == "Employee Input":
    st.markdown("<h3 style='margin-bottom: 20px;'>Profile Registration</h3>", unsafe_allow_html=True)
    
    # 1. Personal Profile Section (with marker for blue top border)
    with st.container(border=True):
        st.markdown('<div class="personal-card-marker"></div>', unsafe_allow_html=True)
        st.markdown("#### Personal Profile", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.slider("Age", 18, 60, 30, key="age_slider")
        with col2:
            gender = st.radio("Gender", ["Male", "Female"], horizontal=True, key="gender_radio")
        with col3:
            marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"], key="marital_select")
            
        col4, col5, col6 = st.columns(3)
        with col4:
            education_field = st.selectbox("Education Field", [
                "Life Sciences", "Medical", "Marketing", "Technical Degree", "Other", "Human Resources"
            ], key="edu_select")
        with col5:
            st.write("")
        with col6:
            st.write("")

    # 2. Job Details Section (with marker for purple top border)
    with st.container(border=True):
        st.markdown('<div class="job-card-marker"></div>', unsafe_allow_html=True)
        st.markdown("#### Job Details", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            department = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"], key="dept_select")
        with col2:
            job_role = st.selectbox("Job Role", [
                "Sales Executive", "Research Scientist", "Laboratory Technician",
                "Manufacturing Director", "Healthcare Representative", "Manager",
                "Sales Representative", "Research Director", "Human Resources"
            ], key="role_select")
        with col3:
            job_level_label = st.select_slider(
                "Job Level",
                options=["1 - Entry Level", "2 - Associate", "3 - Mid-Level", "4 - Senior", "5 - Executive"],
                value="2 - Associate",
                key="level_select_slider"
            )
            job_level_map = {
                "1 - Entry Level": 1,
                "2 - Associate": 2,
                "3 - Mid-Level": 3,
                "4 - Senior": 4,
                "5 - Executive": 5
            }
            job_level = job_level_map[job_level_label]
            
        col4, col5, col6 = st.columns(3)
        with col4:
            business_travel = st.selectbox("Business Travel", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"], key="travel_select")
        with col5:
            st.write("")
        with col6:
            st.write("")

    # 3. Work Conditions Section (with marker for orange top border)
    with st.container(border=True):
        st.markdown('<div class="work-card-marker"></div>', unsafe_allow_html=True)
        st.markdown("#### Work Conditions", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            overtime = st.radio("Overtime Required", ["No", "Yes"], horizontal=True, key="ot_radio")
        with col2:
            distance_from_home = st.slider("Distance From Home (km)", 1, 30, 10, key="dist_slider")
        with col3:
            work_life_balance_label = st.select_slider(
                "Work-Life Balance",
                options=["1 - Poor", "2 - Fair", "3 - Good", "4 - Excellent"],
                value="3 - Good",
                key="wlb_select_slider"
            )
            wlb_map = {
                "1 - Poor": 1,
                "2 - Fair": 2,
                "3 - Good": 3,
                "4 - Excellent": 4
            }
            work_life_balance = wlb_map[work_life_balance_label]
            
        col4, col5, col6 = st.columns(3)
        with col4:
            env_sat_label = st.select_slider(
                "Environment Satisfaction",
                options=["1 - Low", "2 - Medium", "3 - High", "4 - Very High"],
                value="3 - High",
                key="env_select_slider"
            )
            env_sat_map = {
                "1 - Low": 1,
                "2 - Medium": 2,
                "3 - High": 3,
                "4 - Very High": 4
            }
            environment_satisfaction = env_sat_map[env_sat_label]
        with col5:
            st.write("")
        with col6:
            st.write("")

    # 4. Satisfaction & Career Section (with marker for green top border)
    with st.container(border=True):
        st.markdown('<div class="sat-card-marker"></div>', unsafe_allow_html=True)
        st.markdown("#### Satisfaction & Career", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            job_sat_label = st.select_slider(
                "Job Satisfaction",
                options=["1 - Low", "2 - Medium", "3 - High", "4 - Very High"],
                value="3 - High",
                key="job_sat_select_slider"
            )
            job_sat_map = {
                "1 - Low": 1,
                "2 - Medium": 2,
                "3 - High": 3,
                "4 - Very High": 4
            }
            job_satisfaction = job_sat_map[job_sat_label]
        with col2:
            rel_sat_label = st.select_slider(
                "Relationship Satisfaction",
                options=["1 - Low", "2 - Medium", "3 - High", "4 - Very High"],
                value="3 - High",
                key="rel_sat_select_slider"
            )
            rel_sat_map = {
                "1 - Low": 1,
                "2 - Medium": 2,
                "3 - High": 3,
                "4 - Very High": 4
            }
            relationship_satisfaction = rel_sat_map[rel_sat_label]
        with col3:
            monthly_income = st.number_input("Monthly Income ($)", 1000, 20000, 5000, step=100, key="income_input")
            
        col4, col5, col6 = st.columns(3)
        with col4:
            percent_salary_hike = st.slider("Percent Salary Hike", 11, 25, 15, key="hike_slider")
        with col5:
            total_working_years = st.slider("Total Working Years", 0, 40, 8, key="working_slider")
        with col6:
            years_at_company = st.slider("Years At Company", 0, 40, 5, key="tenure_slider")
            
        col7, col8, col9 = st.columns(3)
        with col7:
            years_since_promotion = st.slider("Years Since Last Promotion", 0, 15, 2, key="promo_slider")
        with col8:
            years_with_curr_manager = st.slider("Years With Current Manager", 0, 17, 3, key="mgr_slider")
        with col9:
            num_companies_worked = st.slider("Number of Companies Worked", 0, 10, 2, key="comp_slider")

    # Center-aligned glowing blue button with pulse and sibling CSS markers
    st.markdown('<div class="custom-analyze-btn"></div>', unsafe_allow_html=True)
    predict_btn = st.button("Analyze Attrition Risk", key="predict_action_button", use_container_width=True)

    if predict_btn:
        with st.spinner("Executing XGBoost risk evaluation..."):
            # ── Feature Engineering ──
            satisfaction_score = (job_satisfaction + environment_satisfaction +
                                   relationship_satisfaction + work_life_balance) / 4
            promotion_lag = years_at_company - years_since_promotion
            tenure_per_company = total_working_years / (num_companies_worked + 1)
            overtime_val = 1 if overtime == "Yes" else 0
            worklife_stress_index = overtime_val * (5 - work_life_balance)
            salary_growth_rate = percent_salary_hike / (years_at_company + 1)
            experience_age_ratio = total_working_years / age
            manager_relationship_risk = (5 - relationship_satisfaction) * (1 / (years_with_curr_manager + 1))

            employee_data = {
                'Age': age,
                'OverTime': overtime_val,
                'Satisfaction_Score': satisfaction_score,
                'Promotion_Lag': promotion_lag,
                'Tenure_Per_Company': tenure_per_company,
                'WorkLife_Stress_Index': worklife_stress_index,
                'Salary_Growth_Rate': salary_growth_rate,
                'Experience_Age_Ratio': experience_age_ratio,
                'Manager_Relationship_Risk': manager_relationship_risk,
                'JobRole': job_role
            }

            # ── Create model feature dictionary ──
            input_dict = {col: 0 for col in feature_names}

            raw_values = {
                'Age': age, 'DistanceFromHome': distance_from_home,
                'Gender': 1 if gender == 'Male' else 0,
                'JobLevel': job_level, 'JobSatisfaction': job_satisfaction,
                'EnvironmentSatisfaction': environment_satisfaction,
                'RelationshipSatisfaction': relationship_satisfaction,
                'WorkLifeBalance': work_life_balance,
                'OverTime': overtime_val,
                'MonthlyIncome': monthly_income,
                'PercentSalaryHike': percent_salary_hike,
                'TotalWorkingYears': total_working_years,
                'YearsAtCompany': years_at_company,
                'YearsSinceLastPromotion': years_since_promotion,
                'YearsWithCurrManager': years_with_curr_manager,
                'NumCompaniesWorked': num_companies_worked,
                'Satisfaction_Score': satisfaction_score,
                'Promotion_Lag': promotion_lag,
                'Tenure_Per_Company': tenure_per_company,
                'WorkLife_Stress_Index': worklife_stress_index,
                'Salary_Growth_Rate': salary_growth_rate,
                'Experience_Age_Ratio': experience_age_ratio,
                'Manager_Relationship_Risk': manager_relationship_risk,
            }
            
            for k, v in raw_values.items():
                if k in input_dict:
                    input_dict[k] = v

            # One-hot encoding mapping
            ohe_map = {
                f'Department_{department}': 1,
                f'JobRole_{job_role}': 1,
                f'EducationField_{education_field}': 1,
                f'MaritalStatus_{marital_status}': 1,
                f'BusinessTravel_{business_travel}': 1,
            }
            for k, v in ohe_map.items():
                if k in input_dict:
                    input_dict[k] = v

            X_input = pd.DataFrame([input_dict])[feature_names]

            # ── Perform prediction ──
            risk_proba = model.predict_proba(X_input)[:, 1][0]
            result = get_retention_plan(employee_data, risk_proba)

            # Store in session state and switch to Results tab
            st.session_state.prediction_made = True
            st.session_state.prediction_result = result
            st.session_state.employee_data = employee_data
            st.session_state.show_success = True
            st.session_state.active_tab = "Results & Insights"
            st.rerun()

# ============================================================
# TAB 2 — RESULTS & INSIGHTS
# ============================================================
elif st.session_state.active_tab == "Results & Insights":
    if not st.session_state.prediction_made or st.session_state.prediction_result is None:
        # Elegant empty state card
        st.markdown("""
        <div style="text-align: center; padding: 60px 40px; background-color: #1E293B; border: 1px dashed #475569; border-radius: 12px; margin-top: 20px;">
            <div style="font-size: 48px; margin-bottom: 20px;">🔍</div>
            <h3 style="color: #FFFFFF !important; margin-bottom: 10px;">No Prediction Results Available</h3>
            <p style="color: #94A3B8 !important; max-width: 500px; margin: 0 auto 20px auto; font-size: 15px; line-height: 1.6;">
                Fill in the employee details in the <b>Employee Input</b> tab and select <b>Analyze Attrition Risk</b> to run the AI prediction models and generate retention strategies.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        result = st.session_state.prediction_result
        employee_data = st.session_state.employee_data
        
        # 5. Success Message (renders at the top with fadeInDown animation)
        if st.session_state.show_success:
            st.markdown("""
            <div class="success-alert">
                <span>Analysis complete! View the attrition risk assessment and recommendations below.</span>
            </div>
            """, unsafe_allow_html=True)
            # Toggle off show_success so it doesn't animate on every single action in Tab 2
            st.session_state.show_success = False
            
        # Risk Badge Pills Setup
        badge_text = result['risk_level'].replace(" RISK", "")
        badge_bg = ""
        badge_text_color = ""
        badge_border = ""
        if "CRITICAL" in badge_text:
            badge_bg = "rgba(239, 68, 68, 0.15)"
            badge_text_color = "#EF4444"
            badge_border = "1px solid #EF4444"
        elif "MODERATE" in badge_text:
            badge_bg = "rgba(245, 158, 11, 0.15)"
            badge_text_color = "#F59E0B"
            badge_border = "1px solid #F59E0B"
        else:
            badge_bg = "rgba(16, 185, 129, 0.15)"
            badge_text_color = "#10B981"
            badge_border = "1px solid #10B981"

        col_left, col_right = st.columns([1, 1.2])

        with col_left:
            st.markdown("<h3 style='margin-bottom: 5px;'>Risk Evaluation</h3>", unsafe_allow_html=True)
            
            # Results Tab Summary Sentence
            st.markdown(f"<p style='font-size: 14px; color: #94A3B8 !important; margin-bottom: 15px;'>Based on the profile provided, this employee shows <b>{result['risk_score']:.1f}%</b> probability of leaving.</p>", unsafe_allow_html=True)
            
            # Left side card wrapper
            with st.container(border=True):
                # 1. Gauge chart with smooth fadeIn + scaleUp
                st.markdown('<div class="gauge-container">', unsafe_allow_html=True)
                st.plotly_chart(create_gauge(result['risk_score'], result['risk_color']), use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # 2. Risk Badge with bounceIn animation
                st.markdown(f"""
                <div style="text-align: center; margin-bottom: 24px;">
                    <span class="risk-badge" style="
                        background-color: {badge_bg};
                        color: {badge_text_color};
                        border: {badge_border};
                        box-shadow: 0 0 10px {badge_bg};
                    ">{badge_text} ATTRITION RISK</span>
                </div>
                """, unsafe_allow_html=True)
                
                # 3. Retention probability (metrics comparison boxes formatted with float precision)
                st.markdown("<h5 style='margin-bottom: 12px; font-weight: 600;'>Retention Probability Compare</h5>", unsafe_allow_html=True)
                m_col1, m_col2 = st.columns(2)
                with m_col1:
                    st.markdown(f"""
                    <div style="background-color: #0F172A; border: 1px solid #334155; padding: 16px; border-radius: 8px; text-align: center;">
                        <div style="color: #94A3B8; font-size: 11px; font-weight: 600; text-transform: uppercase; margin-bottom: 4px;">Without HR Action</div>
                        <div style="color: #EF4444; font-size: 28px; font-weight: 700; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{result['current_retention']:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                with m_col2:
                    diff_boost = round(result['retention_if_act'] - result['current_retention'], 1)
                    st.markdown(f"""
                    <div style="background-color: #0F172A; border: 1px solid #334155; padding: 16px; border-radius: 8px; text-align: center;">
                        <div style="color: #94A3B8; font-size: 11px; font-weight: 600; text-transform: uppercase; margin-bottom: 4px;">If HR Acts Now</div>
                        <div style="color: #10B981; font-size: 28px; font-weight: 700; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{result['retention_if_act']:.1f}%</div>
                        <div style="color: #10B981; font-size: 11px; font-weight: 600; margin-top: 4px;">▲ +{diff_boost:.1f}% Boost</div>
                    </div>
                    """, unsafe_allow_html=True)

        with col_right:
            st.markdown("<h3 style='margin-bottom: 15px;'>Strategy & Key Drivers</h3>", unsafe_allow_html=True)
            
            # Root causes list (red left-border cards with stagger animation)
            st.markdown("<h5 style='margin-bottom: 12px; color: #EF4444;'>Key Attrition Drivers</h5>", unsafe_allow_html=True)
            for i, cause in enumerate(result['root_causes']):
                stagger_class = f"stagger-{min(i + 1, 6)}"
                st.markdown(f"""
                <div class="root-cause-card stagger-card {stagger_class}">
                    <h5 style="margin: 0 0 4px 0; font-size: 14px; font-weight: 600; color: #EF4444 !important;">Risk Factor Identified</h5>
                    <p style="margin: 0; color: #E2E8F0; font-size: 14px;">{cause}</p>
                </div>
                """, unsafe_allow_html=True)
                
            st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)
            
            # HR Action Plan list (blue left-border cards with stagger animation)
            st.markdown("<h5 style='margin-bottom: 12px; color: #3B82F6;'>Recommended Action Plan</h5>", unsafe_allow_html=True)
            for i, action in enumerate(result['action_plan']):
                stagger_class = f"stagger-{min(i + 1, 6)}"
                st.markdown(f"""
                <div class="action-plan-card stagger-card {stagger_class}">
                    <h5 style="margin: 0 0 4px 0; font-size: 14px; font-weight: 600; color: #3B82F6 !important;">HR Intervention</h5>
                    <p style="margin: 0; color: #E2E8F0; font-size: 14px;">{action}</p>
                </div>
                """, unsafe_allow_html=True)

# ============================================================
# TAB 3 — ABOUT
# ============================================================
elif st.session_state.active_tab == "About":
    st.markdown("<h3 style='margin-bottom: 20px;'>About the Intelligence System</h3>", unsafe_allow_html=True)
    
    col_about_1, col_about_2 = st.columns([1.2, 1])
    
    with col_about_1:
        with st.container(border=True):
            st.markdown("#### Project Description", unsafe_allow_html=True)
            st.markdown("""
            The **HR Attrition Intelligence System** is an advanced AI-powered decision support tool designed to assist Human Resources professionals in proactively identifying and managing employee turnover risks.
            
            By combining the predictive power of a high-performance **XGBoost machine learning model** with a rule-based **retention recommendation engine**, the system goes beyond simple classification. It analyzes individual employee metrics to identify precise root causes for attrition and generates tailored, prescriptive retention plans.
            """)
            
        with st.container(border=True):
            st.markdown("#### How It Works", unsafe_allow_html=True)
            st.markdown("""
            1. **Employee Profile Input:** HR enters employee demographics, career details, work conditions, and satisfaction ratings in the system.
            2. **XGBoost Risk Score Estimation:** A pre-trained XGBoost Classifier evaluates the input vector, outputting an attrition probability score optimized via GridSearchCV.
            3. **Root Cause Extraction:** The system parses inputs to identify risk factors, such as high-stress overtime, salary growth lagging behind tenure, or long commute distance.
            4. **Actionable HR Intervention:** A specialized rule engine compiles specific strategies corresponding to the root causes, along with estimating the prospective retention improvement rate.
            """)

        with st.container(border=True):
            st.markdown("#### Technology Stack", unsafe_allow_html=True)
            st.markdown("""
            <div style="display: flex; gap: 8px; flex-wrap: wrap; margin-top: 10px;">
                <span style="background-color: #1E3A8A; color: #60A5FA; padding: 6px 14px; border-radius: 20px; font-size: 12px; font-weight: 600; border: 1px solid #3B82F6;">Python</span>
                <span style="background-color: #1E3A8A; color: #60A5FA; padding: 6px 14px; border-radius: 20px; font-size: 12px; font-weight: 600; border: 1px solid #3B82F6;">XGBoost 2.0</span>
                <span style="background-color: #1E3A8A; color: #60A5FA; padding: 6px 14px; border-radius: 20px; font-size: 12px; font-weight: 600; border: 1px solid #3B82F6;">Scikit-Learn</span>
                <span style="background-color: #1E3A8A; color: #60A5FA; padding: 6px 14px; border-radius: 20px; font-size: 12px; font-weight: 600; border: 1px solid #3B82F6;">Plotly</span>
                <span style="background-color: #1E3A8A; color: #60A5FA; padding: 6px 14px; border-radius: 20px; font-size: 12px; font-weight: 600; border: 1px solid #3B82F6;">Streamlit</span>
                <span style="background-color: #1E3A8A; color: #60A5FA; padding: 6px 14px; border-radius: 20px; font-size: 12px; font-weight: 600; border: 1px solid #3B82F6;">Pandas & Numpy</span>
            </div>
            """, unsafe_allow_html=True)

        # 5. Horizontal divider between project info and developer bio
        st.markdown("<hr style='border: 0; border-top: 1px solid #334155; margin: 24px 0;'>", unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown("#### Developer Profile", unsafe_allow_html=True)
            st.markdown("""
            **Isha Maryam**  
            BS Computer Science — Gomal University  
            *Specialization: Machine Learning & Deep Learning*  
            
            [GitHub](https://github.com/) · [LinkedIn](https://linkedin.com/) · [Kaggle](https://kaggle.com/)
            """)

    with col_about_2:
        with st.container(border=True):
            st.markdown("#### Model Performance Table", unsafe_allow_html=True)
            
            perf = config.get('performance', {})
            recall_val = perf.get('recall', 76.6)
            precision_val = perf.get('precision', 33.03)
            f1_val = perf.get('f1_score', 46.15)
            auc_val = perf.get('auc_roc', 76.44)
            
            st.markdown(f"""
            <table style="width: 100%; border-collapse: collapse; text-align: left; background-color: #0F172A; color: white; border-radius: 8px; overflow: hidden; font-size: 13px;">
                <thead>
                    <tr style="background-color: #1E293B; border-bottom: 2px solid #334155;">
                        <th style="padding: 12px 14px; font-weight: 600;">Metric</th>
                        <th style="padding: 12px 14px; font-weight: 600; text-align: right;">Value</th>
                    </tr>
                </thead>
                <tbody>
                    <tr style="border-bottom: 1px solid #334155;">
                        <td style="padding: 12px 14px; font-weight: 500; color: #E2E8F0;">Recall</td>
                        <td style="padding: 12px 14px; text-align: right; font-weight: 700; color: #10B981;">{recall_val}%</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #334155;">
                        <td style="padding: 12px 14px; font-weight: 500; color: #E2E8F0;">Precision</td>
                        <td style="padding: 12px 14px; text-align: right; font-weight: 700; color: #3B82F6;">{precision_val}%</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #334155;">
                        <td style="padding: 12px 14px; font-weight: 500; color: #E2E8F0;">F1 Score</td>
                        <td style="padding: 12px 14px; text-align: right; font-weight: 700; color: #8B5CF6;">{f1_val}%</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #334155;">
                        <td style="padding: 12px 14px; font-weight: 500; color: #E2E8F0;">AUC-ROC</td>
                        <td style="padding: 12px 14px; text-align: right; font-weight: 700; color: #F59E0B;">{auc_val}%</td>
                    </tr>
                </tbody>
            </table>
            """, unsafe_allow_html=True)
            
        with st.container(border=True):
            st.markdown("#### Performance Evaluation Plots", unsafe_allow_html=True)
            with st.expander("Expand to view Confusion Matrix, ROC curve & Feature Importances"):
                plots_dir = "plots"
                plot_files = {
                    "Confusion Matrix": "confusion_matrix.png",
                    "ROC Curve": "roc_curve.png",
                    "Feature Importance": "feature_importance.png",
                    "Model Summary": "model_summary.png"
                }
                for title, fname in plot_files.items():
                    path = os.path.join("plots", fname)
                    st.markdown(f"<p style='font-size: 12px; margin-top: 8px; margin-bottom: 2px; font-weight: 600;'>{title}</p>", unsafe_allow_html=True)
                    if os.path.exists(path):
                        st.image(path, use_container_width=True)
                    else:
                        st.warning(f"Plot not found: {fname}")

# ============================================================
# FOOTER WITH GRADIENT DIVIDER
# ============================================================
st.markdown("<div style='height: 1px; background: linear-gradient(90deg, transparent, #3B82F6, transparent); margin-top: 40px; margin-bottom: 20px;'></div>", unsafe_allow_html=True)
st.caption("<div style='text-align: center; color: #64748B !important; font-size: 12px; margin-bottom: 20px;'>HR Attrition Intelligence System © 2026 | Built with Streamlit & XGBoost</div>", unsafe_allow_html=True)
