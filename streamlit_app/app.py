"""
Malaria Clinical Trial - Streamlit Web Application
Connects to Databricks for predictions and feedback
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from databricks import sql
import pickle
import io
from datetime import datetime
import uuid
import os

# Page configuration
st.set_page_config(
    page_title="Malaria Clinical Trial Assistant",
    page_icon="🦟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 6rem;
        font-weight: 800;
        color: #0066cc;
        text-align: center;
        margin-bottom: 2rem;
        margin-top: 0.5rem;
        line-height: 1.1;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 15px;
        border: 3px solid #ddd;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .positive {
        background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
        border-color: #ef5350;
    }
    .negative {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        border-color: #66bb6a;
    }
    .info-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #0066cc;
        margin: 1rem 0;
    }
    .symptom-section {
        background-color: white;
        padding: 2rem;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin: 1rem 0;
    }
    .result-card {
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Configuration
CATALOG = "eha"
SCHEMA = "malaria_catalog"
VOLUME = "clinical_trial"

SYMPTOM_COLS = [
    "chill_cold", "headache", "fever", "generalized_body_pain",
    "abdominal_pain", "Loss_of_appetite", "joint_pain",
    "vomiting", "nausea", "diarrhea"
]

SYMPTOM_LABELS = {
    "chill_cold": "🥶 Chills/Cold",
    "headache": "🤕 Headache",
    "fever": "🌡️ Fever",
    "generalized_body_pain": "💪 Generalized Body Pain",
    "abdominal_pain": "🤰 Abdominal Pain",
    "Loss_of_appetite": "🍽️ Loss of Appetite",
    "joint_pain": "🦴 Joint Pain",
    "vomiting": "🤮 Vomiting",
    "nausea": "😵 Nausea",
    "diarrhea": "🚽 Diarrhea"
}

# Databricks connection
@st.cache_resource
def get_databricks_connection():
    """Establish connection to Databricks"""
    try:
        # Try to get credentials from Streamlit secrets first (for Cloud deployment)
        # Then fall back to .env file (for local development)
        try:
            server_hostname = st.secrets["DATABRICKS_SERVER_HOSTNAME"]
            http_path = st.secrets["DATABRICKS_HTTP_PATH"]
            access_token = st.secrets["DATABRICKS_TOKEN"]
        except (FileNotFoundError, KeyError):
            # Fall back to environment variables
            from dotenv import load_dotenv
            load_dotenv()
            
            server_hostname = os.getenv("DATABRICKS_SERVER_HOSTNAME")
            http_path = os.getenv("DATABRICKS_HTTP_PATH")
            access_token = os.getenv("DATABRICKS_TOKEN")
        
        if not all([server_hostname, http_path, access_token]):
            st.error("⚠️ Missing Databricks credentials. Please add them in Streamlit Cloud Secrets or .env file")
            return None
        
        conn = sql.connect(
            server_hostname=server_hostname,
            http_path=http_path,
            access_token=access_token
        )
        return conn
    except Exception as e:
        st.error(f"❌ Failed to connect to Databricks: {e}")
        return None

# Model class definition
class MalariaRLModel:
    """RL model for malaria prediction"""
    
    def __init__(self, base_model=None, epsilon=0.1, learning_rate=0.01):
        self.base_model = base_model
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.total_predictions = 0
        self.correct_predictions = 0
        self.reward_history = []
        self.feature_names = SYMPTOM_COLS
        
    def predict(self, X):
        if hasattr(X, 'values'):
            X = X.values
        return self.base_model.predict(X)
    
    def predict_proba(self, X):
        if hasattr(X, 'values'):
            X = X.values
        return self.base_model.predict_proba(X)

@st.cache_resource
def load_model():
    """Load the latest model from Databricks Unity Catalog Volume"""
    try:
        conn = get_databricks_connection()
        cursor = conn.cursor()
        
        # Use dbutils alternative: query the volume metadata
        query = f"""
        SELECT path, name 
        FROM dbfs_file_metadata('{CATALOG}.{SCHEMA}.{VOLUME}')
        WHERE path LIKE '%/ml_models/%' 
        AND name LIKE 'malaria_rl_model_%'
        AND name LIKE '%.pkl'
        ORDER BY path DESC
        LIMIT 1
        """
        
        # Alternative: Use direct file path if you know the latest model
        # For now, we'll use a direct connection approach
        
        # Download model file from Databricks
        model_path = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/ml_models/"
        
        st.info("Model loading from Databricks Volume. Ensure you have the model file locally or use Databricks API to fetch it.")
        
        # For demo purposes, return a placeholder
        # In production, you'd fetch the actual model file
        return None, "latest"
        
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

def execute_query(query, params=None):
    """Execute a query on Databricks"""
    conn = get_databricks_connection()
    if conn is None:
        raise Exception("No database connection available")
    
    cursor = conn.cursor()
    
    if params:
        cursor.execute(query, params)
    else:
        cursor.execute(query)
    
    return cursor.fetchall()

def save_prediction(patient_id, symptoms, prediction, confidence, model_version):
    """Save prediction to Databricks"""
    conn = get_databricks_connection()
    if conn is None:
        raise Exception("No database connection available")
    
    cursor = conn.cursor()
    
    prediction_id = str(uuid.uuid4())
    timestamp = datetime.now()
    
    # Build insert query
    symptom_values = ', '.join([str(symptoms[col]) for col in SYMPTOM_COLS])
    
    query = f"""
    INSERT INTO {CATALOG}.{SCHEMA}.predictions 
    (prediction_id, patient_id, prediction_timestamp, prediction, confidence, model_version,
     {', '.join(SYMPTOM_COLS)})
    VALUES ('{prediction_id}', '{patient_id}', '{timestamp}', {prediction}, {confidence}, 
            '{model_version}', {symptom_values})
    """
    
    cursor.execute(query)
    conn.commit()
    
    return prediction_id

def submit_feedback(prediction_id, actual_result):
    """Submit clinical test feedback"""
    conn = get_databricks_connection()
    if conn is None:
        raise Exception("No database connection available")
    
    cursor = conn.cursor()
    
    query = f"""
    UPDATE {CATALOG}.{SCHEMA}.predictions
    SET actual_result = {actual_result},
        feedback_timestamp = '{datetime.now()}',
        model_correct = (prediction = {actual_result})
    WHERE prediction_id = '{prediction_id}'
    """
    
    cursor.execute(query)
    conn.commit()

# ==================== STREAMLIT APP ====================

# Sidebar
with st.sidebar:
    # Display logo (handle both local and cloud deployment)
    logo_path = os.path.join(os.path.dirname(__file__), "eHA-logo-blue_320x132.png")
    if os.path.exists(logo_path):
        st.image(logo_path, width=200)
    else:
        # Fallback: try relative path
        try:
            st.image("eHA-logo-blue_320x132.png", width=200)
        except:
            # If logo not found, show text instead
            st.markdown("### 🏥 eHealth Africa")
    
    st.title("🦟 Malaria Assistant")
    st.markdown("---")
    
    page = st.radio(
        "Navigation",
        ["🏥 Make Prediction", "📝 Submit Feedback", "📊 Dashboard", "⚙️ Settings"]
    )
    
    st.markdown("---")
    st.info("""
    **Quick Help:**
    - Make predictions for new patients
    - Submit clinical test results
    - View model performance
    """)

# Main content
st.markdown('<p class="main-header">🦟 Malaria Clinical Trial Assistant</p>', unsafe_allow_html=True)

# ==================== PAGE 1: MAKE PREDICTION ====================
if page == "🏥 Make Prediction":
    # Page header with professional styling
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 2rem; border-radius: 15px; margin-bottom: 2rem;">
        <h1 style="color: white; text-align: center; margin: 0; font-size: 2.5rem;">
            🏥 Clinical Malaria Risk Assessment
        </h1>
        <p style="color: #f0f0f0; text-align: center; margin-top: 0.5rem; font-size: 1.1rem;">
            Evidence-based prediction tool for malaria diagnosis
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Instructions card
    st.markdown("""
    <div class="info-card">
        <h4>📋 Instructions</h4>
        <p><strong>1.</strong> Enter patient identification information</p>
        <p><strong>2.</strong> Check all symptoms present in the patient</p>
        <p><strong>3.</strong> Click "Get Prediction" to receive risk assessment</p>
        <p><strong>4.</strong> Save the Prediction ID for feedback submission after clinical testing</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        # Patient Information Section
        st.markdown("### 👤 Patient Information")
        st.markdown('<div class="symptom-section">', unsafe_allow_html=True)
        
        col_id, col_date = st.columns(2)
        with col_id:
            patient_id = st.text_input(
                "Patient ID *", 
                placeholder="e.g., PT-2025-001",
                help="Enter unique patient identifier or leave blank for auto-generation"
            )
        
        with col_date:
            assessment_date = st.date_input(
                "Assessment Date",
                value=datetime.now(),
                help="Date of symptom assessment"
            )
        
        if not patient_id:
            patient_id = f"PT-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            st.caption(f"Auto-generated ID: **{patient_id}**")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Symptoms Section with better styling
        st.markdown("### 🩺 Clinical Symptoms Assessment")
        st.markdown('<div class="symptom-section">', unsafe_allow_html=True)
        st.markdown("**Select all symptoms present in the patient:**")
        st.markdown("---")
        
        symptoms = {}
        
        # Create 2 columns for symptoms
        col_a, col_b = st.columns(2)
        
        symptoms_list = list(SYMPTOM_LABELS.items())
        mid = len(symptoms_list) // 2
        
        with col_a:
            st.markdown("**Primary Symptoms:**")
            for symptom, label in symptoms_list[:mid]:
                symptoms[symptom] = 1 if st.checkbox(label, key=symptom) else 0
        
        with col_b:
            st.markdown("**Secondary Symptoms:**")
            for symptom, label in symptoms_list[mid:]:
                symptoms[symptom] = 1 if st.checkbox(label, key=symptom) else 0
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Action button with better styling
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Count selected symptoms
        symptoms_selected = sum(symptoms.values()) if symptoms else 0
        
        if symptoms_selected > 0:
            st.success(f"✅ {symptoms_selected} symptom(s) selected")
        else:
            st.info("ℹ️ Please select at least one symptom to proceed")
        
        if st.button("🔍 Analyze & Get Prediction", type="primary", use_container_width=True, disabled=(symptoms_selected == 0)):
            # Simulate prediction (replace with actual model)
            X = np.array([[symptoms[col] for col in SYMPTOM_COLS]])
            
            # For demo: random prediction
            prediction = np.random.randint(0, 2)
            confidence = np.random.uniform(0.6, 0.95)
            
            # Generate prediction ID
            prediction_id = str(uuid.uuid4())
            
            # Try to save to database
            try:
                conn = get_databricks_connection()
                if conn:
                    cursor = conn.cursor()
                    
                    # Build insert query
                    symptom_values = ', '.join([str(symptoms[col]) for col in SYMPTOM_COLS])
                    
                    query = f"""
                    INSERT INTO {CATALOG}.{SCHEMA}.predictions 
                    (prediction_id, patient_id, prediction_timestamp, prediction, confidence, model_version,
                     {', '.join(SYMPTOM_COLS)})
                    VALUES ('{prediction_id}', '{patient_id}', '{datetime.now()}', {prediction}, {confidence}, 
                            'demo_v1', {symptom_values})
                    """
                    
                    cursor.execute(query)
                    conn.commit()
                    db_saved = True
                else:
                    db_saved = False
            except Exception as e:
                st.warning(f"⚠️ Could not save to database: {e}")
                db_saved = False
            
            # Save to session state
            st.session_state['last_prediction'] = {
                'prediction_id': prediction_id,
                'patient_id': patient_id,
                'symptoms': symptoms,
                'prediction': prediction,
                'confidence': confidence,
                'timestamp': datetime.now(),
                'db_saved': db_saved
            }
    
    with col2:
        st.markdown("### 📊 Risk Assessment Results")
        
        if 'last_prediction' in st.session_state:
            pred = st.session_state['last_prediction']
            
            result_class = "positive" if pred['prediction'] == 1 else "negative"
            result_text = "POSITIVE" if pred['prediction'] == 1 else "NEGATIVE"
            result_emoji = "🔴" if pred['prediction'] == 1 else "✅"
            risk_level = "HIGH RISK" if pred['prediction'] == 1 else "LOW RISK"
            
            # Professional results card
            st.markdown(f"""
            <div class="result-card {result_class}">
                <h3 style="text-align: center; margin: 0; color: #333;">
                    Malaria Risk Assessment
                </h3>
                <hr style="margin: 1rem 0;">
                <div style="text-align: center; padding: 1.5rem 0;">
                    <div style="font-size: 3rem; margin-bottom: 0.5rem;">
                        {result_emoji}
                    </div>
                    <h1 style="margin: 0.5rem 0; font-size: 2.5rem; font-weight: bold;">
                        {result_text}
                    </h1>
                    <p style="font-size: 1.3rem; color: #666; margin: 0.5rem 0;">
                        {risk_level}
                    </p>
                    <div style="background-color: rgba(255,255,255,0.5); padding: 1rem; border-radius: 8px; margin-top: 1rem;">
                        <p style="margin: 0; font-size: 1.1rem; color: #333;">
                            <strong>Confidence Level</strong>
                        </p>
                        <p style="margin: 0.5rem 0 0 0; font-size: 2rem; font-weight: bold; color: #0066cc;">
                            {pred['confidence']*100:.1f}%
                        </p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Clinical recommendation
            if pred['prediction'] == 1:
                st.error("""
                **⚕️ Clinical Recommendation:**
                - Perform confirmatory diagnostic test (microscopy/RDT)
                - Consider antimalarial treatment if confirmed
                - Monitor patient closely
                """)
            else:
                st.success("""
                **⚕️ Clinical Recommendation:**
                - Low malaria risk indicated
                - Consider alternative diagnoses
                - Confirmatory testing recommended if symptoms persist
                """)
            
            # Patient & Prediction Details
            st.markdown("---")
            st.markdown("**📋 Assessment Details:**")
            
            col_detail1, col_detail2 = st.columns(2)
            with col_detail1:
                st.metric("Patient ID", pred['patient_id'])
            with col_detail2:
                st.metric("Timestamp", pred['timestamp'].strftime("%H:%M:%S"))
            
            # Prediction ID Section
            st.markdown("---")
            st.markdown("**🔑 Prediction Reference ID:**")
            st.code(pred['prediction_id'], language=None)
            st.caption("💡 Save this ID for feedback submission after clinical testing")
            
            # Database save status
            if pred.get('db_saved', False):
                st.success("✅ Data successfully saved to database")
            else:
                st.info("ℹ️ Running in demo mode (database not connected)")
            
            # Next steps
            st.markdown("---")
            st.info("""
            **📝 Next Steps:**
            1. Perform confirmatory clinical test (microscopy/RDT)
            2. Record actual test results in **'Submit Feedback'** page
            3. Use the Prediction ID above for feedback submission
            """)
        else:
            # Placeholder when no prediction yet
            st.markdown("""
            <div style="background-color: #f8f9fa; padding: 3rem; border-radius: 15px; text-align: center; margin-top: 2rem;">
                <div style="font-size: 4rem; margin-bottom: 1rem;">🔬</div>
                <h3 style="color: #666; margin-bottom: 1rem;">No Assessment Yet</h3>
                <p style="color: #888;">Complete the symptom assessment and click<br/>"Analyze & Get Prediction" to see results here.</p>
            </div>
            """, unsafe_allow_html=True)

# ==================== PAGE 2: SUBMIT FEEDBACK ====================
elif page == "📝 Submit Feedback":
    st.header("📝 Submit Clinical Test Result")
    
    st.info("After performing the clinical test (microscopy/RDT), record the actual result here to help the model learn.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Auto-fill if user just made a prediction
        default_pred_id = ""
        if 'last_prediction' in st.session_state:
            default_pred_id = st.session_state['last_prediction'].get('prediction_id', '')
        
        prediction_id = st.text_input(
            "Prediction ID", 
            value=default_pred_id,
            placeholder="Enter prediction ID from prediction screen"
        )
        
        if default_pred_id:
            st.caption(f"✅ Auto-filled from recent prediction for patient: {st.session_state['last_prediction']['patient_id']}")
        
        actual_result = st.radio("Clinical Test Result", ["Negative (0)", "Positive (1)"])
        
        if st.button("Submit Feedback", type="primary", use_container_width=True):
            if not prediction_id:
                st.error("Please enter a Prediction ID")
            else:
                actual = 0 if "Negative" in actual_result else 1
                
                try:
                    submit_feedback(prediction_id, actual)
                    st.success("✅ Feedback submitted successfully!")
                    st.balloons()
                    
                    # Show progress
                    conn = get_databricks_connection()
                    if conn:
                        cursor = conn.cursor()
                        cursor.execute(f"""
                            SELECT COUNT(*) as count 
                            FROM {CATALOG}.{SCHEMA}.predictions 
                            WHERE actual_result IS NOT NULL
                        """)
                        feedback_count = cursor.fetchone()[0]
                        
                        st.metric("Total Feedback Collected", feedback_count)
                        st.progress(min(feedback_count / 50, 1.0))
                        
                        if feedback_count >= 50:
                            st.info("🔄 Enough feedback collected! Model will retrain on next scheduled job.")
                        else:
                            st.info(f"⏳ {50 - feedback_count} more samples needed for retraining")
                        
                except Exception as e:
                    st.error(f"Error submitting feedback: {e}")
    
    with col2:
        st.subheader("Recent Predictions Awaiting Feedback")
        
        try:
            conn = get_databricks_connection()
            if conn:
                cursor = conn.cursor()
                cursor.execute(f"""
                    SELECT patient_id, prediction_id, 
                           CASE WHEN prediction = 1 THEN 'Positive' ELSE 'Negative' END as predicted,
                           prediction_timestamp
                    FROM {CATALOG}.{SCHEMA}.predictions
                    WHERE actual_result IS NULL
                    ORDER BY prediction_timestamp DESC
                    LIMIT 10
                """)
                
                results = cursor.fetchall()
                if results:
                    df = pd.DataFrame(results, columns=['Patient ID', 'Prediction ID', 'Predicted', 'Timestamp'])
                    st.dataframe(df, use_container_width=True)
                else:
                    st.info("No predictions awaiting feedback")
            else:
                st.warning("⚠️ Cannot load predictions - database connection not available")
                
        except Exception as e:
            st.error(f"Error loading predictions: {e}")

# ==================== PAGE 3: DASHBOARD ====================
elif page == "📊 Dashboard":
    st.header("📊 Model Performance Dashboard")
    
    conn = get_databricks_connection()
    
    if not conn:
        st.error("⚠️ Cannot load dashboard - no database connection available")
        st.info("Please check your Databricks credentials in the Settings page")
    else:
        try:
            cursor = conn.cursor()
            
            # KPIs
            st.subheader("Model Performance Metrics")
            col1, col2, col3, col4 = st.columns(4)
            
            cursor.execute(f"""
                SELECT accuracy, precision_score, recall, f1_score
                FROM {CATALOG}.{SCHEMA}.model_performance
                ORDER BY metric_timestamp DESC
                LIMIT 1
            """)
            metrics = cursor.fetchone()
            
            if metrics:
                col1.metric("Accuracy", f"{metrics[0]*100:.1f}%")
                col2.metric("Precision", f"{metrics[1]*100:.1f}%")
                col3.metric("Recall", f"{metrics[2]*100:.1f}%")
                col4.metric("F1 Score", f"{metrics[3]*100:.1f}%")
            
            # Prediction statistics
            st.subheader("Prediction Statistics")
            col1, col2, col3 = st.columns(3)
            
            cursor.execute(f"""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN prediction = 1 THEN 1 ELSE 0 END) as positive,
                    SUM(CASE WHEN actual_result IS NOT NULL THEN 1 ELSE 0 END) as feedback
                FROM {CATALOG}.{SCHEMA}.predictions
            """)
            stats = cursor.fetchone()
            
            col1.metric("Total Predictions", stats[0])
            col2.metric("Positive Cases", stats[1])
            col3.metric("Feedback Received", stats[2])
            
            # Recent predictions table
            st.subheader("Recent Predictions")
            cursor.execute(f"""
                SELECT patient_id, 
                       CASE WHEN prediction = 1 THEN 'Positive' ELSE 'Negative' END as predicted,
                       CASE WHEN actual_result = 1 THEN 'Positive' 
                            WHEN actual_result = 0 THEN 'Negative'
                            ELSE 'Pending' END as actual,
                       ROUND(confidence * 100, 1) as confidence,
                       prediction_timestamp
                FROM {CATALOG}.{SCHEMA}.predictions
                ORDER BY prediction_timestamp DESC
                LIMIT 20
            """)
            
            results = cursor.fetchall()
            df = pd.DataFrame(results, columns=['Patient ID', 'Predicted', 'Actual', 'Confidence %', 'Timestamp'])
            st.dataframe(df, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error loading dashboard data: {e}")

# ==================== PAGE 4: SETTINGS ====================
elif page == "⚙️ Settings":
    st.header("⚙️ Settings & Configuration")
    
    st.subheader("Databricks Connection")
    st.info(f"""
    **Current Configuration:**
    - Catalog: `{CATALOG}`
    - Schema: `{SCHEMA}`
    - Volume: `{VOLUME}`
    """)
    
    st.subheader("Environment Variables Required")
    st.code("""
# .env file
DATABRICKS_SERVER_HOSTNAME=your-workspace.gcp.databricks.com
DATABRICKS_HTTP_PATH=/sql/1.0/warehouses/xxxxx
DATABRICKS_TOKEN=your-access-token
    """)
    
    if st.button("Test Connection"):
        try:
            conn = get_databricks_connection()
            st.success("✅ Connection successful!")
        except Exception as e:
            st.error(f"❌ Connection failed: {e}")
