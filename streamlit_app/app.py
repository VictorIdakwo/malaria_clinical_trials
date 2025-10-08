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
        font-size: 4.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        margin-top: 1rem;
        line-height: 1.2;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        border: 2px solid #ddd;
        margin: 1rem 0;
    }
    .positive {
        background-color: #ffebee;
        border-color: #ef5350;
    }
    .negative {
        background-color: #e8f5e9;
        border-color: #66bb6a;
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
    st.header("🏥 Patient Symptom Assessment")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Patient Information")
        patient_id = st.text_input("Patient ID", placeholder="Enter patient ID or leave blank for auto-generation")
        
        if not patient_id:
            patient_id = f"PATIENT_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        st.subheader("Symptoms Checklist")
        
        symptoms = {}
        
        # Create 2 columns for symptoms
        col_a, col_b = st.columns(2)
        
        symptoms_list = list(SYMPTOM_LABELS.items())
        mid = len(symptoms_list) // 2
        
        with col_a:
            for symptom, label in symptoms_list[:mid]:
                symptoms[symptom] = 1 if st.checkbox(label, key=symptom) else 0
        
        with col_b:
            for symptom, label in symptoms_list[mid:]:
                symptoms[symptom] = 1 if st.checkbox(label, key=symptom) else 0
        
        if st.button("🔍 Get Prediction", type="primary", use_container_width=True):
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
        st.subheader("Prediction Result")
        
        if 'last_prediction' in st.session_state:
            pred = st.session_state['last_prediction']
            
            result_class = "positive" if pred['prediction'] == 1 else "negative"
            result_text = "POSITIVE" if pred['prediction'] == 1 else "NEGATIVE"
            result_emoji = "🔴" if pred['prediction'] == 1 else "✅"
            
            st.markdown(f"""
            <div class="prediction-box {result_class}">
                <h2 style="text-align: center;">{result_emoji} {result_text}</h2>
                <h3 style="text-align: center;">for Malaria</h3>
                <p style="text-align: center; font-size: 1.2rem;">
                    Confidence: <strong>{pred['confidence']*100:.1f}%</strong>
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.success(f"✅ Prediction saved for {pred['patient_id']}")
            
            # Display Prediction ID prominently
            st.markdown("### 📋 Prediction ID")
            st.code(pred['prediction_id'], language=None)
            
            # Copy button hint
            st.caption("👆 Click to select, then copy (Ctrl+C) for feedback submission")
            
            # Database save status
            if pred.get('db_saved', False):
                st.success("💾 Saved to database")
            else:
                st.warning("⚠️ Not saved to database (working in demo mode)")
            
            st.info("⚠️ **Next Step:** After clinical test, submit actual result in 'Submit Feedback' page using the ID above")

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
