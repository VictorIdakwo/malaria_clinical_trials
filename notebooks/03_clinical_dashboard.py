# Databricks notebook source
# MAGIC %md
# MAGIC # Malaria Clinical Trial Dashboard
# MAGIC 
# MAGIC Interactive dashboard for:
# MAGIC - Entering patient symptoms and getting predictions
# MAGIC - Recording actual test results for model feedback
# MAGIC - Viewing prediction history
# MAGIC - Downloading trial data
# MAGIC - Monitoring model performance

# COMMAND ----------

# MAGIC %pip install streamlit plotly pandas numpy mlflow

# COMMAND ----------

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import mlflow
import mlflow.pyfunc
import pickle
from pyspark.sql.functions import col, desc, count
import uuid

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

CATALOG = "eha"
SCHEMA = "malaria_catalog"
VOLUME = "clinical_trial"

SYMPTOM_COLS = [
    "chill_cold", "headache", "fever", "generalized_body_pain",
    "abdominal_pain", "Loss_of_appetite", "joint_pain",
    "vomiting", "nausea", "diarrhea"
]

SYMPTOM_LABELS = {
    "chill_cold": "Chills/Cold",
    "headache": "Headache",
    "fever": "Fever",
    "generalized_body_pain": "Generalized Body Pain",
    "abdominal_pain": "Abdominal Pain",
    "Loss_of_appetite": "Loss of Appetite",
    "joint_pain": "Joint Pain",
    "vomiting": "Vomiting",
    "nausea": "Nausea",
    "diarrhea": "Diarrhea"
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Class Definition

# COMMAND ----------

class MalariaRLModel(mlflow.pyfunc.PythonModel):
    """RL model for malaria prediction"""
    
    def __init__(self, base_model=None, epsilon=0.1, learning_rate=0.01):
        self.base_model = base_model
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.total_predictions = 0
        self.correct_predictions = 0
        self.reward_history = []
        self.feature_names = [
            "chill_cold", "headache", "fever", "generalized_body_pain",
            "abdominal_pain", "Loss_of_appetite", "joint_pain",
            "vomiting", "nausea", "diarrhea"
        ]
        
    def predict(self, X):
        if hasattr(X, 'values'):
            X = X.values
        return self.base_model.predict(X)
    
    def predict_proba(self, X):
        if hasattr(X, 'values'):
            X = X.values
        return self.base_model.predict_proba(X)

# COMMAND ----------

# MAGIC ## Load Latest Model

# COMMAND ----------

@st.cache_resource
def load_model():
    """Load the latest RL model from Unity Catalog Volume"""
    try:
        # List all models in volume
        models = dbutils.fs.ls(f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/ml_models/")
        model_files = [m.path for m in models if m.name.startswith("malaria_rl_model_") and m.name.endswith(".pkl")]
        
        if not model_files:
            st.error("No model found in volume!")
            return None
        
        # Get latest model
        latest_model_path = sorted(model_files)[-1]
        
        # Remove dbfs: prefix if present (Unity Catalog volumes don't use it)
        latest_model_path = latest_model_path.replace("dbfs:", "")
        
        # Load model
        with open(latest_model_path, 'rb') as f:
            model = pickle.load(f)
        
        return model, latest_model_path
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

# COMMAND ----------

# MAGIC %md
# MAGIC ## Helper Functions

# COMMAND ----------

def save_prediction(patient_id, symptoms, prediction, probability, model_version):
    """Save prediction to database"""
    prediction_id = str(uuid.uuid4())
    
    prediction_data = {
        "prediction_id": prediction_id,
        "patient_id": patient_id,
        "prediction_timestamp": datetime.now(),
        **symptoms,
        "predicted_case": int(prediction),
        "prediction_probability": float(probability),
        "model_version": model_version,
        "actual_result": None,
        "feedback_timestamp": None,
        "model_correct": None
    }
    
    pred_df = spark.createDataFrame([prediction_data])
    pred_df.write.format("delta").mode("append").saveAsTable(
        f"{CATALOG}.{SCHEMA}.predictions"
    )
    
    return prediction_id

def update_with_actual_result(prediction_id, actual_result):
    """Update prediction with actual test result"""
    from delta.tables import DeltaTable
    
    predictions_table = DeltaTable.forName(spark, f"{CATALOG}.{SCHEMA}.predictions")
    
    predictions_table.update(
        condition=col("prediction_id") == prediction_id,
        set={
            "actual_result": actual_result,
            "feedback_timestamp": datetime.now(),
            "model_correct": col("predicted_case") == actual_result
        }
    )

def get_recent_predictions(limit=50):
    """Get recent predictions from database"""
    df = spark.table(f"{CATALOG}.{SCHEMA}.predictions") \
        .orderBy(desc("prediction_timestamp")) \
        .limit(limit)
    return df.toPandas()

def get_model_performance():
    """Get model performance metrics"""
    df = spark.table(f"{CATALOG}.{SCHEMA}.model_performance") \
        .orderBy(desc("metric_timestamp"))
    return df.toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Streamlit Dashboard

# COMMAND ----------

st.set_page_config(
    page_title="Malaria Clinical Trial Dashboard",
    page_icon="🦟",
    layout="wide"
)

st.title("🦟 Malaria Clinical Trial Dashboard")
st.markdown("---")

# Load model
model, model_path = load_latest_model()

if model is None:
    st.stop()

model_version = model_path.split("_")[-1].replace(".pkl", "")
st.sidebar.success(f"✅ Model loaded: v{model_version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Main Dashboard Tabs

# COMMAND ----------

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🩺 New Prediction",
    "📊 Prediction History",
    "✅ Enter Test Results",
    "📈 Model Performance",
    "💾 Export Data"
])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Tab 1: New Prediction

# COMMAND ----------

with tab1:
    st.header("Patient Symptom Assessment")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Patient Information")
        patient_id = st.text_input("Patient ID/Lab Number", placeholder="e.g., LT014233")
        
        st.subheader("Select Symptoms")
        st.caption("Check all symptoms the patient is experiencing:")
        
        symptoms = {}
        for col_name, label in SYMPTOM_LABELS.items():
            symptoms[col_name] = 1 if st.checkbox(label, key=f"symptom_{col_name}") else 0
    
    with col2:
        st.subheader("Prediction Results")
        
        if st.button("🔮 Generate Prediction", type="primary", use_container_width=True):
            if not patient_id:
                st.error("Please enter a Patient ID")
            else:
                # Prepare input
                input_df = pd.DataFrame([symptoms])
                
                # Make prediction
                result = model.predict(None, input_df)
                prediction = int(result['prediction'].values[0])
                prob_positive = float(result['probability_positive'].values[0])
                prob_negative = float(result['probability_negative'].values[0])
                
                # Display result
                if prediction == 1:
                    st.error(f"⚠️ POSITIVE for Malaria (Confidence: {prob_positive*100:.1f}%)")
                else:
                    st.success(f"✅ NEGATIVE for Malaria (Confidence: {prob_negative*100:.1f}%)")
                
                # Probability gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob_positive * 100,
                    title={'text': "Malaria Probability"},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkred" if prediction == 1 else "green"},
                        'steps': [
                            {'range': [0, 30], 'color': "lightgreen"},
                            {'range': [30, 70], 'color': "yellow"},
                            {'range': [70, 100], 'color': "lightcoral"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                st.plotly_chart(fig, use_container_width=True)
                
                # Save prediction
                pred_id = save_prediction(
                    patient_id=patient_id,
                    symptoms=symptoms,
                    prediction=prediction,
                    probability=prob_positive,
                    model_version=model_version
                )
                
                st.success(f"✅ Prediction saved! ID: {pred_id[:8]}...")
                st.info("⚠️ Please perform the actual laboratory test and enter results in the 'Enter Test Results' tab")
                
                # Show symptom summary
                st.subheader("Symptom Summary")
                active_symptoms = [SYMPTOM_LABELS[k] for k, v in symptoms.items() if v == 1]
                if active_symptoms:
                    st.write(", ".join(active_symptoms))
                else:
                    st.write("No symptoms reported")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Tab 2: Prediction History

# COMMAND ----------

with tab2:
    st.header("Recent Predictions")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        show_pending = st.checkbox("Show Pending Results", value=True)
    with col2:
        show_confirmed = st.checkbox("Show Confirmed Results", value=True)
    with col3:
        limit = st.number_input("Number of records", min_value=10, max_value=500, value=50)
    
    # Get predictions
    predictions_df = get_recent_predictions(limit=limit)
    
    if not predictions_df.empty:
        # Filter based on checkboxes
        if not show_pending:
            predictions_df = predictions_df[predictions_df['actual_result'].notna()]
        if not show_confirmed:
            predictions_df = predictions_df[predictions_df['actual_result'].isna()]
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Predictions", len(predictions_df))
        with col2:
            pending = predictions_df['actual_result'].isna().sum()
            st.metric("Pending Results", pending)
        with col3:
            if not predictions_df['model_correct'].isna().all():
                accuracy = predictions_df['model_correct'].mean() * 100
                st.metric("Model Accuracy", f"{accuracy:.1f}%")
        with col4:
            positive_pred = (predictions_df['predicted_case'] == 1).sum()
            st.metric("Positive Predictions", positive_pred)
        
        # Display table
        st.subheader("Prediction Records")
        
        display_df = predictions_df[[
            'patient_id', 'prediction_timestamp', 'predicted_case',
            'prediction_probability', 'actual_result', 'model_correct'
        ]].copy()
        
        display_df['predicted_case'] = display_df['predicted_case'].map({0: "Negative", 1: "Positive"})
        display_df['actual_result'] = display_df['actual_result'].map({0.0: "Negative", 1.0: "Positive"})
        display_df['model_correct'] = display_df['model_correct'].map({True: "✅", False: "❌", None: "⏳"})
        
        st.dataframe(display_df, use_container_width=True, height=400)
        
        # Accuracy over time chart
        if not predictions_df['model_correct'].isna().all():
            st.subheader("Model Accuracy Over Time")
            
            accuracy_df = predictions_df[predictions_df['actual_result'].notna()].copy()
            accuracy_df['correct'] = accuracy_df['model_correct'].astype(float)
            accuracy_df['cumulative_accuracy'] = accuracy_df['correct'].expanding().mean() * 100
            
            fig = px.line(
                accuracy_df,
                x='prediction_timestamp',
                y='cumulative_accuracy',
                title='Cumulative Model Accuracy',
                labels={'cumulative_accuracy': 'Accuracy (%)', 'prediction_timestamp': 'Time'}
            )
            fig.add_hline(y=50, line_dash="dash", line_color="gray", annotation_text="Random Baseline")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No predictions found. Start by making predictions in the 'New Prediction' tab.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Tab 3: Enter Test Results

# COMMAND ----------

with tab3:
    st.header("Enter Laboratory Test Results")
    st.caption("Enter the actual lab test results to help the model learn and improve")
    
    # Get pending predictions
    pending_df = get_recent_predictions(limit=100)
    pending_df = pending_df[pending_df['actual_result'].isna()]
    
    if not pending_df.empty:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Select prediction
            patient_options = pending_df.apply(
                lambda x: f"{x['patient_id']} - {x['prediction_timestamp'].strftime('%Y-%m-%d %H:%M')} - Predicted: {'Positive' if x['predicted_case']==1 else 'Negative'}",
                axis=1
            ).tolist()
            
            selected_idx = st.selectbox("Select Patient Record", range(len(patient_options)), format_func=lambda x: patient_options[x])
            
            selected_record = pending_df.iloc[selected_idx]
            
            # Show prediction details
            st.subheader("Prediction Details")
            pred_col1, pred_col2, pred_col3 = st.columns(3)
            
            with pred_col1:
                st.write(f"**Patient ID:** {selected_record['patient_id']}")
                st.write(f"**Prediction:** {'Positive' if selected_record['predicted_case']==1 else 'Negative'}")
            with pred_col2:
                st.write(f"**Confidence:** {selected_record['prediction_probability']*100:.1f}%")
                st.write(f"**Date:** {selected_record['prediction_timestamp'].strftime('%Y-%m-%d %H:%M')}")
            with pred_col3:
                st.write(f"**Model Version:** {selected_record['model_version']}")
        
        with col2:
            st.subheader("Enter Actual Result")
            
            actual_result = st.radio(
                "Laboratory Test Result:",
                options=[0, 1],
                format_func=lambda x: "Negative (No Malaria)" if x == 0 else "Positive (Malaria Detected)",
                key="actual_result"
            )
            
            if st.button("💾 Submit Result", type="primary", use_container_width=True):
                # Update database
                update_with_actual_result(selected_record['prediction_id'], actual_result)
                
                # Show feedback
                was_correct = (selected_record['predicted_case'] == actual_result)
                
                if was_correct:
                    st.success("✅ Model prediction was CORRECT! The model earned a positive reward.")
                else:
                    st.warning("❌ Model prediction was INCORRECT. The model will learn from this feedback.")
                
                st.balloons()
                st.info("Result saved successfully! The model will use this feedback for continuous improvement.")
                
                # Trigger rerun to refresh
                st.rerun()
        
        # Show symptoms
        st.subheader("Reported Symptoms")
        symptoms_list = [SYMPTOM_LABELS[col] for col in SYMPTOM_COLS if selected_record[col] == 1]
        if symptoms_list:
            st.write(", ".join(symptoms_list))
        else:
            st.write("No symptoms reported")
            
    else:
        st.info("📋 No pending test results. All predictions have been confirmed!")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Tab 4: Model Performance

# COMMAND ----------

with tab4:
    st.header("Model Performance Dashboard")
    
    # Get performance data
    perf_df = get_model_performance()
    predictions_df = get_recent_predictions(limit=1000)
    
    # Overall metrics
    st.subheader("Current Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    if not perf_df.empty:
        latest_perf = perf_df.iloc[0]
        
        with col1:
            st.metric("Accuracy", f"{latest_perf['accuracy']*100:.2f}%")
        with col2:
            st.metric("Precision", f"{latest_perf['precision_score']*100:.2f}%")
        with col3:
            st.metric("Recall", f"{latest_perf['recall']*100:.2f}%")
        with col4:
            st.metric("F1 Score", f"{latest_perf['f1_score']*100:.2f}%")
    
    # Real-time accuracy from predictions
    st.subheader("Real-Time Performance (Live Predictions)")
    
    confirmed_df = predictions_df[predictions_df['actual_result'].notna()]
    
    if not confirmed_df.empty:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            rt_accuracy = confirmed_df['model_correct'].mean() * 100
            st.metric("Live Accuracy", f"{rt_accuracy:.1f}%")
        
        with col2:
            total_feedback = len(confirmed_df)
            st.metric("Total Feedback", total_feedback)
        
        with col3:
            correct_count = confirmed_df['model_correct'].sum()
            st.metric("Correct Predictions", int(correct_count))
        
        with col4:
            avg_confidence = confirmed_df['prediction_probability'].mean() * 100
            st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
        
        # Confusion matrix
        st.subheader("Confusion Matrix")
        
        confusion_data = pd.crosstab(
            confirmed_df['predicted_case'].map({0: "Predicted Negative", 1: "Predicted Positive"}),
            confirmed_df['actual_result'].map({0: "Actual Negative", 1: "Actual Positive"})
        )
        
        fig = px.imshow(
            confusion_data,
            text_auto=True,
            color_continuous_scale='Blues',
            labels=dict(x="Actual", y="Predicted", color="Count")
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Prediction distribution
        st.subheader("Prediction Distribution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            pred_dist = confirmed_df['predicted_case'].value_counts()
            fig = px.pie(
                values=pred_dist.values,
                names=['Negative', 'Positive'],
                title='Predicted Cases Distribution'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            actual_dist = confirmed_df['actual_result'].value_counts()
            fig = px.pie(
                values=actual_dist.values,
                names=['Negative', 'Positive'],
                title='Actual Cases Distribution'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Feature analysis
        st.subheader("Symptom Correlation with Positive Cases")
        
        symptom_corr = []
        for symptom in SYMPTOM_COLS:
            positive_with_symptom = confirmed_df[(confirmed_df['actual_result'] == 1) & (confirmed_df[symptom] == 1)].shape[0]
            total_positive = confirmed_df[confirmed_df['actual_result'] == 1].shape[0]
            
            if total_positive > 0:
                correlation = (positive_with_symptom / total_positive) * 100
                symptom_corr.append({
                    'Symptom': SYMPTOM_LABELS[symptom],
                    'Correlation (%)': correlation
                })
        
        if symptom_corr:
            corr_df = pd.DataFrame(symptom_corr).sort_values('Correlation (%)', ascending=True)
            
            fig = px.bar(
                corr_df,
                x='Correlation (%)',
                y='Symptom',
                orientation='h',
                title='Percentage of Positive Cases with Each Symptom'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info("No confirmed predictions yet. Enter test results to see real-time performance.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Tab 5: Export Data

# COMMAND ----------

with tab5:
    st.header("Export Clinical Trial Data")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Export Options")
        
        export_type = st.radio(
            "Select data to export:",
            options=["All Predictions", "Confirmed Results Only", "Pending Results Only"],
            key="export_type"
        )
        
        date_range = st.date_input(
            "Date Range",
            value=(datetime.now().date(), datetime.now().date()),
            key="date_range"
        )
        
        include_symptoms = st.checkbox("Include Symptom Details", value=True)
        
    with col2:
        st.subheader("Download")
        
        if st.button("📥 Generate Export", type="primary", use_container_width=True):
            # Get data based on selection
            export_df = get_recent_predictions(limit=10000)
            
            # Filter by date range
            if len(date_range) == 2:
                start_date, end_date = date_range
                export_df = export_df[
                    (export_df['prediction_timestamp'].dt.date >= start_date) &
                    (export_df['prediction_timestamp'].dt.date <= end_date)
                ]
            
            # Filter by type
            if export_type == "Confirmed Results Only":
                export_df = export_df[export_df['actual_result'].notna()]
            elif export_type == "Pending Results Only":
                export_df = export_df[export_df['actual_result'].isna()]
            
            # Select columns
            base_cols = ['prediction_id', 'patient_id', 'prediction_timestamp',
                        'predicted_case', 'prediction_probability', 'model_version',
                        'actual_result', 'feedback_timestamp', 'model_correct']
            
            if include_symptoms:
                cols = base_cols + SYMPTOM_COLS
            else:
                cols = base_cols
            
            export_df = export_df[cols]
            
            # Convert to CSV
            csv = export_df.to_csv(index=False)
            
            st.download_button(
                label="💾 Download CSV",
                data=csv,
                file_name=f"malaria_clinical_trial_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            st.success(f"✅ Export ready! {len(export_df)} records selected.")
    
    # Show preview
    st.subheader("Data Preview")
    preview_df = get_recent_predictions(limit=10)
    st.dataframe(preview_df.head(10), use_container_width=True)
    
    # Export statistics
    st.subheader("Export Statistics")
    all_predictions = get_recent_predictions(limit=10000)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Records Available", len(all_predictions))
    with col2:
        confirmed = all_predictions['actual_result'].notna().sum()
        st.metric("Confirmed Results", confirmed)
    with col3:
        pending = all_predictions['actual_result'].isna().sum()
        st.metric("Pending Results", pending)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run Dashboard

# COMMAND ----------

# Note: In Databricks, run this using:
# streamlit run <path_to_this_notebook> --server.port 8501

st.sidebar.markdown("---")
st.sidebar.info("""
**How to Use:**
1. **New Prediction**: Enter patient symptoms to get malaria prediction
2. **Prediction History**: View all past predictions and their outcomes
3. **Enter Test Results**: Submit actual lab results for model learning
4. **Model Performance**: Monitor model accuracy and performance metrics
5. **Export Data**: Download clinical trial data for analysis

**Note:** The model continuously learns from your feedback!
""")
