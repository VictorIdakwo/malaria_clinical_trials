# Databricks notebook source
# MAGIC %md
# MAGIC # Malaria Prediction API Service
# MAGIC 
# MAGIC REST API endpoints for programmatic access to the malaria prediction system.
# MAGIC This notebook creates a simple API using Flask for external integrations.

# COMMAND ----------

# MAGIC %pip install flask flask-cors pandas numpy mlflow

# COMMAND ----------

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
import uuid
from datetime import datetime
from pyspark.sql.functions import col
import mlflow

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

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Class Definition

# COMMAND ----------

import mlflow.pyfunc

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

# MAGIC %md
# MAGIC ## Load Model

# COMMAND ----------

def load_model():
    """Load the latest model from volume"""
    models = dbutils.fs.ls(f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/ml_models/")
    model_files = [m.path for m in models if m.name.startswith("malaria_rl_model_") and m.name.endswith(".pkl")]
    
    if not model_files:
        raise Exception("No model found in volume")
    
    latest_model_path = sorted(model_files)[-1]
    
    # Remove dbfs: prefix if present (Unity Catalog volumes don't use it)
    latest_model_path = latest_model_path.replace("dbfs:", "")
    
    with open(latest_model_path, 'rb') as f:
        model = pickle.load(f)
    
    model_version = latest_model_path.split("_")[-1].replace(".pkl", "")
    return model, model_version

# Load model globally
try:
    model, model_version = load_model()
    print(f"✅ Model loaded: version {model_version}")
except Exception as e:
    print(f"❌ Error loading model: {str(e)}")
    model, model_version = None, None

# COMMAND ----------

# MAGIC %md
# MAGIC ## API Helper Functions

# COMMAND ----------

def validate_symptoms(data):
    """Validate symptom input"""
    errors = []
    
    for symptom in SYMPTOM_COLS:
        if symptom not in data:
            errors.append(f"Missing symptom: {symptom}")
        elif data[symptom] not in [0, 1]:
            errors.append(f"Invalid value for {symptom}: must be 0 or 1")
    
    return errors

def save_prediction_to_db(patient_id, symptoms, prediction, probability):
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

def update_feedback_in_db(prediction_id, actual_result):
    """Update prediction with actual result"""
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

# COMMAND ----------

# MAGIC %md
# MAGIC ## Flask API

# COMMAND ----------

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_version": model_version,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/v1/predict', methods=['POST'])
def predict():
    """
    Predict malaria from symptoms
    
    Request body:
    {
        "patient_id": "LT123456",
        "symptoms": {
            "chill_cold": 0,
            "headache": 1,
            "fever": 1,
            "generalized_body_pain": 0,
            "abdominal_pain": 0,
            "Loss_of_appetite": 0,
            "joint_pain": 0,
            "vomiting": 0,
            "nausea": 0,
            "diarrhea": 0
        }
    }
    
    Response:
    {
        "prediction_id": "uuid",
        "patient_id": "LT123456",
        "prediction": 1,
        "probability_positive": 0.85,
        "probability_negative": 0.15,
        "result": "Positive",
        "model_version": "20231007_123456",
        "timestamp": "2023-10-07T12:34:56"
    }
    """
    try:
        # Validate request
        if not request.json:
            return jsonify({"error": "No JSON data provided"}), 400
        
        data = request.json
        
        if "patient_id" not in data:
            return jsonify({"error": "patient_id is required"}), 400
        
        if "symptoms" not in data:
            return jsonify({"error": "symptoms are required"}), 400
        
        patient_id = data["patient_id"]
        symptoms = data["symptoms"]
        
        # Validate symptoms
        errors = validate_symptoms(symptoms)
        if errors:
            return jsonify({"error": "Validation failed", "details": errors}), 400
        
        # Check if model is loaded
        if model is None:
            return jsonify({"error": "Model not available"}), 503
        
        # Make prediction
        input_df = pd.DataFrame([symptoms])
        result = model.predict(None, input_df)
        
        prediction = int(result['prediction'].values[0])
        prob_positive = float(result['probability_positive'].values[0])
        prob_negative = float(result['probability_negative'].values[0])
        
        # Save to database
        prediction_id = save_prediction_to_db(
            patient_id=patient_id,
            symptoms=symptoms,
            prediction=prediction,
            probability=prob_positive
        )
        
        # Return response
        response = {
            "prediction_id": prediction_id,
            "patient_id": patient_id,
            "prediction": prediction,
            "probability_positive": prob_positive,
            "probability_negative": prob_negative,
            "result": "Positive" if prediction == 1 else "Negative",
            "model_version": model_version,
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/v1/feedback', methods=['POST'])
def feedback():
    """
    Submit actual test result for a prediction
    
    Request body:
    {
        "prediction_id": "uuid",
        "actual_result": 1
    }
    
    Response:
    {
        "success": true,
        "message": "Feedback recorded",
        "model_correct": true
    }
    """
    try:
        # Validate request
        if not request.json:
            return jsonify({"error": "No JSON data provided"}), 400
        
        data = request.json
        
        if "prediction_id" not in data:
            return jsonify({"error": "prediction_id is required"}), 400
        
        if "actual_result" not in data:
            return jsonify({"error": "actual_result is required"}), 400
        
        prediction_id = data["prediction_id"]
        actual_result = data["actual_result"]
        
        if actual_result not in [0, 1]:
            return jsonify({"error": "actual_result must be 0 or 1"}), 400
        
        # Get original prediction
        pred_df = spark.table(f"{CATALOG}.{SCHEMA}.predictions") \
            .filter(col("prediction_id") == prediction_id) \
            .collect()
        
        if not pred_df:
            return jsonify({"error": "Prediction ID not found"}), 404
        
        original_prediction = pred_df[0]["predicted_case"]
        
        # Update database
        update_feedback_in_db(prediction_id, actual_result)
        
        # Check if model was correct
        model_correct = (original_prediction == actual_result)
        
        response = {
            "success": True,
            "message": "Feedback recorded successfully",
            "prediction_id": prediction_id,
            "model_correct": model_correct,
            "reward": 1.0 if model_correct else -1.0
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/v1/history/<patient_id>', methods=['GET'])
def get_patient_history(patient_id):
    """
    Get prediction history for a patient
    
    Response:
    {
        "patient_id": "LT123456",
        "predictions": [
            {
                "prediction_id": "uuid",
                "timestamp": "2023-10-07T12:34:56",
                "prediction": 1,
                "actual_result": 1,
                "model_correct": true
            }
        ]
    }
    """
    try:
        # Get predictions for patient
        history_df = spark.table(f"{CATALOG}.{SCHEMA}.predictions") \
            .filter(col("patient_id") == patient_id) \
            .orderBy(col("prediction_timestamp").desc()) \
            .toPandas()
        
        if history_df.empty:
            return jsonify({
                "patient_id": patient_id,
                "predictions": [],
                "count": 0
            }), 200
        
        # Convert to JSON
        predictions = []
        for _, row in history_df.iterrows():
            predictions.append({
                "prediction_id": row["prediction_id"],
                "timestamp": row["prediction_timestamp"].isoformat(),
                "prediction": int(row["predicted_case"]),
                "probability": float(row["prediction_probability"]),
                "actual_result": int(row["actual_result"]) if pd.notna(row["actual_result"]) else None,
                "model_correct": bool(row["model_correct"]) if pd.notna(row["model_correct"]) else None
            })
        
        response = {
            "patient_id": patient_id,
            "predictions": predictions,
            "count": len(predictions)
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/v1/stats', methods=['GET'])
def get_statistics():
    """
    Get overall system statistics
    
    Response:
    {
        "total_predictions": 1000,
        "pending_feedback": 50,
        "accuracy": 0.85,
        "positive_rate": 0.35
    }
    """
    try:
        # Get all predictions
        all_preds = spark.table(f"{CATALOG}.{SCHEMA}.predictions")
        
        total = all_preds.count()
        pending = all_preds.filter(col("actual_result").isNull()).count()
        confirmed = all_preds.filter(col("actual_result").isNotNull())
        
        accuracy = None
        positive_rate = None
        
        if confirmed.count() > 0:
            confirmed_pdf = confirmed.toPandas()
            accuracy = float(confirmed_pdf["model_correct"].mean())
            positive_rate = float((confirmed_pdf["predicted_case"] == 1).mean())
        
        response = {
            "total_predictions": int(total),
            "pending_feedback": int(pending),
            "confirmed_predictions": int(confirmed.count()),
            "accuracy": accuracy,
            "positive_rate": positive_rate,
            "model_version": model_version,
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run API Server

# COMMAND ----------

if __name__ == "__main__":
    print("=" * 60)
    print("🦟 Malaria Prediction API Server")
    print("=" * 60)
    print(f"\nModel Version: {model_version}")
    print("\nAvailable Endpoints:")
    print("  GET  /health                     - Health check")
    print("  POST /api/v1/predict             - Make prediction")
    print("  POST /api/v1/feedback            - Submit feedback")
    print("  GET  /api/v1/history/<patient_id> - Get patient history")
    print("  GET  /api/v1/stats               - Get statistics")
    print("\nStarting server on port 5000...")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=5000, debug=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## API Usage Examples

# COMMAND ----------

# MAGIC %md
# MAGIC ### Example 1: Make a Prediction
# MAGIC 
# MAGIC ```bash
# MAGIC curl -X POST http://localhost:5000/api/v1/predict \
# MAGIC   -H "Content-Type: application/json" \
# MAGIC   -d '{
# MAGIC     "patient_id": "LT123456",
# MAGIC     "symptoms": {
# MAGIC       "chill_cold": 0,
# MAGIC       "headache": 1,
# MAGIC       "fever": 1,
# MAGIC       "generalized_body_pain": 1,
# MAGIC       "abdominal_pain": 0,
# MAGIC       "Loss_of_appetite": 0,
# MAGIC       "joint_pain": 0,
# MAGIC       "vomiting": 0,
# MAGIC       "nausea": 0,
# MAGIC       "diarrhea": 0
# MAGIC     }
# MAGIC   }'
# MAGIC ```
# MAGIC 
# MAGIC ### Example 2: Submit Feedback
# MAGIC 
# MAGIC ```bash
# MAGIC curl -X POST http://localhost:5000/api/v1/feedback \
# MAGIC   -H "Content-Type: application/json" \
# MAGIC   -d '{
# MAGIC     "prediction_id": "abc-123-def-456",
# MAGIC     "actual_result": 1
# MAGIC   }'
# MAGIC ```
# MAGIC 
# MAGIC ### Example 3: Get Patient History
# MAGIC 
# MAGIC ```bash
# MAGIC curl http://localhost:5000/api/v1/history/LT123456
# MAGIC ```
# MAGIC 
# MAGIC ### Example 4: Get Statistics
# MAGIC 
# MAGIC ```bash
# MAGIC curl http://localhost:5000/api/v1/stats
# MAGIC ```
