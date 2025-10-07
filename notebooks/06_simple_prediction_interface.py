# Databricks notebook source
# MAGIC %md
# MAGIC # Simple Malaria Prediction Interface
# MAGIC 
# MAGIC Use this notebook to:
# MAGIC - Make predictions for new patients
# MAGIC - View recent predictions
# MAGIC - Check model performance

# COMMAND ----------

import pickle
import pandas as pd
import numpy as np
from pyspark.sql.functions import col, desc
from datetime import datetime
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

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Class Definition

# COMMAND ----------

import mlflow.pyfunc

class MalariaRLModel(mlflow.pyfunc.PythonModel):
    """
    Reinforcement Learning model for malaria prediction using contextual bandits.
    """
    
    def __init__(self, base_model=None, epsilon=0.1, learning_rate=0.01):
        self.base_model = base_model
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.total_predictions = 0
        self.correct_predictions = 0
        self.reward_history = []
        self.feature_names = SYMPTOM_COLS
        
    def predict(self, X):
        """Make prediction using base model"""
        if hasattr(X, 'values'):
            X = X.values
        return self.base_model.predict(X)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        if hasattr(X, 'values'):
            X = X.values
        return self.base_model.predict_proba(X)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Latest Model

# COMMAND ----------

def load_latest_model():
    """Load the most recent model from volume"""
    models = dbutils.fs.ls(f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/ml_models/")
    model_files = [m.path for m in models if m.name.startswith("malaria_rl_model_") and m.name.endswith(".pkl")]
    
    if not model_files:
        raise Exception("No model found!")
    
    latest_model_path = sorted(model_files)[-1]
    
    # Remove dbfs: prefix if present (Unity Catalog volumes don't use it)
    latest_model_path = latest_model_path.replace("dbfs:", "")
    
    with open(latest_model_path, 'rb') as f:
        model = pickle.load(f)
    
    model_version = latest_model_path.split("_")[-2] + "_" + latest_model_path.split("_")[-1].replace(".pkl", "")
    
    print(f"✅ Loaded model: {latest_model_path}")
    print(f"   Version: {model_version}")
    
    return model, model_version

model, model_version = load_latest_model()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Make a Prediction
# MAGIC 
# MAGIC ### Enter Patient Symptoms (0 = No, 1 = Yes)

# COMMAND ----------

# Create widgets for symptom input
dbutils.widgets.text("patient_id", "", "Patient ID")
dbutils.widgets.dropdown("chill_cold", "0", ["0", "1"], "Chills/Cold")
dbutils.widgets.dropdown("headache", "0", ["0", "1"], "Headache")
dbutils.widgets.dropdown("fever", "0", ["0", "1"], "Fever")
dbutils.widgets.dropdown("generalized_body_pain", "0", ["0", "1"], "Body Pain")
dbutils.widgets.dropdown("abdominal_pain", "0", ["0", "1"], "Abdominal Pain")
dbutils.widgets.dropdown("Loss_of_appetite", "0", ["0", "1"], "Loss of Appetite")
dbutils.widgets.dropdown("joint_pain", "0", ["0", "1"], "Joint Pain")
dbutils.widgets.dropdown("vomiting", "0", ["0", "1"], "Vomiting")
dbutils.widgets.dropdown("nausea", "0", ["0", "1"], "Nausea")
dbutils.widgets.dropdown("diarrhea", "0", ["0", "1"], "Diarrhea")

print("✅ Widgets created! Use the dropdowns above to enter symptoms.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Get Prediction

# COMMAND ----------

# Get values from widgets
patient_id = dbutils.widgets.get("patient_id") or f"PATIENT_{datetime.now().strftime('%Y%m%d%H%M%S')}"

symptoms = {
    "chill_cold": int(dbutils.widgets.get("chill_cold")),
    "headache": int(dbutils.widgets.get("headache")),
    "fever": int(dbutils.widgets.get("fever")),
    "generalized_body_pain": int(dbutils.widgets.get("generalized_body_pain")),
    "abdominal_pain": int(dbutils.widgets.get("abdominal_pain")),
    "Loss_of_appetite": int(dbutils.widgets.get("Loss_of_appetite")),
    "joint_pain": int(dbutils.widgets.get("joint_pain")),
    "vomiting": int(dbutils.widgets.get("vomiting")),
    "nausea": int(dbutils.widgets.get("nausea")),
    "diarrhea": int(dbutils.widgets.get("diarrhea"))
}

# Prepare input for model
X = np.array([[symptoms[col] for col in SYMPTOM_COLS]])

# Make prediction
prediction = model.predict(X)[0]
probability = model.predict_proba(X)[0]

# Display results
print("=" * 60)
print("MALARIA PREDICTION RESULT")
print("=" * 60)
print(f"Patient ID: {patient_id}")
print(f"Model Version: {model_version}")
print(f"\nSymptoms Reported:")
for symptom, value in symptoms.items():
    if value == 1:
        print(f"  ✓ {symptom.replace('_', ' ').title()}")
print(f"\n{'='*60}")
print(f"PREDICTION: {'POSITIVE for Malaria' if prediction == 1 else 'NEGATIVE for Malaria'}")
print(f"Confidence: {max(probability):.1%}")
print(f"{'='*60}")

# Save to predictions table
prediction_id = str(uuid.uuid4())
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, TimestampType

pred_schema = StructType([
    StructField("prediction_id", StringType(), False),
    StructField("patient_id", StringType(), False),
    StructField("prediction_timestamp", TimestampType(), False),
    StructField("prediction", IntegerType(), False),
    StructField("confidence", DoubleType(), False),
    StructField("model_version", StringType(), False)
] + [StructField(col, IntegerType(), False) for col in SYMPTOM_COLS])

pred_data = [{
    "prediction_id": prediction_id,
    "patient_id": patient_id,
    "prediction_timestamp": datetime.now(),
    "prediction": int(prediction),
    "confidence": float(max(probability)),
    "model_version": model_version,
    **symptoms
}]

pred_df = spark.createDataFrame(pred_data, schema=pred_schema)
pred_df.write.format("delta").mode("append").option("mergeSchema", "true").saveAsTable(f"{CATALOG}.{SCHEMA}.predictions")

print(f"\n✅ Prediction saved to database (ID: {prediction_id})")
print(f"\n⚠️  IMPORTANT: After clinical test, update with actual result below!")
print(f"   Prediction ID: {prediction_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📝 Submit Clinical Test Result (Feedback for Model Learning)
# MAGIC 
# MAGIC After performing the clinical test, enter the actual result here.
# MAGIC This feedback helps the model learn and improve!

# COMMAND ----------

# Create widgets for feedback
dbutils.widgets.text("feedback_prediction_id", "", "Prediction ID (from above)")
dbutils.widgets.dropdown("actual_result", "Select", ["Select", "0 - Negative", "1 - Positive"], "Actual Clinical Test Result")

print("✅ Feedback widgets created!")
print("\nInstructions:")
print("1. Copy the Prediction ID from above")
print("2. Paste it in 'Prediction ID' field")
print("3. Select the actual clinical test result")
print("4. Run the next cell to submit feedback")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Submit Feedback

# COMMAND ----------

feedback_pred_id = dbutils.widgets.get("feedback_prediction_id").strip()
actual_result_str = dbutils.widgets.get("actual_result")

if not feedback_pred_id:
    print("❌ Please enter a Prediction ID")
elif actual_result_str == "Select":
    print("❌ Please select the actual clinical test result")
else:
    # Parse actual result
    actual_result = int(actual_result_str.split(" - ")[0])
    
    # Update the prediction with actual result
    from pyspark.sql.functions import col, lit, when
    
    # Get the original prediction
    pred_table = spark.table(f"{CATALOG}.{SCHEMA}.predictions")
    original_pred = pred_table.filter(col("prediction_id") == feedback_pred_id).first()
    
    if not original_pred:
        print(f"❌ Prediction ID '{feedback_pred_id}' not found!")
    else:
        # Check if model was correct
        model_correct = (original_pred["prediction"] == actual_result)
        
        # Update the record with actual result
        updated_df = pred_table.withColumn(
            "actual_result",
            when(col("prediction_id") == feedback_pred_id, lit(actual_result))
            .otherwise(col("actual_result"))
        ).withColumn(
            "model_correct",
            when(col("prediction_id") == feedback_pred_id, lit(model_correct))
            .otherwise(col("model_correct"))
        ).withColumn(
            "feedback_timestamp",
            when(col("prediction_id") == feedback_pred_id, lit(datetime.now()))
            .otherwise(col("feedback_timestamp"))
        )
        
        # Save updated table
        updated_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{CATALOG}.{SCHEMA}.predictions")
        
        # Display result
        print("=" * 60)
        print("FEEDBACK RECORDED")
        print("=" * 60)
        print(f"Prediction ID: {feedback_pred_id}")
        print(f"Patient ID: {original_pred['patient_id']}")
        print(f"Model Predicted: {'POSITIVE' if original_pred['prediction'] == 1 else 'NEGATIVE'}")
        print(f"Actual Result: {'POSITIVE' if actual_result == 1 else 'NEGATIVE'}")
        print(f"Model was: {'✅ CORRECT' if model_correct else '❌ INCORRECT'}")
        print("=" * 60)
        
        # Check if we have enough feedback for retraining
        feedback_count = pred_table.filter(col("actual_result").isNotNull()).count()
        print(f"\n📊 Total feedback received: {feedback_count}")
        
        if feedback_count >= 50:
            print(f"\n🔄 Enough feedback collected! The continuous learning job will:")
            print(f"   1. Retrain the model with new data")
            print(f"   2. Evaluate if performance improved")
            print(f"   3. Update the model if it's better")
            print(f"\n💡 The scheduled job runs daily at 2 AM, or you can run it manually:")
            print(f"   Workflows → Jobs → Malaria_RL_Training_dev → Run Now")
        else:
            print(f"\n⏳ Need {50 - feedback_count} more feedback samples for retraining")
        
        print(f"\n✅ Feedback saved! Thank you for helping the model learn!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## View Recent Predictions

# COMMAND ----------

recent_predictions = spark.table(f"{CATALOG}.{SCHEMA}.predictions") \
    .orderBy(desc("prediction_timestamp")) \
    .limit(10) \
    .toPandas()

print("\n📊 RECENT PREDICTIONS")
print("=" * 60)
display(recent_predictions[[
    "patient_id", "prediction_timestamp", "prediction", 
    "confidence", "model_version"
]])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Performance Summary

# COMMAND ----------

performance = spark.table(f"{CATALOG}.{SCHEMA}.model_performance") \
    .orderBy(desc("metric_timestamp")) \
    .limit(1) \
    .toPandas()

if len(performance) > 0:
    print("\n📈 MODEL PERFORMANCE")
    print("=" * 60)
    print(f"Accuracy:  {performance['accuracy'].iloc[0]:.1%}")
    print(f"Precision: {performance['precision_score'].iloc[0]:.1%}")
    print(f"Recall:    {performance['recall'].iloc[0]:.1%}")
    print(f"F1 Score:  {performance['f1_score'].iloc[0]:.1%}")
    print("=" * 60)
else:
    print("No performance metrics available yet.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Clear Widgets (Optional)

# COMMAND ----------

# Uncomment to clear all widgets
# dbutils.widgets.removeAll()
