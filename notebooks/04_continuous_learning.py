# Databricks notebook source
# MAGIC %md
# MAGIC # Continuous Learning - Model Update Pipeline
# MAGIC 
# MAGIC This notebook implements the continuous learning loop with **dual-threshold retraining**:
# MAGIC 
# MAGIC ## Retraining Triggers:
# MAGIC - **Fast-track**: ≥25 feedback samples in last 24 hours → Immediate retraining
# MAGIC - **Standard**: ≥50 total feedback samples → Regular retraining
# MAGIC 
# MAGIC ## Process:
# MAGIC - Collects feedback from clinical trials
# MAGIC - Checks retraining criteria (daily OR total)
# MAGIC - Retrains the model with new data
# MAGIC - Evaluates improvement
# MAGIC - Updates the model in production if performance improves

# COMMAND ----------

# MAGIC %pip install mlflow scikit-learn numpy pandas xgboost

# COMMAND ----------

import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from pyspark.sql.functions import col, count, lit
from datetime import datetime, timedelta
import pickle
import json

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

# Retraining thresholds
MIN_FEEDBACK_SAMPLES = 50  # Total feedback threshold
DAILY_FAST_TRACK_THRESHOLD = 25  # Daily feedback for immediate retraining

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
        self.feature_names = SYMPTOM_COLS
        
    def predict(self, X):
        if hasattr(X, 'values'):
            X = X.values
        return self.base_model.predict(X)
    
    def predict_proba(self, X):
        if hasattr(X, 'values'):
            X = X.values
        return self.base_model.predict_proba(X)

# Minimum accuracy improvement required to update model
MIN_ACCURACY_IMPROVEMENT = 0.01  # 1%

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Feedback Data

# COMMAND ----------

# Get predictions with actual results (feedback)
feedback_df = spark.table(f"{CATALOG}.{SCHEMA}.predictions") \
    .filter(col("actual_result").isNotNull())

feedback_count = feedback_df.count()

# Check feedback collected in the last 24 hours (fast-track retraining)
from pyspark.sql.functions import current_timestamp, expr
daily_feedback_df = feedback_df.filter(
    col("feedback_timestamp") >= expr("current_timestamp() - INTERVAL 1 DAY")
)
daily_feedback_count = daily_feedback_df.count()

print(f"📊 Feedback Summary:")
print(f"   Total feedback samples: {feedback_count}")
print(f"   Feedback in last 24 hours: {daily_feedback_count}")
print(f"")
print(f"🎯 Retraining Criteria:")
print(f"   • Fast-track: {daily_feedback_count}/{DAILY_FAST_TRACK_THRESHOLD} daily samples")
print(f"   • Standard: {feedback_count}/{MIN_FEEDBACK_SAMPLES} total samples")

# Determine if retraining should proceed
should_retrain = False
retrain_reason = ""

if daily_feedback_count >= DAILY_FAST_TRACK_THRESHOLD:
    should_retrain = True
    retrain_reason = f"Fast-track: {daily_feedback_count} samples collected today (≥{DAILY_FAST_TRACK_THRESHOLD})"
    print(f"")
    print(f"✅ {retrain_reason}")
    print(f"🚀 Initiating fast-track retraining...")
    
elif feedback_count >= MIN_FEEDBACK_SAMPLES:
    should_retrain = True
    retrain_reason = f"Standard: {feedback_count} total samples collected (≥{MIN_FEEDBACK_SAMPLES})"
    print(f"")
    print(f"✅ {retrain_reason}")
    print(f"🔄 Initiating standard retraining...")
    
else:
    print(f"")
    print(f"⚠️ Retraining criteria not met:")
    print(f"   • Need {DAILY_FAST_TRACK_THRESHOLD - daily_feedback_count} more daily samples for fast-track")
    print(f"   OR")
    print(f"   • Need {MIN_FEEDBACK_SAMPLES - feedback_count} more total samples for standard retrain")
    print(f"⏸️ Skipping retraining...")
    dbutils.notebook.exit(json.dumps({
        "status": "skipped",
        "reason": f"Insufficient feedback - Daily: {daily_feedback_count}/{DAILY_FAST_TRACK_THRESHOLD}, Total: {feedback_count}/{MIN_FEEDBACK_SAMPLES}"
    }))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prepare Training Data

# COMMAND ----------

# Combine original training data with feedback
original_data = spark.table(f"{CATALOG}.{SCHEMA}.malaria_training_data")

# Get recent feedback (last 30 days)
recent_feedback = feedback_df.filter(
    col("feedback_timestamp") >= lit(datetime.now() - timedelta(days=30))
)

# Prepare feedback data for training
feedback_train = recent_feedback.select(
    SYMPTOM_COLS + ["actual_result"]
).withColumnRenamed("actual_result", "Cases")

# Combine datasets
combined_data = original_data.select(SYMPTOM_COLS + ["Cases"]) \
    .union(feedback_train)

print(f"Combined training data: {combined_data.count()} samples")

# Convert to pandas
pdf_train = combined_data.toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Split Data for Validation

# COMMAND ----------

from sklearn.model_selection import train_test_split

X = pdf_train[SYMPTOM_COLS].values
y = pdf_train["Cases"].values

# Split: Use feedback data as validation set
feedback_pdf = recent_feedback.select(SYMPTOM_COLS + ["actual_result"]).toPandas()
X_feedback = feedback_pdf[SYMPTOM_COLS].values
y_feedback = feedback_pdf["actual_result"].values

# Use remaining data for training
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape}")
print(f"Validation set: {X_val.shape}")
print(f"Feedback test set: {X_feedback.shape}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Current Model Performance

# COMMAND ----------

# Get latest model performance
latest_perf = spark.table(f"{CATALOG}.{SCHEMA}.model_performance") \
    .orderBy(col("metric_timestamp").desc()) \
    .limit(1) \
    .collect()[0]

current_accuracy = latest_perf["accuracy"]
current_f1 = latest_perf["f1_score"]

print(f"Current model performance:")
print(f"  Accuracy: {current_accuracy:.4f}")
print(f"  F1 Score: {current_f1:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train Updated Model

# COMMAND ----------

mlflow.set_experiment(f"/Shared/malaria_rl_experiment")

with mlflow.start_run(run_name=f"continuous_learning_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
    
    # Log parameters
    mlflow.log_param("model_type", "gradient_boosting")
    mlflow.log_param("training_type", "continuous_learning")
    mlflow.log_param("feedback_samples", len(X_feedback))
    mlflow.log_param("total_training_samples", len(X_train))
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 5)
    
    # Train new model
    new_model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
    
    new_model.fit(X_train, y_train)
    
    # Evaluate on validation set
    y_val_pred = new_model.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_precision = precision_score(y_val, y_val_pred)
    val_recall = recall_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred)
    
    # Evaluate on feedback data (real-world performance)
    y_feedback_pred = new_model.predict(X_feedback)
    feedback_accuracy = accuracy_score(y_feedback, y_feedback_pred)
    feedback_precision = precision_score(y_feedback, y_feedback_pred)
    feedback_recall = recall_score(y_feedback, y_feedback_pred)
    feedback_f1 = f1_score(y_feedback, y_feedback_pred)
    
    # Log metrics
    mlflow.log_metric("val_accuracy", val_accuracy)
    mlflow.log_metric("val_precision", val_precision)
    mlflow.log_metric("val_recall", val_recall)
    mlflow.log_metric("val_f1_score", val_f1)
    
    mlflow.log_metric("feedback_accuracy", feedback_accuracy)
    mlflow.log_metric("feedback_precision", feedback_precision)
    mlflow.log_metric("feedback_recall", feedback_recall)
    mlflow.log_metric("feedback_f1_score", feedback_f1)
    
    # Calculate improvement
    accuracy_improvement = feedback_accuracy - current_accuracy
    f1_improvement = feedback_f1 - current_f1
    
    mlflow.log_metric("accuracy_improvement", accuracy_improvement)
    mlflow.log_metric("f1_improvement", f1_improvement)
    
    print(f"\nNew model performance on feedback data:")
    print(f"  Accuracy:  {feedback_accuracy:.4f} (improvement: {accuracy_improvement:+.4f})")
    print(f"  Precision: {feedback_precision:.4f}")
    print(f"  Recall:    {feedback_recall:.4f}")
    print(f"  F1 Score:  {feedback_f1:.4f} (improvement: {f1_improvement:+.4f})")
    
    # COMMAND ----------
    
    # MAGIC %md
    # MAGIC ## Decide Whether to Update Model
    
    # COMMAND ----------
    
    should_update = accuracy_improvement >= MIN_ACCURACY_IMPROVEMENT
    
    mlflow.log_param("model_updated", should_update)
    
    if should_update:
        print(f"\n✅ Model shows improvement! Updating production model...")
        
        # Create RL wrapper
        from notebooks.train_rl_model import MalariaRLModel
        rl_model = MalariaRLModel(base_model=new_model, epsilon=0.1)
        
        # Log model
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=rl_model,
            registered_model_name="malaria_rl_model"
        )
        
        # Save to volume
        model_version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create ml_models directory if it doesn't exist
        ml_models_dir = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/ml_models"
        try:
            dbutils.fs.mkdirs(ml_models_dir)
        except:
            pass
        
        model_path = f"{ml_models_dir}/malaria_rl_model_{model_version}.pkl"
        
        with open(model_path, 'wb') as f:
            pickle.dump(rl_model, f)
        
        mlflow.log_param("model_path", model_path)
        
        # Save performance metrics
        from pyspark.sql.types import StructType, StructField, TimestampType, StringType, DoubleType, IntegerType
        
        performance_schema = StructType([
            StructField("metric_timestamp", TimestampType(), False),
            StructField("model_version", StringType(), False),
            StructField("accuracy", DoubleType(), False),
            StructField("precision_score", DoubleType(), False),
            StructField("recall", DoubleType(), False),
            StructField("f1_score", DoubleType(), False),
            StructField("total_predictions", IntegerType(), False),
            StructField("correct_predictions", IntegerType(), False),
            StructField("reward_sum", DoubleType(), False),
            StructField("avg_reward", DoubleType(), False)
        ])
        
        performance_data = [{
            "metric_timestamp": datetime.now(),
            "model_version": model_version,
            "accuracy": float(feedback_accuracy),
            "precision_score": float(feedback_precision),
            "recall": float(feedback_recall),
            "f1_score": float(feedback_f1),
            "total_predictions": int(len(y_feedback)),
            "correct_predictions": int(np.sum(y_feedback == y_feedback_pred)),
            "reward_sum": float(np.sum(y_feedback == y_feedback_pred) - np.sum(y_feedback != y_feedback_pred)),
            "avg_reward": float((np.sum(y_feedback == y_feedback_pred) - np.sum(y_feedback != y_feedback_pred)) / len(y_feedback))
        }]
        
        perf_df = spark.createDataFrame(performance_data, schema=performance_schema)
        perf_df.write.format("delta").mode("append").saveAsTable(
            f"{CATALOG}.{SCHEMA}.model_performance"
        )
        
        print(f"✅ Model updated successfully!")
        print(f"   Version: {model_version}")
        print(f"   Path: {model_path}")
        
        result = {
            "status": "updated",
            "model_version": model_version,
            "accuracy_improvement": float(accuracy_improvement),
            "f1_improvement": float(f1_improvement),
            "new_accuracy": float(feedback_accuracy),
            "new_f1": float(feedback_f1)
        }
        
    else:
        print(f"\n⚠️ Model improvement ({accuracy_improvement:.4f}) below threshold ({MIN_ACCURACY_IMPROVEMENT})")
        print("Keeping current model in production.")
        
        result = {
            "status": "not_updated",
            "reason": "insufficient_improvement",
            "accuracy_improvement": float(accuracy_improvement),
            "current_accuracy": float(current_accuracy),
            "new_accuracy": float(feedback_accuracy)
        }
    
    mlflow.log_dict(result, "update_result.json")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary

# COMMAND ----------

print("\n" + "="*60)
print("CONTINUOUS LEARNING SUMMARY")
print("="*60)
print(f"Feedback samples processed: {feedback_count}")
print(f"Current model accuracy: {current_accuracy:.4f}")
print(f"New model accuracy: {feedback_accuracy:.4f}")
print(f"Improvement: {accuracy_improvement:+.4f}")
print(f"Model updated: {should_update}")
print("="*60)

# Return result
dbutils.notebook.exit(json.dumps(result))
