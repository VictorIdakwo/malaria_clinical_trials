# Databricks notebook source
# MAGIC %md
# MAGIC # Reinforcement Learning Model Training - Malaria Prediction
# MAGIC 
# MAGIC This notebook implements a contextual bandit RL approach where:
# MAGIC - **State**: Patient symptoms (10 binary features)
# MAGIC - **Action**: Predict positive (1) or negative (0) for malaria
# MAGIC - **Reward**: +1 for correct prediction, -1 for incorrect
# MAGIC - The model learns from clinical feedback to improve over time

# COMMAND ----------

# MAGIC %pip install mlflow scikit-learn numpy pandas xgboost gymnasium

# COMMAND ----------

import mlflow
import mlflow.sklearn
import mlflow.pyfunc
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from pyspark.sql.functions import col, current_timestamp, lit
from datetime import datetime
import json
import pickle

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

# Get configuration from widgets or use defaults
try:
    CATALOG = dbutils.widgets.get("catalog")
except:
    CATALOG = "eha"

try:
    SCHEMA = dbutils.widgets.get("schema")
except:
    SCHEMA = "malaria_catalog"

try:
    VOLUME = dbutils.widgets.get("volume")
except:
    VOLUME = "clinical_trial"

# Set MLflow experiment
mlflow.set_experiment(f"/Shared/malaria_rl_experiment")

# Symptom columns (features)
SYMPTOM_COLS = [
    "chill_cold", "headache", "fever", "generalized_body_pain",
    "abdominal_pain", "Loss_of_appetite", "joint_pain",
    "vomiting", "nausea", "diarrhea"
]

print(f"Using Catalog: {CATALOG}, Schema: {SCHEMA}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Training Data

# COMMAND ----------

# Load training data
df_train = spark.table(f"{CATALOG}.{SCHEMA}.malaria_training_data")

# Convert to pandas for ML training
pdf_train = df_train.select(SYMPTOM_COLS + ["Cases"]).toPandas()

print(f"Training data shape: {pdf_train.shape}")
print(f"Class distribution:\n{pdf_train['Cases'].value_counts()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prepare Features and Labels

# COMMAND ----------

# Prepare features and labels
X = pdf_train[SYMPTOM_COLS].values
y = pdf_train["Cases"].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")
print(f"Positive cases in train: {np.sum(y_train)} ({np.mean(y_train)*100:.2f}%)")
print(f"Positive cases in test: {np.sum(y_test)} ({np.mean(y_test)*100:.2f}%)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Contextual Bandit RL Model

# COMMAND ----------

class MalariaRLModel(mlflow.pyfunc.PythonModel):
    """
    Reinforcement Learning model for malaria prediction using contextual bandits.
    
    The model maintains:
    - A base classifier (Random Forest or Gradient Boosting)
    - Exploration-exploitation strategy (epsilon-greedy)
    - Online learning capability for continuous improvement
    """
    
    def __init__(self, base_model=None, epsilon=0.1, learning_rate=0.01):
        """
        Initialize RL model.
        
        Args:
            base_model: Base classifier (sklearn model)
            epsilon: Exploration rate for epsilon-greedy strategy
            learning_rate: Learning rate for model updates
        """
        self.base_model = base_model
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.total_predictions = 0
        self.correct_predictions = 0
        self.reward_history = []
        self.feature_names = SYMPTOM_COLS
        
    def predict(self, context, model_input):
        """
        Make prediction with exploration-exploitation strategy.
        
        Args:
            context: MLflow context (not used but required)
            model_input: Input features (pandas DataFrame or numpy array)
        
        Returns:
            Predictions with probabilities
        """
        if isinstance(model_input, pd.DataFrame):
            X = model_input[self.feature_names].values
        else:
            X = model_input
        
        # Get base model predictions
        y_pred_proba = self.base_model.predict_proba(X)
        y_pred = self.base_model.predict(X)
        
        # Epsilon-greedy exploration
        if np.random.random() < self.epsilon:
            # Explore: random action
            y_pred = np.random.randint(0, 2, size=len(y_pred))
        
        # Return predictions with probabilities
        return pd.DataFrame({
            'prediction': y_pred,
            'probability_negative': y_pred_proba[:, 0],
            'probability_positive': y_pred_proba[:, 1]
        })
    
    def compute_reward(self, predicted, actual):
        """
        Compute reward for prediction.
        
        Args:
            predicted: Predicted class
            actual: Actual class
        
        Returns:
            reward: +1 for correct, -1 for incorrect
        """
        return 1.0 if predicted == actual else -1.0
    
    def update_with_feedback(self, X_feedback, y_actual, y_predicted):
        """
        Update model with feedback from clinical trials.
        
        Args:
            X_feedback: Feature matrix of feedback samples
            y_actual: Actual test results
            y_predicted: Previously predicted results
        """
        rewards = [self.compute_reward(pred, actual) 
                   for pred, actual in zip(y_predicted, y_actual)]
        
        self.reward_history.extend(rewards)
        self.total_predictions += len(y_actual)
        self.correct_predictions += sum([r > 0 for r in rewards])
        
        # Retrain model with new feedback (online learning)
        if len(X_feedback) > 0:
            self.base_model.fit(X_feedback, y_actual)
        
        return np.mean(rewards)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train Base Model

# COMMAND ----------

# Train initial model with MLflow tracking
with mlflow.start_run(run_name="malaria_rl_initial_training") as run:
    
    # Log parameters
    mlflow.log_param("model_type", "gradient_boosting")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 5)
    mlflow.log_param("epsilon", 0.1)
    mlflow.log_param("train_size", len(X_train))
    mlflow.log_param("test_size", len(X_test))
    
    # Train base classifier
    base_classifier = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
    
    base_classifier.fit(X_train, y_train)
    
    # Evaluate on test set
    y_pred = base_classifier.predict(X_test)
    y_pred_proba = base_classifier.predict_proba(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    
    print(f"Model Performance:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))
    
    # Create RL model wrapper
    rl_model = MalariaRLModel(base_model=base_classifier, epsilon=0.1)
    
    # Log model with MLflow
    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=rl_model,
        registered_model_name="malaria_rl_model",
        input_example=pd.DataFrame([X_test[0]], columns=SYMPTOM_COLS),
        signature=mlflow.models.infer_signature(
            pd.DataFrame(X_test, columns=SYMPTOM_COLS),
            pd.DataFrame({'prediction': y_pred})
        )
    )
    
    # Save model to volume (in ml_models subfolder)
    model_version = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create ml_models directory if it doesn't exist (use dbutils for Unity Catalog)
    ml_models_dir = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/ml_models"
    try:
        dbutils.fs.mkdirs(ml_models_dir)
    except:
        pass  # Directory already exists
    
    model_path = f"{ml_models_dir}/malaria_rl_model_{model_version}.pkl"
    
    # For Unity Catalog Volumes, use the path directly (no /dbfs/ prefix needed)
    with open(model_path, 'wb') as f:
        pickle.dump(rl_model, f)
    
    mlflow.log_param("model_path", model_path)
    
    print(f"\nModel saved to: {model_path}")
    print(f"MLflow Run ID: {run.info.run_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save Model Performance Metrics

# COMMAND ----------

# Save initial performance metrics to table
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
    "accuracy": float(accuracy),
    "precision_score": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "total_predictions": int(len(y_test)),
    "correct_predictions": int(np.sum(y_test == y_pred)),
    "reward_sum": float(np.sum(y_test == y_pred) - np.sum(y_test != y_pred)),
    "avg_reward": float((np.sum(y_test == y_pred) - np.sum(y_test != y_pred)) / len(y_test))
}]

perf_df = spark.createDataFrame(performance_data, schema=performance_schema)
perf_df.write.format("delta").mode("append").saveAsTable(
    f"{CATALOG}.{SCHEMA}.model_performance"
)

print("Performance metrics saved to model_performance table")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Importance Analysis

# COMMAND ----------

# Get feature importance
feature_importance = pd.DataFrame({
    'feature': SYMPTOM_COLS,
    'importance': base_classifier.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)

# Log feature importance as artifact
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.barh(feature_importance['feature'], feature_importance['importance'])
plt.xlabel('Importance')
plt.title('Feature Importance for Malaria Prediction')
plt.tight_layout()
plt.savefig('/tmp/feature_importance.png')

mlflow.log_artifact('/tmp/feature_importance.png')
plt.close()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Model Metadata

# COMMAND ----------

# Create metadata file
metadata = {
    "model_version": model_version,
    "training_date": datetime.now().isoformat(),
    "model_type": "GradientBoostingClassifier with RL wrapper",
    "feature_columns": SYMPTOM_COLS,
    "target_column": "Cases",
    "metrics": {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1)
    },
    "training_samples": int(len(X_train)),
    "test_samples": int(len(X_test)),
    "epsilon": 0.1,
    "model_path": model_path
}

metadata_path = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/ml_models/model_metadata_{model_version}.json"
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"Metadata saved to: {metadata_path}")
print("\nModel training completed successfully!")
