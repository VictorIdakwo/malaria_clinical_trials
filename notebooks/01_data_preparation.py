# Databricks notebook source
# MAGIC %md
# MAGIC # Data Preparation for Malaria RL Clinical Trial
# MAGIC 
# MAGIC This notebook:
# MAGIC - Loads the clinical data
# MAGIC - Creates Unity Catalog tables
# MAGIC - Prepares data for RL training
# MAGIC - Sets up the feedback collection table

# COMMAND ----------

# MAGIC %pip install mlflow scikit-learn pandas numpy

# COMMAND ----------

import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, current_timestamp
from pyspark.sql.types import *

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

# Configuration
CATALOG = "eha"
SCHEMA = "malaria_catalog"
VOLUME = "clinical_trial"
DATA_PATH = "/Volumes/eha/malaria_catalog/clinical_trial/data/Clinical Main Data for Databricks.csv"

# Use existing catalog, schema, and volume (already created)
spark.sql(f"USE CATALOG {CATALOG}")
spark.sql(f"USE SCHEMA {SCHEMA}")

print(f"✅ Using catalog: {CATALOG}")
print(f"✅ Using schema: {SCHEMA}")
print(f"✅ Using volume: {VOLUME}")
print(f"✅ Data path: {DATA_PATH}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load and Process Clinical Data

# COMMAND ----------

# Define schema for the clinical data
schema = StructType([
    StructField("lab_no", StringType(), True),
    StructField("sex", StringType(), True),
    StructField("age", IntegerType(), True),
    StructField("temperature", DoubleType(), True),
    StructField("Thick_Film", StringType(), True),
    StructField("Cases", IntegerType(), True),
    StructField("chill_cold", IntegerType(), True),
    StructField("headache", IntegerType(), True),
    StructField("fever", IntegerType(), True),
    StructField("generalized_body_pain", IntegerType(), True),
    StructField("abdominal_pain", IntegerType(), True),
    StructField("Loss_of_appetite", IntegerType(), True),
    StructField("joint_pain", IntegerType(), True),
    StructField("vomiting", IntegerType(), True),
    StructField("nausea", IntegerType(), True),
    StructField("diarrhea", IntegerType(), True)
])

# COMMAND ----------

# Load the CSV data
# Note: Update this path to point to your actual data location
# For local file, first upload to DBFS or Volume
df = spark.read.csv(
    DATA_PATH,
    header=True,
    inferSchema=True
)

# Display sample
display(df.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Quality Checks

# COMMAND ----------

# Check for missing values
print("Missing values per column:")
df.select([
    (col(c).isNull().cast("int")).alias(c) 
    for c in df.columns
]).groupBy().sum().show()

# Basic statistics
print("\nDataset Statistics:")
print(f"Total records: {df.count()}")
print(f"Positive cases: {df.filter(col('Cases') == 1).count()}")
print(f"Negative cases: {df.filter(col('Cases') == 0).count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Main Training Table

# COMMAND ----------

# Clean and standardize column names
df_clean = df.withColumnRenamed("Thick Film", "Thick_Film") \
             .withColumnRenamed("generalized body pain", "generalized_body_pain") \
             .withColumnRenamed("abdominal pain", "abdominal_pain") \
             .withColumnRenamed("Loss of appetite", "Loss_of_appetite") \
             .withColumnRenamed("joint pain", "joint_pain")

# Fill null values with 0 for all symptom columns (binary features)
symptom_cols = [
    "chill_cold", "headache", "fever", "generalized_body_pain",
    "abdominal_pain", "Loss_of_appetite", "joint_pain",
    "vomiting", "nausea", "diarrhea", "Cases"
]

# Fill nulls with 0 for symptom columns
for col_name in symptom_cols:
    if col_name in df_clean.columns:
        df_clean = df_clean.fillna(0, subset=[col_name])

print(f"✅ Filled null values in symptom columns")

# Create or replace the main training table
df_clean.write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(f"{CATALOG}.{SCHEMA}.malaria_training_data")

print(f"Training data saved to {CATALOG}.{SCHEMA}.malaria_training_data")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Prediction and Feedback Tables

# COMMAND ----------

# Create table for storing predictions
spark.sql(f"""
CREATE TABLE IF NOT EXISTS {CATALOG}.{SCHEMA}.predictions (
    prediction_id STRING,
    patient_id STRING,
    prediction_timestamp TIMESTAMP,
    chill_cold INT,
    headache INT,
    fever INT,
    generalized_body_pain INT,
    abdominal_pain INT,
    Loss_of_appetite INT,
    joint_pain INT,
    vomiting INT,
    nausea INT,
    diarrhea INT,
    predicted_case INT,
    prediction_probability DOUBLE,
    model_version STRING,
    actual_result INT,
    feedback_timestamp TIMESTAMP,
    model_correct BOOLEAN
) USING DELTA
TBLPROPERTIES (
    'delta.enableChangeDataFeed' = 'true'
)
""")

print(f"Predictions table created: {CATALOG}.{SCHEMA}.predictions")

# COMMAND ----------

# Create table for model performance tracking (drop first to ensure clean schema)
spark.sql(f"DROP TABLE IF EXISTS {CATALOG}.{SCHEMA}.model_performance")

spark.sql(f"""
CREATE TABLE {CATALOG}.{SCHEMA}.model_performance (
    metric_timestamp TIMESTAMP,
    model_version STRING,
    accuracy DOUBLE,
    precision_score DOUBLE,
    recall DOUBLE,
    f1_score DOUBLE,
    total_predictions INT,
    correct_predictions INT,
    reward_sum DOUBLE,
    avg_reward DOUBLE
) USING DELTA
""")

print(f"Model performance table created: {CATALOG}.{SCHEMA}.model_performance")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Engineering

# COMMAND ----------

# Define symptom columns
SYMPTOM_COLS = [
    "chill_cold", "headache", "fever", "generalized_body_pain",
    "abdominal_pain", "Loss_of_appetite", "joint_pain",
    "vomiting", "nausea", "diarrhea"
]

# Create feature vector
from pyspark.ml.feature import VectorAssembler

assembler = VectorAssembler(
    inputCols=SYMPTOM_COLS,
    outputCol="features",
    handleInvalid="skip"  # Skip rows with null values
)

df_features = assembler.transform(df_clean)

# Save feature engineered data
df_features.write.format("delta") \
    .mode("overwrite") \
    .saveAsTable(f"{CATALOG}.{SCHEMA}.malaria_features")

print(f"Feature data saved to {CATALOG}.{SCHEMA}.malaria_features")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Export Configuration

# COMMAND ----------

# Save configuration for other notebooks
dbutils.widgets.text("catalog", CATALOG)
dbutils.widgets.text("schema", SCHEMA)
dbutils.widgets.text("volume", VOLUME)

print("Data preparation completed successfully!")
print(f"\nConfiguration:")
print(f"  Catalog: {CATALOG}")
print(f"  Schema: {SCHEMA}")
print(f"  Volume: {VOLUME}")
print(f"\nTables created:")
print(f"  - {CATALOG}.{SCHEMA}.malaria_training_data")
print(f"  - {CATALOG}.{SCHEMA}.malaria_features")
print(f"  - {CATALOG}.{SCHEMA}.predictions")
print(f"  - {CATALOG}.{SCHEMA}.model_performance")
