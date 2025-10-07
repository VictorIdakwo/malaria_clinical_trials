-- Databricks SQL Dashboard Queries
-- Use these queries to create visualizations in Databricks SQL Dashboard

-- =============================================
-- QUERY 1: Model Performance Overview
-- =============================================
-- Widget: Counter + Line Chart
SELECT 
  metric_timestamp,
  model_version,
  accuracy,
  precision_score as precision,
  recall,
  f1_score,
  total_predictions,
  correct_predictions
FROM eha.malaria_catalog.model_performance
ORDER BY metric_timestamp DESC;

-- =============================================
-- QUERY 2: Latest Model Metrics (KPI Cards)
-- =============================================
-- Widget: Counter (4 separate counters)
SELECT 
  ROUND(accuracy * 100, 2) as accuracy_pct,
  ROUND(precision_score * 100, 2) as precision_pct,
  ROUND(recall * 100, 2) as recall_pct,
  ROUND(f1_score * 100, 2) as f1_pct
FROM eha.malaria_catalog.model_performance
ORDER BY metric_timestamp DESC
LIMIT 1;

-- =============================================
-- QUERY 3: Predictions Summary
-- =============================================
-- Widget: Counter
SELECT 
  COUNT(*) as total_predictions,
  SUM(CASE WHEN prediction = 1 THEN 1 ELSE 0 END) as positive_predictions,
  SUM(CASE WHEN prediction = 0 THEN 1 ELSE 0 END) as negative_predictions,
  SUM(CASE WHEN actual_result IS NOT NULL THEN 1 ELSE 0 END) as with_feedback,
  SUM(CASE WHEN model_correct = true THEN 1 ELSE 0 END) as correct_predictions
FROM eha.malaria_catalog.predictions;

-- =============================================
-- QUERY 4: Prediction Accuracy Over Time
-- =============================================
-- Widget: Line Chart
SELECT 
  DATE(prediction_timestamp) as date,
  COUNT(*) as total,
  SUM(CASE WHEN model_correct = true THEN 1 ELSE 0 END) as correct,
  ROUND(SUM(CASE WHEN model_correct = true THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as accuracy_pct
FROM eha.malaria_catalog.predictions
WHERE actual_result IS NOT NULL
GROUP BY DATE(prediction_timestamp)
ORDER BY date DESC
LIMIT 30;

-- =============================================
-- QUERY 5: Recent Predictions
-- =============================================
-- Widget: Table
SELECT 
  patient_id,
  prediction_timestamp,
  CASE WHEN prediction = 1 THEN 'POSITIVE' ELSE 'NEGATIVE' END as predicted,
  confidence,
  CASE WHEN actual_result = 1 THEN 'POSITIVE' 
       WHEN actual_result = 0 THEN 'NEGATIVE'
       ELSE 'Pending' END as actual,
  CASE WHEN model_correct = true THEN '✓ Correct'
       WHEN model_correct = false THEN '✗ Incorrect'
       ELSE 'Pending' END as result,
  model_version
FROM eha.malaria_catalog.predictions
ORDER BY prediction_timestamp DESC
LIMIT 50;

-- =============================================
-- QUERY 6: Feedback Progress
-- =============================================
-- Widget: Counter + Progress Bar
SELECT 
  COUNT(CASE WHEN actual_result IS NOT NULL THEN 1 END) as feedback_count,
  50 as target,
  ROUND(COUNT(CASE WHEN actual_result IS NOT NULL THEN 1 END) * 100.0 / 50, 2) as progress_pct
FROM eha.malaria_catalog.predictions;

-- =============================================
-- QUERY 7: Symptom Analysis (Most Common)
-- =============================================
-- Widget: Bar Chart
SELECT 
  'Fever' as symptom, SUM(fever) as count FROM eha.malaria_catalog.predictions
UNION ALL
SELECT 'Headache', SUM(headache) FROM eha.malaria_catalog.predictions
UNION ALL
SELECT 'Chills', SUM(chill_cold) FROM eha.malaria_catalog.predictions
UNION ALL
SELECT 'Body Pain', SUM(generalized_body_pain) FROM eha.malaria_catalog.predictions
UNION ALL
SELECT 'Vomiting', SUM(vomiting) FROM eha.malaria_catalog.predictions
UNION ALL
SELECT 'Nausea', SUM(nausea) FROM eha.malaria_catalog.predictions
UNION ALL
SELECT 'Joint Pain', SUM(joint_pain) FROM eha.malaria_catalog.predictions
UNION ALL
SELECT 'Diarrhea', SUM(diarrhea) FROM eha.malaria_catalog.predictions
UNION ALL
SELECT 'Abdominal Pain', SUM(abdominal_pain) FROM eha.malaria_catalog.predictions
UNION ALL
SELECT 'Loss of Appetite', SUM(Loss_of_appetite) FROM eha.malaria_catalog.predictions
ORDER BY count DESC;

-- =============================================
-- QUERY 8: Prediction Distribution by Hour
-- =============================================
-- Widget: Bar Chart
SELECT 
  HOUR(prediction_timestamp) as hour,
  COUNT(*) as prediction_count
FROM eha.malaria_catalog.predictions
GROUP BY HOUR(prediction_timestamp)
ORDER BY hour;

-- =============================================
-- QUERY 9: Model Version Performance Comparison
-- =============================================
-- Widget: Table
SELECT 
  model_version,
  COUNT(*) as predictions,
  SUM(CASE WHEN model_correct = true THEN 1 ELSE 0 END) as correct,
  ROUND(SUM(CASE WHEN model_correct = true THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as accuracy
FROM eha.malaria_catalog.predictions
WHERE actual_result IS NOT NULL
GROUP BY model_version
ORDER BY model_version DESC;

-- =============================================
-- QUERY 10: Confusion Matrix Data
-- =============================================
-- Widget: Table
SELECT 
  SUM(CASE WHEN prediction = 1 AND actual_result = 1 THEN 1 ELSE 0 END) as true_positive,
  SUM(CASE WHEN prediction = 0 AND actual_result = 0 THEN 1 ELSE 0 END) as true_negative,
  SUM(CASE WHEN prediction = 1 AND actual_result = 0 THEN 1 ELSE 0 END) as false_positive,
  SUM(CASE WHEN prediction = 0 AND actual_result = 1 THEN 1 ELSE 0 END) as false_negative
FROM eha.malaria_catalog.predictions
WHERE actual_result IS NOT NULL;
