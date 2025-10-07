# 📊 Create Interactive Databricks Dashboard

This guide shows you how to create a **professional, interactive dashboard** in Databricks.

---

## 🎯 Dashboard Features

Your dashboard will show:
- ✅ **Real-time Model Performance** (Accuracy, Precision, Recall, F1)
- ✅ **Prediction History** (All predictions with actual results)
- ✅ **Feedback Progress** (How many samples until retraining)
- ✅ **Symptom Analysis** (Most common symptoms)
- ✅ **Model Learning Trends** (Accuracy over time)
- ✅ **Confusion Matrix** (True/False Positives/Negatives)

---

## 📋 Step-by-Step Setup (10 minutes)

### **Step 1: Open Databricks SQL** (1 min)

1. Go to your Databricks workspace
2. Click **SQL** in the left sidebar
3. Click **Dashboards** → **Create Dashboard**
4. Name it: **"Malaria Clinical Trial - Model Monitoring"**
5. Click **Create**

---

### **Step 2: Create Visualizations** (8 min)

For each query below, create a visualization:

#### **A) Model Performance KPIs** (4 Counter Cards)

**Query Name:** "Latest Model Metrics"
```sql
SELECT 
  ROUND(accuracy * 100, 2) as accuracy_pct,
  ROUND(precision_score * 100, 2) as precision_pct,
  ROUND(recall * 100, 2) as recall_pct,
  ROUND(f1_score * 100, 2) as f1_pct
FROM eha.malaria_catalog.model_performance
ORDER BY metric_timestamp DESC
LIMIT 1;
```

**Visualization:**
1. Click **Add** → **Visualization**
2. Paste the query above
3. Select **Counter** visualization
4. Create 4 separate counters:
   - Counter 1: `accuracy_pct` - Label: "Accuracy %"
   - Counter 2: `precision_pct` - Label: "Precision %"
   - Counter 3: `recall_pct` - Label: "Recall %"
   - Counter 4: `f1_pct` - Label: "F1 Score %"
5. Add to dashboard

---

#### **B) Predictions Summary** (Counter Cards)

**Query Name:** "Prediction Statistics"
```sql
SELECT 
  COUNT(*) as total_predictions,
  SUM(CASE WHEN prediction = 1 THEN 1 ELSE 0 END) as positive_predictions,
  SUM(CASE WHEN prediction = 0 THEN 1 ELSE 0 END) as negative_predictions,
  SUM(CASE WHEN actual_result IS NOT NULL THEN 1 ELSE 0 END) as with_feedback,
  SUM(CASE WHEN model_correct = true THEN 1 ELSE 0 END) as correct_predictions
FROM eha.malaria_catalog.predictions;
```

**Visualization:** Counter (5 cards)
- Total Predictions
- Positive Predictions
- Negative Predictions
- With Feedback
- Correct Predictions

---

#### **C) Recent Predictions Table**

**Query Name:** "Recent Predictions"
```sql
SELECT 
  patient_id,
  prediction_timestamp,
  CASE WHEN prediction = 1 THEN 'POSITIVE' ELSE 'NEGATIVE' END as predicted,
  ROUND(confidence * 100, 1) as confidence_pct,
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
```

**Visualization:** Table
- Shows last 50 predictions with results

---

#### **D) Accuracy Trend Over Time**

**Query Name:** "Accuracy Trend"
```sql
SELECT 
  DATE(prediction_timestamp) as date,
  ROUND(SUM(CASE WHEN model_correct = true THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as accuracy_pct
FROM eha.malaria_catalog.predictions
WHERE actual_result IS NOT NULL
GROUP BY DATE(prediction_timestamp)
ORDER BY date DESC
LIMIT 30;
```

**Visualization:** Line Chart
- X-axis: `date`
- Y-axis: `accuracy_pct`
- Shows how model accuracy changes over time

---

#### **E) Most Common Symptoms**

**Query Name:** "Symptom Distribution"
```sql
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
ORDER BY count DESC;
```

**Visualization:** Bar Chart
- X-axis: `symptom`
- Y-axis: `count`

---

#### **F) Feedback Progress**

**Query Name:** "Feedback Progress"
```sql
SELECT 
  COUNT(CASE WHEN actual_result IS NOT NULL THEN 1 END) as feedback_count,
  50 as target,
  ROUND(COUNT(CASE WHEN actual_result IS NOT NULL THEN 1 END) * 100.0 / 50, 2) as progress_pct
FROM eha.malaria_catalog.predictions;
```

**Visualization:** Counter + Progress Bar
- Shows how many feedback samples collected
- Progress toward 50 (retraining threshold)

---

### **Step 3: Arrange Dashboard Layout** (1 min)

Organize your dashboard:

```
┌─────────────────────────────────────────────────────────────┐
│  MALARIA CLINICAL TRIAL - MODEL MONITORING DASHBOARD        │
├─────────────┬─────────────┬─────────────┬──────────────────┤
│ Accuracy    │ Precision   │ Recall      │ F1 Score         │
│   87.5%     │   84.2%     │   89.1%     │   86.6%          │
├─────────────┴─────────────┴─────────────┴──────────────────┤
│                  Predictions Summary                         │
├──────────┬──────────┬──────────┬──────────┬────────────────┤
│ Total    │ Positive │ Negative │ Feedback │ Correct        │
│  1,234   │   456    │   778    │   892    │   778          │
├──────────┴──────────┴──────────┴──────────┴────────────────┤
│           Accuracy Trend Over Time (Line Chart)             │
│                                                              │
├──────────────────────────┬───────────────────────────────────┤
│  Most Common Symptoms    │    Feedback Progress              │
│  (Bar Chart)             │    (Progress Bar)                 │
│                          │    42/50 (84%)                    │
├──────────────────────────┴───────────────────────────────────┤
│              Recent Predictions (Table)                      │
│  Patient ID | Predicted | Actual | Correct | Confidence    │
│  ...                                                         │
└──────────────────────────────────────────────────────────────┘
```

---

## 🔄 Auto-Refresh Dashboard

Set dashboard to refresh automatically:

1. Click **Schedule** on your dashboard
2. Set refresh interval: **Every 5 minutes** (or as needed)
3. Enable **Auto-refresh**

---

## 🎯 How to Use the Dashboard

### **For Clinical Staff:**
- View real-time model accuracy
- See recent predictions
- Monitor feedback progress
- Check which symptoms are most common

### **For Data Scientists:**
- Track model performance over time
- Identify when retraining is needed
- Analyze prediction patterns
- Compare model versions

### **For Managers:**
- High-level overview of system health
- Number of patients screened
- Model reliability metrics
- System usage statistics

---

## 📱 Access Your Dashboard

**URL Pattern:**
```
https://YOUR-WORKSPACE.gcp.databricks.com/sql/dashboards/DASHBOARD-ID
```

**Share with team:**
1. Click **Share** button on dashboard
2. Set permissions (View/Edit)
3. Send link to team members

---

## 🎨 Customization Tips

1. **Add filters:**
   - Date range selector
   - Model version filter
   - Patient ID search

2. **Color coding:**
   - Green: Accuracy > 85%
   - Yellow: 70-85%
   - Red: < 70%

3. **Alerts:**
   - Set up alerts when accuracy drops below 80%
   - Notify when 50 feedback samples reached

---

## ✅ Quick Start

**Fastest way to create dashboard:**

1. Open Databricks SQL
2. Click **Dashboards** → **Create Dashboard**
3. Copy-paste queries from `dashboard_sql_queries.sql`
4. Create visualizations
5. Arrange layout
6. Share with team!

**Time:** ~10 minutes for full setup

---

## 🎉 Dashboard is Live!

Your team can now:
- ✅ Monitor model performance in real-time
- ✅ Track predictions and feedback
- ✅ See when retraining will happen
- ✅ Make data-driven decisions

**The dashboard updates automatically as new predictions come in!**
