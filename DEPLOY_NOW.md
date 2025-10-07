# 🚀 Deploy NOW - Simple 10-Minute Setup

## Your Situation
- ❌ Old CLI doesn't support bundles
- ❌ Bundle Resource Explorer is empty
- ✅ **Solution**: Use Databricks Web UI (no CLI needed!)

---

## 📋 Step-by-Step Deployment (10 minutes)

### 1️⃣ Open Your Workspace (1 min)
**URL**: https://1233387161743825.5.gcp.databricks.com

### 2️⃣ Unity Catalog Structure (Already Created ✅)

Your existing catalog structure:
- **Catalog**: `eha`
- **Schema**: `malaria_catalog`
- **Volume**: `clinical_trial`
- **Data Path**: `/Volumes/eha/malaria_catalog/clinical_trial/data/`

No need to create - you're using the existing `eha` catalog.

### 3️⃣ Upload Your Data (2 min)

1. Navigate to: **Data** → **eha** → **malaria_catalog** → **clinical_trial** → **data** (folder)
2. Click **"Upload files"** button
3. Select: `Clinical Main Data for Databricks.csv`
4. Wait for upload to complete ✅

### 4️⃣ Upload Notebooks (3 min)

1. Click **"Workspace"** in left sidebar
2. Navigate to your user folder: `victor.idakwo@ehealthnigeria.org`
3. Right-click → **"Create"** → **"Folder"**
   - Name: `malaria_rl_clinical_trial`

4. Click into the folder → Click **"Import"**

5. Select **"File"** and import each notebook:
   - Browse to: `notebooks/01_data_preparation.py` → **Import**
   - Browse to: `notebooks/02_train_rl_model.py` → **Import**
   - Browse to: `notebooks/03_clinical_dashboard.py` → **Import**
   - Browse to: `notebooks/04_continuous_learning.py` → **Import**
   - Browse to: `notebooks/05_api_service.py` → **Import**

### 5️⃣ Run Initial Setup (2 min)

1. Open notebook: `01_data_preparation.py`
2. Attach to cluster (create one if needed):
   - **Cluster Mode**: Single Node
   - **Databricks Runtime**: 13.3 LTS ML
   - **Node Type**: Standard_DS3_v2 or similar
3. Click **"Run All"** ▶️
4. Wait for completion (~2-3 minutes)

---

## ✅ Initial Training (10 minutes)

1. Open notebook: `02_train_rl_model.py`
2. Same cluster as above
3. Click **"Run All"** ▶️
4. Wait for training (~5-10 minutes)
5. Check output for:
   ```
   Model Performance:
     Accuracy:  0.8xxx
     ...
   ✅ Model saved to: /Volumes/malaria_catalog/...
   ```

---

## 🎉 Launch Dashboard (5 minutes)

1. Open notebook: `03_clinical_dashboard.py`
2. Run first cell to install Streamlit:
   ```python
   %pip install streamlit plotly
   ```
3. Restart Python:
   ```python
   dbutils.library.restartPython()
   ```
4. Run the dashboard cells

**Note**: For full Streamlit experience, you may need to set up Databricks Apps or use the API approach.

---

## 📊 Alternative: Use Databricks Jobs UI

Instead of Streamlit dashboard, create a simple workflow:

### Create Prediction Job

1. Go to **"Workflows"** → **"Jobs"**
2. Click **"Create Job"**
3. Configure:
   - **Name**: Malaria Predictions Daily
   - **Task**: Run `02_train_rl_model.py`
   - **Schedule**: Daily at 2:00 AM
   - **Cluster**: Single Node, Runtime 13.3 ML
   - **Email**: victor.idakwo@ehealthnigeria.org
4. **Save**

---

## 🔍 Verify Everything Works

### Check Tables Created
```sql
-- Run in SQL Editor or notebook
SHOW TABLES IN eha.malaria_catalog;
```

Expected output:
- `malaria_training_data`
- `malaria_features`
- `predictions`
- `model_performance`

### Check Model Saved
```python
# Run in notebook
dbutils.fs.ls("/Volumes/eha/malaria_catalog/clinical_trial/")
```

Expected: Model files like `malaria_rl_model_20231007_*.pkl`

### Check Data Loaded
```sql
SELECT COUNT(*) FROM eha.malaria_catalog.malaria_training_data;
```

Expected: ~48,940 rows

---

## 🎯 Making Predictions (Simple Version)

Since the full dashboard needs the new CLI, here's a simple prediction notebook:

### Create: `predict_patient.py`

```python
# Databricks notebook source
import pickle
import pandas as pd

# Load model
model_files = dbutils.fs.ls("/Volumes/eha/malaria_catalog/clinical_trial/")
latest_model = sorted([m.path for m in model_files if m.name.endswith('.pkl')])[-1]

with open(latest_model.replace("/Volumes/", "/dbfs/Volumes/"), 'rb') as f:
    model = pickle.load(f)

# Enter patient symptoms
symptoms = {
    "chill_cold": 0,
    "headache": 1,
    "fever": 1,
    "generalized_body_pain": 1,
    "abdominal_pain": 0,
    "Loss_of_appetite": 0,
    "joint_pain": 0,
    "vomiting": 0,
    "nausea": 0,
    "diarrhea": 0
}

# Predict
input_df = pd.DataFrame([symptoms])
result = model.predict(None, input_df)

print(f"Prediction: {'POSITIVE' if result['prediction'][0] == 1 else 'NEGATIVE'}")
print(f"Confidence: {result['probability_positive'][0]*100:.1f}%")
```

---

## 📝 Summary

| Step | Status | Time |
|------|--------|------|
| Create Unity Catalog | ✅ Ready | 2 min |
| Upload Data | ✅ Ready | 2 min |
| Upload Notebooks | ✅ Ready | 3 min |
| Run Data Prep | Next | 2 min |
| Train Model | Next | 10 min |
| Make Predictions | Next | 5 min |

**Total Time**: ~25 minutes

---

## 🆘 If You Get Stuck

1. **Cluster issues**: Use Single Node, Runtime 13.3 ML
2. **Import errors**: Make sure you select `.py` files
3. **Volume not found**: Create volumes in Unity Catalog first
4. **Path errors**: Update notebook paths to match your structure

---

## 🎉 You're Ready!

Open your workspace and follow the steps above. Everything will work without needing the new CLI or DAB!

**Workspace**: https://1233387161743825.5.gcp.databricks.com

**Next**: Open your workspace and start with Step 1! 🚀
