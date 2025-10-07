# 🚀 Quick Start Guide

Get your Malaria RL Clinical Trial system up and running in minutes!

## Prerequisites Checklist

- [ ] Databricks workspace (AWS, Azure, or GCP)
- [ ] Unity Catalog enabled
- [ ] Databricks CLI installed (`pip install databricks-cli`)
- [ ] Python 3.9+
- [ ] Access to the clinical data CSV file

## 5-Minute Setup

### Step 1: Configure Databricks CLI (2 min)

```bash
# Install Databricks CLI
pip install databricks-cli

# Configure with your workspace
databricks configure --token

# Enter your Databricks workspace URL and personal access token when prompted
```

### Step 2: Run Setup Script (1 min)

```bash
# Navigate to project directory
cd Clinical_Reinforcement_learning

# Run setup
python setup.py
```

### Step 3: Deploy to Databricks (2 min)

**Option A: Automated Deployment (Linux/Mac)**
```bash
chmod +x deploy.sh
./deploy.sh dev
```

**Option B: Manual Deployment (Windows/All)**
```bash
# Validate bundle
databricks bundle validate --target dev

# Deploy
databricks bundle deploy --target dev
```

## Running the System

### 1️⃣ Upload Your Data

In Databricks workspace:

```python
# Create a notebook and run:
dbutils.fs.cp(
    "file:///path/to/Clinical Main Data for Databricks.csv",
    "dbfs:/Volumes/eha/malaria_catalog/clinical_trial/data/Clinical Main Data for Databricks.csv"
)
```

Or use Databricks UI:
- Navigate to Data → Volumes
- Select `eha` → `malaria_catalog` → `clinical_trial` → `data`
- Click "Upload" and select your CSV file

### 2️⃣ Initialize Database

Open and run: `notebooks/01_data_preparation.py`

This creates:
- ✅ Unity Catalog tables
- ✅ Feature engineering pipeline
- ✅ Prediction and feedback tables

**Expected output:**
```
Training data saved to eha.malaria_catalog.malaria_training_data
Predictions table created: eha.malaria_catalog.predictions
✅ Data preparation completed successfully!
```

### 3️⃣ Train Initial Model

Open and run: `notebooks/02_train_rl_model.py`

This will:
- ✅ Train the RL model
- ✅ Save to MLflow
- ✅ Store in Unity Catalog Volume
- ✅ Log performance metrics

**Expected output:**
```
Model Performance:
  Accuracy:  0.8542
  Precision: 0.8123
  Recall:    0.8765
  F1 Score:  0.8435

✅ Model saved to: /Volumes/eha/malaria_catalog/clinical_trial/malaria_rl_model_20231007_123456.pkl
```

### 4️⃣ Launch Dashboard

Open `notebooks/03_clinical_dashboard.py` and run:

```python
# Cell 1: Install Streamlit
%pip install streamlit

# Cell 2: Run dashboard
!streamlit run notebooks/03_clinical_dashboard.py --server.port 8501
```

Access the dashboard at the provided URL.

## Using the Dashboard

### Make Your First Prediction

1. Go to **"New Prediction"** tab
2. Enter Patient ID (e.g., `TEST001`)
3. Check symptoms (e.g., fever, headache)
4. Click **"Generate Prediction"**
5. View result and confidence score

### Submit Test Results

1. Go to **"Enter Test Results"** tab
2. Select the pending prediction
3. Enter actual lab result (Positive/Negative)
4. Click **"Submit Result"**
5. Model learns from feedback!

### Monitor Performance

1. Go to **"Model Performance"** tab
2. View real-time accuracy
3. Check confusion matrix
4. Analyze symptom correlations

### Export Data

1. Go to **"Export Data"** tab
2. Select date range and options
3. Click **"Generate Export"**
4. Download CSV file

## Scheduling Continuous Learning

The system automatically retrains daily at 2:00 AM. To modify:

Edit `databricks.yml`:
```yaml
schedule:
  quartz_cron_expression: "0 0 2 * * ?"  # Change this
  timezone_id: "Africa/Lagos"
```

## API Access (Optional)

For programmatic access:

1. Run `notebooks/05_api_service.py`
2. API available at `http://<workspace>:5000`

**Example prediction:**
```bash
curl -X POST http://localhost:5000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "patient_id": "LT123456",
    "symptoms": {
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
  }'
```

## Troubleshooting

### Issue: "Model not found"
**Solution:** Run notebook `02_train_rl_model.py` to create initial model

### Issue: "Table not found"
**Solution:** Run notebook `01_data_preparation.py` to create tables

### Issue: "Data file not found"
**Solution:** Upload CSV to `/Volumes/eha/malaria_catalog/clinical_trial/data/`

### Issue: Dashboard won't load
**Solution:** 
```python
%pip install --upgrade streamlit plotly
dbutils.library.restartPython()
```

### Issue: Permission denied
**Solution:** Check Unity Catalog permissions:
```sql
GRANT ALL PRIVILEGES ON CATALOG eha TO `<your-user>`;
GRANT ALL PRIVILEGES ON SCHEMA eha.malaria_catalog TO `<your-user>`;
```

## Verification Checklist

After setup, verify everything works:

- [ ] Tables created in Unity Catalog
- [ ] Model saved in Volume
- [ ] Can make predictions in dashboard
- [ ] Can submit feedback
- [ ] Can export data
- [ ] Scheduled job appears in Workflows
- [ ] MLflow experiment visible

## Next Steps

1. **Train staff** on using the dashboard
2. **Start clinical trials** and collect feedback
3. **Monitor performance** weekly
4. **Review model improvements** monthly
5. **Scale up** as you validate effectiveness

## Support

- 📚 Full documentation: `README.md`
- 🔧 Configuration: `databricks.yml`
- 💬 Questions: Create an issue or contact your team

---

**Ready to save lives! 🦟💊**

*System deployed and operational in under 5 minutes!*
