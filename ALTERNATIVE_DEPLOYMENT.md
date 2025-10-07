# Alternative Deployment (Without DAB)

Since the new Databricks CLI installation is challenging, here's how to deploy **without** using Databricks Asset Bundles.

## ✅ Option 1: Deploy Using Databricks Web UI (Easiest)

### Step 1: Upload Notebooks

1. **Open your Databricks workspace**: https://1233387161743825.5.gcp.databricks.com
2. **Navigate to Workspace** → Click your email folder
3. **Create folder**: `malaria_rl_clinical_trial`
4. **Upload notebooks**:
   - Click "Import" → "File" → Select each notebook:
     - `notebooks/01_data_preparation.py`
     - `notebooks/02_train_rl_model.py`
     - `notebooks/03_clinical_dashboard.py`
     - `notebooks/04_continuous_learning.py`
     - `notebooks/05_api_service.py`

### Step 2: Upload Data

1. **Navigate to Data** → **Catalog**
2. **Use existing catalog**: `eha` (already created)
3. **Create schema**: `malaria_catalog`
   ```sql
   CREATE SCHEMA IF NOT EXISTS eha.malaria_catalog;
   ```
4. **Create volume**: `clinical_trial`
   ```sql
   CREATE VOLUME IF NOT EXISTS eha.malaria_catalog.clinical_trial;
   ```
5. **Upload CSV**:
   - Go to Volumes → `eha` → `malaria_catalog` → `clinical_trial` → `data`
   - Click "Upload" → Select `Clinical Main Data for Databricks.csv`

### Step 3: Run Notebooks in Order

1. **Run notebook 01**: `01_data_preparation.py`
   - Opens notebook → Click "Run All"
   - Wait for completion (~2-3 minutes)

2. **Run notebook 02**: `02_train_rl_model.py`
   - Opens notebook → Click "Run All"
   - Wait for model training (~5-10 minutes)

3. **Launch dashboard**: `03_clinical_dashboard.py`
   - Opens notebook
   - Install Streamlit: `%pip install streamlit`
   - Run: `!streamlit run 03_clinical_dashboard.py`

### Step 4: Create Scheduled Job (Optional)

1. **Navigate to Workflows** → **Jobs**
2. **Click "Create Job"**
3. **Configure**:
   - Name: `Malaria RL Training`
   - Task 1: Run `01_data_preparation.py`
   - Task 2: Run `02_train_rl_model.py` (depends on Task 1)
   - Schedule: Daily at 2:00 AM
   - Email: victor.idakwo@ehealthnigeria.org
4. **Save and Run**

---

## ✅ Option 2: Deploy Using VS Code Databricks Extension

### Prerequisites
- Databricks VS Code Extension installed
- Configured with your workspace (already done)

### Steps

1. **Sync notebooks to workspace**:
   - In VS Code, open Command Palette (`Ctrl+Shift+P`)
   - Type: "Databricks: Sync"
   - Select folder: `notebooks/`
   - Choose destination in workspace

2. **Upload CSV**:
   - Open Databricks explorer in VS Code
   - Navigate to Volumes
   - Right-click → Upload file

3. **Run notebooks**:
   - Right-click notebook → "Run on Databricks"

---

## ✅ Option 3: Manual CLI Upload (Old CLI)

You can still use the old CLI for basic operations:

### Upload Notebooks
```bash
cd "C:\Users\victor.idakwo\Documents\ehealth Africa\ehealth Africa\eHA GitHub\Mian Disease Modelling\Malaria\Clinical_Reinforcement_learning"

# Upload notebooks
databricks workspace import notebooks/01_data_preparation.py /Users/victor.idakwo@ehealthnigeria.org/malaria_rl/01_data_preparation.py --language PYTHON --overwrite

databricks workspace import notebooks/02_train_rl_model.py /Users/victor.idakwo@ehealthnigeria.org/malaria_rl/02_train_rl_model.py --language PYTHON --overwrite

databricks workspace import notebooks/03_clinical_dashboard.py /Users/victor.idakwo@ehealthnigeria.org/malaria_rl/03_clinical_dashboard.py --language PYTHON --overwrite

databricks workspace import notebooks/04_continuous_learning.py /Users/victor.idakwo@ehealthnigeria.org/malaria_rl/04_continuous_learning.py --language PYTHON --overwrite

databricks workspace import notebooks/05_api_service.py /Users/victor.idakwo@ehealthnigeria.org/malaria_rl/05_api_service.py --language PYTHON --overwrite
```

### Upload Data
```bash
# Create directory
databricks fs mkdirs dbfs:/Volumes/eha/malaria_catalog/clinical_trial/data/

# Upload CSV
databricks fs cp "Clinical Main Data for Databricks.csv" dbfs:/Volumes/eha/malaria_catalog/clinical_trial/data/
```

### Create Job via API
Use the old CLI to create jobs using the Jobs API 2.0.

---

## 🎯 Recommended Approach

**Use Option 1 (Web UI)** - It's the most straightforward and doesn't require any CLI setup issues.

### Quick Steps:
1. Open workspace in browser
2. Create folders and upload notebooks (5 mins)
3. Create Unity Catalog structure (2 mins)
4. Upload CSV file (1 min)
5. Run notebooks in order (15 mins)
6. Done! ✅

---

## 📝 After Deployment

Once deployed via any method:

1. **Access your dashboard**:
   - Run notebook `03_clinical_dashboard.py`
   - Dashboard will be accessible via Streamlit

2. **Make predictions**:
   - Enter patient symptoms
   - Get instant malaria predictions

3. **Submit feedback**:
   - Enter actual lab results
   - Model learns continuously

4. **Export data**:
   - Download CSV reports anytime

---

## 🔄 Install New CLI Later (Optional)

If you want DAB features later, manually download:

1. Go to: https://github.com/databricks/cli/releases
2. Download: `databricks_cli_x.x.x_windows_amd64.zip`
3. Extract to: `C:\Program Files\Databricks\`
4. Add to PATH
5. Restart terminal

But for now, **the web UI approach works perfectly!**

---

## Need Help?

- 📧 Email: victor.idakwo@ehealthnigeria.org
- 🌐 Workspace: https://1233387161743825.5.gcp.databricks.com
- 📚 Databricks Docs: https://docs.databricks.com

**Your system is ready to deploy using the Web UI!** 🚀
