# 🦟 Malaria Clinical Trial - Reinforcement Learning System

A production-ready Reinforcement Learning system for malaria prediction with continuous learning capabilities, built on Databricks.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)

---

## 🚀 Quick Deploy Streamlit App

**Deploy the professional web interface in 2 minutes:**

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click **"New app"**
4. Enter:
   - Repository: `VictorIdakwo/malaria_clinical_trials`
   - Branch: `main`
   - Main file path: `streamlit_app/app.py`
5. Add secrets (Advanced settings → Secrets):
   ```toml
   DATABRICKS_SERVER_HOSTNAME = "your-hostname.gcp.databricks.com"
   DATABRICKS_HTTP_PATH = "/sql/1.0/warehouses/your-warehouse-id"
   DATABRICKS_TOKEN = "your-access-token"
   ```
6. Click **Deploy!**

**Your app will be live in ~2 minutes!** 🎉

---

## 📋 Overview

This system uses Reinforcement Learning (Contextual Bandits) to predict malaria cases based on patient symptoms. The model continuously learns from clinical trial feedback, improving its accuracy over time.

### Key Features

- **🤖 Reinforcement Learning**: Contextual bandit approach with epsilon-greedy exploration
- **🔄 Dual-Threshold Retraining**: Fast-track (25 daily samples) OR standard (50 total samples) triggers
- **🎨 Professional Web Interface**: Modern clinical UI with gradient styling and responsive design
- **📊 Interactive Dashboards**: Multiple interfaces (Streamlit Cloud, Databricks notebook, SQL dashboards)
- **⚡ Real-time Predictions**: Instant malaria risk assessment (<2 seconds)
- **📝 Clinical Feedback Loop**: Easy feedback submission with auto-filled prediction IDs
- **🗂️ Model Versioning**: All models stored in Databricks Volumes with complete version tracking
- **📈 Performance Monitoring**: Real-time tracking of accuracy, precision, recall, and F1-score
- **🔐 Secure**: Unity Catalog permissions, encrypted data, no PHI storage

## 🏗️ Architecture

```
Clinical_Reinforcement_learning/
├── databricks.yml                      # DAB configuration
├── requirements.txt                    # Python dependencies
├── notebooks/
│   ├── 01_data_preparation.py         # Data loading and preprocessing
│   ├── 02_train_rl_model.py          # Initial RL model training
│   ├── 03_clinical_dashboard.py      # Interactive dashboard (Databricks)
│   ├── 04_continuous_learning.py     # Dual-threshold retraining pipeline
│   ├── 05_api_service.py              # REST API service
│   └── 06_simple_prediction_interface.py  # Databricks widget interface
├── streamlit_app/
│   ├── app.py                         # Professional Streamlit web app
│   ├── requirements.txt               # Web app dependencies
│   ├── .env.example                   # Configuration template
│   ├── .gitignore                     # Git ignore rules
│   ├── README.md                      # Deployment guide
│   └── eHA-logo-blue_320x132.png     # eHealth Africa logo
├── resources/
│   └── jobs.yml                       # Scheduled jobs configuration
├── dashboard_sql_queries.sql          # SQL dashboard queries
├── Clinical Main Data for Databricks.csv  # Training data (~48,000 records)
├── README.md                          # This file
├── SYSTEM_OVERVIEW.md                 # Complete system documentation
├── DEPLOY_NOW.md                      # Quick deployment guide
├── STREAMLIT_DEPLOYMENT.md            # Streamlit app deployment
└── CREATE_DASHBOARD.md                # SQL dashboard setup guide
```

## 🚀 Quick Start

### Prerequisites

- Databricks workspace (AWS, Azure, or GCP)
- Unity Catalog enabled
- Databricks CLI installed
- Python 3.9+

### Installation

1. **Clone or upload this project to your Databricks workspace**

2. **Configure Databricks CLI**
   ```bash
   databricks configure
   ```

3. **Upload the CSV data to Databricks**
   ```bash
   # Upload data to Unity Catalog volume
   databricks fs cp "Clinical Main Data for Databricks.csv" \
     dbfs:/Volumes/eha/malaria_catalog/clinical_trial/data/
   ```

4. **Deploy the DAB bundle**
   ```bash
   databricks bundle deploy --target dev
   ```

5. **Run the data preparation notebook**
   ```python
   # In Databricks, run:
   # notebooks/01_data_preparation.py
   ```

6. **Train the initial model**
   ```python
   # Run: notebooks/02_train_rl_model.py
   ```

7. **Launch the dashboard**
   ```bash
   # In Databricks notebook:
   %pip install streamlit
   !streamlit run notebooks/03_clinical_dashboard.py
   ```

## 📊 Data Schema

### Input Features (10 Symptoms)

| Feature | Type | Description |
|---------|------|-------------|
| chill_cold | Binary (0/1) | Patient experiencing chills or feeling cold |
| headache | Binary (0/1) | Patient has headache |
| fever | Binary (0/1) | Patient has fever |
| generalized_body_pain | Binary (0/1) | Patient has body aches |
| abdominal_pain | Binary (0/1) | Patient has stomach pain |
| Loss_of_appetite | Binary (0/1) | Patient has reduced appetite |
| joint_pain | Binary (0/1) | Patient has joint pain |
| vomiting | Binary (0/1) | Patient is vomiting |
| nausea | Binary (0/1) | Patient feels nauseous |
| diarrhea | Binary (0/1) | Patient has diarrhea |

### Target Variable

- **Cases**: Binary (0 = Negative, 1 = Positive for Malaria)

## 🎯 How It Works

### 1. Initial Training
- Model trained on historical clinical data (~48,000 patients)
- Gradient Boosting Classifier as base model
- Wrapped in RL framework for continuous learning

### 2. Clinical Prediction
- Clinician enters patient ID and symptoms in dashboard
- Model predicts malaria (Positive/Negative) with confidence score
- Prediction saved to database

### 3. Feedback Loop
- Patient undergoes actual laboratory test
- Clinician enters actual result in dashboard
- System calculates reward (+1 correct, -1 incorrect)
- Model learns from feedback

### 4. Continuous Learning (Dual-Threshold System)
- **Daily automated job** runs at 2:00 AM
- **Fast-track retraining**: ≥25 feedback samples in last 24 hours → Immediate retrain
- **Standard retraining**: ≥50 total feedback samples → Quality-focused retrain
- **Smart deployment**: Only updates if accuracy improves ≥1%
- **Version control**: All model versions preserved in Databricks Volumes

**Benefits:**
- 🚨 Rapid response to disease outbreaks (1-2 days)
- 📊 Quality updates during normal periods (10-15 days)
- 🎯 Adapts to clinic activity levels automatically

## 📱 User Interfaces

### 🌐 Streamlit Web App (Recommended)
**Professional clinical interface deployed on Streamlit Cloud**

#### 🏥 Make Prediction Page
- **Modern UI**: Purple gradient header, card-based design
- **Patient Info**: Auto-generated IDs, date picker
- **Symptom Assessment**: Organized primary/secondary symptoms
- **Smart Validation**: Button disabled until symptoms selected
- **Live Counter**: Shows number of symptoms selected
- **Professional Results**: Risk level, confidence, clinical recommendations
- **Prediction ID**: Prominently displayed for feedback submission

#### 📝 Submit Feedback Page
- **Auto-filled ID**: From recent prediction
- **Simple Interface**: Radio buttons for test results
- **Progress Tracking**: Shows samples until next retrain (25 or 50)
- **Progress Bar**: Visual feedback collection status
- **Recent Predictions**: Table of pending feedback

#### 📊 Dashboard Page
- **KPI Cards**: Accuracy, precision, recall, F1-score
- **Statistics**: Total predictions, positive cases, feedback count
- **Recent Data**: Last 20 predictions with actual results

#### ⚙️ Settings Page
- **Connection Test**: Verify Databricks connectivity
- **Configuration**: View catalog, schema, volume settings
- **System Info**: Current model version, database status

### 🔷 Databricks Notebook Dashboard
**Alternative interface within Databricks workspace**
- Widget-based interaction
- Direct database access
- Same prediction/feedback functionality

### 📊 SQL Dashboards
**Interactive visualizations in Databricks SQL**
- Model performance trends
- Prediction statistics
- Symptom analysis
- Confusion matrix visualization

## 🔧 Configuration

### Unity Catalog Setup

The system uses Unity Catalog with the following structure:

```
eha/                               # Catalog
└── malaria_catalog/               # Schema
    ├── malaria_training_data      # Table: Original training data
    ├── malaria_features           # Table: Feature-engineered data
    ├── predictions               # Table: Predictions and feedback
    ├── model_performance         # Table: Performance metrics
    └── clinical_trial/           # Volume: Model storage
        ├── malaria_rl_model_*.pkl
        └── model_metadata_*.json
```

### Scheduled Jobs

The DAB configuration includes a scheduled job for continuous learning:

- **Frequency**: Daily at 2:00 AM (configurable in `databricks.yml`)
- **Tasks**:
  1. Collect feedback from clinical trials
  2. Evaluate if retraining is needed
  3. Retrain model if criteria met
  4. Update production model if improvement detected

## 📈 Model Performance

### Initial Training Metrics
- Trained on ~39,000 samples
- Test accuracy: ~85-90% (varies by data distribution)
- Precision, Recall, and F1 tracked

### Continuous Learning
- Model improves with clinical feedback
- Tracks cumulative accuracy over time
- Stores all versions for comparison

## 🔐 Security & Privacy

- Patient IDs are stored but no personal information
- All data encrypted at rest in Delta tables
- Access controlled via Unity Catalog permissions
- Audit logs for all predictions and feedback

## 🛠️ Maintenance

### Monitor Model Performance
```python
# Query performance table
SELECT * FROM eha.malaria_catalog.model_performance
ORDER BY metric_timestamp DESC
LIMIT 10;
```

### Check Feedback Status
```python
# Count pending feedback
SELECT 
  COUNT(*) as total_predictions,
  SUM(CASE WHEN actual_result IS NULL THEN 1 ELSE 0 END) as pending_feedback,
  SUM(CASE WHEN model_correct = true THEN 1 ELSE 0 END) as correct_predictions
FROM eha.malaria_catalog.predictions;
```

### Manual Model Update
```bash
# Run continuous learning notebook manually
databricks jobs run-now --job-id <job_id>
```

## 📚 API Reference

### Prediction API

```python
# Load model
import pickle
with open('/Volumes/eha/malaria_catalog/clinical_trial/malaria_rl_model_latest.pkl', 'rb') as f:
    model = pickle.load(f)

# Make prediction
import pandas as pd
symptoms = pd.DataFrame([{
    'chill_cold': 0,
    'headache': 1,
    'fever': 1,
    'generalized_body_pain': 0,
    'abdominal_pain': 0,
    'Loss_of_appetite': 0,
    'joint_pain': 0,
    'vomiting': 0,
    'nausea': 0,
    'diarrhea': 0
}])

result = model.predict(None, symptoms)
print(result)
```

## 🐛 Troubleshooting

### Dashboard not loading
- Ensure Streamlit is installed: `%pip install streamlit`
- Check port availability: `8501`
- Verify model file exists in volume

### Model not updating
- Check if minimum feedback samples reached (50)
- Verify accuracy improvement threshold (1%)
- Review logs in continuous learning notebook

### Data upload issues
- Ensure volume exists: `CREATE VOLUME IF NOT EXISTS ...`
- Check file permissions
- Verify CSV format matches schema

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📚 Documentation

Complete documentation is available:

- **[SYSTEM_OVERVIEW.md](SYSTEM_OVERVIEW.md)** - Comprehensive system documentation (50+ pages)
- **[DEPLOY_NOW.md](DEPLOY_NOW.md)** - Quick deployment guide
- **[STREAMLIT_DEPLOYMENT.md](STREAMLIT_DEPLOYMENT.md)** - Streamlit Cloud deployment
- **[CREATE_DASHBOARD.md](CREATE_DASHBOARD.md)** - SQL dashboard setup
- **[streamlit_app/README.md](streamlit_app/README.md)** - Web app documentation

## 🔗 Quick Links

- **GitHub Repository**: https://github.com/VictorIdakwo/malaria_clinical_trials
- **Streamlit Cloud**: Deploy at [share.streamlit.io](https://share.streamlit.io)
- **Databricks**: Configure Unity Catalog and deploy bundle
- **Issues**: Report bugs or request features on GitHub

## 📝 License

This project is licensed under the MIT License.

## 👥 Authors

**eHealth Africa** - Malaria Disease Modelling Team
- Clinical Decision Support Systems
- Machine Learning & AI Research
- Public Health Technology

## 📞 Support

For issues or questions:
- 📧 **Email**: Contact the data science team
- 🐛 **GitHub Issues**: https://github.com/VictorIdakwo/malaria_clinical_trials/issues
- 📖 **Documentation**: Review SYSTEM_OVERVIEW.md for detailed information
- 🆘 **Databricks Support**: For platform-specific issues

## 🎉 Acknowledgments

- **Clinical staff** providing invaluable feedback for model improvement
- **eHealth Africa** for project support and resources
- **Databricks** for unified data and AI platform capabilities
- **Streamlit** for easy web app deployment
- **Open-source community** for ML libraries and tools

---

## 📊 Project Status

**Version**: 2.0.0  
**Last Updated**: October 8, 2025  
**Status**: ✅ Production Ready  
**Features**: Dual-threshold retraining, Professional UI, Auto-deployment

**Key Improvements (v2.0)**:
- ✅ Dual-threshold retraining system (fast-track + standard)
- ✅ Professional Streamlit web interface with modern design
- ✅ Auto-filled prediction IDs for seamless feedback
- ✅ Progress tracking for retraining thresholds
- ✅ Clinical recommendations based on predictions
- ✅ eHealth Africa branding integrated
- ✅ Comprehensive documentation (SYSTEM_OVERVIEW.md)

---

**⭐ Star this repository if you find it useful!**  
**🍴 Fork it to customize for your facility!**  
**🤝 Contribute to improve malaria diagnosis worldwide!**
