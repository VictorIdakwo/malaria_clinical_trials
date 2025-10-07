# 🦟 Malaria Clinical Trial - Reinforcement Learning System

A production-ready Reinforcement Learning system for malaria prediction with continuous learning capabilities, built on Databricks.

## 📋 Overview

This system uses Reinforcement Learning (Contextual Bandits) to predict malaria cases based on patient symptoms. The model continuously learns from clinical trial feedback, improving its accuracy over time.

### Key Features

- **Reinforcement Learning**: Contextual bandit approach with epsilon-greedy exploration
- **Continuous Learning**: Model automatically updates as it receives feedback from clinical trials
- **Interactive Dashboard**: User-friendly interface for clinicians to make predictions and provide feedback
- **Data Export**: Download clinical trial data for external analysis
- **Model Versioning**: All models stored in Databricks Volumes with version tracking
- **Performance Monitoring**: Real-time tracking of model accuracy and metrics

## 🏗️ Architecture

```
Clinical_Reinforcement_learning/
├── databricks.yml                      # DAB configuration
├── requirements.txt                    # Python dependencies
├── notebooks/
│   ├── 01_data_preparation.py         # Data loading and preprocessing
│   ├── 02_train_rl_model.py          # Initial RL model training
│   ├── 03_clinical_dashboard.py      # Interactive Streamlit dashboard
│   └── 04_continuous_learning.py     # Model update pipeline
├── Clinical Main Data for Databricks.csv  # Training data
└── README.md                          # This file
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

### 4. Continuous Learning
- Scheduled job checks for new feedback
- Model retrained when sufficient feedback collected (≥50 samples)
- Only updates production model if accuracy improves (≥1%)
- Old models preserved in version history

## 📱 Dashboard Features

### 🩺 New Prediction Tab
- Enter patient ID and symptoms
- Get real-time malaria prediction
- View confidence scores and probability gauge

### 📊 Prediction History Tab
- View all predictions with timestamps
- Filter by pending/confirmed results
- Track model accuracy over time

### ✅ Enter Test Results Tab
- Submit actual laboratory results
- Provide feedback for model learning
- See if model was correct

### 📈 Model Performance Tab
- Real-time accuracy metrics
- Confusion matrix visualization
- Symptom correlation analysis
- Performance trends

### 💾 Export Data Tab
- Download predictions as CSV
- Filter by date range
- Include/exclude symptom details
- Export statistics

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

## 📝 License

This project is licensed under the MIT License.

## 👥 Authors

- eHealth Africa - Malaria Disease Modelling Team

## 📞 Support

For issues or questions:
- Create an issue in the repository
- Contact the data science team
- Review Databricks documentation

## 🎉 Acknowledgments

- Clinical staff providing feedback
- eHealth Africa for project support
- Databricks for platform capabilities

---

**Version**: 1.0.0  
**Last Updated**: 2025-10-07  
**Status**: Production Ready ✅
