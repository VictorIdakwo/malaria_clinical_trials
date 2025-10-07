# 🦟 Malaria Clinical Trial - Streamlit Web Application

A professional web application for malaria prediction, clinical feedback collection, and model monitoring.

---

## 🌟 Features

### 🏥 **Make Predictions**
- Interactive symptom checklist
- Real-time predictions with confidence scores
- Patient ID tracking
- Beautiful UI with color-coded results

### 📝 **Submit Feedback**
- Record actual clinical test results
- Track prediction accuracy
- Monitor progress toward model retraining
- View pending predictions

### 📊 **Dashboard**
- Real-time model performance metrics
- Prediction statistics and trends
- Recent predictions table
- Auto-refreshing data

### ⚙️ **Settings**
- Connection configuration
- Environment setup
- Connection testing

---

## 🚀 Quick Start (5 Minutes)

### **Step 1: Install Dependencies**

```bash
cd streamlit_app
pip install -r requirements.txt
```

### **Step 2: Configure Databricks Connection**

1. Copy `.env.example` to `.env`:
```bash
copy .env.example .env
```

2. Edit `.env` with your Databricks credentials:
```env
DATABRICKS_SERVER_HOSTNAME=1233387161743825.5.gcp.databricks.com
DATABRICKS_HTTP_PATH=/sql/1.0/warehouses/your-warehouse-id
DATABRICKS_TOKEN=your-access-token
```

### **Step 3: Get Databricks Credentials**

#### **A) Server Hostname**
Already provided: `1233387161743825.5.gcp.databricks.com`

#### **B) SQL Warehouse HTTP Path**
1. Go to Databricks workspace
2. Click **SQL** → **SQL Warehouses**
3. Click on your warehouse (or create one)
4. Go to **Connection Details**
5. Copy **HTTP Path** (looks like: `/sql/1.0/warehouses/abc123xyz`)

#### **C) Personal Access Token**
1. Go to Databricks workspace
2. Click your user icon → **Settings**
3. Click **Developer** → **Access Tokens**
4. Click **Generate New Token**
5. Give it a name: "Streamlit App"
6. Set lifetime: 90 days (or as needed)
7. Click **Generate**
8. **Copy the token immediately** (you won't see it again!)

### **Step 4: Run the App**

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

---

## 📖 How to Use

### **Making a Prediction**

1. **Select** "🏥 Make Prediction" from sidebar
2. **Enter** Patient ID (or leave blank for auto-generation)
3. **Check** symptoms present in the patient
4. **Click** "Get Prediction" button
5. **View** result (Positive/Negative with confidence)
6. **Note** the prediction for later feedback

### **Submitting Feedback**

1. **Select** "📝 Submit Feedback" from sidebar
2. **Copy** Prediction ID from previous prediction
3. **Enter** the Prediction ID
4. **Select** actual clinical test result
5. **Click** "Submit Feedback"
6. **See** progress toward model retraining

### **Viewing Dashboard**

1. **Select** "📊 Dashboard" from sidebar
2. **View** model performance metrics
3. **Check** prediction statistics
4. **Review** recent predictions

---

## 🌐 Deploy to Production

### **Option 1: Streamlit Community Cloud** (Free)

1. **Push code to GitHub**:
```bash
git add streamlit_app/
git commit -m "Add Streamlit app"
git push
```

2. **Deploy**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Select your repository
   - Set: `streamlit_app/app.py`
   - Add secrets (Settings → Secrets):
     ```toml
     DATABRICKS_SERVER_HOSTNAME = "1233387161743825.5.gcp.databricks.com"
     DATABRICKS_HTTP_PATH = "/sql/1.0/warehouses/xxxxx"
     DATABRICKS_TOKEN = "your-token"
     ```
   - Click "Deploy"

3. **Share URL**:
   - Get public URL: `https://your-app.streamlit.app`
   - Share with clinical staff

### **Option 2: Docker Deployment**

1. **Create Dockerfile**:
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app.py .
COPY .env .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

2. **Build and run**:
```bash
docker build -t malaria-app .
docker run -p 8501:8501 malaria-app
```

### **Option 3: Azure App Service / AWS / GCP**

Follow platform-specific deployment guides for Streamlit apps.

---

## 🔒 Security Best Practices

### **1. Never Commit Secrets**
- ✅ `.env` is in `.gitignore`
- ✅ Use environment variables
- ✅ Rotate tokens regularly

### **2. Use Service Principal** (Recommended)
Instead of personal access token, use a service principal:
1. Create service principal in Databricks
2. Grant minimal required permissions
3. Use service principal credentials

### **3. Implement Authentication**
For production, add user authentication:
```python
import streamlit_authenticator as stauth

# Add login page
authenticator = stauth.Authenticate(...)
name, authentication_status, username = authenticator.login('Login', 'main')
```

---

## 🎨 Customization

### **Branding**

Edit `app.py` to customize:

```python
st.set_page_config(
    page_title="Your Organization Name",
    page_icon="🏥",  # Change emoji
    ...
)
```

Add your logo:
```python
st.sidebar.image("path/to/your/logo.png", width=150)
```

### **Colors**

Customize CSS in the `st.markdown()` block:
```python
st.markdown("""
<style>
    .positive {
        background-color: #your-color;
    }
</style>
""", unsafe_allow_html=True)
```

### **Add Features**

- Export predictions to CSV
- Email notifications for predictions
- SMS alerts for positive cases
- Multi-language support
- Print prescription/referral forms

---

## 🐛 Troubleshooting

### **Connection Error**

```
Error: databricks.sql.exc.Error: Connection failed
```

**Fix:**
- Verify `DATABRICKS_SERVER_HOSTNAME` is correct
- Check SQL Warehouse is running
- Verify access token is valid
- Ensure network connectivity

### **Module Not Found**

```
ModuleNotFoundError: No module named 'databricks'
```

**Fix:**
```bash
pip install -r requirements.txt
```

### **Model Loading Error**

```
Error loading model: Can't get attribute 'MalariaRLModel'
```

**Fix:**
- Model class is defined in the app
- Ensure Databricks Volume path is correct
- Check model file permissions

---

## 📊 Performance Optimization

### **Caching**

The app uses `@st.cache_resource` for:
- Databricks connection (reused across sessions)
- Model loading (loaded once)

### **Database Connection Pool**

For high traffic, implement connection pooling:
```python
from databricks.sql import connection_pool

pool = connection_pool.ConnectionPool(
    max_connections=10,
    ...
)
```

---

## 🆘 Support

**Issues?** 
- Check logs: Streamlit shows errors in the app
- Review Databricks query history
- Test connection in Settings page

**Need Help?**
- Contact: your-support@email.com
- Documentation: [Streamlit Docs](https://docs.streamlit.io)
- Databricks: [SQL Connector Docs](https://docs.databricks.com/dev-tools/python-sql-connector.html)

---

## 📝 License

[Your License Here]

---

## 🎉 You're All Set!

Your Streamlit app is now ready to:
- ✅ Accept patient symptoms
- ✅ Make predictions
- ✅ Collect clinical feedback
- ✅ Monitor model performance
- ✅ Train continuously

**Deploy it and start saving lives!** 🦟💪
