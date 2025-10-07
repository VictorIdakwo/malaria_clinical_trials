# 🚀 Deploy Streamlit App - Complete Guide

Deploy your professional malaria prediction web app that connects to Databricks.

---

## 📁 What Was Created

```
streamlit_app/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── .env.example          # Environment variables template
├── .gitignore            # Git ignore rules
└── README.md             # Full documentation
```

---

## ⚡ Quick Start (Local Testing)

### **1. Install Dependencies**

```bash
cd streamlit_app
pip install -r requirements.txt
```

### **2. Configure Databricks Connection**

Create `.env` file:

```bash
# Windows
copy .env.example .env

# Linux/Mac
cp .env.example .env
```

Edit `.env` with your credentials:

```env
DATABRICKS_SERVER_HOSTNAME=1233387161743825.5.gcp.databricks.com
DATABRICKS_HTTP_PATH=/sql/1.0/warehouses/xxxxxxxxx
DATABRICKS_TOKEN=dapi1234567890abcdef
```

### **3. Get Databricks Credentials**

#### **SQL Warehouse HTTP Path:**
1. Databricks → **SQL** → **SQL Warehouses**
2. Select/Create a warehouse
3. Click **Connection Details** tab
4. Copy **HTTP Path**

Example: `/sql/1.0/warehouses/abc123xyz456`

#### **Personal Access Token:**
1. Databricks → User Settings → **Developer**
2. **Access Tokens** → **Generate New Token**
3. Name: "Streamlit App"
4. Lifetime: 90 days
5. **Generate** → **Copy token immediately!**

### **4. Run Locally**

```bash
streamlit run app.py
```

Opens at: `http://localhost:8501`

---

## 🌐 Deploy to Internet (3 Options)

### **Option 1: Streamlit Community Cloud** (FREE & EASIEST)

#### **Step 1: Push to GitHub**

```bash
# From your project root
git add streamlit_app/
git commit -m "Add Streamlit web app"
git push origin main
```

#### **Step 2: Deploy on Streamlit Cloud**

1. Go to https://share.streamlit.io
2. Sign in with GitHub
3. Click **"New app"**
4. Select:
   - Repository: `your-repo`
   - Branch: `main`
   - Main file path: `streamlit_app/app.py`
5. Click **"Advanced settings"**
6. Add **Secrets** (don't put in repo!):

```toml
DATABRICKS_SERVER_HOSTNAME = "1233387161743825.5.gcp.databricks.com"
DATABRICKS_HTTP_PATH = "/sql/1.0/warehouses/xxxxx"
DATABRICKS_TOKEN = "dapi1234567890abcdef"
```

7. Click **"Deploy!"**

#### **Step 3: Get Public URL**

Your app will be live at:
```
https://your-username-malaria-app-streamlit-app-xyz.streamlit.app
```

**Share this URL with your clinical staff!**

#### **Benefits:**
- ✅ **FREE** for public repos
- ✅ Automatic updates when you push to GitHub
- ✅ SSL/HTTPS included
- ✅ No server management

---

### **Option 2: Azure App Service**

For enterprise deployment with full control.

#### **Step 1: Create Azure App Service**

```bash
# Install Azure CLI
az login

# Create resource group
az group create --name malaria-app-rg --location westeurope

# Create App Service plan
az appservice plan create \
  --name malaria-app-plan \
  --resource-group malaria-app-rg \
  --sku B1 \
  --is-linux

# Create web app
az webapp create \
  --resource-group malaria-app-rg \
  --plan malaria-app-plan \
  --name malaria-prediction-app \
  --runtime "PYTHON|3.9"
```

#### **Step 2: Configure Environment Variables**

```bash
az webapp config appsettings set \
  --resource-group malaria-app-rg \
  --name malaria-prediction-app \
  --settings \
    DATABRICKS_SERVER_HOSTNAME="1233387161743825.5.gcp.databricks.com" \
    DATABRICKS_HTTP_PATH="/sql/1.0/warehouses/xxxxx" \
    DATABRICKS_TOKEN="dapi1234567890"
```

#### **Step 3: Deploy**

```bash
# From streamlit_app directory
az webapp up \
  --resource-group malaria-app-rg \
  --name malaria-prediction-app \
  --runtime "PYTHON:3.9"
```

**URL:** `https://malaria-prediction-app.azurewebsites.net`

---

### **Option 3: Docker + Any Cloud**

Deploy to AWS, GCP, or any cloud with Docker support.

#### **Step 1: Create Dockerfile**

Create `streamlit_app/Dockerfile`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY app.py .

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

#### **Step 2: Build & Test**

```bash
cd streamlit_app

# Build image
docker build -t malaria-app .

# Run locally
docker run -p 8501:8501 \
  -e DATABRICKS_SERVER_HOSTNAME="1233387161743825.5.gcp.databricks.com" \
  -e DATABRICKS_HTTP_PATH="/sql/1.0/warehouses/xxxxx" \
  -e DATABRICKS_TOKEN="dapi1234567890" \
  malaria-app
```

#### **Step 3: Deploy to Cloud**

**AWS ECS:**
```bash
# Push to ECR
aws ecr create-repository --repository-name malaria-app
docker tag malaria-app:latest 123456789.dkr.ecr.us-east-1.amazonaws.com/malaria-app
docker push 123456789.dkr.ecr.us-east-1.amazonaws.com/malaria-app

# Deploy to ECS (use AWS Console or Fargate)
```

**GCP Cloud Run:**
```bash
# Push to GCR
gcloud builds submit --tag gcr.io/PROJECT_ID/malaria-app
gcloud run deploy malaria-app --image gcr.io/PROJECT_ID/malaria-app --platform managed
```

---

## 🔒 Production Security

### **1. Use Service Principal (Not Personal Token)**

Create a service principal in Databricks:

```sql
-- In Databricks SQL
CREATE SERVICE PRINCIPAL 'malaria-app-sp';

-- Grant permissions
GRANT SELECT ON TABLE eha.malaria_catalog.predictions TO `malaria-app-sp`;
GRANT SELECT ON TABLE eha.malaria_catalog.model_performance TO `malaria-app-sp`;
GRANT INSERT ON TABLE eha.malaria_catalog.predictions TO `malaria-app-sp`;
```

Then use service principal credentials in your app.

### **2. Add Authentication**

Install `streamlit-authenticator`:

```bash
pip install streamlit-authenticator
```

Add to `app.py`:

```python
import streamlit_authenticator as stauth

# Load user credentials (from secure storage)
authenticator = stauth.Authenticate(
    credentials,
    'malaria_app',
    'auth_key',
    cookie_expiry_days=30
)

name, authentication_status, username = authenticator.login('Login', 'main')

if authentication_status:
    # Show app
    st.write(f'Welcome {name}')
elif authentication_status == False:
    st.error('Username/password is incorrect')
else:
    st.warning('Please enter username and password')
```

### **3. Enable HTTPS**

All deployment options (Streamlit Cloud, Azure, Docker) support HTTPS by default.

### **4. Implement Rate Limiting**

Prevent abuse:

```python
from streamlit_extras.app_limiter import limit_app

limit_app(max_requests=100, period="1h")
```

---

## 📊 Monitoring & Logging

### **Add Logging to App**

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Log predictions
logger.info(f"Prediction made for patient {patient_id}: {prediction}")
```

### **Monitor with Streamlit Cloud**

- View logs in Streamlit Cloud dashboard
- Set up email alerts
- Monitor app health

### **Custom Monitoring**

Add Application Insights (Azure) or Cloud Monitoring (GCP):

```python
from opencensus.ext.azure.log_exporter import AzureLogHandler

logger.addHandler(AzureLogHandler(connection_string='your-connection-string'))
```

---

## 🎨 Customization

### **1. Add Your Organization Logo**

```python
# In sidebar
st.sidebar.image("https://your-org.com/logo.png", width=150)
```

### **2. Custom Theme**

Create `.streamlit/config.toml`:

```toml
[theme]
primaryColor="#1f77b4"
backgroundColor="#ffffff"
secondaryBackgroundColor="#f0f2f6"
textColor="#262730"
font="sans serif"
```

### **3. Add More Features**

Ideas:
- Export predictions to PDF
- Send SMS notifications
- Integration with EMR systems
- Multi-language support (i18n)
- Voice input for symptoms
- Barcode scanning for patient IDs

---

## 🐛 Troubleshooting

### **App Won't Start**

```
Error: cannot import name 'sql' from 'databricks'
```

**Fix:** Install correct package
```bash
pip install databricks-sql-connector
```

### **Connection Refused**

```
Error: Connection refused at databricks.sql.connect()
```

**Fix:** 
- SQL Warehouse must be running
- Check firewall rules
- Verify credentials

### **Slow Performance**

**Solutions:**
- Use `@st.cache_data` for queries
- Implement connection pooling
- Upgrade SQL Warehouse size

---

## 📱 Mobile Optimization

The app is mobile-responsive by default. Test on:
- iOS Safari
- Android Chrome
- Tablets

**Tips:**
- Keep forms simple
- Use large buttons
- Minimize scrolling
- Test on actual devices

---

## 🎓 Training Clinical Staff

### **Quick User Guide**

1. **Access app:** Open browser → Go to app URL
2. **Make prediction:**
   - Enter patient ID
   - Check symptoms
   - Click "Get Prediction"
3. **Submit feedback:**
   - Copy Prediction ID
   - Go to "Submit Feedback"
   - Enter actual test result
4. **View dashboard:**
   - Check model performance
   - Review recent cases

### **Training Materials**

Create:
- Video tutorial (5 min)
- PDF quick reference card
- In-app help tooltips
- Support contact info

---

## 📈 Analytics

Track usage:

```python
# Add Google Analytics
st.components.v1.html("""
<script async src="https://www.googletagmanager.com/gtag/js?id=G-XXXXXXXXXX"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', 'G-XXXXXXXXXX');
</script>
""")
```

---

## ✅ Deployment Checklist

Before going live:

- [ ] Test all features locally
- [ ] Configure production credentials
- [ ] Enable HTTPS
- [ ] Add authentication
- [ ] Set up monitoring
- [ ] Create user documentation
- [ ] Train clinical staff
- [ ] Test on mobile devices
- [ ] Set up backup strategy
- [ ] Configure auto-scaling (if needed)
- [ ] Create incident response plan

---

## 🎉 You're Ready to Deploy!

**Recommended for you:** Streamlit Community Cloud (FREE & Easy)

**Time to deploy:** 10 minutes

**URL to share:** `https://your-app.streamlit.app`

**Start saving lives with AI!** 🦟💪🏥
