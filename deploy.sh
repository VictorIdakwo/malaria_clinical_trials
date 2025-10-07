#!/bin/bash

# Deployment script for Malaria RL Clinical Trial System
# This script automates the deployment process to Databricks

set -e  # Exit on error

echo "======================================================================"
echo "  🦟 Malaria RL Clinical Trial System - Deployment"
echo "======================================================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
CATALOG="malaria_catalog"
SCHEMA="clinical_trial"
VOLUME="ml_models"
TARGET="${1:-dev}"  # Default to dev if no target specified

echo "📋 Deployment Configuration:"
echo "   Target: $TARGET"
echo "   Catalog: $CATALOG"
echo "   Schema: $SCHEMA"
echo ""

# Check if databricks CLI is installed
if ! command -v databricks &> /dev/null; then
    echo -e "${RED}❌ Databricks CLI not found${NC}"
    echo "   Install: pip install databricks-cli"
    exit 1
fi

echo -e "${GREEN}✅ Databricks CLI found${NC}"

# Validate DAB bundle
echo ""
echo "📋 Validating DAB bundle..."
if databricks bundle validate --target "$TARGET"; then
    echo -e "${GREEN}✅ Bundle validation passed${NC}"
else
    echo -e "${RED}❌ Bundle validation failed${NC}"
    exit 1
fi

# Deploy bundle
echo ""
echo "🚀 Deploying bundle to $TARGET..."
if databricks bundle deploy --target "$TARGET"; then
    echo -e "${GREEN}✅ Bundle deployed successfully${NC}"
else
    echo -e "${RED}❌ Bundle deployment failed${NC}"
    exit 1
fi

# Upload CSV data (if not already uploaded)
echo ""
echo "📤 Checking data upload..."
DATA_FILE="Clinical Main Data for Databricks.csv"
if [ -f "$DATA_FILE" ]; then
    echo "   Found data file: $DATA_FILE"
    echo "   Uploading to Databricks..."
    
    # Note: This requires DBFS or Unity Catalog volumes to be set up
    databricks fs cp "$DATA_FILE" \
        "dbfs:/Volumes/$CATALOG/$SCHEMA/data/$DATA_FILE" \
        --overwrite || echo -e "${YELLOW}⚠️  Data upload failed or already exists${NC}"
else
    echo -e "${YELLOW}⚠️  Data file not found: $DATA_FILE${NC}"
    echo "   Please upload manually to /Volumes/$CATALOG/$SCHEMA/data/"
fi

# Run data preparation (optional - can be run manually)
echo ""
read -p "🤔 Run data preparation notebook now? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "📊 Running data preparation..."
    databricks jobs run-now --job-name "Malaria RL Model Training" \
        --notebook-params '{"task_key":"data_preparation"}' || \
        echo -e "${YELLOW}⚠️  Job execution failed. Run manually in Databricks.${NC}"
else
    echo "⏩ Skipping data preparation. Run manually when ready."
fi

# Summary
echo ""
echo "======================================================================"
echo -e "  ${GREEN}✅ Deployment Complete!${NC}"
echo "======================================================================"
echo ""
echo "📝 Next Steps:"
echo ""
echo "1. Verify Unity Catalog setup:"
echo "   - Catalog: $CATALOG"
echo "   - Schema: $SCHEMA"
echo "   - Volume: $VOLUME"
echo ""
echo "2. Run notebooks in order:"
echo "   a. notebooks/01_data_preparation.py"
echo "   b. notebooks/02_train_rl_model.py"
echo "   c. notebooks/03_clinical_dashboard.py (for dashboard)"
echo ""
echo "3. Access the dashboard:"
echo "   - Run notebook 03_clinical_dashboard.py with Streamlit"
echo ""
echo "4. Monitor the scheduled job:"
echo "   - Job Name: Malaria RL Model Training"
echo "   - Schedule: Daily at 2:00 AM"
echo ""
echo "📚 For more information, see README.md"
echo ""
