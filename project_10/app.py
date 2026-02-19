from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
import io
import base64
import matplotlib.pyplot as plt
import joblib
import os
import sys

# Add src to path if needed (though local import should work)
# sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_preprocessing import (
    preprocess_data,
    verify_recovery_calculation,
    analyze_metal_concentrations,
    compare_feed_distributions,
    remove_anomalies,
    COMMON_COLUMNS_SANS_DATE
)

app = FastAPI()

# HTML Template
# templates = Jinja2Templates(directory="templates") # Not used, returning f-strings directly

# Helper to convert matplotlib fig to base64
def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig) # Clear memory
    return img_str

@app.get("/", response_class=HTMLResponse)
async def read_root():
    return """
    <html>
        <head>
            <title>Gold Recovery Dashboard</title>
            <style>
                body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
                h1 { color: #333; }
                .form-group { margin-bottom: 15px; }
                label { display: block; margin-bottom: 5px; font-weight: bold; }
                input[type="submit"] { background-color: #007bff; color: white; border: none; padding: 10px 20px; cursor: pointer; }
                input[type="submit"]:hover { background-color: #0056b3; }
            </style>
        </head>
        <body>
            <h1>Upload Data for Analysis</h1>
            <form action="/analyze" method="post" enctype="multipart/form-data">
                <div class="form-group">
                    <label>Full Dataset (gold_recovery_full.csv):</label>
                    <input type="file" name="file_full" required>
                </div>
                <div class="form-group">
                    <label>Train Dataset (gold_recovery_train.csv):</label>
                    <input type="file" name="file_train" required>
                </div>
                <div class="form-group">
                    <label>Test Dataset (gold_recovery_test.csv):</label>
                    <input type="file" name="file_test" required>
                </div>
                <input type="submit" value="Analyze">
            </form>
        </body>
    </html>
    """

@app.post("/analyze", response_class=HTMLResponse)
async def analyze(
    file_full: UploadFile = File(...),
    file_train: UploadFile = File(...),
    file_test: UploadFile = File(...)
):
    # 1. Load Data
    full_df = pd.read_csv(file_full.file)
    train_df = pd.read_csv(file_train.file)
    test_df = pd.read_csv(file_test.file)
    
    # 2. Preprocess Data
    full_df, train_df, test_df = preprocess_data(full_df, train_df, test_df)
    
    # Data check output capture (simple list)
    logs = []
    logs.append(f"Loaded {len(full_df)} rows for full dataset.")
    logs.append(f"Loaded {len(train_df)} rows for training dataset.")
    logs.append(f"Loaded {len(test_df)} rows for testing dataset.")

    # 3. Verify Recovery Calculation
    logs.append("Verifying Recovery Calculation...")
    fig_recovery = verify_recovery_calculation(train_df, show_plot=False)
    img_recovery = fig_recovery if fig_recovery else None
    if img_recovery:
        img_recovery_b64 = fig_to_base64(img_recovery)
    
    # 4. Analyze Metal Concentrations
    logs.append("Analyzing Metal Concentrations...")
    fig_metal = analyze_metal_concentrations(full_df, show_plot=False)
    img_metal_b64 = fig_to_base64(fig_metal) if fig_metal else None

    # 5. Remove Anomalies
    logs.append("Removing Anomalies...")
    # Just running it to get clean features, though typically we'd use the provided test set strictly
    train_df_clean, test_df_clean, common_columns = remove_anomalies(train_df, test_df, full_df)
    logs.append(f"Cleaned Train Rows: {len(train_df_clean)}")
    logs.append(f"Cleaned Test Rows: {len(test_df_clean)}")

    # 6. Apply Existing Models
    logs.append("Applying Existing Models...")
    try:
        scaler = joblib.load('models/scaler.pkl')
        model_rougher = joblib.load('models/model_rougher.pkl')
        model_final = joblib.load('models/model_final.pkl')
        
        # Prepare features
        # Assuming test set has the required input features
        features_test = test_df_clean[common_columns].copy() # Ensure copy
        
        # Scale
        features_scaled = scaler.transform(features_test)
        
        # Predict
        predictions_rougher = model_rougher.predict(features_scaled)
        predictions_final = model_final.predict(features_scaled)
        
        # Add to dataframe for display
        test_df_clean['predicted_rougher_recovery'] = predictions_rougher
        test_df_clean['predicted_final_recovery'] = predictions_final
        
        preview_html = test_df_clean[['date', 'predicted_rougher_recovery', 'predicted_final_recovery']].head(10).to_html(classes='table table-striped')
        
    except Exception as e:
        logs.append(f"Error loading models or predicting: {str(e)}")
        preview_html = "<p>Prediction failed.</p>"

    # Render HTML
    return f"""
    <html>
        <head>
            <title>Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; max-width: 1000px; margin: 0 auto; padding: 20px; }}
                h2 {{ color: #333; border-bottom: 2px solid #eee; padding-bottom: 10px; }}
                img {{ max-width: 100%; height: auto; margin: 20px 0; border: 1px solid #ddd; }}
                .log {{ background: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                .table {{ width: 100%; border-collapse: collapse; }}
                .table th, .table td {{ padding: 8px; border: 1px solid #ddd; text-align: left; }}
                .table th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>Analysis Report</h1>
            <div class="log">
                <h3>Process Log</h3>
                <ul>
                    {"".join(f"<li>{log}</li>" for log in logs)}
                </ul>
            </div>
            
            <h2>Recovery Verification</h2>
            {f'<img src="data:image/png;base64,{img_recovery_b64}">' if img_recovery else '<p>No plot generated.</p>'}
            
            <h2>Metal Concentrations</h2>
            {f'<img src="data:image/png;base64,{img_metal_b64}">' if img_metal_b64 else '<p>No plot generated.</p>'}
            
            <h2>Predictions (Preview)</h2>
            {preview_html}
            
            <p><a href="/">Upload New Data</a></p>
        </body>
    </html>
    """
