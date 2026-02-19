
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from jinja2 import Template
import pandas as pd
import io
import os
import secrets
from pathlib import Path
import shutil

# Custom modules
from src.analysis import train_and_predict, bootstrap_profit, analyze_region_profitability
from src.utils import check_duplicate
from src.visualization import generate_profit_distribution_plot, generate_risk_loss_plot
from src.data_preprocessing import load_data, preprocess_data

# Define FastAPI App
app = FastAPI()

# Configuration
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# Templates
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Oil Well Profitability Analysis</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 40px; background-color: #f9f9f9; color: #333; }
        h1, h2, h3 { color: #2c3e50; }
        .container { max-width: 1000px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .success { color: #27ae60; font-weight: bold; }
        .error { color: #c0392b; background: #fadbd8; padding: 10px; border-radius: 4px; border: 1px solid #e6b0aa; }
        table { border-collapse: collapse; width: 100%; margin-top: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
        th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
        th { background-color: #34495e; color: white; }
        tr:nth-child(even) { background-color: #f2f2f2; }
        img { max-width: 100%; height: auto; margin-top: 20px; border: 1px solid #ddd; border-radius: 4px; }
        .plots { display: flex; flex-wrap: wrap; gap: 20px; margin-top: 30px; }
        .plot-container { flex: 1; min-width: 450px; background: white; padding: 15px; border-radius: 8px; border: 1px solid #eee; }
        .form-group { margin-bottom: 20px; padding: 20px; background: #eef2f3; border-radius: 6px; }
        button { background-color: #27ae60 !important; color: white; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer; font-size: 16px; visibility: visible; opacity: 1; }
        button:hover { background-color: #219150 !important; }
        input[type="file"] { margin-bottom: 10px; }
        th { cursor: pointer; }
        th:hover { background-color: #2c3e50; }
    </style>
    <script>
    function sortTable(n) {
      var table, rows, switching, i, x, y, shouldSwitch, dir, switchcount = 0;
      table = document.querySelector("table");
      switching = true;
      dir = "asc"; 
      while (switching) {
        switching = false;
        rows = table.rows;
        for (i = 1; i < (rows.length - 1); i++) {
          shouldSwitch = false;
          x = rows[i].getElementsByTagName("TD")[n];
          y = rows[i + 1].getElementsByTagName("TD")[n];
          var xContent = x.innerHTML.toLowerCase().replace(/[$,%]/g, "");
          var yContent = y.innerHTML.toLowerCase().replace(/[$,%]/g, "");
          if (!isNaN(parseFloat(xContent)) && !isNaN(parseFloat(yContent))) {
              xContent = parseFloat(xContent);
              yContent = parseFloat(yContent);
          }
          if (dir == "asc") {
            if (xContent > yContent) {
              shouldSwitch = true;
              break;
            }
          } else if (dir == "desc") {
            if (xContent < yContent) {
              shouldSwitch = true;
              break;
            }
          }
        }
        if (shouldSwitch) {
          rows[i].parentNode.insertBefore(rows[i + 1], rows[i]);
          switching = true;
          switchcount ++;      
        } else {
          if (switchcount == 0 && dir == "asc") {
            dir = "desc";
            switching = true;
          }
        }
      }
    }
    </script>
</head>
<body>
    <div class="container">
        <h1>Oil Well Profitability Analysis Dashboard</h1>
        
        <div class="form-group">
            <h2>Upload New Geographical Data</h2>
            <form action="/analyze" method="post" enctype="multipart/form-data">
                <label for="file"><strong>Select CSV File:</strong></label><br>
                <input type="file" id="file" name="file" accept=".csv" required><br><br>
                <button type="submit">Run Analysis</button>
            </form>
        </div>
        
        {% if error %}
            <div class="error">
                <strong>Error:</strong> {{ error }}
            </div>
        {% endif %}
        
        {% if results %}
            <hr>
            <h2>Analysis Results</h2>
            <p><strong>Status:</strong> New data processed successfully and compared with existing regions.</p>
            
            <h3>Comparison Summary (Sorted by Profit/Risk Ratio)</h3>
            
            <table>
                <thead>
                    <tr>
                        <th onclick="sortTable(0)">Region (File Name)</th>
                        <th onclick="sortTable(1)">Mean Profit ($)</th>
                        <th onclick="sortTable(2)">Risk of Loss (%)</th>
                        <th>95% CI Lower ($)</th>
                        <th>95% CI Upper ($)</th>
                        <th onclick="sortTable(5)">Profit ($M)/Risk Ratio</th>
                    </tr>
                </thead>
                <tbody>
                    {% for row in summary %}
                    <tr>
                        <td><strong>{{ row.region }}</strong></td>
                        <td>{{ "{:,.2f}".format(row.mean_profit) }}</td>
                        <td style="color: {{ 'red' if row.risk_of_loss_percent > 2.5 else 'green' }}">{{ "{:.2f}".format(row.risk_of_loss_percent) }}%</td>
                        <td>{{ "{:,.2f}".format(row.ci_lower) }}</td>
                        <td>{{ "{:,.2f}".format(row.ci_upper) }}</td>
                        <td>
                            <strong>{{ "{:.2f}".format(row.profit_risk_ratio) }}</strong>
                        </td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
            
            <div class="plots">
                <div class="plot-container">
                    <h3 style="text-align: center;">Profit Distribution</h3>
                    <img src="data:image/png;base64,{{ profit_plot }}" alt="Profit Distribution Plot">
                    <p style="font-size: 0.9em; color: #666;">Distribution of bootstrapped profit samples. Higher peaks indicate more likely profit outcomes.</p>
                </div>
                <div class="plot-container">
                    <h3 style="text-align: center;">Risk of Loss Analysis</h3>
                    <img src="data:image/png;base64,{{ risk_plot }}" alt="Risk Analysis Plot">
                    <p style="font-size: 0.9em; color: #666;">Percentage of bootstrap samples resulting in a loss. Threshold is 2.5%.</p>
                </div>
            </div>
            
        {% endif %}
    </div>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def main_page():
    return Template(HTML_TEMPLATE).render()

@app.post("/analyze", response_class=HTMLResponse)
async def analyze_data(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        return Template(HTML_TEMPLATE).render(error="Invalid file type. Please upload a CSV file.")

    # 1. Read uploaded file content
    content = await file.read()
    
    # 2. Check for Duplicates
    # Convert data dir to str for our util function
    is_duplicate, existing_filename = check_duplicate(content, str(DATA_DIR))
    
    if is_duplicate:
        return Template(HTML_TEMPLATE).render(
            error=f"Duplicate Data Detected! The uploaded file matches existing data: '{existing_filename}'. Analysis rejected."
        )
    
    # 3. Save new file. preserve original filename if possible, but safe against overwrite.
    # We want to display the filename in the table. 
    # Let's clean the filename but keep it recognizable.
    safe_filename = Path(file.filename).name
    # Prepend 'uploaded_' to avoid collision with system files if any, or just use a dedicated folder?
    # Actually, let's just make sure we don't overwrite source data.
    # The user wants to see "geo_data_3" if they uploaded "geo_data_3.csv".
    
    # We will save it as is if it doesn't exist, or append suffix if it does.
    new_filepath = DATA_DIR / safe_filename
    if new_filepath.exists():
        stem = new_filepath.stem
        suffix = new_filepath.suffix
        new_filepath = DATA_DIR / f"{stem}_{secrets.token_hex(4)}{suffix}"
    
    try:
        with open(new_filepath, "wb") as f:
            f.write(content)
        
        # 4. Run Analysis
        # Load ALL data including the new one
        # Use file stem (e.g., 'geo_data_0') as the region name
        data_paths = {f.name: str(f) for f in DATA_DIR.glob("*.csv")}
        
        # Helper functions from src
        raw_dfs = load_data(data_paths)
        dfs = preprocess_data(raw_dfs)
        
        results_summary = []
        profits_dict = {}
        regions_risk = {}
        regions_risk_raw = {}
        
        # Constants
        BUDGET = 100_000_000
        WELLS_TO_SELECT = 200
        REVENUE_PER_UNIT = 4500
        POINTS_STUDIED = 500
        BOOTSTRAP_SAMPLES = 1000
        RANDOM_STATE = 42
        
        summary_rows = []

        for region, df in dfs.items():
            # Train & Predict
            # We recreate model each time to be safe
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            
            predictions_df, score, rmse = train_and_predict(
                df, 
                target_col='product',
                model=model,
                test_size=0.25, 
                random_state=RANDOM_STATE
            )
            
            # Bootstrap
            profits = bootstrap_profit(
                results_df=predictions_df,
                count=WELLS_TO_SELECT,
                revenue_per_unit=REVENUE_PER_UNIT,
                budget=BUDGET,
                repeats=BOOTSTRAP_SAMPLES,
                n_samples=POINTS_STUDIED,
                random_state=RANDOM_STATE
            )
            
            # Stats
            stats = analyze_region_profitability(profits)
            stats['region'] = region

            # Calculate Profit/Risk Ratio (Profit in Millions / Risk %)
            # Avoid division by zero: if risk is very low (< 0.1), use 0.1
            effective_risk = max(stats['risk_of_loss_percent'], 0.1)
            profit_millions = stats['mean_profit'] / 1_000_000
            ratio = profit_millions / effective_risk
            
            stats['profit_risk_ratio'] = ratio
            results_summary.append(stats)
            
            # Collect for plotting/summary
            profits_dict[region] = profits
            regions_risk[region] = stats['risk_of_loss_percent']
            
            summary_rows.append(stats)

        # 5. Determine Best Region
        summary_df = pd.DataFrame(results_summary).set_index('region')
        
        # Filter regions with risk < 2.5%
        valid_regions = summary_df[summary_df['risk_of_loss_percent'] < 2.5]
        
        # 5. Determine default sort (Desc by Ratio)
        summary_rows.sort(key=lambda x: x['profit_risk_ratio'], reverse=True)

        best_region = None # No longer highlighting
            
        # 6. Generate Plots
        profit_plot_b64 = generate_profit_distribution_plot(profits_dict)
        risk_plot_b64 = generate_risk_loss_plot(regions_risk)
        
        # 7. Render Template
        return Template(HTML_TEMPLATE).render(
            results=True,
            summary=summary_rows,
            recommended_region="Decide based on Profit/Risk Ratio below",
            profit_plot=profit_plot_b64,
            risk_plot=risk_plot_b64
        )

    except Exception as e:
        # Clean up the file if analysis fails
        if new_filepath.exists():
            os.remove(new_filepath)
        return Template(HTML_TEMPLATE).render(error=f"Analysis failed: {str(e)}")
