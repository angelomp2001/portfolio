
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import pandas as pd

def generate_profit_distribution_plot(profits_dict: dict):
    """
    Generates a histogram plot of profit distributions for all regions.
    Returns: Base64 encoded image string.
    """
    plt.figure(figsize=(10, 6))
    
    for region, profits in profits_dict.items():
        sns.kdeplot(profits, label=region, fill=True, alpha=0.3)
        plt.axvline(profits.mean(), linestyle='--', linewidth=1, label=f"Mean: {region}")
    
    plt.title("Expected Profit Distribution by Region")
    plt.xlabel("Profit ($)")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    
    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    
    # Encode
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return img_base64

def generate_risk_loss_plot(risk_dict: dict):
    """
    Generates a bar chart comparing risk of loss across regions.
    Returns: Base64 encoded image string.
    """
    regions = list(risk_dict.keys())
    risks = list(risk_dict.values())
    
    plt.figure(figsize=(8, 5))
    bars = plt.bar(regions, risks, color=['red' if r > 2.5 else 'green' for r in risks])
    
    plt.axhline(2.5, color='black', linestyle='--', label="Risk Threshold (2.5%)")
    plt.title("Risk of Loss (%) by Region")
    plt.ylabel("Risk of Loss (%)")
    plt.legend()
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.2f}%',
                 ha='center', va='bottom')
                 
    plt.tight_layout()
    
    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    
    # Encode
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return img_base64
