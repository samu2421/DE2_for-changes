import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def analyze_ecommerce_data():
    """
    Comprehensive analysis of the Online Retail dataset
    Designed for data engineering pipeline understanding
    """
    
    print("Starting E-Commerce Data Analysis...")
    print("=" * 50)
    
    # Load the Kaggle dataset
    try:
        df = pd.read_csv('data/OnlineRetail.csv', encoding='latin1')
        print(f"Dataset loaded successfully!")
    except FileNotFoundError:
        print("OnlineRetail.csv not found in data/ folder")
        print("Please download from: https://www.kaggle.com/datasets/vijayuv/onlineretail")
        return None
    
    # Basic dataset information
    print(f"\nDATASET OVERVIEW")
    print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"Memory usage: {df.memory_usage().sum() / 1024**2:.2f} MB")
    print(f"Date range: {df['InvoiceDate'].min()} to {df['InvoiceDate'].max()}")
    
    # Column information
    print(f"\nCOLUMNS:")
    for i, col in enumerate(df.columns, 1):
        dtype = df[col].dtype
        null_count = df[col].isnull().sum()
        null_pct = (null_count / len(df)) * 100
        print(f"{i}. {col:15} | {str(dtype):10} | Nulls: {null_count:6,} ({null_pct:5.1f}%)")
    
    # Data quality assessment
    print(f"\nDATA QUALITY ASSESSMENT:")
    
    # Missing values analysis
    missing_summary = df.isnull().sum()
    if missing_summary.sum() > 0:
        print("Missing values detected:")
        for col, missing in missing_summary[missing_summary > 0].items():
            print(f"  • {col}: {missing:,} ({missing/len(df)*100:.1f}%)")
    
    # Duplicate analysis
    duplicates = df.duplicated().sum()
    print(f"Duplicate rows: {duplicates:,} ({duplicates/len(df)*100:.1f}%)")
    
    # Business metrics analysis
    print(f"\nBUSINESS METRICS:")
    
    # Clean data for analysis (remove nulls and negatives)
    df_clean = df.dropna()
    df_clean = df_clean[df_clean['Quantity'] > 0]
    df_clean = df_clean[df_clean['UnitPrice'] > 0]
    
    # Calculate revenue
    df_clean['Revenue'] = df_clean['Quantity'] * df_clean['UnitPrice']
    
    print(f"Clean dataset: {len(df_clean):,} transactions")
    print(f"Total revenue: £{df_clean['Revenue'].sum():,.2f}")
    print(f"Average order value: £{df_clean['Revenue'].mean():.2f}")
    print(f"Unique customers: {df_clean['CustomerID'].nunique():,}")
    print(f"Unique products: {df_clean['StockCode'].nunique():,}")
    print(f"Countries served: {df_clean['Country'].nunique()}")
    
    # Top performing segments
    print(f"\nTOP PERFORMERS:")
    
    # Top countries by revenue
    top_countries = df_clean.groupby('Country')['Revenue'].sum().sort_values(ascending=False).head(5)
    print("Top 5 countries by revenue:")
    for country, revenue in top_countries.items():
        print(f"  • {country}: £{revenue:,.2f}")
    
    # Top products by quantity sold
    top_products = df_clean.groupby('StockCode')['Quantity'].sum().sort_values(ascending=False).head(5)
    print(f"\nTop 5 products by quantity:")
    for product, qty in top_products.items():
        product_desc = df_clean[df_clean['StockCode'] == product]['Description'].iloc[0]
        print(f"  • {product} ({product_desc[:30]}): {qty:,} units")
    
    # Time-based patterns
    print(f"\nTEMPORAL PATTERNS:")
    df_clean['InvoiceDate'] = pd.to_datetime(df_clean['InvoiceDate'])
    df_clean['Hour'] = df_clean['InvoiceDate'].dt.hour
    df_clean['DayOfWeek'] = df_clean['InvoiceDate'].dt.dayofweek
    df_clean['Month'] = df_clean['InvoiceDate'].dt.month
    
    # Peak hours
    hourly_revenue = df_clean.groupby('Hour')['Revenue'].sum()
    peak_hour = hourly_revenue.idxmax()
    print(f"Peak revenue hour: {peak_hour}:00 (£{hourly_revenue.max():,.2f})")
    
    # Peak day of week (0=Monday)
    daily_revenue = df_clean.groupby('DayOfWeek')['Revenue'].sum()
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    peak_day = days[daily_revenue.idxmax()]
    print(f"Peak revenue day: {peak_day} (£{daily_revenue.max():,.2f})")
    
    # Data engineering insights
    print(f"\nDATA ENGINEERING INSIGHTS:")
    print(f"• Dataset spans {(df_clean['InvoiceDate'].max() - df_clean['InvoiceDate'].min()).days} days")
    print(f"• Average {len(df_clean) / df_clean['InvoiceDate'].dt.date.nunique():.0f} transactions per day")
    print(f"• Data volume: ~{len(df_clean) * 8 / 1024**2:.2f} MB (assuming 8 bytes per field)")
    print(f"• Recommended batch size: {min(10000, len(df_clean) // 100)} records")
    
    # Save analysis results
    print(f"\nSAVING ANALYSIS RESULTS...")
    
    # Ensure docs directory exists
    Path('docs').mkdir(exist_ok=True)
    
    # Summary statistics
    summary_stats = df_clean.describe()
    summary_stats.to_csv('docs/dataset_summary.csv')
    print("Summary statistics saved to docs/dataset_summary.csv")
    
    # Business metrics
    business_metrics = {
        'total_transactions': len(df_clean),
        'total_revenue': df_clean['Revenue'].sum(),
        'unique_customers': df_clean['CustomerID'].nunique(),
        'unique_products': df_clean['StockCode'].nunique(),
        'countries_served': df_clean['Country'].nunique(),
        'date_range_days': (df_clean['InvoiceDate'].max() - df_clean['InvoiceDate'].min()).days,
        'avg_daily_transactions': len(df_clean) / df_clean['InvoiceDate'].dt.date.nunique()
    }
    
    # Save to file for pipeline use
    import json
    with open('docs/business_metrics.json', 'w') as f:
        json.dump(business_metrics, f, indent=2, default=str)
    print("Business metrics saved to docs/business_metrics.json")
    
    print(f"\nRECOMMENDATIONS FOR PIPELINE:")
    print("• Use batch processing for historical analysis")
    print("• Implement streaming for real-time revenue tracking") 
    print("• Focus ML on predicting: order value, customer behavior, demand")
    print("• Monitor: transaction volume, revenue, data quality")
    print("• Key features for ML: Hour, DayOfWeek, Country, Quantity, UnitPrice")
    
    return df_clean


if __name__ == "__main__":
    analyzed_data = analyze_ecommerce_data()
    
    if analyzed_data is not None:
        print(f"\nAnalysis complete! Check docs/ folder for saved results.")
        print(f"Use this data understanding for designing your batch pipeline.")
    else:
        print("Analysis failed. Please check your data file.")